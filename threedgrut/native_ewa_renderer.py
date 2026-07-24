"""Classic EWA rasterization with optional layered-depth buffers."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from enum import Enum
from functools import partial
from typing import Protocol, cast

import torch
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint as checkpoint_function

from threedgrut.datasets.protocols import Batch
from threedgrut.layered_depth_compositor import composite_layered_depth
from threedgrut.native_lidar_renderer import camera_to_world_to_viewmat
from threedgrut.rectangular_ewa_compositor import (
    composite_rectangular_ewa,
    masked_rectangular_ewa_forward_contribution,
    masked_rectangular_ewa_forward_visibility,
)
from threedgrut.utils.render import SH2RGB

GsplatRasterization = Callable[
    ...,
    tuple[torch.Tensor, torch.Tensor, dict[str, object]],
]
GsplatRasterizeToIndicesInRange = Callable[
    ...,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]

_NATIVE_EPS2D = 0.1
_NATIVE_NEAR_PLANE = 0.1
_NATIVE_TILE_SIZE = 16
_LAYERED_DEPTH_DEFAULT_ALPHA_CUTOFF = 1.0 / 255.0
_EWA_TILE_BOUND_MARGIN = 0.1
_LAYERED_DEPTH_DEFAULT_TRANSPARENCY_THRESHOLD = 0.0
_LAYERED_DEPTH_DEFAULT_TILE_ROWS_PER_BATCH = 16
_EWA_RENDER_TRANSMITTANCE_CUTOFF = 0.01
_EWA_CONTRIBUTION_TRANSMITTANCE_CUTOFF = 0.01
_LAYERED_DEPTH_DEFAULT_ACTIVATION_CHECKPOINT = True


class LayeredDepthCompositor(str, Enum):
    """Selectable implementation for layered-depth buffer compositing."""

    REFERENCE = "reference"
    FUSED = "fused"


class FOVClampBackward(str, Enum):
    """Backward convention for EWA covariance FOV clipping."""

    AUTODIFF = "autodiff"
    FROZEN = "frozen"


class NativeEWAGaussians(Protocol):
    """Gaussian state consumed by the native classic-EWA renderer."""

    max_sh_degree: int
    n_active_features: int
    num_gaussians: int
    positions: torch.Tensor
    environment_mask: torch.Tensor
    background: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool],
        tuple[torch.Tensor, torch.Tensor],
    ]

    def get_positions(self) -> torch.Tensor:
        """Return world-space Gaussian means."""

    def get_rotation(self) -> torch.Tensor:
        """Return normalized Gaussian rotations."""

    def get_scale(self) -> torch.Tensor:
        """Return physical Gaussian scales."""

    def get_density(self) -> torch.Tensor:
        """Return physical Gaussian opacities."""

    def get_features_albedo(self) -> torch.Tensor:
        """Return SH0 native albedo coefficients."""


class LocalProjectionFieldSampler(Protocol):
    """Sample one source-frame local field at projected pixel centers."""

    def sample(
        self,
        *,
        source_frame_idx: int,
        means2d: torch.Tensor,
        resolution: tuple[int, int],
    ) -> torch.Tensor:
        """Return normalized two-axis offsets for each projected center."""


def _load_gsplat_rasterization() -> GsplatRasterization:
    """Load gsplat's differentiable classic Gaussian rasterizer."""
    try:
        rendering = importlib.import_module("gsplat.rendering")
    except ModuleNotFoundError as exc:
        msg = (
            "native_ewa rendering requires gsplat. Install the CUDA splat "
            "dependency with ./scripts/post_build.sh --deps gsplat."
        )
        raise RuntimeError(msg) from exc
    rasterization = getattr(rendering, "rasterization", None)
    if not callable(rasterization):
        msg = "gsplat.rendering.rasterization is not callable."
        raise TypeError(msg)
    return cast(GsplatRasterization, rasterization)


def _load_gsplat_rasterize_to_indices_in_range(
) -> GsplatRasterizeToIndicesInRange:
    """Load gsplat's static packed-index enumerator."""
    try:
        wrapper = importlib.import_module("gsplat.cuda._wrapper")
    except ModuleNotFoundError as exc:
        msg = (
            "native_ewa rendering requires gsplat. Install the CUDA splat "
            "dependency with ./scripts/post_build.sh --deps gsplat."
        )
        raise RuntimeError(msg) from exc
    rasterize_to_indices_in_range = getattr(
        wrapper,
        "rasterize_to_indices_in_range",
        None,
    )
    if not callable(rasterize_to_indices_in_range):
        msg = "gsplat.rasterize_to_indices_in_range is not callable."
        raise TypeError(msg)
    return cast(GsplatRasterizeToIndicesInRange, rasterize_to_indices_in_range)


def _meta_tensor(meta: dict[str, object], key: str) -> torch.Tensor:
    """Return one tensor-valued gsplat metadata entry."""
    value = meta.get(key)
    if not torch.is_tensor(value):
        msg = f"native_ewa gsplat metadata {key!r} is not a tensor."
        raise RuntimeError(msg)
    return value


def _pair(value: object | None, label: str) -> tuple[float, float]:
    """Normalize one native pinhole pair from a batch intrinsics record."""
    if value is None:
        raise ValueError(f"native_ewa requires {label}.")
    if torch.is_tensor(value):
        values = value.detach().flatten().tolist()
    elif isinstance(value, (str, bytes)):
        msg = f"native_ewa {label} must be a two-value sequence."
        raise TypeError(msg)
    else:
        try:
            values = list(cast(Iterable[object], value))
        except TypeError as exc:
            msg = f"native_ewa {label} must be a two-value sequence."
            raise TypeError(msg) from exc
    if len(values) != 2:
        msg = f"native_ewa {label} must have exactly two values."
        raise ValueError(msg)
    return float(values[0]), float(values[1])


def _capture_screen_gradient(
    gradient: torch.Tensor,
    *,
    gaussian_ids: torch.Tensor,
    destination: torch.Tensor,
) -> torch.Tensor:
    """Preserve true EWA pixel-coordinate gradients for native topology."""
    with torch.no_grad():
        destination.index_copy_(0, gaussian_ids, gradient)
    return gradient


def _screen_gradient_pixel_norm(
    gradient: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert viewport-normalized screen gradients to native pixel norms."""
    pixel_gradient = gradient * gradient.new_tensor(
        (0.5 * width, 0.5 * height)
    )
    return torch.linalg.vector_norm(pixel_gradient, dim=1)


def _scatter_metadata(
    *,
    gaussian_ids: torch.Tensor,
    point_count: int,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    means2d: torch.Tensor,
    radii: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expand packed gsplat projection state into native per-point arrays."""
    if gaussian_ids.ndim != 1:
        raise ValueError("native_ewa expects one packed Gaussian-id vector.")
    if conics.shape != (gaussian_ids.numel(), 3):
        raise ValueError("native_ewa conics do not align with Gaussian ids.")
    if opacities.shape != (gaussian_ids.numel(),):
        raise ValueError("native_ewa opacities do not align with Gaussian ids.")
    if means2d.shape != (gaussian_ids.numel(), 2):
        raise ValueError("native_ewa means2d do not align with Gaussian ids.")
    if radii.shape != (gaussian_ids.numel(), 2):
        raise ValueError("native_ewa radii do not align with Gaussian ids.")
    if gaussian_ids.numel() and (
        gaussian_ids.min() < 0 or gaussian_ids.max() >= point_count
    ):
        raise ValueError("native_ewa Gaussian ids are outside the model range.")

    device = means2d.device
    dtype = means2d.dtype
    visibility = torch.zeros((point_count, 1), device=device, dtype=torch.bool)
    projected_position = torch.zeros((point_count, 2), device=device, dtype=dtype)
    projected_conic_opacity = torch.zeros(
        (point_count, 4),
        device=device,
        dtype=dtype,
    )
    projected_extent = torch.zeros((point_count, 2), device=device, dtype=dtype)
    visibility[gaussian_ids, 0] = True
    projected_position.index_copy_(0, gaussian_ids, means2d.detach())
    projected_conic_opacity.index_copy_(
        0,
        gaussian_ids,
        torch.cat((conics.detach(), opacities.detach()[:, None]), dim=1),
    )
    projected_extent.index_copy_(0, gaussian_ids, radii.to(dtype=dtype))
    return {
        "mog_visibility": visibility,
        "mog_projected_position": projected_position,
        "mog_projected_conic_opacity": projected_conic_opacity,
        "mog_projected_extent": projected_extent,
    }


def _current_ewa_contribution_weights(
    *,
    gaussian_ids: torch.Tensor,
    point_count: int,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    environment_mask: torch.Tensor,
    pixel_mask: torch.Tensor | None,
    width: int,
    height: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
) -> torch.Tensor:
    """Measure each splat's EWA contribution before base termination."""
    contribution = masked_rectangular_ewa_forward_contribution(
        means2d=means2d.detach(),
        conics=conics.detach(),
        opacities=opacities.detach(),
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        offsets=isect_offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
        alpha_cutoff=_LAYERED_DEPTH_DEFAULT_ALPHA_CUTOFF,
        transmittance_cutoff=_EWA_CONTRIBUTION_TRANSMITTANCE_CUTOFF,
    )
    result = torch.zeros(
        (point_count, 1),
        device=contribution.device,
        dtype=contribution.dtype,
    )
    result.index_copy_(0, gaussian_ids, contribution[:, None])
    return result


def _current_ewa_forward_visibility(
    *,
    gaussian_ids: torch.Tensor,
    point_count: int,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    environment_mask: torch.Tensor,
    pixel_mask: torch.Tensor | None,
    width: int,
    height: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
) -> torch.Tensor:
    """Measure per-Gaussian EWA reachability before base termination."""
    packed_visibility = masked_rectangular_ewa_forward_visibility(
        means2d=means2d.detach(),
        conics=conics.detach(),
        opacities=opacities.detach(),
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        offsets=isect_offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
        alpha_cutoff=_LAYERED_DEPTH_DEFAULT_ALPHA_CUTOFF,
        transmittance_cutoff=_EWA_CONTRIBUTION_TRANSMITTANCE_CUTOFF,
    )
    result = torch.zeros(
        (point_count, 1),
        device=packed_visibility.device,
        dtype=torch.bool,
    )
    result.index_copy_(0, gaussian_ids, packed_visibility[:, None])
    return result


@torch.no_grad()
def _layered_depth_tile_lists(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    alpha_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build depth-sorted tile lists for one selected Gaussian layer."""
    point_count = means2d.shape[0]
    if means2d.shape != (point_count, 2):
        raise ValueError("Layer means2d must have shape (N, 2).")
    if conics.shape != (point_count, 3):
        raise ValueError("Layer conics must have shape (N, 3).")
    if opacities.shape != (point_count,):
        raise ValueError("Layer opacities must have shape (N,).")
    if depths.shape != (point_count,):
        raise ValueError("Layer depths must have shape (N,).")
    if width <= 0 or height <= 0:
        raise ValueError("Layer image dimensions must be positive.")
    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("Layer tile dimensions must be positive.")
    if alpha_cutoff <= 0.0:
        raise ValueError("Layer alpha cutoff must be positive.")

    tiles_width = (width + tile_width - 1) // tile_width
    tiles_height = (height + tile_height - 1) // tile_height
    offsets = torch.zeros(
        (1, tiles_height, tiles_width),
        device=means2d.device,
        dtype=torch.int32,
    )
    if point_count == 0:
        return offsets, torch.empty(
            (0,),
            device=means2d.device,
            dtype=torch.int32,
        )

    determinant = conics[:, 0] * conics[:, 2] - conics[:, 1].square()
    support = torch.log(opacities / alpha_cutoff)
    usable = (
        torch.isfinite(means2d).all(dim=1)
        & torch.isfinite(conics).all(dim=1)
        & torch.isfinite(opacities)
        & torch.isfinite(depths)
        & (opacities >= alpha_cutoff)
        & (determinant > 0.0)
        & (support >= 0.0)
    )
    usable_ids = torch.nonzero(usable, as_tuple=False).squeeze(1)
    if usable_ids.numel() == 0:
        return offsets, torch.empty(
            (0,),
            device=means2d.device,
            dtype=torch.int32,
        )

    usable_means = means2d.index_select(0, usable_ids)
    usable_conics = conics.index_select(0, usable_ids)
    usable_support = support.index_select(0, usable_ids)
    usable_determinant = determinant.index_select(0, usable_ids)
    radius_x = torch.sqrt(
        2.0 * usable_support * usable_conics[:, 2] / usable_determinant
    ) + _EWA_TILE_BOUND_MARGIN
    radius_y = torch.sqrt(
        2.0 * usable_support * usable_conics[:, 0] / usable_determinant
    ) + _EWA_TILE_BOUND_MARGIN
    minimum_x = torch.floor(
        (usable_means[:, 0] - radius_x) / tile_width
    ).to(dtype=torch.int64)
    maximum_x = torch.floor(
        (usable_means[:, 0] + radius_x) / tile_width
    ).to(dtype=torch.int64)
    minimum_y = torch.floor(
        (usable_means[:, 1] - radius_y) / tile_height
    ).to(dtype=torch.int64)
    maximum_y = torch.floor(
        (usable_means[:, 1] + radius_y) / tile_height
    ).to(dtype=torch.int64)
    intersects = (
        (maximum_x >= 0)
        & (minimum_x < tiles_width)
        & (maximum_y >= 0)
        & (minimum_y < tiles_height)
    )
    usable_ids = usable_ids[intersects]
    if usable_ids.numel() == 0:
        return offsets, torch.empty(
            (0,),
            device=means2d.device,
            dtype=torch.int32,
        )

    minimum_x = minimum_x[intersects].clamp(0, tiles_width - 1)
    maximum_x = maximum_x[intersects].clamp(0, tiles_width - 1)
    minimum_y = minimum_y[intersects].clamp(0, tiles_height - 1)
    maximum_y = maximum_y[intersects].clamp(0, tiles_height - 1)
    span_x = maximum_x - minimum_x + 1
    span_y = maximum_y - minimum_y + 1
    tile_counts = span_x * span_y
    starts = torch.cumsum(tile_counts, dim=0) - tile_counts
    local_ids = torch.repeat_interleave(usable_ids, tile_counts)
    local_starts = torch.repeat_interleave(starts, tile_counts)
    local_span_x = torch.repeat_interleave(span_x, tile_counts)
    local_minimum_x = torch.repeat_interleave(minimum_x, tile_counts)
    local_minimum_y = torch.repeat_interleave(minimum_y, tile_counts)
    local_offset = torch.arange(
        local_ids.numel(),
        device=means2d.device,
        dtype=torch.int64,
    ) - local_starts
    tile_x = local_minimum_x + torch.remainder(local_offset, local_span_x)
    tile_y = local_minimum_y + torch.div(
        local_offset,
        local_span_x,
        rounding_mode="floor",
    )
    tile_ids = tile_y * tiles_width + tile_x
    depth_order = torch.argsort(
        depths.index_select(0, local_ids),
        stable=True,
    )
    tile_order = torch.argsort(tile_ids.index_select(0, depth_order), stable=True)
    ordered = depth_order.index_select(0, tile_order)
    sorted_tile_ids = tile_ids.index_select(0, ordered)
    flatten_ids = local_ids.index_select(0, ordered).to(dtype=torch.int32)
    counts_per_tile = torch.bincount(
        sorted_tile_ids,
        minlength=tiles_width * tiles_height,
    )
    starts_per_tile = torch.cumsum(counts_per_tile, dim=0) - counts_per_tile
    offsets.copy_(starts_per_tile.reshape(1, tiles_height, tiles_width))
    return offsets, flatten_ids


def _layered_depth_accumulation(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    tile_width: int,
    vertical_scale: int,
    alpha_cutoff: float,
    tile_rows_per_batch: int,
    activation_checkpoint: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite one selected Gaussian layer in bounded tile-row batches."""
    stretched_height = height * vertical_scale
    expected_tiles_width = (width + tile_width - 1) // tile_width
    expected_tiles_height = (
        stretched_height + tile_width - 1
    ) // tile_width
    if offsets.shape != (1, expected_tiles_height, expected_tiles_width):
        raise ValueError("Layer tile offsets do not match the render shape.")
    if tile_rows_per_batch <= 0:
        raise ValueError("Layer tile-row batch size must be positive.")

    raw_depth_bands: list[torch.Tensor] = []
    transmittance_bands: list[torch.Tensor] = []
    median_depth_bands: list[torch.Tensor] = []
    flat_offsets = offsets.reshape(-1)
    for tile_row_start in range(
        0,
        expected_tiles_height,
        tile_rows_per_batch,
    ):
        tile_row_end = min(
            tile_row_start + tile_rows_per_batch,
            expected_tiles_height,
        )
        first_tile = tile_row_start * expected_tiles_width
        end_tile = tile_row_end * expected_tiles_width
        list_start = int(flat_offsets[first_tile].item())
        list_end = (
            int(flat_offsets[end_tile].item())
            if end_tile < flat_offsets.numel()
            else flatten_ids.numel()
        )
        band_offsets = (
            offsets[:, tile_row_start:tile_row_end] - list_start
        ).contiguous()
        band_flatten_ids = flatten_ids[list_start:list_end].contiguous()
        y_offset = tile_row_start * tile_width
        band_height = min(
            stretched_height - y_offset,
            (tile_row_end - tile_row_start) * tile_width,
        )
        def accumulate_band(
            current_means2d: torch.Tensor,
            current_conics: torch.Tensor,
            current_opacities: torch.Tensor,
            current_depths: torch.Tensor,
            current_offsets: torch.Tensor = band_offsets,
            current_flatten_ids: torch.Tensor = band_flatten_ids,
            current_height: int = band_height,
            current_y_offset: int = y_offset,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return _layered_depth_accumulation_band(
                means2d=current_means2d,
                conics=current_conics,
                opacities=current_opacities,
                depths=current_depths,
                offsets=current_offsets,
                flatten_ids=current_flatten_ids,
                width=width,
                height=current_height,
                tile_width=tile_width,
                alpha_cutoff=alpha_cutoff,
                y_offset=current_y_offset,
            )

        should_checkpoint = activation_checkpoint and torch.is_grad_enabled()
        if should_checkpoint:
            raw_depth, transmittance, median_depth = checkpoint_function(
                accumulate_band,
                means2d,
                conics,
                opacities,
                depths,
                use_reentrant=False,
            )
        else:
            raw_depth, transmittance, median_depth = accumulate_band(
                means2d,
                conics,
                opacities,
                depths,
            )
        raw_depth_bands.append(raw_depth)
        transmittance_bands.append(transmittance)
        median_depth_bands.append(median_depth)
    return (
        torch.cat(raw_depth_bands, dim=1),
        torch.cat(transmittance_bands, dim=1),
        torch.cat(median_depth_bands, dim=1),
    )


def _layered_depth_accumulation_band(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    tile_width: int,
    alpha_cutoff: float,
    y_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite one contiguous tile-row band into depth buffers."""
    if y_offset:
        means2d = torch.stack(
            (
                means2d[:, 0],
                means2d[:, 1] - float(y_offset),
            ),
            dim=1,
        )
    total_pixels = width * height
    transmittance = torch.ones(
        (total_pixels,),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    raw_depth = torch.zeros_like(transmittance)
    median_depth = torch.zeros_like(transmittance)
    flat_offsets = offsets.reshape(-1)
    offsets_with_end = torch.cat(
        (
            flat_offsets,
            torch.tensor(
                (flatten_ids.numel(),),
                device=flat_offsets.device,
                dtype=flat_offsets.dtype,
            ),
        )
    )
    max_range = int(
        torch.max(offsets_with_end[1:] - offsets_with_end[:-1]).item()
    )
    if max_range == 0:
        shape = (1, height, width, 1)
        return (
            raw_depth.reshape(shape),
            transmittance.reshape(shape),
            median_depth.reshape(shape),
        )

    rasterize_to_indices_in_range = _load_gsplat_rasterize_to_indices_in_range()
    chunk_size = tile_width * tile_width
    chunk_count = (max_range + chunk_size - 1) // chunk_size
    for chunk_index in range(chunk_count):
        gaussian_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
            chunk_index,
            chunk_index + 1,
            transmittance.reshape(1, height, width),
            means2d.unsqueeze(0),
            conics.unsqueeze(0),
            opacities.unsqueeze(0),
            width,
            height,
            tile_width,
            offsets,
            flatten_ids,
        )
        if gaussian_ids.numel() == 0:
            continue
        if not bool(torch.all(image_ids == 0)):
            raise RuntimeError("Layered depth expects one camera.")
        if pixel_ids.numel() > 1 and not bool(
            torch.all(pixel_ids[1:] >= pixel_ids[:-1])
        ):
            raise RuntimeError("Layered depth expects row-major pixel ids.")
        pixel_x = (pixel_ids % width).to(dtype=means2d.dtype) + 0.5
        pixel_y = torch.div(pixel_ids, width, rounding_mode="floor").to(
            dtype=means2d.dtype
        ) + 0.5
        delta_x = pixel_x - means2d[gaussian_ids, 0]
        delta_y = pixel_y - means2d[gaussian_ids, 1]
        selected_conics = conics[gaussian_ids]
        sigma = 0.5 * (
            selected_conics[:, 0] * delta_x.square()
            + selected_conics[:, 2] * delta_y.square()
        ) + selected_conics[:, 1] * delta_x * delta_y
        alpha = torch.clamp_max(
            opacities[gaussian_ids] * torch.exp(-sigma),
            0.99,
        )
        keep = alpha >= alpha_cutoff
        gaussian_ids = gaussian_ids[keep]
        pixel_ids = pixel_ids[keep]
        alpha = alpha[keep]
        if gaussian_ids.numel() == 0:
            continue
        unique_pixels, counts = torch.unique_consecutive(
            pixel_ids,
            return_counts=True,
        )
        log_alpha = torch.log1p(-alpha).to(dtype=torch.float64)
        global_after = torch.cumsum(log_alpha, dim=0)
        starts = torch.cumsum(counts, dim=0) - counts
        group_before = torch.zeros(
            (counts.numel(),),
            device=alpha.device,
            dtype=log_alpha.dtype,
        )
        if counts.numel() > 1:
            group_before[1:] = global_after[starts[1:] - 1]
        groups = torch.repeat_interleave(
            torch.arange(counts.numel(), device=alpha.device),
            counts,
        )
        relative_after = torch.exp(global_after - group_before[groups])
        before = (
            transmittance[pixel_ids] * relative_after / (1.0 - alpha)
        )
        raw_depth.scatter_add_(
            0,
            pixel_ids,
            (before * alpha * depths[gaussian_ids]).to(dtype=raw_depth.dtype),
        )
        after = before * (1.0 - alpha)
        with torch.no_grad():
            crosses_median = (
                (median_depth[pixel_ids] == 0.0)
                & (before.detach() > 0.5)
                & (after.detach() <= 0.5)
            )
            if bool(torch.any(crosses_median)):
                median_depth[pixel_ids[crosses_median]] = depths[
                    gaussian_ids[crosses_median]
                ].detach()
        final = torch.exp(torch.segment_reduce(log_alpha, "sum", lengths=counts))
        transmittance[unique_pixels] *= final.to(dtype=transmittance.dtype)
    shape = (1, height, width, 1)
    return (
        raw_depth.reshape(shape),
        transmittance.reshape(shape),
        median_depth.reshape(shape),
    )


def _render_layered_depth_buffers(
    *,
    gaussian_ids: torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    selected_layer_mask: torch.Tensor,
    semantic_mask: torch.Tensor,
    width: int,
    height: int,
    near_plane: float,
    tile_width: int,
    tile_height: int,
    alpha_cutoff: float,
    transparency_threshold: float,
    tile_rows_per_batch: int,
    activation_checkpoint: bool,
    compositor: LayeredDepthCompositor,
) -> dict[str, torch.Tensor]:
    """Render raw, median, and thresholded transparency for one layer."""
    packed_count = gaussian_ids.numel()
    if depths.shape != (packed_count,):
        raise ValueError("Layer depths do not align with Gaussian ids.")
    if selected_layer_mask.dtype != torch.bool:
        raise ValueError("Selected layer mask must use bool dtype.")
    if semantic_mask.shape != (1, height, width, 1):
        raise ValueError("Semantic mask must match the rendered image shape.")
    if semantic_mask.dtype != torch.uint8:
        raise ValueError("Semantic mask must use uint8 labels.")
    if semantic_mask.device != means2d.device:
        raise ValueError("Semantic mask must share the render device.")
    if gaussian_ids.numel() and (
        int(gaussian_ids.max().item()) >= selected_layer_mask.numel()
    ):
        raise ValueError("Selected layer mask does not cover Gaussian ids.")
    if tile_width % tile_height != 0:
        raise ValueError(
            "Layered depth requires tile_width to be divisible by tile_height."
        )
    if near_plane <= 0.0:
        raise ValueError("Layered depth near plane must be positive.")
    vertical_scale = tile_width // tile_height
    if vertical_scale < 1:
        raise ValueError("Layered depth vertical scale must be positive.")
    if transparency_threshold < 0.0:
        raise ValueError("Layer transparency threshold cannot be negative.")

    layer_rows = selected_layer_mask.index_select(0, gaussian_ids)
    layer_rows = layer_rows & (depths >= near_plane)
    layer_means = means2d[layer_rows]
    layer_conics = conics[layer_rows]
    layer_opacities = opacities[layer_rows]
    layer_depths = depths[layer_rows]
    offsets, flatten_ids = _layered_depth_tile_lists(
        means2d=layer_means,
        conics=layer_conics,
        opacities=layer_opacities,
        depths=layer_depths,
        width=width,
        height=height,
        tile_width=tile_width,
        tile_height=tile_height,
        alpha_cutoff=alpha_cutoff,
    )
    stretch = float(vertical_scale)
    stretched_means = torch.stack(
        (
            layer_means[:, 0],
            stretch * layer_means[:, 1] - 0.5 * (stretch - 1.0),
        ),
        dim=1,
    )
    stretched_conics = torch.stack(
        (
            layer_conics[:, 0],
            layer_conics[:, 1] / stretch,
            layer_conics[:, 2] / (stretch * stretch),
        ),
        dim=1,
    )
    if compositor is LayeredDepthCompositor.FUSED:
        raw_depth, transmittance, median_depth = composite_layered_depth(
            means2d=stretched_means,
            conics=stretched_conics,
            opacities=layer_opacities,
            depths=layer_depths,
            offsets=offsets,
            flatten_ids=flatten_ids,
            width=width,
            height=height * vertical_scale,
            tile_size=tile_width,
            alpha_cutoff=alpha_cutoff,
        )
    else:
        raw_depth, transmittance, median_depth = (
            _layered_depth_accumulation(
                means2d=stretched_means,
                conics=stretched_conics,
                opacities=layer_opacities,
                depths=layer_depths,
                offsets=offsets,
                flatten_ids=flatten_ids,
                width=width,
                height=height,
                tile_width=tile_width,
                vertical_scale=vertical_scale,
                alpha_cutoff=alpha_cutoff,
                tile_rows_per_batch=tile_rows_per_batch,
                activation_checkpoint=activation_checkpoint,
            )
        )
    raw_depth = raw_depth[:, 0::vertical_scale]
    transmittance = transmittance[:, 0::vertical_scale]
    median_depth = median_depth[:, 0::vertical_scale]
    layer_transparency = torch.where(
        transmittance >= transparency_threshold,
        transmittance,
        torch.zeros_like(transmittance),
    )
    invalid = semantic_mask == 0
    return {
        "layered_depth_raw": torch.where(
            invalid,
            torch.zeros_like(raw_depth),
            raw_depth,
        ),
        "layered_transparency": torch.where(
            invalid,
            torch.zeros_like(layer_transparency),
            layer_transparency,
        ),
        "layered_median_depth": torch.where(
            invalid,
            torch.zeros_like(median_depth),
            median_depth,
        ),
    }


def _fov_clamp_projection_position_vjp(
    *,
    positions: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    densities: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    means2d_gradient: torch.Tensor,
    conics_gradient: torch.Tensor,
    depths_gradient: torch.Tensor,
    opacities_gradient: torch.Tensor,
    width: int,
    height: int,
    eps2d: float,
    freeze_clamp_backward: bool,
) -> torch.Tensor:
    """Recompute one packed pinhole projection VJP for selected splats."""
    if positions.shape[0] == 0:
        return torch.zeros_like(positions)
    with torch.enable_grad():
        world = positions.detach().requires_grad_(True)
        world_to_camera = torch.linalg.inv(
            camera_to_world.to(device=world.device, dtype=world.dtype)
        )
        camera_rotation = world_to_camera[:3, :3]
        camera_translation = world_to_camera[:3, 3]
        mean_camera = world @ camera_rotation.T + camera_translation
        x, y, z = mean_camera.unbind(dim=1)
        camera_intrinsics = intrinsics.to(
            device=world.device,
            dtype=world.dtype,
        )
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        tan_fovx = 0.5 * width / fx
        tan_fovy = 0.5 * height / fy
        limit_x_positive = (width - cx) / fx + 0.3 * tan_fovx
        limit_x_negative = cx / fx + 0.3 * tan_fovx
        limit_y_positive = (height - cy) / fy + 0.3 * tan_fovy
        limit_y_negative = cy / fy + 0.3 * tan_fovy
        ratio_x = x / z
        ratio_y = y / z
        projected_x = z * torch.clamp(
            ratio_x,
            min=-limit_x_negative,
            max=limit_x_positive,
        )
        projected_y = z * torch.clamp(
            ratio_y,
            min=-limit_y_negative,
            max=limit_y_positive,
        )
        if freeze_clamp_backward:
            x_clipped = (ratio_x < -limit_x_negative) | (
                ratio_x > limit_x_positive
            )
            y_clipped = (ratio_y < -limit_y_negative) | (
                ratio_y > limit_y_positive
            )
            projected_x = torch.where(
                x_clipped,
                projected_x.detach(),
                projected_x,
            )
            projected_y = torch.where(
                y_clipped,
                projected_y.detach(),
                projected_y,
            )
        normalized_rotation = torch.nn.functional.normalize(rotations, dim=1)
        qw, qx, qy, qz = normalized_rotation.unbind(dim=1)
        rotation_matrix = torch.stack(
            (
                1.0 - 2.0 * (qy.square() + qz.square()),
                2.0 * (qx * qy - qw * qz),
                2.0 * (qx * qz + qw * qy),
                2.0 * (qx * qy + qw * qz),
                1.0 - 2.0 * (qx.square() + qz.square()),
                2.0 * (qy * qz - qw * qx),
                2.0 * (qx * qz - qw * qy),
                2.0 * (qy * qz + qw * qx),
                1.0 - 2.0 * (qx.square() + qy.square()),
            ),
            dim=1,
        ).reshape(-1, 3, 3)
        covariance_matrix = rotation_matrix * scales[:, None, :]
        covariance_world = covariance_matrix @ covariance_matrix.transpose(
            1,
            2,
        )
        covariance_camera = (
            camera_rotation @ covariance_world @ camera_rotation.T
        )
        zeros = torch.zeros_like(z)
        jacobian = torch.stack(
            (
                fx / z,
                zeros,
                -fx * projected_x / z.square(),
                zeros,
                fy / z,
                -fy * projected_y / z.square(),
            ),
            dim=1,
        ).reshape(-1, 2, 3)
        covariance_2d_raw = (
            jacobian @ covariance_camera @ jacobian.transpose(1, 2)
        )
        determinant_raw = (
            covariance_2d_raw[:, 0, 0] * covariance_2d_raw[:, 1, 1]
            - covariance_2d_raw[:, 0, 1] * covariance_2d_raw[:, 1, 0]
        )
        covariance_2d = covariance_2d_raw + torch.eye(
            2,
            device=world.device,
            dtype=world.dtype,
        ).expand_as(covariance_2d_raw) * eps2d
        determinant = (
            covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1]
            - covariance_2d[:, 0, 1] * covariance_2d[:, 1, 0]
        ).clamp_min(1e-10)
        conics = torch.stack(
            (
                covariance_2d[:, 1, 1] / determinant,
                -(
                    covariance_2d[:, 0, 1] + covariance_2d[:, 1, 0]
                )
                / (2.0 * determinant),
                covariance_2d[:, 0, 0] / determinant,
            ),
            dim=1,
        )
        compensation = torch.sqrt(
            (determinant_raw / determinant).clamp_min(0.005 * 0.005)
        )
        effective_opacities = densities * compensation
        means2d = torch.stack(
            (fx * x / z + cx, fy * y / z + cy),
            dim=1,
        )
        objective = (
            (means2d * means2d_gradient.detach()).sum()
            + (conics * conics_gradient.detach()).sum()
            + (z * depths_gradient.detach()).sum()
            + (effective_opacities * opacities_gradient.detach()).sum()
        )
        gradient = torch.autograd.grad(objective, world)[0]
    return gradient.detach()


class _FrozenFOVClampBackwardCorrection(torch.autograd.Function):
    """Add the legacy FOV-clamp position VJP without changing render values."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        densities: torch.Tensor,
        gaussian_ids: torch.Tensor,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        width: int,
        height: int,
        eps2d: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return zero-valued paths that carry the recovered VJP on backward."""
        viewmat = torch.linalg.inv(camera_to_world.detach())[0]
        packed_positions = positions.detach().index_select(0, gaussian_ids)
        mean_camera = (
            packed_positions @ viewmat[:3, :3].T + viewmat[:3, 3]
        )
        fx = intrinsics[0, 0, 0]
        fy = intrinsics[0, 1, 1]
        cx = intrinsics[0, 0, 2]
        cy = intrinsics[0, 1, 2]
        limit_x_positive = (width - cx) / fx + 0.15 * width / fx
        limit_x_negative = cx / fx + 0.15 * width / fx
        limit_y_positive = (height - cy) / fy + 0.15 * height / fy
        limit_y_negative = cy / fy + 0.15 * height / fy
        ratio_x = mean_camera[:, 0] / mean_camera[:, 2]
        ratio_y = mean_camera[:, 1] / mean_camera[:, 2]
        clipped_packed = torch.nonzero(
            (ratio_x < -limit_x_negative)
            | (ratio_x > limit_x_positive)
            | (ratio_y < -limit_y_negative)
            | (ratio_y > limit_y_positive),
            as_tuple=False,
        ).reshape(-1)
        ctx.point_count = positions.shape[0]
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.has_clipped = bool(clipped_packed.numel())
        if ctx.has_clipped:
            clipped_ids = gaussian_ids.index_select(0, clipped_packed)
            ctx.save_for_backward(
                positions.detach().index_select(0, clipped_ids),
                scales.detach().index_select(0, clipped_ids),
                rotations.detach().index_select(0, clipped_ids),
                densities.detach().index_select(0, clipped_ids),
                clipped_ids,
                clipped_packed,
                camera_to_world.detach()[0],
                intrinsics.detach()[0],
            )
        return (
            torch.zeros_like(means2d),
            torch.zeros_like(conics),
            torch.zeros_like(opacities),
            torch.zeros_like(depths),
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        means2d_gradient: torch.Tensor | None,
        conics_gradient: torch.Tensor | None,
        opacities_gradient: torch.Tensor | None,
        depths_gradient: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        """Inject frozen-minus-autodiff position gradients for clipped splats."""
        if not ctx.has_clipped:
            return (None,) * 14
        (
            positions,
            scales,
            rotations,
            densities,
            clipped_ids,
            clipped_packed,
            camera_to_world,
            intrinsics,
        ) = ctx.saved_tensors
        if means2d_gradient is None:
            means2d_gradient = positions.new_zeros((clipped_packed.numel(), 2))
        else:
            means2d_gradient = means2d_gradient.index_select(
                0,
                clipped_packed,
            )
        if conics_gradient is None:
            conics_gradient = positions.new_zeros((clipped_packed.numel(), 3))
        else:
            conics_gradient = conics_gradient.index_select(0, clipped_packed)
        if opacities_gradient is None:
            opacities_gradient = positions.new_zeros((clipped_packed.numel(),))
        else:
            opacities_gradient = opacities_gradient.index_select(
                0,
                clipped_packed,
            )
        if depths_gradient is None:
            depths_gradient = positions.new_zeros((clipped_packed.numel(),))
        else:
            depths_gradient = depths_gradient.index_select(0, clipped_packed)
        autodiff_gradient = _fov_clamp_projection_position_vjp(
            positions=positions,
            scales=scales,
            rotations=rotations,
            densities=densities,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            means2d_gradient=means2d_gradient,
            conics_gradient=conics_gradient,
            depths_gradient=depths_gradient,
            opacities_gradient=opacities_gradient,
            width=ctx.width,
            height=ctx.height,
            eps2d=ctx.eps2d,
            freeze_clamp_backward=False,
        )
        frozen_gradient = _fov_clamp_projection_position_vjp(
            positions=positions,
            scales=scales,
            rotations=rotations,
            densities=densities,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            means2d_gradient=means2d_gradient,
            conics_gradient=conics_gradient,
            depths_gradient=depths_gradient,
            opacities_gradient=opacities_gradient,
            width=ctx.width,
            height=ctx.height,
            eps2d=ctx.eps2d,
            freeze_clamp_backward=True,
        )
        position_gradient = positions.new_zeros((ctx.point_count, 3))
        position_gradient.index_add_(
            0,
            clipped_ids,
            frozen_gradient - autodiff_gradient,
        )
        return (position_gradient,) + (None,) * 13


def _apply_frozen_fov_clamp_backward(
    *,
    mode: FOVClampBackward,
    positions: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    densities: torch.Tensor,
    gaussian_ids: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    width: int,
    height: int,
    eps2d: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return projection outputs with an optional frozen-clamp VJP side path."""
    if mode is FOVClampBackward.AUTODIFF:
        return means2d, conics, opacities, depths
    if mode is not FOVClampBackward.FROZEN:
        raise ValueError(f"Unsupported FOV clamp backward mode {mode!r}.")
    corrections = _FrozenFOVClampBackwardCorrection.apply(
        positions,
        scales,
        rotations,
        densities,
        gaussian_ids,
        camera_to_world,
        intrinsics,
        means2d,
        conics,
        opacities,
        depths,
        width,
        height,
        eps2d,
    )
    return tuple(
        value + correction
        for value, correction in zip(
            (means2d, conics, opacities, depths),
            corrections,
            strict=True,
        )
    )


class Tracer:
    """Differentiable classic-EWA renderer with optional layer buffers."""

    def __init__(self, conf: DictConfig) -> None:
        self.conf = conf
        self.local_projection_field: LocalProjectionFieldSampler | None = None
        native_conf = conf.render.get("native_ewa", {})
        self.eps2d = float(native_conf.get("eps2d", _NATIVE_EPS2D))
        self.near_plane = float(
            native_conf.get("near_plane", _NATIVE_NEAR_PLANE)
        )
        self.tile_size = int(
            native_conf.get("tile_size", _NATIVE_TILE_SIZE)
        )
        self.global_z_order = bool(native_conf.get("global_z_order", True))
        fov_clamp_backward_value = str(
            native_conf.get(
                "fov_clamp_backward",
                FOVClampBackward.AUTODIFF.value,
            )
        )
        try:
            self.fov_clamp_backward = FOVClampBackward(
                fov_clamp_backward_value
            )
        except ValueError as exc:
            msg = (
                "render.native_ewa.fov_clamp_backward must be one of "
                f"{[member.value for member in FOVClampBackward]}; got "
                f"{fov_clamp_backward_value!r}."
            )
            raise ValueError(msg) from exc
        layered_depth_conf = native_conf.get("layered_depth", {})
        self.layered_depth_tile_height = int(
            layered_depth_conf.get("tile_height", self.tile_size)
        )
        self.layered_depth_alpha_cutoff = float(
            layered_depth_conf.get(
                "alpha_cutoff",
                _LAYERED_DEPTH_DEFAULT_ALPHA_CUTOFF,
            )
        )
        self.layered_depth_transparency_threshold = float(
            layered_depth_conf.get(
                "transparency_threshold",
                _LAYERED_DEPTH_DEFAULT_TRANSPARENCY_THRESHOLD,
            )
        )
        self.layered_depth_tile_rows_per_batch = int(
            layered_depth_conf.get(
                "tile_rows_per_batch",
                _LAYERED_DEPTH_DEFAULT_TILE_ROWS_PER_BATCH,
            )
        )
        self.layered_depth_activation_checkpoint = bool(
            layered_depth_conf.get(
                "activation_checkpoint",
                _LAYERED_DEPTH_DEFAULT_ACTIVATION_CHECKPOINT,
            )
        )
        compositor_value = str(
            layered_depth_conf.get(
                "compositor",
                LayeredDepthCompositor.REFERENCE.value,
            )
        )
        try:
            self.layered_depth_compositor = LayeredDepthCompositor(
                compositor_value
            )
        except ValueError as exc:
            msg = (
                "render.native_ewa.layered_depth.compositor must be one of "
                f"{[member.value for member in LayeredDepthCompositor]}; got "
                f"{compositor_value!r}."
            )
            raise ValueError(msg) from exc
        loss_conf = conf.get("loss", {})
        self.use_layered_depth_adjoint = bool(
            loss_conf.get("use_layered_depth_adjoint", False)
        )
        if self.eps2d <= 0.0:
            raise ValueError("render.native_ewa.eps2d must be positive.")
        if self.near_plane <= 0.0:
            raise ValueError("render.native_ewa.near_plane must be positive.")
        if self.tile_size != _NATIVE_TILE_SIZE:
            raise ValueError(
                "native_ewa requires the recovered 16-pixel tile size."
            )
        if not self.global_z_order:
            raise ValueError("native_ewa requires global depth ordering.")
        if self.use_layered_depth_adjoint:
            if self.layered_depth_tile_height <= 0:
                raise ValueError(
                    "render.native_ewa.layered_depth.tile_height must be "
                    "positive."
                )
            if self.tile_size % self.layered_depth_tile_height != 0:
                raise ValueError(
                    "render.native_ewa.layered_depth.tile_height must divide "
                    "render.native_ewa.tile_size."
                )
            if self.layered_depth_alpha_cutoff <= 0.0:
                raise ValueError(
                    "render.native_ewa.layered_depth.alpha_cutoff must be "
                    "positive."
                )
            if self.layered_depth_transparency_threshold < 0.0:
                raise ValueError(
                    "render.native_ewa.layered_depth.transparency_threshold "
                    "cannot be negative."
                )
            if self.layered_depth_tile_rows_per_batch <= 0:
                raise ValueError(
                    "render.native_ewa.layered_depth.tile_rows_per_batch "
                    "must be positive."
                )

    @property
    def timings(self) -> dict[str, float]:
        """Return the same timing surface as native tracer backends."""
        return {}

    def set_local_projection_field(
        self,
        field: LocalProjectionFieldSampler | None,
    ) -> None:
        """Bind the trainer-owned local field for source-frame rendering."""
        self.local_projection_field = field

    def _apply_local_projection_field(
        self,
        *,
        means2d: torch.Tensor,
        intrinsics: torch.Tensor,
        gpu_batch: Batch,
        width: int,
        height: int,
    ) -> torch.Tensor:
        field = self.local_projection_field
        if field is None:
            return means2d
        source_frame_idx = int(gpu_batch.source_frame_idx)
        offsets = field.sample(
            source_frame_idx=source_frame_idx,
            means2d=means2d,
            resolution=(width, height),
        )
        if offsets.shape != means2d.shape:
            raise RuntimeError(
                "Local projection field offsets must match projected centers."
            )
        focal_length = torch.stack(
            (intrinsics[0, 0, 0], intrinsics[0, 1, 1])
        )
        return means2d + focal_length.reshape(1, 2) * offsets

    @torch.no_grad()
    def build_acc(
        self,
        gaussians: NativeEWAGaussians,
        rebuild: bool = True,
    ) -> None:
        """Classic EWA uses per-frame tiles and has no persistent BVH."""
        del gaussians, rebuild

    @staticmethod
    def _intrinsics(
        gpu_batch: Batch,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters is None:
            raise ValueError(
                "native_ewa requires OpenCVPinholeCameraModelParameters."
            )
        focal_length = _pair(
            gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters.get(
                "focal_length"
            ),
            "focal_length",
        )
        principal_point = _pair(
            gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters.get(
                "principal_point"
            ),
            "principal_point",
        )
        intrinsics = torch.zeros((1, 3, 3), device=device, dtype=dtype)
        intrinsics[0, 0, 0] = focal_length[0]
        intrinsics[0, 1, 1] = focal_length[1]
        intrinsics[0, 0, 2] = principal_point[0]
        intrinsics[0, 1, 2] = principal_point[1]
        intrinsics[0, 2, 2] = 1.0
        return intrinsics

    @staticmethod
    def _ray_distance_from_axial_depth(
        *,
        axial_depth: torch.Tensor,
        gpu_batch: Batch,
    ) -> torch.Tensor:
        ray_z = gpu_batch.depth_ray_z
        if ray_z is None:
            ray_z = torch.abs(gpu_batch.rays_dir[..., 2:3])
        ray_z = ray_z.to(device=axial_depth.device, dtype=axial_depth.dtype)
        if ray_z.shape != axial_depth.shape:
            msg = (
                "native_ewa axial depth and camera ray-z tensors must have "
                "the same shape."
            )
            raise ValueError(msg)
        epsilon = torch.finfo(axial_depth.dtype).eps
        return axial_depth / ray_z.clamp_min(epsilon)

    def render(
        self,
        gaussians: NativeEWAGaussians,
        gpu_batch: Batch,
        train: bool = False,
        frame_id: int = 0,
    ) -> dict[str, torch.Tensor | float]:
        """Rasterize one native SH0 pinhole frame through classic EWA."""
        del frame_id
        if gaussians.max_sh_degree != 0 or gaussians.n_active_features != 0:
            raise ValueError(
                "native_ewa requires the recovered native SH0 radiance "
                "configuration."
            )
        if gpu_batch.rgb_gt is None:
            raise ValueError("native_ewa requires RGB ground truth dimensions.")
        if gpu_batch.T_to_world.shape != (1, 4, 4):
            raise ValueError(
                "native_ewa supports one global-shutter camera per render."
            )
        if gpu_batch.T_to_world_end is not None:
            raise ValueError("native_ewa does not support rolling-shutter poses.")
        if train and not torch.is_grad_enabled():
            raise RuntimeError(
                "native_ewa training requires gradients to recover EWA "
                "contribution weights."
            )

        height = int(gpu_batch.rgb_gt.shape[1])
        width = int(gpu_batch.rgb_gt.shape[2])
        if height <= 0 or width <= 0:
            raise ValueError("native_ewa requires a positive image resolution.")
        positions = gaussians.get_positions()
        scales = gaussians.get_scale()
        densities = gaussians.get_density().squeeze(-1)
        rotations = gaussians.get_rotation()
        intrinsics = self._intrinsics(
            gpu_batch,
            device=positions.device,
            dtype=positions.dtype,
        )
        colors = SH2RGB(gaussians.get_features_albedo())
        rasterization = _load_gsplat_rasterization()
        _, _, meta = rasterization(
            means=positions,
            quats=rotations,
            scales=scales,
            opacities=densities,
            colors=colors,
            viewmats=camera_to_world_to_viewmat(gpu_batch.T_to_world),
            Ks=intrinsics,
            width=width,
            height=height,
            near_plane=self.near_plane,
            packed=True,
            rasterize_mode="antialiased",
            render_mode="RGB+ED",
            eps2d=self.eps2d,
            tile_size=self.tile_size,
            global_z_order=self.global_z_order,
        )
        gaussian_ids = _meta_tensor(meta, "gaussian_ids")
        means2d = _meta_tensor(meta, "means2d")
        conics = _meta_tensor(meta, "conics")
        opacities = _meta_tensor(meta, "opacities")
        depths = _meta_tensor(meta, "depths")
        radii = _meta_tensor(meta, "radii")
        if train:
            means2d, conics, opacities, depths = (
                _apply_frozen_fov_clamp_backward(
                    mode=self.fov_clamp_backward,
                    positions=positions,
                    scales=scales,
                    rotations=rotations,
                    densities=densities,
                    gaussian_ids=gaussian_ids,
                    camera_to_world=gpu_batch.T_to_world,
                    intrinsics=intrinsics,
                    means2d=means2d,
                    conics=conics,
                    opacities=opacities,
                    depths=depths,
                    width=width,
                    height=height,
                    eps2d=self.eps2d,
                )
            )
        means2d = self._apply_local_projection_field(
            means2d=means2d,
            intrinsics=intrinsics,
            gpu_batch=gpu_batch,
            width=width,
            height=height,
        )
        isect_offsets, flatten_ids = _layered_depth_tile_lists(
            means2d=means2d,
            conics=conics,
            opacities=opacities,
            depths=depths,
            width=width,
            height=height,
            tile_width=_NATIVE_TILE_SIZE,
            tile_height=_NATIVE_TILE_SIZE // 2,
            alpha_cutoff=_LAYERED_DEPTH_DEFAULT_ALPHA_CUTOFF,
        )
        packed_environment_mask = gaussians.environment_mask.index_select(
            0,
            gaussian_ids,
        )
        rendered_rgb, alpha, axial_depth = composite_rectangular_ewa(
            means2d=means2d,
            conics=conics,
            opacities=opacities,
            colors=colors.index_select(0, gaussian_ids),
            depths=depths,
            environment_mask=packed_environment_mask,
            pixel_mask=gpu_batch.mask,
            offsets=isect_offsets,
            flatten_ids=flatten_ids,
            width=width,
            height=height,
            alpha_cutoff=_LAYERED_DEPTH_DEFAULT_ALPHA_CUTOFF,
            transmittance_cutoff=_EWA_RENDER_TRANSMITTANCE_CUTOFF,
        )
        rendered = torch.cat((rendered_rgb, axial_depth), dim=3)
        if rendered.shape != (1, height, width, 4):
            msg = (
                "native_ewa expected RGB plus expected-depth output with "
                f"shape (1, {height}, {width}, 4), got {tuple(rendered.shape)}."
            )
            raise RuntimeError(msg)
        if alpha.shape != (1, height, width, 1):
            msg = (
                "native_ewa expected one opacity image with shape "
                f"(1, {height}, {width}, 1), got {tuple(alpha.shape)}."
            )
            raise RuntimeError(msg)

        tiles_per_gauss = torch.bincount(
            flatten_ids.to(dtype=torch.int64),
            minlength=gaussian_ids.numel(),
        )
        layered_depth_buffers: dict[str, torch.Tensor] = {}
        if self.use_layered_depth_adjoint:
            if gpu_batch.semantic_mask is None:
                raise RuntimeError(
                    "Layered depth adjoint requires a raw semantic mask."
                )
            layered_depth_buffers = _render_layered_depth_buffers(
                gaussian_ids=gaussian_ids,
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                depths=depths,
                selected_layer_mask=~gaussians.environment_mask,
                semantic_mask=gpu_batch.semantic_mask,
                width=width,
                height=height,
                near_plane=self.near_plane,
                tile_width=self.tile_size,
                tile_height=self.layered_depth_tile_height,
                alpha_cutoff=self.layered_depth_alpha_cutoff,
                transparency_threshold=(
                    self.layered_depth_transparency_threshold
                ),
                tile_rows_per_batch=self.layered_depth_tile_rows_per_batch,
                activation_checkpoint=(
                    self.layered_depth_activation_checkpoint
                ),
                compositor=self.layered_depth_compositor,
            )
        state = _scatter_metadata(
            gaussian_ids=gaussian_ids,
            point_count=gaussians.num_gaussians,
            conics=conics,
            opacities=opacities,
            means2d=means2d,
            radii=radii,
        )
        projected_gradient = torch.zeros_like(
            state["mog_projected_position"]
        )
        projected_gradient_pixels = torch.zeros(
            (gaussians.num_gaussians,),
            device=positions.device,
            dtype=positions.dtype,
        )
        if train:
            means2d.register_hook(
                partial(
                    _capture_screen_gradient,
                    gaussian_ids=gaussian_ids,
                    destination=projected_gradient,
                )
            )
            def capture_gradient_pixels(gradient: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    projected_gradient_pixels.index_copy_(
                        0,
                        gaussian_ids,
                        _screen_gradient_pixel_norm(
                            gradient,
                            width=width,
                            height=height,
                        ),
                    )
                return gradient

            means2d.register_hook(capture_gradient_pixels)
            accumulated_weight = _current_ewa_contribution_weights(
                gaussian_ids=gaussian_ids,
                point_count=gaussians.num_gaussians,
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                environment_mask=packed_environment_mask,
                pixel_mask=gpu_batch.mask,
                width=width,
                height=height,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
            )
            forward_visibility = _current_ewa_forward_visibility(
                gaussian_ids=gaussian_ids,
                point_count=gaussians.num_gaussians,
                means2d=means2d,
                conics=conics,
                opacities=opacities,
                environment_mask=packed_environment_mask,
                pixel_mask=gpu_batch.mask,
                width=width,
                height=height,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
            )
        else:
            accumulated_weight = torch.zeros(
                (gaussians.num_gaussians, 1),
                device=positions.device,
                dtype=positions.dtype,
            )
            forward_visibility = torch.zeros(
                (gaussians.num_gaussians, 1),
                device=positions.device,
                dtype=torch.bool,
            )

        tiles_count = torch.zeros(
            (gaussians.num_gaussians, 1),
            device=positions.device,
            dtype=positions.dtype,
        )
        tiles_count.index_copy_(
            0,
            gaussian_ids,
            tiles_per_gauss.to(dtype=positions.dtype)[:, None],
        )
        pred_rgb, pred_opacity = gaussians.background(
            gpu_batch.rays_ori,
            gpu_batch.rays_dir,
            rendered[..., :3],
            alpha,
            train,
        )
        axial_depth = rendered[..., 3:]
        pred_dist = self._ray_distance_from_axial_depth(
            axial_depth=axial_depth,
            gpu_batch=gpu_batch,
        )
        return {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(
                torch.ones_like(pred_rgb),
                dim=3,
            ),
            "hits_count": (pred_opacity > 0.0).to(dtype=pred_opacity.dtype),
            "frame_time_ms": 0.0,
            **state,
            "mog_tiles_count": tiles_count,
            "mog_accumulated_weight": accumulated_weight,
            "mog_forward_visibility": forward_visibility,
            "mog_projected_position_gradient": projected_gradient,
            "mog_projected_gradient_pixels": projected_gradient_pixels,
            **layered_depth_buffers,
        }

    def render_diagnostic(
        self,
        gaussians: NativeEWAGaussians,
        gpu_batch: Batch,
        frame_id: int = 0,
        features_override: torch.Tensor | None = None,
        sph_degree_override: int | None = None,
    ) -> dict[str, torch.Tensor | float]:
        """Render native EWA diagnostics without altering native radiance."""
        del frame_id
        if features_override is not None or sph_degree_override is not None:
            raise ValueError(
                "native_ewa diagnostics do not support feature overrides."
            )
        return self.render(gaussians, gpu_batch, train=False)

    @staticmethod
    def get_bvh_stats() -> dict[str, float]:
        """Classic EWA has no BVH statistics."""
        return {}
