"""Classic Gaussian depth rendering used by visibility-adaptive LiDAR updates."""

import importlib
import math
from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple, cast

import torch

GsplatFullyFusedProjection = Callable[
    ...,
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ],
]

_NATIVE_EPS2D = 0.1
_NATIVE_MIN_PROJECTED_COVARIANCE_DETERMINANT = 0.005
_NATIVE_LIDAR_TILE_SIZE = 4
_NATIVE_ALPHA_THRESHOLD = 1.0 / 255.0
_NATIVE_MAX_ALPHA = 0.99
_NATIVE_LIDAR_BATCH_SIZE = 32
_NATIVE_TRANSMITTANCE_THRESHOLD = 0.01


class NativeLidarRenderer(StrEnum):
    """Available LiDAR depth-rendering implementations."""

    GUT_HIT_DISTANCE = "gut_hit_distance"
    """3DGUT hit-distance rendering retained for existing runs."""

    CLASSIC_EXPECTED_DEPTH = "classic_expected_depth"
    """Classic Gaussian expected projected-depth rendering."""


class NativeLidarGaussianState(NamedTuple):
    """Physical Gaussian tensors consumed by the classic LiDAR renderer."""

    positions: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    skip_mask: torch.Tensor | None = None
    """Optional per-Gaussian mask for splats the native renderer skips."""


class NativeLidarRenderResult(NamedTuple):
    """Depth, opacity, and native point-status visibility for one LiDAR pass."""

    depth: torch.Tensor
    alpha: torch.Tensor
    visibility: torch.Tensor


def parse_native_lidar_renderer(value: str) -> NativeLidarRenderer:
    """Parse a configured LiDAR renderer with an actionable failure."""
    try:
        return NativeLidarRenderer(value)
    except ValueError as exc:
        supported = ", ".join(
            renderer.value for renderer in NativeLidarRenderer
        )
        msg = (
            f"Unsupported native LiDAR renderer {value!r}; "
            f"expected one of: {supported}."
        )
        raise ValueError(msg) from exc


def camera_to_world_to_viewmat(
    camera_to_world: torch.Tensor,
) -> torch.Tensor:
    """Convert one native camera-to-world pose into gsplat view-matrix form."""
    if camera_to_world.shape != (1, 4, 4):
        msg = (
            "Native LiDAR rendering requires one camera-to-world matrix; "
            f"got {tuple(camera_to_world.shape)}."
        )
        raise ValueError(msg)
    return torch.linalg.inv(camera_to_world)


def _load_gsplat_fully_fused_projection() -> GsplatFullyFusedProjection:
    """Load the projection primitive used to reproduce native culling."""
    try:
        wrapper = importlib.import_module("gsplat.cuda._wrapper")
    except ModuleNotFoundError as exc:
        msg = (
            "classic_expected_depth LiDAR rendering requires gsplat. "
            "Install the CUDA splat dependency with "
            "./scripts/post_build.sh --deps gsplat."
        )
        raise RuntimeError(msg) from exc
    projection = getattr(wrapper, "fully_fused_projection", None)
    if not callable(projection):
        msg = "gsplat.cuda._wrapper.fully_fused_projection is not callable."
        raise TypeError(msg)
    return cast(GsplatFullyFusedProjection, projection)


def native_projected_covariance_is_valid(conics: torch.Tensor) -> torch.Tensor:
    """Return the DLL's pre-regularization projected-covariance gate."""
    if conics.shape[-1] != 3:
        msg = f"Expected conics ending in three values, got {tuple(conics.shape)}."
        raise ValueError(msg)
    conic_xx, conic_xy, conic_yy = conics.unbind(dim=-1)
    inverse_determinant = conic_xx * conic_yy - conic_xy.square()
    covariance_xx = conic_yy / inverse_determinant - _NATIVE_EPS2D
    covariance_xy = -conic_xy / inverse_determinant
    covariance_yy = conic_xx / inverse_determinant - _NATIVE_EPS2D
    determinant = covariance_xx * covariance_yy - covariance_xy.square()
    return torch.isfinite(determinant) & (
        determinant > _NATIVE_MIN_PROJECTED_COVARIANCE_DETERMINANT
    )


def _project_native_lidar_gaussians(
    *,
    positions: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    viewmats: torch.Tensor,
    intrinsics_matrix: torch.Tensor,
    image_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Project one LiDAR camera with the native classic parameters."""
    projection = _load_gsplat_fully_fused_projection()
    radii, means2d, depths, conics, compensations = projection(
        positions,
        None,
        rotations,
        scales,
        viewmats,
        intrinsics_matrix,
        image_size,
        image_size,
        eps2d=_NATIVE_EPS2D,
        near_plane=0.01,
        packed=False,
        radius_clip=0.0,
        sparse_grad=False,
        calc_compensations=True,
        camera_model="pinhole",
        opacities=opacities.squeeze(-1),
    )
    if radii.shape != (1, positions.shape[0], 2):
        msg = (
            "Native LiDAR projection requires one virtual camera and one "
            f"radius pair per splat, got {tuple(radii.shape)}."
        )
        raise ValueError(msg)
    if compensations is None:
        msg = "Native LiDAR projection did not return opacity compensations."
        raise RuntimeError(msg)
    if compensations.shape != (1, positions.shape[0]):
        msg = (
            "Native LiDAR projection requires one opacity compensation per "
            f"splat, got {tuple(compensations.shape)}."
        )
        raise ValueError(msg)
    return radii, means2d, depths, conics, compensations


def _native_lidar_active_tile_mask(
    *,
    sample_grid: torch.Tensor,
    image_size: int,
    tile_width: int,
    tile_height: int,
) -> torch.Tensor:
    """Mark 4x4 tiles containing one sampled LiDAR ray."""
    query_xy = (
        sample_grid.reshape(-1, 2) + 1.0
    ) * (image_size * 0.5) - 0.5
    tile_x = torch.floor(
        query_xy[:, 0] / _NATIVE_LIDAR_TILE_SIZE
    ).to(dtype=torch.long)
    tile_y = torch.floor(
        query_xy[:, 1] / _NATIVE_LIDAR_TILE_SIZE
    ).to(dtype=torch.long)
    tile_x = tile_x.clamp(min=0, max=tile_width - 1)
    tile_y = tile_y.clamp(min=0, max=tile_height - 1)
    active_tiles = torch.zeros(
        (tile_height, tile_width),
        device=sample_grid.device,
        dtype=torch.bool,
    )
    active_tiles[tile_y, tile_x] = True
    return active_tiles


def _native_lidar_float_tile_bounds(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    compensations: torch.Tensor,
    tile_width: int,
    tile_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the native opacity-compensated float rectangle for each splat."""
    conic_xx, conic_xy, conic_yy = conics[0].unbind(dim=-1)
    inverse_determinant = conic_xx * conic_yy - conic_xy.square()
    covariance_xx = conic_yy / inverse_determinant
    covariance_yy = conic_xx / inverse_determinant
    covariance_xx = torch.nan_to_num(covariance_xx).clamp_min(0.0)
    covariance_yy = torch.nan_to_num(covariance_yy).clamp_min(0.0)
    bound_opacity = opacities.squeeze(-1) * compensations.squeeze(0)
    extent_squared = 2.0 * torch.log(
        bound_opacity.clamp_min(_NATIVE_ALPHA_THRESHOLD)
        / _NATIVE_ALPHA_THRESHOLD
    )
    extent_squared = extent_squared.clamp_min(0.0)
    radii = torch.stack(
        (
            torch.sqrt(extent_squared * covariance_xx),
            torch.sqrt(extent_squared * covariance_yy),
        ),
        dim=-1,
    )
    radii = torch.nan_to_num(radii).add(_NATIVE_EPS2D)
    lower = torch.trunc(
        (means2d[0] - radii) / _NATIVE_LIDAR_TILE_SIZE
    ).to(dtype=torch.long)
    upper = torch.trunc(
        (means2d[0] + radii + _NATIVE_LIDAR_TILE_SIZE)
        / _NATIVE_LIDAR_TILE_SIZE
    ).to(dtype=torch.long)
    lower_x = lower[:, 0].clamp(min=0, max=tile_width)
    lower_y = lower[:, 1].clamp(min=0, max=tile_height)
    upper_x = upper[:, 0].clamp(min=0, max=tile_width)
    upper_y = upper[:, 1].clamp(min=0, max=tile_height)
    return lower_x, lower_y, upper_x, upper_y


def _native_lidar_sorted_tile_lists(
    *,
    lower_x: torch.Tensor,
    lower_y: torch.Tensor,
    upper_x: torch.Tensor,
    upper_y: torch.Tensor,
    depths: torch.Tensor,
    culling_mask: torch.Tensor,
    active_tiles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit active float-bound intersections in native tile/depth order."""
    tile_height, tile_width = active_tiles.shape
    rectangle_width = (upper_x - lower_x).clamp_min(0)
    rectangle_height = (upper_y - lower_y).clamp_min(0)
    intersections_per_splat = rectangle_width * rectangle_height
    intersections_per_splat = torch.where(
        culling_mask,
        intersections_per_splat,
        torch.zeros_like(intersections_per_splat),
    )
    pair_count = int(intersections_per_splat.sum().item())
    if pair_count == 0:
        offsets = torch.zeros(
            (1, tile_height, tile_width),
            device=depths.device,
            dtype=torch.long,
        )
        flatten_ids = torch.empty(
            (0,),
            device=depths.device,
            dtype=torch.long,
        )
        return offsets, flatten_ids

    gaussian_ids = torch.repeat_interleave(
        torch.arange(depths.shape[0], device=depths.device),
        intersections_per_splat,
    )
    pair_offsets = torch.cumsum(intersections_per_splat, dim=0)
    pair_offsets = pair_offsets - intersections_per_splat
    local_indices = torch.arange(pair_count, device=depths.device)
    local_indices = local_indices - torch.repeat_interleave(
        pair_offsets,
        intersections_per_splat,
    )
    widths = rectangle_width[gaussian_ids]
    tile_x = lower_x[gaussian_ids] + torch.remainder(local_indices, widths)
    tile_y = lower_y[gaussian_ids] + torch.div(
        local_indices,
        widths,
        rounding_mode="floor",
    )
    tile_ids = tile_y * tile_width + tile_x
    active = active_tiles.reshape(-1)[tile_ids]
    gaussian_ids = gaussian_ids[active]
    tile_ids = tile_ids[active]
    if gaussian_ids.numel() == 0:
        offsets = torch.zeros(
            (1, tile_height, tile_width),
            device=depths.device,
            dtype=torch.long,
        )
        return offsets, gaussian_ids

    depth_order = torch.argsort(depths[gaussian_ids], stable=True)
    gaussian_ids = gaussian_ids[depth_order]
    tile_ids = tile_ids[depth_order]
    tile_order = torch.argsort(tile_ids, stable=True)
    gaussian_ids = gaussian_ids[tile_order]
    tile_ids = tile_ids[tile_order]
    intersections_per_tile = torch.bincount(
        tile_ids,
        minlength=tile_width * tile_height,
    )
    offsets = torch.cumsum(intersections_per_tile, dim=0)
    offsets = offsets - intersections_per_tile
    return offsets.reshape(1, tile_height, tile_width), gaussian_ids


def _native_lidar_masked_tile_lists(
    *,
    means2d: torch.Tensor,
    depths: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    compensations: torch.Tensor,
    sample_grid: torch.Tensor,
    image_size: int,
    tile_width: int,
    tile_height: int,
    culling_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build native float-bound LiDAR lists only for queried tiles."""
    active_tiles = _native_lidar_active_tile_mask(
        sample_grid=sample_grid,
        image_size=image_size,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    lower_x, lower_y, upper_x, upper_y = _native_lidar_float_tile_bounds(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        compensations=compensations,
        tile_width=tile_width,
        tile_height=tile_height,
    )
    return _native_lidar_sorted_tile_lists(
        lower_x=lower_x,
        lower_y=lower_y,
        upper_x=upper_x,
        upper_y=upper_y,
        depths=depths[0],
        culling_mask=culling_mask.squeeze(0),
        active_tiles=active_tiles,
    )


def _segmented_prefix_transmittance(
    alpha: torch.Tensor,
    segment_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return transmittance before each splat and after each query ray."""
    log_transmittance = torch.log1p(-alpha)
    prefix_log_transmittance = torch.cumsum(log_transmittance, dim=0)
    starts_mask = torch.ones_like(segment_indices, dtype=torch.bool)
    starts_mask[1:] = segment_indices[1:] != segment_indices[:-1]
    segment_starts = torch.nonzero(starts_mask, as_tuple=False).squeeze(-1)
    segment_ends = torch.cat(
        (
            segment_starts[1:] - 1,
            segment_starts.new_tensor((alpha.shape[0] - 1,)),
        )
    )
    segment_sizes = segment_ends - segment_starts + 1
    segment_prefixes = torch.zeros_like(
        segment_starts,
        dtype=prefix_log_transmittance.dtype,
    )
    segment_prefixes[1:] = prefix_log_transmittance[
        segment_starts[1:] - 1
    ]
    per_splat_prefixes = torch.repeat_interleave(
        segment_prefixes,
        segment_sizes,
    )
    transmittance_before = torch.exp(
        prefix_log_transmittance
        - log_transmittance
        - per_splat_prefixes
    )
    transmittance_after = torch.exp(
        prefix_log_transmittance[segment_ends] - segment_prefixes
    )
    return transmittance_before, transmittance_after, segment_starts


def _mask_low_transmittance_batches(
    *,
    alpha: torch.Tensor,
    segment_indices: torch.Tensor,
) -> torch.Tensor:
    """Stop later warp-sized batches once their ray is effectively opaque."""
    transmittance_before, _, segment_starts = (
        _segmented_prefix_transmittance(alpha, segment_indices)
    )
    pair_indices = torch.arange(alpha.shape[0], device=alpha.device)
    segment_ends = torch.cat(
        (
            segment_starts[1:] - 1,
            segment_starts.new_tensor((alpha.shape[0] - 1,)),
        )
    )
    segment_sizes = segment_ends - segment_starts + 1
    per_pair_segment_starts = torch.repeat_interleave(
        segment_starts,
        segment_sizes,
    )
    local_pair_indices = pair_indices - per_pair_segment_starts
    batch_starts = torch.remainder(
        local_pair_indices,
        _NATIVE_LIDAR_BATCH_SIZE,
    ).eq(0)
    batch_indices = torch.cumsum(batch_starts, dim=0) - 1
    batch_transmittance = transmittance_before[batch_starts]
    batch_is_active = (
        batch_transmittance >= _NATIVE_TRANSMITTANCE_THRESHOLD
    )
    return torch.where(
        batch_is_active[batch_indices],
        alpha,
        torch.zeros_like(alpha),
    )


def _render_native_sparse_expected_depth(
    *,
    means2d: torch.Tensor,
    depths: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    sample_grid: torch.Tensor,
    image_size: int,
    tile_width: int,
    tile_height: int,
    isect_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    skip_mask: torch.Tensor | None = None,
) -> NativeLidarRenderResult:
    """Evaluate sorted classic Gaussian tiles at continuous LiDAR pixels."""
    n_splats = opacities.shape[0]
    if means2d.shape != (1, n_splats, 2):
        msg = f"Expected projected means [1, N, 2], got {tuple(means2d.shape)}."
        raise ValueError(msg)
    if depths.shape != (1, n_splats):
        msg = f"Expected projected depths [1, N], got {tuple(depths.shape)}."
        raise ValueError(msg)
    if conics.shape != (1, n_splats, 3):
        msg = f"Expected projected conics [1, N, 3], got {tuple(conics.shape)}."
        raise ValueError(msg)
    if isect_offsets.shape != (1, tile_height, tile_width):
        msg = (
            "Expected one native LiDAR tile-offset image with shape "
            f"[1, {tile_height}, {tile_width}], got "
            f"{tuple(isect_offsets.shape)}."
        )
        raise ValueError(msg)

    query_shape = sample_grid.shape[:-1]
    query_xy = (
        sample_grid.reshape(-1, 2) + 1.0
    ) * (image_size * 0.5) - 0.5
    tile_x = torch.floor(
        query_xy[:, 0] / _NATIVE_LIDAR_TILE_SIZE
    ).to(dtype=torch.long)
    tile_y = torch.floor(
        query_xy[:, 1] / _NATIVE_LIDAR_TILE_SIZE
    ).to(dtype=torch.long)
    tile_x = tile_x.clamp(min=0, max=tile_width - 1)
    tile_y = tile_y.clamp(min=0, max=tile_height - 1)
    query_tile_ids = tile_y * tile_width + tile_x
    query_order = torch.argsort(query_tile_ids, stable=True)
    ordered_tile_ids = query_tile_ids[query_order]
    tile_ids, queries_per_tile = torch.unique_consecutive(
        ordered_tile_ids,
        return_counts=True,
    )
    query_offsets = torch.cumsum(queries_per_tile, dim=0)
    query_offsets = query_offsets - queries_per_tile

    tile_offsets = isect_offsets.reshape(-1).to(dtype=torch.long)
    tile_end_offsets = torch.cat(
        (
            tile_offsets[1:],
            tile_offsets.new_full((1,), flatten_ids.numel()),
        )
    )
    splats_per_tile = tile_end_offsets[tile_ids] - tile_offsets[tile_ids]
    pairs_per_tile = queries_per_tile * splats_per_tile
    pair_count = int(pairs_per_tile.sum().item())
    output_shape = (*query_shape, 1)
    if pair_count == 0:
        empty = torch.zeros(
            output_shape,
            device=means2d.device,
            dtype=means2d.dtype,
        )
        visibility = torch.zeros(
            n_splats,
            device=means2d.device,
            dtype=torch.bool,
        )
        return NativeLidarRenderResult(empty, empty, visibility)

    pair_tile_indices = torch.repeat_interleave(
        torch.arange(tile_ids.shape[0], device=means2d.device),
        pairs_per_tile,
    )
    pair_offsets = torch.cumsum(pairs_per_tile, dim=0) - pairs_per_tile
    local_pair_indices = torch.arange(
        pair_count,
        device=means2d.device,
    ) - torch.repeat_interleave(pair_offsets, pairs_per_tile)
    pair_splats_per_tile = splats_per_tile[pair_tile_indices]
    segment_indices = query_offsets[pair_tile_indices] + torch.div(
        local_pair_indices,
        pair_splats_per_tile,
        rounding_mode="floor",
    )
    query_ids = query_order[segment_indices]
    gaussian_ids = flatten_ids[
        tile_offsets[tile_ids[pair_tile_indices]]
        + torch.remainder(local_pair_indices, pair_splats_per_tile)
    ].to(dtype=torch.long)

    deltas = means2d[0, gaussian_ids] - query_xy[query_ids]
    selected_conics = conics[0, gaussian_ids]
    sigma = 0.5 * (
        selected_conics[:, 0] * deltas[:, 0].square()
        + selected_conics[:, 2] * deltas[:, 1].square()
    ) + selected_conics[:, 1] * deltas[:, 0] * deltas[:, 1]
    unclamped_alpha = opacities[gaussian_ids] * torch.exp(-sigma)
    clamped_alpha = torch.minimum(
        torch.full_like(sigma, _NATIVE_MAX_ALPHA),
        unclamped_alpha,
    )
    alpha = unclamped_alpha + (clamped_alpha - unclamped_alpha).detach()
    alpha = torch.where(
        (sigma >= 0.0) & (alpha >= _NATIVE_ALPHA_THRESHOLD),
        alpha,
        torch.zeros_like(alpha),
    )
    if skip_mask is not None:
        alpha = torch.where(
            skip_mask[gaussian_ids],
            torch.zeros_like(alpha),
            alpha,
        )
    alpha = _mask_low_transmittance_batches(
        alpha=alpha,
        segment_indices=segment_indices,
    )
    transmittance_before, transmittance_after, segment_starts = (
        _segmented_prefix_transmittance(alpha, segment_indices)
    )
    weights = alpha * transmittance_before
    accumulated_depth = torch.zeros(
        query_xy.shape[0],
        device=means2d.device,
        dtype=means2d.dtype,
    ).index_add(
        0,
        query_ids,
        # The DLL's LiDAR backward only emits projected-mean, conic, and
        # opacity gradients. It does not return a direct axial-depth gradient.
        weights * depths[0, gaussian_ids].detach(),
    )
    accumulated_alpha = 1.0 - transmittance_after
    expected_depth = accumulated_depth[query_ids[segment_starts]]
    expected_depth = expected_depth / accumulated_alpha.clamp_min(1e-10)
    expected_depth = torch.where(
        accumulated_alpha > 0.0,
        expected_depth,
        torch.zeros_like(expected_depth),
    )
    sampled_depth = torch.zeros(
        query_xy.shape[0],
        device=means2d.device,
        dtype=means2d.dtype,
    ).index_copy(0, query_ids[segment_starts], expected_depth)
    sampled_alpha = torch.zeros_like(sampled_depth).index_copy(
        0,
        query_ids[segment_starts],
        accumulated_alpha,
    )
    with torch.no_grad():
        visibility = torch.zeros(
            n_splats,
            device=means2d.device,
            dtype=torch.bool,
        )
        visibility.index_fill_(
            0,
            gaussian_ids[alpha.detach() > 0.0],
            True,
        )
    return NativeLidarRenderResult(
        sampled_depth.reshape(output_shape),
        sampled_alpha.reshape(output_shape),
        visibility,
    )


def render_classic_expected_depth(
    *,
    gaussian_state: NativeLidarGaussianState,
    camera_to_world: torch.Tensor,
    intrinsics: list[float],
    sample_grid: torch.Tensor,
    image_size: int,
) -> NativeLidarRenderResult:
    """Render native-style expected depth at continuous LiDAR query pixels."""
    positions = gaussian_state.positions
    rotations = gaussian_state.rotations
    scales = gaussian_state.scales
    opacities = gaussian_state.opacities
    if positions.ndim != 2 or positions.shape[1] != 3:
        msg = f"Expected positions [N, 3], got {tuple(positions.shape)}."
        raise ValueError(msg)
    if rotations.shape != (positions.shape[0], 4):
        msg = f"Expected rotations [N, 4], got {tuple(rotations.shape)}."
        raise ValueError(msg)
    if scales.shape != positions.shape:
        msg = f"Expected scales [N, 3], got {tuple(scales.shape)}."
        raise ValueError(msg)
    if opacities.shape != (positions.shape[0], 1):
        msg = f"Expected opacities [N, 1], got {tuple(opacities.shape)}."
        raise ValueError(msg)
    skip_mask = gaussian_state.skip_mask
    if skip_mask is not None:
        if skip_mask.shape != (positions.shape[0],):
            msg = (
                "Native LiDAR skip mask must have one value per Gaussian, got "
                f"{tuple(skip_mask.shape)}."
            )
            raise ValueError(msg)
        if skip_mask.dtype != torch.bool:
            msg = "Native LiDAR skip mask must use bool dtype."
            raise ValueError(msg)
        if skip_mask.device != positions.device:
            msg = "Native LiDAR skip mask must share the Gaussian device."
            raise ValueError(msg)
    if len(intrinsics) != 4:
        msg = "Native LiDAR intrinsics must be [fx, fy, cx, cy]."
        raise ValueError(msg)
    if sample_grid.ndim != 4 or sample_grid.shape[:2] != (1, 1):
        msg = (
            "Native LiDAR sample grid must have shape [1, 1, R, 2], got "
            f"{tuple(sample_grid.shape)}."
        )
        raise ValueError(msg)
    if sample_grid.shape[-1] != 2:
        msg = "Native LiDAR sample grid must end in the two image coordinates."
        raise ValueError(msg)
    if image_size <= 0:
        msg = f"Native LiDAR image size must be positive, got {image_size}."
        raise ValueError(msg)

    fx, fy, cx, cy = intrinsics
    intrinsics_matrix = torch.zeros(
        (1, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    intrinsics_matrix[0, 0, 0] = fx
    intrinsics_matrix[0, 1, 1] = fy
    intrinsics_matrix[0, 0, 2] = cx
    intrinsics_matrix[0, 1, 2] = cy
    intrinsics_matrix[0, 2, 2] = 1.0
    viewmats = camera_to_world_to_viewmat(camera_to_world)
    radii, means2d, depths, conics, compensations = (
        _project_native_lidar_gaussians(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            viewmats=viewmats,
            intrinsics_matrix=intrinsics_matrix,
            image_size=image_size,
        )
    )
    with torch.no_grad():
        native_culling_mask = (radii > 0).all(
            dim=-1
        ) & native_projected_covariance_is_valid(conics) & (depths > 0.1)
    if native_culling_mask.shape != (1, positions.shape[0]):
        msg = (
            "Native LiDAR culling requires one virtual camera and one mask "
            f"value per splat, got {tuple(native_culling_mask.shape)}."
        )
        raise ValueError(msg)
    filtered_opacities = torch.where(
        native_culling_mask.squeeze(0),
        opacities.squeeze(-1),
        torch.zeros_like(opacities.squeeze(-1)),
    )
    effective_opacities = filtered_opacities * compensations.squeeze(0)
    tile_width = math.ceil(image_size / _NATIVE_LIDAR_TILE_SIZE)
    tile_height = math.ceil(image_size / _NATIVE_LIDAR_TILE_SIZE)
    with torch.no_grad():
        isect_offsets, flatten_ids = _native_lidar_masked_tile_lists(
            means2d=means2d,
            depths=depths,
            conics=conics,
            opacities=opacities,
            compensations=compensations,
            sample_grid=sample_grid,
            image_size=image_size,
            tile_width=tile_width,
            tile_height=tile_height,
            culling_mask=native_culling_mask,
        )
    return _render_native_sparse_expected_depth(
        means2d=means2d,
        depths=depths,
        conics=conics,
        opacities=effective_opacities,
        sample_grid=sample_grid,
        image_size=image_size,
        tile_width=tile_width,
        tile_height=tile_height,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        skip_mask=skip_mask,
    )
