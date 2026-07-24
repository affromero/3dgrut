"""Differentiable rectangular-tile classic-EWA compositing."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.utils.cpp_extension

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


_TILE_WIDTH = 16
_TILE_HEIGHT = 8


class _RectangularEWACompositorExtension(Protocol):
    """Typed surface exposed by the fused rectangular EWA extension."""

    def forward(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        environment_mask: torch.Tensor,
        pixel_mask: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        alpha_cutoff: float,
        transmittance_cutoff: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Composite RGB plus all/base transmittance state."""

    def backward(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        environment_mask: torch.Tensor,
        pixel_mask: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        all_transmittance: torch.Tensor,
        base_transmittance: torch.Tensor,
        last_all: torch.Tensor,
        last_base: torch.Tensor,
        rgb_gradient: torch.Tensor,
        raw_depth_gradient: torch.Tensor,
        base_transmittance_gradient: torch.Tensor,
        all_transmittance_gradient: torch.Tensor,
        width: int,
        height: int,
        alpha_cutoff: float,
        transmittance_cutoff: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Accumulate packed EWA parameter gradients."""

    def masked_contribution(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        pixel_mask: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        alpha_cutoff: float,
        transmittance_cutoff: float,
    ) -> torch.Tensor:
        """Accumulate masked front-to-back contribution per packed row."""

    def masked_forward_contribution(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        environment_mask: torch.Tensor,
        pixel_mask: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        alpha_cutoff: float,
        transmittance_cutoff: float,
    ) -> torch.Tensor:
        """Accumulate rows reached before base-layer termination."""

    def masked_forward_visibility(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        environment_mask: torch.Tensor,
        pixel_mask: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        alpha_cutoff: float,
        transmittance_cutoff: float,
    ) -> torch.Tensor:
        """Mark packed rows reached before base-layer termination."""


@lru_cache(maxsize=1)
def _load_extension() -> _RectangularEWACompositorExtension:
    """Load the fused CUDA extension once per process."""
    module_name = "lib_rectangular_ewa_compositor"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        root_dir = os.path.abspath(os.path.dirname(__file__))
        cpp_standard = 17
        nvcc_flags = [
            f"-std=c++{cpp_standard}",
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.name == "posix":
            cflags = [f"-std=c++{cpp_standard}"]
            nvcc_flags.append("-Xcompiler=-fno-strict-aliasing")
            ldflags = ["-lcuda", "-lnvrtc"]
        elif os.name == "nt":
            cflags = [f"/std:c++{cpp_standard}", "/DNOMINMAX"]
            ldflags = ["cuda.lib", "advapi32.lib", "nvrtc.lib"]
        else:
            msg = f"Unsupported operating system: {os.name}."
            raise RuntimeError(msg)
        build_dir = torch.utils.cpp_extension._get_build_directory(
            module_name,
            verbose=True,
        )
        module = torch.utils.cpp_extension.load(
            name=module_name,
            sources=[
                os.path.join(root_dir, "rectangular_ewa_compositor.cpp"),
                os.path.join(root_dir, "rectangular_ewa_compositor.cu"),
            ],
            extra_cflags=cflags,
            extra_cuda_cflags=nvcc_flags,
            extra_ldflags=ldflags,
            extra_include_paths=[root_dir],
            build_directory=build_dir,
            with_cuda=True,
            verbose=True,
        )
    return cast(_RectangularEWACompositorExtension, module)


def _flatten_pixel_mask(
    pixel_mask: torch.Tensor | None,
    *,
    device: torch.device,
    width: int,
    height: int,
) -> torch.Tensor:
    """Normalize the pipeline mask to one optional rectangular image mask."""
    if pixel_mask is None:
        return torch.empty((0,), device=device, dtype=torch.bool)
    expected_shape = (1, height, width, 1)
    if pixel_mask.shape != expected_shape:
        raise ValueError("EWA pixel mask must match the rendered image shape.")
    if pixel_mask.device != device:
        raise ValueError("EWA pixel mask must share the rendered device.")
    return (pixel_mask[0, :, :, 0] > 0).contiguous()


def _terminal_state_masks(
    *,
    pixel_mask: torch.Tensor,
    base_transmittance: torch.Tensor,
    transmittance_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return enabled open and closed masks for terminal EWA state."""
    if pixel_mask.numel() == 0:
        enabled = torch.ones_like(base_transmittance, dtype=torch.bool)
    else:
        enabled = pixel_mask[None, :, :, None]
    base_open = enabled & (base_transmittance >= transmittance_cutoff)
    return base_open, enabled & ~base_open


def _validate_inputs(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    environment_mask: torch.Tensor,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
) -> None:
    """Check the stable public tensor contract before dispatching to CUDA."""
    point_count = means2d.shape[0]
    if means2d.shape != (point_count, 2):
        raise ValueError("EWA means2d must have shape (N, 2).")
    if conics.shape != (point_count, 3):
        raise ValueError("EWA conics must have shape (N, 3).")
    if opacities.shape != (point_count,):
        raise ValueError("EWA opacities must have shape (N,).")
    if colors.shape != (point_count, 3):
        raise ValueError("EWA colors must have shape (N, 3).")
    if depths.shape != (point_count,):
        raise ValueError("EWA depths must have shape (N,).")
    if environment_mask.shape != (point_count,):
        raise ValueError("EWA environment mask must have shape (N,).")
    if environment_mask.dtype != torch.bool:
        raise ValueError("EWA environment mask must use bool dtype.")
    if width <= 0 or height <= 0:
        raise ValueError("EWA image dimensions must be positive.")
    expected_offsets_shape = (
        1,
        (height + _TILE_HEIGHT - 1) // _TILE_HEIGHT,
        (width + _TILE_WIDTH - 1) // _TILE_WIDTH,
    )
    if offsets.shape != expected_offsets_shape:
        raise ValueError("EWA tile offsets do not match the render shape.")
    if offsets.dtype != torch.int32 or flatten_ids.dtype != torch.int32:
        raise ValueError("EWA tile lists must use int32 indices.")


class _RectangularEWAComposite(torch.autograd.Function):
    """Autograd adapter for the recovered rectangular EWA recurrence."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        environment_mask: torch.Tensor,
        pixel_mask: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        alpha_cutoff: float,
        transmittance_cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Composite RGB, alpha, and expected base-layer depth."""
        needs_grad = any(ctx.needs_input_grad[:5])
        (
            rgb,
            all_transmittance,
            raw_depth,
            base_transmittance,
            last_all,
            last_base,
        ) = _load_extension().forward(
            means2d.contiguous(),
            conics.contiguous(),
            opacities.contiguous(),
            colors.contiguous(),
            depths.contiguous(),
            environment_mask.contiguous(),
            pixel_mask.contiguous(),
            offsets.contiguous(),
            flatten_ids.contiguous(),
            width,
            height,
            alpha_cutoff,
            transmittance_cutoff,
        )
        base_open, base_closed = _terminal_state_masks(
            pixel_mask=pixel_mask,
            base_transmittance=base_transmittance,
            transmittance_cutoff=transmittance_cutoff,
        )
        epsilon = torch.finfo(raw_depth.dtype).eps
        all_alpha = 1.0 - all_transmittance
        base_alpha = 1.0 - base_transmittance
        terminal_rgb = torch.where(
            base_closed.expand_as(rgb),
            rgb / all_alpha.clamp_min(epsilon),
            rgb,
        )
        expected_depth = torch.where(
            base_closed,
            raw_depth / base_alpha.clamp_min(epsilon),
            raw_depth,
        )
        reported_all_transmittance = torch.where(
            base_open,
            all_transmittance,
            torch.zeros_like(all_transmittance),
        )
        ctx.width = width
        ctx.height = height
        ctx.alpha_cutoff = alpha_cutoff
        ctx.transmittance_cutoff = transmittance_cutoff
        if needs_grad:
            ctx.save_for_backward(
                means2d,
                conics,
                opacities,
                colors,
                depths,
                environment_mask,
                pixel_mask,
                offsets,
                flatten_ids,
                rgb,
                all_transmittance,
                raw_depth,
                base_transmittance,
                last_all,
                last_base,
            )
        return terminal_rgb, 1.0 - reported_all_transmittance, expected_depth

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        rgb_gradient: torch.Tensor | None,
        alpha_gradient: torch.Tensor | None,
        expected_depth_gradient: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        """Map RGB, alpha, and expected-depth adjoints to packed inputs."""
        (
            means2d,
            conics,
            opacities,
            colors,
            depths,
            environment_mask,
            pixel_mask,
            offsets,
            flatten_ids,
            rgb,
            all_transmittance,
            raw_depth,
            base_transmittance,
            last_all,
            last_base,
        ) = ctx.saved_tensors
        image_shape = (1, ctx.height, ctx.width, 1)
        rgb_shape = (1, ctx.height, ctx.width, 3)
        if rgb_gradient is None:
            rgb_gradient = torch.zeros(
                rgb_shape,
                device=means2d.device,
                dtype=means2d.dtype,
            )
        if alpha_gradient is None:
            alpha_gradient = torch.zeros(
                image_shape,
                device=means2d.device,
                dtype=means2d.dtype,
            )
        if expected_depth_gradient is None:
            expected_depth_gradient = torch.zeros(
                image_shape,
                device=means2d.device,
                dtype=means2d.dtype,
            )
        base_open, base_closed = _terminal_state_masks(
            pixel_mask=pixel_mask,
            base_transmittance=base_transmittance,
            transmittance_cutoff=ctx.transmittance_cutoff,
        )
        epsilon = torch.finfo(base_transmittance.dtype).eps
        all_alpha = 1.0 - all_transmittance
        base_alpha = 1.0 - base_transmittance
        raw_rgb_gradient = torch.where(
            base_closed.expand_as(rgb_gradient),
            rgb_gradient / all_alpha.clamp_min(epsilon),
            rgb_gradient,
        )
        rgb_all_transmittance_gradient = (
            (rgb_gradient * rgb).sum(dim=-1, keepdim=True)
            / all_alpha.square().clamp_min(epsilon)
        )
        all_transmittance_gradient = torch.where(
            base_open,
            -alpha_gradient,
            torch.where(
                base_closed,
                rgb_all_transmittance_gradient,
                torch.zeros_like(alpha_gradient),
            ),
        )
        raw_depth_gradient = torch.where(
            base_closed,
            expected_depth_gradient / base_alpha.clamp_min(epsilon),
            expected_depth_gradient,
        )
        base_transmittance_gradient = torch.where(
            base_closed,
            expected_depth_gradient * raw_depth
            / base_alpha.square().clamp_min(epsilon),
            torch.zeros_like(expected_depth_gradient),
        )
        gradients = _load_extension().backward(
            means2d.contiguous(),
            conics.contiguous(),
            opacities.contiguous(),
            colors.contiguous(),
            depths.contiguous(),
            environment_mask.contiguous(),
            pixel_mask.contiguous(),
            offsets.contiguous(),
            flatten_ids.contiguous(),
            all_transmittance.contiguous(),
            base_transmittance.contiguous(),
            last_all.contiguous(),
            last_base.contiguous(),
            raw_rgb_gradient.contiguous(),
            raw_depth_gradient.contiguous(),
            base_transmittance_gradient.contiguous(),
            all_transmittance_gradient.contiguous(),
            ctx.width,
            ctx.height,
            ctx.alpha_cutoff,
            ctx.transmittance_cutoff,
        )
        return (*gradients, None, None, None, None, None, None, None, None)


def composite_rectangular_ewa(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    environment_mask: torch.Tensor,
    pixel_mask: torch.Tensor | None,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    alpha_cutoff: float = 1.0 / 255.0,
    transmittance_cutoff: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite depth-sorted packed rows with 16x8 EWA tiles."""
    _validate_inputs(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        colors=colors,
        depths=depths,
        environment_mask=environment_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
    )
    flattened_pixel_mask = _flatten_pixel_mask(
        pixel_mask,
        device=means2d.device,
        width=width,
        height=height,
    )
    return _RectangularEWAComposite.apply(
        means2d,
        conics,
        opacities,
        colors,
        depths,
        environment_mask,
        flattened_pixel_mask,
        offsets,
        flatten_ids,
        width,
        height,
        alpha_cutoff,
        transmittance_cutoff,
    )


@torch.no_grad()
def masked_rectangular_ewa_contribution(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    pixel_mask: torch.Tensor | None,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    alpha_cutoff: float = 1.0 / 255.0,
    transmittance_cutoff: float = 0.01,
) -> torch.Tensor:
    """Accumulate masked current-frame EWA contribution for packed rows."""
    empty_colors = torch.empty(
        (means2d.shape[0], 3),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    empty_depths = torch.empty(
        (means2d.shape[0],),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    _validate_inputs(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        colors=empty_colors,
        depths=empty_depths,
        environment_mask=torch.zeros_like(opacities, dtype=torch.bool),
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
    )
    flattened_pixel_mask = _flatten_pixel_mask(
        pixel_mask,
        device=means2d.device,
        width=width,
        height=height,
    )
    return _load_extension().masked_contribution(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        flattened_pixel_mask,
        offsets.contiguous(),
        flatten_ids.contiguous(),
        width,
        height,
        alpha_cutoff,
        transmittance_cutoff,
    )


@torch.no_grad()
def masked_rectangular_ewa_forward_contribution(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    environment_mask: torch.Tensor,
    pixel_mask: torch.Tensor | None,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    alpha_cutoff: float = 1.0 / 255.0,
    transmittance_cutoff: float = 0.01,
) -> torch.Tensor:
    """Accumulate samples reached by EWA before base-layer termination."""
    point_count = means2d.shape[0]
    empty_colors = torch.empty(
        (point_count, 3),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    empty_depths = torch.empty(
        (point_count,),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    _validate_inputs(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        colors=empty_colors,
        depths=empty_depths,
        environment_mask=environment_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
    )
    flattened_pixel_mask = _flatten_pixel_mask(
        pixel_mask,
        device=means2d.device,
        width=width,
        height=height,
    )
    contribution = _load_extension().masked_forward_contribution(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        environment_mask.contiguous(),
        flattened_pixel_mask,
        offsets.contiguous(),
        flatten_ids.contiguous(),
        width,
        height,
        alpha_cutoff,
        transmittance_cutoff,
    )
    if contribution.shape != (point_count,):
        raise RuntimeError("EWA forward contribution has an invalid shape.")
    return contribution


@torch.no_grad()
def masked_rectangular_ewa_forward_visibility(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    environment_mask: torch.Tensor,
    pixel_mask: torch.Tensor | None,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    alpha_cutoff: float = 1.0 / 255.0,
    transmittance_cutoff: float = 0.01,
) -> torch.Tensor:
    """Mark samples reached by masked EWA before base-layer termination."""
    point_count = means2d.shape[0]
    empty_colors = torch.empty(
        (point_count, 3),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    empty_depths = torch.empty(
        (point_count,),
        device=means2d.device,
        dtype=means2d.dtype,
    )
    _validate_inputs(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        colors=empty_colors,
        depths=empty_depths,
        environment_mask=environment_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=width,
        height=height,
    )
    flattened_pixel_mask = _flatten_pixel_mask(
        pixel_mask,
        device=means2d.device,
        width=width,
        height=height,
    )
    visibility = _load_extension().masked_forward_visibility(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        environment_mask.contiguous(),
        flattened_pixel_mask,
        offsets.contiguous(),
        flatten_ids.contiguous(),
        width,
        height,
        alpha_cutoff,
        transmittance_cutoff,
    )
    if visibility.shape != (point_count,):
        raise RuntimeError("EWA forward visibility has an invalid shape.")
    return visibility.to(dtype=torch.bool)
