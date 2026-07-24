"""Fused differentiable compositor for tiled layered-depth buffers."""

from __future__ import annotations

import importlib
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

import torch
import torch.utils.cpp_extension

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


class _LayeredDepthCompositorExtension(Protocol):
    """Typed surface exposed by the fused CUDA extension."""

    def forward(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
        alpha_cutoff: float,
        maximum_chunk_count: int,
        save_chunk_transparencies: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render layered buffers and optional backward state."""

    def backward(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        chunk_transparencies: torch.Tensor,
        raw_depth_gradient: torch.Tensor,
        transparency_gradient: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
        alpha_cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Accumulate gradients for packed layered-depth parameters."""

    def masked_ewa_contribution(
        self,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        pixel_mask: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
        alpha_cutoff: float,
        transmittance_cutoff: float,
    ) -> torch.Tensor:
        """Accumulate masked front-to-back EWA contribution per row."""


@lru_cache(maxsize=1)
def _load_extension() -> _LayeredDepthCompositorExtension:
    """Load the fused CUDA extension once per process."""
    module_name = "lib_layered_depth_compositor"
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
                os.path.join(root_dir, "layered_depth_compositor.cpp"),
                os.path.join(root_dir, "layered_depth_compositor.cu"),
            ],
            extra_cflags=cflags,
            extra_cuda_cflags=nvcc_flags,
            extra_ldflags=ldflags,
            extra_include_paths=[root_dir],
            build_directory=build_dir,
            with_cuda=True,
            verbose=True,
        )
    return cast(_LayeredDepthCompositorExtension, module)


def _maximum_chunk_count(
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
) -> int:
    """Return the largest 16x16 tile-list chunk count."""
    flat_offsets = offsets.flatten()
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
    largest_range = int(
        torch.max(offsets_with_end[1:] - offsets_with_end[:-1]).item()
    )
    return max(1, (largest_range + 255) // 256)


class _LayeredDepthComposite(torch.autograd.Function):
    """Autograd adapter for the fused packed layered-depth compositor."""

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
        alpha_cutoff: float,
        maximum_chunk_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render raw depth, transparency, and non-differentiable median depth."""
        needs_grad = any(ctx.needs_input_grad[:4])
        raw_depth, transparency, median_depth, chunk_transparencies = (
            _load_extension().forward(
                means2d.contiguous(),
                conics.contiguous(),
                opacities.contiguous(),
                depths.contiguous(),
                offsets.contiguous(),
                flatten_ids.contiguous(),
                width,
                height,
                tile_size,
                alpha_cutoff,
                maximum_chunk_count,
                needs_grad,
            )
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.alpha_cutoff = alpha_cutoff
        if needs_grad:
            ctx.save_for_backward(
                means2d,
                conics,
                opacities,
                depths,
                offsets,
                flatten_ids,
                chunk_transparencies,
            )
        ctx.mark_non_differentiable(median_depth)
        return raw_depth, transparency, median_depth

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        raw_depth_gradient: torch.Tensor | None,
        transparency_gradient: torch.Tensor | None,
        median_depth_gradient: torch.Tensor | None,
    ) -> tuple[
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
    ]:
        """Propagate gradients through raw depth and transparency outputs."""
        del median_depth_gradient
        (
            means2d,
            conics,
            opacities,
            depths,
            offsets,
            flatten_ids,
            chunk_transparencies,
        ) = ctx.saved_tensors
        output_shape = (1, ctx.height, ctx.width, 1)
        if raw_depth_gradient is None:
            raw_depth_gradient = torch.zeros(
                output_shape,
                device=means2d.device,
                dtype=means2d.dtype,
            )
        if transparency_gradient is None:
            transparency_gradient = torch.zeros(
                output_shape,
                device=means2d.device,
                dtype=means2d.dtype,
            )
        gradients = _load_extension().backward(
            means2d.contiguous(),
            conics.contiguous(),
            opacities.contiguous(),
            depths.contiguous(),
            offsets.contiguous(),
            flatten_ids.contiguous(),
            chunk_transparencies,
            raw_depth_gradient.contiguous(),
            transparency_gradient.contiguous(),
            ctx.width,
            ctx.height,
            ctx.tile_size,
            ctx.alpha_cutoff,
        )
        return (*gradients, None, None, None, None, None, None, None)


def composite_layered_depth(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    depths: torch.Tensor,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    alpha_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Composite one depth-sorted layer with a fused CUDA implementation."""
    return _LayeredDepthComposite.apply(
        means2d,
        conics,
        opacities,
        depths,
        offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        alpha_cutoff,
        _maximum_chunk_count(offsets, flatten_ids),
    )


@torch.no_grad()
def masked_ewa_contribution(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    pixel_mask: torch.Tensor | None,
    offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
    tile_size: int,
    alpha_cutoff: float,
    transmittance_cutoff: float,
) -> torch.Tensor:
    """Accumulate one masked classic-EWA contribution value per packed row."""
    if pixel_mask is None:
        flattened_pixel_mask = torch.empty(
            (0,),
            device=means2d.device,
            dtype=torch.bool,
        )
    else:
        expected_shape = (1, height, width, 1)
        if pixel_mask.shape != expected_shape:
            raise ValueError(
                "EWA contribution mask must match the rendered image shape."
            )
        if pixel_mask.device != means2d.device:
            raise ValueError(
                "EWA contribution mask must share the rendered device."
            )
        flattened_pixel_mask = (pixel_mask[0, :, :, 0] > 0).contiguous()
    return _load_extension().masked_ewa_contribution(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        flattened_pixel_mask,
        offsets.contiguous(),
        flatten_ids.contiguous(),
        width,
        height,
        tile_size,
        alpha_cutoff,
        transmittance_cutoff,
    )
