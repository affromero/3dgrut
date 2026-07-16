"""Isolated visibility-and-shape-derived scale-shape topology primitives for 3DGUT."""

import math
from typing import NamedTuple

import torch

from threedgrut.utils.misc import quaternion_to_so3

DETERMINISTIC_CHILD_OFFSET = 0.6744800209999084
DETERMINISTIC_CHILD_SCALE = 0.8


class ScaleShapeThresholds(NamedTuple):
    """Frozen physical and status gates for the matched topology arm."""

    anisotropy: float
    min_largest_scale: float
    min_observations: int


def scale_shape_split_mask(
    *,
    split_candidates: torch.Tensor,
    physical_scales: torch.Tensor,
    observation_count: torch.Tensor,
    thresholds: ScaleShapeThresholds,
) -> torch.Tensor:
    """Return the strict, status-aware reroute subset of split candidates."""
    point_count = split_candidates.shape[0]
    if split_candidates.shape != (point_count,) or split_candidates.dtype != (torch.bool):
        message = "scale-shape split candidates must be a bool vector."
        raise ValueError(message)
    if physical_scales.shape != (point_count, 3):
        message = "scale-shape physical scales must have shape (N, 3)."
        raise ValueError(message)
    if observation_count.shape not in (
        (point_count,),
        (point_count, 1),
    ):
        message = "scale-shape observation counts must have shape (N,) " "or (N, 1)."
        raise ValueError(message)
    if observation_count.is_floating_point() or observation_count.dtype == (torch.bool):
        message = "scale-shape observation counts must use an integer dtype."
        raise ValueError(message)
    if not math.isfinite(thresholds.anisotropy) or thresholds.anisotropy <= 1.0:
        message = "scale-shape anisotropy threshold must be finite and " "greater than one."
        raise ValueError(message)
    if not math.isfinite(thresholds.min_largest_scale) or thresholds.min_largest_scale <= 0.0:
        message = "scale-shape minimum largest scale must be finite and " "positive."
        raise ValueError(message)
    if isinstance(thresholds.min_observations, bool) or thresholds.min_observations < 1:
        message = "scale-shape minimum observations must be a positive " "integer."
        raise ValueError(message)

    candidate_scales = physical_scales[split_candidates]
    if candidate_scales.numel() and (
        not bool(torch.isfinite(candidate_scales).all()) or bool((candidate_scales <= 0.0).any())
    ):
        message = "scale-shape split candidates require finite, positive " "physical scales."
        raise RuntimeError(message)
    largest_scale = physical_scales.amax(dim=1)
    smallest_scale = physical_scales.amin(dim=1)
    anisotropy = largest_scale / smallest_scale
    observations = observation_count.reshape(-1)
    return (
        split_candidates
        & (observations >= thresholds.min_observations)
        & (anisotropy > thresholds.anisotropy)
        & (largest_scale > thresholds.min_largest_scale)
    )


def deterministic_split_children(
    *,
    positions: torch.Tensor,
    physical_scales: torch.Tensor,
    rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place two children symmetrically along each parent's largest axis."""
    point_count = positions.shape[0]
    if positions.shape != (point_count, 3):
        message = "deterministic split positions must have shape (N, 3)."
        raise ValueError(message)
    if physical_scales.shape != (point_count, 3):
        message = "deterministic split scales must have shape (N, 3)."
        raise ValueError(message)
    if rotations.shape != (point_count, 4):
        message = "deterministic split rotations must have shape (N, 4)."
        raise ValueError(message)
    if not bool(torch.isfinite(positions).all()):
        message = "deterministic split positions must be finite."
        raise RuntimeError(message)
    if not bool(torch.isfinite(rotations).all()):
        message = "deterministic split rotations must be finite."
        raise RuntimeError(message)
    if not bool(torch.isfinite(physical_scales).all()) or bool((physical_scales <= 0.0).any()):
        message = "deterministic split physical scales must be finite and positive."
        raise RuntimeError(message)

    largest_axis = torch.argmax(physical_scales, dim=1)
    local_axes = torch.nn.functional.one_hot(
        largest_axis,
        num_classes=3,
    ).to(dtype=positions.dtype)
    world_axes = torch.bmm(
        quaternion_to_so3(rotations),
        local_axes.unsqueeze(-1),
    ).squeeze(-1)
    largest_scale = physical_scales.gather(1, largest_axis[:, None])
    offsets = world_axes * largest_scale * DETERMINISTIC_CHILD_OFFSET
    child_positions = torch.cat(
        (positions + offsets, positions - offsets),
        dim=0,
    )
    child_scales = torch.cat(
        (
            physical_scales * DETERMINISTIC_CHILD_SCALE,
            physical_scales * DETERMINISTIC_CHILD_SCALE,
        ),
        dim=0,
    )
    return child_positions, child_scales


def transmittance_preserving_child_opacity(
    physical_opacity: torch.Tensor,
) -> torch.Tensor:
    """Split base alpha into two children with exact co-located alpha algebra."""
    point_count = physical_opacity.shape[0]
    if physical_opacity.shape != (point_count, 1):
        message = "deterministic split opacity must have shape (N, 1)."
        raise ValueError(message)
    if not bool(torch.isfinite(physical_opacity).all()) or bool(
        ((physical_opacity < 0.0) | (physical_opacity >= 1.0)).any()
    ):
        message = "deterministic split physical opacity must be finite and in [0, 1)."
        raise RuntimeError(message)

    child_opacity = -torch.expm1(0.5 * torch.log1p(-physical_opacity))
    return torch.cat((child_opacity, child_opacity), dim=0)
