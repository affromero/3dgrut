"""Gauss-Hermite moment-preserving Gaussian split primitives."""

import math
from typing import NamedTuple

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from threedgrut.utils.misc import quaternion_to_so3

GAUSS_HERMITE_CHILD_COUNT = 3
GAUSS_HERMITE_NODES = (-1.0, 0.0, 1.0)
GAUSS_HERMITE_WEIGHTS = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)


class MomentPreservingSplitChildren(NamedTuple):
    """Physical parameters for child-major ordered split rows."""

    positions: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor


def _validate_inputs(
    *,
    positions: torch.Tensor,
    physical_scales: torch.Tensor,
    rotations: torch.Tensor,
    physical_opacities: torch.Tensor,
    beta: float,
) -> None:
    """Validate physical Gaussian parameters before constructing children."""
    parent_count = positions.shape[0]
    expected_shapes = (
        ("positions", positions, (parent_count, 3)),
        ("physical scales", physical_scales, (parent_count, 3)),
        ("rotations", rotations, (parent_count, 4)),
        ("physical opacities", physical_opacities, (parent_count, 1)),
    )
    for name, value, expected_shape in expected_shapes:
        if value.shape != expected_shape:
            message = (
                f"Moment-preserving split {name} must have shape "
                f"{expected_shape}, got {tuple(value.shape)}."
            )
            raise ValueError(message)
        if value.device != positions.device:
            message = (
                "Moment-preserving split tensors must share device "
                f"{positions.device}; {name} uses {value.device}."
            )
            raise ValueError(message)
        if value.dtype != positions.dtype:
            message = (
                "Moment-preserving split tensors must share dtype "
                f"{positions.dtype}; {name} uses {value.dtype}."
            )
            raise ValueError(message)
        if not bool(torch.isfinite(value).all()):
            message = f"Moment-preserving split {name} must be finite."
            raise RuntimeError(message)
    if bool((physical_scales <= 0.0).any()):
        message = "Moment-preserving split physical scales must be positive."
        raise RuntimeError(message)
    invalid_opacity = (physical_opacities < 0.0) | (physical_opacities >= 1.0)
    if bool(invalid_opacity.any()):
        message = (
            "Moment-preserving split physical opacities must be in [0, 1)."
        )
        raise RuntimeError(message)
    if not math.isfinite(beta) or not 0.0 < beta < 1.0:
        message = (
            "Moment-preserving split beta must be finite and in (0, 1), "
            f"got {beta}."
        )
        raise ValueError(message)


@jaxtyped(typechecker=beartype)
def gauss_hermite_split_children(
    *,
    positions: Float[torch.Tensor, "parent 3"],
    physical_scales: Float[torch.Tensor, "parent 3"],
    rotations: Float[torch.Tensor, "parent 4"],
    physical_opacities: Float[torch.Tensor, "parent 1"],
    beta: float,
) -> MomentPreservingSplitChildren:
    """Construct deterministic children preserving Gaussian moments.

    Under the declared Gauss-Hermite weights, the component geometry preserves
    each parent's mean, covariance, fourth central moment on the selected axis,
    and odd central moments through degree five. Those weights allocate parent
    optical depth exactly. The resulting alpha values preserve transmittance
    only in the co-located limit, not under arbitrary rendered displacement.
    """
    _validate_inputs(
        positions=positions,
        physical_scales=physical_scales,
        rotations=rotations,
        physical_opacities=physical_opacities,
        beta=beta,
    )
    parent_count = positions.shape[0]
    largest_axis = torch.argmax(physical_scales, dim=1)
    local_axis = torch.nn.functional.one_hot(
        largest_axis,
        num_classes=3,
    ).to(dtype=positions.dtype)
    world_axis = torch.bmm(
        quaternion_to_so3(rotations),
        local_axis.unsqueeze(-1),
    ).squeeze(-1)
    largest_scale = physical_scales.gather(
        1,
        largest_axis[:, None],
    )
    offset = world_axis * largest_scale * math.sqrt(3.0 * (1.0 - beta))
    child_positions = torch.cat(
        tuple(positions + node * offset for node in GAUSS_HERMITE_NODES),
        dim=0,
    )

    child_scales = physical_scales.repeat(
        GAUSS_HERMITE_CHILD_COUNT,
        1,
    )
    child_largest_axis = largest_axis.repeat(
        GAUSS_HERMITE_CHILD_COUNT,
    )[:, None]
    child_largest_scale = largest_scale.repeat(
        GAUSS_HERMITE_CHILD_COUNT,
        1,
    ) * math.sqrt(beta)
    child_scales.scatter_(
        1,
        child_largest_axis,
        child_largest_scale,
    )

    child_weights = torch.tensor(
        GAUSS_HERMITE_WEIGHTS,
        dtype=positions.dtype,
        device=positions.device,
    ).repeat_interleave(parent_count)[:, None]
    repeated_opacities = physical_opacities.repeat(
        GAUSS_HERMITE_CHILD_COUNT,
        1,
    )
    child_opacities = -torch.expm1(
        child_weights * torch.log1p(-repeated_opacities)
    )
    return MomentPreservingSplitChildren(
        positions=child_positions,
        scales=child_scales,
        opacities=child_opacities,
    )
