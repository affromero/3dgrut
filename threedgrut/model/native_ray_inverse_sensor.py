"""Differentiable inverse-sensor evidence on native range rays."""

import math
from dataclasses import replace

import torch

from threedgrut.datasets.protocols import Batch


def shift_ray_origins(
    batch: Batch,
    distance: torch.Tensor,
) -> Batch:
    """Move every origin along its unchanged unit ray."""
    if batch.depth_gt is None:
        message = "Native-ray origin shifts require depth_gt."
        raise ValueError(message)
    if distance.shape != batch.depth_gt.shape:
        message = (
            "Native-ray shift distance must match depth_gt: "
            f"{tuple(distance.shape)} versus "
            f"{tuple(batch.depth_gt.shape)}."
        )
        raise ValueError(message)
    shifted_origins = (
        batch.rays_ori
        + distance.to(
            device=batch.rays_ori.device,
            dtype=batch.rays_ori.dtype,
        )
        * batch.rays_dir
    )
    return replace(
        batch,
        rays_ori=shifted_origins.contiguous(),
    )


def interval_transmittance_losses(
    *,
    full_opacity: torch.Tensor,
    after_lower_opacity: torch.Tensor,
    after_upper_opacity: torch.Tensor,
    valid_mask: torch.Tensor,
    return_weight: torch.Tensor,
    probability_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover free-prefix and return-window likelihoods.

    For residual transmittances ``T0``, ``T-``, and ``T+`` from
    collinear half-rays starting at the native origin and the lower and
    upper return-window bounds, respectively:

    ``T_free = T0 / T-`` and ``T_return = T- / T+``.
    """
    if not 0.0 < probability_floor < 0.5:
        message = (
            "Interval probability floor must be in (0, 0.5): "
            f"{probability_floor}."
        )
        raise ValueError(message)
    expected_shape = full_opacity.shape
    for name, value in (
        ("after_lower_opacity", after_lower_opacity),
        ("after_upper_opacity", after_upper_opacity),
        ("valid_mask", valid_mask),
        ("return_weight", return_weight),
    ):
        if value.shape != expected_shape:
            message = (
                f"{name} must match full_opacity: "
                f"{tuple(value.shape)} versus "
                f"{tuple(expected_shape)}."
            )
            raise ValueError(message)
    finite = (
        torch.isfinite(full_opacity)
        & torch.isfinite(after_lower_opacity)
        & torch.isfinite(after_upper_opacity)
        & torch.isfinite(return_weight)
    )
    valid = finite & (valid_mask > 0.5)
    if bool(
        torch.any(
            valid
            & ((return_weight < 0.0) | (return_weight > 1.0))
        ).detach()
    ):
        message = "Native-ray return weights must be in [0, 1]."
        raise ValueError(message)
    valid_count = valid.to(full_opacity.dtype).sum()
    if int(valid_count.detach().item()) == 0:
        message = "Native-ray interval loss has no finite valid rays."
        raise ValueError(message)

    maximum_opacity = 1.0 - probability_floor
    full = torch.clamp(
        full_opacity,
        min=0.0,
        max=maximum_opacity,
    )
    lower = torch.clamp(
        after_lower_opacity,
        min=0.0,
        max=maximum_opacity,
    )
    upper = torch.clamp(
        after_upper_opacity,
        min=0.0,
        max=maximum_opacity,
    )
    log_full_transmittance = torch.log1p(-full)
    log_lower_transmittance = torch.log1p(-lower)
    log_upper_transmittance = torch.log1p(-upper)
    log_probability_floor = math.log(probability_floor)
    log_free_transmittance = torch.clamp(
        log_full_transmittance - log_lower_transmittance,
        min=log_probability_floor,
        max=0.0,
    )
    log_return_transmittance = torch.clamp(
        log_lower_transmittance - log_upper_transmittance,
        min=log_probability_floor,
        max=0.0,
    )
    return_occupancy = torch.clamp(
        -torch.expm1(log_return_transmittance),
        min=probability_floor,
        max=1.0,
    )
    free_denominator = torch.clamp(valid_count, min=1.0)
    occupied_weight = torch.where(
        valid,
        return_weight.to(full_opacity.dtype),
        torch.zeros_like(full_opacity),
    )
    occupied_denominator = occupied_weight.sum()
    if float(occupied_denominator.detach().item()) <= 0.0:
        message = "Native-ray interval loss has no positive return weight."
        raise ValueError(message)
    free_loss = (
        torch.where(
            valid,
            -log_free_transmittance,
            torch.zeros_like(log_free_transmittance),
        ).sum()
        / free_denominator
    )
    occupied_error = torch.where(
        valid,
        -torch.log(return_occupancy),
        torch.zeros_like(return_occupancy),
    )
    occupied_loss = (
        (occupied_weight * occupied_error).sum()
        / occupied_denominator
    )
    monotonic_tolerance = 1e-5
    prefix_violation = (
        torch.where(
            valid,
            (lower > full + monotonic_tolerance).to(full.dtype),
            torch.zeros_like(full),
        ).sum()
        / free_denominator
    )
    window_violation = (
        torch.where(
            valid,
            (upper > lower + monotonic_tolerance).to(full.dtype),
            torch.zeros_like(full),
        ).sum()
        / free_denominator
    )
    return (
        free_loss,
        occupied_loss,
        prefix_violation,
        window_violation,
    )
