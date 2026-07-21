"""Tests for exact native range-ray inverse-sensor evidence."""

import pytest
import torch
from threedgrut.datasets.protocols import Batch
from threedgrut.model.native_ray_inverse_sensor import (
    interval_transmittance_losses,
    shift_ray_origins,
)


def _batch() -> Batch:
    depth = torch.tensor([[[[2.0], [3.0]]]])
    return Batch(
        rays_ori=torch.zeros((1, 1, 2, 3)),
        rays_dir=torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]),
        T_to_world=torch.eye(4).unsqueeze(0),
        depth_gt=depth,
        depth_ray_z=torch.ones_like(depth),
        mask=torch.ones_like(depth),
    )


def test_shift_ray_origins_preserves_collinear_metric_rays() -> None:
    """A metric range shift changes only each ray's origin."""
    batch = _batch()
    distance = torch.tensor([[[[0.5], [1.25]]]])

    shifted = shift_ray_origins(batch, distance)

    assert torch.equal(shifted.rays_dir, batch.rays_dir)
    assert torch.equal(shifted.T_to_world, batch.T_to_world)
    assert torch.allclose(
        shifted.rays_ori,
        torch.tensor([[[[0.5, 0.0, 0.0], [0.0, 1.25, 0.0]]]]),
    )


def test_interval_losses_recover_known_prefix_and_window() -> None:
    """Porter-Duff products isolate prefix and return probabilities."""
    prefix_transmittance = 0.8
    window_transmittance = 0.5
    suffix_transmittance = 0.75
    full_opacity = torch.tensor(
        [
            [
                [
                    [
                        1.0
                        - (
                            prefix_transmittance
                            * window_transmittance
                            * suffix_transmittance
                        )
                    ]
                ]
            ]
        ],
        requires_grad=True,
    )
    lower_opacity = torch.tensor(
        [[[[1.0 - (window_transmittance * suffix_transmittance)]]]],
        requires_grad=True,
    )
    upper_opacity = torch.tensor(
        [[[[1.0 - suffix_transmittance]]]],
        requires_grad=True,
    )

    free, occupied, prefix_violation, window_violation = (
        interval_transmittance_losses(
            full_opacity=full_opacity,
            after_lower_opacity=lower_opacity,
            after_upper_opacity=upper_opacity,
            valid_mask=torch.ones_like(full_opacity),
            return_weight=torch.ones_like(full_opacity),
            probability_floor=1e-6,
        )
    )

    assert free.item() == pytest.approx(
        -torch.log(torch.tensor(prefix_transmittance)).item()
    )
    assert occupied.item() == pytest.approx(
        -torch.log(torch.tensor(1.0 - window_transmittance)).item()
    )
    assert prefix_violation.item() == 0.0
    assert window_violation.item() == 0.0
    (free + occupied).backward()
    assert full_opacity.grad is not None
    assert lower_opacity.grad is not None
    assert upper_opacity.grad is not None
    assert torch.isfinite(full_opacity.grad).all()
    assert torch.isfinite(lower_opacity.grad).all()
    assert torch.isfinite(upper_opacity.grad).all()


def test_interval_losses_mask_invalid_rays_and_report_ordering() -> None:
    """Invalid pixels do not contribute and finite ordering errors report."""
    full = torch.tensor([[[[0.4], [0.2]]]])
    lower = torch.tensor([[[[0.5], [float("nan")]]]])
    upper = torch.tensor([[[[0.6], [0.0]]]])
    mask = torch.tensor([[[[1.0], [0.0]]]])

    free, occupied, prefix_violation, window_violation = (
        interval_transmittance_losses(
            full_opacity=full,
            after_lower_opacity=lower,
            after_upper_opacity=upper,
            valid_mask=mask,
            return_weight=torch.ones_like(mask),
            probability_floor=1e-6,
        )
    )

    assert torch.isfinite(free)
    assert torch.isfinite(occupied)
    assert prefix_violation.item() == 1.0
    assert window_violation.item() == 1.0


def test_interval_return_loss_uses_amplitude_as_confidence() -> None:
    """Weak Leica returns contribute less occupied evidence."""
    full = torch.tensor([[[[0.75], [0.75]]]])
    lower = torch.tensor([[[[0.5], [0.2]]]])
    upper = torch.tensor([[[[0.0], [0.0]]]])
    weights = torch.tensor([[[[0.9], [0.1]]]])

    _, occupied, _, _ = interval_transmittance_losses(
        full_opacity=full,
        after_lower_opacity=lower,
        after_upper_opacity=upper,
        valid_mask=torch.ones_like(full),
        return_weight=weights,
        probability_floor=1e-6,
    )

    expected = (
        0.9 * -torch.log(torch.tensor(0.5))
        + 0.1 * -torch.log(torch.tensor(0.2))
    )
    assert occupied.item() == pytest.approx(expected.item())
