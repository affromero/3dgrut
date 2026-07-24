"""Tests for fixed native projection-distortion replay."""

import pytest
import torch
from threedgrut.datasets.utils import create_pixel_coords
from threedgrut.model.native_projection_distortion import (
    sample_native_distortion,
    solve_native_base_pixels,
    warp_native_render,
)


def test_zero_native_distortion_preserves_render() -> None:
    generator = torch.Generator().manual_seed(7)
    pred_rgb = torch.rand((1, 5, 7, 3), generator=generator)
    pixel_coords = create_pixel_coords(7, 5)

    warped = warp_native_render(
        pred_rgb,
        pixel_coords,
        torch.zeros((12, 12, 2)),
        focal_length=torch.tensor([4.0, 4.0]),
        sign=1.0,
        iterations=3,
    )

    assert torch.allclose(warped, pred_rgb, atol=1e-6)


def test_native_distortion_warp_preserves_render_gradient() -> None:
    pred_rgb = torch.rand((1, 5, 7, 3), requires_grad=True)
    pixel_coords = create_pixel_coords(7, 5)
    distortion = torch.zeros((12, 12, 2))
    distortion[..., 0] = 0.01

    warped = warp_native_render(
        pred_rgb,
        pixel_coords,
        distortion,
        focal_length=torch.tensor([4.0, 4.0]),
        sign=1.0,
        iterations=3,
    )
    warped.square().mean().backward()

    assert pred_rgb.grad is not None
    assert torch.isfinite(pred_rgb.grad).all()
    assert pred_rgb.grad.abs().max() > 0


def test_constant_native_distortion_inverse_round_trip() -> None:
    field = torch.empty((12, 12, 2))
    field[..., 0] = 0.25
    field[..., 1] = -0.5
    target = torch.tensor([[[[10.5, 20.5]]]])
    focal = torch.tensor([8.0, 6.0])

    base = solve_native_base_pixels(
        field,
        target,
        focal_length=focal,
        resolution=(40, 30),
        sign=1.0,
        iterations=2,
    )
    forward = base + focal.reshape(1, 1, 1, 2) * sample_native_distortion(
        field,
        base,
        resolution=(40, 30),
    )

    assert torch.allclose(base, torch.tensor([[[[8.5, 23.5]]]]))
    assert torch.allclose(forward, target)


def test_native_distortion_samples_four_neighbors() -> None:
    field = torch.zeros((2, 2, 2))
    field[0, 1, 0] = 2.0
    field[1, 0, 0] = 4.0
    field[1, 1, 0] = 6.0

    sampled = sample_native_distortion(
        field,
        torch.tensor([[[[1.0, 1.0]]]]),
        resolution=(2, 2),
    )

    assert torch.allclose(sampled, torch.tensor([[[[3.0, 0.0]]]]))


def test_native_distortion_rejects_invalid_sign() -> None:
    with pytest.raises(ValueError, match="sign"):
        solve_native_base_pixels(
            torch.zeros((12, 12, 2)),
            torch.zeros((1, 1, 1, 2)),
            focal_length=torch.ones(2),
            resolution=(1, 1),
            sign=0.0,
            iterations=1,
        )
