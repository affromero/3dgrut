"""Tests for clean-rim high-frequency appearance supervision."""

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from threedgrut.model.losses import rim_high_frequency_loss


@jaxtyped(typechecker=beartype)
def _rim_depth_ray_z(
    batch: int,
    height: int,
    width: int,
) -> Float[torch.Tensor, "batch height width 1"]:
    return torch.full((batch, height, width, 1), 0.3)


@jaxtyped(typechecker=beartype)
def _center_depth_ray_z(
    batch: int,
    height: int,
    width: int,
) -> Float[torch.Tensor, "batch height width 1"]:
    return torch.full((batch, height, width, 1), 0.9)


@jaxtyped(typechecker=beartype)
def _loss(
    rgb_pred: Float[torch.Tensor, "batch height width channel"],
    rgb_gt: Float[torch.Tensor, "batch height width channel"],
    depth_ray_z: Float[torch.Tensor, "batch height width 1"],
    mask: Float[torch.Tensor, "batch height width 1"] | None = None,
) -> Float[torch.Tensor, ""]:
    return rim_high_frequency_loss(
        rgb_pred=rgb_pred,
        rgb_gt=rgb_gt,
        depth_ray_z=depth_ray_z,
        mask=mask,
        theta_min_deg=60.0,
        theta_max_deg=80.0,
        kernel_size=3,
        sigma=1.0,
        loss_type="charbonnier",
        charbonnier_epsilon=0.01,
    )


def test_rim_high_frequency_loss_is_zero_for_matching_images() -> None:
    """Identical prediction and target have no high-frequency residual."""
    torch.manual_seed(1)
    rgb = torch.rand((1, 12, 10, 3))
    loss = _loss(rgb, rgb, _rim_depth_ray_z(1, 12, 10))
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_rim_high_frequency_loss_ignores_pixels_outside_theta_band() -> None:
    """Pixels outside theta 60-80 do not contribute to the loss."""
    rgb_gt = torch.zeros((1, 8, 8, 3))
    rgb_pred = torch.rand((1, 8, 8, 3))
    loss = _loss(rgb_pred, rgb_gt, _center_depth_ray_z(1, 8, 8))
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_rim_high_frequency_loss_respects_rgb_mask() -> None:
    """The RGB validity mask removes otherwise valid rim pixels."""
    rgb_gt = torch.zeros((1, 8, 8, 3))
    rgb_pred = torch.rand((1, 8, 8, 3))
    mask = torch.zeros((1, 8, 8, 1))
    loss = _loss(rgb_pred, rgb_gt, _rim_depth_ray_z(1, 8, 8), mask)
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_rim_high_frequency_loss_detects_rim_texture_mismatch() -> None:
    """Rim-band high-frequency texture mismatch produces a positive loss."""
    yy = torch.arange(12).view(12, 1)
    xx = torch.arange(12).view(1, 12)
    checker = ((xx + yy) % 2).float().view(1, 12, 12, 1)
    rgb_gt = checker.expand(-1, -1, -1, 3)
    rgb_pred = torch.zeros_like(rgb_gt)
    loss = _loss(rgb_pred, rgb_gt, _rim_depth_ray_z(1, 12, 12))
    assert loss > 0.1
