"""Tests for the RGB loss recovered from visibility-adaptive PTX."""

import pytest
import torch
from threedgrut.model.losses import (
    indexed_camera_loss_weight,
    native_huber_loss,
    fixed_image_loss_denominator,
)


def test_native_huber_matches_type_two_values_and_gradients() -> None:
    """Type two is quadratic inside 0.1 and L1 outside it."""
    residual = torch.tensor(
        (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2),
        requires_grad=True,
    )

    loss = native_huber_loss(residual, delta=0.1)
    loss.sum().backward()

    expected_loss = torch.tensor((0.15, 0.05, 0.0125, 0.0, 0.0125, 0.05, 0.15))
    expected_gradient = torch.tensor((-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0))
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(residual.grad, expected_gradient)


@pytest.mark.parametrize(
    ("valid_pixels", "expected_denominator"),
    (
        (10, 30.0),
        (8, 24.0),
        (5, 24.0),
        (0, 24.0),
    ),
)
def test_native_rgb_denominator_has_eighty_percent_floor(
    valid_pixels: int,
    expected_denominator: float,
) -> None:
    """Native reduction uses total RGB pixels and a valid-fraction floor."""
    rgb = torch.zeros((1, 2, 5, 3))
    mask = torch.zeros((1, 2, 5, 1))
    mask.reshape(-1)[:valid_pixels] = 1.0

    denominator = fixed_image_loss_denominator(rgb=rgb, mask=mask)

    torch.testing.assert_close(
        denominator,
        torch.tensor(expected_denominator),
    )


def test_native_rgb_denominator_without_mask_uses_all_pixels() -> None:
    """Unmasked images retain their full RGB element count."""
    rgb = torch.zeros((1, 3, 7, 3))

    denominator = fixed_image_loss_denominator(rgb=rgb, mask=None)

    torch.testing.assert_close(denominator, torch.tensor(63.0))


def test_native_rgb_denominator_includes_image_scale() -> None:
    """Native reduction divides by width * height * channels * image scale."""
    rgb = torch.zeros((1, 10, 10, 3))
    mask = torch.ones((1, 10, 10, 1))

    denominator = fixed_image_loss_denominator(
        rgb=rgb,
        mask=mask,
        image_scale=0.8,
    )

    torch.testing.assert_close(denominator, torch.tensor(240.0))


def test_native_rgb_denominator_accepts_a_configured_validity_floor() -> None:
    """A profile can set the validity floor without changing image scale."""
    rgb = torch.zeros((1, 2, 5, 3))
    mask = torch.zeros((1, 2, 5, 1))
    mask.reshape(-1)[:5] = 1.0

    denominator = fixed_image_loss_denominator(
        rgb=rgb,
        mask=mask,
        min_valid_fraction=2.0 / 3.0,
    )

    torch.testing.assert_close(denominator, torch.tensor(20.0))


@pytest.mark.parametrize("image_scale", (0.0, -1.0, float("nan")))
def test_native_rgb_denominator_rejects_invalid_image_scale(
    image_scale: float,
) -> None:
    """Malformed image scales fail instead of silently changing gradients."""
    rgb = torch.zeros((1, 2, 2, 3))

    with pytest.raises(ValueError, match="image scale"):
        fixed_image_loss_denominator(
            rgb=rgb,
            mask=None,
            image_scale=image_scale,
        )


@pytest.mark.parametrize("min_valid_fraction", (0.0, -1.0, 1.1, float("nan")))
def test_native_rgb_denominator_rejects_invalid_validity_floor(
    min_valid_fraction: float,
) -> None:
    """Malformed validity floors fail instead of changing gradients silently."""
    rgb = torch.zeros((1, 2, 2, 3))

    with pytest.raises(ValueError, match="minimum valid fraction"):
        fixed_image_loss_denominator(
            rgb=rgb,
            mask=None,
            min_valid_fraction=min_valid_fraction,
        )


def test_indexed_camera_weight_uses_explicit_physical_index() -> None:
    """Projected view IDs cannot substitute for the physical camera ID."""
    weights = [1.0, 1.0, 2.0, 2.0]

    assert indexed_camera_loss_weight(
        camera_index=2,
        configured_weights=weights,
    ) == 2.0


@pytest.mark.parametrize("camera_index", (-1, 4, 7))
def test_indexed_camera_weight_rejects_out_of_range_index(
    camera_index: int,
) -> None:
    """Invalid physical camera IDs fail instead of indexing another weight."""
    with pytest.raises(ValueError, match="camera index"):
        indexed_camera_loss_weight(
            camera_index=camera_index,
            configured_weights=[1.0, 1.0, 2.0, 2.0],
        )
