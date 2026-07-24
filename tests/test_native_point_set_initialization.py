"""Tests for recovered visibility-adaptive point-set initialization rules."""

import numpy as np
import pytest
import torch
from threedgrut.model import geometry
from threedgrut.model.model import (
    MixtureOfGaussians,
    native_environment_parameters,
    native_initial_scale_from_mean_dist2,
    position_learning_rate_scale,
)


def _environment_mask_model(point_count: int) -> MixtureOfGaussians:
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    model.positions = torch.nn.Parameter(torch.zeros((point_count, 3)))
    model.register_buffer(
        "environment_mask",
        torch.empty((0,), dtype=torch.bool),
    )
    return model


def test_environment_mask_is_model_owned_structural_state() -> None:
    """The model validates and retains one environment bit per Gaussian."""
    model = _environment_mask_model(3)
    mask = torch.tensor((False, True, True))

    model._set_environment_mask(mask)

    torch.testing.assert_close(model.environment_mask, mask)


@pytest.mark.parametrize(
    "invalid_mask",
    (
        torch.zeros((3,), dtype=torch.float32),
        torch.zeros((3, 1), dtype=torch.bool),
    ),
)
def test_environment_mask_rejects_invalid_state(
    invalid_mask: torch.Tensor,
) -> None:
    """Checkpoint identity cannot silently change dtype or row shape."""
    model = _environment_mask_model(3)

    with pytest.raises(ValueError, match="environment mask"):
        model._set_environment_mask(invalid_mask)


def test_position_learning_rate_scale_preserves_profile_semantics() -> None:
    """Only profiles opting into scene scaling inherit the scene extent."""
    assert position_learning_rate_scale(
        scale_by_scene_extent=True, scene_extent=104.0
    ) == pytest.approx(104.0)
    assert position_learning_rate_scale(
        scale_by_scene_extent=False, scene_extent=104.0
    ) == pytest.approx(1.0)


def test_native_simple_knn_fails_loudly_when_extension_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing native KNN extension must not select a different algorithm."""
    monkeypatch.setattr(geometry, "_dist_cuda2", None)

    with pytest.raises(RuntimeError, match="post_build.sh"):
        geometry.native_simple_knn_mean_dist2(torch.zeros((4, 3)))


def test_native_simple_knn_rejects_cpu_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA extension must never reinterpret CPU tensor memory."""
    monkeypatch.setattr(geometry, "_dist_cuda2", lambda points: points[:, 0])

    with pytest.raises(ValueError, match="CUDA"):
        geometry.native_simple_knn_mean_dist2(
            torch.zeros((4, 3), dtype=torch.float32)
        )


def test_native_scale_floor_follows_upper_cap_for_duplicate_points() -> None:
    """A zero median still produces the native finite minimum scale."""
    mean_dist2 = torch.tensor((0.0, 0.0, 4.0), dtype=torch.float32)

    initial_scale = native_initial_scale_from_mean_dist2(mean_dist2)

    torch.testing.assert_close(
        initial_scale,
        torch.full((3,), 1.0e-4, dtype=torch.float32),
    )


def test_native_environment_parameters_align_local_z_to_radial_direction() -> (
    None
):
    """Environmental anisotropy must point along each radial direction."""
    positions = torch.tensor(
        ((0.0, 0.0, 2.0), (0.0, 0.0, -4.0)), dtype=torch.float32
    )

    scales, rotations = native_environment_parameters(
        positions=positions,
        center=torch.zeros(3),
        xy_scale=6.0,
        z_scale=0.2,
    )

    torch.testing.assert_close(
        scales,
        torch.tensor(
            (
                (12.0 / np.sqrt(2.0), 12.0 / np.sqrt(2.0), 0.4 / np.sqrt(2.0)),
                (24.0 / np.sqrt(2.0), 24.0 / np.sqrt(2.0), 0.8 / np.sqrt(2.0)),
            ),
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        rotations[0], torch.tensor((1.0, 0.0, 0.0, 0.0))
    )
    torch.testing.assert_close(
        rotations[1], torch.tensor((0.0, 1.0, 0.0, 0.0))
    )
