import pytest
import torch

from threedgrut.model.acquisition_appearance import (
    SH_DC_NORMALIZATION,
    AcquisitionAppearance,
)


def _appearance(
    *,
    min_sequence_idx: int = 0,
    max_sequence_idx: int = 10,
) -> AcquisitionAppearance:
    return AcquisitionAppearance(
        num_gaussians=4,
        num_cameras=3,
        num_knots=3,
        min_sequence_idx=min_sequence_idx,
        max_sequence_idx=max_sequence_idx,
        max_rgb_delta=0.24,
        magnitude_regularization=0.1,
        curvature_regularization=0.2,
        device="cpu",
        dtype=torch.float32,
    )


def test_zero_temporal_knots_preserve_incumbent_radiance() -> None:
    appearance = _appearance()

    delta = appearance.rgb_delta(camera_idx=1, sequence_idx=5)

    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_temporal_coefficients_use_piecewise_linear_interpolation() -> None:
    appearance = _appearance()
    with torch.no_grad():
        appearance.temporal_raw[1, 0] = torch.atanh(
            torch.tensor((0.0, 0.5, 1.0 - 1e-6))
        )

    at_start = appearance.sample_coefficients(
        camera_idx=1,
        sequence_idx=0,
    )
    at_middle = appearance.sample_coefficients(
        camera_idx=1,
        sequence_idx=5,
    )

    torch.testing.assert_close(at_start[0], torch.tensor(0.0))
    torch.testing.assert_close(at_middle[0], torch.tensor(0.5))


def test_nonzero_sequence_domain_maps_both_endpoints_to_knots() -> None:
    appearance = _appearance(
        min_sequence_idx=7,
        max_sequence_idx=278,
    )
    with torch.no_grad():
        appearance.temporal_raw[0, 0] = torch.atanh(
            torch.tensor((0.1, 0.2, 0.3))
        )

    first = appearance.sample_coefficients(
        camera_idx=0,
        sequence_idx=7,
    )
    last = appearance.sample_coefficients(
        camera_idx=0,
        sequence_idx=278,
    )

    torch.testing.assert_close(first[0], torch.tensor(0.1))
    torch.testing.assert_close(last[0], torch.tensor(0.3))


@pytest.mark.parametrize(
    ("camera_idx", "sequence_idx"),
    ((-1, 0), (3, 0), (0, -1), (0, 11)),
)
def test_conditioning_outside_registered_domain_fails_closed(
    camera_idx: int,
    sequence_idx: int,
) -> None:
    appearance = _appearance()

    with pytest.raises(ValueError, match="outside the registered interval"):
        appearance.sample_coefficients(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )


def test_rgb_delta_is_bounded_by_registered_channel_limit() -> None:
    appearance = _appearance()
    with torch.no_grad():
        appearance.direction_raw.fill_(100.0)
        appearance.temporal_raw.fill_(100.0)

    delta = appearance.rgb_delta(camera_idx=2, sequence_idx=10)

    assert float(delta.detach().abs().max()) <= appearance.max_rgb_delta


def test_materialization_changes_only_dc_radiance() -> None:
    appearance = _appearance()
    with torch.no_grad():
        appearance.temporal_raw[0, :, 0] = 1.0
    positions = torch.randn(4, 3)
    rotation = torch.randn(4, 4)
    scale = torch.randn(4, 3)
    density = torch.randn(4, 1)
    albedo = torch.randn(4, 3)
    specular = torch.randn(4, 45)
    background = torch.nn.Identity()

    view = appearance.materialize(
        positions=positions,
        rotation=rotation,
        scale=scale,
        density=density,
        features_albedo=albedo,
        features_specular=specular,
        background=background,
        n_active_features=3,
        camera_idx=0,
        sequence_idx=0,
    )

    expected_albedo = albedo + appearance.rgb_delta(
        camera_idx=0,
        sequence_idx=0,
    ) / SH_DC_NORMALIZATION
    torch.testing.assert_close(view.positions, positions)
    torch.testing.assert_close(view.rotation, rotation)
    torch.testing.assert_close(view.scale, scale)
    torch.testing.assert_close(view.density, density)
    torch.testing.assert_close(view.get_features()[:, :3], expected_albedo)
    torch.testing.assert_close(view.get_features()[:, 3:], specular)


def test_regularization_penalizes_temporal_curvature() -> None:
    appearance = _appearance()
    with torch.no_grad():
        appearance.temporal_raw[0, 0, 1] = 1.0

    loss = appearance.get_regularization_loss()
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert appearance.temporal_raw.grad is not None
    assert bool(torch.isfinite(appearance.temporal_raw.grad).all())


def test_state_round_trip_preserves_prediction() -> None:
    appearance = _appearance()
    with torch.no_grad():
        appearance.direction_raw.normal_()
        appearance.temporal_raw.normal_()
    restored = _appearance()
    restored.load_state_dict(appearance.state_dict(), strict=True)

    expected = appearance.rgb_delta(camera_idx=2, sequence_idx=7)
    actual = restored.rgb_delta(camera_idx=2, sequence_idx=7)

    torch.testing.assert_close(actual, expected)
