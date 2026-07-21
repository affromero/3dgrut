import pytest
import torch

from threedgrut.model.acquisition_appearance import SH_DC_NORMALIZATION
from threedgrut.model.surface_acquisition_spline import (
    SurfaceAcquisitionSpline,
)


def _spline(
    *,
    min_sequence_idx: int = 0,
    max_sequence_idx: int = 4,
) -> SurfaceAcquisitionSpline:
    return SurfaceAcquisitionSpline(
        num_gaussians=5,
        num_cameras=3,
        num_knots=5,
        min_sequence_idx=min_sequence_idx,
        max_sequence_idx=max_sequence_idx,
        max_rgb_delta=0.25,
        magnitude_regularization=0.01,
        curvature_regularization=0.05,
        device="cpu",
        dtype=torch.float32,
    )


def test_zero_coefficient_knots_preserve_parent_radiance() -> None:
    spline = _spline()

    delta = spline.rgb_delta(camera_idx=1, sequence_idx=2)

    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_each_camera_and_knot_owns_independent_surface_coefficients() -> None:
    spline = _spline()

    assert len(spline.coefficient_knots) == 15
    assert all(
        parameter.shape == (5, 2) for parameter in spline.coefficient_knots
    )
    assert (
        spline.coefficient_knots[0].data_ptr()
        != spline.coefficient_knots[5].data_ptr()
    )


def test_piecewise_linear_interpolation_is_surface_local() -> None:
    spline = _spline(max_sequence_idx=8)
    with torch.no_grad():
        spline.coefficient_knots[1][:, 0] = torch.atanh(
            torch.linspace(0.1, 0.5, 5)
        )
        spline.coefficient_knots[2][:, 0] = torch.atanh(
            torch.linspace(0.3, 0.7, 5)
        )

    coefficients = spline.sample_coefficients(
        camera_idx=0,
        sequence_idx=3,
    )

    torch.testing.assert_close(
        coefficients[:, 0],
        torch.linspace(0.2, 0.6, 5),
    )


@pytest.mark.parametrize(
    ("camera_idx", "sequence_idx"),
    [(-1, 0), (3, 0), (0, -1), (0, 5)],
)
def test_conditioning_outside_registered_domain_fails_closed(
    camera_idx: int,
    sequence_idx: int,
) -> None:
    spline = _spline()

    with pytest.raises(ValueError, match="outside the registered interval"):
        spline.sample_coefficients(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )


def test_rgb_delta_respects_registered_channel_bound() -> None:
    spline = _spline()
    with torch.no_grad():
        spline.direction_raw.fill_(100.0)
        for parameter in spline.coefficient_knots:
            parameter.fill_(100.0)

    delta = spline.rgb_delta(camera_idx=2, sequence_idx=4)

    assert float(delta.detach().abs().max()) <= 0.25


def test_local_regularization_touches_only_active_camera_neighbors() -> None:
    spline = _spline()
    with torch.no_grad():
        spline.coefficient_knots[0][0, 0] = 0.5

    loss = spline.get_local_regularization_loss(
        camera_idx=0,
        sequence_idx=0,
    )
    loss.backward()

    active_gradient_indices = {
        index
        for index, parameter in enumerate(spline.coefficient_knots)
        if parameter.grad is not None
    }
    assert active_gradient_indices == {0, 1, 2}
    assert spline.direction_raw.grad is None


def test_materialization_changes_only_dc_radiance() -> None:
    spline = _spline()
    with torch.no_grad():
        spline.coefficient_knots[0][:, 0] = 0.5
    positions = torch.randn(5, 3)
    rotation = torch.randn(5, 4)
    scale = torch.randn(5, 3)
    density = torch.randn(5, 1)
    albedo = torch.randn(5, 3)
    specular = torch.randn(5, 45)
    background = torch.nn.Identity()

    view = spline.materialize(
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

    expected_albedo = (
        albedo
        + spline.rgb_delta(
            camera_idx=0,
            sequence_idx=0,
        )
        / SH_DC_NORMALIZATION
    )
    torch.testing.assert_close(view.positions, positions)
    torch.testing.assert_close(view.rotation, rotation)
    torch.testing.assert_close(view.scale, scale)
    torch.testing.assert_close(view.density, density)
    torch.testing.assert_close(view.get_features()[:, :3], expected_albedo)
    torch.testing.assert_close(view.get_features()[:, 3:], specular)


def test_state_round_trip_preserves_surface_local_prediction() -> None:
    spline = _spline()
    with torch.no_grad():
        spline.direction_raw.normal_()
        for parameter in spline.coefficient_knots:
            parameter.normal_()
    restored = _spline()
    restored.load_state_dict(spline.state_dict(), strict=True)

    expected = spline.rgb_delta(camera_idx=2, sequence_idx=3)
    actual = restored.rgb_delta(camera_idx=2, sequence_idx=3)

    torch.testing.assert_close(actual, expected)
