import pytest
import torch

from threedgrut.model.gaussian_track_acquisition import (
    GaussianTrackAcquisition,
)


def _treatment(
    *,
    min_sequence_idx: int = 0,
    max_sequence_idx: int = 4,
) -> GaussianTrackAcquisition:
    return GaussianTrackAcquisition(
        gaussian_track_ids=torch.tensor((0, 1, 1, 2, 3)),
        num_cameras=3,
        num_knots=5,
        min_sequence_idx=min_sequence_idx,
        max_sequence_idx=max_sequence_idx,
        max_rgb_delta=0.15,
        magnitude_regularization=0.01,
        curvature_regularization=0.10,
        device="cpu",
        dtype=torch.float32,
    )


def test_zero_knots_exactly_preserve_parent_radiance() -> None:
    treatment = _treatment()

    delta = treatment.rgb_delta(camera_idx=1, sequence_idx=2)

    torch.testing.assert_close(delta, torch.zeros_like(delta))


def test_track_zero_abstains_even_when_knots_are_nonzero() -> None:
    treatment = _treatment()
    with torch.no_grad():
        treatment.rgb_knots[0].fill_(10.0)

    delta = treatment.rgb_delta(camera_idx=0, sequence_idx=0)

    torch.testing.assert_close(delta[0], torch.zeros(3))
    assert bool(torch.all(delta[1:] > 0.0))


def test_gaussians_on_the_same_track_share_the_same_residual() -> None:
    treatment = _treatment()
    with torch.no_grad():
        treatment.rgb_knots[0][1] = torch.tensor((0.1, 0.2, 0.3))

    delta = treatment.rgb_delta(camera_idx=0, sequence_idx=0)

    torch.testing.assert_close(delta[1], delta[2])


def test_piecewise_linear_interpolation_uses_neighboring_knots() -> None:
    treatment = _treatment(max_sequence_idx=8)
    with torch.no_grad():
        treatment.rgb_knots[1][1, 0] = torch.atanh(torch.tensor(0.2))
        treatment.rgb_knots[2][1, 0] = torch.atanh(torch.tensor(0.6))

    delta = treatment.rgb_delta(camera_idx=0, sequence_idx=3)

    torch.testing.assert_close(
        delta[1, 0],
        torch.tensor(0.4 * treatment.max_rgb_delta),
    )


@pytest.mark.parametrize(
    ("camera_idx", "sequence_idx"),
    ((-1, 0), (3, 0), (0, -1), (0, 5)),
)
def test_conditioning_outside_registered_domain_fails_closed(
    camera_idx: int,
    sequence_idx: int,
) -> None:
    treatment = _treatment()

    with pytest.raises(ValueError, match="outside the registered interval"):
        treatment.rgb_delta(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )


def test_rgb_delta_respects_registered_channel_bound() -> None:
    treatment = _treatment()
    with torch.no_grad():
        for knot in treatment.rgb_knots:
            knot.fill_(100.0)

    delta = treatment.rgb_delta(camera_idx=2, sequence_idx=4)

    assert (
        float(delta.detach().abs().max()) <= treatment.max_rgb_delta + 1.0e-6
    )


def test_regularization_touches_only_active_camera_neighbors() -> None:
    treatment = _treatment()
    with torch.no_grad():
        treatment.rgb_knots[0][1, 0] = 0.5

    loss = treatment.get_local_regularization_loss(
        camera_idx=0,
        sequence_idx=0,
    )
    loss.backward()

    active = {
        index
        for index, parameter in enumerate(treatment.rgb_knots)
        if parameter.grad is not None
    }
    assert active == {0, 1, 2}


def test_state_round_trip_preserves_tracks_and_prediction() -> None:
    treatment = _treatment()
    with torch.no_grad():
        for knot in treatment.rgb_knots:
            knot.normal_()
    restored = _treatment()
    restored.load_state_dict(treatment.state_dict(), strict=True)

    expected = treatment.rgb_delta(camera_idx=2, sequence_idx=3)
    actual = restored.rgb_delta(camera_idx=2, sequence_idx=3)

    torch.testing.assert_close(
        restored.gaussian_track_ids,
        treatment.gaussian_track_ids,
    )
    torch.testing.assert_close(actual, expected)
