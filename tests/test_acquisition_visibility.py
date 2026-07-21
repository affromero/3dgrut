import pytest
import torch

from threedgrut.model.acquisition_visibility import AcquisitionVisibility


def _visibility(
    *,
    min_sequence_idx: int = 7,
    max_sequence_idx: int = 278,
) -> AcquisitionVisibility:
    return AcquisitionVisibility(
        num_gaussians=5,
        num_cameras=3,
        num_knots=5,
        min_sequence_idx=min_sequence_idx,
        max_sequence_idx=max_sequence_idx,
        max_logit_delta=4.0,
        response_sparsity_regularization=0.001,
        magnitude_regularization=0.01,
        curvature_regularization=0.05,
        device="cpu",
        dtype=torch.float32,
    )


def test_zero_spatial_response_preserves_parent_opacity_logits() -> None:
    visibility = _visibility()
    parent_logits = torch.randn(5, 1)

    delta = visibility.logit_delta(
        camera_idx=1,
        sequence_idx=140,
    )
    treatment = torch.sigmoid(parent_logits + delta)
    control = torch.sigmoid(parent_logits)

    torch.testing.assert_close(delta, torch.zeros_like(delta))
    torch.testing.assert_close(treatment, control, rtol=0.0, atol=0.0)


def test_temporal_initialization_contains_four_independent_modes() -> None:
    visibility = _visibility()

    temporal = torch.tanh(visibility.temporal_raw[0])

    torch.testing.assert_close(
        temporal[0],
        torch.full_like(temporal[0], 0.25),
    )
    torch.testing.assert_close(
        temporal[1],
        torch.linspace(-0.25, 0.25, 5),
    )
    torch.testing.assert_close(temporal[2, (0, 2, 4)], torch.zeros(3))
    torch.testing.assert_close(
        temporal[3, (0, 2, 4)],
        torch.tensor((0.25, -0.25, 0.25)),
    )


def test_piecewise_linear_interpolation_uses_registered_sequence_domain() -> (
    None
):
    visibility = _visibility()
    with torch.no_grad():
        visibility.temporal_raw[2, 0] = torch.atanh(
            torch.tensor((0.0, 0.2, 0.4, 0.6, 0.8))
        )

    first = visibility.sample_coefficients(
        camera_idx=2,
        sequence_idx=7,
    )
    middle = visibility.sample_coefficients(
        camera_idx=2,
        sequence_idx=142,
    )
    last = visibility.sample_coefficients(
        camera_idx=2,
        sequence_idx=278,
    )

    torch.testing.assert_close(first[0], torch.tensor(0.0))
    assert 0.39 < float(middle[0].detach()) < 0.41
    torch.testing.assert_close(last[0], torch.tensor(0.8))


@pytest.mark.parametrize(
    ("camera_idx", "sequence_idx"),
    ((-1, 7), (3, 7), (0, 6), (0, 279)),
)
def test_conditioning_outside_registered_domain_fails_closed(
    camera_idx: int,
    sequence_idx: int,
) -> None:
    visibility = _visibility()

    with pytest.raises(ValueError, match="outside the registered interval"):
        visibility.sample_coefficients(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )


def test_opacity_logit_delta_respects_registered_bound() -> None:
    visibility = _visibility()
    with torch.no_grad():
        visibility.response_raw.fill_(100.0)
        visibility.temporal_raw.fill_(100.0)

    delta = visibility.logit_delta(
        camera_idx=2,
        sequence_idx=278,
    )

    assert float(delta.detach().abs().max()) <= 4.0


def test_regularization_penalizes_spatial_response_and_temporal_curvature() -> (
    None
):
    visibility = _visibility()
    with torch.no_grad():
        visibility.response_raw[0, 0] = 1.0
        visibility.temporal_raw[0, 0, 2] = 1.0

    loss = visibility.get_regularization_loss()
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert visibility.response_raw.grad is not None
    assert visibility.temporal_raw.grad is not None
    assert bool(torch.isfinite(visibility.response_raw.grad).all())
    assert bool(torch.isfinite(visibility.temporal_raw.grad).all())


def test_state_round_trip_preserves_prediction() -> None:
    visibility = _visibility()
    with torch.no_grad():
        visibility.response_raw.normal_()
        visibility.temporal_raw.normal_()
    restored = _visibility()
    restored.load_state_dict(visibility.state_dict(), strict=True)

    expected = visibility.logit_delta(
        camera_idx=2,
        sequence_idx=173,
    )
    actual = restored.logit_delta(
        camera_idx=2,
        sequence_idx=173,
    )

    torch.testing.assert_close(actual, expected)
