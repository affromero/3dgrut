"""Behavior tests for the optimizer recovered from visibility-adaptive."""

from typing import cast

import pytest
import torch

from threedgrut.optimizers.visibility_selective_adam import (
    DENSITY_REGULARIZER,
    ENCODER_REGULARIZER,
    FP16GlobalAdam,
    POSITION_UPDATE_DECAY,
    VISIBILITY_SELECTIVE_DEFAULT_EPSILON,
    VisibilitySelectiveAdam,
    scale_regularizer,
)
from threedgrut.trainer import _step_gaussian_optimizer


def _parameters(point_count: int = 3) -> dict[str, torch.nn.Parameter]:
    return {
        "positions": torch.nn.Parameter(torch.zeros((point_count, 3))),
        "density": torch.nn.Parameter(torch.zeros((point_count, 1))),
        "features_albedo": torch.nn.Parameter(torch.zeros((point_count, 3))),
        "features_specular": torch.nn.Parameter(
            torch.zeros((point_count, 8))
        ),
        "rotation": torch.nn.Parameter(torch.zeros((point_count, 4))),
        "scale": torch.nn.Parameter(torch.zeros((point_count, 3))),
    }


def _optimizer(
    parameters: dict[str, torch.nn.Parameter],
    *,
    eps: float = VISIBILITY_SELECTIVE_DEFAULT_EPSILON,
    radiance_eps: float | None = None,
) -> VisibilitySelectiveAdam:
    groups = [
        {"name": name, "params": [parameter], "lr": 1.0}
        for name, parameter in parameters.items()
    ]
    if radiance_eps is not None:
        for group in groups:
            if group["name"] in {"features_albedo", "features_specular"}:
                group["eps"] = radiance_eps
    return VisibilitySelectiveAdam(groups, eps=eps)


def _adam_bias_scale(step: int) -> float:
    """Return the native per-Gaussian Adam bias scale for one age."""
    step_tensor = torch.tensor(float(step))
    return float(
        torch.sqrt(1.0 - torch.pow(0.99, step_tensor))
        / (1.0 - torch.pow(0.9, step_tensor))
    )


def test_visibility_selective_adam_updates_only_visible_rows() -> None:
    """Visible rows use age decay and zero-gradient regularizers."""
    parameters = _parameters()
    optimizer = _optimizer(parameters)
    parameters["positions"].grad = torch.ones_like(parameters["positions"])
    parameters["features_albedo"].grad = torch.ones_like(
        parameters["features_albedo"]
    )
    parameters["features_specular"].grad = torch.ones_like(
        parameters["features_specular"]
    )

    optimizer.step(torch.tensor((True, False, True)))

    expected_positions = torch.tensor(
        (
            (-0.9998, -0.9998, -0.9998),
            (0.0, 0.0, 0.0),
            (-0.9998, -0.9998, -0.9998),
        )
    )
    expected_color = torch.tensor(-0.9501953125)
    assert torch.allclose(parameters["positions"], expected_positions)
    assert torch.all(parameters["features_albedo"][[0, 2]] == expected_color)
    assert torch.equal(
        parameters["features_albedo"][1],
        torch.zeros(3),
    )
    assert torch.all(
        parameters["features_specular"][[0, 2]] == expected_color
    )
    assert torch.equal(
        parameters["features_specular"][1],
        torch.zeros(8),
    )
    assert torch.allclose(
        parameters["density"].detach().squeeze(),
        torch.tensor((-0.001, 0.0, -0.001)),
    )
    assert torch.equal(parameters["scale"], torch.zeros((3, 3)))

    position_state = optimizer.state[parameters["positions"]]
    color_state = optimizer.state[parameters["features_albedo"]]
    latent_state = optimizer.state[parameters["features_specular"]]
    assert torch.equal(
        position_state["gaussian_steps"],
        torch.tensor((1, 0, 1), dtype=torch.int32),
    )
    assert color_state["exp_avg"].dtype is torch.float16
    assert color_state["exp_avg_sq"].dtype is torch.float16
    assert latent_state["exp_avg"].dtype is torch.float16
    assert latent_state["exp_avg_sq"].dtype is torch.float16

def test_trainer_passes_renderer_visibility_to_selective_optimizer() -> None:
    """LiDAR and RGB updates share the visibility-selective step."""
    parameters = _parameters()
    optimizer = _optimizer(parameters)
    parameters["positions"].grad = torch.ones_like(parameters["positions"])

    _step_gaussian_optimizer(
        optimizer=optimizer,
        density=parameters["density"],
        outputs={
            "mog_visibility": torch.tensor(
                ((True,), (False,), (True,))
            )
        },
    )

    assert torch.equal(
        optimizer.state[parameters["positions"]]["gaussian_steps"],
        torch.tensor((1, 0, 1), dtype=torch.int32),
    )


def test_visibility_selective_adam_can_preserve_age_on_auxiliary_update() -> None:
    """An auxiliary update changes moments without advancing image age."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)
    parameters["positions"].grad = torch.ones_like(parameters["positions"])

    optimizer.step(torch.tensor((True,)), advance_age=False)
    optimizer.step(torch.tensor((True,)))

    second_update = (
        0.19
        / (0.0199**0.5)
        * _adam_bias_scale(1)
        * POSITION_UPDATE_DECAY
    )
    torch.testing.assert_close(
        parameters["positions"].detach(),
        torch.full((1, 3), -POSITION_UPDATE_DECAY - second_update),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert torch.equal(
        optimizer.state[parameters["positions"]]["gaussian_steps"],
        torch.tensor((1,), dtype=torch.int32),
    )


def test_visibility_selective_adam_uses_incremented_age_after_current_update() -> (
    None
):
    """A larger stored-age increment does not alter the current update."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)
    parameters["positions"].grad = torch.ones_like(parameters["positions"])

    optimizer.step(torch.tensor((True,)), age_increment=5)

    state = optimizer.state[parameters["positions"]]
    torch.testing.assert_close(
        parameters["positions"].detach(),
        torch.full((1, 3), -POSITION_UPDATE_DECAY),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert torch.equal(
        state["gaussian_steps"],
        torch.tensor((5,), dtype=torch.int32),
    )

    optimizer.step(torch.tensor((True,)), age_increment=5)

    second_update = (
        0.19
        / (0.0199**0.5)
        * _adam_bias_scale(6)
        * POSITION_UPDATE_DECAY**6
    )
    torch.testing.assert_close(
        parameters["positions"].detach(),
        torch.full((1, 3), -POSITION_UPDATE_DECAY - second_update),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert torch.equal(
        state["gaussian_steps"],
        torch.tensor((10,), dtype=torch.int32),
    )


def test_visibility_selective_adam_increment_saturates_before_storage() -> None:
    """A multi-step age increment cannot overflow native counter storage."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)
    optimizer.step(torch.tensor((True,)))
    state = optimizer.state[parameters["positions"]]
    state["gaussian_steps"].fill_(65_533)

    optimizer.step(torch.tensor((True,)), age_increment=5)
    optimizer.step(torch.tensor((True,)), age_increment=5)

    assert torch.equal(
        state["gaussian_steps"],
        torch.tensor((65_535,), dtype=torch.int32),
    )


@pytest.mark.parametrize("age_increment", (0, -1, True, 1.5))
def test_visibility_selective_adam_rejects_invalid_age_increment(
    age_increment: object,
) -> None:
    """The public age-increment API accepts only positive integer values."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)

    with pytest.raises((TypeError, ValueError), match="age increment"):
        optimizer.step(
            torch.tensor((True,)),
            age_increment=cast(int, age_increment),
        )


def test_visibility_selective_adam_uses_bias_corrected_per_gaussian_age() -> None:
    """Each visible Gaussian applies bias-corrected Adam at its own age."""
    parameters = _parameters(point_count=2)
    optimizer = _optimizer(parameters)
    parameters["positions"].grad = torch.ones_like(parameters["positions"])
    parameters["features_albedo"].grad = torch.ones_like(
        parameters["features_albedo"]
    )

    optimizer.step(torch.tensor((True, False)))
    optimizer.step(torch.tensor((True, True)))

    second_visible_update = (
        0.19
        / (0.0199**0.5)
        * _adam_bias_scale(2)
        * POSITION_UPDATE_DECAY**2
    )
    assert torch.allclose(
        parameters["positions"].detach(),
        torch.tensor(
            (
                (
                    -POSITION_UPDATE_DECAY - second_visible_update,
                    -POSITION_UPDATE_DECAY - second_visible_update,
                    -POSITION_UPDATE_DECAY - second_visible_update,
                ),
                (
                    -POSITION_UPDATE_DECAY,
                    -POSITION_UPDATE_DECAY,
                    -POSITION_UPDATE_DECAY,
                ),
            )
        ),
        atol=1.0e-6,
    )
    assert torch.equal(
        optimizer.state[parameters["positions"]]["gaussian_steps"],
        torch.tensor((2, 1), dtype=torch.int32),
    )


def test_visibility_selective_adam_preserves_configured_radiance_epsilon() -> None:
    """Radiance groups can override the generic optimizer epsilon."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters, radiance_eps=2.0e-5)
    color_group = next(
        group
        for group in optimizer.param_groups
        if group["name"] == "features_albedo"
    )
    specular_group = next(
        group
        for group in optimizer.param_groups
        if group["name"] == "features_specular"
    )
    position_group = next(
        group
        for group in optimizer.param_groups
        if group["name"] == "positions"
    )
    assert color_group["eps"] == 2.0e-5
    assert specular_group["eps"] == 2.0e-5
    assert position_group["eps"] == VISIBILITY_SELECTIVE_DEFAULT_EPSILON


def test_scale_regularizer_orders_axes_and_handles_ties() -> None:
    """Scale ordering, threshold, permutations, and ties are stable."""
    scales = torch.tensor(
        (
            (-1.0, 0.0, 7.0),
            (-1.0, 0.0, 8.0),
            (9.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (2.0, 2.0, 2.0),
        )
    )

    regularizer = scale_regularizer(scales)

    expected = torch.tensor(
        (
            (0.01, -0.07, 0.07),
            (-0.01, -0.08, 0.08),
            (0.09, -0.09, -0.01),
            (0.01, -0.01, 0.01),
            (0.01, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    assert torch.allclose(regularizer, expected)


def test_visibility_selective_counter_saturates_like_native_storage() -> None:
    """The update at age 65536 preserves the saturated native counter."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)
    optimizer.step(torch.tensor((True,)))
    state = optimizer.state[parameters["positions"]]
    state["gaussian_steps"].fill_(65_535)
    before = parameters["density"].detach().clone()

    optimizer.step(torch.tensor((True,)))

    assert torch.equal(
        state["gaussian_steps"],
        torch.tensor((65_535,), dtype=torch.int32),
    )
    torch.testing.assert_close(
        parameters["density"].detach(),
        before - DENSITY_REGULARIZER,
    )


def test_visibility_selective_state_round_trip_restores_storage_dtypes() -> None:
    """PyTorch checkpoint loading must not cast FP16 state to FP32."""
    source_parameters = _parameters()
    source_optimizer = _optimizer(source_parameters)
    source_optimizer.step(torch.tensor((True, False, True)))

    target_parameters = _parameters()
    target_optimizer = _optimizer(target_parameters)
    target_optimizer.load_state_dict(source_optimizer.state_dict())

    position_state = target_optimizer.state[target_parameters["positions"]]
    color_state = target_optimizer.state[
        target_parameters["features_albedo"]
    ]
    latent_state = target_optimizer.state[
        target_parameters["features_specular"]
    ]
    assert position_state["gaussian_steps"].dtype is torch.int32
    assert color_state["exp_avg"].dtype is torch.float16
    assert color_state["exp_avg_sq"].dtype is torch.float16
    assert latent_state["exp_avg"].dtype is torch.float16
    assert latent_state["exp_avg_sq"].dtype is torch.float16


def test_global_fp16_encoder_adam_matches_recovered_update() -> None:
    """Encoder Adam quantizes state and applies its recovered L2 term."""
    encoder = torch.nn.Parameter(torch.full((2, 3), 0.5))
    optimizer = FP16GlobalAdam(
        [{"name": "color_encoder", "params": [encoder], "lr": 0.001}]
    )
    encoder.grad = torch.ones_like(encoder)

    optimizer.step()

    expected = torch.tensor(
        0.5 - 0.001 * (1.0 + ENCODER_REGULARIZER * 0.5)
    ).to(dtype=torch.float16)
    torch.testing.assert_close(
        encoder,
        expected.to(dtype=encoder.dtype).expand_as(encoder),
    )
    state = optimizer.state[encoder]
    assert state["step"] == 1
    assert state["exp_avg"].dtype is torch.float16
    assert state["exp_avg_sq"].dtype is torch.float16


def test_global_fp16_encoder_state_round_trip_preserves_fp16() -> None:
    """Global FP16 Adam resumes with its scalar age and moments."""
    source = torch.nn.Parameter(torch.ones((2, 3)))
    source_optimizer = FP16GlobalAdam(
        [{"name": "color_encoder", "params": [source]}]
    )
    source.grad = torch.ones_like(source)
    source_optimizer.step()

    target = torch.nn.Parameter(torch.ones((2, 3)))
    target_optimizer = FP16GlobalAdam(
        [{"name": "color_encoder", "params": [target]}]
    )
    target_optimizer.load_state_dict(source_optimizer.state_dict())

    state = target_optimizer.state[target]
    assert state["step"] == 1
    assert state["exp_avg"].dtype is torch.float16
    assert state["exp_avg_sq"].dtype is torch.float16
