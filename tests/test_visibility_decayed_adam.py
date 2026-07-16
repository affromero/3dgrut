"""Behavior tests for two-stream per-Gaussian optimizer kernels."""

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers.visibility_decayed_adam import (
    COLOR_EPSILON,
    OPACITY_REGULARIZER,
    VisibilityDecayedAdam,
    ordered_log_scale_regularizer,
)
from threedgrut.strategy.gs import GSStrategy


def _parameters(
    point_count: int = 3,
) -> dict[str, torch.nn.Parameter]:
    return {
        "positions": torch.nn.Parameter(torch.zeros((point_count, 3))),
        "density": torch.nn.Parameter(torch.zeros((point_count, 1))),
        "features_albedo": torch.nn.Parameter(torch.zeros((point_count, 3))),
        "features_specular": torch.nn.Parameter(torch.zeros((point_count, 8))),
        "rotation": torch.nn.Parameter(torch.zeros((point_count, 4))),
        "scale": torch.nn.Parameter(torch.zeros((point_count, 3))),
    }


def _optimizer(
    parameters: dict[str, torch.nn.Parameter],
) -> VisibilityDecayedAdam:
    groups = [{"name": name, "params": [parameter], "lr": 1.0} for name, parameter in parameters.items()]
    return VisibilityDecayedAdam(groups)


def test_image_step_updates_only_renderer_visible_rows() -> None:
    """The image kernel must update only renderer-visible rows."""
    parameters = _parameters()
    optimizer = _optimizer(parameters)
    parameters["positions"].grad = torch.ones_like(parameters["positions"])
    parameters["features_albedo"].grad = torch.ones_like(parameters["features_albedo"])

    optimizer.step(torch.tensor((True, False, True)))

    expected_positions = torch.tensor(
        (
            (-0.9998, -0.9998, -0.9998),
            (0.0, 0.0, 0.0),
            (-0.9998, -0.9998, -0.9998),
        )
    )
    torch.testing.assert_close(parameters["positions"], expected_positions)
    assert torch.all(parameters["features_albedo"][[0, 2]] == torch.tensor(-0.94970703125))
    assert torch.equal(
        parameters["features_albedo"][1],
        torch.zeros(3),
    )
    assert optimizer.state[parameters["features_albedo"]]["exp_avg"].dtype is torch.float16
    assert next(group for group in optimizer.param_groups if group["name"] == "features_albedo")["eps"] == COLOR_EPSILON


def test_geometry_step_never_initializes_or_mutates_color_state() -> None:
    """The LiDAR kernel must omit every color parameter and moment."""
    parameters = _parameters(point_count=2)
    optimizer = _optimizer(parameters)
    parameters["positions"].grad = torch.ones_like(parameters["positions"])
    parameters["features_albedo"].grad = torch.full_like(
        parameters["features_albedo"],
        7.0,
    )
    color_before = parameters["features_albedo"].detach().clone()

    optimizer.step_geometry(torch.tensor((True, False)))

    assert torch.equal(parameters["features_albedo"], color_before)
    assert optimizer.state[parameters["features_albedo"]] == {}
    assert optimizer.state[parameters["features_specular"]] == {}
    assert torch.equal(
        optimizer.state[parameters["positions"]]["gaussian_steps"],
        torch.tensor((1, 0), dtype=torch.int32),
    )
    torch.testing.assert_close(
        parameters["density"].detach().squeeze(),
        torch.tensor((-OPACITY_REGULARIZER, 0.0)),
    )


def test_image_color_decay_uses_shared_age_after_geometry_step() -> None:
    """Both optimizer passes must advance the shared per-Gaussian age."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)

    optimizer.step_geometry(torch.tensor((True,)))
    parameters["features_albedo"].grad = torch.ones_like(parameters["features_albedo"])
    optimizer.step(torch.tensor((True,)))

    age_two_bias_correction = torch.sqrt(torch.tensor(1.0 - 0.99**2)) / (1.0 - 0.9**2)
    expected_age_two_decay = (0.95**2 * age_two_bias_correction).to(dtype=torch.float16)
    assert torch.all(parameters["features_albedo"] == -expected_age_two_decay)
    assert torch.equal(
        optimizer.state[parameters["positions"]]["gaussian_steps"],
        torch.tensor((2,), dtype=torch.int32),
    )


def test_geometry_only_checkpoint_restores_lazy_color_state() -> None:
    """A geometry-only checkpoint must keep color moments lazily initialized."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)
    optimizer.step_geometry(torch.tensor((True,)))

    restored_parameters = _parameters(point_count=1)
    restored_optimizer = _optimizer(restored_parameters)
    restored_optimizer.load_state_dict(optimizer.state_dict())

    assert restored_optimizer.state[restored_parameters["features_albedo"]] == {}
    assert restored_optimizer.state[restored_parameters["features_specular"]] == {}
    assert torch.equal(
        restored_optimizer.state[restored_parameters["positions"]]["gaussian_steps"],
        torch.tensor((1,), dtype=torch.int32),
    )

    restored_parameters["features_albedo"].grad = torch.ones_like(restored_parameters["features_albedo"])
    restored_optimizer.step(torch.tensor((True,)))

    color_state = restored_optimizer.state[restored_parameters["features_albedo"]]
    assert color_state["exp_avg"].dtype is torch.float16
    assert torch.equal(
        restored_optimizer.state[restored_parameters["positions"]]["gaussian_steps"],
        torch.tensor((2,), dtype=torch.int32),
    )


def test_ordered_scale_regularizer_matches_recovered_kernel_probes() -> None:
    """Scale ordering and the recovered threshold must match probes."""
    scales = torch.tensor(
        (
            (-1.0, 0.0, 7.0),
            (-1.0, 0.0, 8.0),
            (9.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
            (2.0, 2.0, 2.0),
        )
    )

    regularizer = ordered_log_scale_regularizer(scales)

    expected = torch.tensor(
        (
            (0.01, -0.07, 0.07),
            (-0.01, -0.08, 0.08),
            (0.09, -0.09, -0.01),
            (0.01, -0.01, 0.01),
            (0.0, 0.0, 0.0),
        )
    )
    torch.testing.assert_close(regularizer, expected)


def test_counter_wraps_after_the_uint16_age_update() -> None:
    """The update at age 65536 must precede uint16 counter wrapping."""
    parameters = _parameters(point_count=1)
    optimizer = _optimizer(parameters)
    optimizer.step_geometry(torch.tensor((True,)))
    state = optimizer.state[parameters["positions"]]
    state["gaussian_steps"].fill_(65_535)
    density_before = parameters["density"].detach().clone()

    optimizer.step_geometry(torch.tensor((True,)))

    assert torch.equal(
        state["gaussian_steps"],
        torch.tensor((0,), dtype=torch.int32),
    )
    torch.testing.assert_close(
        parameters["density"].detach(),
        density_before - OPACITY_REGULARIZER,
    )


def test_model_builds_optimizer_with_absolute_position_rate() -> None:
    """Position LR must remain absolute when scene scaling is disabled."""
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    for name, parameter in _parameters(point_count=2).items():
        setattr(model, name, parameter)
    model.scene_extent = 17.0
    model.conf = OmegaConf.create(
        {
            "optimizer": {
                "type": "visibility_decayed_adam",
                "lr": 0.0,
                "betas": [0.9, 0.99],
                "eps": 1.0e-10,
                "resume_lr_scale": 1.0,
                "scale_position_lr_by_scene_extent": False,
                "params": {
                    "positions": {"lr": 0.001},
                    "density": {"lr": 0.05},
                    "features_albedo": {"lr": 0.1},
                    "features_specular": {"lr": 0.1},
                    "rotation": {"lr": 0.001},
                    "scale": {"lr": 0.005},
                },
            },
            "scheduler": {
                "positions": {
                    "type": "skip",
                    "lr_init": 0.001,
                    "lr_final": 1.0e-6,
                    "max_steps": 30_000,
                }
            },
        }
    )

    model.setup_optimizer()

    assert isinstance(model.optimizer, VisibilityDecayedAdam)
    group_lrs = {str(group["name"]): float(group["lr"]) for group in model.optimizer.param_groups}
    assert group_lrs == {
        "positions": 0.001,
        "density": 0.05,
        "features_albedo": 0.1,
        "features_specular": 0.1,
        "rotation": 0.001,
        "scale": 0.005,
    }
    model.scheduler_step(10_000)
    position_group = next(group for group in model.optimizer.param_groups if group["name"] == "positions")
    assert position_group["lr"] == 0.001


def test_gs_clone_preserves_optimizer_state_dtypes() -> None:
    """Generic GS topology changes must retain optimizer state storage."""
    parameters = _parameters(point_count=2)
    optimizer = _optimizer(parameters)
    optimizer.step(torch.tensor((True, True)))
    model = SimpleNamespace(optimizer=optimizer, **parameters)
    strategy = GSStrategy.__new__(GSStrategy)
    strategy.model = model
    strategy.conf = OmegaConf.create({"strategy": {"print_stats": False}})
    strategy._densify_candidate_masks = lambda *_args: (
        torch.tensor((True, False)),
        torch.tensor((True, False)),
        torch.tensor((False, False)),
    )
    strategy.reset_densification_buffers = lambda: None

    strategy.clone_gaussians(
        densify_grad_norm=torch.ones(2),
        scene_extent=1.0,
    )

    position_state = optimizer.state[model.positions]
    color_state = optimizer.state[model.features_albedo]
    assert position_state["gaussian_steps"].shape == (3,)
    assert position_state["gaussian_steps"].dtype is torch.int32
    assert color_state["exp_avg"].shape == (3, 3)
    assert color_state["exp_avg"].dtype is torch.float16
    assert color_state["exp_avg_sq"].dtype is torch.float16
    assert torch.count_nonzero(color_state["exp_avg"][-1]) == 0
