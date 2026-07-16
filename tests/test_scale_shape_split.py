"""Behavior tests for the isolated visibility-and-shape-derived 3DGUT topology arm."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.strategy.gs import GSStrategy
from threedgrut.strategy.scale_shape_split import (
    DETERMINISTIC_CHILD_OFFSET,
    DETERMINISTIC_CHILD_SCALE,
    ScaleShapeThresholds,
    deterministic_split_children,
    scale_shape_split_mask,
    transmittance_preserving_child_opacity,
)


class _StrategyModel:
    """Small CPU model exposing the Gaussian contract used by GSStrategy."""

    def __init__(
        self,
        scales: tuple[tuple[float, float, float], ...],
        *,
        opacity: float = 0.36,
    ) -> None:
        self.device = torch.device("cpu")
        point_count = len(scales)
        positions = torch.arange(point_count, dtype=torch.float32) * 10.0
        self.positions = torch.nn.Parameter(
            torch.stack(
                (
                    positions,
                    torch.zeros(point_count),
                    torch.zeros(point_count),
                ),
                dim=1,
            )
        )
        opacity_tensor = torch.full((point_count, 1), opacity)
        self.density = torch.nn.Parameter(torch.logit(opacity_tensor))
        self.features_albedo = torch.nn.Parameter(torch.ones((point_count, 3)))
        self.features_specular = torch.nn.Parameter(torch.ones((point_count, 3)))
        self.rotation = torch.nn.Parameter(torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(point_count, 1))
        self.scale = torch.nn.Parameter(torch.log(torch.tensor(scales)))
        parameter_names = (
            "positions",
            "density",
            "features_albedo",
            "features_specular",
            "rotation",
            "scale",
        )
        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": [getattr(self, name)],
                    "name": name,
                    "lr": 0.001,
                }
                for name in parameter_names
            ]
        )

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_density(self) -> torch.Tensor:
        return torch.sigmoid(self.density)

    def get_scale(self) -> torch.Tensor:
        return torch.exp(self.scale)

    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.rotation, dim=1)

    @staticmethod
    def density_activation_inv(value: torch.Tensor) -> torch.Tensor:
        return torch.logit(value)

    @staticmethod
    def scale_activation(value: torch.Tensor) -> torch.Tensor:
        return torch.exp(value)

    @staticmethod
    def scale_activation_inv(value: torch.Tensor) -> torch.Tensor:
        return torch.log(value)


def _strategy(
    *,
    enabled: bool = True,
    overrides: dict[str, object] | None = None,
) -> tuple[GSStrategy, _StrategyModel]:
    values = overrides or {}
    model = _StrategyModel(
        (
            (0.02, 0.002, 0.001),
            (0.02, 0.002, 0.001),
        )
    )
    config = OmegaConf.create(
        {
            "render": {"method": values.get("render_method", "3dgut")},
            "model": {
                "density_activation": values.get("density_activation", "sigmoid"),
                "scale_activation": values.get("scale_activation", "exp"),
            },
            "strategy": {
                "print_stats": False,
                "densify": {
                    "frequency": 300,
                    "start_iteration": 500,
                    "end_iteration": 15_000,
                    "clone_grad_threshold": 0.2,
                    "split_grad_threshold": 0.2,
                    "relative_size_threshold": 0.005,
                    "split": {"n_gaussians": values.get("split_children", 2)},
                    "theta_aware": {"enabled": False},
                    "feature_grad": {"enabled": False},
                    "projected_extent_split": {"enabled": False},
                    "covariance_gradient_split": {"enabled": False},
                    "scale_shape_split": {
                        "enabled": enabled,
                        "anisotropy_threshold": values.get("anisotropy_threshold", 8.0),
                        "min_largest_scale": values.get("min_largest_scale", 0.01),
                        "min_observations": values.get("min_observations", 32),
                    },
                },
                "prune": {"density_threshold": 0.005},
                "reset_density": {"new_max_density": 0.01},
            },
        }
    )
    strategy = GSStrategy(config, model)
    strategy.init_densification_buffer()
    return strategy, model


def test_scale_shape_mask_uses_strict_shape_scale_and_status_gates() -> None:
    """Only existing split candidates clearing every frozen gate reroute."""
    mask = scale_shape_split_mask(
        split_candidates=torch.tensor((True, True, True, True, False)),
        physical_scales=torch.tensor(
            (
                (0.02, 0.0025, 0.0025),
                (0.0201, 0.0025, 0.0025),
                (0.0201, 0.0025, 0.0025),
                (0.01, 0.001, 0.001),
                (1.0, 0.001, 0.001),
            )
        ),
        observation_count=torch.tensor(((32,), (31,), (32,), (32,), (32,))),
        thresholds=ScaleShapeThresholds(
            anisotropy=8.0,
            min_largest_scale=0.01,
            min_observations=32,
        ),
    )

    torch.testing.assert_close(
        mask,
        torch.tensor((False, False, True, False, False)),
    )


def test_deterministic_children_follow_largest_rotated_axis() -> None:
    """The recovered offset and 0.8 shrink are deterministic and axis-aware."""
    positions = torch.tensor(((1.0, 2.0, 3.0),))
    scales = torch.tensor(((0.02, 0.002, 0.001),))
    rotations = torch.tensor(((1.0, 0.0, 0.0, 0.0),))

    child_positions, child_scales = deterministic_split_children(
        positions=positions,
        physical_scales=scales,
        rotations=rotations,
    )

    offset = 0.02 * DETERMINISTIC_CHILD_OFFSET
    torch.testing.assert_close(
        child_positions,
        torch.tensor(((1.0 + offset, 2.0, 3.0), (1.0 - offset, 2.0, 3.0))),
    )
    torch.testing.assert_close(
        child_scales,
        scales.repeat(2, 1) * DETERMINISTIC_CHILD_SCALE,
    )


def test_child_opacity_preserves_parent_transmittance_exactly() -> None:
    """Two equal child alphas reproduce each parent's aggregate alpha."""
    parent = torch.tensor(((0.001,), (0.36,), (0.9,)), dtype=torch.float64)

    children = transmittance_preserving_child_opacity(parent)
    first, second = children.chunk(2, dim=0)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        (1.0 - first) * (1.0 - second),
        1.0 - parent,
        rtol=1e-14,
        atol=1e-14,
    )


def test_visibility_status_counts_only_visible_rows_and_saturates() -> None:
    """The same-window status reaches but never exceeds the frozen gate."""
    strategy, _ = _strategy()
    strategy.densify_scale_shape_observation_count[:, 0] = torch.tensor((31, 2))

    strategy.update_scale_shape_observation_count({"mog_visibility": torch.tensor(((True,), (False,)))})
    strategy.update_scale_shape_observation_count({"mog_visibility": torch.tensor(((True,), (True,)))})

    torch.testing.assert_close(
        strategy.densify_scale_shape_observation_count[:, 0],
        torch.tensor((32, 3), dtype=torch.int32),
    )


def test_matched_split_reroutes_only_status_qualified_candidate() -> None:
    """Candidate count stays fixed while one split receives deterministic children."""
    strategy, model = _strategy()
    strategy.densify_grad_norm_accum[:, 0] = 0.5
    strategy.densify_grad_norm_denom[:, 0] = 1
    strategy.densify_scale_shape_observation_count[:, 0] = torch.tensor((32, 31))
    torch.manual_seed(19)

    strategy.densify_gaussians(scene_extent=1.0)

    assert model.num_gaussians == 4
    offset = 0.02 * DETERMINISTIC_CHILD_OFFSET
    torch.testing.assert_close(
        model.positions[[0, 2]],
        torch.tensor(((offset, 0.0, 0.0), (-offset, 0.0, 0.0))),
    )
    expected_child_scale = torch.tensor((0.02, 0.002, 0.001)) * 0.8
    torch.testing.assert_close(
        model.get_scale()[[0, 2]],
        expected_child_scale.repeat(2, 1),
    )
    expected_standard_scale = torch.tensor((0.02, 0.002, 0.001)) / 1.6
    torch.testing.assert_close(
        model.get_scale()[[1, 3]],
        expected_standard_scale.repeat(2, 1),
    )
    child_alpha = model.get_density()[[0, 2]]
    torch.testing.assert_close(
        (1.0 - child_alpha[0]) * (1.0 - child_alpha[1]),
        torch.tensor((1.0 - 0.36,)),
    )
    torch.testing.assert_close(
        model.get_density()[[1, 3]],
        torch.full((2, 1), 0.36),
    )
    torch.testing.assert_close(
        strategy.densify_scale_shape_observation_count,
        torch.zeros((4, 1), dtype=torch.int32),
    )


def test_disabled_arm_needs_no_visibility_or_checkpoint_state() -> None:
    """Default-off runs preserve renderer and checkpoint contracts."""
    strategy, _ = _strategy(
        enabled=False,
        overrides={"render_method": "3dgrt"},
    )

    assert "densify_scale_shape_observation_count" not in (strategy.get_strategy_parameters())
    assert not strategy.post_backward(
        step=15_001,
        scene_extent=1.0,
        train_dataset=(),
        outputs=None,
    )


def test_enabled_resume_rejects_missing_mid_window_status() -> None:
    """A resume cannot silently erase status that changes split routing."""
    strategy, _ = _strategy()
    checkpoint = strategy.get_strategy_parameters()
    checkpoint.pop("densify_scale_shape_observation_count")
    checkpoint["global_step"] = 100

    restored, _ = _strategy()
    with pytest.raises(ValueError, match="mid-window observation-count"):
        restored.init_densification_buffer(checkpoint)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"render_method": "3dgrt"}, "render.method=3dgut"),
        ({"anisotropy_threshold": 1.0}, "greater than one"),
        ({"min_largest_scale": math.inf}, "finite and positive"),
        ({"min_observations": 0}, "positive integer"),
        ({"split_children": 3}, "exactly two"),
        ({"density_activation": "none"}, "density_activation=sigmoid"),
        ({"scale_activation": "none"}, "scale_activation=exp"),
    ],
)
def test_enabled_arm_rejects_non_preregistered_contracts(
    overrides: dict[str, object],
    message: str,
) -> None:
    """Frozen units and activation assumptions fail before training."""
    with pytest.raises(ValueError, match=message):
        _strategy(overrides=overrides)
