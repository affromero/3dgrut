"""Behavior tests for cancellation-conditioned clone-to-split routing."""

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.strategy.gs import GSStrategy


class _Model:
    def __init__(self, count: int = 3) -> None:
        self.device = torch.device("cpu")
        self.positions = torch.nn.Parameter(
            torch.stack(
                (
                    torch.arange(1, count + 1, dtype=torch.float32),
                    torch.zeros(count),
                    torch.zeros(count),
                ),
                dim=1,
            )
        )
        self.scale = torch.nn.Parameter(torch.log(torch.full((count, 3), 0.5)))
        self.features_specular = torch.nn.Parameter(torch.ones((count, 3)))

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    @property
    def protected_gaussian_count(self) -> int:
        return 0

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_scale(self) -> torch.Tensor:
        return torch.exp(self.scale)


def _strategy(
    *,
    enabled: bool = True,
    render_method: str = "3dgut",
    cancellation_threshold: float = 0.5,
    extent_px: float = 8.0,
    min_joint_observations: object = 2,
    min_joint_fraction: float = 0.5,
    max_reroute_fraction: float = 0.5,
    projected_extent_enabled: bool = False,
) -> GSStrategy:
    config = OmegaConf.create(
        {
            "render": {"method": render_method},
            "model": {
                "density_activation": "sigmoid",
                "scale_activation": "exp",
            },
            "strategy": {
                "print_stats": False,
                "densify": {
                    "frequency": 300,
                    "start_iteration": 500,
                    "end_iteration": 15_000,
                    "clone_grad_threshold": 0.2,
                    "split_grad_threshold": 0.4,
                    "relative_size_threshold": 1.0,
                    "split": {"n_gaussians": 2},
                    "theta_aware": {"enabled": False},
                    "feature_grad": {"enabled": False},
                    "tile_coverage_weighted_gradient": {"enabled": False},
                    "absolute_ray_gradient_diagnostics": {"enabled": False},
                    "absolute_ray_gradient_densification": {"enabled": False},
                    "cancellation_conditioned_split": {
                        "enabled": enabled,
                        "cancellation_threshold": cancellation_threshold,
                        "extent_px": extent_px,
                        "min_joint_observations": min_joint_observations,
                        "min_joint_fraction": min_joint_fraction,
                        "max_reroute_fraction": max_reroute_fraction,
                    },
                    "projected_extent_split": {
                        "enabled": projected_extent_enabled,
                        "max_px": 8.0,
                    },
                    "covariance_gradient_split": {"enabled": False},
                    "scale_shape_split": {"enabled": False},
                },
                "prune": {"density_threshold": 0.005},
                "reset_density": {"new_max_density": 0.01},
            },
        }
    )
    strategy = GSStrategy(config, _Model())
    strategy.init_densification_buffer()
    return strategy


def _outputs(
    signed_x: tuple[float, float, float],
    absolute_x: tuple[float, float, float],
    extent_x: tuple[float, float, float],
) -> dict[str, object]:
    zeros = torch.zeros(3)
    return {
        "mog_signed_ray_position_grad": torch.stack(
            (torch.tensor(signed_x), zeros, zeros),
            dim=1,
        ),
        "mog_abs_ray_position_grad": torch.stack(
            (torch.tensor(absolute_x), zeros, zeros),
            dim=1,
        ),
        "mog_projected_extent": torch.stack(
            (torch.tensor(extent_x), torch.ones(3)),
            dim=1,
        ),
        "mog_tiles_count": torch.ones(3, dtype=torch.int32),
    }


def _candidate_masks(
    strategy: GSStrategy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return strategy._densify_candidate_masks(
        torch.full((3,), 0.3),
        1.0,
        None,
        None,
        cancellation_joint_observations=(strategy.densify_cancellation_joint_observations),
        cancellation_valid_observations=(strategy.densify_cancellation_valid_observations),
    )


def test_joint_events_reroute_only_ordinary_clone_candidates() -> None:
    strategy = _strategy()
    outputs = _outputs(
        signed_x=(1.0, 1.0, 1.0),
        absolute_x=(3.0, 3.0, 1.0),
        extent_x=(10.0, 7.0, 10.0),
    )

    strategy.update_cancellation_conditioned_split_buffer(outputs)
    strategy.update_cancellation_conditioned_split_buffer(outputs)

    torch.testing.assert_close(
        strategy.densify_cancellation_joint_observations,
        torch.tensor(((2,), (0,), (0,)), dtype=torch.int32),
    )
    torch.testing.assert_close(
        strategy.densify_cancellation_valid_observations,
        torch.full((3, 1), 2, dtype=torch.int32),
    )
    clone, split, rerouted = _candidate_masks(strategy)
    torch.testing.assert_close(clone, torch.tensor((False, True, True)))
    torch.testing.assert_close(split, torch.tensor((True, False, False)))
    torch.testing.assert_close(rerouted, torch.tensor((True, False, False)))


def test_cancellation_and_extent_must_cooccur_in_the_same_observation() -> None:
    strategy = _strategy(min_joint_observations=1)

    strategy.update_cancellation_conditioned_split_buffer(
        _outputs(
            signed_x=(1.0, 1.0, 1.0),
            absolute_x=(3.0, 1.0, 1.0),
            extent_x=(7.0, 7.0, 7.0),
        )
    )
    strategy.update_cancellation_conditioned_split_buffer(
        _outputs(
            signed_x=(1.0, 1.0, 1.0),
            absolute_x=(1.0, 1.0, 1.0),
            extent_x=(10.0, 10.0, 10.0),
        )
    )

    assert not bool(strategy.densify_cancellation_joint_observations.any())
    clone, split, rerouted = _candidate_masks(strategy)
    torch.testing.assert_close(clone, torch.tensor((True, True, True)))
    torch.testing.assert_close(split, torch.tensor((False, False, False)))
    torch.testing.assert_close(rerouted, torch.tensor((False, False, False)))


def test_joint_fraction_rejects_a_single_incidental_event() -> None:
    strategy = _strategy(
        min_joint_observations=1,
        min_joint_fraction=0.75,
    )
    strategy.update_cancellation_conditioned_split_buffer(
        _outputs(
            signed_x=(1.0, 1.0, 1.0),
            absolute_x=(3.0, 1.0, 1.0),
            extent_x=(10.0, 7.0, 7.0),
        )
    )
    strategy.update_cancellation_conditioned_split_buffer(
        _outputs(
            signed_x=(1.0, 1.0, 1.0),
            absolute_x=(1.0, 1.0, 1.0),
            extent_x=(10.0, 7.0, 7.0),
        )
    )

    clone, split, rerouted = _candidate_masks(strategy)
    torch.testing.assert_close(clone, torch.tensor((True, True, True)))
    torch.testing.assert_close(split, torch.tensor((False, False, False)))
    torch.testing.assert_close(rerouted, torch.tensor((False, False, False)))


def test_reroute_ceiling_fails_closed() -> None:
    strategy = _strategy(
        min_joint_observations=1,
        max_reroute_fraction=0.5,
    )
    strategy.update_cancellation_conditioned_split_buffer(
        _outputs(
            signed_x=(1.0, 1.0, 1.0),
            absolute_x=(3.0, 3.0, 1.0),
            extent_x=(10.0, 10.0, 7.0),
        )
    )

    with pytest.raises(RuntimeError, match="reroute fraction"):
        _candidate_masks(strategy)


def test_checkpoint_reset_and_prune_preserve_aligned_counts() -> None:
    strategy = _strategy()
    strategy.densify_cancellation_joint_observations[:] = torch.tensor(
        ((2,), (1,), (0,)),
        dtype=torch.int32,
    )
    strategy.densify_cancellation_valid_observations[:] = torch.tensor(
        ((3,), (2,), (1,)),
        dtype=torch.int32,
    )
    checkpoint = strategy.get_strategy_parameters()
    restored = _strategy()
    restored.init_densification_buffer(checkpoint)

    torch.testing.assert_close(
        restored.densify_cancellation_joint_observations,
        strategy.densify_cancellation_joint_observations,
    )
    restored.prune_densification_buffers(torch.tensor((True, False, True)))
    torch.testing.assert_close(
        restored.densify_cancellation_joint_observations,
        torch.tensor(((2,), (0,)), dtype=torch.int32),
    )
    restored.model.positions = torch.nn.Parameter(restored.model.positions.detach()[torch.tensor((True, False, True))])
    restored.reset_densification_buffers()
    assert restored.densify_cancellation_joint_observations.shape == (2, 1)
    assert not bool(restored.densify_cancellation_joint_observations.any())


@pytest.mark.parametrize("render_method", ("3dgrt", "torch"))
def test_enabled_arm_requires_3dgut(render_method: str) -> None:
    with pytest.raises(ValueError, match="render.method=3dgut"):
        _strategy(render_method=render_method)


def test_enabled_arm_rejects_other_structural_routing() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _strategy(projected_extent_enabled=True)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    (
        ("cancellation_threshold", 1.0, "cancellation_threshold"),
        ("extent_px", 0.0, "extent_px"),
        ("min_joint_observations", True, "positive integer"),
        ("min_joint_fraction", 0.0, "min_joint_fraction"),
        ("max_reroute_fraction", 1.0, "max_reroute_fraction"),
    ),
)
def test_enabled_arm_validates_preregistered_thresholds(
    parameter: str,
    value: object,
    message: str,
) -> None:
    arguments = {parameter: value}
    with pytest.raises(ValueError, match=message):
        _strategy(**arguments)
