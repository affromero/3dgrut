"""Behavior tests for tile-coverage-weighted densification evidence."""

import pytest
import torch
from omegaconf import OmegaConf
from threedgrut.strategy.gs import GSStrategy


class _Model:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.positions = torch.nn.Parameter(torch.tensor(((1.0, 0.0, 0.0),)))
        self.features_specular = torch.nn.Parameter(torch.ones((1, 3)))

    @property
    def num_gaussians(self) -> int:
        return 1


def _strategy(
    *,
    enabled: bool,
    render_method: str = "3dgut",
) -> tuple[GSStrategy, _Model]:
    model = _Model()
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
                    "clone_grad_threshold": 0.0002,
                    "split_grad_threshold": 0.0002,
                    "relative_size_threshold": 0.01,
                    "split": {"n_gaussians": 2},
                    "theta_aware": {"enabled": False},
                    "feature_grad": {"enabled": False},
                    "tile_coverage_weighted_gradient": {
                        "enabled": enabled,
                    },
                    "projected_extent_split": {"enabled": False},
                    "covariance_gradient_split": {"enabled": False},
                    "scale_shape_split": {"enabled": False},
                },
                "prune": {"density_threshold": 0.005},
                "reset_density": {"new_max_density": 0.01},
            },
        }
    )
    strategy = GSStrategy(config, model)
    strategy.init_densification_buffer()
    return strategy, model


def _set_x_gradient(model: _Model, value: float) -> None:
    model.positions.grad = torch.tensor(((value, 0.0, 0.0),))


def test_enabled_weighting_computes_tile_weighted_view_mean() -> None:
    """One-tile and three-tile views contribute in a 1:3 ratio."""
    strategy, model = _strategy(enabled=True)
    sensor = torch.zeros(3)
    _set_x_gradient(model, 2.0)
    strategy.update_gradient_buffer(
        sensor,
        outputs={"mog_tiles_count": torch.tensor((1,), dtype=torch.int32)},
    )
    _set_x_gradient(model, 4.0)
    strategy.update_gradient_buffer(
        sensor,
        outputs={"mog_tiles_count": torch.tensor((3,), dtype=torch.int32)},
    )

    score = strategy.densify_grad_norm_accum / strategy.densify_grad_norm_denom
    torch.testing.assert_close(score, torch.tensor(((1.75,),)))
    torch.testing.assert_close(
        strategy.densify_grad_norm_denom,
        torch.tensor(((4,),), dtype=torch.int32),
    )


def test_disabled_weighting_preserves_per_view_mean_without_outputs() -> None:
    """The default path remains the ordinary two-observation average."""
    strategy, model = _strategy(enabled=False, render_method="3dgrt")
    sensor = torch.zeros(3)
    _set_x_gradient(model, 2.0)
    strategy.update_gradient_buffer(sensor)
    _set_x_gradient(model, 4.0)
    strategy.update_gradient_buffer(sensor)

    score = strategy.densify_grad_norm_accum / strategy.densify_grad_norm_denom
    torch.testing.assert_close(score, torch.tensor(((1.5,),)))
    torch.testing.assert_close(
        strategy.densify_grad_norm_denom,
        torch.tensor(((2,),), dtype=torch.int32),
    )


@pytest.mark.parametrize("render_method", ["3dgrt", "torch"])
def test_enabled_weighting_requires_3dgut(render_method: str) -> None:
    """The intervention cannot run without 3DGUT coverage evidence."""
    with pytest.raises(ValueError, match=r"render\.method=3dgut"):
        _strategy(enabled=True, render_method=render_method)


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ({}, "mog_tiles_count"),
        (
            {"mog_tiles_count": torch.tensor((1.0,))},
            "integer dtype",
        ),
        (
            {"mog_tiles_count": torch.tensor((-1,), dtype=torch.int32)},
            "non-negative",
        ),
        (
            {"mog_tiles_count": torch.tensor((1, 2), dtype=torch.int32)},
            "one value per Gaussian",
        ),
    ],
)
def test_enabled_weighting_rejects_invalid_renderer_evidence(
    outputs: dict[str, object],
    message: str,
) -> None:
    """Malformed coverage evidence cannot silently alter topology."""
    strategy, model = _strategy(enabled=True)
    _set_x_gradient(model, 2.0)

    with pytest.raises((RuntimeError, ValueError), match=message):
        strategy.update_gradient_buffer(torch.zeros(3), outputs=outputs)
