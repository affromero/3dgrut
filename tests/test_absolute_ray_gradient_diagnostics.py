"""Behavior tests for diagnostic 3DGUT ray-gradient cancellation evidence."""

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.strategy.gs import GSStrategy
from threedgut_tracer.tracer import Tracer


class _Model:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.positions = torch.nn.Parameter(
            torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        )
        self.features_specular = torch.nn.Parameter(torch.ones((2, 3)))
        self.scale = torch.nn.Parameter(
            torch.log(
                torch.tensor(((1.0, 0.1, 0.1), (1.0, 1.0, 1.0)))
            )
        )

    @property
    def num_gaussians(self) -> int:
        return 2

    def get_scale(self) -> torch.Tensor:
        return torch.exp(self.scale)


class _Writer:
    def __init__(self) -> None:
        self.values: dict[str, tuple[float, int]] = {}

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.values[name] = (value, step)


def _strategy(
    *,
    diagnostics_enabled: bool = True,
    densification_enabled: bool = False,
    render_method: str = "3dgut",
    tile_weighting: bool = False,
    frequency: object = 10,
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
                    "clone_grad_threshold": 0.0002,
                    "split_grad_threshold": 0.0002,
                    "relative_size_threshold": 0.01,
                    "split": {"n_gaussians": 2},
                    "theta_aware": {"enabled": False},
                    "feature_grad": {"enabled": False},
                    "tile_coverage_weighted_gradient": {
                        "enabled": tile_weighting,
                    },
                    "absolute_ray_gradient_diagnostics": {
                        "enabled": diagnostics_enabled,
                        "frequency": frequency,
                    },
                    "absolute_ray_gradient_densification": {
                        "enabled": densification_enabled,
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
    return GSStrategy(config, _Model())


def _outputs() -> dict[str, object]:
    return {
        "mog_signed_ray_position_grad": torch.tensor(
            ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
        ),
        "mog_abs_ray_position_grad": torch.tensor(
            ((3.0, 0.0, 0.0), (0.0, 2.0, 0.0))
        ),
        "mog_projected_extent": torch.tensor(((10.0, 4.0), (6.0, 5.0))),
    }


def test_metrics_measure_cancellation_and_demand_weighted_shape() -> None:
    strategy = _strategy()
    writer = _Writer()

    metrics = strategy.log_absolute_ray_gradient_diagnostics(
        outputs=_outputs(),
        sensor_position=torch.zeros(3),
        writer=writer,
        step=20,
    )

    assert metrics["active_gaussians"] == 2.0
    assert metrics["cancellation_global"] == pytest.approx(0.4)
    assert metrics["cancellation_weighted_mean"] == pytest.approx(0.4)
    assert metrics["cancellation_gt_0p5_mass_fraction"] == pytest.approx(0.6)
    assert metrics["cancellation_gt_0p9_mass_fraction"] == 0.0
    assert metrics["extent_weighted_mean_px"] == pytest.approx(8.4)
    assert metrics["extent_gt_8_mass_fraction"] == pytest.approx(0.6)
    assert metrics["extent_gt_16_mass_fraction"] == 0.0
    assert metrics["condition_gt_8_mass_fraction"] == pytest.approx(0.6)
    key = "train/densify/ray_abs/cancellation_global"
    assert writer.values[key] == pytest.approx((0.4, 20))


def test_metrics_reject_signed_mass_above_absolute_upper_bound() -> None:
    strategy = _strategy()
    outputs = _outputs()
    outputs["mog_signed_ray_position_grad"] = torch.tensor(
        ((4.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    )

    with pytest.raises(RuntimeError, match="upper bound"):
        strategy.log_absolute_ray_gradient_diagnostics(
            outputs=outputs,
            sensor_position=torch.zeros(3),
            writer=None,
            step=20,
        )


def test_mass_matched_densification_redistributes_but_conserves_dose() -> None:
    strategy = _strategy(
        diagnostics_enabled=False,
        densification_enabled=True,
    )
    strategy.init_densification_buffer()

    strategy.update_gradient_buffer(
        sensor_position=torch.zeros(3),
        outputs=_outputs(),
    )

    torch.testing.assert_close(
        strategy.densify_grad_norm_accum,
        torch.tensor(((0.9,), (0.6,))),
    )
    assert strategy.densify_grad_norm_accum.sum().item() == pytest.approx(1.5)
    torch.testing.assert_close(
        strategy.densify_grad_norm_denom,
        torch.ones((2, 1), dtype=torch.int32),
    )


@pytest.mark.parametrize("render_method", ("3dgrt", "torch"))
def test_enabled_diagnostics_require_3dgut(render_method: str) -> None:
    with pytest.raises(ValueError, match=r"render\.method=3dgut"):
        _strategy(render_method=render_method)


def test_diagnostics_reject_tile_weighted_signed_control() -> None:
    with pytest.raises(ValueError, match="not tile weighting"):
        _strategy(tile_weighting=True)


def test_densification_rejects_other_structural_dose() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _strategy(
            diagnostics_enabled=False,
            densification_enabled=True,
            tile_weighting=True,
        )


@pytest.mark.parametrize("frequency", (True, 0, 1.5))
def test_diagnostics_require_positive_integer_frequency(
    frequency: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _strategy(frequency=frequency)


def test_tracer_treats_mcmc_strategy_as_no_gs_densification() -> None:
    config = OmegaConf.create(
        {
            "strategy": {
                "method": "MCMCStrategy",
                "add": {"max_n_gaussians": 1_000_000},
            }
        }
    )

    densify = Tracer._optional_densify_config(config)

    assert dict(densify) == {}
