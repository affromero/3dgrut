"""Behavior tests for covariance-conditioned gradient split routing."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.strategy.gs import GSStrategy


class _StrategyModel:
    def __init__(self, scales: tuple[float, ...]) -> None:
        self.device = torch.device("cpu")
        count = len(scales)
        x_positions = torch.arange(1, count + 1, dtype=torch.float32)
        self.positions = torch.nn.Parameter(
            torch.stack(
                (x_positions, torch.zeros(count), torch.zeros(count)),
                dim=1,
            )
        )
        self.density = torch.nn.Parameter(torch.zeros((count, 1)))
        self.features_albedo = torch.nn.Parameter(torch.ones((count, 3)))
        self.features_specular = torch.nn.Parameter(torch.ones((count, 3)))
        self.rotation = torch.nn.Parameter(torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(count, 1))
        physical_scales = torch.tensor(scales)[:, None].repeat(1, 3)
        self.scale = torch.nn.Parameter(torch.log(physical_scales))
        parameter_names = (
            "positions",
            "density",
            "features_albedo",
            "features_specular",
            "rotation",
            "scale",
        )
        parameter_groups = [
            {
                "params": [getattr(self, name)],
                "name": name,
                "lr": 0.001,
            }
            for name in parameter_names
        ]
        self.optimizer = torch.optim.Adam(parameter_groups)

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_density(self) -> torch.Tensor:
        return torch.sigmoid(self.density)

    def get_scale(self) -> torch.Tensor:
        return self.scale_activation(self.scale)

    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.rotation, dim=1)

    @staticmethod
    def scale_activation(value: torch.Tensor) -> torch.Tensor:
        return torch.exp(value)

    @staticmethod
    def scale_activation_inv(value: torch.Tensor) -> torch.Tensor:
        return torch.log(value)


def _strategy(
    *,
    enabled: bool = True,
    projected_extent_enabled: bool = False,
    render_method: str = "3dgut",
    radius_px: float = 8.0,
    min_large_observations: int = 6,
    max_reroute_fraction: float = 0.22385751031338305,
    max_invalid_fraction: float = 1e-5,
    scales: tuple[float, ...] = (0.5, 0.5),
    print_stats: bool = False,
) -> tuple[GSStrategy, _StrategyModel]:
    model = _StrategyModel(scales)
    config = OmegaConf.create(
        {
            "render": {"method": render_method},
            "model": {
                "density_activation": "sigmoid",
                "scale_activation": "exp",
            },
            "strategy": {
                "print_stats": print_stats,
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
                    "projected_extent_split": {
                        "enabled": projected_extent_enabled,
                        "max_px": 8.0,
                    },
                    "covariance_gradient_split": {
                        "enabled": enabled,
                        "radius_px": radius_px,
                        "min_large_observations": min_large_observations,
                        "max_reroute_fraction": max_reroute_fraction,
                        "max_invalid_fraction": max_invalid_fraction,
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


def _renderer_outputs(
    radii: tuple[float, ...],
    tile_counts: tuple[int, ...] | None = None,
) -> dict[str, object]:
    inverse_variances = torch.tensor(
        tuple(1.0 / radius**2 for radius in radii),
        dtype=torch.float32,
    )
    conic_opacity = torch.stack(
        (
            inverse_variances,
            torch.zeros_like(inverse_variances),
            inverse_variances,
            torch.ones_like(inverse_variances),
        ),
        dim=1,
    )
    if tile_counts is None:
        tile_counts = tuple(1 for _ in radii)
    return {
        "mog_projected_conic_opacity": conic_opacity,
        "mog_tiles_count": torch.tensor(tile_counts, dtype=torch.int32),
    }


def _packet_outputs(
    packets: torch.Tensor,
    tile_counts: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if tile_counts is None:
        tile_counts = tuple(1 for _ in range(packets.shape[0]))
    return {
        "mog_projected_conic_opacity": packets,
        "mog_tiles_count": torch.tensor(tile_counts, dtype=torch.int32),
    }


def test_disabled_arm_preserves_world_decisions_and_checkpoint_contract() -> None:
    strategy, _ = _strategy(
        enabled=False,
        render_method="3dgrt",
        scales=(0.5, 2.0),
    )
    clone, split, rerouted = strategy._densify_candidate_masks(
        torch.tensor((0.3, 0.5)),
        1.0,
        None,
        None,
    )

    torch.testing.assert_close(clone, torch.tensor((True, False)))
    torch.testing.assert_close(split, torch.tensor((False, True)))
    torch.testing.assert_close(rerouted, torch.tensor((False, False)))
    parameters = strategy.get_strategy_parameters()
    assert "densify_covariance_gradient_mass" not in parameters
    assert "densify_covariance_large_observations" not in parameters
    assert "densify_covariance_packet_observations" not in parameters
    assert "densify_covariance_invalid_packet_observations" not in parameters
    assert "densify_covariance_invalid_reason_counts" not in parameters
    assert "densify_covariance_invalid_example_indices" not in parameters
    assert "densify_covariance_invalid_example_values" not in parameters


def test_covariance_and_extent_arms_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        _strategy(projected_extent_enabled=True)


@pytest.mark.parametrize("render_method", ("3dgrt", "torch"))
def test_enabled_arm_requires_3dgut(render_method: str) -> None:
    with pytest.raises(ValueError, match="render.method=3dgut"):
        _strategy(render_method=render_method)


@pytest.mark.parametrize("radius_px", (0.0, -1.0, float("inf"), float("nan")))
def test_radius_must_be_finite_and_positive(radius_px: float) -> None:
    with pytest.raises(ValueError, match="radius_px"):
        _strategy(radius_px=radius_px)


@pytest.mark.parametrize("min_large_observations", (0, -1, 1.5, True))
def test_minimum_observation_count_must_be_positive_integer(
    min_large_observations: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _strategy(min_large_observations=min_large_observations)


@pytest.mark.parametrize(
    "max_reroute_fraction",
    (0.0, 1.0, -0.1, float("inf"), float("nan")),
)
def test_reroute_ceiling_must_define_a_strict_subset(
    max_reroute_fraction: float,
) -> None:
    with pytest.raises(ValueError, match=r"in \(0, 1\)"):
        _strategy(max_reroute_fraction=max_reroute_fraction)


@pytest.mark.parametrize(
    "max_invalid_fraction",
    (-0.1, 1.0, True, float("inf"), float("nan")),
)
def test_invalid_packet_ceiling_must_be_a_finite_fraction(
    max_invalid_fraction: float,
) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\)"):
        _strategy(max_invalid_fraction=max_invalid_fraction)


def test_conic_math_recovers_major_radius_after_rotation() -> None:
    angle = torch.tensor(torch.pi / 4.0)
    rotation = torch.tensor(
        (
            (torch.cos(angle), -torch.sin(angle)),
            (torch.sin(angle), torch.cos(angle)),
        )
    )
    covariance = rotation @ torch.diag(torch.tensor((4.0, 64.0))) @ rotation.T
    inverse_covariance = torch.linalg.inv(covariance)
    conic = torch.tensor(
        (
            (
                inverse_covariance[0, 0],
                inverse_covariance[0, 1],
                inverse_covariance[1, 1],
            ),
        )
    )

    radius = GSStrategy._major_eigen_radius_from_inverse_covariance(conic)

    torch.testing.assert_close(
        radius,
        torch.tensor((8.0,), dtype=torch.float64),
        rtol=1e-5,
        atol=1e-5,
    )


def test_float64_conic_math_recovers_float32_false_negative() -> None:
    strategy, model = _strategy(scales=(0.5,))
    model.positions.grad = torch.tensor(((2.0, 0.0, 0.0),))
    conic = torch.tensor(
        ((0.9325225353240967, 0.25084713101387024, 0.06747749447822571),),
        dtype=torch.float32,
    )
    determinant_32 = conic[0, 0] * conic[0, 2] - conic[0, 1].square()
    conic_64 = conic.to(torch.float64)
    determinant_64 = conic_64[0, 0] * conic_64[0, 2] - conic_64[0, 1].square()
    assert determinant_32 <= 0.0
    assert determinant_64 > 0.0
    packet = torch.cat((conic, torch.ones((1, 1))), dim=1)

    strategy.update_covariance_gradient_buffer(
        _packet_outputs(packet),
        sensor_position=torch.zeros(3),
    )

    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass,
        torch.tensor(((1.0,),)),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.tensor(((1,),), dtype=torch.int32),
    )
    assert strategy.densify_covariance_packet_observations.item() == 1
    assert strategy.densify_covariance_invalid_packet_observations.item() == 0


def test_accumulation_pairs_gradient_with_same_large_observation() -> None:
    strategy, model = _strategy(scales=(0.5, 0.5, 0.5))
    model.positions.grad = torch.tensor(((2.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    sensor_position = torch.zeros(3)

    strategy.update_covariance_gradient_buffer(
        _renderer_outputs((9.0, 7.0, 10.0), (1, 1, 0)),
        sensor_position=sensor_position,
    )

    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass,
        torch.tensor(((1.0,), (0.0,), (0.0,))),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.tensor(((1,), (0,), (0,)), dtype=torch.int32),
    )

    strategy.update_covariance_gradient_buffer(
        _renderer_outputs((8.0, 9.0, 10.0), (1, 1, 0)),
        sensor_position=sensor_position,
    )

    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass,
        torch.tensor(((1.0,), (2.0,), (0.0,))),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.tensor(((1,), (1,), (0,)), dtype=torch.int32),
    )


def test_zero_gradient_is_absent_from_both_existing_denominators() -> None:
    strategy, model = _strategy()
    model.positions.grad = torch.zeros_like(model.positions)

    strategy.update_covariance_gradient_buffer(
        _renderer_outputs((9.0, 7.0)),
        sensor_position=torch.zeros(3),
    )

    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass,
        torch.zeros((2, 1)),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.zeros((2, 1), dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("outputs", "message"),
    (
        ({}, "mog_projected_conic_opacity"),
        (
            {"mog_projected_conic_opacity": torch.zeros((2, 4))},
            "mog_tiles_count",
        ),
        (
            {
                "mog_projected_conic_opacity": torch.zeros((2, 3)),
                "mog_tiles_count": torch.ones(2),
            },
            "must have shape",
        ),
    ),
)
def test_accumulation_rejects_missing_or_misaligned_renderer_state(
    outputs: dict[str, object],
    message: str,
) -> None:
    strategy, model = _strategy()
    model.positions.grad = torch.ones_like(model.positions)

    with pytest.raises((RuntimeError, ValueError), match=message):
        strategy.update_covariance_gradient_buffer(
            outputs,
            sensor_position=torch.zeros(3),
        )


def test_invalid_predicates_are_disjoint_and_record_first_examples() -> None:
    strategy, model = _strategy(
        scales=(0.5,) * 5,
        max_invalid_fraction=0.9,
    )
    model.positions.grad = torch.full_like(model.positions, 2.0)
    inverse_variance = 1.0 / 9.0**2
    packets = torch.tensor(
        (
            (float("nan"), 0.0, inverse_variance, 1.0),
            (-1.0, 0.0, 1.0, 1.0),
            (1.0, 2.0, 1.0, 1.0),
            (inverse_variance, 0.0, inverse_variance, 0.0),
            (inverse_variance, 0.0, inverse_variance, 1.0),
        ),
        dtype=torch.float32,
    )

    strategy.update_covariance_gradient_buffer(
        _packet_outputs(packets),
        sensor_position=torch.zeros(3),
    )

    torch.testing.assert_close(
        strategy.densify_covariance_invalid_reason_counts,
        torch.ones(4, dtype=torch.int64),
    )
    assert strategy.densify_covariance_packet_observations.item() == 5
    assert strategy.densify_covariance_invalid_packet_observations.item() == 4
    torch.testing.assert_close(
        strategy.densify_covariance_invalid_example_indices[:, 0],
        torch.arange(4, dtype=torch.int64),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_invalid_example_values[:, 0],
        packets[:4].to(torch.float64),
        equal_nan=True,
    )
    assert torch.isnan(strategy.densify_covariance_invalid_example_values[:, 1:]).all()
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.tensor(((0,), (0,), (0,), (0,), (1,)), dtype=torch.int32),
    )
    calls: dict[str, tuple[float, int]] = {}

    def add_scalar(name: str, value: float, step: int) -> None:
        calls[name] = (value, step)

    strategy._validate_covariance_packet_window(
        writer=SimpleNamespace(add_scalar=add_scalar),
        step=600,
    )
    for reason_name in (
        "nonfinite",
        "nonpositive_diagonal",
        "nonpositive_determinant",
        "nonpositive_opacity",
    ):
        assert calls[f"train/densify/covgrad_invalid_{reason_name}"] == (
            1.0,
            600,
        )


def test_true_indefinite_packet_is_excluded_but_total_n_is_unchanged() -> None:
    strategy, model = _strategy()
    model.positions.grad = torch.ones_like(model.positions)
    batch = SimpleNamespace(T_to_world=torch.eye(4).unsqueeze(0))
    packets = torch.tensor(
        (
            (1.0, 2.0, 1.0, 1.0),
            (1.0, 0.0, 1.0, 1.0),
        )
    )

    assert not strategy.post_backward(
        step=1,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=_packet_outputs(packets),
    )

    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass,
        torch.zeros((2, 1)),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.zeros((2, 1), dtype=torch.int32),
    )
    torch.testing.assert_close(
        strategy.densify_grad_norm_denom,
        torch.ones((2, 1), dtype=torch.int32),
    )
    assert strategy.densify_covariance_invalid_packet_observations.item() == 1


def test_covariance_window_closes_after_last_reachable_densification() -> None:
    strategy, model = _strategy()
    assert strategy._covariance_gradient_window_is_open(14_700)
    assert not strategy._covariance_gradient_window_is_open(14_701)
    model.positions.grad = torch.ones_like(model.positions)
    batch = SimpleNamespace(T_to_world=torch.eye(4).unsqueeze(0))
    packets = torch.tensor(
        (
            (1.0, 2.0, 1.0, 1.0),
            (1.0 / 81.0, 0.0, 1.0 / 81.0, 1.0),
        )
    )

    assert not strategy.post_backward(
        step=14_701,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=_packet_outputs(packets),
    )

    assert strategy.densify_covariance_packet_observations.item() == 0
    assert strategy.densify_covariance_invalid_packet_observations.item() == 0
    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass,
        torch.zeros((2, 1)),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.zeros((2, 1), dtype=torch.int32),
    )
    torch.testing.assert_close(
        strategy.densify_grad_norm_denom,
        torch.ones((2, 1), dtype=torch.int32),
    )


def test_invalid_examples_are_capped_at_eight_per_predicate() -> None:
    strategy, model = _strategy(scales=(0.5,) * 10)
    model.positions.grad = torch.ones_like(model.positions)
    packets = torch.tensor(((1.0, 2.0, 1.0, 1.0),) * 10)

    strategy.update_covariance_gradient_buffer(
        _packet_outputs(packets),
        sensor_position=torch.zeros(3),
    )

    assert strategy.densify_covariance_invalid_reason_counts[2].item() == 10
    torch.testing.assert_close(
        strategy.densify_covariance_invalid_example_indices[2],
        torch.arange(8, dtype=torch.int64),
    )


def test_candidate_partition_requires_count_and_conditioned_gradient() -> None:
    strategy, _ = _strategy(scales=(0.5,) * 5)
    gradient_mass = torch.tensor(((1.8,), (1.8,), (1.14,), (1.14,), (0.0,)))
    observation_count = torch.tensor(((6,), (5,), (6,), (6,), (0,)))
    total_observation_count = torch.tensor(((6,), (6,), (6,), (6,), (0,)))
    clone, split, rerouted = strategy._densify_candidate_masks(
        torch.full((5,), 0.3),
        1.0,
        None,
        None,
        gradient_mass,
        observation_count,
        total_observation_count,
    )

    torch.testing.assert_close(
        rerouted,
        torch.tensor((True, False, False, False, False)),
    )
    torch.testing.assert_close(
        clone,
        torch.tensor((False, True, True, True, True)),
    )
    torch.testing.assert_close(split, rerouted)


def test_candidate_partition_fails_closed_above_reroute_ceiling() -> None:
    strategy, _ = _strategy(scales=(0.5,) * 5)
    gradient_mass = torch.tensor(((1.8,), (1.8,), (0.0,), (0.0,), (0.0,)))
    observation_count = torch.tensor(((6,), (6,), (0,), (0,), (0,)))
    total_observation_count = torch.tensor(((6,), (6,), (0,), (0,), (0,)))

    with pytest.raises(RuntimeError, match="preregistered ceiling"):
        strategy._densify_candidate_masks(
            torch.full((5,), 0.3),
            1.0,
            None,
            None,
            gradient_mass,
            observation_count,
            total_observation_count,
        )


def test_candidate_partition_validates_total_observation_denominator() -> None:
    strategy, _ = _strategy(scales=(0.5,) * 5)
    gradient_mass = torch.tensor(((1.8,), (0.0,), (0.0,), (0.0,), (0.0,)))
    large_observation_count = torch.tensor(((6,), (0,), (0,), (0,), (0,)))

    with pytest.raises(RuntimeError, match="total-observation denominator"):
        strategy._densify_candidate_masks(
            torch.full((5,), 0.3),
            1.0,
            None,
            None,
            gradient_mass,
            large_observation_count,
        )

    invalid_total_count = torch.tensor(((5,), (0,), (0,), (0,), (0,)))
    with pytest.raises(RuntimeError, match="bounded by the existing total"):
        strategy._densify_candidate_masks(
            torch.full((5,), 0.3),
            1.0,
            None,
            None,
            gradient_mass,
            large_observation_count,
            invalid_total_count,
        )


def test_conditioned_mass_uses_total_observation_denominator() -> None:
    strategy, _ = _strategy(scales=(0.5,) * 5)
    large_gradient_mass = torch.tensor(((1.8,), (0.0,), (0.0,), (0.0,), (0.0,)))
    large_observation_count = torch.tensor(((6,), (0,), (0,), (0,), (0,)))
    total_observation_count = torch.tensor(((30,), (0,), (0,), (0,), (0,)))

    clone, split, rerouted = strategy._densify_candidate_masks(
        torch.full((5,), 0.3),
        1.0,
        None,
        None,
        large_gradient_mass,
        large_observation_count,
        total_observation_count,
    )

    torch.testing.assert_close(
        clone,
        torch.tensor((True, True, True, True, True)),
    )
    torch.testing.assert_close(split, torch.zeros(5, dtype=torch.bool))
    torch.testing.assert_close(rerouted, torch.zeros(5, dtype=torch.bool))


def test_split_children_reset_and_prune_keep_buffers_aligned() -> None:
    strategy, model = _strategy(scales=(0.5,) * 5)
    strategy.densify_grad_norm_accum[:, 0] = 1.8
    strategy.densify_grad_norm_denom[:, 0] = 6
    strategy.densify_covariance_gradient_mass[0, 0] = 1.8
    strategy.densify_covariance_large_observations[0, 0] = 6
    strategy.densify_covariance_packet_observations[0] = 100_000
    strategy.densify_covariance_invalid_packet_observations[0] = 1
    strategy.densify_covariance_invalid_reason_counts[2] = 1
    strategy.densify_covariance_invalid_example_indices[2, 0] = 0
    strategy.densify_covariance_invalid_example_values[2, 0] = torch.tensor(
        (1.0, 2.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    torch.manual_seed(42)

    strategy.densify_gaussians(scene_extent=1.0)

    assert model.num_gaussians == 10
    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass,
        torch.zeros((10, 1)),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations,
        torch.zeros((10, 1), dtype=torch.int32),
    )
    assert strategy.densify_covariance_packet_observations.item() == 0
    assert strategy.densify_covariance_invalid_packet_observations.item() == 0
    torch.testing.assert_close(
        strategy.densify_covariance_invalid_reason_counts,
        torch.zeros(4, dtype=torch.int64),
    )
    assert (strategy.densify_covariance_invalid_example_indices == -1).all()
    assert torch.isnan(strategy.densify_covariance_invalid_example_values).all()
    strategy.densify_covariance_gradient_mass[:, 0] = torch.arange(10)
    strategy.densify_covariance_large_observations[:, 0] = torch.arange(10)
    model.density.data[:, 0] = 0.0
    model.density.data[3, 0] = -10.0

    strategy.prune_gaussians_opacity()

    assert model.num_gaussians == 9
    expected = torch.tensor((0, 1, 2, 4, 5, 6, 7, 8, 9))
    torch.testing.assert_close(
        strategy.densify_covariance_gradient_mass[:, 0],
        expected.float(),
    )
    torch.testing.assert_close(
        strategy.densify_covariance_large_observations[:, 0],
        expected.int(),
    )


def test_checkpoint_restore_requires_mid_window_buffers() -> None:
    strategy, _ = _strategy()
    strategy.densify_covariance_gradient_mass[:, 0] = torch.tensor((1.0, 2.0))
    strategy.densify_covariance_large_observations[:, 0] = torch.tensor((6, 7))
    strategy.densify_covariance_packet_observations[0] = 10
    strategy.densify_covariance_invalid_packet_observations[0] = 1
    strategy.densify_covariance_invalid_reason_counts[2] = 1
    strategy.densify_covariance_invalid_example_indices[2, 0] = 1
    strategy.densify_covariance_invalid_example_values[2, 0] = torch.tensor(
        (1.0, 2.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    checkpoint = strategy.get_strategy_parameters()
    checkpoint["global_step"] = 100

    restored, _ = _strategy()
    restored.init_densification_buffer(checkpoint)
    torch.testing.assert_close(
        restored.densify_covariance_gradient_mass,
        torch.tensor(((1.0,), (2.0,))),
    )
    torch.testing.assert_close(
        restored.densify_covariance_large_observations,
        torch.tensor(((6,), (7,)), dtype=torch.int32),
    )
    assert restored.densify_covariance_packet_observations.item() == 10
    assert restored.densify_covariance_invalid_packet_observations.item() == 1
    torch.testing.assert_close(
        restored.densify_covariance_invalid_reason_counts,
        torch.tensor((0, 0, 1, 0), dtype=torch.int64),
    )
    assert restored.densify_covariance_invalid_example_indices[2, 0] == 1
    torch.testing.assert_close(
        restored.densify_covariance_invalid_example_values[2, 0],
        torch.tensor((1.0, 2.0, 1.0, 1.0), dtype=torch.float64),
    )

    partial = dict(checkpoint)
    partial.pop("densify_covariance_invalid_example_values")
    with pytest.raises(ValueError, match="complete invalid-packet telemetry"):
        restored.init_densification_buffer(partial)

    missing = dict(checkpoint)
    covariance_state_names = (
        "densify_covariance_gradient_mass",
        "densify_covariance_large_observations",
        "densify_covariance_packet_observations",
        "densify_covariance_invalid_packet_observations",
        "densify_covariance_invalid_reason_counts",
        "densify_covariance_invalid_example_indices",
        "densify_covariance_invalid_example_values",
    )
    for name in covariance_state_names:
        missing.pop(name)
    with pytest.raises(ValueError, match="missing mid-window"):
        restored.init_densification_buffer(missing)

    reset_boundary = dict(missing)
    reset_boundary["global_step"] = 601
    restored.init_densification_buffer(reset_boundary)
    torch.testing.assert_close(
        restored.densify_covariance_gradient_mass,
        torch.zeros((2, 1)),
    )
    assert restored.densify_covariance_packet_observations.item() == 0
    assert restored.densify_covariance_invalid_packet_observations.item() == 0

    closed_tail = dict(missing)
    closed_tail["global_step"] = 14_702
    restored.init_densification_buffer(closed_tail)
    assert restored.densify_covariance_packet_observations.item() == 0


def test_disabled_resume_ignores_covariance_buffers_from_enabled_arm() -> None:
    enabled, _ = _strategy()
    checkpoint = enabled.get_strategy_parameters()
    checkpoint["global_step"] = 100
    disabled, _ = _strategy(enabled=False, render_method="3dgrt")

    disabled.init_densification_buffer(checkpoint)

    torch.testing.assert_close(
        disabled.densify_covariance_gradient_mass,
        torch.zeros((2, 1)),
    )
    assert disabled.densify_covariance_packet_observations.item() == 0


def test_invalid_fraction_aborts_at_boundary_before_topology_mutation() -> None:
    strategy, model = _strategy()
    strategy.densify_covariance_packet_observations[0] = 99_999
    strategy.densify_covariance_invalid_packet_observations[0] = 1
    strategy.densify_covariance_invalid_reason_counts[2] = 1
    strategy.densify_covariance_invalid_example_indices[2, 0] = 0
    strategy.densify_covariance_invalid_example_values[2, 0] = torch.tensor(
        (1.0, 2.0, 1.0, 1.0),
        dtype=torch.float64,
    )
    parameter_names = (
        "positions",
        "density",
        "features_albedo",
        "features_specular",
        "rotation",
        "scale",
    )
    parameter_ids_before = tuple(id(getattr(model, name)) for name in parameter_names)
    parameter_values_before = tuple(getattr(model, name).detach().clone() for name in parameter_names)
    optimizer_parameter_ids_before = tuple(id(group["params"][0]) for group in model.optimizer.param_groups)
    optimizer_state_ids_before = tuple(id(state) for state in model.optimizer.state.values())

    with pytest.raises(RuntimeError, match="invalid packet fraction"):
        strategy.densify_gaussians(scene_extent=1.0, step=600)

    assert model.num_gaussians == 2
    assert tuple(id(getattr(model, name)) for name in parameter_names) == (parameter_ids_before)
    for name, expected in zip(parameter_names, parameter_values_before):
        torch.testing.assert_close(getattr(model, name), expected)
    assert tuple(id(group["params"][0]) for group in model.optimizer.param_groups) == optimizer_parameter_ids_before
    assert tuple(id(state) for state in model.optimizer.state.values()) == (optimizer_state_ids_before)


def test_exact_invalid_fraction_ceiling_is_allowed() -> None:
    strategy, _ = _strategy()
    strategy.densify_covariance_packet_observations[0] = 100_000
    strategy.densify_covariance_invalid_packet_observations[0] = 1
    strategy.densify_covariance_invalid_reason_counts[2] = 1
    strategy.densify_covariance_invalid_example_indices[2, 0] = 0
    strategy.densify_covariance_invalid_example_values[2, 0] = torch.tensor(
        (1.0, 2.0, 1.0, 1.0),
        dtype=torch.float64,
    )

    strategy._validate_covariance_packet_window(writer=None, step=600)


def test_zero_invalid_window_emits_exact_packet_telemetry() -> None:
    strategy, model = _strategy()
    model.positions.grad = torch.ones_like(model.positions)
    strategy.update_covariance_gradient_buffer(
        _renderer_outputs((7.0, 7.0)),
        sensor_position=torch.zeros(3),
    )
    calls: dict[str, tuple[float, int]] = {}

    def add_scalar(name: str, value: float, step: int) -> None:
        calls[name] = (value, step)

    strategy._validate_covariance_packet_window(
        writer=SimpleNamespace(add_scalar=add_scalar),
        step=600,
    )

    assert calls["train/densify/covgrad_conic_packets"] == (2.0, 600)
    assert calls["train/densify/covgrad_invalid_packets"] == (0.0, 600)
    assert calls["train/densify/covgrad_invalid_fraction"] == (0.0, 600)
    assert calls["train/densify/covgrad_invalid_nonfinite"] == (0.0, 600)
    assert calls["train/densify/covgrad_invalid_nonpositive_diagonal"] == (
        0.0,
        600,
    )
    assert calls["train/densify/covgrad_invalid_nonpositive_determinant"] == (
        0.0,
        600,
    )
    assert calls["train/densify/covgrad_invalid_nonpositive_opacity"] == (
        0.0,
        600,
    )


def test_densify_event_emits_covariance_telemetry() -> None:
    strategy, _ = _strategy(scales=(0.5,) * 5)
    strategy.densify_covariance_gradient_mass[0, 0] = 1.8
    strategy.densify_covariance_large_observations[0, 0] = 6
    calls: dict[str, tuple[float, int]] = {}

    def add_scalar(name: str, value: float, step: int) -> None:
        calls[name] = (value, step)

    writer = SimpleNamespace(add_scalar=add_scalar)
    strategy.log_densify_stats(
        densify_grad_norm=torch.full((5,), 0.3),
        scene_extent=1.0,
        projected_extent_max=torch.zeros(5),
        covariance_gradient_mass=strategy.densify_covariance_gradient_mass,
        covariance_large_observations=(strategy.densify_covariance_large_observations),
        covariance_total_observations=torch.tensor(
            ((30,), (6,), (6,), (6,), (6,)),
            dtype=torch.int32,
        ),
        writer=writer,
        step=600,
    )

    assert calls["train/densify/covgrad_conditioned_eligible"] == (0.0, 600)
    assert calls["train/densify/covgrad_rerouted_splits"] == (0.0, 600)
    assert calls["train/densify/covgrad_reroute_fraction"] == (0.0, 600)
