"""Behavior tests for recovered visibility-adaptive densification primitives."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from threedgrut.optimizers.visibility_selective_adam import VisibilitySelectiveAdam
from threedgrut.strategy.visibility_adaptive import (
    VISIBILITY_ADAPTIVE_CHILD_OFFSET,
    VISIBILITY_ADAPTIVE_COHERENCE_INTERVAL,
    VISIBILITY_ADAPTIVE_COHERENCE_PARTIAL_PREFIX,
    VISIBILITY_ADAPTIVE_FORWARD_VISIBILITY_SOURCE,
    VISIBILITY_ADAPTIVE_IMAGE_HUBER_WEIGHT,
    VISIBILITY_ADAPTIVE_IMAGE_SSIM_WEIGHT,
    VISIBILITY_ADAPTIVE_MAX_CLOUD_COUNT,
    VISIBILITY_ADAPTIVE_MAX_ITERATIONS,
    VISIBILITY_ADAPTIVE_POSITION_LEARNING_RATE,
    VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN,
    VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_TANGENT,
    VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT,
    VISIBILITY_ADAPTIVE_REGULAR_PRUNE_RADIUS_THRESHOLD,
    VISIBILITY_ADAPTIVE_SH_LEARNING_RATE,
    VISIBILITY_ADAPTIVE_STATUS_WINDOW_SIZE,
    VISIBILITY_ADAPTIVE_TANGENT_PROJECTED_GRADIENT_SCALE,
    VisibilityAdaptiveStrategy,
    visibility_adaptive_coherence_permutation,
    visibility_adaptive_compact_point_weights,
    visibility_adaptive_densify_mask,
    visibility_adaptive_jacobian_projected_gradient_pixels,
    visibility_adaptive_normalized_radius,
    visibility_adaptive_preprocess_gradients,
    visibility_adaptive_projected_gradient_pixels,
    visibility_adaptive_projected_radius_pixels,
    visibility_adaptive_projected_size_pixels,
    visibility_adaptive_projected_size_from_weight,
    visibility_adaptive_regular_prune_keep_mask,
    visibility_adaptive_split_children,
    visibility_adaptive_split_opacity,
    visibility_adaptive_tangent_projected_gradient_pixels,
    visibility_adaptive_topology_schedule,
    visibility_adaptive_update_point_status,
)


class _NativeStrategyModel:
    def __init__(
        self,
        *,
        specular_width: int = 3,
        optimize_specular: bool = True,
    ) -> None:
        self.device = torch.device("cpu")
        self.positions = torch.nn.Parameter(
            torch.tensor(((0.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
        )
        self.density = torch.nn.Parameter(torch.zeros((2, 1)))
        self.features_albedo = torch.nn.Parameter(torch.ones((2, 3)))
        self.features_specular = torch.nn.Parameter(
            torch.ones((2, specular_width))
        )
        self.rotation = torch.nn.Parameter(
            torch.tensor(((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
        )
        self.scale = torch.nn.Parameter(torch.zeros((2, 3)))
        self.environment_mask = torch.zeros(2, dtype=torch.bool)
        parameter_names = [
            "positions",
            "density",
            "features_albedo",
        ]
        if optimize_specular:
            parameter_names.append("features_specular")
        parameter_names.extend(("rotation", "scale"))
        groups = [
            {"params": [getattr(self, name)], "name": name, "lr": 0.001}
            for name in parameter_names
        ]
        self.optimizer = VisibilitySelectiveAdam(groups)
        self.optimizer.step(torch.zeros((2, 1)))

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

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
    def scale_activation_inv(value: torch.Tensor) -> torch.Tensor:
        return torch.log(value)


def _native_strategy(
    *,
    specular_width: int = 3,
    optimize_specular: bool = True,
    strategy_overrides: dict[str, object] | None = None,
) -> tuple[VisibilityAdaptiveStrategy, _NativeStrategyModel]:
    model = _NativeStrategyModel(
        specular_width=specular_width,
        optimize_specular=optimize_specular,
    )
    config_data = {
        "model": {
            "density_activation": "sigmoid",
            "scale_activation": "exp",
        },
        "loss": {
            "use_fixed_image_loss_denominator": True,
            "use_camera_loss_weights": True,
            "camera_loss_weights": [1.0, 1.0, 2.0, 2.0],
            "camera_loss_weights_use_physical_camera": True,
        },
        "n_iterations": 10_001,
        "strategy": {
            "print_stats": False,
            "max_gaussians": 14_000_000,
            "densify": {
                "frequency": 100,
                "start_iteration": 0,
                "end_iteration": -1,
            },
            "prune": {
                "frequency": 100,
                "start_iteration": 0,
                "end_iteration": -1,
                "density_threshold": 0.005,
            },
        },
    }
    if strategy_overrides is not None:
        config_data["strategy"].update(strategy_overrides)
    config = OmegaConf.create(config_data)
    strategy = VisibilityAdaptiveStrategy(config, model)
    strategy.init_densification_buffer()
    return strategy, model


def _repeat_native_strategy_rows(
    *,
    strategy: VisibilityAdaptiveStrategy,
    model: _NativeStrategyModel,
    repeat_count: int,
) -> None:
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        parameter.data = parameter.data.repeat((repeat_count, 1))
        state = model.optimizer.state[parameter]
        state["exp_avg"] = state["exp_avg"].repeat((repeat_count, 1))
        state["exp_avg_sq"] = state["exp_avg_sq"].repeat((repeat_count, 1))
        if group["name"] == "positions":
            state["gaussian_steps"] = state["gaussian_steps"].repeat(
                repeat_count
            )
    strategy.point_status = strategy.point_status.repeat((repeat_count, 1))
    model.environment_mask = model.environment_mask.repeat(
        repeat_count
    )


def test_native_static_training_contract_matches_recovered_defaults() -> None:
    """Static limits and image objective retain native constructor values."""
    assert VISIBILITY_ADAPTIVE_MAX_ITERATIONS == 100_000
    assert VISIBILITY_ADAPTIVE_MAX_CLOUD_COUNT == 16_777_216
    assert VISIBILITY_ADAPTIVE_STATUS_WINDOW_SIZE == 32.0
    assert VISIBILITY_ADAPTIVE_COHERENCE_INTERVAL == 100
    assert VISIBILITY_ADAPTIVE_IMAGE_HUBER_WEIGHT == 0.8
    assert VISIBILITY_ADAPTIVE_IMAGE_SSIM_WEIGHT == 0.2
    assert VISIBILITY_ADAPTIVE_POSITION_LEARNING_RATE == 0.001
    assert VISIBILITY_ADAPTIVE_SH_LEARNING_RATE == 0.1


@pytest.mark.parametrize(
    "invalid_mask",
    (
        torch.zeros((2,), dtype=torch.float32),
        torch.zeros((2, 1), dtype=torch.bool),
    ),
)
def test_native_strategy_rejects_invalid_environment_mask(
    invalid_mask: torch.Tensor,
) -> None:
    """Structural environment identity must stay typed and row aligned."""
    strategy, model = _native_strategy()
    model.environment_mask = invalid_mask

    with pytest.raises(ValueError, match="environment mask"):
        strategy.init_densification_buffer()


def test_native_split_appends_children_along_rotated_largest_axis() -> None:
    """Each selected parent produces two ordered deterministic children."""
    half_turn_z = torch.tensor(
        (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)), dtype=torch.float32
    )
    positions = torch.tensor(((1.0, 2.0, 3.0),), dtype=torch.float32)
    scales = torch.tensor(((2.0, 4.0, 1.0),), dtype=torch.float32)

    child_positions, child_scales, child_rotations = (
        visibility_adaptive_split_children(
            positions=positions,
            physical_scales=scales,
            rotations=half_turn_z[None, :],
        )
    )

    offset = 4.0 * VISIBILITY_ADAPTIVE_CHILD_OFFSET
    torch.testing.assert_close(
        child_positions,
        torch.tensor(
            ((1.0 - offset, 2.0, 3.0), (1.0 + offset, 2.0, 3.0)),
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        child_scales,
        torch.tensor(((1.6, 3.2, 0.8), (1.6, 3.2, 0.8))),
    )
    torch.testing.assert_close(
        child_rotations, half_turn_z[None, :].expand(2, -1)
    )


def test_native_split_uses_first_axis_for_equal_maximum_scales() -> None:
    """The native largest-axis tie follows first-index argmax semantics."""
    child_positions, _, _ = visibility_adaptive_split_children(
        positions=torch.zeros((1, 3)),
        physical_scales=torch.tensor(((2.0, 2.0, 1.0),)),
        rotations=torch.tensor(((1.0, 0.0, 0.0, 0.0),)),
    )

    assert child_positions[0, 0] > 0.0
    assert child_positions[1, 0] < 0.0
    torch.testing.assert_close(child_positions[:, 1:], torch.zeros((2, 2)))


def test_native_densify_selector_uses_strict_recovered_thresholds() -> None:
    """Equality at any recovered threshold must reject a candidate."""
    mask = visibility_adaptive_densify_mask(
        status_grad=torch.tensor((0.0002, 0.00021, 0.00021, 0.00021)),
        status_size=torch.tensor((9.0, 8.0, 8.1, 8.1)),
        accumulated_weight=torch.tensor((0.0, 0.0, 150.0, 149.9)),
    )

    torch.testing.assert_close(mask, torch.tensor((False, False, False, True)))


def test_native_compact_weights_match_projected_pixel_recurrence() -> None:
    """Visibility fusion uses relative consistency around rendered values."""
    weights = visibility_adaptive_compact_point_weights(
        numerator=torch.tensor((0.4, 0.2, 0.0)),
        transmittance=torch.tensor((0.5, 0.5, 0.0)),
        reference=torch.tensor((0.8, 0.2, 1.0)),
        multiplier=torch.tensor((2.0, 3.0, 4.0)),
        sigma=0.5,
    )

    torch.testing.assert_close(
        weights,
        torch.tensor(
            (2.0, 3.0 * np.exp(-0.5), 0.0),
            dtype=torch.float32,
        ),
        rtol=2e-6,
        atol=2e-6,
    )


def test_native_compact_weights_reject_invalid_inputs() -> None:
    """Both derived and reference values must clear the native strict gate."""
    weights = visibility_adaptive_compact_point_weights(
        numerator=torch.tensor((1.0e-6, 1.0)),
        transmittance=torch.zeros(2),
        reference=torch.tensor((1.0, 1.0e-6)),
        multiplier=torch.ones(2),
        sigma=1.0,
    )

    torch.testing.assert_close(weights, torch.zeros(2))
    with np.testing.assert_raises_regex(ValueError, "sigma"):
        visibility_adaptive_compact_point_weights(
            numerator=torch.ones(1),
            transmittance=torch.zeros(1),
            reference=torch.ones(1),
            multiplier=torch.ones(1),
            sigma=0.0,
        )


def test_native_topology_schedule_matches_host_warmup_and_boundaries() -> None:
    """The host enables Step intervals after its effective image batch."""
    iteration_per_batch = 3_192

    assert visibility_adaptive_topology_schedule(
        step=0,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (False, False, False)
    assert visibility_adaptive_topology_schedule(
        step=100,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (False, False, False)
    assert visibility_adaptive_topology_schedule(
        step=3_190,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (False, False, False)
    assert visibility_adaptive_topology_schedule(
        step=3_191,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (True, True, True)
    assert visibility_adaptive_topology_schedule(
        step=3_200,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (False, False, False)
    assert visibility_adaptive_topology_schedule(
        step=3_291,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (True, True, False)
    assert visibility_adaptive_topology_schedule(
        step=6_383,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (False, False, True)
    assert visibility_adaptive_topology_schedule(
        step=6_391,
        total_iterations=10_001,
        iteration_per_batch=iteration_per_batch,
    ) == (True, False, False)


def test_native_topology_defaults_to_the_input_image_count() -> None:
    """Native point-status windows close after one input-image pass."""
    strategy, model = _native_strategy()
    strategy.accumulated_weight[:, 0] = torch.tensor((0.0, 1.0))

    changed = strategy._post_optimizer_step(
        step=3_191,
        scene_extent=1.0,
        train_dataset=range(3_192),
    )

    assert changed
    assert model.num_gaussians == 1
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.zeros((1, 1)),
    )


def test_native_batch_prune_precedes_opacity_prune() -> None:
    """The batch pivot includes rows that opacity pruning will remove."""
    strategy, model = _native_strategy(
        strategy_overrides={
            "topology_total_iterations": 10_000,
            "topology_batch_size": 600,
            "densify": {"frequency": 0},
            "prune": {
                "frequency": 600,
                "density_threshold": 0.005,
            },
        }
    )
    _repeat_native_strategy_rows(
        strategy=strategy,
        model=model,
        repeat_count=5,
    )
    strategy.accumulated_weight = torch.arange(1.0, 11.0)[:, None]
    model.density.data[:, 0] = torch.tensor(
        (-10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )

    assert strategy._post_optimizer_step(
        step=599,
        scene_extent=1.0,
        train_dataset=range(1),
    )

    assert model.num_gaussians == 8
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.zeros((8, 1)),
    )


def test_native_scheduled_prune_requires_low_opacity_and_radius() -> None:
    """Scheduled pruning preserves faint rows that were recently large."""
    keep = visibility_adaptive_regular_prune_keep_mask(
        physical_opacity=torch.tensor((0.0049, 0.0049, 0.0050, 0.0051)),
        status_radius=torch.tensor((0.0099, 0.0100, 0.0099, 0.0100)),
        opacity_threshold=0.005,
        radius_threshold=VISIBILITY_ADAPTIVE_REGULAR_PRUNE_RADIUS_THRESHOLD,
    )

    torch.testing.assert_close(
        keep,
        torch.tensor((False, True, True, True)),
    )


def test_native_scheduled_prune_compacts_only_small_faint_rows() -> None:
    """Scheduled pruning compacts all aligned state after the radius gate."""
    strategy, model = _native_strategy()
    model.density.data[:, 0] = torch.tensor((-10.0, -10.0))
    strategy.point_status[:, 2] = torch.tensor((0.009, 0.010))
    strategy.accumulated_weight[:, 0] = torch.tensor((11.0, 22.0))

    assert strategy._scheduled_prune()

    assert model.num_gaussians == 1
    torch.testing.assert_close(
        strategy.point_status[:, 2], torch.tensor((0.010,)))
    torch.testing.assert_close(
        strategy.accumulated_weight, torch.tensor(((22.0,),)))


def test_native_final_prune_ignores_scheduled_radius_gate() -> None:
    """Final cleanup retains its native density-only behavior."""
    strategy, model = _native_strategy()
    model.density.data[:, 0] = torch.tensor((-10.0, 0.0))
    strategy.point_status[:, 2] = torch.tensor((0.020, 0.002))

    assert strategy._prune()

    assert model.num_gaussians == 1


def test_native_coherence_uses_block_local_zyx_morton_order() -> None:
    """Coherence sorts Morton keys without crossing 1024-row blocks."""
    first_block = torch.zeros((1024, 3))
    first_block[:3] = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    )
    first_block[3:, 0] = 2.0
    positions = torch.cat((first_block, torch.zeros((1, 3))))

    permutation = visibility_adaptive_coherence_permutation(
        positions=positions,
        scale=1.0,
    )

    torch.testing.assert_close(permutation[:3], torch.tensor((1, 2, 0)))
    assert permutation[-1] == 1024


def test_native_partial_coherence_preserves_first_512_rows() -> None:
    """The first pass offsets by 512 rows then sorts 1024-row blocks."""
    positions = torch.zeros((1537, 3))
    positions[512:] = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
        + ((2.0, 0.0, 0.0),) * 1022
    )

    permutation = visibility_adaptive_coherence_permutation(
        positions=positions,
        scale=1.0,
        include_prefix=False,
    )

    torch.testing.assert_close(
        permutation[:VISIBILITY_ADAPTIVE_COHERENCE_PARTIAL_PREFIX],
        torch.arange(VISIBILITY_ADAPTIVE_COHERENCE_PARTIAL_PREFIX),
    )
    torch.testing.assert_close(
        permutation[512:515], torch.tensor((513, 514, 512))
    )
    assert permutation[-1] == 1536


def test_native_coherence_permutes_parameters_and_state_together() -> None:
    """Every Gaussian row follows the recovered Morton permutation."""
    strategy, model = _native_strategy()
    model.positions.data[:] = torch.tensor(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    model.features_albedo.data[:, 0] = torch.tensor((10.0, 20.0))
    model.environment_mask[:] = torch.tensor((False, True))
    strategy.point_status[:, 0] = torch.tensor((30.0, 40.0))
    strategy.accumulated_weight[:, 0] = torch.tensor((50.0, 60.0))
    position_state = model.optimizer.state[model.positions]
    position_state["gaussian_steps"][:] = torch.tensor((70, 80))

    assert not strategy._coherence()
    assert strategy.coherence_pass_count == 1
    assert strategy._coherence()
    assert strategy.coherence_pass_count == 2

    torch.testing.assert_close(
        model.positions,
        torch.tensor(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))),
    )
    torch.testing.assert_close(
        model.features_albedo[:, 0], torch.tensor((20.0, 10.0))
    )
    torch.testing.assert_close(
        strategy.point_status[:, 0], torch.tensor((40.0, 30.0))
    )
    torch.testing.assert_close(
        strategy.accumulated_weight[:, 0], torch.tensor((60.0, 50.0))
    )
    torch.testing.assert_close(
        model.environment_mask,
        torch.tensor((True, False)),
    )
    position_state = model.optimizer.state[model.positions]
    torch.testing.assert_close(
        position_state["gaussian_steps"],
        torch.tensor((80, 70), dtype=torch.int32),
    )


def test_native_short_run_keeps_prune_but_disables_densify() -> None:
    """The host final window disables only native Step densification."""
    assert visibility_adaptive_topology_schedule(
        step=3_191,
        total_iterations=6_500,
        iteration_per_batch=3_192,
    ) == (True, False, True)
    assert visibility_adaptive_topology_schedule(
        step=3_291,
        total_iterations=6_500,
        iteration_per_batch=3_192,
    ) == (True, False, False)


def test_native_topology_keeps_global_interval_phase_in_final_window() -> None:
    """Regular pruning stays phase-anchored after the final batch prune."""
    assert visibility_adaptive_topology_schedule(
        step=76_691,
        total_iterations=79_800,
        iteration_per_batch=3_192,
    ) == (True, False, False)
    assert visibility_adaptive_topology_schedule(
        step=76_700,
        total_iterations=79_800,
        iteration_per_batch=3_192,
    ) == (False, False, False)


def test_partial_replay_can_use_native_topology_horizon() -> None:
    """Truncated diagnostics can still replay native batch-window pruning."""
    dataset = range(3_192)
    strategy, model = _native_strategy()
    strategy.conf.n_iterations = 3_202
    strategy.accumulated_weight[:, 0] = torch.tensor((60.0, 40.0))

    assert not strategy._post_optimizer_step(
        step=3_191,
        scene_extent=1.0,
        train_dataset=dataset,
    )
    assert model.num_gaussians == 2

    strategy, model = _native_strategy(
        strategy_overrides={"topology_total_iterations": 6_500}
    )
    strategy.conf.n_iterations = 3_202
    strategy.accumulated_weight[:, 0] = torch.tensor((60.0, 40.0))
    model.density.data[:, 0] = torch.tensor((10.0, -10.0))

    assert strategy._post_optimizer_step(
        step=3_191,
        scene_extent=1.0,
        train_dataset=dataset,
    )
    assert model.num_gaussians == 1
    assert model.get_density().item() >= 0.005
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.zeros((1, 1)),
    )


def test_native_zero_topology_intervals_disable_regular_events() -> None:
    """Zero native intervals disable regular prune and densify calls."""
    assert visibility_adaptive_topology_schedule(
        step=3_200,
        total_iterations=10_001,
        iteration_per_batch=3_192,
        prune_interval=0,
        densify_interval=0,
    ) == (False, False, False)
    assert visibility_adaptive_topology_schedule(
        step=3_191,
        total_iterations=10_001,
        iteration_per_batch=3_192,
        prune_interval=0,
        densify_interval=0,
    ) == (False, False, True)


def test_native_split_opacity_matches_vendor_cubin() -> None:
    """The recovered volume split matches direct SM89 kernel execution."""
    physical_opacity = torch.sigmoid(
        torch.tensor(((-1.0986123,), (0.0,), (2.0,)))
    )

    children = visibility_adaptive_split_opacity(physical_opacity=physical_opacity)

    expected_logits = torch.tensor(
        (-2.349962, -2.349962, -1.496959, -1.496959, -0.6107988, -0.6107988)
    )
    torch.testing.assert_close(
        torch.logit(children.squeeze(-1)),
        expected_logits,
        rtol=2e-5,
        atol=2e-5,
    )


def test_native_split_opacity_rejects_nonphysical_values() -> None:
    """Opacity outside the activated range cannot enter the native split."""
    with np.testing.assert_raises_regex(ValueError, "in \\[0, 1\\]"):
        visibility_adaptive_split_opacity(physical_opacity=torch.tensor(((1.01,),)))


def test_native_point_status_tracks_decayed_maxima_and_rolling_gradient() -> (
    None
):
    """Visible points follow the four-field recurrence recovered from SASS."""
    status = torch.tensor(
        (
            (0.0, 10.0, 0.5, 0.0),
            (4.0, 10.0, 0.5, 2.0),
            (7.0, 3.0, 0.25, 1.0),
        )
    )

    updated = visibility_adaptive_update_point_status(
        status=status,
        point_indices=torch.tensor((0, 1)),
        projected_size_pixels=torch.tensor((8.0, 12.0, 100.0)),
        visible_measurements=torch.tensor(((0.6, 2.0), (0.4, 3.0))),
        window_size=4.0,
    )

    torch.testing.assert_close(
        updated,
        torch.tensor(
            (
                (1.0, 9.0, 0.6, 0.5),
                (4.0, 12.0, 0.49, 2.2),
                (7.0, 3.0, 0.25, 1.0),
            )
        ),
    )


def test_native_point_status_rejects_nonpositive_window() -> None:
    """A zero status window cannot represent the native recurrence."""
    with np.testing.assert_raises_regex(ValueError, "window size"):
        visibility_adaptive_update_point_status(
            status=torch.zeros((1, 4)),
            point_indices=torch.zeros((0,), dtype=torch.long),
            projected_size_pixels=torch.zeros((1,)),
            visible_measurements=torch.zeros((0, 2)),
            window_size=0.0,
        )


def test_native_projected_radius_matches_vendor_cubin() -> None:
    """Conic eigenvalues and opacity recover the native normalized radius."""
    packets = torch.tensor(
        (
            (1.0, 0.0, 1.0, 1.0),
            (4.0, 0.0, 4.0, 1.0),
            (1.0, 0.0, 1.0, 0.5),
            (4.0, 0.0, 1.0, 1.0),
            (2.0, 0.5, 1.0, 1.0),
        )
    )

    radii = visibility_adaptive_normalized_radius(
        projected_conic_opacity=packets,
        image_width=1_000,
        image_height=500,
    )

    torch.testing.assert_close(
        radii,
        torch.tensor(
            (
                0.00332904304,
                0.00166452152,
                0.00311387773,
                0.00332904304,
                0.00373862637,
            )
        ),
        rtol=2e-6,
        atol=1e-8,
    )


def test_native_projected_radius_rejects_invalid_vendor_packets() -> None:
    """The native kernel records zero for invalid opacity or conic packets."""
    packets = torch.tensor(
        (
            (1.0, 0.0, 1.0, 1.0 / 255.0),
            (1.0, 0.0, 1.0, 0.5 / 255.0),
            (1.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0, 1.0),
            (-1.0, 0.0, 1.0, 1.0),
        )
    )

    radii = visibility_adaptive_normalized_radius(
        projected_conic_opacity=packets,
        image_width=1_000,
        image_height=500,
    )

    torch.testing.assert_close(radii, torch.zeros(5))


def test_native_projected_radius_uses_larger_viewport_dimension() -> None:
    """Portrait and landscape viewports share the native max-side scale."""
    packet = torch.tensor(((1.0, 0.0, 1.0, 1.0),))

    landscape = visibility_adaptive_normalized_radius(
        projected_conic_opacity=packet,
        image_width=1_000,
        image_height=500,
    )
    portrait = visibility_adaptive_normalized_radius(
        projected_conic_opacity=packet,
        image_width=500,
        image_height=1_000,
    )

    torch.testing.assert_close(landscape, portrait)


def test_native_projected_size_defaults_to_rendered_weight() -> None:
    """Topology state defaults to the renderer's accumulated weight."""
    projected_size = visibility_adaptive_projected_size_pixels(
        rendered_weight=torch.tensor(((2.0,),)),
        projected_extent=torch.tensor(((4.0, 3.0),)),
        projected_conic_opacity=torch.tensor(((1.0, 0.0, 1.0, 1.0),)),
        image_width=10,
        image_height=10,
    )

    torch.testing.assert_close(
        projected_size,
        torch.tensor((2.0,)),
    )


def test_native_projected_size_can_use_rendered_weight() -> None:
    """Raw renderer contribution remains available as an explicit source."""
    projected_size = visibility_adaptive_projected_size_from_weight(
        rendered_weight=torch.tensor(
            (
                (2.0,),
                (-1.0,),
                (float("nan"),),
            )
        ),
    )

    torch.testing.assert_close(
        projected_size,
        torch.tensor((2.0, 0.0, 0.0)),
    )

    selected_size = visibility_adaptive_projected_size_pixels(
        rendered_weight=torch.tensor(((2.0,), (-1.0,), (float("nan"),))),
        projected_extent=torch.zeros((3, 2)),
        projected_conic_opacity=torch.zeros((3, 4)),
        image_width=10,
        image_height=10,
        source=VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT,
    )
    torch.testing.assert_close(selected_size, projected_size)


def test_native_projected_size_multiplier_scales_selected_source() -> None:
    """Projected-size scale is an explicit topology gate ablation knob."""
    projected_size = visibility_adaptive_projected_size_pixels(
        rendered_weight=torch.tensor(((2.0,), (-1.0,))),
        projected_extent=torch.tensor(((4.0, 3.0), (6.0, 1.0))),
        projected_conic_opacity=torch.zeros((2, 4)),
        image_width=10,
        image_height=10,
        source=VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT,
        multiplier=0.5,
    )

    torch.testing.assert_close(projected_size, torch.tensor((1.0, 0.0)))


@pytest.mark.parametrize("multiplier", (0.0, -1.0, float("nan")))
def test_native_projected_size_rejects_invalid_multiplier(
    multiplier: float,
) -> None:
    """Malformed projected-size scales fail instead of changing topology."""
    with pytest.raises(ValueError, match="projected-size multiplier"):
        visibility_adaptive_projected_size_pixels(
            rendered_weight=torch.tensor(((2.0,),)),
            projected_extent=torch.zeros((1, 2)),
            projected_conic_opacity=torch.zeros((1, 4)),
            image_width=10,
            image_height=10,
            multiplier=multiplier,
        )


def test_native_projected_size_rejects_invalid_shape() -> None:
    """Projected-size parity requires a renderer row per Gaussian."""
    with np.testing.assert_raises_regex(ValueError, "rendered weight"):
        visibility_adaptive_projected_size_from_weight(
            rendered_weight=torch.zeros((2, 3)),
        )


def test_native_projected_gradient_pixels_uses_renderer_gradient() -> None:
    """Native projected gradients are normalized coordinates scaled to pixels."""
    projected_gradient = visibility_adaptive_projected_gradient_pixels(
        projected_position_gradient=torch.tensor(((0.002, 0.004),)),
        image_width=10,
        image_height=10,
        position_gradient=torch.tensor(((3.0, 4.0, 0.0),)),
    )

    torch.testing.assert_close(
        projected_gradient,
        torch.sqrt(torch.tensor((0.01**2 + 0.02**2,))),
    )


def test_native_projected_gradient_pixels_uses_non_square_pixel_scaling(
) -> None:
    """Renderer gradients retain independent horizontal and vertical scales."""
    projected_gradient = visibility_adaptive_projected_gradient_pixels(
        projected_position_gradient=torch.tensor(((0.2, 0.2),)),
        image_width=20,
        image_height=10,
    )

    torch.testing.assert_close(
        projected_gradient,
        torch.sqrt(torch.tensor((5.0,))),
    )


def test_native_projected_gradient_pixels_preserves_mixed_renderer_rows(
) -> None:
    """A live renderer buffer retains rows whose native gradient is zero."""
    projected_gradient = visibility_adaptive_projected_gradient_pixels(
        projected_position_gradient=torch.tensor(((0.2, 0.0), (0.0, 0.0))),
        image_width=10,
        image_height=10,
        position_gradient=torch.tensor(((3.0, 4.0, 0.0), (6.0, 8.0, 0.0))),
        positions=torch.tensor(((0.0, 0.0, 2.0), (0.0, 0.0, 2.0))),
        camera_to_world=torch.eye(4),
        focal_length=torch.tensor((20.0, 20.0)),
    )

    torch.testing.assert_close(projected_gradient, torch.tensor((1.0, 0.0)))


def test_native_projected_gradient_pixels_defaults_to_jacobian() -> None:
    """A fully dead renderer buffer reconstructs native pixel gradients."""
    projected_gradient = visibility_adaptive_projected_gradient_pixels(
        projected_position_gradient=torch.zeros((1, 2)),
        image_width=10,
        image_height=10,
        position_gradient=torch.tensor(((5.0, 0.0, 0.0),)),
        positions=torch.tensor(((0.0, 0.0, 2.0),)),
        camera_to_world=torch.eye(4),
        focal_length=torch.tensor((20.0, 20.0)),
    )

    torch.testing.assert_close(projected_gradient, torch.tensor((2.5,)))


def test_native_projected_gradient_pixels_falls_back_without_focal_length(
) -> None:
    """Non-pinhole batches retain the raw world-gradient fallback."""
    projected_gradient = visibility_adaptive_projected_gradient_pixels(
        projected_position_gradient=torch.zeros((1, 2)),
        image_width=10,
        image_height=10,
        position_gradient=torch.tensor(((3.0, 4.0, 0.0),)),
        positions=torch.tensor(((0.0, 0.0, 2.0),)),
        camera_to_world=torch.eye(4),
    )

    torch.testing.assert_close(projected_gradient, torch.tensor((5.0,)))


def test_native_tangent_projected_gradient_matches_measured_proxy() -> None:
    """Native paired dumps support distance-scaled tangent world gradients."""
    projected_gradient = visibility_adaptive_tangent_projected_gradient_pixels(
        positions=torch.tensor(((0.0, 0.0, 2.0), (3.0, 0.0, 4.0))),
        position_gradient=torch.tensor(((5.0, 0.0, 1.0), (1.0, 0.0, 0.0))),
        camera_to_world=torch.eye(4),
        scale=VISIBILITY_ADAPTIVE_TANGENT_PROJECTED_GRADIENT_SCALE,
    )

    torch.testing.assert_close(
        projected_gradient,
        torch.tensor((8.0, 3.2)),
    )


def test_native_projected_gradient_pixels_uses_jacobian_proxy() -> None:
    """Pinhole batches can reconstruct image-plane gradients from xyz grads."""
    projected_gradient = visibility_adaptive_projected_gradient_pixels(
        projected_position_gradient=torch.zeros((1, 2)),
        image_width=10,
        image_height=10,
        position_gradient=torch.tensor(((5.0, 0.0, 0.0),)),
        positions=torch.tensor(((0.0, 0.0, 2.0),)),
        camera_to_world=torch.eye(4),
        focal_length=torch.tensor((20.0, 20.0)),
        proxy_mode=VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN,
    )

    torch.testing.assert_close(projected_gradient, torch.tensor((2.5,)))


def test_native_jacobian_projected_gradient_rejects_bad_pose() -> None:
    """The camera-Jacobian proxy needs a single camera-to-world matrix."""
    with np.testing.assert_raises_regex(ValueError, "camera pose"):
        visibility_adaptive_jacobian_projected_gradient_pixels(
            positions=torch.zeros((1, 3)),
            position_gradient=torch.zeros((1, 3)),
            camera_to_world=torch.eye(3),
            focal_length=torch.ones(2),
            image_width=10,
            image_height=10,
        )


def test_native_jacobian_projected_gradient_zeros_degenerate_geometry(
) -> None:
    """Zero-depth and non-finite camera-space rows cannot affect topology."""
    projected_gradient = visibility_adaptive_jacobian_projected_gradient_pixels(
        positions=torch.tensor(((0.0, 0.0, 0.0), (float("nan"), 0.0, 2.0))),
        position_gradient=torch.tensor(((5.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
        camera_to_world=torch.eye(4),
        focal_length=torch.tensor((20.0, 20.0)),
        image_width=10,
        image_height=10,
    )

    torch.testing.assert_close(projected_gradient, torch.zeros(2))


def test_native_projected_gradient_pixels_requires_proxy_gradient() -> None:
    """A zero renderer gradient must not silently produce a dead status path."""
    with np.testing.assert_raises_regex(ValueError, "position gradients"):
        visibility_adaptive_projected_gradient_pixels(
            projected_position_gradient=torch.zeros((2, 2)),
            image_width=10,
            image_height=10,
        )


def test_native_gradient_preprocess_uses_fixed_scale_shrink() -> None:
    """Large status always applies the cubin's fixed five-percent shrink."""
    physical_scales = torch.tensor(
        ((4.0, 1.0, 1.0), (4.0, 1.0, 1.0), (2.0, 3.0, 1.0))
    )
    raw_scales = torch.nn.Parameter(torch.log(physical_scales))
    scale_gradient = torch.tensor(
        ((8.0, 0.0, 0.0), (8.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )

    visibility_adaptive_preprocess_gradients(
        raw_scales=raw_scales,
        raw_rotations=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 3),
        positions=torch.zeros((3, 3)),
        scale_gradient=scale_gradient,
        rotation_gradient=torch.zeros((3, 4)),
        point_status=torch.tensor(
            ((0.0, 0.0, 2.0, 0.0), (0.0, 0.0, 3.0, 0.0), (0.0, 0.0, 0.5, 0.0))
        ),
        point_indices=torch.arange(3, dtype=torch.int64),
        environment_mask=torch.zeros(3, dtype=torch.bool),
        camera_position=torch.zeros(3),
        image_gradient_scale=0.25,
        accumulation_weight_source=torch.zeros((3, 1)),
        accumulated_weight=torch.zeros((3, 1)),
        weight_accumulation_multiplier=1.0,
    )

    torch.testing.assert_close(
        torch.exp(raw_scales),
        torch.tensor(((3.8, 1.0, 1.0), (3.8, 1.0, 1.0), (2.0, 3.0, 1.0))),
    )
    torch.testing.assert_close(
        scale_gradient,
        torch.tensor(((8.475, 0.0, 0.0), (8.725, 0.0, 0.0), (0.0, 0.1, 0.0))),
        rtol=2e-6,
        atol=2e-6,
    )


def test_native_gradient_preprocess_accumulates_weight_for_visible_rows() -> (
    None
):
    """Native accumulation uses the launch's projected-size source argument."""
    accumulated_weight = torch.tensor(((10.0,), (20.0,), (30.0,)))

    visibility_adaptive_preprocess_gradients(
        raw_scales=torch.zeros((3, 3)),
        raw_rotations=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 3),
        positions=torch.zeros((3, 3)),
        scale_gradient=torch.zeros((3, 3)),
        rotation_gradient=torch.zeros((3, 4)),
        point_status=torch.zeros((3, 4)),
        point_indices=torch.tensor((0, 2), dtype=torch.int64),
        environment_mask=torch.tensor((False, False, True)),
        camera_position=torch.tensor((0.0, 0.0, -1.0)),
        image_gradient_scale=0.0,
        accumulation_weight_source=torch.tensor(((2.0,), (5.0,), (7.0,))),
        accumulated_weight=accumulated_weight,
        weight_accumulation_multiplier=0.5,
    )

    torch.testing.assert_close(
        accumulated_weight,
        torch.tensor(((11.0,), (20.0,), (33.5,))),
    )


def test_native_environment_gradient_uses_raw_quaternion_and_new_scale() -> (
    None
):
    """Environment rows bypass status and backprop through clamped scale."""
    raw_scales = torch.nn.Parameter(
        torch.log(torch.tensor(((6.0, 4.0, 3.0),)))
    )
    parameter_identity = id(raw_scales)
    scale_gradient = torch.tensor(((6.0, 0.0, 0.0),))
    rotation_gradient = torch.zeros((1, 4))

    visibility_adaptive_preprocess_gradients(
        raw_scales=raw_scales,
        raw_rotations=torch.tensor(((2.0, 0.3, -0.4, 0.5),)),
        positions=torch.tensor(((0.0, 0.0, 10.0),)),
        scale_gradient=scale_gradient,
        rotation_gradient=rotation_gradient,
        point_status=torch.tensor(((0.0, 0.0, 100.0, 0.0),)),
        point_indices=torch.tensor((0,), dtype=torch.int64),
        environment_mask=torch.ones(1, dtype=torch.bool),
        camera_position=torch.zeros(3),
        image_gradient_scale=100.0,
        accumulation_weight_source=torch.zeros((1, 1)),
        accumulated_weight=torch.zeros((1, 1)),
        weight_accumulation_multiplier=1.0,
    )

    assert id(raw_scales) == parameter_identity
    torch.testing.assert_close(
        torch.exp(raw_scales), torch.tensor(((5.0, 4.0, 3.0),))
    )
    torch.testing.assert_close(
        scale_gradient,
        torch.tensor(((6.0, 4.0e-9, 5.6666667e-9),)),
        rtol=2e-6,
        atol=2e-12,
    )
    torch.testing.assert_close(
        rotation_gradient,
        torch.tensor(((2.0e-7, 5.7e-7, -7.6e-7, 0.5e-7),)),
        rtol=2e-6,
        atol=2e-12,
    )


def test_native_environment_scale_clamp_is_strict() -> None:
    """A radial scale equal to half the range is not rewritten."""
    raw_scales = torch.log(torch.tensor(((5.0, 5.0, 1.0), (5.001, 4.0, 1.0))))
    original_first = raw_scales[0].clone()

    visibility_adaptive_preprocess_gradients(
        raw_scales=raw_scales,
        raw_rotations=torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2),
        positions=torch.tensor(((0.0, 0.0, 10.0),) * 2),
        scale_gradient=torch.zeros((2, 3)),
        rotation_gradient=torch.zeros((2, 4)),
        point_status=torch.zeros((2, 4)),
        point_indices=torch.arange(2, dtype=torch.int64),
        environment_mask=torch.ones(2, dtype=torch.bool),
        camera_position=torch.zeros(3),
        image_gradient_scale=0.0,
        accumulation_weight_source=torch.zeros((2, 1)),
        accumulated_weight=torch.zeros((2, 1)),
        weight_accumulation_multiplier=1.0,
    )

    assert torch.equal(raw_scales[0], original_first)
    torch.testing.assert_close(
        torch.exp(raw_scales[1]), torch.tensor((5.0, 4.0, 1.0))
    )


def test_native_post_backward_updates_status_then_preprocesses_gradients() -> (
    None
):
    """The strategy reproduces the status, gradient, and weight launch order."""
    strategy, model = _native_strategy()
    parameter_identity = id(model.scale)
    with torch.no_grad():
        model.positions.copy_(
            torch.tensor(((10.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
        )
    model.positions.grad = torch.tensor(
        ((100.0, 0.0, 0.0), (100.0, 0.0, 0.0))
    )
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.full_like(model.density, 7.0)
    scale_gradient_identity = id(model.scale.grad)
    density_gradient = model.density.grad.clone()
    projected_conic_opacity = torch.tensor(
        (
            (0.01, 0.0, 0.01, 1.0),
            (1.0, 0.0, 1.0, 1.0),
        )
    )
    outputs = {
        "mog_visibility": torch.tensor((True, False)),
        "mog_projected_conic_opacity": projected_conic_opacity,
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (0.0, 0.0))),
        "mog_projected_position": torch.tensor(((1.0, 1.0), (9.0, 9.0))),
        "mog_projected_position_gradient": torch.tensor(
            ((0.002, 0.004), (0.0, 0.0))
        ),
        "mog_accumulated_weight": torch.tensor((4.0, 9.0)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        mask=torch.ones((1, 10, 10, 1)),
        post_processing_camera_idx=2,
        native_image_scale=0.8,
    )

    changed = strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    radius = visibility_adaptive_normalized_radius(
        projected_conic_opacity=projected_conic_opacity,
        image_width=10,
        image_height=10,
    )[0]
    image_gradient_scale = 2.0 / (300.0 * 0.8)
    expected_scale_gradient = (radius - 0.1) * image_gradient_scale
    assert changed is False
    assert id(model.scale) == parameter_identity
    assert id(model.scale.grad) == scale_gradient_identity
    assert model.optimizer.param_groups[5]["params"][0] is model.scale
    torch.testing.assert_close(
        strategy.point_status[0, 1],
        torch.tensor(4.0),
    )
    torch.testing.assert_close(strategy.point_status[0, 2], radius)
    torch.testing.assert_close(
        strategy.point_status[0, 3],
        torch.sqrt(torch.tensor(0.01**2 + 0.02**2)) / 32.0,
    )
    torch.testing.assert_close(
        torch.exp(model.scale[0]),
        torch.tensor((0.95, 1.0, 1.0)),
    )
    torch.testing.assert_close(
        model.scale.grad[0],
        torch.stack(
            (
                expected_scale_gradient,
                torch.zeros_like(expected_scale_gradient),
                torch.zeros_like(expected_scale_gradient),
            )
        ),
    )
    torch.testing.assert_close(model.density.grad, density_gradient)
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.tensor(((4.0,), (0.0,))),
    )


def test_native_post_backward_uses_ray_gradient_when_projection_is_zero() -> (
    None
):
    """The default zero-projection fallback uses raw position-grad norm."""
    strategy, model = _native_strategy()
    with torch.no_grad():
        model.positions.copy_(
            torch.tensor(((0.0, 0.0, 2.0), (0.0, 0.0, 4.0)))
        )
    model.positions.grad = torch.tensor(
        ((3.0, 4.0, 0.0), (0.0, 0.0, 0.0))
    )
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.zeros_like(model.density)
    outputs = {
        "mog_visibility": torch.tensor((True, False)),
        "mog_projected_conic_opacity": torch.tensor(
            (
                (0.01, 0.0, 0.01, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            )
        ),
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (0.0, 0.0))),
        "mog_projected_position": torch.tensor(((1.0, 1.0), (9.0, 9.0))),
        "mog_projected_position_gradient": torch.zeros((2, 2)),
        "mog_accumulated_weight": torch.tensor((4.0, 9.0)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        mask=torch.ones((1, 10, 10, 1)),
        post_processing_camera_idx=0,
        native_image_scale=1.0,
    )

    strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    torch.testing.assert_close(
        strategy.point_status[0, 3],
        torch.tensor(5.0 / 32.0),
    )


@pytest.mark.parametrize(
    ("status_gradient_scale", "expected_status"),
    ((1.0, 5.0 / 32.0), (4.0, 20.0 / 32.0)),
)
def test_native_post_backward_scales_renderer_pixel_gradient_for_status(
    status_gradient_scale: float,
    expected_status: float,
) -> None:
    """Renderer pixel gradients update only the configured status signal."""
    strategy, model = _native_strategy(
        strategy_overrides={"status_gradient_scale": status_gradient_scale}
    )
    model.positions.grad = torch.zeros_like(model.positions)
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.zeros_like(model.density)
    outputs = {
        "mog_visibility": torch.tensor((True, False)),
        "mog_projected_conic_opacity": torch.tensor(
            (
                (0.01, 0.0, 0.01, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            )
        ),
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (0.0, 0.0))),
        "mog_projected_position": torch.tensor(((1.0, 1.0), (9.0, 9.0))),
        "mog_projected_position_gradient": torch.zeros((2, 2)),
        "mog_projected_gradient_pixels": torch.tensor((5.0, 0.0)),
        "mog_accumulated_weight": torch.tensor((4.0, 9.0)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        mask=torch.ones((1, 10, 10, 1)),
        post_processing_camera_idx=0,
        native_image_scale=1.0,
    )

    strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    torch.testing.assert_close(
        strategy.point_status[0, 3],
        torch.tensor(expected_status),
    )
    torch.testing.assert_close(model.positions.grad, torch.zeros_like(model.positions))


def test_native_post_backward_can_use_jacobian_projected_gradient() -> None:
    """Pinhole proxy remains available as an explicit diagnostic mode."""
    strategy, model = _native_strategy(
        strategy_overrides={
            "projected_gradient_proxy_mode": (
                VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN
            ),
        }
    )
    with torch.no_grad():
        model.positions.copy_(
            torch.tensor(((0.0, 0.0, 2.0), (0.0, 0.0, 4.0)))
        )
    model.positions.grad = torch.tensor(
        ((5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.zeros_like(model.density)
    outputs = {
        "mog_visibility": torch.tensor((True, False)),
        "mog_projected_conic_opacity": torch.tensor(
            (
                (0.01, 0.0, 0.01, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            )
        ),
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (0.0, 0.0))),
        "mog_projected_position": torch.tensor(((1.0, 1.0), (9.0, 9.0))),
        "mog_projected_position_gradient": torch.zeros((2, 2)),
        "mog_accumulated_weight": torch.tensor((4.0, 9.0)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        intrinsics_OpenCVPinholeCameraModelParameters={
            "focal_length": torch.tensor((20.0, 20.0)),
        },
        mask=torch.ones((1, 10, 10, 1)),
        post_processing_camera_idx=0,
        native_image_scale=1.0,
    )

    strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    torch.testing.assert_close(
        strategy.point_status[0, 3],
        torch.tensor(2.5 / 32.0),
    )


def test_native_post_backward_uses_tangent_projected_gradient() -> None:
    """The native paired-dump tangent proxy remains opt-in."""
    strategy, model = _native_strategy(
        strategy_overrides={
            "projected_gradient_proxy_mode": (
                VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_TANGENT
            ),
        }
    )
    with torch.no_grad():
        model.positions.copy_(
            torch.tensor(((0.0, 0.0, 2.0), (0.0, 0.0, 4.0)))
        )
    model.positions.grad = torch.tensor(
        ((5.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.zeros_like(model.density)
    outputs = {
        "mog_visibility": torch.tensor((True, False)),
        "mog_projected_conic_opacity": torch.tensor(
            (
                (0.01, 0.0, 0.01, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            )
        ),
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (0.0, 0.0))),
        "mog_projected_position": torch.tensor(((1.0, 1.0), (9.0, 9.0))),
        "mog_projected_position_gradient": torch.zeros((2, 2)),
        "mog_accumulated_weight": torch.tensor((4.0, 9.0)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        intrinsics_OpenCVPinholeCameraModelParameters={
            "focal_length": torch.tensor((20.0, 20.0)),
        },
        mask=torch.ones((1, 10, 10, 1)),
        post_processing_camera_idx=0,
        native_image_scale=1.0,
    )

    strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    torch.testing.assert_close(
        strategy.point_status[0, 3],
        torch.tensor(8.0 / 32.0),
    )


def test_native_post_backward_can_use_contribution_visibility() -> None:
    """Opt-in contribution visibility gates status and optimizer rows."""
    strategy, model = _native_strategy(
        strategy_overrides={
            "optimizer_visibility_source": "color_gradient_weight",
            "optimizer_visibility_min_rendered_weight": 0.015,
            "optimizer_visibility_min_color_gradient_norm": 0.0,
            "projected_size_source": (
                VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT
            ),
            "weight_accumulation_multiplier": 1.0,
        }
    )
    model.positions.grad = torch.zeros_like(model.positions)
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.zeros_like(model.density)
    model.features_albedo.grad = torch.tensor(
        (
            (0.1, 0.0, 0.0),
            (0.1, 0.0, 0.0),
        )
    )
    outputs = {
        "mog_visibility": torch.tensor((True, True)),
        "mog_projected_conic_opacity": torch.tensor(
            (
                (0.01, 0.0, 0.01, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            )
        ),
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (3.0, 2.0))),
        "mog_projected_position": torch.tensor(((1.0, 1.0), (9.0, 9.0))),
        "mog_projected_position_gradient": torch.zeros((2, 2)),
        "mog_accumulated_weight": torch.tensor((0.01, 0.02)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        mask=torch.ones((1, 10, 10, 1)),
        post_processing_camera_idx=0,
        native_image_scale=1.0,
    )

    strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    torch.testing.assert_close(
        outputs["mog_visibility"],
        torch.tensor((False, True)),
    )
    torch.testing.assert_close(strategy.point_status[0], torch.zeros(4))
    assert strategy.point_status[1, 0] == 1.0
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.tensor(((0.0,), (0.02,))),
    )


def test_native_post_backward_can_use_forward_visibility() -> None:
    """Opt-in forward visibility gates status and optimizer rows."""
    strategy, model = _native_strategy(
        strategy_overrides={
            "optimizer_visibility_source": (
                VISIBILITY_ADAPTIVE_FORWARD_VISIBILITY_SOURCE
            ),
            "projected_size_source": (
                VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT
            ),
            "weight_accumulation_multiplier": 1.0,
        }
    )
    model.positions.grad = torch.zeros_like(model.positions)
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.zeros_like(model.density)
    model.features_albedo.grad = torch.zeros_like(model.features_albedo)
    outputs = {
        "mog_visibility": torch.tensor((True, True)),
        "mog_forward_visibility": torch.tensor((False, True)),
        "mog_projected_conic_opacity": torch.tensor(
            (
                (0.01, 0.0, 0.01, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            )
        ),
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (3.0, 2.0))),
        "mog_projected_position": torch.tensor(((1.0, 1.0), (9.0, 9.0))),
        "mog_projected_position_gradient": torch.zeros((2, 2)),
        "mog_accumulated_weight": torch.tensor((0.01, 0.02)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        mask=torch.ones((1, 10, 10, 1)),
        post_processing_camera_idx=0,
        native_image_scale=1.0,
    )

    strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    torch.testing.assert_close(
        outputs["mog_visibility"],
        torch.tensor((False, True)),
    )
    torch.testing.assert_close(strategy.point_status[0], torch.zeros(4))
    assert strategy.point_status[1, 0] == 1.0
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.tensor(((0.0,), (0.02,))),
    )


def test_native_post_backward_can_use_mask_center_weight_visibility() -> None:
    """Opt-in mask-center visibility gates invalid-pixel optimizer rows."""
    strategy, model = _native_strategy(
        strategy_overrides={
            "optimizer_visibility_source": "mask_center_weight",
            "optimizer_visibility_min_rendered_weight": 0.015,
            "projected_size_source": (
                VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT
            ),
            "weight_accumulation_multiplier": 1.0,
        }
    )
    model.positions.grad = torch.zeros_like(model.positions)
    model.scale.grad = torch.zeros_like(model.scale)
    model.rotation.grad = torch.zeros_like(model.rotation)
    model.density.grad = torch.zeros_like(model.density)
    outputs = {
        "mog_visibility": torch.tensor((True, True)),
        "mog_projected_conic_opacity": torch.tensor(
            (
                (0.01, 0.0, 0.01, 1.0),
                (1.0, 0.0, 1.0, 1.0),
            )
        ),
        "mog_projected_extent": torch.tensor(((2.0, 1.0), (3.0, 2.0))),
        "mog_projected_position": torch.tensor(((2.0, 2.0), (7.0, 7.0))),
        "mog_projected_position_gradient": torch.zeros((2, 2)),
        "mog_accumulated_weight": torch.tensor((0.02, 0.02)),
        "pred_rgb": torch.zeros((1, 10, 10, 3)),
    }
    mask = torch.zeros((1, 10, 10, 1), dtype=torch.bool)
    mask[0, 2, 2, 0] = True
    batch = SimpleNamespace(
        T_to_world=torch.eye(4).unsqueeze(0),
        mask=mask,
        post_processing_camera_idx=0,
        native_image_scale=1.0,
    )

    strategy._post_backward(
        step=0,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )

    torch.testing.assert_close(
        outputs["mog_visibility"],
        torch.tensor((True, False)),
    )
    assert strategy.point_status[0, 0] == 1.0
    torch.testing.assert_close(strategy.point_status[1], torch.zeros(4))
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.tensor(((0.02,), (0.0,))),
    )


def test_native_strategy_split_keeps_all_parameter_and_state_rows_aligned() -> (
    None
):
    """A native split mutates its parent and appends two aligned children."""
    strategy, model = _native_strategy()
    model.environment_mask[:] = torch.tensor((True, False))
    strategy.point_status[0, 1] = 9.0
    strategy.point_status[0, 3] = 0.001
    strategy.accumulated_weight[:, 0] = torch.tensor((10.0, 200.0))
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        state = model.optimizer.state[parameter]
        state["exp_avg"].fill_(3.0)
        state["exp_avg_sq"].fill_(4.0)
        if group["name"] == "positions":
            state["gaussian_steps"][:] = torch.tensor((7, 9))

    assert strategy._densify()

    assert model.num_gaussians == 4
    assert model.features_specular.shape == (4, 3)
    assert strategy.point_status.shape == (4, 4)
    torch.testing.assert_close(
        model.positions,
        torch.tensor(
            (
                (0.0, 0.0, 0.0),
                (10.0, 0.0, 0.0),
                (VISIBILITY_ADAPTIVE_CHILD_OFFSET, 0.0, 0.0),
                (-VISIBILITY_ADAPTIVE_CHILD_OFFSET, 0.0, 0.0),
            )
        ),
    )
    torch.testing.assert_close(
        model.get_scale(),
        torch.tensor(
            (
                (0.8, 0.8, 0.8),
                (1.0, 1.0, 1.0),
                (0.8, 0.8, 0.8),
                (0.8, 0.8, 0.8),
            )
        ),
    )
    expected_opacity = visibility_adaptive_split_opacity(
        physical_opacity=torch.tensor(((0.5,),))
    )[0]
    torch.testing.assert_close(
        model.get_density()[torch.tensor((0, 2, 3))],
        expected_opacity.expand(3, 1),
    )
    torch.testing.assert_close(
        strategy.point_status,
        torch.zeros((4, 4)),
    )
    torch.testing.assert_close(
        strategy.accumulated_weight[:, 0],
        torch.tensor((10.0, 200.0, 10.0, 10.0)),
    )
    torch.testing.assert_close(
        model.environment_mask,
        torch.tensor((True, False, True, True)),
    )
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        state = model.optimizer.state[parameter]
        torch.testing.assert_close(
            state["exp_avg"][2:], torch.zeros_like(state["exp_avg"][2:])
        )
        torch.testing.assert_close(
            state["exp_avg_sq"][2:],
            torch.zeros_like(state["exp_avg_sq"][2:]),
        )
        if group["name"] == "positions":
            torch.testing.assert_close(
                state["gaussian_steps"],
                torch.tensor((7, 9, 7, 7), dtype=torch.int32),
            )


def test_native_strategy_skips_entire_split_over_capacity() -> None:
    """Native capacity checks never truncate the selected parent set."""
    strategy, model = _native_strategy()
    strategy.conf.strategy.max_gaussians = 5
    strategy.point_status[:, 1] = 9.0
    strategy.point_status[:, 3] = 0.001

    assert not strategy._densify()

    assert model.num_gaussians == 2
    torch.testing.assert_close(model.get_scale(), torch.ones((2, 3)))
    torch.testing.assert_close(
        strategy.point_status[:, 1], torch.full((2,), 9.0)
    )


def test_native_strategy_prune_compacts_every_state_row() -> None:
    """Strict native opacity pruning compacts model, moments, and buffers."""
    strategy, model = _native_strategy()
    model.density.data[:, 0] = torch.tensor((-10.0, 0.0))
    model.environment_mask[:] = torch.tensor((True, False))
    strategy.point_status[:, 0] = torch.tensor((11.0, 22.0))
    strategy.accumulated_weight[:, 0] = torch.tensor((33.0, 44.0))

    assert strategy._prune()

    assert model.num_gaussians == 1
    assert model.features_specular.shape == (1, 3)
    assert strategy.point_status[0, 0] == 22.0
    assert strategy.accumulated_weight[0, 0] == 44.0
    assert not model.environment_mask[0]
    for group in model.optimizer.param_groups:
        parameter = group["params"][0]
        state = model.optimizer.state[parameter]
        assert state["exp_avg"].shape[0] == 1
        assert state["exp_avg_sq"].shape[0] == 1
        if group["name"] == "positions":
            assert state["gaussian_steps"].shape == (1,)


def test_native_strategy_prune_compacts_unoptimized_sh0_tensor() -> None:
    """SH0's empty specular tensor must follow Gaussian topology changes."""
    strategy, model = _native_strategy(
        specular_width=0,
        optimize_specular=False,
    )
    model.density.data[:, 0] = torch.tensor((-10.0, 0.0))

    assert strategy._prune()

    assert model.num_gaussians == 1
    assert model.features_specular.shape == (1, 0)


def test_native_finalization_prunes_before_evaluation() -> None:
    """Finalization applies the native final opacity threshold."""
    strategy, model = _native_strategy()
    model.density.data[:, 0] = torch.tensor((-10.0, 0.0))

    assert strategy.finalize_training()

    assert model.num_gaussians == 1
    torch.testing.assert_close(model.get_density(), torch.tensor(((0.5,),)))


@pytest.mark.parametrize(
    ("weights", "expected_changed", "expected_retained_count"),
    (
        (
            (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0),
            True,
            8,
        ),
        (
            (1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
            True,
            6,
        ),
        (
            (51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0),
            False,
            10,
        ),
    ),
)
def test_native_iterative_prune_uses_quantile_capped_native_weight_pivot(
    weights: tuple[float, ...],
    expected_changed: bool,
    expected_retained_count: int,
) -> None:
    """Batch pruning retains rows strictly above the capped 10th-percentile pivot."""
    strategy, model = _native_strategy()
    _repeat_native_strategy_rows(
        strategy=strategy,
        model=model,
        repeat_count=5,
    )
    strategy.accumulated_weight = torch.tensor(weights)[:, None]

    assert strategy._iterative_prune() is expected_changed

    assert model.num_gaussians == expected_retained_count
    torch.testing.assert_close(
        strategy.accumulated_weight,
        torch.zeros((expected_retained_count, 1)),
    )


def test_native_iterative_prune_raises_when_all_rows_tie() -> None:
    """An all-tied batch must not silently discard every Gaussian."""
    strategy, model = _native_strategy()
    _repeat_native_strategy_rows(
        strategy=strategy,
        model=model,
        repeat_count=5,
    )
    original_weight = torch.ones((10, 1))
    strategy.accumulated_weight = original_weight.clone()

    with pytest.raises(RuntimeError, match="would remove every Gaussian"):
        strategy._iterative_prune()

    torch.testing.assert_close(strategy.accumulated_weight, original_weight)


def test_native_strategy_checkpoint_round_trip_preserves_status() -> None:
    """Native topology evidence survives the normal checkpoint contract."""
    strategy, model = _native_strategy()
    strategy.point_status[0] = torch.tensor((1.0, 2.0, 3.0, 4.0))
    strategy.accumulated_weight[0, 0] = 5.0
    assert not strategy._coherence()
    checkpoint = strategy.get_strategy_parameters()
    restored = VisibilityAdaptiveStrategy(strategy.conf, model)

    restored.init_densification_buffer(checkpoint)

    torch.testing.assert_close(restored.point_status, strategy.point_status)
    torch.testing.assert_close(
        restored.accumulated_weight, strategy.accumulated_weight
    )
    assert restored.coherence_pass_count == 1


def test_native_strategy_repeats_zero_width_sh0_rows() -> None:
    """SH0 has no specular columns but still needs two rows per child."""
    mask = torch.tensor((True, False))

    repeated = VisibilityAdaptiveStrategy._repeat_selected(torch.empty((2, 0)), mask)

    assert repeated.shape == (2, 0)
