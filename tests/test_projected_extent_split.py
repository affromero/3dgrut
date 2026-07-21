"""Behavior tests for opt-in 3DGUT projected-extent split routing."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.strategy.gs import GSStrategy
from threedgrut.trainer import Trainer3DGRUT


class _StrategyModel:
    """Small CPU Gaussian model with the parameter contract GSStrategy uses."""

    def __init__(self, scales: tuple[float, ...]) -> None:
        self.device = torch.device("cpu")
        self.protected_gaussian_count = 0
        count = len(scales)
        positions = torch.arange(1, count + 1, dtype=torch.float32)
        self.positions = torch.nn.Parameter(
            torch.stack(
                (positions, torch.zeros(count), torch.zeros(count)),
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
        groups = [
            {
                "params": [getattr(self, name)],
                "name": name,
                "lr": 0.001,
            }
            for name in parameter_names
        ]
        self.optimizer = torch.optim.Adam(groups)

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

    @staticmethod
    def refresh_protected_gradient_hooks() -> None:
        """The test model has no protected prefix hooks to refresh."""


def _strategy(
    *,
    enabled: bool = True,
    render_method: str = "3dgut",
    max_px: float = 8.0,
    clone_threshold: float = 0.2,
    split_threshold: float = 0.4,
    scales: tuple[float, ...] = (0.5, 0.5),
    print_stats: bool = False,
    theta_aware: bool = False,
    densify_start_iteration: int = 500,
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
                    "start_iteration": densify_start_iteration,
                    "end_iteration": 15_000,
                    "clone_grad_threshold": clone_threshold,
                    "split_grad_threshold": split_threshold,
                    "relative_size_threshold": 1.0,
                    "split": {"n_gaussians": 2},
                    "theta_aware": {"enabled": theta_aware},
                    "feature_grad": {"enabled": False},
                    "projected_extent_split": {
                        "enabled": enabled,
                        "max_px": max_px,
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
    extents: torch.Tensor,
    tile_counts: torch.Tensor,
) -> dict[str, object]:
    return {
        "mog_projected_extent": extents,
        "mog_tiles_count": tile_counts,
    }


def test_disabled_gate_preserves_world_decisions_and_requires_no_outputs() -> None:
    """The default-off path ignores extents and supports the 3DGRT renderer."""
    strategy, _ = _strategy(
        enabled=False,
        render_method="3dgrt",
        scales=(0.5, 2.0),
    )
    clone, split, projected_split = strategy._densify_candidate_masks(
        torch.tensor((0.3, 0.5)),
        1.0,
        None,
        None,
    )

    torch.testing.assert_close(clone, torch.tensor((True, False)))
    torch.testing.assert_close(split, torch.tensor((False, True)))
    torch.testing.assert_close(
        projected_split,
        torch.tensor((False, False)),
    )
    assert "densify_projected_extent_max" not in (strategy.get_strategy_parameters())
    assert not strategy.post_backward(
        step=15_001,
        scene_extent=1.0,
        train_dataset=(),
        outputs=None,
    )


def test_disabled_densification_requires_no_position_gradient() -> None:
    """A negative start disables gradient accumulation for frozen geometry."""
    strategy, model = _strategy(densify_start_iteration=-1)
    model.positions.requires_grad_(False)

    assert not strategy.post_backward(
        step=1,
        scene_extent=1.0,
        train_dataset=(),
        outputs=None,
    )


@pytest.mark.parametrize("render_method", ("3dgrt", "torch"))
def test_enabled_gate_requires_3dgut(render_method: str) -> None:
    """Only 3DGUT exposes the renderer buffers used by the gate."""
    with pytest.raises(ValueError, match="render.method=3dgut"):
        _strategy(enabled=True, render_method=render_method)


@pytest.mark.parametrize("max_px", (0.0, -1.0, float("inf"), float("nan")))
def test_enabled_gate_requires_finite_positive_threshold(
    max_px: float,
) -> None:
    """Invalid support thresholds fail before training starts."""
    with pytest.raises(ValueError, match="finite and positive"):
        _strategy(max_px=max_px)


def test_extent_accumulation_tracks_valid_visible_maxima_only() -> None:
    """Maxima ignore off-screen, nonfinite, and negative renderer rows."""
    strategy, _ = _strategy(scales=(0.5, 0.5, 0.5, 0.5))
    strategy.update_projected_extent_buffer(
        _renderer_outputs(
            torch.tensor(
                (
                    (3.0, 4.0),
                    (9.0, 1.0),
                    (float("nan"), 2.0),
                    (-5.0, 6.0),
                )
            ),
            torch.tensor((1, 0, 1, 1)),
        )
    )
    strategy.update_projected_extent_buffer(
        _renderer_outputs(
            torch.tensor(
                (
                    (8.0, 2.0),
                    (9.0, 1.0),
                    (7.0, 3.0),
                    (4.0, 4.0),
                )
            ),
            torch.tensor((1, 1, 1, 1)),
        )
    )

    torch.testing.assert_close(
        strategy.densify_projected_extent_max,
        torch.tensor((8.0, 9.0, 7.0, 4.0)),
    )


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ({}, "mog_projected_extent"),
        (
            {"mog_projected_extent": torch.zeros((2, 2))},
            "mog_tiles_count",
        ),
        (
            {
                "mog_projected_extent": torch.zeros((2, 3)),
                "mog_tiles_count": torch.ones(2),
            },
            "must have shape",
        ),
        (
            {
                "mog_projected_extent": torch.zeros((2, 2)),
                "mog_tiles_count": torch.ones(3),
            },
            "one value per Gaussian",
        ),
    ],
)
def test_extent_accumulation_rejects_missing_or_misaligned_outputs(
    outputs: dict[str, object],
    message: str,
) -> None:
    """Enabled routing fails loudly when renderer diagnostics are unavailable."""
    strategy, _ = _strategy()

    with pytest.raises((RuntimeError, ValueError), match=message):
        strategy.update_projected_extent_buffer(outputs)


def test_candidate_partition_reroutes_only_small_world_clone_candidates() -> None:
    """Extent routing is strict at 8 px and keeps world thresholds independent."""
    strategy, _ = _strategy(scales=(0.5, 0.5, 2.0, 0.5))
    clone, split, projected_split = strategy._densify_candidate_masks(
        torch.tensor((0.3, 0.3, 0.5, 0.1)),
        1.0,
        None,
        torch.tensor((8.0, 8.01, 100.0, 9.0)),
    )

    torch.testing.assert_close(
        clone,
        torch.tensor((True, False, False, False)),
    )
    torch.testing.assert_close(
        split,
        torch.tensor((False, True, True, False)),
    )
    torch.testing.assert_close(
        projected_split,
        torch.tensor((False, True, False, False)),
    )


def test_theta_aware_densify_stats_accepts_projected_extent_routing() -> None:
    """Existing theta-aware diagnostics remain compatible with extent stats."""
    strategy, _ = _strategy(print_stats=True, theta_aware=True)

    strategy.log_densify_stats(
        densify_grad_norm=torch.tensor((0.3, 0.5)),
        scene_extent=1.0,
        mean_cos=torch.tensor((0.25, 0.9)),
        projected_extent_max=torch.tensor((9.0, 4.0)),
    )


def test_clone_then_split_snapshot_adds_one_smaller_population_row() -> None:
    """A rerouted clone candidate survives clone's buffer reset and splits."""
    strategy, model = _strategy(
        clone_threshold=0.2,
        split_threshold=0.4,
    )
    strategy.densify_grad_norm_accum[:, 0] = torch.tensor((0.3, 0.0))
    strategy.densify_grad_norm_denom[:, 0] = 1
    strategy.densify_projected_extent_max[:] = torch.tensor((9.0, 0.0))
    torch.manual_seed(42)

    strategy.densify_gaussians(scene_extent=1.0)

    assert model.num_gaussians == 3
    for name in (
        "positions",
        "density",
        "features_albedo",
        "features_specular",
        "rotation",
        "scale",
    ):
        assert getattr(model, name).shape[0] == 3
    torch.testing.assert_close(
        model.get_scale()[:, 0],
        torch.tensor((0.5, 0.3125, 0.3125)),
    )
    torch.testing.assert_close(
        strategy.densify_projected_extent_max,
        torch.zeros(3),
    )


def test_topology_reset_and_prune_keep_extent_rows_aligned() -> None:
    """Split children reset the buffer and opacity pruning compacts its rows."""
    strategy, model = _strategy()
    strategy.densify_grad_norm_accum[:, 0] = torch.tensor((0.3, 0.0))
    strategy.densify_grad_norm_denom[:, 0] = 1
    strategy.densify_projected_extent_max[:] = torch.tensor((9.0, 0.0))
    torch.manual_seed(42)
    strategy.densify_gaussians(scene_extent=1.0)
    strategy.densify_projected_extent_max[:] = torch.tensor((1.0, 2.0, 3.0))
    model.density.data[:, 0] = torch.tensor((0.0, -10.0, 0.0))

    strategy.prune_gaussians_opacity()

    assert model.num_gaussians == 2
    torch.testing.assert_close(
        strategy.densify_projected_extent_max,
        torch.tensor((1.0, 3.0)),
    )
    assert strategy.densify_grad_norm_accum.shape == (2, 1)
    assert strategy.densify_grad_norm_denom.shape == (2, 1)


def test_checkpoint_restore_supports_new_old_and_permuted_buffers() -> None:
    """Resume restores extent rows, while old checkpoints start with zeros."""
    strategy, _ = _strategy()
    strategy.densify_projected_extent_max[:] = torch.tensor((3.0, 7.0))
    checkpoint = strategy.get_strategy_parameters()

    restored, restored_model = _strategy()
    restored_model.device = "cpu"
    restored.init_densification_buffer(checkpoint)
    torch.testing.assert_close(
        restored.densify_projected_extent_max,
        torch.tensor((3.0, 7.0)),
    )

    old_checkpoint = dict(checkpoint)
    old_checkpoint.pop("densify_projected_extent_max")
    old_restored, _ = _strategy()
    old_restored.init_densification_buffer(old_checkpoint)
    torch.testing.assert_close(
        old_restored.densify_projected_extent_max,
        torch.zeros(2),
    )

    permutation = torch.tensor((1, 0))
    permuted_checkpoint = {key: tuple(value[permutation] for value in values) for key, values in checkpoint.items()}
    permuted, _ = _strategy()
    permuted.init_densification_buffer(permuted_checkpoint)
    torch.testing.assert_close(
        permuted.densify_projected_extent_max,
        torch.tensor((7.0, 3.0)),
    )


def test_checkpoint_restore_rejects_extent_row_mismatch() -> None:
    """A malformed resume cannot silently desynchronize topology state."""
    strategy, _ = _strategy()
    checkpoint = strategy.get_strategy_parameters()
    checkpoint["densify_projected_extent_max"] = (torch.zeros(3),)

    with pytest.raises(ValueError, match="must have shape"):
        strategy.init_densification_buffer(checkpoint)


def test_base_callback_forwards_renderer_outputs_to_extent_accumulation() -> None:
    """The public strategy callback delivers raw renderer diagnostics."""
    strategy, model = _strategy()
    model.positions.grad = torch.ones_like(model.positions)
    batch = SimpleNamespace(T_to_world=torch.eye(4).unsqueeze(0))
    outputs = _renderer_outputs(
        torch.tensor(((4.0, 9.0), (12.0, 3.0))),
        torch.tensor((1, 0)),
    )

    assert not strategy.post_backward(
        step=1,
        scene_extent=1.0,
        train_dataset=(),
        batch=batch,
        outputs=outputs,
    )
    torch.testing.assert_close(
        strategy.densify_projected_extent_max,
        torch.tensor((9.0, 0.0)),
    )


def test_trainer_forwards_raw_outputs_to_post_backward_strategy() -> None:
    """The training boundary preserves the renderer diagnostic dictionary."""
    calls: dict[str, object] = {}

    def post_backward(**kwargs: object) -> bool:
        calls.update(kwargs)
        return True

    trainer = object.__new__(Trainer3DGRUT)
    trainer.strategy = SimpleNamespace(post_backward=post_backward)
    trainer.scene_extent = 4.0
    trainer.train_dataset = object()
    trainer.tracking = SimpleNamespace(writer=object())
    gpu_batch = object()
    outputs = {"mog_projected_extent": object()}

    assert trainer._post_backward_strategy_step(
        global_step=7,
        gpu_batch=gpu_batch,
        outputs=outputs,
    )
    assert calls["step"] == 7
    assert calls["scene_extent"] == 4.0
    assert calls["train_dataset"] is trainer.train_dataset
    assert calls["batch"] is gpu_batch
    assert calls["writer"] is trainer.tracking.writer
    assert calls["outputs"] is outputs
