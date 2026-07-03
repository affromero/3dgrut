"""Tests for validation checkpointing."""

import json
import os

import pytest
from omegaconf import OmegaConf
from threedgrut.trainer import Trainer3DGRUT


class _ScalarWriter:
    """Minimal scalar writer used by checkpointing-only tests."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, value, step))


class _Tracking:
    """Minimal tracking shell used by checkpointing-only tests."""

    def __init__(self, output_dir: str = "") -> None:
        self.writer = _ScalarWriter()
        self.output_dir = output_dir


class _Model:
    """Minimal model shell for metrics logging."""

    num_gaussians = 123


class _Scheduler:
    """Minimal PPISP scheduler shell."""

    def __init__(self, last_epoch: int) -> None:
        self.last_epoch = last_epoch


class _PostProcessing:
    """Minimal PPISP post-processing shell."""

    def __init__(self, *, last_epoch: int, activation_step: int) -> None:
        self._ppisp_scheduler = _Scheduler(last_epoch)
        self._controller_activation_step = activation_step


def _checkpointing_trainer(saved_paths: list[str]) -> Trainer3DGRUT:
    """Create a Trainer3DGRUT shell without initializing CUDA state."""
    trainer = object.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "checkpoint": {
                "save_last_on_validation": False,
                "save_best_on_validation": True,
            },
            "early_stopping": {
                "enabled": True,
                "metric": "masked_psnr",
                "patience": 3,
                "min_delta": 0.03,
                "min_score": 0.0,
                "min_score_after_step": 0,
                "restore_best_on_end": True,
            },
        }
    )
    trainer.global_step = 22000
    trainer._should_stop_training = False
    trainer._best_validation_score = 26.408542533419027
    trainer._best_validation_step = 20000
    trainer._early_stopping_reference_score = 26.408542533419027
    trainer._early_stopping_reference_step = 20000
    trainer._stale_validation_count = 0
    trainer._best_checkpoint_path = "/tmp/previous_best.pt"

    def save_checkpoint(
        *,
        last_checkpoint: bool = False,
        best_checkpoint: bool = False,
    ) -> str:
        assert not last_checkpoint
        assert best_checkpoint
        saved_paths.append("/tmp/current_best.pt")
        return "/tmp/current_best.pt"

    setattr(trainer, "save_checkpoint", save_checkpoint)
    trainer.tracking = _Tracking()
    return trainer


def test_best_checkpoint_tracks_small_positive_validation_gain() -> None:
    """Small PSNR gains should save ckpt_best without resetting patience."""
    saved_paths: list[str] = []
    trainer = _checkpointing_trainer(saved_paths)
    metrics = {"masked_psnr": [26.435249677344935]}

    trainer._handle_validation_checkpointing(metrics)

    assert saved_paths == ["/tmp/current_best.pt"]
    assert trainer._best_checkpoint_path == "/tmp/current_best.pt"
    assert trainer._best_validation_score == pytest.approx(26.435249677344935)
    assert trainer._best_validation_step == 22000
    assert trainer._early_stopping_reference_score == pytest.approx(
        26.408542533419027
    )
    assert trainer._early_stopping_reference_step == 20000
    assert trainer._stale_validation_count == 1


def test_validation_score_floor_stops_after_probe_step() -> None:
    """A live quality floor stops runs before long bad continuations."""
    saved_paths: list[str] = []
    trainer = _checkpointing_trainer(saved_paths)
    trainer.conf.early_stopping["min_score"] = 22.0
    trainer.conf.early_stopping["min_score_after_step"] = 5000
    trainer.global_step = 5000
    metrics = {"masked_psnr": [17.0]}

    trainer._handle_validation_checkpointing(metrics)

    assert saved_paths == []
    assert trainer._should_stop_training is True
    assert trainer._stale_validation_count == 0


def test_ppisp_pre_distillation_validation_tracks_geometry_plateau() -> None:
    """Pre-PPISP validation should track staleness without checkpointing."""
    saved_paths: list[str] = []
    trainer = _checkpointing_trainer(saved_paths)
    trainer.post_processing = _PostProcessing(
        last_epoch=32_000,
        activation_step=95_000,
    )
    trainer._distillation_start_step = 95_000
    trainer.global_step = 32_000
    trainer._best_validation_score = None
    trainer._best_validation_step = None
    trainer._best_checkpoint_path = None
    trainer._early_stopping_reference_score = None
    trainer._early_stopping_reference_step = None
    trainer._stale_validation_count = 0
    metrics = {"masked_psnr": [25.0]}

    trainer._handle_validation_checkpointing(metrics)

    assert saved_paths == []
    assert trainer._should_stop_training is False
    assert trainer._distillation_start_step == 95_000
    assert trainer.post_processing._controller_activation_step == 95_000
    assert trainer._early_stopping_reference_score == pytest.approx(25.0)
    assert trainer._early_stopping_reference_step == 32_000
    assert trainer._stale_validation_count == 0
    assert trainer._best_checkpoint_path is None


def test_ppisp_geometry_plateau_activates_distillation_instead_of_stopping() -> None:
    """A pre-PPISP plateau should start PPISP, not early-stop the run."""
    saved_paths: list[str] = []
    trainer = _checkpointing_trainer(saved_paths)
    trainer.conf.early_stopping["patience"] = 1
    trainer.post_processing = _PostProcessing(
        last_epoch=50_000,
        activation_step=95_000,
    )
    trainer._distillation_start_step = 95_000
    trainer.global_step = 50_000
    trainer._best_validation_score = None
    trainer._best_validation_step = None
    trainer._best_checkpoint_path = None
    trainer._early_stopping_reference_score = 25.0
    trainer._early_stopping_reference_step = 48_000
    trainer._stale_validation_count = 0
    metrics = {"masked_psnr": [25.01]}

    trainer._handle_validation_checkpointing(metrics)

    assert saved_paths == []
    assert trainer._should_stop_training is False
    assert trainer._distillation_start_step == 50_000
    assert trainer.post_processing._controller_activation_step == 50_000
    assert trainer._early_stopping_reference_score is None
    assert trainer._early_stopping_reference_step is None
    assert trainer._stale_validation_count == 0
    assert trainer._best_checkpoint_path is None


def test_ppisp_scheduled_distillation_starts_fresh_checkpoint_state() -> None:
    """Scheduled PPISP distillation should not inherit geometry patience."""
    saved_paths: list[str] = []
    trainer = _checkpointing_trainer(saved_paths)
    trainer.post_processing = _PostProcessing(
        last_epoch=95_000,
        activation_step=95_000,
    )
    trainer._distillation_start_step = 95_000
    trainer.global_step = 96_000
    trainer._best_validation_score = None
    trainer._best_validation_step = None
    trainer._best_checkpoint_path = None
    trainer._early_stopping_reference_score = 25.0
    trainer._early_stopping_reference_step = 94_000
    trainer._stale_validation_count = 2
    metrics = {"masked_psnr": [24.8]}

    trainer._handle_validation_checkpointing(metrics)

    assert saved_paths == ["/tmp/current_best.pt"]
    assert trainer._should_stop_training is False
    assert trainer._best_validation_score == pytest.approx(24.8)
    assert trainer._best_validation_step == 96_000
    assert trainer._early_stopping_reference_score == pytest.approx(24.8)
    assert trainer._early_stopping_reference_step == 96_000
    assert trainer._stale_validation_count == 0


def test_training_metric_indices_are_deterministic() -> None:
    """Train metric sampling should be stable across repeated runs."""
    assert Trainer3DGRUT._selected_training_metric_indices(10, 4) == [
        0,
        3,
        6,
        9,
    ]
    assert Trainer3DGRUT._selected_training_metric_indices(5, 0) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert Trainer3DGRUT._selected_training_metric_indices(0, 4) == []


def test_training_metrics_pass_writes_comparable_summary(
    tmp_path: object,
) -> None:
    """Train metrics should log scalars and a step-indexed JSON summary."""
    trainer = object.__new__(Trainer3DGRUT)
    trainer.tracking = _Tracking(output_dir=str(tmp_path))
    trainer.global_step = 42
    trainer.model = _Model()
    metrics = {
        "psnr": [30.0, 32.0],
        "masked_psnr": [29.0, 31.0],
        "ssim": [0.8, 0.9],
        "lpips": [0.2, 0.1],
        "mask_coverage": [0.5, 0.7],
        "gradient_l1": [0.2, 0.4],
        "source_scan_id": ["scan_a", "scan_a"],
        "camera_idx": [0, 1],
        "losses": {"total_loss": [0.1, 0.2]},
    }

    trainer.log_training_metrics_pass(metrics)

    scalars = dict(
        (name, value)
        for name, value, step in trainer.tracking.writer.scalars
        if step == 42
    )
    assert scalars["train_eval/psnr"] == pytest.approx(31.0)
    assert scalars["train_eval/masked_psnr"] == pytest.approx(30.0)
    assert scalars["train_eval/loss/total"] == pytest.approx(0.15)

    summary_path = os.path.join(
        str(tmp_path),
        "training_metrics_step_000042.json",
    )
    with open(summary_path, encoding="utf-8") as file:
        summary = json.load(file)
    assert summary["global_step"] == pytest.approx(42.0)
    assert summary["num_gaussians"] == pytest.approx(123.0)
    assert summary["evaluated_views"] == pytest.approx(2.0)
    assert summary["psnr"] == pytest.approx(31.0)
    assert summary["source_scan_metrics"]["scan_a"]["view_count"] == (
        pytest.approx(2.0)
    )
