"""Tests for validation checkpointing."""

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

    def __init__(self) -> None:
        self.writer = _ScalarWriter()


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
                "min_step": 0,
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

    trainer.save_checkpoint = save_checkpoint
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


def test_plateau_early_stopping_waits_for_min_step() -> None:
    """Plateau patience should not accrue before the configured min step."""
    saved_paths: list[str] = []
    trainer = _checkpointing_trainer(saved_paths)
    trainer.conf.early_stopping["min_step"] = 30000
    metrics = {"masked_psnr": [26.40]}

    trainer._handle_validation_checkpointing(metrics)

    assert saved_paths == []
    assert trainer._should_stop_training is False
    assert trainer._stale_validation_count == 0
