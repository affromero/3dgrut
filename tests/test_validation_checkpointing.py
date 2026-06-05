"""Tests for validation checkpointing."""

import pytest
from omegaconf import OmegaConf
from threedgrut.trainer import Trainer3DGRUT


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
