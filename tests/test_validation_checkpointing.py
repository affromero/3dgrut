"""Tests for validation checkpointing."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
import threedgrut.trainer as trainer_module
from threedgrut.trainer import (
    SOURCE_SCAN_METRIC_NAMES,
    Trainer3DGRUT,
    _group_metric_summary,
    _group_metric_summary_by_keys,
    _uses_known_frame_reconstruction_validation,
    _validate_native_camera_configuration,
)


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


class _NativeAppearancePostProcessing:
    """Minimal native appearance shell for validation-routing tests."""

    use_native_appearance_grid = True


def test_checkpoint_serializes_lidar_sampler_state(tmp_path: object) -> None:
    """LiDAR-enabled checkpoints must resume the exact packet stream."""
    lidar_state = {
        "packet_count": 2,
        "packet_queue": [(0, 3, 1), (1, 7, 0)],
        "pcg_state": 23,
        "rng_state": {"state": 29},
        "source_fingerprint": "fingerprint",
    }
    trainer = object.__new__(Trainer3DGRUT)
    trainer.global_step = 17
    trainer.n_epochs = 1
    trainer.tracking = _Tracking(str(tmp_path))
    trainer.model = SimpleNamespace(
        get_model_parameters=lambda: {"positions": torch.zeros((1, 3))}
    )
    trainer.strategy = SimpleNamespace(get_strategy_parameters=lambda: {})
    trainer.lidar_ray_sampler = SimpleNamespace(state_dict=lambda: lidar_state)
    trainer.feature_decoder = None
    trainer.post_processing = None
    trainer.camera_residual = None
    trainer.geometry_only_optimizer = None
    trainer._pending_geometry_optimizer_state = None
    trainer.conf = OmegaConf.create(
        {"checkpoint": {"keep_step_checkpoints": True}}
    )

    checkpoint_path = trainer.save_checkpoint()
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    assert checkpoint["lidar_supervision"] == lidar_state


def test_init_dataloaders_builds_validation_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation setup uses the platform-adjusted loader configuration."""
    train_dataset = object()
    val_dataset = torch.utils.data.TensorDataset(torch.arange(2))

    def make_datasets(
        *,
        name: str,
        config: object,
        ray_jitter: object,
    ) -> tuple[object, torch.utils.data.TensorDataset]:
        assert name == "colmap"
        assert config is conf
        assert ray_jitter is None
        return train_dataset, val_dataset

    trainer = object.__new__(Trainer3DGRUT)
    trainer._load_native_image_replay = lambda _conf: None
    trainer._make_train_dataloader = lambda _conf: "train-loader"
    conf = OmegaConf.create(
        {"dataset": {"type": "colmap"}, "num_workers": 0}
    )
    monkeypatch.setattr(trainer_module.datasets, "make", make_datasets)

    trainer.init_dataloaders(conf)

    assert trainer.train_dataset is train_dataset
    assert trainer.train_dataloader == "train-loader"
    assert trainer.val_dataset is val_dataset
    assert trainer.val_dataloader.dataset is val_dataset
    assert trainer.val_dataloader.batch_size == 1


def test_run_training_finalizes_before_final_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal topology mutation must precede final rendered metrics."""
    events: list[str] = []

    def finalize_training() -> bool:
        events.append("finalize")
        return True

    trainer = object.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "render": {"method": "native_ewa"},
            "camera_residual": {
                "finite_difference_audit": {
                    "enabled": False,
                    "exit_after": False,
                },
            },
            "camera_intrinsics_audit": {
                "enabled": False,
                "exit_after": False,
            },
            "validate_only": False,
            "validate_final": True,
            "n_iterations": 0,
        }
    )
    trainer.model = SimpleNamespace(
        optimizer=object(),
        build_acc=lambda *, rebuild: events.append(
            f"build_acc:{rebuild}"
        ),
    )
    trainer.strategy = SimpleNamespace(finalize_training=finalize_training)
    trainer.n_epochs = 0
    trainer.global_step = 0
    trainer._should_stop_training = False
    trainer.diagnostics = None
    trainer.gui = None
    trainer.run_training_metrics_pass = lambda _conf: (
        events.append("training_metrics") or {"psnr": [30.0]}
    )
    trainer.run_validation_pass = lambda _conf, training_metrics: (
        events.append("validation") or training_metrics
    )
    trainer._handle_validation_checkpointing = lambda _metrics: events.append(
        "checkpoint"
    )
    trainer.on_training_end = lambda: events.append("end")
    test_logger = SimpleNamespace(
        finished_tasks={"Training": {"elapsed": 1.0}},
        log_rule=lambda *_args, **_kwargs: None,
        start_progress=lambda **_kwargs: None,
        end_progress=lambda **_kwargs: None,
        log_table=lambda *_args, **_kwargs: None,
        info=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(trainer_module, "logger", test_logger)

    trainer.run_training()

    assert events == [
        "finalize",
        "build_acc:True",
        "training_metrics",
        "validation",
        "checkpoint",
        "end",
    ]


def test_native_camera_rejects_pretransformed_world_rays() -> None:
    """Absolute COLMAP ownership requires unposed camera-space rays."""
    conf = OmegaConf.create(
        {
            "dataset": {
                "shutter_type": "GLOBAL",
                "blur_samples": 1,
                "rs_ray_injection": True,
            },
            "post_processing": {"replay_native_extrinsics": False},
            "camera_residual": {
                "optimize_global": False,
                "optimize_per_camera": False,
                "optimize_per_image": True,
                "optimize_rolling_per_camera": False,
                "rotation_lr": 1.0e-5,
                "translation_lr": 1.0e-3,
                "eps": 1.0e-7,
                "reg_lambda": 0.0,
                "warmup_steps": 0,
                "lr_end_fraction": 1.0,
                "betas": [0.8, 0.95],
                "finite_difference_audit": {"enabled": False},
            },
        }
    )

    with pytest.raises(ValueError, match="camera-space rays"):
        _validate_native_camera_configuration(conf)


@pytest.mark.parametrize(
    ("validate_only", "test_split_interval", "holdout_path", "expected"),
    (
        (False, 0, None, True),
        (False, 8, None, False),
        (False, 0, "holdout.txt", False),
        (True, 0, None, False),
    ),
)
def test_native_reconstruction_validation_requires_the_training_split(
    validate_only: bool,
    test_split_interval: int,
    holdout_path: str | None,
    expected: bool,
) -> None:
    """Known appearance must never leak into a held-out validation split."""
    conf = OmegaConf.create(
        {
            "validate_only": validate_only,
            "dataset": {
                "test_split_interval": test_split_interval,
                "holdout_image_list_path": holdout_path,
            },
        }
    )

    actual = _uses_known_frame_reconstruction_validation(
        conf=conf,
        post_processing=_NativeAppearancePostProcessing(),
    )

    assert actual is expected


def test_optional_metrics_keep_their_own_group_alignment() -> None:
    """Masked metrics should group only the views carrying a mask."""
    metrics = {
        "source_scan_id": ["scan_a", "scan_a", "scan_b"],
        "camera_idx": [0, 1, 0],
        "psnr": [10.0, 20.0, 30.0],
        "masked_psnr": [11.0, 31.0],
        "masked_psnr_source_scan_id": ["scan_a", "scan_b"],
        "masked_psnr_camera_idx": [0, 0],
    }

    by_scan = _group_metric_summary(
        metrics=metrics,
        group_key="source_scan_id",
        metric_names=SOURCE_SCAN_METRIC_NAMES,
    )
    by_scan_camera = _group_metric_summary_by_keys(
        metrics=metrics,
        group_keys=("source_scan_id", "camera_idx"),
        metric_names=SOURCE_SCAN_METRIC_NAMES,
    )

    assert by_scan["scan_a"]["psnr"] == pytest.approx(15.0)
    assert by_scan["scan_a"]["masked_psnr"] == pytest.approx(11.0)
    assert by_scan["scan_b"]["masked_psnr"] == pytest.approx(31.0)
    assert by_scan_camera["scan_a/0"]["masked_psnr"] == pytest.approx(
        11.0
    )
    assert "masked_psnr" not in by_scan_camera["scan_a/1"]


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
