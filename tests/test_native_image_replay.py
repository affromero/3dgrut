"""Behavior tests for immutable visibility-adaptive image replay."""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from threedgrut.datasets.native_image_replay import (
    NativeImageReplay,
    NativeImageReplaySampler,
    NativeImageSelection,
)
from threedgrut.trainer import Trainer3DGRUT


def _write_trace(tmp_path, rows: tuple[str, ...]) -> str:
    trace_path = tmp_path / "sampler.tsv"
    trace_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(trace_path)


def test_replay_preserves_native_order_and_dimensions(tmp_path) -> None:
    """Replay keeps the measured selection, revisit, and output size."""
    trace_path = _write_trace(
        tmp_path,
        (
            "event\tsequence\tfields",
            "image\t1\t7\t11\t0",
            "update\t1\t99\t1024\t1024",
            "image\t2\t3\t29\t1",
            "update\t2\t77\t1280\t1226",
        ),
    )

    replay = NativeImageReplay(trace_path)

    assert len(replay) == 2
    assert replay.selection_at(0) == NativeImageSelection(7, 11, 0, 1024, 1024, 1)
    assert replay.selection_at(1) == NativeImageSelection(3, 29, 1, 1280, 1226, 2)
    assert list(
        NativeImageReplaySampler(replay, start_step=1, end_step=2)
    ) == [replay.selection_at(1)]


def test_replay_rejects_noncontiguous_or_unpaired_rows(tmp_path) -> None:
    """A partial capture cannot silently become a different training run."""
    missing_update = _write_trace(
        tmp_path,
        ("image\t1\t0\t1\t0",),
    )
    with pytest.raises(ValueError, match="one image and update"):
        NativeImageReplay(missing_update)

    gap = _write_trace(
        tmp_path,
        (
            "image\t2\t0\t1\t0",
            "update\t2\t10\t1024\t1024",
        ),
    )
    with pytest.raises(ValueError, match="contiguous"):
        NativeImageReplay(gap)


def test_replay_manifest_rejects_changed_trace(tmp_path) -> None:
    """Resuming is valid only against the exact captured selection stream."""
    trace_path = _write_trace(
        tmp_path,
        (
            "image\t1\t0\t1\t0",
            "update\t1\t10\t1024\t1024",
        ),
    )
    replay = NativeImageReplay(trace_path)
    manifest = replay.manifest()
    _write_trace(
        tmp_path,
        (
            "image\t1\t0\t2\t0",
            "update\t1\t10\t1024\t1024",
        ),
    )

    with pytest.raises(ValueError, match="hash differs"):
        NativeImageReplay(trace_path).validate_manifest(manifest)


class _ReplayDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, selection: NativeImageSelection) -> dict[str, int]:
        return {
            "ordinal": selection.ordinal,
            "sequence": selection.sequence,
            "width": selection.width,
        }

    def get_intrinsics_idx(self, frame_idx: int) -> int:
        return 17 if frame_idx >= 4 else 10

    def get_camera_idx(self, frame_idx: int) -> int:
        return 1 if frame_idx >= 4 else 0


def _native_replay_config(
    *,
    enabled: bool,
    trace_path: str | None,
    age_increment_by_camera: dict[str, int],
) -> DictConfig:
    return OmegaConf.create(
        {
            "n_iterations": 2,
            "dataset": {
                "type": "colmap",
                "test_split_interval": 0,
                "native_image_scale": {"enabled": True},
                "holdout_image_list_path": None,
                "train_exclude_image_list_path": None,
                "train_focus_image_list_path": None,
                "native_image_replay": {
                    "enabled": enabled,
                    "trace_path": trace_path,
                    "age_increment_by_camera": age_increment_by_camera,
                },
            },
        }
    )


def test_trainer_replay_starts_at_committed_global_step(tmp_path) -> None:
    """Rebuilt loaders derive selections from the checkpoint step, not prefetch."""
    trace_path = _write_trace(
        tmp_path,
        (
            "image\t1\t7\t11\t0",
            "update\t1\t99\t1024\t1024",
            "image\t2\t3\t29\t1",
            "update\t2\t77\t1280\t1226",
            "image\t3\t5\t31\t0",
            "update\t3\t55\t1024\t1024",
        ),
    )
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.global_step = 1
    trainer.run_generator = torch.Generator().manual_seed(7)
    trainer.train_dataset = _ReplayDataset()
    trainer.native_image_replay = NativeImageReplay(trace_path)
    conf = OmegaConf.create({"num_workers": 0, "n_iterations": 3})

    loader = trainer._make_train_dataloader(conf)
    batches = list(loader)

    assert [int(batch["sequence"].item()) for batch in batches] == [2, 3]
    assert [int(batch["ordinal"].item()) for batch in batches] == [3, 5]
    loader.shutdown()


def test_trainer_rebuilds_replay_loader_after_global_step_restore(
    tmp_path,
) -> None:
    """A checkpoint restore replaces the pre-restore replay sampler."""
    trace_path = _write_trace(
        tmp_path,
        (
            "image\t1\t7\t11\t0",
            "update\t1\t99\t1024\t1024",
            "image\t2\t3\t29\t1",
            "update\t2\t77\t1280\t1226",
            "image\t3\t5\t31\t0",
            "update\t3\t55\t1024\t1024",
        ),
    )
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.global_step = 0
    trainer.run_generator = torch.Generator().manual_seed(7)
    trainer.train_dataset = _ReplayDataset()
    trainer.native_image_replay = NativeImageReplay(trace_path)
    conf = OmegaConf.create({"num_workers": 0, "n_iterations": 3})
    trainer.train_dataloader = trainer._make_train_dataloader(conf)

    trainer.global_step = 1
    trainer._rebuild_native_replay_dataloader_for_global_step(conf)
    batches = list(trainer.train_dataloader)

    assert [int(batch["sequence"].item()) for batch in batches] == [2, 3]
    trainer.train_dataloader.shutdown()


def test_trainer_maps_native_age_increment_from_source_camera_id(tmp_path) -> None:
    """Replay config uses stable source IDs, not derived camera indices."""
    trace_path = _write_trace(
        tmp_path,
        (
            "image\t1\t0\t11\t0",
            "update\t1\t99\t1024\t1024",
            "image\t2\t4\t29\t0",
            "update\t2\t77\t1280\t1226",
        ),
    )
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.train_dataset = _ReplayDataset()
    conf = _native_replay_config(
        enabled=True,
        trace_path=trace_path,
        age_increment_by_camera={"17": 5},
    )

    replay = trainer._load_native_image_replay(conf)

    assert replay is not None
    assert trainer.native_image_age_increments_by_camera_id == {17: 5}
    assert trainer.native_image_age_increments_by_camera_idx == {1: 5}


def test_trainer_rejects_native_age_schedule_without_replay() -> None:
    """A configured schedule cannot silently do nothing when replay is off."""
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.train_dataset = _ReplayDataset()
    conf = _native_replay_config(
        enabled=False,
        trace_path=None,
        age_increment_by_camera={"17": 5},
    )

    with pytest.raises(ValueError, match="requires native_image_replay"):
        trainer._load_native_image_replay(conf)


def test_native_age_schedule_manifest_rejects_changed_resume() -> None:
    """Resume requires the same stable source-camera age schedule."""
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.native_image_age_increments_by_camera_id = {17: 5}

    manifest = {
        "age_increment_by_camera": (
            trainer._native_replay_age_increment_manifest()
        )
    }
    trainer._validate_native_replay_age_increment_manifest(manifest)

    with pytest.raises(ValueError, match="age increments differ"):
        trainer._validate_native_replay_age_increment_manifest(
            {"age_increment_by_camera": "{}"}
        )
