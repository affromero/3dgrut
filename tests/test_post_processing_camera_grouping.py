"""Tests for physical-camera grouping used by appearance correction."""

from types import SimpleNamespace

import pytest

import threedgrut.trainer as trainer_module
from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.trainer import Trainer3DGRUT


def _extrinsic(name: str, camera_id: int) -> SimpleNamespace:
    """Return the COLMAP extrinsic fields used by camera grouping."""
    return SimpleNamespace(name=name, camera_id=camera_id)


@pytest.mark.parametrize(
    ("name", "camera_id", "expected"),
    [
        ("front_0007.png", 11, "front"),
        ("left_0007.png", 12, "left"),
        ("right_0007.png", 13, "right"),
        ("rig/front_0007.png", 14, "front"),
        ("rig\\left_0007.png", 14, "left"),
        ("front/0007.png", 14, "front"),
        ("rig/right/0007.png", 14, "right"),
        ("0007.png", 15, "camera_id_15"),
        ("capture_left.png", 16, "camera_id_16"),
        ("image_0007.png", 17, "camera_id_17"),
    ],
)
def test_physical_camera_key_recovers_canonical_camera_from_wrappers(
    name: str,
    camera_id: int,
    expected: str,
) -> None:
    """Canonical basename or camera folder survives wrapper directories."""
    extrinsic = _extrinsic(name, camera_id)

    assert ColmapDataset._post_processing_camera_key(extrinsic) == expected


def test_physical_camera_counts_indices_and_names_share_stable_order() -> None:
    """Camera counts, indices, and labels use the same lexical key order."""
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.cam_extrinsics = [
        _extrinsic("right_0007.png", 31),
        _extrinsic("front_0008.png", 32),
        _extrinsic("left_0007.png", 33),
        _extrinsic("front_0007.png", 34),
    ]
    keys = sorted({dataset._post_processing_camera_key(extrinsic) for extrinsic in dataset.cam_extrinsics})
    dataset._post_processing_camera_key_to_idx = {key: idx for idx, key in enumerate(keys)}

    assert dataset.get_post_processing_camera_names() == [
        "front",
        "left",
        "right",
    ]
    assert dataset.get_post_processing_frames_per_camera() == [2, 1, 1]
    assert [dataset.get_post_processing_camera_idx(idx) for idx in range(len(dataset.cam_extrinsics))] == [2, 0, 1, 0]


def test_expected_physical_camera_count_rejects_collapsed_rig() -> None:
    """A known three-camera recipe must not silently train one bucket."""
    with pytest.raises(ValueError, match="expected 3, got 1"):
        Trainer3DGRUT._validate_post_processing_camera_count(
            num_cameras=1,
            expected_num_cameras=3,
        )


def test_trainer_uses_physical_camera_names() -> None:
    """PPISP labels follow physical buckets rather than camera-ID rows."""
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer._post_processing_camera_index_mode = "dataset"
    trainer.train_dataset = SimpleNamespace(
        get_post_processing_camera_names=lambda: [
            "front",
            "left",
            "right",
        ],
        get_camera_names=lambda: ["camera_0", "camera_1"],
        get_post_processing_frames_per_camera=lambda: [2, 1, 1],
        get_frames_per_camera=lambda: [1, 1, 1, 1],
    )

    assert trainer._post_processing_camera_names() == [
        "front",
        "left",
        "right",
    ]


def _metadata_trainer(
    *,
    camera_names: list[str] | None = None,
    frame_counts: list[int] | None = None,
    global_step: int = 101,
    scheduler_last_epoch: int = 101,
    activation_step: int = 100,
) -> Trainer3DGRUT:
    """Return a minimal trainer with a real physical-camera contract."""
    resolved_names = camera_names or ["front", "left", "right"]
    resolved_counts = frame_counts or [212, 212, 212]
    scheduler = SimpleNamespace(last_epoch=scheduler_last_epoch)
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer._post_processing_camera_index_mode = "dataset"
    trainer.train_dataset = SimpleNamespace(
        get_post_processing_camera_names=lambda: resolved_names,
        get_post_processing_frames_per_camera=lambda: resolved_counts,
        get_frames_per_camera=lambda: resolved_counts,
    )
    trainer.conf = SimpleNamespace(post_processing=SimpleNamespace(method="ppisp"))
    trainer.post_processing = SimpleNamespace(
        config=SimpleNamespace(use_controller=True),
        _controller_activation_step=activation_step,
        _ppisp_scheduler=scheduler,
    )
    trainer.global_step = global_step
    return trainer


def test_checkpoint_metadata_seals_camera_order_and_controller_proof() -> None:
    """Checkpoint metadata exactly records camera slots and PPISP readiness."""
    trainer = _metadata_trainer()

    assert trainer._post_processing_checkpoint_metadata() == {
        "camera_keys": ["front", "left", "right"],
        "camera_frame_counts": [212, 212, 212],
        "camera_index_mode": "dataset",
        "inference_contract": {
            "schema_version": 1,
            "checkpoint_global_step": 101,
            "controller_enabled": True,
            "controller_trained": True,
            "controller_activation_step": 100,
            "scheduler_last_epoch": 101,
        },
    }


def test_save_checkpoint_embeds_common_eval_contract(monkeypatch) -> None:
    """The checkpoint payload consumed by common eval includes the contract."""
    trainer = _metadata_trainer()
    trainer.model = SimpleNamespace(get_model_parameters=lambda: {})
    trainer.strategy = SimpleNamespace(get_strategy_parameters=lambda: {})
    trainer.tracking = SimpleNamespace(output_dir="/tmp/checkpoint-contract")
    trainer.n_epochs = 1
    trainer.conf.checkpoint = {"keep_step_checkpoints": False}
    trainer.post_processing.state_dict = lambda: {"module": "state"}
    trainer.post_processing_optimizers = [SimpleNamespace(state_dict=lambda: {"optimizer": "state"})]
    trainer.post_processing_schedulers = [SimpleNamespace(state_dict=lambda: {"last_epoch": 101})]
    trainer.camera_residual = None
    trainer.camera_residual_optimizer = None
    trainer.camera_residual_scheduler = None
    saved = {}
    monkeypatch.setattr(
        trainer_module.torch,
        "save",
        lambda parameters, path: saved.update(
            parameters=parameters,
            path=path,
        ),
    )
    monkeypatch.setattr(
        trainer_module.os,
        "replace",
        lambda source, dest: None,
    )

    trainer.save_checkpoint(last_checkpoint=True)

    post_processing = saved["parameters"]["post_processing"]
    assert post_processing["camera_keys"] == ["front", "left", "right"]
    assert post_processing["camera_frame_counts"] == [212, 212, 212]
    assert post_processing["camera_index_mode"] == "dataset"
    assert post_processing["inference_contract"]["controller_trained"] is True


def test_checkpoint_metadata_does_not_claim_untrained_controller() -> None:
    """The activation step itself is not proof of a controller update."""
    trainer = _metadata_trainer(
        scheduler_last_epoch=100,
        activation_step=100,
    )

    contract = trainer._post_processing_checkpoint_metadata()["inference_contract"]

    assert isinstance(contract, dict)
    assert contract["controller_trained"] is False


@pytest.mark.parametrize(
    ("camera_names", "frame_counts", "message"),
    [
        (
            ["front", "front", "right"],
            [212, 212, 212],
            "camera keys must be unique",
        ),
        (
            ["front", "left", "right"],
            [212, 212, 0],
            "frame counts must be positive",
        ),
    ],
)
def test_checkpoint_metadata_rejects_ambiguous_camera_contract(
    camera_names: list[str],
    frame_counts: list[int],
    message: str,
) -> None:
    """An unusable physical-camera contract cannot enter a checkpoint."""
    trainer = _metadata_trainer(
        camera_names=camera_names,
        frame_counts=frame_counts,
    )

    with pytest.raises(ValueError, match=message):
        trainer._post_processing_checkpoint_metadata()


def test_expected_physical_camera_count_accepts_matching_rig() -> None:
    """The launch guard accepts the intended physical rig."""
    Trainer3DGRUT._validate_post_processing_camera_count(
        num_cameras=3,
        expected_num_cameras=3,
        frames_per_camera=[212, 212, 212],
        camera_names=["front", "left", "right"],
        expected_camera_names=["front", "left", "right"],
    )


def test_expected_physical_camera_count_rejects_inactive_bucket() -> None:
    """A declared camera without train frames cannot pass the rig guard."""
    with pytest.raises(ValueError, match="inactive indices: \\[2\\]"):
        Trainer3DGRUT._validate_post_processing_camera_count(
            num_cameras=3,
            expected_num_cameras=3,
            frames_per_camera=[212, 212, 0],
        )


def test_expected_physical_camera_names_reject_wrong_mapping() -> None:
    """The count alone cannot substitute for expected camera identities."""
    with pytest.raises(ValueError, match="names mismatch"):
        Trainer3DGRUT._validate_post_processing_camera_count(
            num_cameras=3,
            expected_num_cameras=3,
            frames_per_camera=[212, 212, 212],
            camera_names=["camera_id_1", "camera_id_2", "camera_id_3"],
            expected_camera_names=["front", "left", "right"],
        )
