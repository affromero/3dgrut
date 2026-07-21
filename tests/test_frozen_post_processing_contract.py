from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.post_processing.checkpoint_contract import (
    inherited_controller_inference_contract,
    module_state_sha256,
)
from threedgrut.render import _ppisp_controller_restore_manifest
from threedgrut.trainer import Trainer3DGRUT


def _trainer() -> Trainer3DGRUT:
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "post_processing": {
                "method": "ppisp",
                "frozen_inference_during_training": True,
            }
        }
    )
    trainer._post_processing_camera_index_mode = "dataset"
    trainer.train_dataset = SimpleNamespace(
        get_post_processing_camera_names=lambda: [
            "front",
            "left",
            "right",
        ],
        get_post_processing_frames_per_camera=lambda: [156, 156, 156],
        get_frames_per_camera=lambda: [156, 156, 156],
    )
    trainer.global_step = 5000
    trainer.post_processing = SimpleNamespace(
        state_dict=lambda: {},
    )
    return trainer


def _parent_post_processing() -> dict[str, object]:
    return {
        "module": {},
        "camera_keys": ["front", "left", "right"],
        "camera_frame_counts": [156, 156, 156],
        "camera_index_mode": "dataset",
        "inference_contract": {
            "schema_version": 3,
            "checkpoint_global_step": 12000,
            "controller_enabled": True,
            "controller_trained": True,
            "controller_activation_step": 7000,
            "scheduler_last_epoch": 12000,
            "multiscale_controller_enabled": False,
            "multiscale_controller_trained": False,
            "multiscale_view_context_enabled": False,
            "multiscale_view_context_trained": False,
            "multiscale_view_context_contract": None,
        },
    }


def test_frozen_post_processing_preserves_parent_inference_proof() -> None:
    trainer = _trainer()
    parent = _parent_post_processing()

    trainer._capture_frozen_post_processing_metadata(
        {"post_processing": parent}
    )
    metadata = trainer._post_processing_checkpoint_metadata()

    contract = metadata["inference_contract"]
    assert isinstance(contract, dict)
    assert contract["schema_version"] == 4
    assert contract["checkpoint_global_step"] == 5000
    assert contract["scheduler_last_epoch"] is None
    inherited = contract["frozen_parent_controller"]
    assert inherited["parent_checkpoint_global_step"] == 12000
    assert inherited["parent_scheduler_last_epoch"] == 12000
    assert metadata["camera_keys"] == ["front", "left", "right"]


def test_repeated_frozen_continuation_preserves_original_training_proof() -> (
    None
):
    module_sha256 = module_state_sha256({"weight": torch.tensor((1.0, 2.0))})
    parent_contract = {
        "schema_version": 4,
        "checkpoint_global_step": 5000,
        "controller_trained": True,
        "controller_activation_step": 7000,
        "scheduler_last_epoch": None,
        "frozen_parent_controller": {
            "schema_version": 1,
            "module_state_sha256": module_sha256,
            "parent_checkpoint_global_step": 12000,
            "parent_controller_activation_step": 7000,
            "parent_scheduler_last_epoch": 12000,
            "parent_controller_trained": True,
        },
    }

    contract = inherited_controller_inference_contract(
        parent_contract,
        module_sha256=module_sha256,
        checkpoint_global_step=5000,
    )

    inherited = contract["frozen_parent_controller"]
    assert inherited["parent_checkpoint_global_step"] == 12000
    assert inherited["parent_scheduler_last_epoch"] == 12000
    assert contract["scheduler_last_epoch"] is None


def test_frozen_post_processing_rejects_untrained_parent() -> None:
    trainer = _trainer()
    parent = _parent_post_processing()
    contract = parent["inference_contract"]
    assert isinstance(contract, dict)
    contract["controller_trained"] = False
    trainer._capture_frozen_post_processing_metadata(
        {"post_processing": parent}
    )

    with pytest.raises(ValueError, match="does not prove"):
        trainer._post_processing_checkpoint_metadata()


def test_frozen_post_processing_requires_complete_parent_metadata() -> None:
    trainer = _trainer()

    with pytest.raises(ValueError, match="incomplete"):
        trainer._capture_frozen_post_processing_metadata(
            {"post_processing": {"module": {}}}
        )


def _inherited_checkpoint() -> dict[str, object]:
    module = {"weight": torch.tensor((1.0, 2.0))}
    return {
        "global_step": 5000,
        "config": OmegaConf.create(
            {
                "post_processing": {
                    "frozen_inference_during_training": True,
                }
            }
        ),
        "post_processing": {
            "module": module,
            "schedulers": [],
            "inference_contract": {
                "schema_version": 4,
                "checkpoint_global_step": 5000,
                "controller_activation_step": 7000,
                "scheduler_last_epoch": None,
                "controller_trained": True,
                "frozen_parent_controller": {
                    "schema_version": 1,
                    "module_state_sha256": module_state_sha256(module),
                    "parent_checkpoint_global_step": 12000,
                    "parent_controller_activation_step": 7000,
                    "parent_scheduler_last_epoch": 12000,
                    "parent_controller_trained": True,
                },
            },
        },
    }


def test_inherited_controller_proof_authorizes_strict_inference() -> None:
    manifest = _ppisp_controller_restore_manifest(
        _inherited_checkpoint(),
        use_controller=True,
        configured_activation_step=4000,
        require_controller_ready=True,
    )

    assert manifest["ready_for_controller_inference"] is True
    assert manifest["proof_source"] == ("checkpoint_frozen_parent_contract")


def test_inherited_controller_proof_rejects_module_mutation() -> None:
    checkpoint = _inherited_checkpoint()
    post_processing = checkpoint["post_processing"]
    assert isinstance(post_processing, dict)
    module = post_processing["module"]
    assert isinstance(module, dict)
    module["weight"] = torch.tensor((1.0, 3.0))

    with pytest.raises(ValueError, match="module-state hash mismatch"):
        _ppisp_controller_restore_manifest(
            checkpoint,
            use_controller=True,
            configured_activation_step=4000,
            require_controller_ready=True,
        )


def test_inherited_controller_proof_requires_frozen_config() -> None:
    checkpoint = _inherited_checkpoint()
    config = checkpoint["config"]
    config.post_processing.frozen_inference_during_training = False

    with pytest.raises(ValueError, match="configured for frozen inference"):
        _ppisp_controller_restore_manifest(
            checkpoint,
            use_controller=True,
            configured_activation_step=4000,
            require_controller_ready=True,
        )
