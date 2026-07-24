"""Checkpoint-render coverage for native absolute camera state."""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from threedgrut import render
from threedgrut.datasets.protocols import Batch
from threedgrut.model.native_camera_extrinsics import (
    NativeCameraExtrinsics,
)


class _Dataset:
    """Small stable source-frame table for checkpoint-render tests."""

    def __init__(
        self,
        *,
        qvecs: torch.Tensor,
        tvecs: torch.Tensor,
        manifest_hash: str,
    ) -> None:
        self.qvecs = qvecs
        self.tvecs = tvecs
        self.manifest_hash = manifest_hash

    def get_source_frame_colmap_extrinsics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the source poses used to construct the checkpoint."""
        return self.qvecs.clone(), self.tvecs.clone()

    def get_source_frame_manifest_hash(self) -> str:
        """Return the stable source-frame ordering fingerprint."""
        return self.manifest_hash


class _NativeAppearance:
    """Small post-processing surface for known-frame routing tests."""

    use_native_appearance_grid = True


def _source_poses() -> tuple[torch.Tensor, torch.Tensor]:
    qvecs = torch.tensor(
        ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        dtype=torch.float32,
    )
    tvecs = torch.tensor(
        ((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
        dtype=torch.float32,
    )
    return qvecs, tvecs


def _conf(
    *,
    holdout_path: str | None = None,
    apply_known_frame_in_eval: bool = False,
) -> DictConfig:
    return OmegaConf.create(
        {
            "camera_residual": {
                "enabled": True,
                "native_absolute_colmap": True,
            },
            "dataset": {
                "test_split_interval": 0,
                "holdout_image_list_path": holdout_path,
            },
            "post_processing": {
                "apply_known_frame_in_eval": apply_known_frame_in_eval,
            },
        }
    )


def _dataset(manifest_hash: str = "source-manifest") -> _Dataset:
    qvecs, tvecs = _source_poses()
    return _Dataset(
        qvecs=qvecs,
        tvecs=tvecs,
        manifest_hash=manifest_hash,
    )


def _checkpoint(dataset: _Dataset) -> dict:
    module = NativeCameraExtrinsics(
        initial_qvecs=dataset.qvecs,
        initial_tvecs=dataset.tvecs,
    )
    with torch.no_grad():
        module.tvecs[0, 0] = 0.5
    return {
        "camera_residual": {
            "format_version": module.checkpoint_format_version,
            "algorithm": module.checkpoint_algorithm,
            "optimizer_group_manifest": module.optimizer_group_manifest(),
            "source_frame_manifest_hash": dataset.manifest_hash,
            "module": {
                name: value.detach().clone()
                for name, value in module.state_dict().items()
            },
        }
    }


def _restore(
    checkpoint: dict,
    dataset: _Dataset,
) -> NativeCameraExtrinsics:
    restored = render._restore_native_camera_extrinsics(
        checkpoint,
        conf=_conf(),
        dataset=dataset,
        device=torch.device("cpu"),
    )
    assert restored is not None
    return restored


def _batch(source_frame_idx: int = 0) -> Batch:
    return Batch(
        rays_ori=torch.zeros((1, 1, 1, 3)),
        rays_dir=torch.tensor([[[[0.0, 0.0, 1.0]]]]),
        T_to_world=torch.eye(4).reshape(1, 4, 4),
        source_frame_idx=source_frame_idx,
    )


def test_checkpoint_render_restores_native_camera_extrinsics() -> None:
    """Checkpoint rendering restores the trained absolute source poses."""
    dataset = _dataset()
    checkpoint = _checkpoint(dataset)

    restored = _restore(checkpoint, dataset)

    assert not restored.training
    torch.testing.assert_close(
        restored.qvecs,
        checkpoint["camera_residual"]["module"]["qvecs"],
    )
    torch.testing.assert_close(
        restored.tvecs,
        checkpoint["camera_residual"]["module"]["tvecs"],
    )


def test_native_camera_render_rejects_missing_checkpoint_state() -> None:
    """An enabled native mode cannot silently omit its pose checkpoint."""
    with pytest.raises(ValueError, match="camera_residual checkpoint state"):
        render._restore_native_camera_extrinsics(
            {},
            conf=_conf(),
            dataset=_dataset(),
            device=torch.device("cpu"),
        )


def test_native_camera_render_rejects_mismatched_manifest() -> None:
    """A source-frame reordering fails before render-time pose lookup."""
    checkpoint_dataset = _dataset()
    render_dataset = _dataset(manifest_hash="other-manifest")

    with pytest.raises(ValueError, match="manifest mismatch"):
        _restore(_checkpoint(checkpoint_dataset), render_dataset)


def test_native_camera_render_rejects_invalid_checkpoint_contract() -> None:
    """An incompatible camera algorithm cannot be interpreted as native."""
    dataset = _dataset()
    checkpoint = _checkpoint(dataset)
    checkpoint["camera_residual"]["algorithm"] = "unexpected"

    with pytest.raises(ValueError, match="algorithm is invalid"):
        _restore(checkpoint, dataset)


def test_native_camera_render_rejects_different_source_poses() -> None:
    """Matching frame names alone cannot substitute different source poses."""
    checkpoint_dataset = _dataset()
    render_dataset = _dataset()
    render_dataset.tvecs[0, 1] = 0.25

    with pytest.raises(ValueError, match="initial poses do not match"):
        _restore(_checkpoint(checkpoint_dataset), render_dataset)


def test_native_camera_render_applies_saved_pose_at_checkpoint_step() -> None:
    """The saved native pose is active at the checkpoint validation step."""
    restored = _restore(_checkpoint(_dataset()), _dataset())
    batch = _batch()

    at_checkpoint_step = render._apply_native_camera_extrinsics(
        restored,
        batch,
        global_step=12768,
    )
    at_negative_step = render._apply_native_camera_extrinsics(
        restored,
        batch,
        global_step=-1,
    )

    assert not torch.equal(at_checkpoint_step.T_to_world, batch.T_to_world)
    torch.testing.assert_close(
        at_checkpoint_step.T_to_world,
        at_negative_step.T_to_world,
    )


def test_native_camera_render_requires_source_frame_idx() -> None:
    """Unindexed standalone batches fail before pose or appearance lookup."""
    restored = _restore(_checkpoint(_dataset()), _dataset())

    with pytest.raises(ValueError, match="requires source_frame_idx"):
        render._apply_native_camera_extrinsics(
            restored,
            object(),
            global_step=12768,
        )


def test_known_frame_rendering_matches_native_reconstruction_contract() -> (
    None
):
    """No-holdout native evaluation uses known appearance like validation."""
    restored = _restore(_checkpoint(_dataset()), _dataset())

    assert render._use_known_frame_post_processing(
        conf=_conf(),
        post_processing=_NativeAppearance(),
        camera_residual=restored,
    )
    assert not render._use_known_frame_post_processing(
        conf=_conf(holdout_path="heldout.txt"),
        post_processing=_NativeAppearance(),
        camera_residual=restored,
    )
    assert render._use_known_frame_post_processing(
        conf=_conf(
            holdout_path="heldout.txt",
            apply_known_frame_in_eval=True,
        ),
        post_processing=_NativeAppearance(),
        camera_residual=restored,
    )
