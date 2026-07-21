"""Tests for authenticated native range-ray geometry batches."""

import hashlib
import json
import os

import numpy as np
import pytest
import torch

from threedgrut.datasets.native_ray_evidence import NativeRayEvidence


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _write_fixture(folder: str) -> str:
    ray_path = os.path.join(folder, "rays_0007.npz")
    np.savez(
        ray_path,
        origins=np.asarray(
            [
                [10.0, 20.0, 30.0],
                [10.2, 20.0, 30.0],
                [10.0, 20.2, 30.0],
                [10.0, 20.0, 30.2],
            ],
            dtype=np.float32,
        ),
        directions=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        depths=np.asarray(
            [1.0, 2.0, 3.0, 4.0],
            dtype=np.float32,
        ),
        return_weights=np.asarray(
            [0.1, 0.2, 0.3, 0.4],
            dtype=np.float32,
        ),
        timestamps=np.arange(4, dtype=np.uint64),
    )
    manifest = {
        "schema_version": 3,
        "return_weight_semantics": (
            "normalized_leica_device_w_amplitude"
        ),
        "sealed_test_used": False,
        "development_data_used": False,
        "trajectory_fit_parity": "even",
        "supervised_return_parity": "odd",
        "rows": [
            {
                "exposure_index": 7,
                "file": "rays_0007.npz",
                "ray_count": 4,
                "sha256": _sha256_file(ray_path),
            }
        ],
    }
    manifest_path = os.path.join(folder, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return _sha256_file(manifest_path)


def test_native_ray_sample_uses_projector_camera_frame_and_is_deterministic(
    tmp_path,
) -> None:
    """The same step selects the same authenticated native rays."""
    folder = str(tmp_path)
    manifest_sha256 = _write_fixture(folder)
    evidence = NativeRayEvidence(
        folder=folder,
        manifest_sha256=manifest_sha256,
        sample_count=3,
        seed=11,
        equirect_width=8,
        equirect_height=4,
    )

    first = evidence.sample(
        exposure_index=7,
        global_step=19,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = evidence.sample(
        exposure_index=7,
        global_step=19,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert not first.rays_in_world_space
    assert first.rays_ori.shape == (1, 4, 8, 3)
    assert first.depth_gt.shape == (1, 4, 8, 1)
    assert torch.equal(
        first.depth_ray_z,
        torch.ones_like(first.depth_gt),
    )
    assert first.mask.sum().item() == 3
    active_weights = first.range_return_weight[first.mask > 0.5]
    assert any(
        sorted(active_weights.tolist()) == pytest.approx(expected)
        for expected in (
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.4],
            [0.1, 0.3, 0.4],
            [0.2, 0.3, 0.4],
        )
    )
    assert first.intrinsics is None
    assert (
        first.intrinsics_EquirectCameraModelParameters[
            "shutter_type"
        ]
        == "GLOBAL"
    )
    assert torch.equal(first.rays_ori, second.rays_ori)
    assert torch.equal(first.rays_dir, second.rays_dir)
    assert torch.equal(first.depth_gt, second.depth_gt)
    assert torch.equal(first.T_to_world, first.T_to_world_end)
    assert torch.allclose(
        first.T_to_world[0, :3, 3],
        torch.tensor([10.0, 20.0, 30.0]),
    )
    active = first.mask[0, ..., 0] > 0.5
    world_origins = (
        first.rays_ori[0][active]
        + first.T_to_world[0, :3, 3]
    )
    source_origins = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [10.2, 20.0, 30.0],
            [10.0, 20.2, 30.0],
            [10.0, 20.0, 30.2],
        ],
    )
    observed_origin_keys = {
        tuple(torch.round(origin * 10).to(torch.int64).tolist())
        for origin in world_origins
    }
    source_origin_keys = {
        tuple(torch.round(origin * 10).to(torch.int64).tolist())
        for origin in source_origins
    }
    assert len(observed_origin_keys) == 3
    assert observed_origin_keys < source_origin_keys


def test_native_ray_manifest_rejects_development_data(tmp_path) -> None:
    """A native-ray package that opened development data fails closed."""
    folder = str(tmp_path)
    _write_fixture(folder)
    manifest_path = os.path.join(folder, "manifest.json")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["development_data_used"] = True
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="train-only"):
        NativeRayEvidence(
            folder=folder,
            manifest_sha256=_sha256_file(manifest_path),
            sample_count=3,
            seed=11,
        )


def test_native_ray_equirect_raster_requires_two_to_one_shape(
    tmp_path,
) -> None:
    """The projector contract rejects a non-spherical raster shape."""
    folder = str(tmp_path)
    manifest_sha256 = _write_fixture(folder)

    with pytest.raises(ValueError, match="2:1"):
        NativeRayEvidence(
            folder=folder,
            manifest_sha256=manifest_sha256,
            sample_count=3,
            seed=11,
            equirect_width=7,
            equirect_height=4,
        )
