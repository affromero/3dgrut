"""Tests for the generic checksummed point-set artifact loader."""

import hashlib
import json
import os

import numpy as np
import pytest
from threedgrut.datasets.point_set_artifact import load_point_set_artifact


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()


def _write_artifact(root: str, *, schema_version: int = 5) -> None:
    arrays = {
        "positions": np.asarray(
            ((1.0, 2.0, 3.0), (0.0, 0.0, 4.0)), dtype=np.float32
        ),
        "sh0": np.zeros((2, 3), dtype=np.float32),
        "opacity": np.asarray((0.25, 0.1), dtype=np.float32),
    }
    metadata = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        path = os.path.join(root, filename)
        np.save(path, array, allow_pickle=False)
        metadata[name] = {
            "filename": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(path),
        }
    with open(
        os.path.join(root, "manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": schema_version,
                "count_base": 1,
                "environmental": {
                    "start_index": 1,
                    "count": 1,
                    "center": [0.0, 0.0, 0.0],
                    "scene_radius": 1.0,
                    "near_multiplier": 2.5,
                    "far_multiplier": 10.0,
                    "xy_scale": 6.0,
                    "z_scale": 0.2,
                },
                "arrays": metadata,
            },
            handle,
        )


def _write_materialized_artifact(root: str) -> None:
    arrays = {
        "positions": np.asarray(
            ((1.0, 2.0, 3.0), (0.0, 0.0, 4.0)), dtype=np.float32
        ),
        "sh0": np.zeros((2, 3), dtype=np.float32),
        "density_logits": np.asarray((-1.0, -2.0), dtype=np.float32),
        "log_scales": np.full((2, 3), -3.0, dtype=np.float32),
        "rotations_wxyz": np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            dtype=np.float32,
        ),
        "environment_mask": np.asarray((0, 1), dtype=np.uint8),
    }
    metadata = {}
    for name, array in arrays.items():
        filename = f"{name}.npy"
        path = os.path.join(root, filename)
        np.save(path, array, allow_pickle=False)
        metadata[name] = {
            "filename": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(path),
        }
    with open(
        os.path.join(root, "manifest.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "schema_version": 6,
                "parameterization": {
                    "density": "logit",
                    "scale": "log",
                    "rotation": "wxyz",
                },
                "arrays": metadata,
            },
            handle,
        )


@pytest.mark.parametrize("schema_version", (4, 5))
def test_loader_preserves_values_and_order(
    tmp_path: object, schema_version: int
) -> None:
    """Verified arrays should load without dtype or ordering changes."""
    root = str(tmp_path)
    _write_artifact(root, schema_version=schema_version)

    positions, sh0, opacity, manifest = load_point_set_artifact(root)

    np.testing.assert_array_equal(
        positions, ((1.0, 2.0, 3.0), (0.0, 0.0, 4.0))
    )
    np.testing.assert_array_equal(sh0, np.zeros((2, 3)))
    np.testing.assert_array_equal(
        opacity, np.asarray((0.25, 0.1), dtype=np.float32)
    )
    assert manifest["schema_version"] == schema_version


@pytest.mark.parametrize("schema_version", (2, 3))
def test_loader_rejects_stale_native_initialization_schema(
    tmp_path: object, schema_version: int
) -> None:
    """Known-wrong environmental shells must be regenerated."""
    root = str(tmp_path)
    _write_artifact(root, schema_version=schema_version)

    with pytest.raises(ValueError, match="Unsupported point-set schema"):
        load_point_set_artifact(root)


def test_loader_rejects_checksum_mismatch(tmp_path: object) -> None:
    """Corrupted arrays should fail before model initialization."""
    root = str(tmp_path)
    _write_artifact(root)
    opacity_path = os.path.join(root, "opacity.npy")
    with open(opacity_path, "r+b") as handle:
        handle.seek(-1, 2)
        byte = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes((byte[0] ^ 1,)))

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_point_set_artifact(root)


def test_loader_preserves_materialized_gaussian_state(tmp_path: object) -> None:
    """Schema six must preserve raw optimization parameters exactly."""
    root = str(tmp_path)
    _write_materialized_artifact(root)

    artifact = load_point_set_artifact(root)

    assert artifact.has_materialized_state
    assert artifact.opacity is None
    np.testing.assert_array_equal(
        artifact.density_logits, np.asarray((-1.0, -2.0), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        artifact.log_scales, np.full((2, 3), -3.0, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        artifact.rotations_wxyz,
        np.asarray(
            ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        artifact.environment_mask, np.asarray((0, 1), dtype=np.uint8)
    )
