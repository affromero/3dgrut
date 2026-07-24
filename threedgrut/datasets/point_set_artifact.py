import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

POINT_SET_SCHEMA_VERSIONS = frozenset((4, 5, 6))
_SEED_ARRAY_NAMES = ("positions", "sh0", "opacity")
_STATE_ARRAY_NAMES = (
    "positions",
    "sh0",
    "density_logits",
    "log_scales",
    "rotations_wxyz",
    "environment_mask",
)


@dataclass(frozen=True)
class PointSetArtifact:
    """Verified point-set seed or materialized Gaussian state."""

    positions: np.ndarray
    sh0: np.ndarray
    opacity: np.ndarray | None
    manifest: dict
    density_logits: np.ndarray | None = None
    log_scales: np.ndarray | None = None
    rotations_wxyz: np.ndarray | None = None
    environment_mask: np.ndarray | None = None

    @property
    def has_materialized_state(self) -> bool:
        """Whether this artifact carries all optimizable Gaussian fields."""
        return self.density_logits is not None

    def __iter__(self) -> Iterator[np.ndarray | dict | None]:
        """Preserve the legacy four-value unpacking contract."""
        yield self.positions
        yield self.sh0
        yield self.opacity
        yield self.manifest


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arrays(
    artifact_path: str,
    arrays: dict,
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    loaded = {}
    for name in names:
        metadata = arrays.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Point-set manifest is missing {name!r}.")
        filename = metadata.get("filename")
        if not isinstance(filename, str) or not filename.endswith(".npy"):
            raise ValueError(f"Invalid point-set filename for {name!r}.")
        array_path = os.path.join(artifact_path, filename)
        if not os.path.isfile(array_path):
            raise FileNotFoundError(
                f"Point-set array does not exist: {array_path}"
            )
        if _sha256(array_path) != metadata.get("sha256"):
            raise ValueError(f"Point-set checksum mismatch for {name!r}.")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata.get("shape"):
            raise ValueError(f"Point-set shape mismatch for {name!r}.")
        if str(array.dtype) != metadata.get("dtype"):
            raise ValueError(f"Point-set dtype mismatch for {name!r}.")
        loaded[name] = array
    return loaded


def _validate_seed_artifact(
    loaded: dict[str, np.ndarray],
    manifest: dict,
) -> PointSetArtifact:
    positions = loaded["positions"]
    sh0 = loaded["sh0"]
    opacity = loaded["opacity"]
    point_count = positions.shape[0]
    if positions.shape != (point_count, 3):
        raise ValueError("Point-set positions must have shape (N, 3).")
    if sh0.shape != (point_count, 3):
        raise ValueError("Point-set sh0 must have shape (N, 3).")
    if opacity.shape != (point_count,):
        raise ValueError("Point-set opacity must have shape (N,).")
    if positions.dtype != np.float32 or sh0.dtype != np.float32:
        raise ValueError("Point-set positions and sh0 must be float32.")
    if opacity.dtype != np.float32:
        raise ValueError("Point-set opacity must be float32.")
    count_base = manifest.get("count_base")
    environmental = manifest.get("environmental")
    if not isinstance(count_base, int) or not 0 < count_base < point_count:
        raise ValueError("Point-set manifest has an invalid base point count.")
    if not isinstance(environmental, dict):
        raise ValueError(
            "Point-set manifest is missing environmental metadata."
        )
    if environmental.get("start_index") != count_base:
        raise ValueError("Environmental points must follow all base points.")
    if environmental.get("count") != point_count - count_base:
        raise ValueError("Environmental point count does not match arrays.")
    center = environmental.get("center")
    if not isinstance(center, list) or len(center) != 3:
        raise ValueError("Environmental center must contain three values.")
    scalar_names = (
        "scene_radius",
        "near_multiplier",
        "far_multiplier",
        "xy_scale",
        "z_scale",
    )
    if any(
        not isinstance(environmental.get(name), (int, float))
        or not np.isfinite(environmental[name])
        or environmental[name] <= 0.0
        for name in scalar_names
    ):
        raise ValueError("Environmental scalar metadata must be positive.")
    return PointSetArtifact(
        positions=positions,
        sh0=sh0,
        opacity=opacity,
        manifest=manifest,
    )


def _validate_materialized_state(
    loaded: dict[str, np.ndarray],
    manifest: dict,
) -> PointSetArtifact:
    positions = loaded["positions"]
    sh0 = loaded["sh0"]
    density_logits = loaded["density_logits"]
    log_scales = loaded["log_scales"]
    rotations_wxyz = loaded["rotations_wxyz"]
    environment_mask = loaded["environment_mask"]
    point_count = positions.shape[0]
    if positions.shape != (point_count, 3):
        raise ValueError("Point-set positions must have shape (N, 3).")
    if sh0.shape != (point_count, 3):
        raise ValueError("Point-set sh0 must have shape (N, 3).")
    if density_logits.shape != (point_count,):
        raise ValueError("Point-set density_logits must have shape (N,).")
    if log_scales.shape != (point_count, 3):
        raise ValueError("Point-set log_scales must have shape (N, 3).")
    if rotations_wxyz.shape != (point_count, 4):
        raise ValueError(
            "Point-set rotations_wxyz must have shape (N, 4)."
        )
    if environment_mask.shape != (point_count,):
        raise ValueError(
            "Point-set environment_mask must have shape (N,)."
        )
    float_arrays = (
        positions,
        sh0,
        density_logits,
        log_scales,
        rotations_wxyz,
    )
    if any(array.dtype != np.float32 for array in float_arrays):
        raise ValueError("Materialized Gaussian fields must be float32.")
    if environment_mask.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise ValueError("Point-set environment_mask must be bool or uint8.")
    if not all(np.isfinite(array).all() for array in float_arrays):
        raise ValueError("Materialized Gaussian fields must be finite.")
    if not np.all(np.isin(environment_mask, (0, 1))):
        raise ValueError("Point-set environment_mask values must be 0 or 1.")
    parameterization = manifest.get("parameterization")
    expected_parameterization = {
        "density": "logit",
        "scale": "log",
        "rotation": "wxyz",
    }
    if parameterization != expected_parameterization:
        raise ValueError(
            "Materialized point-set parameterization must define logit "
            "density, log scale, and WXYZ rotation."
        )
    return PointSetArtifact(
        positions=positions,
        sh0=sh0,
        opacity=None,
        manifest=manifest,
        density_logits=density_logits,
        log_scales=log_scales,
        rotations_wxyz=rotations_wxyz,
        environment_mask=environment_mask,
    )


def load_point_set_artifact(
    artifact_path: str,
) -> PointSetArtifact:
    """Load and verify a generic Gaussian point-set artifact."""
    manifest_path = os.path.join(artifact_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Point-set manifest does not exist: {manifest_path}"
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    schema_version = manifest.get("schema_version")
    if schema_version not in POINT_SET_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported point-set schema version: "
            f"{manifest.get('schema_version')!r}"
        )

    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict):
        raise ValueError("Point-set manifest is missing array metadata.")

    if schema_version == 6:
        return _validate_materialized_state(
            _load_arrays(artifact_path, arrays, _STATE_ARRAY_NAMES),
            manifest,
        )
    return _validate_seed_artifact(
        _load_arrays(artifact_path, arrays, _SEED_ARRAY_NAMES),
        manifest,
    )
