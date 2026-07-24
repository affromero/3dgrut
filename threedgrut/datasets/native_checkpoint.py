"""Validated reader for native per-image checkpoint state."""

import hashlib
from collections.abc import Mapping, Sequence
from typing import Never

import msgpack
import numpy as np
from klogr.path import path_open

NATIVE_APPEARANCE_SHAPE = (33, 33, 8)
NATIVE_DISTORTION_SHAPE = (12, 12, 2)
_APPEARANCE_BYTES = int(np.prod(NATIVE_APPEARANCE_SHAPE)) * 2
_DISTORTION_BYTES = int(np.prod(NATIVE_DISTORTION_SHAPE)) * 4


class NativeCheckpointImage:
    """Decoded native state for one source image."""

    def __init__(
        self,
        *,
        name: str,
        appearance: np.ndarray,
        distortion: np.ndarray,
        qvec: np.ndarray,
        tvec: np.ndarray,
    ) -> None:
        """Store validated native tensors for one image."""
        self.name = name
        self.appearance = appearance
        self.distortion = distortion
        self.qvec = qvec
        self.tvec = tvec


class NativeCheckpointReplay:
    """Name-keyed native checkpoint with stable source-frame ordering."""

    def __init__(
        self,
        *,
        images_by_name: dict[str, NativeCheckpointImage],
        source_image_names: Sequence[str],
    ) -> None:
        """Store native images and the authoritative dataset ordering."""
        self.images_by_name = images_by_name
        self.source_image_names = tuple(source_image_names)
        manifest = "\n".join(self.source_image_names).encode("utf-8")
        self.source_manifest_hash = hashlib.sha256(manifest).hexdigest()

    def appearance_rows(self) -> np.ndarray:
        """Return FP16 appearance rows in dataset source-frame order."""
        return np.stack(
            [
                self.images_by_name[name].appearance.reshape(-1)
                for name in self.source_image_names
            ]
        )

    def distortion_rows(self) -> np.ndarray:
        """Return FP32 distortion rows in dataset source-frame order."""
        return np.stack(
            [
                self.images_by_name[name].distortion
                for name in self.source_image_names
            ]
        )


class NativeCheckpointError(ValueError):
    """Native checkpoint data is invalid or mismatched."""


def _fail(message: str) -> Never:
    raise NativeCheckpointError(message)


def _mapping(value: object, *, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        _fail(f"Native checkpoint {label} must be a mapping.")
    return value


def _required(
    mapping: Mapping[object, object], key: str, *, label: str
) -> object:
    if key not in mapping:
        _fail(f"Native checkpoint {label} is missing {key!r}.")
    return mapping[key]


def _blob(
    mapping: Mapping[object, object],
    *,
    label: str,
    expected_bytes: int,
) -> bytes:
    value = _required(mapping, "data", label=label)
    if not isinstance(value, bytes):
        _fail(f"Native checkpoint {label}.data must be bytes.")
    if len(value) != expected_bytes:
        _fail(
            f"Native checkpoint {label}.data has {len(value)} bytes; "
            f"expected {expected_bytes}."
        )
    return value


def _vector(
    mapping: Mapping[object, object],
    *,
    key: str,
    size: int,
    label: str,
) -> np.ndarray:
    value = _required(mapping, key, label=label)
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _fail(f"Native checkpoint {label}.{key} must be numeric.")
    if vector.shape != (size,) or not np.isfinite(vector).all():
        _fail(
            f"Native checkpoint {label}.{key} must contain {size} finite "
            "values."
        )
    return vector


def _decode_image(value: object, *, index: int) -> NativeCheckpointImage:
    label = f"images[{index}]"
    image = _mapping(value, label=label)
    name = _required(image, "name", label=label)
    if not isinstance(name, str) or not name:
        _fail(f"Native checkpoint {label}.name must be non-empty.")
    appearance_blob = _blob(
        _mapping(_required(image, "appearance", label=label), label=label),
        label=f"{label}.appearance",
        expected_bytes=_APPEARANCE_BYTES,
    )
    distortion_blob = _blob(
        _mapping(_required(image, "distortion", label=label), label=label),
        label=f"{label}.distortion",
        expected_bytes=_DISTORTION_BYTES,
    )
    appearance = (
        np.frombuffer(appearance_blob, dtype="<f2")
        .copy()
        .reshape(NATIVE_APPEARANCE_SHAPE)
    )
    distortion = (
        np.frombuffer(distortion_blob, dtype="<f4")
        .copy()
        .reshape(NATIVE_DISTORTION_SHAPE)
    )
    if not np.isfinite(appearance).all():
        _fail(f"Native checkpoint {label}.appearance is non-finite.")
    if not np.isfinite(distortion).all():
        _fail(f"Native checkpoint {label}.distortion is non-finite.")
    extrinsic = _mapping(
        _required(image, "extrinsic", label=label),
        label=f"{label}.extrinsic",
    )
    qvec = _vector(
        extrinsic,
        key="qvec",
        size=4,
        label=f"{label}.extrinsic",
    )
    qvec_norm = float(np.linalg.norm(qvec))
    if abs(qvec_norm - 1.0) > 1e-3:
        _fail(
            f"Native checkpoint {label}.extrinsic.qvec norm is "
            f"{qvec_norm:.8f}; expected a unit COLMAP quaternion."
        )
    return NativeCheckpointImage(
        name=name,
        appearance=appearance,
        distortion=distortion,
        qvec=qvec,
        tvec=_vector(
            extrinsic,
            key="tvec",
            size=3,
            label=f"{label}.extrinsic",
        ),
    )


def load_native_checkpoint(
    path: str,
    *,
    source_image_names: Sequence[str],
) -> NativeCheckpointReplay:
    """Load native state and require an exact source image-name match."""
    with path_open(path, "rb") as handle:
        root = _mapping(
            msgpack.unpack(handle, raw=False, strict_map_key=False),
            label="root",
        )
    raw_images = _required(root, "images", label="root")
    if not isinstance(raw_images, list):
        _fail("Native checkpoint images must be a list.")
    images_by_name: dict[str, NativeCheckpointImage] = {}
    for index, raw_image in enumerate(raw_images):
        image = _decode_image(raw_image, index=index)
        if image.name in images_by_name:
            _fail(
                f"Native checkpoint contains duplicate image {image.name!r}."
            )
        images_by_name[image.name] = image

    ordered_names = tuple(source_image_names)
    if len(ordered_names) != len(set(ordered_names)):
        _fail("Dataset source image names contain duplicates.")
    expected = set(ordered_names)
    actual = set(images_by_name)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(
            "Native checkpoint image set does not match the dataset: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    return NativeCheckpointReplay(
        images_by_name=images_by_name,
        source_image_names=ordered_names,
    )
