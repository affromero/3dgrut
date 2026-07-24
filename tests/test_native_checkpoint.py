"""Tests for strict native checkpoint decoding."""

import msgpack
import numpy as np
import pytest
from klogr.path import path_open
from threedgrut.datasets.native_checkpoint import load_native_checkpoint


def _image(
    name: str,
    *,
    appearance_value: float = 0.0,
    nonfinite_appearance: bool = False,
) -> dict[str, object]:
    appearance = np.full((33, 33, 8), appearance_value, dtype="<f2")
    if nonfinite_appearance:
        appearance[0, 0, 0] = np.nan
    distortion = np.zeros((12, 12, 2), dtype="<f4")
    return {
        "name": name,
        "appearance": {"data": appearance.tobytes()},
        "distortion": {"data": distortion.tobytes()},
        "extrinsic": {
            "qvec": [1.0, 0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
        },
    }


def _write_checkpoint(path: str, images: list[dict[str, object]]) -> None:
    with path_open(path, "wb") as handle:
        msgpack.pack({"images": images}, handle)


def test_checkpoint_rows_follow_dataset_names_not_payload_order(
    tmp_path: object,
) -> None:
    """Replay tensors should follow stable dataset source-frame names."""
    path = str(tmp_path / "checkpoint.msgpack")
    _write_checkpoint(
        path,
        [
            _image("b.jpg", appearance_value=2.0),
            _image("a.jpg", appearance_value=1.0),
        ],
    )

    checkpoint = load_native_checkpoint(
        path,
        source_image_names=("a.jpg", "b.jpg"),
    )

    rows = checkpoint.appearance_rows()
    assert rows.shape == (2, 33 * 33 * 8)
    assert np.all(rows[0] == np.float16(1.0))
    assert np.all(rows[1] == np.float16(2.0))
    assert checkpoint.images_by_name["a.jpg"].qvec.tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert checkpoint.images_by_name["a.jpg"].tvec.tolist() == [
        0.0,
        0.0,
        0.0,
    ]


def test_checkpoint_rejects_image_set_mismatch(tmp_path: object) -> None:
    """Missing or extra native images should fail loudly."""
    path = str(tmp_path / "checkpoint.msgpack")
    _write_checkpoint(path, [_image("a.jpg")])

    with pytest.raises(ValueError, match="image set does not match"):
        load_native_checkpoint(
            path,
            source_image_names=("a.jpg", "b.jpg"),
        )


def test_checkpoint_rejects_duplicate_images(tmp_path: object) -> None:
    """Duplicate native names cannot define stable source-frame state."""
    path = str(tmp_path / "checkpoint.msgpack")
    _write_checkpoint(path, [_image("a.jpg"), _image("a.jpg")])

    with pytest.raises(ValueError, match="duplicate image"):
        load_native_checkpoint(path, source_image_names=("a.jpg",))


def test_checkpoint_rejects_nonfinite_appearance(tmp_path: object) -> None:
    """Non-finite native parameters should be rejected before replay."""
    path = str(tmp_path / "checkpoint.msgpack")
    _write_checkpoint(path, [_image("a.jpg", nonfinite_appearance=True)])

    with pytest.raises(ValueError, match="appearance is non-finite"):
        load_native_checkpoint(path, source_image_names=("a.jpg",))
