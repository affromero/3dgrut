"""Behavioral checks for train-only COLMAP materialization contracts."""

from pathlib import Path

import pycolmap
import pytest

from threedgrut.datasets.train_only_colmap import (
    validate_fixed_camera_metadata,
    validate_registration,
)


def _write_model(path: Path, *, tvec_x: float = 0.0) -> None:
    path.mkdir()
    (path / "cameras.txt").write_text("1 PINHOLE 8 8 4 4 4 4\n", encoding="utf-8")
    (path / "images.txt").write_text(
        f"1 1 0 0 0 {tvec_x} 0 0 1 a.jpg\n\n",
        encoding="utf-8",
    )
    (path / "points3D.txt").write_text("", encoding="utf-8")


def test_registered_training_views_are_an_exact_strict_subset() -> None:
    """A sorted strict fold passes without permitting held-out names."""
    validate_registration(
        source_names=["a.jpg", "b.jpg", "c.jpg"],
        all_names=["a.jpg", "b.jpg", "c.jpg"],
        training_names=["a.jpg", "c.jpg"],
    )


@pytest.mark.parametrize(
    ("all_names", "training_names", "message"),
    [
        (["a.jpg", "c.jpg"], ["a.jpg"], "all-view list"),
        (
            ["a.jpg", "b.jpg", "c.jpg"],
            ["a.jpg", "missing.jpg"],
            "strict subset",
        ),
        (
            ["a.jpg", "b.jpg", "c.jpg"],
            ["a.jpg", "b.jpg", "c.jpg"],
            "strict subset",
        ),
    ],
)
def test_registered_training_views_reject_leaks(all_names: list[str], training_names: list[str], message: str) -> None:
    """Incomplete universes, missing names, and all-view training fail."""
    with pytest.raises(ValueError, match=message):
        validate_registration(
            source_names=["a.jpg", "b.jpg", "c.jpg"],
            all_names=all_names,
            training_names=training_names,
        )


def test_fixed_camera_metadata_accepts_identical_public_pose(
    tmp_path: Path,
) -> None:
    """The manifest digest covers IDs, intrinsics, qvecs, and tvecs."""
    source_path = tmp_path / "source"
    output_path = tmp_path / "output"
    _write_model(source_path)
    _write_model(output_path)
    source_sha, output_sha = validate_fixed_camera_metadata(
        source=pycolmap.Reconstruction(str(source_path)),
        output=pycolmap.Reconstruction(str(output_path)),
        training_names=["a.jpg"],
    )
    assert source_sha == output_sha


def test_fixed_camera_metadata_rejects_pose_change(tmp_path: Path) -> None:
    """A triangulator may not alter a fixed public camera pose."""
    source_path = tmp_path / "source"
    output_path = tmp_path / "output"
    _write_model(source_path)
    _write_model(output_path, tvec_x=0.01)
    with pytest.raises(ValueError, match="changed field tvec"):
        validate_fixed_camera_metadata(
            source=pycolmap.Reconstruction(str(source_path)),
            output=pycolmap.Reconstruction(str(output_path)),
            training_names=["a.jpg"],
        )
