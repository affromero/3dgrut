"""Tests for resolving native per-frame B2G depth."""

import os

from threedgrut.datasets.dataset_colmap import ColmapDataset


def _dataset(dataset_root: str, depth_folder: str) -> ColmapDataset:
    dataset = object.__new__(ColmapDataset)
    dataset.path = dataset_root
    dataset.depth_folder = depth_folder
    return dataset


def test_flat_camera_frame_name_resolves_native_depth(
    tmp_path: os.PathLike[str],
) -> None:
    """Flat B2G names resolve their structured native-depth frame."""
    dataset_root = os.fspath(tmp_path)
    depth_root = os.path.join(dataset_root, "native_depth")
    expected = os.path.join(depth_root, "front", "depth", "0007.npy")
    os.makedirs(os.path.dirname(expected))
    with open(expected, "wb"):
        pass

    resolved = _dataset(dataset_root, depth_root).resolve_depth_path(
        "front_0007.png"
    )

    assert resolved == expected


def test_flat_sidecar_takes_precedence_over_native_depth(
    tmp_path: os.PathLike[str],
) -> None:
    """A direct sidecar wins when both supported layouts exist."""
    dataset_root = os.fspath(tmp_path)
    depth_root = os.path.join(dataset_root, "native_depth")
    direct = os.path.join(depth_root, "front_0007.npy")
    structured = os.path.join(depth_root, "front", "depth", "0007.npy")
    os.makedirs(os.path.dirname(structured))
    for depth_path in (direct, structured):
        with open(depth_path, "wb"):
            pass

    resolved = _dataset(dataset_root, depth_root).resolve_depth_path(
        "front_0007.png"
    )

    assert resolved == direct


def test_arbitrary_flat_name_does_not_guess_native_depth(
    tmp_path: os.PathLike[str],
) -> None:
    """Names without a camera and numeric frame keep legacy resolution."""
    dataset_root = os.fspath(tmp_path)
    depth_root = os.path.join(dataset_root, "native_depth")

    resolved = _dataset(dataset_root, depth_root).resolve_depth_path(
        "lobby.png"
    )

    assert resolved == os.path.join(depth_root, "lobby.npy")
