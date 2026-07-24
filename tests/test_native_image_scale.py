"""Behavior tests for the visibility-adaptive image-resolution schedule."""

import numpy as np
import pytest
import torch
from ncore.data import OpenCVPinholeCameraModelParameters, ShutterType
from omegaconf import OmegaConf
from PIL import Image
from threedgrut import datasets
from threedgrut.datasets import dataset_colmap
from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.datasets.native_image_replay import NativeImageSelection
from threedgrut.trainer import Trainer3DGRUT


def make_scale_dataset(
    scale: float = 0.05,
    factors: tuple[float, ...] = (),
    split: str = "train",
) -> ColmapDataset:
    """Build the minimal dataset state needed by scale helpers."""
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.split = split
    dataset.native_image_scale_enabled = True
    dataset.native_image_scale = scale
    dataset.native_image_min_size = 160
    dataset.native_image_max_size = 5120
    dataset.native_image_physical_camera_factors = factors
    dataset.image_horizontal_flip = False
    dataset._worker_gpu_cache = {}
    dataset.device = "cpu"
    dataset.ray_jitter = None
    return dataset


def test_native_image_size_matches_native_minimum_clamp() -> None:
    """Floor both camera families at the recovered 160-pixel minimum."""
    dataset = make_scale_dataset()

    assert dataset.native_image_size(1024, 1024) == (160, 160, 0.15625)
    width, height, effective_scale = dataset.native_image_size(3200, 3061)

    assert (width, height) == (167, 160)
    assert effective_scale == pytest.approx(160 / 3061)


def test_native_image_size_applies_maximum_before_minimum() -> None:
    """Apply the native maximum-size clamp before the minimum floor."""
    dataset = make_scale_dataset(scale=2.0)
    dataset.native_image_max_size = 512

    assert dataset.native_image_size(1024, 768) == (512, 384, 0.5)


def test_native_image_size_applies_physical_camera_factor_before_clamps() -> (
    None
):
    """DeviceHD's recovered two-thirds scale composes with dynamic scale."""
    dataset = make_scale_dataset(
        scale=1.0,
        factors=(1.0, 1.0, 0.6666667, 0.6666667),
    )
    dataset.native_image_min_size = 1

    normal = dataset.native_image_size(
        3200,
        3061,
        physical_camera_index=0,
    )
    device_hd = dataset.native_image_size(
        3200,
        3061,
        physical_camera_index=2,
    )

    assert normal == (3200, 3061, 1.0)
    assert device_hd[:2] == (2133, 2041)
    assert device_hd[2] == pytest.approx(0.6666667)


def test_native_validation_preserves_physical_camera_resolution() -> None:
    """Keep native camera factors while pinning validation globally to one."""
    dataset = make_scale_dataset(
        scale=1.0,
        factors=(1.0, 1.0, 0.6666667, 0.6666667),
        split="val",
    )
    dataset.native_image_min_size = 160

    dataset._validate_native_image_scale()

    assert dataset.native_image_size(
        3200,
        3065,
        physical_camera_index=2,
    )[:2] == (2133, 2043)


def test_colmap_factories_pin_native_validation_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire native camera factors into validation and render datasets."""
    calls: list[dict[str, object]] = []

    class FakeColmapDataset:
        def __init__(self, _path: str, **kwargs: object) -> None:
            self.kwargs = kwargs
            calls.append(kwargs)

    monkeypatch.setattr(datasets, "ColmapDataset", FakeColmapDataset)
    conf = OmegaConf.create(
        {
            "path": "/dataset",
            "dataset": {
                "load_exif": False,
                "downsample_factor": 1,
                "test_split_interval": 0,
                "native_image_scale": {
                    "enabled": True,
                    "start": 0.05,
                    "min_size": 160,
                    "max_size": 4608,
                    "physical_camera_factors": [1.0, 0.6666667],
                },
            },
            "render": {"method": "3dgut"},
            "post_processing": {},
        }
    )

    train_dataset, val_dataset = datasets.make("colmap", conf, None)
    test_dataset = datasets.make_test("colmap", conf)

    assert train_dataset.kwargs["native_image_scale"] == 0.05
    for dataset in (val_dataset, test_dataset):
        assert dataset.kwargs["native_image_scale_enabled"] is True
        assert dataset.kwargs["native_image_scale"] == 1.0
        assert tuple(
            dataset.kwargs["native_image_physical_camera_factors"]
        ) == (1.0, 0.6666667)
    assert len(calls) == 3


def test_colmap_mask_resolution_accepts_stem_matched_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve native PNG masks for COLMAP image names ending in JPG."""
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.path = "/dataset"
    dataset.downsample_factor = 1
    expected = "/dataset/masks/map_0/camera_1/frame.png"
    monkeypatch.setattr(
        dataset_colmap.os.path,
        "exists",
        lambda path: path == expected,
    )

    resolved = dataset.resolve_mask_path(
        "/dataset/images/map_0/camera_1/frame.jpg",
        "map_0/camera_1/frame.jpg",
    )

    assert resolved == expected


def test_native_image_factors_validate_mapping_and_lookup() -> None:
    """Positional factors must exactly match the physical-camera mapping."""
    dataset = make_scale_dataset(factors=(1.0, 0.6666667))
    dataset._post_processing_camera_key_to_idx = {"camera_0": 0}

    with pytest.raises(ValueError, match="one value per physical camera"):
        dataset._validate_native_physical_camera_factors()

    dataset._post_processing_camera_key_to_idx["camera_1"] = 1
    dataset._validate_native_physical_camera_factors()
    for invalid_index in (None, -1, 2):
        with pytest.raises(ValueError, match="physical-camera"):
            dataset.native_image_size(
                1024,
                1024,
                physical_camera_index=invalid_index,
            )


def test_native_image_factors_must_be_finite_and_positive() -> None:
    """Malformed immutable source-camera factors fail before loading data."""
    dataset = make_scale_dataset(factors=(1.0, float("nan")))

    with pytest.raises(ValueError, match="finite and positive"):
        dataset._validate_native_image_scale()


def test_native_scaled_pinhole_intrinsics_use_rounded_axis_ratios() -> None:
    """Scale intrinsics by the rounded output dimensions per axis."""
    dataset = make_scale_dataset()
    params = OpenCVPinholeCameraModelParameters(
        resolution=np.array([3200, 3061], dtype=np.uint64),
        shutter_type=ShutterType.GLOBAL,
        principal_point=np.array([1552.35, 1754.39], dtype=np.float32),
        focal_length=np.array([1914.64, 1918.97], dtype=np.float32),
        radial_coeffs=np.zeros(6, dtype=np.float32),
        tangential_coeffs=np.zeros(2, dtype=np.float32),
        thin_prism_coeffs=np.zeros(4, dtype=np.float32),
    ).to_dict()
    dataset.intrinsics = {
        6: (params, None, None, "OpenCVPinholeCameraModelParameters", None)
    }
    scale = 160 / 3061

    scaled, _origins, directions, _name, pixels = (
        dataset._native_scaled_intrinsics(6, scale, 167, 160)
    )

    np.testing.assert_allclose(
        scaled["focal_length"],
        np.asarray(params["focal_length"]) * [167 / 3200, 160 / 3061],
    )
    np.testing.assert_allclose(
        scaled["principal_point"],
        np.asarray(params["principal_point"]) * [167 / 3200, 160 / 3061],
    )
    np.testing.assert_array_equal(scaled["resolution"], [167, 160])
    assert directions.shape == (1, 160, 167, 3)
    assert pixels.shape[-3:-1] == (160, 167)


def test_native_getitem_resizes_rgb_linearly_and_mask_with_nearest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trace selections override global scale with native image dimensions."""
    dataset = make_scale_dataset(scale=1.0, factors=(1.0, 1.0))
    dataset.native_image_min_size = 1
    dataset.image_paths = ["frame_0042.png"]
    dataset.mask_paths = ["frame_0042_mask.png"]
    dataset.sky_mask_paths = None
    dataset.depth_paths = None
    dataset.poses = [np.eye(4, dtype=np.float32)]
    dataset.poses_end = None
    dataset.source_frame_indices = [17]
    dataset.exif_exposures = None
    dataset.get_intrinsics_idx = lambda _idx: 0
    dataset.get_camera_idx = lambda _idx: 0
    dataset.get_post_processing_camera_idx = lambda _idx: 1
    rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = np.asarray(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
        ],
        dtype=np.uint8,
    )
    monkeypatch.setattr(dataset_colmap, "_read_rgb_image_array", lambda _: rgb)
    monkeypatch.setattr(dataset_colmap.os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        dataset_colmap.Image,
        "open",
        lambda _: Image.fromarray(mask),
    )

    sample = dataset[
        NativeImageSelection(
            ordinal=0,
            priority=11,
            subindex=0,
            width=2,
            height=2,
            sequence=1,
        )
    ]

    expected_rgb = np.asarray(
        [
            [[8, 9, 10], [14, 15, 16]],
            [[32, 33, 34], [38, 39, 40]],
        ],
        dtype=np.uint8,
    )
    expected_mask = np.asarray([[0, 2], [8, 10]], dtype=np.uint8)
    torch.testing.assert_close(
        sample["data"], torch.from_numpy(expected_rgb)[None]
    )
    torch.testing.assert_close(
        sample["mask"], torch.from_numpy(expected_mask)[None, ..., None]
    )
    assert sample["sequence_idx"] == 42
    assert sample["source_frame_idx"] == 17
    assert sample["native_image_scale"] == 0.5
    assert sample["native_replay_sequence"] == 1
    assert sample["native_replay_subindex"] == 0


def test_native_getitem_flips_pixel_aligned_supervision_horizontally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Visibility-adaptive replay mirrors RGB and masks after native resizing."""
    dataset = make_scale_dataset(scale=1.0, factors=(1.0, 1.0))
    dataset.native_image_min_size = 1
    dataset.image_horizontal_flip = True
    dataset.image_paths = ["frame_0042.png"]
    dataset.mask_paths = ["frame_0042_mask.png"]
    dataset.sky_mask_paths = None
    dataset.depth_paths = None
    dataset.poses = [np.eye(4, dtype=np.float32)]
    dataset.poses_end = None
    dataset.source_frame_indices = [17]
    dataset.exif_exposures = None
    dataset.get_intrinsics_idx = lambda _idx: 0
    dataset.get_camera_idx = lambda _idx: 0
    dataset.get_post_processing_camera_idx = lambda _idx: 1
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    mask = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    monkeypatch.setattr(dataset_colmap, "_read_rgb_image_array", lambda _: rgb)
    monkeypatch.setattr(dataset_colmap.os.path, "exists", lambda _: True)
    monkeypatch.setattr(
        dataset_colmap.Image,
        "open",
        lambda _: Image.fromarray(mask),
    )

    sample = dataset[0]

    torch.testing.assert_close(
        sample["data"],
        torch.from_numpy(np.ascontiguousarray(np.fliplr(rgb)))[None],
    )
    torch.testing.assert_close(
        sample["mask"],
        torch.from_numpy(np.ascontiguousarray(np.fliplr(mask)))[None, ..., None],
    )


def test_native_gpu_batch_carries_effective_image_scale() -> None:
    """Loss and gradient scaling consume the same scale as resized intrinsics."""
    dataset = make_scale_dataset()
    dataset.rs_ray_injection = False
    rays = torch.zeros((1, 2, 2, 3))
    rays[..., 2] = 1.0
    pixels = torch.zeros((1, 2, 2, 2))
    dataset._native_scaled_intrinsics = (
        lambda _intr, _scale, _width, _height: (
            {"shutter_type": ShutterType.GLOBAL.name},
            torch.zeros_like(rays),
            rays,
            "OpenCVPinholeCameraModelParameters",
            pixels,
        )
    )
    batch = {
        "data": torch.full((1, 1, 2, 2, 3), 255, dtype=torch.uint8),
        "pose": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
        "intr": torch.tensor([0]),
        "camera_idx": torch.tensor([0]),
        "post_processing_camera_idx": torch.tensor([1]),
        "frame_idx": torch.tensor([2]),
        "source_frame_idx": torch.tensor([7]),
        "sequence_idx": torch.tensor([42]),
        "image_path": ["camera_0/frame_0042.jpg"],
        "native_image_scale": torch.tensor([0.8]),
        "image_width": torch.tensor([2]),
        "image_height": torch.tensor([2]),
    }

    gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)

    assert gpu_batch.native_image_scale == pytest.approx(0.8)
    assert gpu_batch.post_processing_camera_idx == 1
    assert gpu_batch.source_frame_idx == 7


def test_native_scale_schedule_changes_only_at_native_boundaries() -> None:
    """Compute the recovered linear scale at 3000-step boundaries."""
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    conf = OmegaConf.create(
        {
            "n_iterations": 99799,
            "dataset": {
                "type": "colmap",
                "native_image_scale": {
                    "enabled": True,
                    "start": 0.05,
                    "end": 0.95,
                    "interval": 3000,
                },
            },
        }
    )

    assert trainer._native_image_scale_for_step(conf, 0) == 0.05
    assert trainer._native_image_scale_for_step(conf, 3000) == pytest.approx(
        0.05 + 0.9 * 3000 / 99799
    )
    assert trainer._native_image_scale_for_step(conf, 5999) == pytest.approx(
        0.05 + 0.9 * 3000 / 99799
    )
    assert trainer._native_image_scale_for_step(conf, 99000) == pytest.approx(
        0.05 + 0.9 * 99000 / 99799
    )


def test_native_scale_replacement_shuts_down_prefetched_loader_first() -> None:
    """Retire old workers before exposing a loader at the new scale."""
    events: list[str] = []

    class FakeDataset:
        def set_native_image_scale(self, scale: float) -> bool:
            events.append(f"scale:{scale}")
            return True

    class FakeLoader:
        def shutdown(self) -> None:
            events.append("shutdown")

    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.train_dataset = FakeDataset()
    trainer.train_dataloader = FakeLoader()
    trainer.global_step = 3000

    def make_loader(_conf: object) -> FakeLoader:
        events.append("replace")
        return FakeLoader()

    trainer._make_train_dataloader = make_loader

    trainer._replace_train_dataloader_for_native_scale(
        OmegaConf.create({}), 0.077
    )

    assert events == ["scale:0.077", "shutdown", "replace"]
