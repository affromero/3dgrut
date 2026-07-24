"""Tests for COLMAP camera ray angle sidecars."""

import pytest
import torch
from threedgrut.datasets.dataset_colmap import ColmapDataset


class _ColmapDatasetForDepthRayZ(ColmapDataset):
    def __init__(
        self,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        pixel_coords: torch.Tensor,
    ) -> None:
        self.device = "cpu"
        self.rs_ray_injection = False
        self.native_image_scale_enabled = False
        self.worker_intrinsics = {
            5: (
                {"shutter_type": "GLOBAL"},
                rays_ori,
                rays_dir,
                "OpenCVFisheyeCameraModelParameters",
                pixel_coords,
            )
        }

    def _lazy_worker_intrinsics_cache(
        self,
    ) -> dict[
        int,
        tuple[dict[str, str], torch.Tensor, torch.Tensor, str, torch.Tensor],
    ]:
        return self.worker_intrinsics


def test_colmap_batch_includes_depth_ray_z_without_depth_gt() -> None:
    """COLMAP batches expose camera ray angle without a depth sidecar."""
    rays_ori = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    rays_dir = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0], [0.0, 0.6, 0.8], [0.0, 0.8, 0.6]],
                [[0.0, 1.0, 0.0], [0.0, -0.6, 0.8], [0.0, -0.8, 0.6]],
            ]
        ],
        dtype=torch.float32,
    )
    pixel_coords = torch.zeros((1, 2, 3, 2), dtype=torch.float32)
    dataset = _ColmapDatasetForDepthRayZ(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        pixel_coords=pixel_coords,
    )

    batch = {
        "data": torch.zeros((1, 1, 2, 3, 3), dtype=torch.uint8),
        "pose": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
        "intr": torch.tensor([5]),
        "camera_idx": torch.tensor([0]),
        "post_processing_camera_idx": torch.tensor([0]),
        "frame_idx": torch.tensor([7]),
        "source_frame_idx": torch.tensor([17]),
        "sequence_idx": torch.tensor([-1]),
        "image_path": ["cam0/frame0007.png"],
    }

    gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)

    assert gpu_batch.depth_gt is None
    assert gpu_batch.depth_ray_z is not None
    assert torch.allclose(
        gpu_batch.depth_ray_z,
        torch.abs(rays_dir[..., 2:3]),
    )
    assert gpu_batch.source_frame_idx == 17


def test_colmap_batch_requires_stable_source_frame_index() -> None:
    """Split-local frame IDs cannot silently replace source-frame IDs."""
    rays_ori = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    rays_dir = torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32)
    pixel_coords = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
    dataset = _ColmapDatasetForDepthRayZ(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        pixel_coords=pixel_coords,
    )
    batch = {
        "data": torch.zeros((1, 1, 1, 1, 3), dtype=torch.uint8),
        "pose": torch.eye(4, dtype=torch.float32).reshape(1, 1, 4, 4),
        "intr": torch.tensor([5]),
        "camera_idx": torch.tensor([0]),
        "frame_idx": torch.tensor([7]),
        "sequence_idx": torch.tensor([-1]),
        "image_path": ["cam0/frame0007.png"],
    }

    with pytest.raises(ValueError, match="require source_frame_idx"):
        dataset.get_gpu_batch_with_intrinsics(batch)
