"""Tests for metric depth sidecar geometry."""

import torch

from threedgrut.trainer import _predicted_axial_depth


def test_world_ray_depth_uses_preserved_camera_space_z() -> None:
    pred_dist = torch.full((2, 3, 4, 1), 10.0)
    rays_dir_world = torch.zeros((2, 3, 4, 3))
    depth_ray_z = torch.full((1, 3, 4, 1), 0.25)

    pred_depth = _predicted_axial_depth(
        pred_dist=pred_dist,
        rays_dir=rays_dir_world,
        rays_in_world_space=True,
        depth_ray_z=depth_ray_z,
    )

    assert pred_depth.shape == pred_dist.shape
    assert torch.allclose(pred_depth, torch.full_like(pred_depth, 2.5))


def test_camera_space_depth_uses_ray_z_fallback() -> None:
    pred_dist = torch.full((1, 2, 2, 1), 8.0)
    rays_dir = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]], [[0.0, 0.8, 0.6], [1.0, 0.0, 0.0]]]]
    )

    pred_depth = _predicted_axial_depth(
        pred_dist=pred_dist,
        rays_dir=rays_dir,
        rays_in_world_space=False,
        depth_ray_z=None,
    )

    expected = torch.tensor([[[[8.0], [6.4]], [[4.8], [0.0]]]])
    assert torch.allclose(pred_depth, expected)
