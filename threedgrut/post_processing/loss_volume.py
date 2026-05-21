"""Per-tile loss volume — voxelized post-training reconstruction error.

After training completes, re-renders a stride-sampled subset of train views,
projects per-pixel L2 loss through the predicted depth into world space, and
aggregates the loss into a sparse 3D voxel grid. Saves the resulting volume
as `loss_volume.npz` and a top-down `loss_volume_topdown.png` for offline /
wandb inspection.

Useful for spatial diagnosis — "where in the scene is the model undertrained?"
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image

from threedgrut.utils.logger import logger


@torch.no_grad()
def compute_loss_volume(
    model: Any,
    dataset: Any,
    *,
    voxel_size_m: float = 0.1,
    view_stride: int = 4,
    max_views: int = 256,
) -> dict[str, np.ndarray] | None:
    """Return a sparse loss volume aggregated across stride-sampled train views.

    Returns dict with keys:
      - `coords`     [N, 3] int32 voxel coordinates
      - `loss_sum`   [N]    float32 summed L2 loss in each voxel
      - `loss_count` [N]    int32 sample count per voxel
      - `loss_mean`  [N]    float32 mean L2 loss per voxel
      - `voxel_size_m` (float)
      - `views_used` (int)

    Returns None on failure (missing dataset method, empty stride window).
    """
    if not hasattr(dataset, "get_gpu_batch_with_intrinsics"):
        logger.warning("loss_volume: dataset has no get_gpu_batch_with_intrinsics; skipping")
        return None
    n = len(dataset)
    if n <= 0:
        logger.warning("loss_volume: empty dataset; skipping")
        return None

    view_indices = list(range(0, n, max(1, view_stride)))[:max_views]
    if not view_indices:
        return None

    # Aggregate in float64 / int64 over all views, then convert at the end.
    voxel_to_idx: dict[tuple[int, int, int], int] = {}
    loss_sum: list[float] = []
    loss_count: list[int] = []

    for idx in logger.track(view_indices, description="Computing loss volume"):
        try:
            sample = dataset[idx]
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(sample)
            outputs = model(gpu_batch, train=False)
        except Exception as exc:
            logger.warning(f"loss_volume: view {idx} render failed: {exc}")
            continue

        pred_rgb = outputs["pred_rgb"][0]
        if gpu_batch.rgb_gt is None:
            continue
        gt = gpu_batch.rgb_gt[0]
        if gt.shape[:2] != pred_rgb.shape[:2]:
            gt = torch.nn.functional.interpolate(
                gt.permute(2, 0, 1).unsqueeze(0),
                size=(pred_rgb.shape[0], pred_rgb.shape[1]),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0)
        loss = (pred_rgb - gt.to(pred_rgb.device)).pow(2).mean(dim=-1)  # [H, W]
        dist = outputs["pred_dist"][0]
        if dist.ndim == 3 and dist.shape[-1] == 1:
            dist = dist[..., 0]
        # Project pixels into world space: cam_pts = origin + dir * dist
        rays_ori = gpu_batch.rays_ori[0]  # [H, W, 3]
        rays_dir = gpu_batch.rays_dir[0]  # [H, W, 3] normalized
        cam_pts = rays_ori + rays_dir * dist.unsqueeze(-1)
        T = gpu_batch.T_to_world[0]
        if T.shape == (4, 4):
            R = T[:3, :3]
            t = T[:3, 3]
        else:
            R = T[:3, :3]
            t = T[:3, 3] if T.shape[1] >= 4 else torch.zeros(3, device=T.device, dtype=T.dtype)
        world_pts = cam_pts @ R.T + t

        # Voxelize: floor(xyz / voxel_size)
        voxel = (world_pts / voxel_size_m).floor().to(torch.int32)
        voxel_flat = voxel.reshape(-1, 3).cpu().numpy()
        loss_flat = loss.reshape(-1).detach().cpu().numpy()
        # Drop non-finite
        finite_mask = np.isfinite(voxel_flat).all(axis=1) & np.isfinite(loss_flat)
        voxel_flat = voxel_flat[finite_mask]
        loss_flat = loss_flat[finite_mask]

        # Aggregate via dict (simple, robust). For >5M points consider unique+bincount.
        for c, l in zip(map(tuple, voxel_flat.tolist()), loss_flat.tolist()):
            existing = voxel_to_idx.get(c)
            if existing is None:
                voxel_to_idx[c] = len(loss_sum)
                loss_sum.append(float(l))
                loss_count.append(1)
            else:
                loss_sum[existing] += float(l)
                loss_count[existing] += 1

    if not voxel_to_idx:
        return None

    coords = np.array(list(voxel_to_idx.keys()), dtype=np.int32)
    loss_sum_arr = np.array(loss_sum, dtype=np.float32)
    loss_count_arr = np.array(loss_count, dtype=np.int32)
    loss_mean_arr = loss_sum_arr / np.maximum(loss_count_arr, 1)
    return {
        "coords": coords,
        "loss_sum": loss_sum_arr,
        "loss_count": loss_count_arr,
        "loss_mean": loss_mean_arr,
        "voxel_size_m": np.float32(voxel_size_m),
        "views_used": np.int32(len(view_indices)),
    }


def save_loss_volume(volume: dict[str, np.ndarray], output_dir: str) -> tuple[str, str]:
    """Save the volume as npz + a top-down max-projection PNG. Returns paths."""
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "loss_volume.npz")
    np.savez_compressed(npz_path, **volume)
    png_path = os.path.join(output_dir, "loss_volume_topdown.png")
    try:
        coords = volume["coords"]
        loss_mean = volume["loss_mean"]
        # Top-down: project along Y axis (3DGRUT uses -y-up convention)
        x = coords[:, 0]
        z = coords[:, 2]
        x0, x1 = int(x.min()), int(x.max())
        z0, z1 = int(z.min()), int(z.max())
        w = max(1, x1 - x0 + 1)
        h = max(1, z1 - z0 + 1)
        img = np.zeros((h, w), dtype=np.float32)
        for xi, zi, lm in zip(x, z, loss_mean):
            img[zi - z0, xi - x0] = max(img[zi - z0, xi - x0], float(lm))
        # Normalize + colormap
        if img.max() > 0:
            img = img / img.max()
        try:
            from matplotlib import cm
            rgba = cm.hot(img)
            img_rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        except ImportError:
            img_rgb = np.stack([img * 255, img * 128, np.zeros_like(img)], axis=-1).astype(np.uint8)
        Image.fromarray(img_rgb).save(png_path)
    except Exception as exc:
        logger.warning(f"loss_volume: top-down PNG render failed: {exc}")
        png_path = ""
    return npz_path, png_path
