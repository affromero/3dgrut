"""Per-tile loss volume — voxelized post-training reconstruction error.

After training completes, re-renders a stride-sampled subset of train views,
projects per-pixel L2 loss through the predicted depth into world space, and
aggregates the loss into a sparse 3D voxel grid. Saves the resulting volume
as `loss_volume.npz`, `loss_volume_summary.json`, and a context-rich top-down
`loss_volume_topdown.png` for offline / wandb inspection.

Useful for spatial diagnosis — "where in the scene is the model undertrained?"
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import default_collate

from threedgrut.utils.logger import logger


@torch.no_grad()
def compute_loss_volume(
    model: Any,
    dataset: Any,
    *,
    voxel_size_m: float = 0.1,
    view_stride: int = 4,
    pixel_stride: int = 8,
    max_views: int = 256,
) -> dict[str, np.ndarray] | None:
    """Return a sparse loss volume aggregated across stride-sampled train views.

    Returns dict with keys:
      - `coords`     [N, 3] int32 voxel coordinates
      - `loss_sum`   [N]    float32 summed L2 loss in each voxel
      - `loss_count` [N]    int32 sample count per voxel
      - `loss_mean`  [N]    float32 mean L2 loss per voxel
      - `voxel_size_m`   (float)
      - `views_used`     (int)
      - `pixel_stride`   (int)
      - `camera_centers` [V, 3] float32 camera origins used in the volume

    Returns None on failure (missing dataset method, empty stride window).
    """
    if not hasattr(dataset, "get_gpu_batch_with_intrinsics"):
        logger.warning(
            "loss_volume: dataset has no get_gpu_batch_with_intrinsics; skipping"
        )
        return None
    n = len(dataset)
    if n <= 0:
        logger.warning("loss_volume: empty dataset; skipping")
        return None

    view_indices = list(range(0, n, max(1, view_stride)))[:max_views]
    if not view_indices:
        return None
    resolved_pixel_stride = max(1, int(pixel_stride))

    # Aggregate in float64 / int64 over all views, then convert at the end.
    voxel_to_idx: dict[tuple[int, int, int], int] = {}
    loss_sum: list[float] = []
    loss_count: list[int] = []
    camera_centers: list[np.ndarray] = []
    rendered_view_indices: list[int] = []
    views_rendered = 0

    for idx in logger.track(view_indices, description="Computing loss volume"):
        try:
            sample = dataset[idx]
            batch = default_collate([sample])
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
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
        loss = (
            (pred_rgb - gt.to(pred_rgb.device)).pow(2).mean(dim=-1)
        )  # [H, W]
        dist = outputs["pred_dist"][0]
        if dist.ndim == 3 and dist.shape[-1] == 1:
            dist = dist[..., 0]
        # Project pixels into world space: cam_pts = origin + dir * dist
        rays_ori = gpu_batch.rays_ori[0]  # [H, W, 3]
        rays_dir = gpu_batch.rays_dir[0]  # [H, W, 3] normalized
        if resolved_pixel_stride > 1:
            loss = loss[::resolved_pixel_stride, ::resolved_pixel_stride]
            dist = dist[::resolved_pixel_stride, ::resolved_pixel_stride]
            rays_ori = rays_ori[
                ::resolved_pixel_stride, ::resolved_pixel_stride
            ]
            rays_dir = rays_dir[
                ::resolved_pixel_stride, ::resolved_pixel_stride
            ]
        cam_pts = rays_ori + rays_dir * dist.unsqueeze(-1)
        transform = gpu_batch.T_to_world[0]
        if transform.shape == (4, 4):
            rotation = transform[:3, :3]
            t = transform[:3, 3]
        else:
            rotation = transform[:3, :3]
            t = (
                transform[:3, 3]
                if transform.shape[1] >= 4
                else torch.zeros(
                    3, device=transform.device, dtype=transform.dtype
                )
            )
        camera_centers.append(t.detach().float().cpu().numpy())
        rendered_view_indices.append(int(idx))
        world_pts = cam_pts @ rotation.T + t

        # Voxelize: floor(xyz / voxel_size)
        voxel = (world_pts / voxel_size_m).floor().to(torch.int32)
        voxel_flat = voxel.reshape(-1, 3).cpu().numpy()
        loss_flat = loss.reshape(-1).detach().cpu().numpy()
        # Drop non-finite
        finite_mask = np.isfinite(voxel_flat).all(axis=1) & np.isfinite(
            loss_flat
        )
        voxel_flat = voxel_flat[finite_mask]
        loss_flat = loss_flat[finite_mask]

        # Aggregate via dict (simple, robust). For >5M points consider unique+bincount.
        for coord, loss_value in zip(
            map(tuple, voxel_flat.tolist()),
            loss_flat.tolist(),
            strict=True,
        ):
            existing = voxel_to_idx.get(coord)
            if existing is None:
                voxel_to_idx[coord] = len(loss_sum)
                loss_sum.append(float(loss_value))
                loss_count.append(1)
            else:
                loss_sum[existing] += float(loss_value)
                loss_count[existing] += 1
        views_rendered += 1

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
        "views_used": np.int32(views_rendered),
        "pixel_stride": np.int32(resolved_pixel_stride),
        "camera_centers": np.array(camera_centers, dtype=np.float32),
        "view_indices": np.array(rendered_view_indices, dtype=np.int32),
    }


def summarize_loss_volume(
    volume: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Return scalar summaries suitable for wandb and JSON sidecars."""
    loss_mean = np.asarray(volume["loss_mean"], dtype=np.float64)
    loss_count = np.asarray(volume["loss_count"], dtype=np.int64)
    finite = np.isfinite(loss_mean)
    if not np.any(finite):
        return {
            "active_voxels": int(loss_mean.size),
            "sample_count": int(loss_count.sum()),
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "views_used": int(np.asarray(volume["views_used"]).item()),
            "pixel_stride": int(np.asarray(volume["pixel_stride"]).item()),
            "voxel_size_m": float(np.asarray(volume["voxel_size_m"]).item()),
        }

    finite_loss = loss_mean[finite]
    return {
        "active_voxels": int(loss_mean.size),
        "sample_count": int(loss_count.sum()),
        "mean": float(np.mean(finite_loss)),
        "p50": float(np.quantile(finite_loss, 0.50)),
        "p95": float(np.quantile(finite_loss, 0.95)),
        "p99": float(np.quantile(finite_loss, 0.99)),
        "max": float(np.max(finite_loss)),
        "views_used": int(np.asarray(volume["views_used"]).item()),
        "pixel_stride": int(np.asarray(volume["pixel_stride"]).item()),
        "voxel_size_m": float(np.asarray(volume["voxel_size_m"]).item()),
    }


def _normalize_grid(grid: np.ndarray, *, quantile: float = 0.99) -> np.ndarray:
    """Normalize a non-negative image grid with robust percentile scaling."""
    finite = np.isfinite(grid)
    positive = grid[finite & (grid > 0)]
    if positive.size == 0:
        return np.zeros_like(grid, dtype=np.float32)
    scale = float(np.quantile(positive, quantile))
    if scale <= 0.0:
        scale = float(positive.max())
    if scale <= 0.0:
        return np.zeros_like(grid, dtype=np.float32)
    return np.clip(grid / scale, 0.0, 1.0).astype(np.float32)


def _hot_colormap(values: np.ndarray) -> np.ndarray:
    """Return a black-red-yellow-white heatmap for normalized values."""
    v = np.clip(values, 0.0, 1.0)
    red = np.clip(3.0 * v, 0.0, 1.0)
    green = np.clip(3.0 * v - 1.0, 0.0, 1.0)
    blue = np.clip(3.0 * v - 2.0, 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def _grid_to_rgb_panel(grid_rgb: np.ndarray, *, scale: int) -> Image.Image:
    """Upscale an RGB grid into a panel without smoothing away hotspots."""
    image = Image.fromarray(np.flipud(grid_rgb))
    size = (max(1, image.width * scale), max(1, image.height * scale))
    return image.resize(size, Image.Resampling.NEAREST)


def _draw_camera_centers(
    panel: Image.Image,
    *,
    camera_centers: np.ndarray,
    x0: int,
    z0: int,
    voxel_size_m: float,
    scale: int,
) -> None:
    """Overlay sampled camera origins on a top-down panel."""
    if camera_centers.size == 0:
        return
    draw = ImageDraw.Draw(panel)
    height = panel.height
    radius = max(3, scale * 2)
    for center in camera_centers:
        xi = int(np.floor(float(center[0]) / voxel_size_m)) - x0
        zi = int(np.floor(float(center[2]) / voxel_size_m)) - z0
        cx = xi * scale + scale // 2
        cy = height - (zi * scale + scale // 2)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(0, 255, 255),
            width=max(1, scale),
        )


def _labeled_panel(
    *,
    image: Image.Image,
    title: str,
    subtitle: str,
) -> Image.Image:
    """Add compact labels above one loss-volume panel."""
    label_height = 36
    panel = Image.new(
        "RGB", (image.width, image.height + label_height), "black"
    )
    panel.paste(image, (0, label_height))
    draw = ImageDraw.Draw(panel)
    draw.text((6, 4), title, fill=(255, 255, 255))
    draw.text((6, 19), subtitle, fill=(180, 180, 180))
    return panel


def _occupied_axis_bounds(values: np.ndarray) -> tuple[int, int]:
    """Return robust display bounds for sparse voxel coordinates."""
    if values.size == 0:
        return 0, 0
    if values.size < 100:
        return int(values.min()), int(values.max())
    lower = int(np.floor(np.quantile(values, 0.005)))
    upper = int(np.ceil(np.quantile(values, 0.995)))
    return lower, upper


def _save_topdown_png(volume: dict[str, np.ndarray], png_path: str) -> None:
    """Save a context-rich top-down loss-volume contact sheet."""
    coords = volume["coords"]
    loss_mean = volume["loss_mean"]
    loss_count = volume["loss_count"]
    voxel_size_m = float(np.asarray(volume["voxel_size_m"]).item())
    x = coords[:, 0]
    z = coords[:, 2]
    x0, x1 = _occupied_axis_bounds(x)
    z0, z1 = _occupied_axis_bounds(z)
    x0 -= 2
    x1 += 2
    z0 -= 2
    z1 += 2
    w = max(1, x1 - x0 + 1)
    h = max(1, z1 - z0 + 1)
    loss_grid = np.zeros((h, w), dtype=np.float32)
    count_grid = np.zeros((h, w), dtype=np.float32)
    for xi, zi, lm, lc in zip(x, z, loss_mean, loss_count, strict=True):
        yy = zi - z0
        xx = xi - x0
        if yy < 0 or yy >= h or xx < 0 or xx >= w:
            continue
        loss_grid[yy, xx] = max(loss_grid[yy, xx], float(lm))
        count_grid[yy, xx] += float(lc)

    loss_norm = _normalize_grid(loss_grid)
    count_norm = _normalize_grid(np.log1p(count_grid))
    heat_rgb = _hot_colormap(loss_norm)
    coverage_rgb = np.repeat(
        (count_norm[..., None] * 180.0).astype(np.uint8), 3, axis=-1
    )
    alpha = np.clip(loss_norm[..., None] * 0.85, 0.0, 0.85)
    overlay_rgb = (
        coverage_rgb.astype(np.float32) * (1.0 - alpha)
        + heat_rgb.astype(np.float32) * alpha
    ).astype(np.uint8)

    scale = max(1, min(8, int(np.floor(900 / max(w, h)))))
    heat_panel = _grid_to_rgb_panel(heat_rgb, scale=scale)
    coverage_panel = _grid_to_rgb_panel(coverage_rgb, scale=scale)
    overlay_panel = _grid_to_rgb_panel(overlay_rgb, scale=scale)
    _draw_camera_centers(
        overlay_panel,
        camera_centers=np.asarray(
            volume.get("camera_centers", np.empty((0, 3))),
            dtype=np.float32,
        ),
        x0=x0,
        z0=z0,
        voxel_size_m=voxel_size_m,
        scale=scale,
    )

    summary = summarize_loss_volume(volume)
    subtitle = (
        f"{summary['active_voxels']} voxels, "
        f"{summary['views_used']} views, "
        f"p95={summary['p95']:.4f}, max={summary['max']:.4f}"
    )
    panels = [
        _labeled_panel(
            image=heat_panel,
            title="Residual heat",
            subtitle="p99-normalized per-voxel mean RGB L2",
        ),
        _labeled_panel(
            image=coverage_panel,
            title="Sample coverage",
            subtitle=f"log sample count, voxel={voxel_size_m:.3f}m",
        ),
        _labeled_panel(
            image=overlay_panel,
            title="Residual over coverage",
            subtitle=subtitle,
        ),
    ]
    gap = 8
    output = Image.new(
        "RGB",
        (
            sum(panel.width for panel in panels) + gap * (len(panels) - 1),
            max(panel.height for panel in panels),
        ),
        "black",
    )
    cursor = 0
    for panel in panels:
        output.paste(panel, (cursor, 0))
        cursor += panel.width + gap
    output.save(png_path)


def save_loss_volume(
    volume: dict[str, np.ndarray], output_dir: str
) -> tuple[str, str]:
    """Save the volume, summary sidecar, and top-down contact sheet."""
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, "loss_volume.npz")
    np.savez_compressed(npz_path, **volume)
    summary_path = os.path.join(output_dir, "loss_volume_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summarize_loss_volume(volume), fh, indent=2, sort_keys=True)
    png_path = os.path.join(output_dir, "loss_volume_topdown.png")
    try:
        _save_topdown_png(volume, png_path)
    except Exception as exc:
        logger.warning(f"loss_volume: top-down PNG render failed: {exc}")
        png_path = ""
    return npz_path, png_path
