"""Per-pixel residual visualization helpers.

Used by the live GUIs to display `|render - gt|` heatmaps sampled against
the nearest training-view GT image. Available only when a training dataset
is present (training-time GUIs); the post-training playground (`ps_gui`)
has no dataset and so does not support residual mode.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch


_GT_CACHE: "OrderedDict[tuple[int, int], torch.Tensor]" = OrderedDict()
_GT_CACHE_MAX = 16


def _camera_center_from_c2w(c2w: np.ndarray) -> np.ndarray:
    """Extract camera center [3] from a 3x4 or 4x4 camera-to-world matrix."""
    return c2w[:3, 3]


def find_closest_train_camera(current_c2w: np.ndarray, train_dataset: Any) -> tuple[int, float] | None:
    """Return (idx, distance) of the train pose with the closest camera center.

    Returns None when the dataset doesn't expose `get_poses()` or has no poses.
    """
    if not hasattr(train_dataset, "get_poses"):
        return None
    poses = train_dataset.get_poses()
    if poses is None:
        return None
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[0] == 0 or poses.shape[1] < 3 or poses.shape[2] < 4:
        return None
    centers = poses[:, :3, 3]
    current_center = np.asarray(current_c2w)[:3, 3]
    distances = np.linalg.norm(centers - current_center[None, :], axis=1)
    idx = int(np.argmin(distances))
    return idx, float(distances[idx])


def load_gt_for_train_view(train_dataset: Any, idx: int) -> torch.Tensor | None:
    """Return GT RGB tensor [H, W, 3] in [0, 1] for train view `idx`, cached.

    Tries common dataset return shapes — Colmap-style uint8 `data` field,
    plain `image`/`rgb`/`rgb_gt`. Returns None when the format is unknown.
    """
    key = (id(train_dataset), idx)
    cached = _GT_CACHE.get(key)
    if cached is not None:
        _GT_CACHE.move_to_end(key)
        return cached

    try:
        sample = train_dataset[idx]
    except Exception:
        return None

    img: torch.Tensor | np.ndarray | None = None
    if isinstance(sample, dict):
        for k in ("data", "rgb_gt", "image", "rgb"):
            if k in sample:
                img = sample[k]
                break
    else:
        img = getattr(sample, "rgb_gt", None)
    if img is None:
        return None

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    if not isinstance(img, torch.Tensor):
        return None

    # Collapse a leading batch dim if present.
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    if img.ndim != 3:
        return None

    # [3, H, W] -> [H, W, 3]
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img = img.permute(1, 2, 0)
    if img.shape[-1] != 3:
        return None

    img = img.float()
    if img.max() > 1.5:
        img = img / 255.0
    img = img.clamp(0.0, 1.0).contiguous().cpu()

    while len(_GT_CACHE) >= _GT_CACHE_MAX:
        _GT_CACHE.popitem(last=False)
    _GT_CACHE[key] = img
    return img


def compute_residual_heatmap(rendered_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
    """Return per-pixel `|rendered - gt|.mean(-1)` aligned to `rendered_rgb` resolution.

    Both inputs are `[H, W, 3]` in `[0, 1]`. GT is bilinearly resized to match
    `rendered_rgb` if its resolution differs.
    """
    device = rendered_rgb.device
    if rendered_rgb.shape[:2] != gt_rgb.shape[:2]:
        gt_resized = torch.nn.functional.interpolate(
            gt_rgb.permute(2, 0, 1).unsqueeze(0).to(device),
            size=(rendered_rgb.shape[0], rendered_rgb.shape[1]),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
    else:
        gt_resized = gt_rgb.to(device)
    return (rendered_rgb - gt_resized).abs().mean(dim=-1)
