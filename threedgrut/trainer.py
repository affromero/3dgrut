# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import wandb

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.datasets.utils import DEFAULT_DEVICE, MultiEpochsDataLoader, PointCloud
from threedgrut.model.camera_residual import CameraResidual
from threedgrut.model.losses import ssim
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers import SelectiveAdam
from threedgrut.post_processing import LuminanceAffine
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, create_summary_writer, jet_map
from threedgrut.utils.render import apply_post_processing
from threedgrut.utils.timer import CudaTimer


def _robust_jet_map(
    value_map: torch.Tensor,
    validity_mask: torch.Tensor | None = None,
    *,
    quantile: float = 0.95,
) -> torch.Tensor:
    """Colorize a scalar map with per-frame robust scaling."""
    finite_mask = torch.isfinite(value_map)
    if validity_mask is not None:
        finite_mask = finite_mask & (validity_mask > 0.5)

    valid_values = value_map[finite_mask]
    if valid_values.numel() == 0:
        max_value = torch.tensor(1.0, device=value_map.device, dtype=value_map.dtype)
    else:
        max_value = torch.quantile(valid_values, quantile).clamp_min(1e-6)
    return jet_map(value_map, max_value)


def _tensor_to_bgr_image(image: torch.Tensor) -> np.ndarray:
    """Convert an HWC RGB float tensor to a BGR uint8 image."""
    rgb = (
        image.detach()
        .clip(0, 1)
        .mul(255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _ensure_bhwc(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return a batched BHWC tensor and whether a batch dim was added."""
    if image.ndim == 3:
        return image.unsqueeze(0), True
    return image, False


def _restore_image_rank(image: torch.Tensor, squeezed: bool) -> torch.Tensor:
    """Restore an image tensor to its original rank."""
    if squeezed:
        return image.squeeze(0)
    return image


def _rgb_to_luma(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB image tensor in BHWC/HWC layout to luma."""
    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        device=image.device,
        dtype=image.dtype,
    )
    return (image[..., :3] * weights).sum(dim=-1, keepdim=True)


def _box_blur_bhwc(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a per-channel box low-pass filter to a BHWC/HWC image tensor."""
    batched, squeezed = _ensure_bhwc(image)
    bchw = batched.permute(0, 3, 1, 2)
    blurred = F.avg_pool2d(
        bchw,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return _restore_image_rank(blurred.permute(0, 2, 3, 1), squeezed)


def _sobel_grad_bhwc(image: torch.Tensor) -> torch.Tensor:
    """Return Sobel gradient magnitude for a BHWC/HWC scalar image."""
    batched, squeezed = _ensure_bhwc(image)
    bchw = batched.permute(0, 3, 1, 2)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(bchw, kernel_x, padding=1)
    grad_y = F.conv2d(bchw, kernel_y, padding=1)
    grad = torch.sqrt(torch.clamp_min(grad_x.square() + grad_y.square(), 1e-12))
    return _restore_image_rank(grad.permute(0, 2, 3, 1), squeezed)


def _laplacian_bhwc(image: torch.Tensor) -> torch.Tensor:
    """Return Laplacian response for a BHWC/HWC scalar image."""
    batched, squeezed = _ensure_bhwc(image)
    bchw = batched.permute(0, 3, 1, 2)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3)
    response = F.conv2d(bchw, kernel, padding=1)
    return _restore_image_rank(response.permute(0, 2, 3, 1), squeezed)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Compute a mean over valid pixels and channels."""
    if mask is None:
        return value.mean()
    denominator = torch.clamp_min(mask.sum() * value.shape[-1], 1.0)
    return (value * mask).sum() / denominator


def _frequency_high_ratio(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Return predicted/GT high-frequency FFT energy ratio on luma."""
    pred_luma = _rgb_to_luma(rgb_pred)
    gt_luma = _rgb_to_luma(rgb_gt)
    if mask is not None:
        pred_luma = pred_luma * mask
        gt_luma = gt_luma * mask
    pred_fft = torch.fft.rfft2(pred_luma.squeeze(-1), norm="ortho")
    gt_fft = torch.fft.rfft2(gt_luma.squeeze(-1), norm="ortho")
    height = pred_luma.shape[1]
    width = pred_luma.shape[2]
    fy = torch.fft.fftfreq(height, device=pred_luma.device).reshape(1, height, 1)
    fx = torch.fft.rfftfreq(width, device=pred_luma.device).reshape(1, 1, -1)
    high_mask = (torch.sqrt(fx.square() + fy.square()) > 0.18).to(pred_luma.dtype)
    pred_energy = (pred_fft.abs().square() * high_mask).mean()
    gt_energy = (gt_fft.abs().square() * high_mask).mean()
    return pred_energy / torch.clamp_min(gt_energy, 1e-12)


def _frequency_band_ratios(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return predicted/GT FFT energy ratios over radial frequency bands."""
    pred_luma = _rgb_to_luma(rgb_pred)
    gt_luma = _rgb_to_luma(rgb_gt)
    if mask is not None:
        pred_luma = pred_luma * mask
        gt_luma = gt_luma * mask

    pred_fft = torch.fft.rfft2(pred_luma.squeeze(-1), norm="ortho")
    gt_fft = torch.fft.rfft2(gt_luma.squeeze(-1), norm="ortho")
    height = pred_luma.shape[1]
    width = pred_luma.shape[2]
    fy = torch.fft.fftfreq(height, device=pred_luma.device).reshape(1, height, 1)
    fx = torch.fft.rfftfreq(width, device=pred_luma.device).reshape(1, 1, -1)
    radius = torch.sqrt(fx.square() + fy.square())
    pred_energy = pred_fft.abs().square()
    gt_energy = gt_fft.abs().square()

    bands = {
        "fft_energy_ratio_low": (0.00, 0.08),
        "fft_energy_ratio_mid": (0.08, 0.16),
        "fft_energy_ratio_high": (0.16, 0.28),
        "fft_energy_ratio_ultra": (0.28, float("inf")),
    }
    ratios = {}
    for name, (lower, upper) in bands.items():
        band_mask = (radius >= lower) & (radius < upper)
        band_mask = band_mask.to(pred_luma.dtype)
        pred_band_energy = (pred_energy * band_mask).mean()
        gt_band_energy = (gt_energy * band_mask).mean()
        ratios[name] = pred_band_energy / torch.clamp_min(gt_band_energy, 1e-12)
    return ratios


def _tensor_distribution_stats(
    *,
    prefix: str,
    values: torch.Tensor,
) -> dict[str, float]:
    """Return compact distribution stats for a model-space tensor."""
    flat = values.detach().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {}
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.5, 0.95, 0.99], device=finite.device),
    )
    return {
        f"{prefix}_mean": finite.mean().item(),
        f"{prefix}_p50": quantiles[0].item(),
        f"{prefix}_p95": quantiles[1].item(),
        f"{prefix}_p99": quantiles[2].item(),
        f"{prefix}_max": finite.max().item(),
    }


def _gaussian_geometry_metrics(model: MixtureOfGaussians) -> dict[str, float]:
    """Measure Gaussian geometry to expose representation failure modes."""
    with torch.no_grad():
        scales = model.get_scale().detach().float()
        density = model.get_density().detach().float()
        min_axis = torch.clamp_min(scales.min(dim=1).values, 1e-12)
        max_axis = scales.max(dim=1).values
        geom_mean_scale = torch.clamp_min(scales.prod(dim=1), 1e-36).pow(1.0 / 3.0)
        anisotropy = max_axis / min_axis

        metrics = {
            "num_gaussians": float(model.num_gaussians),
            "scale_axis_min": min_axis.min().item(),
            "scale_axis_max": max_axis.max().item(),
        }
        metrics.update(
            _tensor_distribution_stats(
                prefix="scale_geom",
                values=geom_mean_scale,
            )
        )
        metrics.update(
            _tensor_distribution_stats(
                prefix="scale_anisotropy",
                values=anisotropy,
            )
        )
        metrics.update(
            _tensor_distribution_stats(
                prefix="density",
                values=density,
            )
        )
    return metrics


def _camera_focal_mean(gpu_batch) -> float:
    camera_params = (
        gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters
        or gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters
        or gpu_batch.intrinsics_RationalCameraModelParameters
    )
    if camera_params is None:
        return 1.0
    focal_length = camera_params.get("focal_length", np.array([1.0, 1.0]))
    return float(np.asarray(focal_length, dtype=np.float32).mean())


def _screen_space_footprint_metrics(
    *,
    model: MixtureOfGaussians,
    gpu_batch,
    max_samples: int,
) -> dict[str, float]:
    """Approximate projected Gaussian footprint distribution for one view."""
    with torch.no_grad():
        positions = model.get_positions().detach()
        scales = model.get_scale().detach().float().amax(dim=1)
        sample_count = min(max_samples, positions.shape[0])
        if sample_count <= 0:
            return {}
        if sample_count < positions.shape[0]:
            sample_idx = torch.linspace(
                0,
                positions.shape[0] - 1,
                steps=sample_count,
                device=positions.device,
            ).long()
            positions = positions[sample_idx]
            scales = scales[sample_idx]
        pose = gpu_batch.T_to_world.squeeze(0)
        world_to_camera = torch.linalg.inv(pose)
        homogeneous = torch.cat(
            (positions, torch.ones_like(positions[:, :1])),
            dim=1,
        )
        camera_positions = homogeneous @ world_to_camera.transpose(0, 1)
        depth = camera_positions[:, 2]
        in_front = depth > 1e-3
        if not in_front.any():
            return {"footprint_front_fraction": 0.0}
        focal_mean = _camera_focal_mean(gpu_batch)
        projected_radius = scales[in_front] * focal_mean / depth[in_front].clamp_min(1e-3)
        quantiles = torch.quantile(
            projected_radius,
            torch.tensor([0.5, 0.95, 0.99], device=projected_radius.device),
        )
        return {
            "footprint_front_fraction": in_front.float().mean().item(),
            "footprint_radius_px_mean": projected_radius.mean().item(),
            "footprint_radius_px_p50": quantiles[0].item(),
            "footprint_radius_px_p95": quantiles[1].item(),
            "footprint_radius_px_p99": quantiles[2].item(),
            "footprint_radius_px_max": projected_radius.max().item(),
            "footprint_radius_lt_0p5_fraction": (projected_radius < 0.5).float().mean().item(),
            "footprint_radius_gt_8_fraction": (projected_radius > 8.0).float().mean().item(),
        }


def _diagnostic_metrics(
    *,
    rgb_gt: torch.Tensor,
    rgb_pred: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, float]:
    """Compute validation diagnostics that separate failure modes."""
    low_gt = _box_blur_bhwc(rgb_gt, 31)
    low_pred = _box_blur_bhwc(rgb_pred, 31)
    high_gt = rgb_gt - low_gt
    high_pred = rgb_pred - low_pred
    luma_gt = _rgb_to_luma(rgb_gt)
    luma_pred = _rgb_to_luma(rgb_pred)
    grad_gt = _sobel_grad_bhwc(luma_gt)
    grad_pred = _sobel_grad_bhwc(luma_pred)
    lap_gt = _laplacian_bhwc(luma_gt)
    lap_pred = _laplacian_bhwc(luma_pred)
    metrics = {
        "low_freq_l1": _masked_mean(torch.abs(low_pred - low_gt), mask).item(),
        "high_freq_l1": _masked_mean(torch.abs(high_pred - high_gt), mask).item(),
        "gradient_l1": _masked_mean(torch.abs(grad_pred - grad_gt), mask).item(),
        "laplacian_l1": _masked_mean(torch.abs(lap_pred - lap_gt), mask).item(),
        "fft_high_energy_ratio": _frequency_high_ratio(rgb_pred, rgb_gt, mask).item(),
    }
    metrics.update(
        {
            key: value.item()
            for key, value in _frequency_band_ratios(
                rgb_pred,
                rgb_gt,
                mask,
            ).items()
        }
    )
    return metrics


def _fit_image_to_panel(
    image: np.ndarray,
    *,
    width: int,
    height: int,
    bg_color: tuple[int, int, int] = (18, 18, 18),
) -> np.ndarray:
    """Letterbox one BGR image into a fixed-size panel."""
    panel = np.full((height, width, 3), bg_color, dtype=np.uint8)
    image_h, image_w = image.shape[:2]
    scale = min(width / float(image_w), height / float(image_h))
    resized = cv2.resize(
        image,
        (
            max(1, round(image_w * scale)),
            max(1, round(image_h * scale)),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    y0 = (height - resized.shape[0]) // 2
    x0 = (width - resized.shape[1]) // 2
    panel[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return panel


def _text_panel(
    label: str,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Create a readable row-label panel."""
    panel = np.full((height, width, 3), (12, 12, 12), dtype=np.uint8)
    cv2.putText(
        panel,
        label,
        (12, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _save_validation_grid_jpeg(
    *,
    tile_groups: dict[str, torch.Tensor],
    image_paths: list[str],
    column_names: list[str],
    row_start: int,
    row_end: int,
    output_path: str,
    panel_width: int = 360,
    panel_height: int = 480,
    quality: int = 85,
) -> str:
    """Save one grouped validation grid as a JPEG image."""
    gap_px = 8
    label_width = 220
    header_height = 54
    rows = row_end - row_start
    grid_width = (
        label_width
        + gap_px
        + len(column_names) * (panel_width + gap_px)
        + gap_px
    )
    grid_height = header_height + rows * (panel_height + gap_px) + gap_px
    grid = np.full((grid_height, grid_width, 3), (8, 8, 8), dtype=np.uint8)

    for column_idx, column_name in enumerate(column_names):
        x = label_width + gap_px + column_idx * (panel_width + gap_px)
        header = np.full((header_height, panel_width, 3), (0, 0, 0), dtype=np.uint8)
        cv2.putText(
            header,
            column_name,
            (10, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        grid[0:header_height, x : x + panel_width] = header

    for row_offset, row_idx in enumerate(range(row_start, row_end)):
        y = header_height + row_offset * (panel_height + gap_px)
        row_label = os.path.splitext(os.path.basename(image_paths[row_idx]))[0]
        grid[y : y + panel_height, 0:label_width] = _text_panel(
            row_label,
            width=label_width,
            height=panel_height,
        )
        for column_idx, column_name in enumerate(column_names):
            x = label_width + gap_px + column_idx * (panel_width + gap_px)
            panel = _fit_image_to_panel(
                _tensor_to_bgr_image(tile_groups[column_name][row_idx]),
                width=panel_width,
                height=panel_height,
            )
            grid[y : y + panel_height, x : x + panel_width] = panel

    cv2.imwrite(output_path, grid, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return output_path


def _make_validation_image_tiles(
    *,
    rgb_gt: torch.Tensor,
    rgb_pred: torch.Tensor,
    pred_dist: torch.Tensor,
    pred_opacity: torch.Tensor,
    hit_counts: torch.Tensor,
    mask: torch.Tensor | None,
    sky_mask: torch.Tensor | None,
    max_hit_count: int,
) -> dict[str, torch.Tensor]:
    """Build aligned validation tiles for grouped W&B image grids."""
    rgb_gt = rgb_gt.clip(0, 1.0)
    rgb_pred = rgb_pred.clip(0, 1.0)
    if mask is None:
        mask_rgb = torch.ones_like(rgb_gt)
        valid_mask = None
    else:
        valid_mask = mask.clip(0, 1.0)
        mask_rgb = valid_mask.expand_as(rgb_gt)

    error = torch.abs(rgb_pred - rgb_gt).mean(dim=2, keepdim=True)
    if valid_mask is not None:
        error = error * valid_mask

    low_gt = _box_blur_bhwc(rgb_gt, 31)
    low_pred = _box_blur_bhwc(rgb_pred, 31)
    low_error = torch.abs(low_pred - low_gt).mean(dim=2, keepdim=True)
    high_gt = rgb_gt - low_gt
    high_pred = rgb_pred - low_pred
    high_error = torch.abs(high_pred - high_gt).mean(dim=2, keepdim=True)
    edge_error = torch.abs(
        _sobel_grad_bhwc(_rgb_to_luma(rgb_pred))
        - _sobel_grad_bhwc(_rgb_to_luma(rgb_gt))
    )
    if valid_mask is not None:
        low_error = low_error * valid_mask
        high_error = high_error * valid_mask
        edge_error = edge_error * valid_mask

    error_rgb = jet_map(error, 0.25)
    low_error_rgb = jet_map(low_error, 0.25)
    high_error_rgb = jet_map(high_error, 0.12)
    edge_error_rgb = jet_map(edge_error, 1.0)
    depth_rgb = _robust_jet_map(pred_dist, valid_mask)
    opacity_rgb = jet_map(pred_opacity, 1)
    hits_rgb = jet_map(hit_counts, max_hit_count)
    hit_coverage = (hit_counts > 0).to(rgb_gt.dtype)
    opacity_coverage = (pred_opacity > 0.01).to(rgb_gt.dtype)
    if valid_mask is not None:
        hit_coverage = hit_coverage * valid_mask
        opacity_coverage = opacity_coverage * valid_mask
    tiles = {
        "input_rgb": rgb_gt,
        "training_mask": mask_rgb,
        "prediction_rgb": rgb_pred,
        "rgb_error": error_rgb,
        "low_freq_error": low_error_rgb,
        "high_freq_error": high_error_rgb,
        "edge_error": edge_error_rgb,
        "predicted_depth": depth_rgb,
        "opacity": opacity_rgb,
        "ray_hits": hits_rgb,
        "hit_coverage": hit_coverage.expand_as(rgb_gt),
        "opacity_coverage": opacity_coverage.expand_as(rgb_gt),
    }
    if sky_mask is not None:
        tiles["sky_opacity_mask"] = sky_mask.clip(0, 1.0).expand_as(rgb_gt)
    return tiles


class Trainer3DGRUT:
    """Trainer for paper: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes" """

    model: MixtureOfGaussians
    """ Gaussian Model """

    train_dataset: BoundedMultiViewDataset
    val_dataset: BoundedMultiViewDataset

    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader

    scene_extent: float = 1.0
    """TODO: Add docstring"""

    scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
    """TODO: Add docstring"""

    strategy: BaseStrategy
    """ Strategy for optimizing the Gaussian model in terms of densification, pruning, etc. """

    gui = None
    """ If GUI is enabled, references the GUI interface """

    criterions: Dict
    """ Contains functors required to compute evaluation metrics, i.e. psnr, ssim, lpips """

    tracking: Dict
    """ Contains all components used to report progress of training """

    post_processing: Optional[nn.Module] = None
    """ Post-processing module """

    post_processing_optimizers: Optional[list] = None
    """ Optimizers for post-processing module """

    post_processing_schedulers: Optional[list] = None
    """ Schedulers for post-processing module optimizers """

    camera_residual: Optional[CameraResidual] = None
    """ Optional camera residual module """

    camera_residual_optimizer: Optional[torch.optim.Optimizer] = None
    """ Optimizer for camera residual module """

    _distillation_start_step: int = -1
    """ Step at which distillation starts (-1 means disabled) """

    @staticmethod
    def create_from_checkpoint(resume: str, conf: DictConfig):
        """Create a new trainer from a checkpoint file"""

        conf.resume = resume
        conf.import_ply.enabled = False
        return Trainer3DGRUT(conf)

    @staticmethod
    def create_from_ply(ply_path: str, conf: DictConfig):
        """Create a new trainer from a PLY file"""

        conf.resume = ""
        conf.import_ply.enabled = True
        conf.import_ply.path = ply_path
        return Trainer3DGRUT(conf)

    @torch.cuda.nvtx.range("setup-trainer")
    def __init__(self, conf: DictConfig, device=None):
        """Set up a new training session, or continue an existing one based on configuration"""

        # Keep track of useful fields
        self.conf = conf
        """ Global configuration of model, scene, optimization, etc"""
        self.device = device if device is not None else DEFAULT_DEVICE
        """ Device used for training and visualizations """
        self.global_step = 0
        """ Current global iteration of the trainer """
        self.n_iterations = conf.n_iterations
        """ Total number of train iterations to take (for multiple passes over the dataset) """
        self.n_epochs = 0
        """ Total number of train epochs / passes, e.g. single pass over the dataset."""
        self.val_frequency = conf.val_frequency
        """ Validation frequency, in terms on global steps """

        # Setup the trainer and components
        logger.log_rule("Load Datasets")
        self.init_dataloaders(conf)
        self.init_scene_extents(self.train_dataset)
        logger.log_rule("Initialize Model")
        self.init_model(conf, self.scene_extent)
        self.init_densification_and_pruning_strategy(conf)
        logger.log_rule("Setup Model Weights & Training")
        self.init_metrics()
        self.init_post_processing(conf)
        self.init_camera_residual(conf)
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)
        self.init_gui(conf, self.model, self.train_dataset, self.val_dataset, self.scene_bbox)

    def init_dataloaders(self, conf: DictConfig):
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        train_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": True,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        val_dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": conf.num_workers,
                "batch_size": 1,
                "shuffle": False,
                "pin_memory": True,
                "persistent_workers": True if conf.num_workers > 0 else False,
            }
        )

        train_dataloader = MultiEpochsDataLoader(train_dataset, **train_dataloader_kwargs)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, **val_dataloader_kwargs)

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

    def teardown_dataloaders(self):
        if self.train_dataloader is not None:
            del self.train_dataloader
        if self.val_dataloader is not None:
            del self.val_dataloader
        if self.train_dataset is not None:
            del self.train_dataset
        if self.val_dataset is not None:
            del self.val_dataset

    def init_scene_extents(self, train_dataset: BoundedMultiViewDataset) -> None:
        scene_bbox: tuple[torch.Tensor, torch.Tensor]  # Tuple of vec3 (min,max)
        scene_extent = train_dataset.get_scene_extent()
        scene_bbox = train_dataset.get_scene_bbox()
        self.scene_extent = scene_extent
        self.scene_bbox = scene_bbox

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        """Initializes the gaussian model and the optix context"""
        self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_densification_and_pruning_strategy(self, conf: DictConfig) -> None:
        """Set pre-train / post-train iteration logic. i.e. densification and pruning"""
        assert self.model is not None
        match self.conf.strategy.method:
            case "GSStrategy":
                from threedgrut.strategy.gs import GSStrategy

                self.strategy = GSStrategy(conf, self.model)
                logger.info("🔆 Using GS strategy")
            case "MCMCStrategy":
                from threedgrut.strategy.mcmc import MCMCStrategy

                self.strategy = MCMCStrategy(conf, self.model)
                logger.info("🔆 Using MCMC strategy")
            case _:
                raise ValueError(f"unrecognized model.strategy {conf.strategy.method}")

    def setup_training(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
    ):
        """
        Performs required steps to setup the optimization:
        1. Initialize the gaussian model fields: load previous weights from checkpoint, or initialize from scratch.
        2. Build BVH acceleration structure for gaussian model, if not loaded with checkpoint
        3. Set up the optimizer to optimize the gaussian model params
        4. Initialize the densification buffers in the densificaiton strategy
        """

        # Initialize
        if conf.resume:  # Load a checkpoint
            logger.info(f"🤸 Loading a pretrained checkpoint from {conf.resume}!")
            checkpoint = torch.load(conf.resume, weights_only=False)
            model.init_from_checkpoint(checkpoint)
            self.strategy.init_densification_buffer(checkpoint)
            global_step = checkpoint["global_step"]

            # Restore post-processing state
            if "post_processing" in checkpoint and self.post_processing is not None:
                self.post_processing.load_state_dict(checkpoint["post_processing"]["module"])
                for opt, opt_state in zip(
                    self.post_processing_optimizers,
                    checkpoint["post_processing"]["optimizers"],
                ):
                    opt.load_state_dict(opt_state)
                for sched, sched_state in zip(
                    self.post_processing_schedulers,
                    checkpoint["post_processing"]["schedulers"],
                ):
                    sched.load_state_dict(sched_state)
                logger.info("📷 Post-processing state restored from checkpoint")
            if "camera_residual" in checkpoint and self.camera_residual is not None:
                self.camera_residual.load_state_dict(checkpoint["camera_residual"]["module"])
                self.camera_residual_optimizer.load_state_dict(
                    checkpoint["camera_residual"]["optimizer"]
                )
                logger.info("📷 Camera residual state restored from checkpoint")
        elif conf.import_ply.enabled:
            ply_path = (
                conf.import_ply.path
                if conf.import_ply.path
                else f"{conf.out_dir}/{conf.experiment_name}/export_last.ply"
            )
            logger.info(f"Loading a ply model from {ply_path}!")
            model.init_from_ply(ply_path)
            self.strategy.init_densification_buffer()
            model.build_acc()
            global_step = conf.import_ply.init_global_step
        else:
            logger.info(f"🤸 Initiating new 3dgrut training..")
            match conf.initialization.method:
                case "random":
                    model.init_from_random_point_cloud(
                        num_gaussians=conf.initialization.num_gaussians,
                        xyz_max=conf.initialization.xyz_max,
                        xyz_min=conf.initialization.xyz_min,
                    )
                case "colmap":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    model.init_from_colmap(conf.path, observer_points)
                case "fused_point_cloud":
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    ply_path = conf.initialization.fused_point_cloud_path
                    logger.info(f"Initializing from accumulated point cloud: {ply_path}")
                    model.init_from_fused_point_cloud(ply_path, observer_points)
                case "point_cloud":
                    try:
                        ply_path = os.path.join(conf.path, "point_cloud.ply")
                        model.init_from_pretrained_point_cloud(ply_path)
                    except FileNotFoundError as e:
                        logger.error(e)
                        raise e
                case "checkpoint":
                    checkpoint = torch.load(conf.initialization.path, weights_only=False)
                    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
                case "lidar":
                    assert isinstance(
                        train_dataset, datasets.NCoreDataset
                    ), "can only initialize from lidar with NCoreDataset"
                    pc = PointCloud.from_sequence(
                        list(train_dataset.get_point_clouds(step_frame=1, non_dynamic_points_only=True)),
                        device="cpu",
                    )
                    if conf.initialization.num_points < len(pc.xyz_end):
                        # Deterministically random subsample points if there are more points than the specified number of gaussians
                        rng = torch.Generator().manual_seed(conf.seed_initialization)
                        idxs = torch.randperm(len(pc.xyz_end), generator=rng)[: conf.initialization.num_points]
                        pc = pc.selected_idxs(idxs)
                    observer_points = torch.tensor(
                        train_dataset.get_observer_points(),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    model.init_from_lidar(pc, observer_points)
                case _:
                    raise ValueError(
                        f"unrecognized initialization.method {conf.initialization.method}, choose from [colmap, point_cloud, random, checkpoint, lidar]"
                    )

            self.strategy.init_densification_buffer()

            model.build_acc()
            model.setup_optimizer()
            global_step = 0

        self.global_step = global_step
        self.n_epochs = int((conf.n_iterations + len(train_dataset) - 1) / len(train_dataset))

    def init_gui(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
        val_dataset: BoundedMultiViewDataset,
        scene_bbox,
    ):
        gui = None

        if conf.with_gui:
            from threedgrut.utils.gui import GUI

            gui = GUI(conf, model, train_dataset, val_dataset, scene_bbox)

        elif conf.with_viser_gui:
            from threedgrut.utils.viser_gui_util import ViserGUI

            gui = ViserGUI(conf, model, train_dataset, val_dataset, scene_bbox)

        self.gui = gui

    def init_metrics(self):
        self.criterions = Dict(
            psnr=PeakSignalNoiseRatio(data_range=1).to(self.device),
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device),
            lpips=LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device),
        )

    def init_experiments_tracking(self, conf: DictConfig):
        # Initialize the tensorboard writer
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(
            conf, object_name, conf.out_dir, conf.experiment_name, conf.use_wandb
        )
        logger.info(f"📊 Training logs & will be saved to: {out_dir}")

        # Store parsed config for reference
        with open(os.path.join(out_dir, "parsed.yaml"), "w") as fp:
            OmegaConf.save(config=conf, f=fp)

        # Pack all components used to track progress of training
        self.tracking = Dict(
            writer=writer,
            run_name=run_name,
            object_name=object_name,
            output_dir=out_dir,
        )

    def init_post_processing(self, conf: DictConfig):
        """Initialize post-processing module based on config."""
        method = conf.post_processing.method

        if method is None:
            return

        if method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            frames_per_camera = self.train_dataset.get_frames_per_camera()
            num_cameras = len(frames_per_camera)
            num_frames = sum(frames_per_camera)

            use_controller = conf.post_processing.get("use_controller", True)

            # Distillation mode: controller activates after main training
            # Total iterations = n_iterations, distillation starts at n_iterations - n_distillation_steps
            n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = main_training_steps / conf.n_iterations
                controller_distillation = True
                self._distillation_start_step = main_training_steps
                logger.info(f"📷 PPISP distillation mode: controller activates at step {main_training_steps}")
            elif use_controller:
                controller_activation_ratio = 0.8
                controller_distillation = False
                self._distillation_start_step = -1
            else:
                controller_activation_ratio = 0.0
                controller_distillation = False
                self._distillation_start_step = -1

            ppisp_config = PPISPConfig(
                use_controller=use_controller,
                controller_distillation=controller_distillation,
                controller_activation_ratio=controller_activation_ratio,
            )

            self.post_processing = PPISP(
                num_cameras=num_cameras,
                num_frames=num_frames,
                config=ppisp_config,
            ).to(self.device)

            self.post_processing_optimizers = self.post_processing.create_optimizers()
            self.post_processing_schedulers = self.post_processing.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=conf.n_iterations,
            )

            logger.info(f"📷 {method.upper()} initialized: {num_cameras} cameras, {num_frames} frames")
        elif method == "luminance_affine":
            frames_per_camera = self.train_dataset.get_frames_per_camera()
            num_cameras = len(frames_per_camera)
            num_frames = sum(frames_per_camera)

            self.post_processing = LuminanceAffine(
                num_cameras=num_cameras,
                num_frames=num_frames,
                lr=conf.post_processing.get("lr", 1e-3),
                reg_lambda=conf.post_processing.get("reg_lambda", 1e-2),
                use_frame_residual=conf.post_processing.get(
                    "use_frame_residual",
                    False,
                ),
                max_log_gain=conf.post_processing.get("max_log_gain", 0.25),
                max_bias=conf.post_processing.get("max_bias", 0.10),
            ).to(self.device)

            self.post_processing_optimizers = self.post_processing.create_optimizers()
            self.post_processing_schedulers = self.post_processing.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=conf.n_iterations,
            )

            logger.info(
                f"📷 LUMINANCE_AFFINE initialized: {num_cameras} cameras, "
                f"{num_frames} frames"
            )
        else:
            raise ValueError(f"Unknown post-processing method: {method}")

    def init_camera_residual(self, conf: DictConfig) -> None:
        """Initialize optional bounded camera residual calibration."""
        if not conf.camera_residual.enabled:
            return
        frames_per_camera = self.train_dataset.get_frames_per_camera()
        self.camera_residual = CameraResidual(
            num_cameras=len(frames_per_camera),
            lr=conf.camera_residual.lr,
            reg_lambda=conf.camera_residual.reg_lambda,
            max_rotation_rad=conf.camera_residual.max_rotation_rad,
            max_translation_m=conf.camera_residual.max_translation_m,
            optimize_global=conf.camera_residual.optimize_global,
            optimize_per_camera=conf.camera_residual.optimize_per_camera,
        ).to(self.device)
        self.camera_residual_optimizer = self.camera_residual.create_optimizer()
        logger.warning(
            "📷 CAMERA_RESIDUAL enabled. Current 3DGUT CUDA backward does not "
            "return ray/sensor gradients; monitor camera_residual/max_abs_grad."
        )

    def _apply_camera_residual(self, gpu_batch):
        if self.camera_residual is None:
            return gpu_batch
        return self.camera_residual(gpu_batch)

    def _validation_log_image_views(self) -> set[int]:
        """Return validation iteration indices to log as W&B image grids."""
        eval_image_count = int(self.conf.writer.get("eval_image_count", 5))
        if eval_image_count <= 0:
            return {int(idx) for idx in self.conf.writer.log_image_views}
        total_views = len(self.val_dataloader)
        if total_views <= 0:
            return set()
        selected_count = min(eval_image_count, total_views)
        indices = np.linspace(
            0,
            total_views - 1,
            num=selected_count,
            dtype=np.int64,
        )
        return {int(idx) for idx in indices}

    @torch.cuda.nvtx.range("get_metrics")
    def get_metrics(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        losses: dict[str, torch.Tensor],
        profilers: dict[str, CudaTimer],
        split: str = "training",
        iteration: Optional[int] = None,
    ) -> dict[str, Union[int, float]]:
        """Computes dictionary of single batch metrics based on current batch output.
        Args:
            gpu_batch: GT data of current batch
            output: model prediction for current batch
            losses: dictionary of loss terms computed for current batch
            split: name of split metrics are computed for - 'training' or 'validation'
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        step = self.global_step

        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]

        psnr = self.criterions["psnr"]
        ssim = self.criterions["ssim"]
        lpips = self.criterions["lpips"]

        # Move losses to cpu once
        metrics["losses"] = {k: v.detach().item() for k, v in losses.items()}

        is_compute_train_hit_metrics = (split == "training") and (step % self.conf.writer.hit_stat_frequency == 0)
        is_compute_validation_metrics = split == "validation"

        if is_compute_train_hit_metrics or is_compute_validation_metrics:
            metrics["hits_mean"] = outputs["hits_count"].mean().item()
            metrics["hits_std"] = outputs["hits_count"].std().item()
            metrics["hits_min"] = outputs["hits_count"].min().item()
            metrics["hits_max"] = outputs["hits_count"].max().item()
            hit_mask = (outputs["hits_count"] > 0).float()
            opacity_mask = (outputs["pred_opacity"] > 0.01).float()
            if gpu_batch.mask is not None:
                valid = gpu_batch.mask
                valid_denominator = torch.clamp_min(valid.sum(), 1.0)
                metrics["valid_hit_coverage"] = ((hit_mask * valid).sum() / valid_denominator).item()
                metrics["valid_opacity_coverage"] = ((opacity_mask * valid).sum() / valid_denominator).item()
            else:
                metrics["valid_hit_coverage"] = hit_mask.mean().item()
                metrics["valid_opacity_coverage"] = opacity_mask.mean().item()

        if is_compute_validation_metrics:
            with torch.cuda.nvtx.range(f"criterions_psnr"):
                metrics["psnr"] = psnr(rgb_pred, rgb_gt).item()
                if gpu_batch.mask is not None:
                    mask = gpu_batch.mask
                    masked_error = torch.square(rgb_pred - rgb_gt) * mask
                    masked_denominator = torch.clamp_min(
                        mask.sum() * rgb_gt.shape[-1],
                        1.0,
                    )
                    masked_mse = masked_error.sum() / masked_denominator
                    metrics["masked_psnr"] = (
                        -10.0 * torch.log10(torch.clamp_min(masked_mse, 1e-12))
                    ).item()
                    metrics["mask_coverage"] = mask.mean().item()

            rgb_gt_full = rgb_gt.permute(0, 3, 1, 2)
            pred_rgb_full = rgb_pred.permute(0, 3, 1, 2)
            pred_rgb_full_clipped = rgb_pred.clip(0, 1).permute(0, 3, 1, 2)

            with torch.cuda.nvtx.range(f"criterions_ssim"):
                metrics["ssim"] = ssim(pred_rgb_full, rgb_gt_full).item()
            with torch.cuda.nvtx.range(f"criterions_lpips"):
                metrics["lpips"] = lpips(pred_rgb_full_clipped, rgb_gt_full).item()

            metrics.update(
                _diagnostic_metrics(
                    rgb_gt=rgb_gt,
                    rgb_pred=rgb_pred,
                    mask=gpu_batch.mask,
                )
            )
            metrics.update(
                _screen_space_footprint_metrics(
                    model=self.model,
                    gpu_batch=gpu_batch,
                    max_samples=int(self.conf.writer.get("footprint_sample_count", 200000)),
                )
            )

            if iteration in self._validation_log_image_views():
                mask = gpu_batch.mask[-1] if gpu_batch.mask is not None else None
                sky_mask = gpu_batch.sky_mask[-1] if gpu_batch.sky_mask is not None else None
                metrics["eval_image_path"] = gpu_batch.image_path
                metrics["img_eval_tiles"] = _make_validation_image_tiles(
                    rgb_gt=gpu_batch.rgb_gt[-1],
                    rgb_pred=outputs["pred_rgb"][-1],
                    pred_dist=outputs["pred_dist"][-1],
                    pred_opacity=outputs["pred_opacity"][-1],
                    hit_counts=outputs["hits_count"][-1],
                    mask=mask,
                    sky_mask=sky_mask,
                    max_hit_count=self.conf.writer.max_num_hits,
                )

        if profilers:
            timings = {}
            for key, timer in profilers.items():
                if timer.enabled:
                    timings[key] = timer.timing()
            if timings:
                metrics["timings"] = timings

        return metrics

    @torch.cuda.nvtx.range("get_losses")
    def get_losses(
        self, gpu_batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Computes dictionary of losses for current batch.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
        Returns:
            losses: dictionary of loss terms computed for current batch.
        """
        rgb_gt = gpu_batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]
        mask = gpu_batch.mask
        loss_denominator = torch.tensor(
            rgb_gt.numel(), dtype=rgb_gt.dtype, device=self.device
        )

        # Mask out the invalid pixels if the mask is provided
        if mask is not None:
            rgb_gt = rgb_gt * mask
            rgb_pred = rgb_pred * mask
            loss_denominator = torch.clamp(
                mask.sum() * rgb_gt.shape[-1],
                min=1.0,
            )

        # L1 loss
        loss_l1 = torch.zeros(1, device=self.device)
        lambda_l1 = 0.0
        if self.conf.loss.use_l1:
            with torch.cuda.nvtx.range(f"loss-l1"):
                loss_l1 = torch.abs(rgb_pred - rgb_gt).sum() / loss_denominator
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range(f"loss-l2"):
                squared_error = torch.square(rgb_pred - rgb_gt)
                loss_l2 = squared_error.sum() / loss_denominator
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range(f"loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_rgb_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = 1.0 - ssim(pred_rgb_full, rgb_gt_full)
                lambda_ssim = self.conf.loss.lambda_ssim

        # Opacity regularization
        loss_opacity = torch.zeros(1, device=self.device)
        lambda_opacity = 0.0
        if self.conf.loss.use_opacity:
            with torch.cuda.nvtx.range(f"loss-opacity"):
                loss_opacity = torch.abs(self.model.get_density()).mean()
                lambda_opacity = self.conf.loss.lambda_opacity

        # Scale regularization
        loss_scale = torch.zeros(1, device=self.device)
        lambda_scale = 0.0
        if self.conf.loss.use_scale:
            with torch.cuda.nvtx.range(f"loss-scale"):
                loss_scale = torch.abs(self.model.get_scale()).mean()
                lambda_scale = self.conf.loss.lambda_scale

        loss_sky_opacity = torch.zeros(1, device=self.device)
        lambda_sky_opacity = 0.0
        if self.conf.loss.use_sky_opacity:
            if gpu_batch.sky_mask is None:
                raise RuntimeError("loss.use_sky_opacity requires dataset.sky_mask_folder.")
            with torch.cuda.nvtx.range(f"loss-sky-opacity"):
                sky_mask = gpu_batch.sky_mask
                sky_denominator = torch.clamp(sky_mask.sum(), min=1.0)
                loss_sky_opacity = (outputs["pred_opacity"] * sky_mask).sum() / sky_denominator
                lambda_sky_opacity = self.conf.loss.lambda_sky_opacity

        # Total loss
        loss = (
            lambda_l1 * loss_l1
            + lambda_ssim * loss_ssim
            + lambda_opacity * loss_opacity
            + lambda_scale * loss_scale
            + lambda_sky_opacity * loss_sky_opacity
        )
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            opacity_loss=lambda_opacity * loss_opacity,
            scale_loss=lambda_scale * loss_scale,
            sky_opacity_loss=lambda_sky_opacity * loss_sky_opacity,
            sky_opacity_loss_raw=loss_sky_opacity,
        )

    @torch.cuda.nvtx.range("log_validation_iter")
    def log_validation_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single validation iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        logger.log_progress(
            task_name="Validation",
            advance=1,
            iteration=f"{str(iteration)}",
            psnr=batch_metrics["psnr"],
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range("log_validation_pass")
    def log_validation_pass(self, metrics: dict[str, Any]) -> None:
        """Log information after a single validation pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if self.conf.use_wandb and "img_eval_tiles" in metrics:
            jpg_dir = os.path.join(
                self.tracking.output_dir,
                "wandb_eval_jpg",
                f"step_{global_step:06d}",
            )
            os.makedirs(jpg_dir, exist_ok=True)
            grid_columns = [
                "input_rgb",
                "training_mask",
                "sky_opacity_mask",
                "prediction_rgb",
                "rgb_error",
                "low_freq_error",
                "high_freq_error",
                "edge_error",
                "predicted_depth",
                "opacity",
                "ray_hits",
                "hit_coverage",
                "opacity_coverage",
            ]
            tile_groups = metrics["img_eval_tiles"]
            grid_columns = [column for column in grid_columns if column in tile_groups]
            image_paths = metrics["eval_image_path"]
            group_size = int(self.conf.writer.get("eval_image_group_size", 5))
            if group_size < 1:
                group_size = len(image_paths)
            for group_start in range(0, len(image_paths), group_size):
                group_idx = group_start // group_size
                group_end = min(group_start + group_size, len(image_paths))
                grid_path = os.path.join(
                    jpg_dir,
                    f"group_{group_idx:03d}.jpg",
                )
                _save_validation_grid_jpeg(
                    tile_groups=tile_groups,
                    image_paths=image_paths,
                    column_names=grid_columns,
                    row_start=group_start,
                    row_end=group_end,
                    output_path=grid_path,
                )
                caption = (
                    "rows=image path, columns="
                    f"{', '.join(grid_columns)}"
                )
                wandb.log(
                    {
                        f"eval/image_grid/group_{group_idx:03d}": wandb.Image(
                            grid_path,
                            caption=caption,
                        )
                    }
                )

        mean_timings = {}
        if "timings" in metrics:
            for time_key in metrics["timings"]:
                mean_timings[time_key] = np.mean(metrics["timings"][time_key])
                writer.add_scalar(f"time/val/{time_key}", mean_timings[time_key], global_step)

        writer.add_scalar(
            "geometry/num_gaussians",
            self.model.num_gaussians,
            self.global_step,
        )

        mean_psnr = np.mean(metrics["psnr"])
        writer.add_scalar("val/psnr", mean_psnr, global_step)
        if "masked_psnr" in metrics:
            writer.add_scalar(
                "val/masked_psnr",
                np.mean(metrics["masked_psnr"]),
                global_step,
            )
        if "mask_coverage" in metrics:
            writer.add_scalar(
                "val/mask_coverage",
                np.mean(metrics["mask_coverage"]),
                global_step,
            )
        writer.add_scalar("val/ssim", np.mean(metrics["ssim"]), global_step)
        writer.add_scalar("val/lpips", np.mean(metrics["lpips"]), global_step)
        writer.add_scalar(
            "diagnostics/residual/low_freq_l1",
            np.mean(metrics["low_freq_l1"]),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/residual/high_freq_l1",
            np.mean(metrics["high_freq_l1"]),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/residual/gradient_l1",
            np.mean(metrics["gradient_l1"]),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/residual/laplacian_l1",
            np.mean(metrics["laplacian_l1"]),
            global_step,
        )
        writer.add_scalar(
            "diagnostics/frequency/ratio_high_legacy",
            np.mean(metrics["fft_high_energy_ratio"]),
            global_step,
        )
        frequency_metric_names = (
            ("fft_energy_ratio_low", "diagnostics/frequency/ratio_low"),
            ("fft_energy_ratio_mid", "diagnostics/frequency/ratio_mid"),
            ("fft_energy_ratio_high", "diagnostics/frequency/ratio_high"),
            ("fft_energy_ratio_ultra", "diagnostics/frequency/ratio_ultra"),
        )
        for metric_name, wandb_name in frequency_metric_names:
            writer.add_scalar(
                wandb_name,
                np.mean(metrics[metric_name]),
                global_step,
            )
        geometry_metric_names = (
            ("num_gaussians", "geometry/num_gaussians"),
            ("scale_axis_min", "geometry/scale/axis_min"),
            ("scale_axis_max", "geometry/scale/axis_max"),
            ("scale_geom_mean", "geometry/scale/geom_mean"),
            ("scale_geom_p50", "geometry/scale/geom_p50"),
            ("scale_geom_p95", "geometry/scale/geom_p95"),
            ("scale_geom_p99", "geometry/scale/geom_p99"),
            ("scale_geom_max", "geometry/scale/geom_max"),
            ("scale_anisotropy_mean", "geometry/anisotropy/mean"),
            ("scale_anisotropy_p50", "geometry/anisotropy/p50"),
            ("scale_anisotropy_p95", "geometry/anisotropy/p95"),
            ("scale_anisotropy_p99", "geometry/anisotropy/p99"),
            ("scale_anisotropy_max", "geometry/anisotropy/max"),
            ("density_mean", "geometry/density/mean"),
            ("density_p50", "geometry/density/p50"),
            ("density_p95", "geometry/density/p95"),
            ("density_p99", "geometry/density/p99"),
            ("density_max", "geometry/density/max"),
        )
        geometry_metrics = _gaussian_geometry_metrics(self.model)
        for metric_name, wandb_name in geometry_metric_names:
            if metric_name in geometry_metrics:
                writer.add_scalar(wandb_name, geometry_metrics[metric_name], global_step)
        if self.camera_residual is not None:
            for metric_name, value in self.camera_residual.stats().items():
                writer.add_scalar(
                    f"camera_residual/{metric_name}",
                    value,
                    global_step,
                )
        writer.add_scalar(
            "val/hits/min",
            np.mean(metrics["hits_min"]),
            global_step,
        )
        writer.add_scalar(
            "val/hits/max",
            np.mean(metrics["hits_max"]),
            global_step,
        )
        writer.add_scalar(
            "val/hits/mean",
            np.mean(metrics["hits_mean"]),
            global_step,
        )
        if "valid_hit_coverage" in metrics:
            writer.add_scalar(
                "diagnostics/coverage/valid_hit_fraction",
                np.mean(metrics["valid_hit_coverage"]),
                global_step,
            )
        if "valid_opacity_coverage" in metrics:
            writer.add_scalar(
                "diagnostics/coverage/valid_opacity_fraction",
                np.mean(metrics["valid_opacity_coverage"]),
                global_step,
            )
        footprint_metric_names = (
            ("footprint_front_fraction", "diagnostics/footprint/front_fraction"),
            ("footprint_radius_px_mean", "diagnostics/footprint/radius_px_mean"),
            ("footprint_radius_px_p50", "diagnostics/footprint/radius_px_p50"),
            ("footprint_radius_px_p95", "diagnostics/footprint/radius_px_p95"),
            ("footprint_radius_px_p99", "diagnostics/footprint/radius_px_p99"),
            ("footprint_radius_px_max", "diagnostics/footprint/radius_px_max"),
            (
                "footprint_radius_lt_0p5_fraction",
                "diagnostics/footprint/radius_lt_0p5_fraction",
            ),
            (
                "footprint_radius_gt_8_fraction",
                "diagnostics/footprint/radius_gt_8_fraction",
            ),
        )
        for metric_name, wandb_name in footprint_metric_names:
            if metric_name in metrics:
                writer.add_scalar(wandb_name, np.mean(metrics[metric_name]), global_step)

        loss = np.mean(metrics["losses"]["total_loss"])
        writer.add_scalar("val/loss/total", loss, global_step)
        if self.conf.loss.use_l1:
            l1_loss = np.mean(metrics["losses"]["l1_loss"])
            writer.add_scalar("val/loss/l1", l1_loss, global_step)
        if self.conf.loss.use_l2:
            l2_loss = np.mean(metrics["losses"]["l2_loss"])
            writer.add_scalar("val/loss/l2", l2_loss, global_step)
        if self.conf.loss.use_ssim:
            ssim_loss = np.mean(metrics["losses"]["ssim_loss"])
            writer.add_scalar("val/loss/ssim", ssim_loss, global_step)
        if self.conf.loss.use_sky_opacity:
            sky_opacity_loss = np.mean(metrics["losses"]["sky_opacity_loss"])
            sky_opacity_loss_raw = np.mean(metrics["losses"]["sky_opacity_loss_raw"])
            writer.add_scalar("val/loss/sky_opacity", sky_opacity_loss, global_step)
            writer.add_scalar("val/loss/sky_opacity_raw", sky_opacity_loss_raw, global_step)

        table = {
            k: np.mean(v)
            for k, v in metrics.items()
            if k
            in (
                "psnr",
                "masked_psnr",
                "ssim",
                "lpips",
                "mask_coverage",
                "low_freq_l1",
                "high_freq_l1",
                "gradient_l1",
                "laplacian_l1",
                "fft_high_energy_ratio",
                "fft_energy_ratio_low",
                "fft_energy_ratio_mid",
                "fft_energy_ratio_high",
                "fft_energy_ratio_ultra",
            )
        }
        for time_key in mean_timings:
            table[time_key] = f"{'{:.2f}'.format(mean_timings[time_key])}" + " ms/it"
        logger.log_table(f"📊 Validation Metrics - Step {global_step}", record=table)

    @torch.cuda.nvtx.range(f"log_training_iter")
    def log_training_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: Optional[int] = None,
    ) -> None:
        """Log information after a single training iteration.
        Args:
            gpu_batch: GT data of current batch
            outputs: model prediction for current batch
            batch_metrics: dictionary of metrics computed for current batch
            iteration: optional, local iteration number within the current pass, e.g 0 <= iter < len(dataset).
        """
        writer = self.tracking.writer
        global_step = self.global_step

        if self.conf.enable_writer and global_step > 0 and global_step % self.conf.log_frequency == 0:
            loss = np.mean(batch_metrics["losses"]["total_loss"])
            writer.add_scalar("train/loss/total", loss, global_step)
            if self.conf.loss.use_l1:
                l1_loss = np.mean(batch_metrics["losses"]["l1_loss"])
                writer.add_scalar("train/loss/l1", l1_loss, global_step)
            if self.conf.loss.use_l2:
                l2_loss = np.mean(batch_metrics["losses"]["l2_loss"])
                writer.add_scalar("train/loss/l2", l2_loss, global_step)
            if self.conf.loss.use_ssim:
                ssim_loss = np.mean(batch_metrics["losses"]["ssim_loss"])
                writer.add_scalar("train/loss/ssim", ssim_loss, global_step)
            if self.conf.loss.use_opacity:
                opacity_loss = np.mean(batch_metrics["losses"]["opacity_loss"])
                writer.add_scalar("train/loss/opacity", opacity_loss, global_step)
            if self.conf.loss.use_scale:
                scale_loss = np.mean(batch_metrics["losses"]["scale_loss"])
                writer.add_scalar("train/loss/scale", scale_loss, global_step)
            if self.conf.loss.use_sky_opacity:
                sky_opacity_loss = np.mean(batch_metrics["losses"]["sky_opacity_loss"])
                sky_opacity_loss_raw = np.mean(batch_metrics["losses"]["sky_opacity_loss_raw"])
                writer.add_scalar("train/loss/sky_opacity", sky_opacity_loss, global_step)
                writer.add_scalar("train/loss/sky_opacity_raw", sky_opacity_loss_raw, global_step)
            if self.post_processing is not None and "post_processing_reg_loss" in batch_metrics["losses"]:
                post_processing_reg_loss = np.mean(batch_metrics["losses"]["post_processing_reg_loss"])
                writer.add_scalar(
                    "train/loss/post_processing_reg",
                    post_processing_reg_loss,
                    global_step,
                )
            if self.camera_residual is not None and "camera_residual_reg_loss" in batch_metrics["losses"]:
                camera_residual_reg_loss = np.mean(batch_metrics["losses"]["camera_residual_reg_loss"])
                writer.add_scalar(
                    "train/loss/camera_residual_reg",
                    camera_residual_reg_loss,
                    global_step,
                )
                for metric_name, value in self.camera_residual.stats().items():
                    writer.add_scalar(
                        f"camera_residual/{metric_name}",
                        value,
                        global_step,
                    )
            if "psnr" in batch_metrics:
                writer.add_scalar("train/psnr", batch_metrics["psnr"], self.global_step)
            if "ssim" in batch_metrics:
                writer.add_scalar("train/ssim", batch_metrics["ssim"], self.global_step)
            if "lpips" in batch_metrics:
                writer.add_scalar("train/lpips", batch_metrics["lpips"], self.global_step)
            if "hits_mean" in batch_metrics:
                writer.add_scalar("train/hits/mean", batch_metrics["hits_mean"], self.global_step)
            if "hits_std" in batch_metrics:
                writer.add_scalar("train/hits/std", batch_metrics["hits_std"], self.global_step)
            if "hits_min" in batch_metrics:
                writer.add_scalar("train/hits/min", batch_metrics["hits_min"], self.global_step)
            if "hits_max" in batch_metrics:
                writer.add_scalar("train/hits/max", batch_metrics["hits_max"], self.global_step)
            if "valid_hit_coverage" in batch_metrics:
                writer.add_scalar(
                    "diagnostics/coverage/train_valid_hit_fraction",
                    batch_metrics["valid_hit_coverage"],
                    self.global_step,
                )
            if "valid_opacity_coverage" in batch_metrics:
                writer.add_scalar(
                    "diagnostics/coverage/train_valid_opacity_fraction",
                    batch_metrics["valid_opacity_coverage"],
                    self.global_step,
                )

            if "timings" in batch_metrics:
                for time_key in batch_metrics["timings"]:
                    writer.add_scalar(
                        f"time/train/{time_key}",
                        batch_metrics["timings"][time_key],
                        self.global_step,
                    )

            writer.add_scalar("train/iteration", self.global_step, self.global_step)
            if hasattr(self, "_training_start_time"):
                elapsed_seconds = time.perf_counter() - self._training_start_time
                writer.add_scalar("time/train/elapsed_seconds", elapsed_seconds, self.global_step)

            writer.add_scalar("geometry/num_gaussians", self.model.num_gaussians, self.global_step)

            # # NOTE: hack to easily compare with 3DGS
            # writer.add_scalar("train_loss_patches/total_loss", loss, global_step)
            # writer.add_scalar("gaussians/count", self.model.num_gaussians, self.global_step)

        logger.log_progress(
            task_name="Training",
            advance=1,
            step=f"{str(self.global_step)}",
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range(f"log_training_pass")
    def log_training_pass(self, metrics):
        """Log information after a single training pass.
        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.
        """
        pass

    @torch.cuda.nvtx.range(f"on_training_end")
    def on_training_end(self):
        """Callback that prompts at the end of training."""
        conf = self.conf
        out_dir = self.tracking.output_dir

        # Export the mixture-of-3d-gaussians
        logger.log_rule("Exporting Models")

        if conf.export_ply.enabled:
            from threedgrut.export import PLYExporter

            ply_path = conf.export_ply.path if conf.export_ply.path else os.path.join(out_dir, "export_last.ply")
            exporter = PLYExporter()
            exporter.export(self.model, Path(ply_path), dataset=self.train_dataset, conf=conf)

        if conf.export_usd.enabled:
            from threedgrut.export import NuRecExporter, USDExporter

            # Determine format for filename suffix
            usdz_format = getattr(conf.export_usd, "format", "nurec")
            if usdz_format == "standard":
                format_suffix = "lightfield"
                exporter = USDExporter.from_config(conf)
            else:
                format_suffix = "nurec"
                exporter = NuRecExporter()

            # Handle path: if not set or relative, put in output directory
            if conf.export_usd.path:
                usdz_path = conf.export_usd.path
                if not os.path.isabs(usdz_path):
                    usdz_path = os.path.join(out_dir, usdz_path)
            else:
                # Default filename includes format suffix
                usdz_path = os.path.join(out_dir, f"export_last_{format_suffix}.usdz")

            exporter.export(
                self.model,
                Path(usdz_path),
                dataset=self.train_dataset,
                conf=conf,
                background=getattr(self, "background", None),
            )

        # Export post-processing report (PPISP-based)
        if self.post_processing is not None and conf.post_processing.method == "ppisp":
            from ppisp.report import export_ppisp_report

            logger.info("📊 Exporting PPISP report...")

            ppisp_report_dir = Path(out_dir) / "ppisp_report"
            frames_per_camera = self.train_dataset.get_frames_per_camera()

            # Get camera names if available
            camera_names = None
            if hasattr(self.train_dataset, "get_camera_names"):
                camera_names = self.train_dataset.get_camera_names()

            export_ppisp_report(
                self.post_processing,
                frames_per_camera=frames_per_camera,
                output_dir=ppisp_report_dir,
                camera_names=camera_names,
            )
            logger.info(f"📊 PPISP report saved to: {ppisp_report_dir}")

        self.teardown_dataloaders()
        self.save_checkpoint(last_checkpoint=True)

        # Evaluate on test set
        if conf.test_last:
            logger.log_rule("Evaluation on Test Set")

            # Renderer test split
            renderer = Renderer.from_preloaded_model(
                model=self.model,
                out_dir=out_dir,
                path=conf.path,
                save_gt=False,
                writer=self.tracking.writer,
                global_step=self.global_step,
                compute_extra_metrics=conf.compute_extra_metrics,
                post_processing=self.post_processing,
            )
            renderer.render_all()

    @torch.cuda.nvtx.range(f"save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        """Saves checkpoint to a path under {conf.out_dir}/{conf.experiment_name}.
        Args:
            last_checkpoint: If true, will update checkpoint title to 'last'.
                             Otherwise uses global step
        """
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        parameters = self.model.get_model_parameters()
        parameters |= {"global_step": self.global_step, "epoch": self.n_epochs - 1}

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        # Add post-processing state to checkpoint (module + optimizers + schedulers)
        if self.post_processing is not None:
            parameters["post_processing"] = {
                "module": self.post_processing.state_dict(),
                "optimizers": [opt.state_dict() for opt in self.post_processing_optimizers],
                "schedulers": [sched.state_dict() for sched in self.post_processing_schedulers],
            }
        if self.camera_residual is not None:
            parameters["camera_residual"] = {
                "module": self.camera_residual.state_dict(),
                "optimizer": self.camera_residual_optimizer.state_dict(),
            }

        os.makedirs(os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True)
        if not last_checkpoint:
            ckpt_path = os.path.join(out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt")
        else:
            ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
        torch.save(parameters, ckpt_path)
        logger.info(f'💾 Saved checkpoint to: "{os.path.abspath(ckpt_path)}"')

    def render_gui(self, scene_updated):
        """Render & refresh a single frame for the gui"""
        gui = self.gui
        if gui is not None:
            import polyscope as ps

            if gui.live_update:
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_cloud_viz()
                gui.update_render_view_viz()

            ps.frame_tick()
            while not gui.viz_do_train:
                ps.frame_tick()

            if ps.window_requests_close():
                logger.warning(
                    "Terminating training from GUI window is not supported. Please terminate it from the terminal."
                )

    def render_gui_viser(self, scene_updated):
        gui = self.gui
        if gui is not None:
            if gui.live_update:
                # update render view
                if scene_updated or self.model.positions.requires_grad:
                    gui.update_point_cloud()
                for client in gui.server.get_clients().values():
                    gui.update_render_view(client, force=True)
                while not gui.viz_do_train:
                    time.sleep(0.0001)

    @torch.cuda.nvtx.range(f"run_train_iter")
    def run_train_iter(
        self,
        global_step: int,
        batch: dict,
        profilers: dict,
        metrics: list,
        conf: DictConfig,
    ):
        # Freeze Gaussians and suspend strategy when distillation starts
        if self._distillation_start_step >= 0 and global_step >= self._distillation_start_step:
            self.model.freeze_gaussians()
            self.strategy.suspend()

        # Access the GPU-cache batch data
        with torch.cuda.nvtx.range(f"train_iter{global_step}_get_gpu_batch"):
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
            gpu_batch = self._apply_camera_residual(gpu_batch)

        # Perform validation if required
        is_time_to_validate = (global_step > 0 or conf.validate_first) and (global_step % self.val_frequency == 0)
        if is_time_to_validate:
            self.run_validation_pass(conf)

        # Compute the outputs of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_fwd"):
            profilers["inference"].start()
            outputs = self.model(gpu_batch, train=True, frame_id=global_step)
            profilers["inference"].end()

        # Apply post-processing to rendered output
        if self.post_processing is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing"):
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=True)

        # Compute the losses of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
            batch_losses = self.get_losses(gpu_batch, outputs)
            # Add post-processing regularization loss
            if self.post_processing is not None:
                post_processing_reg_loss = self.post_processing.get_regularization_loss()
                batch_losses["total_loss"] = batch_losses["total_loss"] + post_processing_reg_loss
                batch_losses["post_processing_reg_loss"] = post_processing_reg_loss
            if self.camera_residual is not None:
                camera_residual_reg_loss = self.camera_residual.get_regularization_loss()
                batch_losses["total_loss"] = batch_losses["total_loss"] + camera_residual_reg_loss
                batch_losses["camera_residual_reg_loss"] = camera_residual_reg_loss

        # Backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_pre_bwd"):
            self.strategy.pre_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Back-propagate the gradients and update the parameters
        with torch.cuda.nvtx.range(f"train_{global_step}_bwd"):
            profilers["backward"].start()
            batch_losses["total_loss"].backward()
            profilers["backward"].end()

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
            scene_updated = self.strategy.post_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Optimizer step
        with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
            if isinstance(self.model.optimizer, SelectiveAdam):
                assert (
                    outputs["mog_visibility"].shape == self.model.density.shape
                ), f"Visibility shape {outputs['mog_visibility'].shape} does not match density shape {self.model.density.shape}"
                self.model.optimizer.step(outputs["mog_visibility"])
            else:
                self.model.optimizer.step()
            self.model.optimizer.zero_grad()

        # Scheduler step
        with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
            self.model.scheduler_step(global_step)

        # Post-processing optimizer/scheduler step
        if self.post_processing_optimizers is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_post_processing_opt"):
                for opt in self.post_processing_optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in self.post_processing_schedulers:
                    sched.step()
        if self.camera_residual_optimizer is not None:
            with torch.cuda.nvtx.range(f"train_{global_step}_camera_residual_opt"):
                self.camera_residual_optimizer.step()
                self.camera_residual_optimizer.zero_grad()

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_opt_step"):
            scene_updated = self.strategy.post_optimizer_step(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
            )

        # Update the SH if required
        if self.model.progressive_training and check_step_condition(
            global_step, 0, 1e6, self.model.feature_dim_increase_interval
        ):
            self.model.increase_num_active_features()

        # Update the BVH if required
        if scene_updated or (
            conf.model.bvh_update_frequency > 0 and global_step % conf.model.bvh_update_frequency == 0
        ):
            with torch.cuda.nvtx.range(f"train_{global_step}_bvh"):
                profilers["build_as"].start()
                self.model.build_acc(rebuild=True)
                profilers["build_as"].end()

        # Increment the global step
        global_step += 1
        self.global_step = global_step

        # Compute metrics
        batch_metrics = self.get_metrics(
            gpu_batch,
            outputs,
            batch_losses,
            profilers,
            split="training",
            iteration=iter,
        )
        if "forward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["forward_render_cuda"] = self.model.renderer.timings["forward_render"]
        if "backward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["backward_render_cuda"] = self.model.renderer.timings["backward_render"]
        metrics.append(batch_metrics)

        # !!! Below global step has been incremented !!!
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_log_iter"):
            self.log_training_iter(gpu_batch, outputs, batch_metrics, iter)
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_save_ckpt"):
            if global_step in conf.checkpoint.iterations:
                self.save_checkpoint()

        # Updating the GUI
        with torch.cuda.nvtx.range(f"train_{global_step - 1}_update_gui"):
            if self.conf.with_viser_gui:
                self.render_gui_viser(scene_updated)
            elif self.conf.with_gui:
                self.render_gui(scene_updated)

    @torch.cuda.nvtx.range(f"run_train_pass")
    def run_train_pass(self, conf: DictConfig):
        """Runs a single train epoch over the dataset."""
        metrics = []
        profilers = {
            "inference": CudaTimer(enabled=self.conf.enable_frame_timings),
            "backward": CudaTimer(enabled=self.conf.enable_frame_timings),
            "build_as": CudaTimer(enabled=self.conf.enable_frame_timings),
        }

        for iter, batch in enumerate(self.train_dataloader):
            # Check if we have reached the maximum number of iterations
            if self.global_step >= conf.n_iterations:
                return

            # Step for training iteration
            self.run_train_iter(self.global_step, batch, profilers, metrics, conf)

        self.log_training_pass(metrics)

    @torch.cuda.nvtx.range(f"run_validation_pass")
    @torch.no_grad()
    def run_validation_pass(self, conf: DictConfig) -> dict[str, Any]:
        """Runs a single validation epoch over the dataset.
        Returns:
             dictionary of metrics computed and aggregated over validation set.
        """

        profilers = {
            "inference": CudaTimer(),
        }
        metrics = []
        logger.info(f"Step {self.global_step} -- Running validation..")
        logger.start_progress(
            task_name="Validation",
            total_steps=len(self.val_dataloader),
            color="medium_purple3",
        )

        for val_iteration, batch_idx in enumerate(self.val_dataloader):
            # Access the GPU-cache batch data
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch_idx)
            gpu_batch = self._apply_camera_residual(gpu_batch)

            # Compute the outputs of a single batch
            with torch.cuda.nvtx.range(f"train.validation_step_{self.global_step}"):
                profilers["inference"].start()
                outputs = self.model(gpu_batch, train=False)
                # Apply post-processing for validation (novel view mode)
                if self.post_processing is not None:
                    outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)
                profilers["inference"].end()

                batch_losses = self.get_losses(gpu_batch, outputs)
                batch_metrics = self.get_metrics(
                    gpu_batch,
                    outputs,
                    batch_losses,
                    profilers,
                    split="validation",
                    iteration=val_iteration,
                )

                self.log_validation_iter(gpu_batch, outputs, batch_metrics, iteration=val_iteration)
                metrics.append(batch_metrics)

        logger.end_progress(task_name="Validation")

        metrics = self._flatten_list_of_dicts(metrics)
        self.log_validation_pass(metrics)
        return metrics

    @staticmethod
    def _flatten_list_of_dicts(list_of_dicts):
        """
        Converts list of dicts -> dict of lists.
        Supports flattening of up to 2 levels of dict hierarchies
        """
        flat_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                if isinstance(v, dict):
                    flat_dict[k] = defaultdict(list) if k not in flat_dict else flat_dict[k]
                    for inner_k, inner_v in v.items():
                        flat_dict[k][inner_k].append(inner_v)
                else:
                    flat_dict[k].append(v)
        return flat_dict

    def run_training(self):
        """Initiate training logic for n_epochs.
        Training and validation are controlled by the config.
        """
        assert self.model.optimizer is not None, "Optimizer needs to be initialized before the training can start!"
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")

        # Training loop
        self._training_start_time = time.perf_counter()
        logger.start_progress(task_name="Training", total_steps=conf.n_iterations, color="spring_green1")

        for epoch_idx in range(self.n_epochs):
            self.run_train_pass(conf)

        logger.end_progress(task_name="Training")

        # Report training statistics
        stats = logger.finished_tasks["Training"]
        table = dict(
            n_steps=f"{self.global_step}",
            n_epochs=f"{self.n_epochs}",
            training_time=f"{stats['elapsed']:.2f} s",
            iteration_speed=f"{self.global_step / stats['elapsed']:.2f} it/s",
        )
        logger.log_table(f"🎊 Training Statistics", record=table)

        # Perform testing
        self.on_training_end()
        logger.info(f"🥳 Training Complete.")

        # Updating the GUI
        if self.gui is not None:
            self.gui.training_done = True
            logger.info(f"🎨 GUI Blocking... Terminate GUI to Stop.")
            self.gui.block_in_rendering_loop(fps=60)
