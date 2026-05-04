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

import json
import os
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import wandb
from addict import Dict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from threedgrut import datasets
from threedgrut.datasets.protocols import BoundedMultiViewDataset
from threedgrut.datasets.utils import (
    DEFAULT_DEVICE,
    MultiEpochsDataLoader,
    PointCloud,
)
from threedgrut.model.camera_residual import CameraResidual
from threedgrut.model.losses import (
    dense_depth_gradient_l1_loss,
    dense_depth_l1_loss,
    equirect_consistency_l1_loss,
    ssim,
)
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers import SelectiveAdam
from threedgrut.post_processing import LuminanceAffine
from threedgrut.render import Renderer
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    check_step_condition,
    create_summary_writer,
    jet_map,
)
from threedgrut.utils.render import apply_post_processing
from threedgrut.utils.timer import CudaTimer


DIAGNOSTIC_QUANTILE_MAX_SAMPLES = 1_000_000


def _quantile_values(value: torch.Tensor) -> torch.Tensor:
    """Bound tensor size before quantile diagnostics on dense point clouds."""
    if value.numel() <= DIAGNOSTIC_QUANTILE_MAX_SAMPLES:
        return value
    step = (
        value.numel() + DIAGNOSTIC_QUANTILE_MAX_SAMPLES - 1
    ) // DIAGNOSTIC_QUANTILE_MAX_SAMPLES
    return value[::step]


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
        max_value = torch.tensor(
            1.0, device=value_map.device, dtype=value_map.dtype
        )
    else:
        max_value = torch.quantile(valid_values, quantile).clamp_min(1e-6)
    return jet_map(value_map, max_value)


def _scalar_tensor_stats(
    value: torch.Tensor,
    prefix: str,
    validity_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Return finite scalar statistics for a tensor."""
    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    if validity_mask is not None:
        valid = validity_mask.detach()
        if valid.shape != detached.shape:
            valid = valid.expand_as(detached)
        finite_mask = finite_mask & (valid > 0.5)

    total_count = max(float(detached.numel()), 1.0)
    finite_fraction = float(finite_mask.sum().item()) / total_count
    if not finite_mask.any():
        return {
            f"{prefix}_finite_fraction": finite_fraction,
            f"{prefix}_nonzero_fraction": 0.0,
        }

    values = detached[finite_mask].float()
    nonzero_fraction = float((values.abs() > 0.0).sum().item()) / max(
        float(values.numel()), 1.0
    )
    quantile_values = _quantile_values(values)
    return {
        f"{prefix}_finite_fraction": finite_fraction,
        f"{prefix}_nonzero_fraction": nonzero_fraction,
        f"{prefix}_min": values.min().item(),
        f"{prefix}_mean": values.mean().item(),
        f"{prefix}_p50": torch.quantile(quantile_values, 0.50).item(),
        f"{prefix}_p95": torch.quantile(quantile_values, 0.95).item(),
        f"{prefix}_p99": torch.quantile(quantile_values, 0.99).item(),
        f"{prefix}_max": values.max().item(),
    }


def _gradient_tensor_stats(
    parameter: torch.nn.Parameter,
    prefix: str,
) -> dict[str, float]:
    """Return finite scalar statistics for an optimizer parameter gradient."""
    if parameter.grad is None:
        return {
            f"{prefix}/has_grad": 0.0,
            f"{prefix}/finite_fraction": 0.0,
            f"{prefix}/nonzero_fraction": 0.0,
        }

    grad = parameter.grad.detach().abs().float()
    finite_mask = torch.isfinite(grad)
    total_count = max(float(grad.numel()), 1.0)
    if not finite_mask.any():
        return {
            f"{prefix}/has_grad": 1.0,
            f"{prefix}/finite_fraction": 0.0,
            f"{prefix}/nonzero_fraction": 0.0,
        }
    values = grad[finite_mask]
    nonzero_fraction = float((values > 0.0).sum().item()) / max(
        float(values.numel()), 1.0
    )
    quantile_values = _quantile_values(values)
    return {
        f"{prefix}/has_grad": 1.0,
        f"{prefix}/finite_fraction": float(finite_mask.sum().item())
        / total_count,
        f"{prefix}/nonzero_fraction": nonzero_fraction,
        f"{prefix}/mean": values.mean().item(),
        f"{prefix}/p50": torch.quantile(quantile_values, 0.50).item(),
        f"{prefix}/p95": torch.quantile(quantile_values, 0.95).item(),
        f"{prefix}/p99": torch.quantile(quantile_values, 0.99).item(),
        f"{prefix}/max": values.max().item(),
    }


def _tensor_to_bgr_image(image: torch.Tensor) -> np.ndarray:
    """Convert an HWC RGB float tensor to a BGR uint8 image."""
    rgb = image.detach().clip(0, 1).mul(255).to(torch.uint8).cpu().numpy()
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
    grad = torch.sqrt(
        torch.clamp_min(grad_x.square() + grad_y.square(), 1e-12)
    )
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


def _masked_mean(
    value: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    """Compute a mean over valid pixels and channels."""
    if mask is None:
        return value.mean()
    denominator = torch.clamp_min(mask.sum() * value.shape[-1], 1.0)
    return (value * mask).sum() / denominator


def _resolve_torch_hub_repo_dir(repo_name_prefix: str) -> str | None:
    """Resolve a cached torch hub repo by prefix without hardcoded paths."""
    hub_dir = torch.hub.get_dir()
    if not os.path.isdir(hub_dir):
        return None
    candidates = [
        os.path.join(hub_dir, item)
        for item in os.listdir(hub_dir)
        if item.startswith(repo_name_prefix)
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


class FoundationFeatureProbe(nn.Module):
    """Frozen image foundation-model probe for structural diagnostics/losses."""

    def __init__(
        self,
        *,
        method: str,
        repo_dir: str,
        weights: str,
        image_size: int,
    ) -> None:
        super().__init__()
        self.method = method
        self.image_size = image_size
        self.model = self._load_model(
            method=method,
            repo_dir=repo_dir,
            weights=weights,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _load_model(
        self,
        *,
        method: str,
        repo_dir: str,
        weights: str,
    ) -> nn.Module:
        if method == "tuna_pixel_patch":
            return nn.Identity()
        source = "github"
        repo = repo_dir
        if not repo:
            if method.startswith("dinov2_"):
                repo = _resolve_torch_hub_repo_dir("facebookresearch_dinov2")
                if repo is None:
                    repo = "facebookresearch/dinov2"
                else:
                    source = "local"
            elif method.startswith("dinov3_"):
                if not weights:
                    msg = (
                        "DINOv3 weights are gated by Meta/Hugging Face. "
                        "Set foundation_features.weights to an accepted "
                        "checkpoint path or URL."
                    )
                    raise ValueError(msg)
                repo = _resolve_torch_hub_repo_dir("facebookresearch_dinov3")
                if repo is None:
                    repo = "facebookresearch/dinov3"
                else:
                    source = "local"
            else:
                msg = f"Unsupported foundation feature method: {method}"
                raise ValueError(msg)
        else:
            source = "local"
        if weights:
            model = torch.hub.load(
                repo,
                method,
                source=source,
                pretrained=True,
                weights=weights,
            )
        else:
            model = torch.hub.load(
                repo, method, source=source, pretrained=True
            )
        if not isinstance(model, nn.Module):
            msg = (
                f"Foundation feature method did not return nn.Module: {method}"
            )
            raise TypeError(msg)
        return model

    def _pixel_patch_features(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batched, _ = _ensure_bhwc(image.clip(0, 1))
        if mask is not None:
            batched_mask, _ = _ensure_bhwc(mask.clip(0, 1))
            batched = batched * batched_mask
        bchw = batched.permute(0, 3, 1, 2)
        resized = F.interpolate(
            bchw,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        patch_size = 16
        patches = F.unfold(resized, kernel_size=patch_size, stride=patch_size)
        patch_side = self.image_size // patch_size
        if patch_side * patch_side != patches.shape[-1]:
            msg = (
                "tuna_pixel_patch requires image_size divisible by 16; got "
                f"{self.image_size}."
            )
            raise ValueError(msg)
        patch_tokens = F.normalize(patches.transpose(1, 2), dim=2)
        patch_grid = patch_tokens.reshape(
            patch_tokens.shape[0],
            patch_side,
            patch_side,
            patch_tokens.shape[2],
        )
        return {
            "cls": patch_tokens.mean(dim=1),
            "patch": patch_grid.permute(0, 3, 1, 2),
        }

    def _prepare(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batched, _ = _ensure_bhwc(image.clip(0, 1))
        if mask is not None:
            batched_mask, _ = _ensure_bhwc(mask.clip(0, 1))
            batched = batched * batched_mask
        bchw = batched.permute(0, 3, 1, 2)
        resized = F.interpolate(
            bchw,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        return (resized - self.mean.to(resized)) / self.std.to(resized)

    def _features(
        self, image: torch.Tensor, mask: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        if self.method == "tuna_pixel_patch":
            return self._pixel_patch_features(image, mask)
        prepared = self._prepare(image, mask)
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(prepared)
            if isinstance(features, dict) and "x_norm_clstoken" in features:
                result = {"cls": features["x_norm_clstoken"]}
                patch_tokens = features.get("x_norm_patchtokens")
                if patch_tokens is not None:
                    token_count = patch_tokens.shape[1]
                    side = int(round(token_count**0.5))
                    if side * side == token_count:
                        patch = patch_tokens.reshape(
                            patch_tokens.shape[0],
                            side,
                            side,
                            patch_tokens.shape[2],
                        )
                        result["patch"] = patch.permute(0, 3, 1, 2)
                return result
        output = self.model(prepared)
        if not torch.is_tensor(output):
            msg = f"Unsupported feature output for {self.method}"
            raise TypeError(msg)
        if output.ndim > 2:
            output = output.flatten(start_dim=1)
        return {"cls": output}

    def compare(
        self,
        *,
        rgb_pred: torch.Tensor,
        rgb_gt: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        pred_features = self._features(rgb_pred, mask)
        with torch.no_grad():
            gt_features = self._features(rgb_gt, mask)
        cls_cosine = F.cosine_similarity(
            pred_features["cls"],
            gt_features["cls"],
            dim=1,
        )
        result = {
            "foundation_feature_cosine": cls_cosine.mean(),
            "foundation_feature_distance": (1.0 - cls_cosine).mean(),
        }
        if "patch" in pred_features and "patch" in gt_features:
            patch_cosine = F.cosine_similarity(
                pred_features["patch"],
                gt_features["patch"],
                dim=1,
            )
            patch_error = 1.0 - patch_cosine
            result["foundation_feature_patch_cosine"] = patch_cosine.mean()
            result["foundation_feature_patch_error"] = patch_error.mean()
            target_size = rgb_gt.shape[1:3]
            error_map = F.interpolate(
                patch_error[:, None],
                size=target_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            if mask is not None:
                error_map = error_map * mask
            result["foundation_feature_error_map"] = error_map
        return result


def _image_radius(
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return image-plane radius from the image center."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.sqrt(xx.square() + yy.square())


def _region_mask(
    radius: torch.Tensor,
    *,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Return a BHWC-compatible radial region mask."""
    return ((radius >= lower) & (radius < upper)).to(radius.dtype)[
        None, :, :, None
    ]


def _radial_residual_metrics(
    *,
    rgb_gt: torch.Tensor,
    rgb_pred: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, float]:
    """Compute residual diagnostics over image-center to fisheye-rim bands."""
    batched_gt, squeezed_gt = _ensure_bhwc(rgb_gt)
    batched_pred, squeezed_pred = _ensure_bhwc(rgb_pred)
    if squeezed_gt != squeezed_pred:
        raise RuntimeError(
            "GT and prediction rank mismatch in radial diagnostics."
        )

    height = batched_gt.shape[1]
    width = batched_gt.shape[2]
    radius = _image_radius(
        height=height,
        width=width,
        device=batched_gt.device,
        dtype=batched_gt.dtype,
    )

    low_gt = _box_blur_bhwc(batched_gt, 31)
    low_pred = _box_blur_bhwc(batched_pred, 31)
    high_abs = torch.abs((batched_pred - low_pred) - (batched_gt - low_gt))
    rgb_abs = torch.abs(batched_pred - batched_gt)
    grad_gt = _sobel_grad_bhwc(_rgb_to_luma(batched_gt))
    grad_pred = _sobel_grad_bhwc(_rgb_to_luma(batched_pred))
    grad_abs = torch.abs(grad_pred - grad_gt)

    batched_mask = mask
    if batched_mask is not None and batched_mask.ndim == 3:
        batched_mask = batched_mask.unsqueeze(0)
    if batched_mask is None:
        radius = radius / torch.clamp_min(radius.max(), 1e-6)
    else:
        valid_pixels = (batched_mask > 0.5).any(dim=0).squeeze(-1)
        valid_radius = radius[valid_pixels]
        if valid_radius.numel() > 0:
            radius = radius / torch.clamp_min(valid_radius.max(), 1e-6)
        else:
            radius = radius / torch.clamp_min(radius.max(), 1e-6)

    bands = {
        "center": (0.00, 0.33),
        "mid": (0.33, 0.66),
        "outer": (0.66, 0.90),
        "rim": (0.90, 1.01),
    }
    metrics = {}
    for band_name, (lower, upper) in bands.items():
        radial_mask = _region_mask(radius, lower=lower, upper=upper)
        combined_mask = (
            radial_mask if batched_mask is None else radial_mask * batched_mask
        )
        metrics[f"radial_{band_name}_rgb_l1"] = _masked_mean(
            rgb_abs,
            combined_mask,
        ).item()
        metrics[f"radial_{band_name}_high_freq_l1"] = _masked_mean(
            high_abs,
            combined_mask,
        ).item()
        metrics[f"radial_{band_name}_gradient_l1"] = _masked_mean(
            grad_abs,
            combined_mask,
        ).item()
        pred_edge = _masked_mean(grad_pred, combined_mask)
        gt_edge = _masked_mean(grad_gt, combined_mask)
        metrics[f"radial_{band_name}_edge_energy_ratio"] = (
            pred_edge / torch.clamp_min(gt_edge, 1e-12)
        ).item()
    return metrics


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
    fy = torch.fft.fftfreq(height, device=pred_luma.device).reshape(
        1, height, 1
    )
    fx = torch.fft.rfftfreq(width, device=pred_luma.device).reshape(1, 1, -1)
    high_mask = (torch.sqrt(fx.square() + fy.square()) > 0.18).to(
        pred_luma.dtype
    )
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
    fy = torch.fft.fftfreq(height, device=pred_luma.device).reshape(
        1, height, 1
    )
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
        ratios[name] = pred_band_energy / torch.clamp_min(
            gt_band_energy, 1e-12
        )
    return ratios


def _frequency_band_errors(
    rgb_pred: torch.Tensor,
    rgb_gt: torch.Tensor,
    mask: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Return normalized complex FFT error over radial frequency bands."""
    pred_luma = _rgb_to_luma(rgb_pred)
    gt_luma = _rgb_to_luma(rgb_gt)
    if mask is not None:
        pred_luma = pred_luma * mask
        gt_luma = gt_luma * mask

    pred_fft = torch.fft.rfft2(pred_luma.squeeze(-1), norm="ortho")
    gt_fft = torch.fft.rfft2(gt_luma.squeeze(-1), norm="ortho")
    height = pred_luma.shape[1]
    width = pred_luma.shape[2]
    fy = torch.fft.fftfreq(height, device=pred_luma.device).reshape(
        1, height, 1
    )
    fx = torch.fft.rfftfreq(width, device=pred_luma.device).reshape(1, 1, -1)
    radius = torch.sqrt(fx.square() + fy.square())
    error_energy = (pred_fft - gt_fft).abs().square()
    gt_energy = gt_fft.abs().square()

    bands = {
        "fft_error_ratio_low": (0.00, 0.08),
        "fft_error_ratio_mid": (0.08, 0.16),
        "fft_error_ratio_high": (0.16, 0.28),
        "fft_error_ratio_ultra": (0.28, float("inf")),
    }
    ratios = {}
    for name, (lower, upper) in bands.items():
        band_mask = (radius >= lower) & (radius < upper)
        band_mask = band_mask.to(pred_luma.dtype)
        band_error = (error_energy * band_mask).mean()
        band_gt = (gt_energy * band_mask).mean()
        ratios[name] = band_error / torch.clamp_min(band_gt, 1e-12)
    return ratios


def _top_fraction_mask(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    """Return a mask for the strongest valid responses by fixed fraction."""
    valid_values = values[valid_mask]
    if valid_values.numel() == 0:
        return torch.zeros_like(values, dtype=torch.bool)
    keep_count = max(1, int(round(valid_values.numel() * fraction)))
    keep_count = min(keep_count, valid_values.numel())
    threshold = torch.topk(valid_values, keep_count).values[-1]
    return (values >= threshold) & valid_mask


def _edge_alignment_metrics(
    *,
    rgb_gt: torch.Tensor,
    rgb_pred: torch.Tensor,
    mask: torch.Tensor | None,
    top_fraction: float = 0.15,
) -> dict[str, float]:
    """Measure whether predicted edge energy is spatially aligned to GT edges."""
    luma_gt = _rgb_to_luma(rgb_gt)
    luma_pred = _rgb_to_luma(rgb_pred)
    grad_gt = _sobel_grad_bhwc(luma_gt).squeeze(-1)
    grad_pred = _sobel_grad_bhwc(luma_pred).squeeze(-1)

    if mask is None:
        valid_mask = torch.ones_like(grad_gt, dtype=torch.bool)
    else:
        valid_mask = mask.squeeze(-1) > 0.5
    gt_edges = _top_fraction_mask(
        grad_gt,
        valid_mask,
        fraction=top_fraction,
    )
    pred_edges = _top_fraction_mask(
        grad_pred,
        valid_mask,
        fraction=top_fraction,
    )
    true_positive = (gt_edges & pred_edges).sum().to(rgb_gt.dtype)
    pred_count = pred_edges.sum().to(rgb_gt.dtype).clamp_min(1.0)
    gt_count = gt_edges.sum().to(rgb_gt.dtype).clamp_min(1.0)
    precision = true_positive / pred_count
    recall = true_positive / gt_count
    f1 = 2.0 * precision * recall / torch.clamp_min(precision + recall, 1e-12)

    edge_mask = gt_edges.unsqueeze(-1).to(rgb_gt.dtype)
    nonedge_mask = (valid_mask & ~gt_edges).unsqueeze(-1).to(rgb_gt.dtype)
    rgb_abs = torch.abs(rgb_pred - rgb_gt)
    low_gt = _box_blur_bhwc(rgb_gt, 31)
    low_pred = _box_blur_bhwc(rgb_pred, 31)
    high_abs = torch.abs((rgb_pred - low_pred) - (rgb_gt - low_gt))

    return {
        "edge_top15_precision": precision.item(),
        "edge_top15_recall": recall.item(),
        "edge_top15_f1": f1.item(),
        "edge_rgb_l1": _masked_mean(rgb_abs, edge_mask).item(),
        "nonedge_rgb_l1": _masked_mean(rgb_abs, nonedge_mask).item(),
        "edge_high_freq_l1": _masked_mean(high_abs, edge_mask).item(),
        "nonedge_high_freq_l1": _masked_mean(high_abs, nonedge_mask).item(),
    }


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
        geom_mean_scale = torch.clamp_min(scales.prod(dim=1), 1e-36).pow(
            1.0 / 3.0
        )
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
        projected_radius = (
            scales[in_front] * focal_mean / depth[in_front].clamp_min(1e-3)
        )
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
            "footprint_radius_lt_0p5_fraction": (projected_radius < 0.5)
            .float()
            .mean()
            .item(),
            "footprint_radius_gt_8_fraction": (projected_radius > 8.0)
            .float()
            .mean()
            .item(),
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
        "high_freq_l1": _masked_mean(
            torch.abs(high_pred - high_gt), mask
        ).item(),
        "gradient_l1": _masked_mean(
            torch.abs(grad_pred - grad_gt), mask
        ).item(),
        "laplacian_l1": _masked_mean(
            torch.abs(lap_pred - lap_gt), mask
        ).item(),
        "fft_high_energy_ratio": _frequency_high_ratio(
            rgb_pred, rgb_gt, mask
        ).item(),
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
    metrics.update(
        {
            key: value.item()
            for key, value in _frequency_band_errors(
                rgb_pred,
                rgb_gt,
                mask,
            ).items()
        }
    )
    metrics.update(
        _edge_alignment_metrics(
            rgb_gt=rgb_gt,
            rgb_pred=rgb_pred,
            mask=mask,
        )
    )
    metrics.update(
        _radial_residual_metrics(
            rgb_gt=rgb_gt,
            rgb_pred=rgb_pred,
            mask=mask,
        )
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
        header = np.full(
            (header_height, panel_width, 3), (0, 0, 0), dtype=np.uint8
        )
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
    gt_depth: torch.Tensor | None = None,
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
    # Diagnostic inversions: the binary `hit > 0` and `opacity > 0.01`
    # coverage panels go all-white once the model is dense (~12M splats),
    # carrying no information. Flip them to expose the *minority* pixels
    # that are diagnostically interesting — sparse-ray regions and
    # low-opacity regions, which usually mark missing geometry, edges,
    # and rim degeneracies.
    hits_sparse = (hit_counts < 4).to(rgb_gt.dtype)
    opacity_low = (pred_opacity < 0.5).to(rgb_gt.dtype)
    if valid_mask is not None:
        hits_sparse = hits_sparse * valid_mask
        opacity_low = opacity_low * valid_mask
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
        "hits_sparse": hits_sparse.expand_as(rgb_gt),
        "opacity_low": opacity_low.expand_as(rgb_gt),
    }
    if sky_mask is not None:
        tiles["sky_opacity_mask"] = sky_mask.clip(0, 1.0).expand_as(rgb_gt)
    if gt_depth is not None:
        # `_robust_jet_map` / `jet_map` expect the same shape `pred_dist`
        # has at this point — sliced last batch element, ``[H, W, 1]``.
        # Anything else silently produces a black panel.
        gt = gt_depth
        if gt.dim() == 4:
            gt = gt[0]
        if gt.dim() == 2:
            gt = gt.unsqueeze(-1)
        gt_valid = (gt > 0.0).to(gt.dtype)
        if valid_mask is not None and gt_valid.shape == valid_mask.shape:
            gt_valid = gt_valid * valid_mask
        if gt.shape[:2] != depth_rgb.shape[:2]:
            h = min(gt.shape[0], depth_rgb.shape[0])
            w = min(gt.shape[1], depth_rgb.shape[1])
            gt = gt[:h, :w]
            gt_valid = gt_valid[:h, :w]
        tiles["depth_gt"] = _robust_jet_map(gt, gt_valid)
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

    post_processing: nn.Module | None = None
    """ Post-processing module """

    post_processing_optimizers: list | None = None
    """ Optimizers for post-processing module """

    post_processing_schedulers: list | None = None
    """ Schedulers for post-processing module optimizers """

    camera_residual: CameraResidual | None = None
    """ Optional camera residual module """

    camera_residual_optimizer: torch.optim.Optimizer | None = None
    """ Optimizer for camera residual module """

    foundation_feature_probe: FoundationFeatureProbe | None = None
    """ Optional frozen foundation-model probe for feature diagnostics """

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
        self.init_foundation_feature_probe(conf)
        self.init_post_processing(conf)
        self.init_camera_residual(conf)
        self.setup_training(conf, self.model, self.train_dataset)
        self.init_experiments_tracking(conf)
        self.init_gui(
            conf,
            self.model,
            self.train_dataset,
            self.val_dataset,
            self.scene_bbox,
        )

    def init_dataloaders(self, conf: DictConfig):
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        train_dataset, val_dataset = datasets.make(
            name=conf.dataset.type, config=conf, ray_jitter=None
        )
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

        train_dataloader = MultiEpochsDataLoader(
            train_dataset, **train_dataloader_kwargs
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, **val_dataloader_kwargs
        )

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

    def init_scene_extents(
        self, train_dataset: BoundedMultiViewDataset
    ) -> None:
        scene_bbox: tuple[
            torch.Tensor, torch.Tensor
        ]  # Tuple of vec3 (min,max)
        scene_extent = train_dataset.get_scene_extent()
        scene_bbox = train_dataset.get_scene_bbox()
        self.scene_extent = scene_extent
        self.scene_bbox = scene_bbox

    def init_model(self, conf: DictConfig, scene_extent=None) -> None:
        """Initializes the gaussian model and the optix context"""
        self.model = MixtureOfGaussians(conf, scene_extent=scene_extent)

    def init_densification_and_pruning_strategy(
        self, conf: DictConfig
    ) -> None:
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
                raise ValueError(
                    f"unrecognized model.strategy {conf.strategy.method}"
                )

    def setup_training(
        self,
        conf: DictConfig,
        model: MixtureOfGaussians,
        train_dataset: BoundedMultiViewDataset,
    ):
        """Performs required steps to setup the optimization:
        1. Initialize the gaussian model fields: load previous weights from checkpoint, or initialize from scratch.
        2. Build BVH acceleration structure for gaussian model, if not loaded with checkpoint
        3. Set up the optimizer to optimize the gaussian model params
        4. Initialize the densification buffers in the densificaiton strategy
        """
        # Initialize
        if conf.resume:  # Load a checkpoint
            logger.info(
                f"🤸 Loading a pretrained checkpoint from {conf.resume}!"
            )
            checkpoint = torch.load(conf.resume, weights_only=False)
            model.init_from_checkpoint(checkpoint)
            self.strategy.init_densification_buffer(checkpoint)
            global_step = checkpoint["global_step"]

            # Restore post-processing state
            if (
                "post_processing" in checkpoint
                and self.post_processing is not None
            ):
                self.post_processing.load_state_dict(
                    checkpoint["post_processing"]["module"]
                )
                for opt, opt_state in zip(
                    self.post_processing_optimizers,
                    checkpoint["post_processing"]["optimizers"],
                    strict=False,
                ):
                    opt.load_state_dict(opt_state)
                for sched, sched_state in zip(
                    self.post_processing_schedulers,
                    checkpoint["post_processing"]["schedulers"],
                    strict=False,
                ):
                    sched.load_state_dict(sched_state)
                logger.info(
                    "📷 Post-processing state restored from checkpoint"
                )
            if (
                "camera_residual" in checkpoint
                and self.camera_residual is not None
            ):
                self.camera_residual.load_state_dict(
                    checkpoint["camera_residual"]["module"]
                )
                self.camera_residual_optimizer.load_state_dict(
                    checkpoint["camera_residual"]["optimizer"]
                )
                logger.info(
                    "📷 Camera residual state restored from checkpoint"
                )
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
            logger.info("🤸 Initiating new 3dgrut training..")
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
                    logger.info(
                        f"Initializing from accumulated point cloud: {ply_path}"
                    )
                    model.init_from_fused_point_cloud(
                        ply_path, observer_points
                    )
                case "point_cloud":
                    try:
                        ply_path = os.path.join(conf.path, "point_cloud.ply")
                        model.init_from_pretrained_point_cloud(ply_path)
                    except FileNotFoundError as e:
                        logger.error(e)
                        raise e
                case "checkpoint":
                    checkpoint = torch.load(
                        conf.initialization.path, weights_only=False
                    )
                    model.init_from_checkpoint(
                        checkpoint, setup_optimizer=False
                    )
                    if (
                        "post_processing" in checkpoint
                        and self.post_processing is not None
                    ):
                        self.post_processing.load_state_dict(
                            checkpoint["post_processing"]["module"]
                        )
                        logger.info(
                            "📷 Post-processing module restored from initialization checkpoint"
                        )
                case "lidar":
                    assert isinstance(train_dataset, datasets.NCoreDataset), (
                        "can only initialize from lidar with NCoreDataset"
                    )
                    pc = PointCloud.from_sequence(
                        list(
                            train_dataset.get_point_clouds(
                                step_frame=1, non_dynamic_points_only=True
                            )
                        ),
                        device="cpu",
                    )
                    if conf.initialization.num_points < len(pc.xyz_end):
                        # Deterministically random subsample points if there are more points than the specified number of gaussians
                        rng = torch.Generator().manual_seed(
                            conf.seed_initialization
                        )
                        idxs = torch.randperm(len(pc.xyz_end), generator=rng)[
                            : conf.initialization.num_points
                        ]
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
        self.n_epochs = int(
            (conf.n_iterations + len(train_dataset) - 1) / len(train_dataset)
        )

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
            ssim=StructuralSimilarityIndexMeasure(data_range=1.0).to(
                self.device
            ),
            lpips=LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=True
            ).to(self.device),
        )

    def init_foundation_feature_probe(self, conf: DictConfig) -> None:
        """Initialize optional frozen image feature probe."""
        feature_conf = conf.get("foundation_features", {})
        if not bool(feature_conf.get("enabled", False)):
            return
        method = str(feature_conf.get("method", "dinov2_vitl14"))
        repo_dir = str(feature_conf.get("repo_dir", ""))
        weights = str(feature_conf.get("weights", ""))
        image_size = int(feature_conf.get("image_size", 224))
        self.foundation_feature_probe = FoundationFeatureProbe(
            method=method,
            repo_dir=repo_dir,
            weights=weights,
            image_size=image_size,
        ).to(self.device)
        logger.info(f"🧠 Foundation feature probe enabled: {method}")

    def init_experiments_tracking(self, conf: DictConfig):
        # Initialize the tensorboard writer
        object_name = Path(conf.path).stem
        writer, out_dir, run_name = create_summary_writer(
            conf,
            object_name,
            conf.out_dir,
            conf.experiment_name,
            conf.use_wandb,
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
            n_distillation_steps = conf.post_processing.get(
                "n_distillation_steps", 5000
            )
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = (
                    main_training_steps / conf.n_iterations
                )
                controller_distillation = True
                self._distillation_start_step = main_training_steps
                logger.info(
                    f"📷 PPISP distillation mode: controller activates at step {main_training_steps}"
                )
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

            self.post_processing_optimizers = (
                self.post_processing.create_optimizers()
            )
            self.post_processing_schedulers = (
                self.post_processing.create_schedulers(
                    self.post_processing_optimizers,
                    max_optimization_iters=conf.n_iterations,
                )
            )

            logger.info(
                f"📷 {method.upper()} initialized: {num_cameras} cameras, {num_frames} frames"
            )
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
                use_residual_grid=conf.post_processing.get(
                    "use_residual_grid",
                    False,
                ),
                residual_grid_size=conf.post_processing.get(
                    "residual_grid_size",
                    32,
                ),
                residual_grid_max=conf.post_processing.get(
                    "residual_grid_max",
                    0.05,
                ),
                residual_grid_reg_lambda=conf.post_processing.get(
                    "residual_grid_reg_lambda",
                    0.01,
                ),
            ).to(self.device)

            self.post_processing_optimizers = (
                self.post_processing.create_optimizers()
            )
            self.post_processing_schedulers = (
                self.post_processing.create_schedulers(
                    self.post_processing_optimizers,
                    max_optimization_iters=conf.n_iterations,
                )
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
        self.camera_residual_optimizer = (
            self.camera_residual.create_optimizer()
        )
        logger.warning(
            "📷 CAMERA_RESIDUAL enabled. Current 3DGUT CUDA backward does not "
            "return ray/sensor gradients; monitor camera_residual/max_abs_grad."
        )

    def _validate_camera_residual_gradient(self, global_step: int) -> None:
        """Abort invalid camera-residual runs with no pose gradients."""
        if self.camera_residual is None:
            return
        camera_conf = self.conf.camera_residual
        if not bool(camera_conf.get("fail_on_zero_grad", True)):
            return
        fail_after_steps = int(camera_conf.get("fail_after_steps", 5))
        if global_step < fail_after_steps:
            return
        max_abs_grad = self.camera_residual.max_abs_grad()
        min_abs_grad = float(camera_conf.get("min_abs_grad", 1e-12))
        if max_abs_grad > min_abs_grad:
            return
        msg = (
            "Camera residual is enabled, but no gradient reached the SO3/SE3 "
            f"residual parameters by step {global_step}. "
            f"camera_residual/max_abs_grad={max_abs_grad:.3e}, "
            f"threshold={min_abs_grad:.3e}. "
            "This is not a valid camera finetuning run. Fix the tracer "
            "backward to propagate ray/sensor gradients, or disable "
            "camera_residual.fail_on_zero_grad only for an explicit diagnostic."
        )
        raise RuntimeError(msg)

    def _apply_camera_residual(self, gpu_batch):
        if self.camera_residual is None:
            return gpu_batch
        return self.camera_residual(gpu_batch)

    def _camera_residual_audit_axis_candidates(
        self,
        *,
        prefix: str,
        camera_idx: int | None,
    ) -> list[tuple[str, int | None, torch.Tensor, torch.Tensor]]:
        """Return fixed axis perturbations for finite-difference audit."""
        audit_conf = self.conf.camera_residual.finite_difference_audit
        rotation_step = float(audit_conf.rotation_step_rad)
        translation_step = float(audit_conf.translation_step_m)
        candidates = [
            (
                f"{prefix}_baseline",
                camera_idx,
                torch.zeros(3, device=self.device),
                torch.zeros(3, device=self.device),
            )
        ]
        axes = ("x", "y", "z")
        for axis_idx, axis_name in enumerate(axes):
            for sign in (-1.0, 1.0):
                rotation = torch.zeros(3, device=self.device)
                rotation[axis_idx] = sign * rotation_step
                candidates.append(
                    (
                        f"{prefix}_rot_{axis_name}_{sign:+.0f}",
                        camera_idx,
                        rotation,
                        torch.zeros(3, device=self.device),
                    )
                )
        for axis_idx, axis_name in enumerate(axes):
            for sign in (-1.0, 1.0):
                translation = torch.zeros(3, device=self.device)
                translation[axis_idx] = sign * translation_step
                candidates.append(
                    (
                        f"{prefix}_trans_{axis_name}_{sign:+.0f}",
                        camera_idx,
                        torch.zeros(3, device=self.device),
                        translation,
                    )
                )
        return candidates

    def _camera_residual_audit_candidates(
        self,
    ) -> list[tuple[str, int | None, torch.Tensor, torch.Tensor]]:
        """Return global and optional per-camera pose audit candidates."""
        candidates = self._camera_residual_audit_axis_candidates(
            prefix="global",
            camera_idx=None,
        )
        if not self.camera_residual.optimize_per_camera:
            return candidates
        camera_count = int(self.camera_residual.camera_rotation_raw.shape[0])
        for camera_idx in range(camera_count):
            candidates.extend(
                self._camera_residual_audit_axis_candidates(
                    prefix=f"camera_{camera_idx}",
                    camera_idx=camera_idx,
                )
            )
        return candidates

    @torch.no_grad()
    def run_camera_residual_finite_difference_audit(self) -> None:
        """Evaluate fixed SO3/SE3 residual nudges without ray gradients."""
        if self.camera_residual is None:
            raise RuntimeError(
                "camera_residual.finite_difference_audit requires "
                "camera_residual.enabled=true."
            )
        audit_conf = self.conf.camera_residual.finite_difference_audit
        max_views = int(audit_conf.max_views)
        if max_views <= 0:
            raise ValueError(
                "camera_residual.finite_difference_audit.max_views must be positive."
            )

        original_rotation = (
            self.camera_residual.global_rotation_raw.detach().clone()
        )
        original_translation = (
            self.camera_residual.global_translation_raw.detach().clone()
        )
        rows = []
        try:
            for (
                name,
                camera_idx,
                rotation,
                translation,
            ) in self._camera_residual_audit_candidates():
                if camera_idx is None:
                    self.camera_residual.set_global_delta(
                        rotation=rotation,
                        translation=translation,
                    )
                else:
                    self.camera_residual.set_camera_delta(
                        camera_idx=camera_idx,
                        rotation=rotation,
                        translation=translation,
                    )
                psnr_values = []
                masked_psnr_values = []
                logger.info(
                    f"Camera residual finite-difference candidate: {name}"
                )
                for _, batch_idx in enumerate(self.val_dataloader):
                    gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(
                        batch_idx
                    )
                    if (
                        camera_idx is not None
                        and gpu_batch.camera_idx != camera_idx
                    ):
                        continue
                    if len(psnr_values) >= max_views:
                        break
                    gpu_batch = self._apply_camera_residual(gpu_batch)
                    outputs = self.model(gpu_batch, train=False)
                    if self.post_processing is not None:
                        outputs = apply_post_processing(
                            self.post_processing,
                            outputs,
                            gpu_batch,
                            training=False,
                        )
                    rgb_error = torch.square(
                        outputs["pred_rgb"] - gpu_batch.rgb_gt
                    )
                    psnr_values.append(
                        (
                            -10.0
                            * torch.log10(
                                torch.clamp_min(rgb_error.mean(), 1e-12)
                            )
                        ).item()
                    )
                    if gpu_batch.mask is not None:
                        mask = gpu_batch.mask
                        masked_error = rgb_error * mask
                        denominator = torch.clamp_min(
                            mask.sum() * gpu_batch.rgb_gt.shape[-1], 1.0
                        )
                        masked_mse = masked_error.sum() / denominator
                        masked_psnr_values.append(
                            (
                                -10.0
                                * torch.log10(
                                    torch.clamp_min(masked_mse, 1e-12)
                                )
                            ).item()
                        )
                mean_psnr = (
                    float(np.mean(psnr_values))
                    if psnr_values
                    else float("nan")
                )
                mean_masked_psnr = (
                    float(np.mean(masked_psnr_values))
                    if masked_psnr_values
                    else float("nan")
                )
                rows.append(
                    {
                        "candidate": name,
                        "camera_idx": -1 if camera_idx is None else camera_idx,
                        "rotation_x": float(rotation[0].item()),
                        "rotation_y": float(rotation[1].item()),
                        "rotation_z": float(rotation[2].item()),
                        "translation_x": float(translation[0].item()),
                        "translation_y": float(translation[1].item()),
                        "translation_z": float(translation[2].item()),
                        "mean_psnr": mean_psnr,
                        "mean_masked_psnr": mean_masked_psnr,
                    }
                )
        finally:
            with torch.no_grad():
                self.camera_residual.global_rotation_raw.copy_(
                    original_rotation
                )
                self.camera_residual.global_translation_raw.copy_(
                    original_translation
                )
                self.camera_residual.camera_rotation_raw.zero_()
                self.camera_residual.camera_translation_raw.zero_()

        score_key = "mean_masked_psnr"
        if not rows or all(np.isnan(row[score_key]) for row in rows):
            score_key = "mean_psnr"
        finite_rows = [row for row in rows if not np.isnan(row[score_key])]
        best = max(finite_rows, key=lambda row: row[score_key])
        if best["camera_idx"] >= 0:
            baseline_name = f"camera_{best['camera_idx']}_baseline"
        else:
            baseline_name = "global_baseline"
        baseline = next(
            row for row in rows if row["candidate"] == baseline_name
        )
        for row in rows:
            display_row = {
                key: str(value) if isinstance(value, int) else value
                for key, value in row.items()
            }
            logger.log_table(
                f"Camera Residual Finite-Difference Audit - {row['candidate']}",
                record=display_row,
            )
        output_path = os.path.join(
            self.tracking.output_dir,
            "camera_residual_finite_difference_audit.json",
        )
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "score_key": score_key,
                    "baseline": baseline,
                    "best": best,
                    "rows": rows,
                },
                fp,
                indent=2,
            )
        logger.info(
            "Camera residual finite-difference best: "
            f"{best['candidate']} {score_key}={best[score_key]:.6f}; "
            f"baseline={baseline[score_key]:.6f}; "
            f"delta={best[score_key] - baseline[score_key]:.6f}"
        )
        logger.info(
            f"Camera residual finite-difference audit JSON: {output_path}"
        )

    def _camera_intrinsics_audit_candidates(self) -> list[dict[str, Any]]:
        """Return renderer-side RATIONAL intrinsics perturbation candidates."""
        audit_conf = self.conf.camera_intrinsics_audit
        candidates = [
            {"name": "baseline", "target": "", "index": -1, "scale": 0.0}
        ]
        candidate_specs = (
            (
                "fx",
                "focal_length",
                0,
                float(audit_conf.focal_delta_fraction),
                True,
            ),
            (
                "fy",
                "focal_length",
                1,
                float(audit_conf.focal_delta_fraction),
                True,
            ),
            (
                "cx",
                "principal_point",
                0,
                float(audit_conf.principal_delta_px),
                False,
            ),
            (
                "cy",
                "principal_point",
                1,
                float(audit_conf.principal_delta_px),
                False,
            ),
            ("skew", "skew", 0, float(audit_conf.skew_delta_px), False),
            (
                "b1",
                "numerator_coeffs",
                0,
                float(audit_conf.numerator_delta),
                False,
            ),
            (
                "d1",
                "denominator_coeffs",
                0,
                float(audit_conf.denominator_delta),
                False,
            ),
            ("a1", "affine_coeffs", 0, float(audit_conf.affine_delta), False),
            (
                "p1",
                "tangential_coeffs",
                0,
                float(audit_conf.tangential_delta),
                False,
            ),
            (
                "p2",
                "tangential_coeffs",
                1,
                float(audit_conf.tangential_delta),
                False,
            ),
        )
        for label, target, index, scale, relative in candidate_specs:
            for sign in (-1.0, 1.0):
                candidates.append(
                    {
                        "name": f"{label}_{sign:+.0f}",
                        "target": target,
                        "index": index,
                        "scale": sign * scale,
                        "relative": relative,
                    }
                )
        return candidates

    def _with_intrinsics_audit_delta(
        self, gpu_batch, candidate: dict[str, Any]
    ):
        """Return a batch with one renderer-side RATIONAL intrinsic delta."""
        params = gpu_batch.intrinsics_RationalCameraModelParameters
        if params is None:
            raise RuntimeError(
                "camera_intrinsics_audit requires RATIONAL cameras."
            )
        if candidate["name"] == "baseline":
            return gpu_batch

        updated_params = {}
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                updated_params[key] = value.copy()
            else:
                updated_params[key] = value

        target = candidate["target"]
        index = int(candidate["index"])
        scale = float(candidate["scale"])
        if target == "skew":
            updated_params[target] = float(updated_params[target]) + scale
        else:
            values = np.asarray(
                updated_params[target], dtype=np.float32
            ).copy()
            delta = scale
            if bool(candidate.get("relative", False)):
                delta = float(values[index]) * scale
            values[index] = values[index] + delta
            updated_params[target] = values
        return replace(
            gpu_batch, intrinsics_RationalCameraModelParameters=updated_params
        )

    @torch.no_grad()
    def _camera_intrinsics_candidate_metrics(
        self,
        *,
        camera_idx: int,
        candidate: dict[str, Any],
        max_views: int,
    ) -> dict[str, float | int | str]:
        """Evaluate one intrinsics candidate on one camera head."""
        metrics = []
        for batch_idx in self.val_dataloader:
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(
                batch_idx
            )
            if int(gpu_batch.camera_idx) != camera_idx:
                continue
            audited_batch = self._with_intrinsics_audit_delta(
                gpu_batch, candidate
            )
            outputs = self.model(audited_batch, train=False)
            if self.post_processing is not None:
                outputs = apply_post_processing(
                    self.post_processing,
                    outputs,
                    audited_batch,
                    training=False,
                )
            rgb_gt = audited_batch.rgb_gt
            rgb_pred = outputs["pred_rgb"]
            rgb_error = torch.square(rgb_pred - rgb_gt)
            if audited_batch.mask is not None:
                mask = audited_batch.mask
                rgb_error = rgb_error * mask
                denominator = torch.clamp_min(
                    mask.sum() * rgb_gt.shape[-1], 1.0
                )
            else:
                denominator = torch.tensor(
                    rgb_gt.numel(),
                    dtype=rgb_gt.dtype,
                    device=rgb_gt.device,
                )
            mse = rgb_error.sum() / denominator
            diagnostics = _diagnostic_metrics(
                rgb_gt=rgb_gt,
                rgb_pred=rgb_pred,
                mask=audited_batch.mask,
            )
            metrics.append(
                {
                    "masked_psnr": (
                        -10.0 * torch.log10(torch.clamp_min(mse, 1e-12))
                    ).item(),
                    "gradient_l1": diagnostics["gradient_l1"],
                    "high_ratio": diagnostics["fft_energy_ratio_high"],
                    "rim_rgb_l1": diagnostics["radial_rim_rgb_l1"],
                    "rim_edge_ratio": diagnostics[
                        "radial_rim_edge_energy_ratio"
                    ],
                    "center_edge_ratio": diagnostics[
                        "radial_center_edge_energy_ratio"
                    ],
                }
            )
            if len(metrics) >= max_views:
                break
        if not metrics:
            raise RuntimeError(
                f"No validation views found for camera {camera_idx}."
            )
        return {
            "camera_idx": camera_idx,
            "candidate": str(candidate["name"]),
            "num_views": len(metrics),
            "masked_psnr": float(np.mean([m["masked_psnr"] for m in metrics])),
            "gradient_l1": float(np.mean([m["gradient_l1"] for m in metrics])),
            "high_ratio": float(np.mean([m["high_ratio"] for m in metrics])),
            "rim_rgb_l1": float(np.mean([m["rim_rgb_l1"] for m in metrics])),
            "rim_edge_ratio": float(
                np.mean([m["rim_edge_ratio"] for m in metrics])
            ),
            "center_edge_ratio": float(
                np.mean([m["center_edge_ratio"] for m in metrics])
            ),
        }

    @torch.no_grad()
    def run_camera_intrinsics_finite_difference_audit(self) -> None:
        """Rank small renderer-side per-camera RATIONAL intrinsics nudges."""
        audit_conf = self.conf.camera_intrinsics_audit
        max_views = int(audit_conf.max_views_per_camera)
        if max_views <= 0:
            raise ValueError(
                "camera_intrinsics_audit.max_views_per_camera must be positive."
            )
        camera_indices = [int(idx) for idx in audit_conf.camera_indices]
        rows = []
        for camera_idx in camera_indices:
            logger.info(f"Camera intrinsics audit: camera {camera_idx}")
            for candidate in self._camera_intrinsics_audit_candidates():
                rows.append(
                    self._camera_intrinsics_candidate_metrics(
                        camera_idx=camera_idx,
                        candidate=candidate,
                        max_views=max_views,
                    )
                )

        baselines = {
            int(row["camera_idx"]): row
            for row in rows
            if row["candidate"] == "baseline"
        }
        for row in rows:
            baseline = baselines[int(row["camera_idx"])]
            row["delta_masked_psnr"] = float(row["masked_psnr"]) - float(
                baseline["masked_psnr"]
            )
            row["delta_rim_edge_ratio"] = float(row["rim_edge_ratio"]) - float(
                baseline["rim_edge_ratio"]
            )
            row["delta_rim_rgb_l1"] = float(row["rim_rgb_l1"]) - float(
                baseline["rim_rgb_l1"]
            )

        ranked_rows = sorted(
            rows,
            key=lambda row: (
                int(row["camera_idx"]),
                -float(row["delta_masked_psnr"]),
            ),
        )
        report = {
            "note": (
                "Renderer-side RATIONAL intrinsics audit. Rays are still cached "
                "dataset rays; positive candidates require a follow-up path that "
                "regenerates rays from corrected intrinsics."
            ),
            "max_views_per_camera": max_views,
            "rows": ranked_rows,
        }
        output_path = os.path.join(
            self.tracking.output_dir,
            "camera_intrinsics_finite_difference_audit.json",
        )
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        logger.info(
            f"Camera intrinsics finite-difference audit JSON: {output_path}"
        )
        for camera_idx in camera_indices:
            best = next(
                row
                for row in ranked_rows
                if int(row["camera_idx"]) == camera_idx
            )
            logger.log_table(
                f"📷 Best camera {camera_idx} intrinsics audit candidate",
                record={
                    "candidate": str(best["candidate"]),
                    "num_views": str(best["num_views"]),
                    "masked_psnr": float(best["masked_psnr"]),
                    "delta_masked_psnr": float(best["delta_masked_psnr"]),
                    "rim_edge_ratio": float(best["rim_edge_ratio"]),
                    "delta_rim_edge_ratio": float(
                        best["delta_rim_edge_ratio"]
                    ),
                    "rim_rgb_l1": float(best["rim_rgb_l1"]),
                    "delta_rim_rgb_l1": float(best["delta_rim_rgb_l1"]),
                },
            )
        if self.conf.use_wandb:
            wandb.log(
                {
                    "diagnostics/camera_intrinsics_audit": wandb.Table(
                        data=[
                            [
                                row["camera_idx"],
                                row["candidate"],
                                row["num_views"],
                                row["masked_psnr"],
                                row["delta_masked_psnr"],
                                row["rim_edge_ratio"],
                                row["delta_rim_edge_ratio"],
                                row["rim_rgb_l1"],
                                row["delta_rim_rgb_l1"],
                                row["gradient_l1"],
                                row["high_ratio"],
                            ]
                            for row in ranked_rows
                        ],
                        columns=[
                            "camera_idx",
                            "candidate",
                            "num_views",
                            "masked_psnr",
                            "delta_masked_psnr",
                            "rim_edge_ratio",
                            "delta_rim_edge_ratio",
                            "rim_rgb_l1",
                            "delta_rim_rgb_l1",
                            "gradient_l1",
                            "high_ratio",
                        ],
                    )
                },
                step=self.global_step,
            )

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
        iteration: int | None = None,
    ) -> dict[str, int | float]:
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

        is_compute_train_hit_metrics = (split == "training") and (
            step % self.conf.writer.hit_stat_frequency == 0
        )
        is_compute_validation_metrics = split == "validation"

        if is_compute_train_hit_metrics or is_compute_validation_metrics:
            metrics["hits_mean"] = outputs["hits_count"].mean().item()
            metrics["hits_std"] = outputs["hits_count"].std().item()
            metrics["hits_min"] = outputs["hits_count"].min().item()
            metrics["hits_max"] = outputs["hits_count"].max().item()
            stats_mask = gpu_batch.mask if gpu_batch.mask is not None else None
            metrics.update(
                _scalar_tensor_stats(
                    outputs["pred_opacity"],
                    "pred_opacity",
                    stats_mask,
                )
            )
            metrics.update(
                _scalar_tensor_stats(
                    outputs["pred_dist"],
                    "pred_dist",
                    stats_mask,
                )
            )
            hit_mask = (outputs["hits_count"] > 0).float()
            opacity_mask = (outputs["pred_opacity"] > 0.01).float()
            if gpu_batch.mask is not None:
                valid = gpu_batch.mask
                valid_denominator = torch.clamp_min(valid.sum(), 1.0)
                metrics["valid_hit_coverage"] = (
                    (hit_mask * valid).sum() / valid_denominator
                ).item()
                metrics["valid_opacity_coverage"] = (
                    (opacity_mask * valid).sum() / valid_denominator
                ).item()
            else:
                metrics["valid_hit_coverage"] = hit_mask.mean().item()
                metrics["valid_opacity_coverage"] = opacity_mask.mean().item()

        if is_compute_validation_metrics:
            metrics["camera_idx"] = int(gpu_batch.camera_idx)
            with torch.cuda.nvtx.range("criterions_psnr"):
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

            with torch.cuda.nvtx.range("criterions_ssim"):
                metrics["ssim"] = ssim(pred_rgb_full, rgb_gt_full).item()
            with torch.cuda.nvtx.range("criterions_lpips"):
                metrics["lpips"] = lpips(
                    pred_rgb_full_clipped, rgb_gt_full
                ).item()

            foundation_feature_error_map = None
            feature_conf = self.conf.get("foundation_features", {})
            use_logged_views_only = bool(
                feature_conf.get("eval_on_logged_views_only", True)
            )
            is_logged_view = iteration in self._validation_log_image_views()
            should_compute_features = (
                self.foundation_feature_probe is not None
                and (not use_logged_views_only or is_logged_view)
            )
            if should_compute_features:
                feature_metrics = self.foundation_feature_probe.compare(
                    rgb_pred=rgb_pred.clip(0, 1),
                    rgb_gt=rgb_gt.clip(0, 1),
                    mask=gpu_batch.mask,
                )
                for metric_name, metric_value in feature_metrics.items():
                    if metric_name == "foundation_feature_error_map":
                        foundation_feature_error_map = metric_value
                    else:
                        metrics[metric_name] = metric_value.detach().item()

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
                    max_samples=int(
                        self.conf.writer.get("footprint_sample_count", 200000)
                    ),
                )
            )

            if is_logged_view:
                mask = (
                    gpu_batch.mask[-1] if gpu_batch.mask is not None else None
                )
                sky_mask = (
                    gpu_batch.sky_mask[-1]
                    if gpu_batch.sky_mask is not None
                    else None
                )
                metrics["eval_image_path"] = gpu_batch.image_path
                gt_depth_for_tile = None
                dense_depth_dir = (
                    self.conf.loss.get("dense_depth_dir", "")
                    if hasattr(self.conf.loss, "get")
                    else getattr(self.conf.loss, "dense_depth_dir", "")
                )
                if dense_depth_dir:
                    image_path = getattr(gpu_batch, "image_path", "")
                    stem = (
                        os.path.splitext(os.path.basename(image_path))[0]
                        if image_path
                        else ""
                    )
                    cache = getattr(self, "_dense_depth_cache", None)
                    if cache is None:
                        cache = {}
                        self._dense_depth_cache = cache
                    if stem and stem in cache:
                        gt_depth_for_tile = cache[stem]
                    elif stem:
                        npy_path = os.path.join(dense_depth_dir, f"{stem}.npy")
                        if os.path.exists(npy_path):
                            gt_arr = np.load(npy_path).astype("float32")
                            gt_depth_for_tile = torch.from_numpy(gt_arr).to(
                                self.device
                            )
                            cache[stem] = gt_depth_for_tile
                metrics["img_eval_tiles"] = _make_validation_image_tiles(
                    rgb_gt=gpu_batch.rgb_gt[-1],
                    rgb_pred=outputs["pred_rgb"][-1],
                    pred_dist=outputs["pred_dist"][-1],
                    pred_opacity=outputs["pred_opacity"][-1],
                    hit_counts=outputs["hits_count"][-1],
                    mask=mask,
                    sky_mask=sky_mask,
                    max_hit_count=self.conf.writer.max_num_hits,
                    gt_depth=gt_depth_for_tile,
                )
                if foundation_feature_error_map is not None and bool(
                    feature_conf.get("log_error_tile", True)
                ):
                    error_map = foundation_feature_error_map[-1]
                    metrics["img_eval_tiles"]["foundation_feature_error"] = (
                        _robust_jet_map(error_map, mask, quantile=0.95)
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
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
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
            with torch.cuda.nvtx.range("loss-l1"):
                loss_l1 = torch.abs(rgb_pred - rgb_gt).sum() / loss_denominator
                lambda_l1 = self.conf.loss.lambda_l1

        # L2 loss
        loss_l2 = torch.zeros(1, device=self.device)
        lambda_l2 = 0.0
        if self.conf.loss.use_l2:
            with torch.cuda.nvtx.range("loss-l2"):
                squared_error = torch.square(rgb_pred - rgb_gt)
                loss_l2 = squared_error.sum() / loss_denominator
                lambda_l2 = self.conf.loss.lambda_l2

        # DSSIM loss
        loss_ssim = torch.zeros(1, device=self.device)
        lambda_ssim = 0.0
        if self.conf.loss.use_ssim:
            with torch.cuda.nvtx.range("loss-ssim"):
                rgb_gt_full = torch.permute(rgb_gt, (0, 3, 1, 2))
                pred_rgb_full = torch.permute(rgb_pred, (0, 3, 1, 2))
                loss_ssim = 1.0 - ssim(pred_rgb_full, rgb_gt_full)
                lambda_ssim = self.conf.loss.lambda_ssim

        loss_foundation_feature = torch.zeros(1, device=self.device)
        lambda_foundation_feature = 0.0
        if self.conf.loss.use_foundation_feature:
            if self.foundation_feature_probe is None:
                msg = (
                    "loss.use_foundation_feature requires "
                    "foundation_features.enabled=true."
                )
                raise RuntimeError(msg)
            with torch.cuda.nvtx.range("loss-foundation-feature"):
                feature_metrics = self.foundation_feature_probe.compare(
                    rgb_pred=rgb_pred.clip(0, 1),
                    rgb_gt=rgb_gt.clip(0, 1),
                    mask=mask,
                )
                loss_foundation_feature = feature_metrics[
                    "foundation_feature_distance"
                ]
                if "foundation_feature_patch_error" in feature_metrics:
                    loss_foundation_feature = 0.5 * (
                        loss_foundation_feature
                        + feature_metrics["foundation_feature_patch_error"]
                    )
                lambda_foundation_feature = (
                    self.conf.loss.lambda_foundation_feature
                )

        # Opacity regularization
        loss_opacity = torch.zeros(1, device=self.device)
        lambda_opacity = 0.0
        if self.conf.loss.use_opacity:
            with torch.cuda.nvtx.range("loss-opacity"):
                loss_opacity = torch.abs(self.model.get_density()).mean()
                lambda_opacity = self.conf.loss.lambda_opacity

        # Scale regularization
        loss_scale = torch.zeros(1, device=self.device)
        lambda_scale = 0.0
        if self.conf.loss.use_scale:
            with torch.cuda.nvtx.range("loss-scale"):
                loss_scale = torch.abs(self.model.get_scale()).mean()
                lambda_scale = self.conf.loss.lambda_scale

        loss_sky_opacity = torch.zeros(1, device=self.device)
        lambda_sky_opacity = 0.0
        if self.conf.loss.use_sky_opacity:
            if gpu_batch.sky_mask is None:
                raise RuntimeError(
                    "loss.use_sky_opacity requires dataset.sky_mask_folder."
                )
            with torch.cuda.nvtx.range("loss-sky-opacity"):
                sky_mask = gpu_batch.sky_mask
                sky_denominator = torch.clamp(sky_mask.sum(), min=1.0)
                loss_sky_opacity = (
                    outputs["pred_opacity"] * sky_mask
                ).sum() / sky_denominator
                lambda_sky_opacity = self.conf.loss.lambda_sky_opacity

        # Dense depth supervision (Phase 3c: DA3 z-extended)
        loss_dense_depth = torch.zeros(1, device=self.device)
        lambda_dense_depth = 0.0
        if self.conf.loss.use_dense_depth:
            dense_depth_dir = self.conf.loss.dense_depth_dir
            if not dense_depth_dir:
                raise RuntimeError(
                    "loss.use_dense_depth requires loss.dense_depth_dir."
                )
            with torch.cuda.nvtx.range("loss-dense-depth"):
                image_path = getattr(gpu_batch, "image_path", "")
                stem = (
                    os.path.splitext(os.path.basename(image_path))[0]
                    if image_path
                    else ""
                )
                cache = getattr(self, "_dense_depth_cache", None)
                if cache is None:
                    cache = {}
                    self._dense_depth_cache = cache
                gt_depth = cache.get(stem) if stem else None
                if gt_depth is None and stem:
                    npy_path = os.path.join(dense_depth_dir, f"{stem}.npy")
                    if os.path.exists(npy_path):
                        gt_arr = np.load(npy_path).astype("float32")
                        gt_depth = torch.from_numpy(gt_arr).to(self.device)
                        cache[stem] = gt_depth
                if gt_depth is not None:
                    pred_dist = outputs["pred_dist"]
                    if pred_dist.dim() == 4 and pred_dist.shape[-1] == 1:
                        pred_dist_2d = pred_dist[0, ..., 0]
                    else:
                        pred_dist_2d = pred_dist[0]
                    # `pred_dist` is hit distance ALONG the ray; `gt_depth`
                    # is z-depth (DA3 inferred at the pinhole proxy then
                    # remapped to fisheye coords). For ~180° fisheyes those
                    # differ by a per-pixel `cos(theta)` factor that ranges
                    # from 1.0 at the optical axis to ~0.17 near the rim;
                    # without this conversion the loss compares apples to
                    # oranges and silently fights the cosine rather than
                    # the geometry.
                    rays_dir = getattr(gpu_batch, "rays_dir", None)
                    if (
                        rays_dir is not None
                        and not getattr(
                            gpu_batch, "rays_in_world_space", False
                        )
                        and rays_dir.dim() == 4
                        and rays_dir.shape[-1] == 3
                    ):
                        cos_theta = rays_dir[0, ..., 2]
                        if cos_theta.shape == pred_dist_2d.shape:
                            pred_dist_2d = pred_dist_2d * cos_theta
                    if pred_dist_2d.shape != gt_depth.shape:
                        # Renderer sometimes pads/crops; align to GT shape.
                        h = min(pred_dist_2d.shape[0], gt_depth.shape[0])
                        w = min(pred_dist_2d.shape[1], gt_depth.shape[1])
                        pred_dist_2d = pred_dist_2d[:h, :w]
                        gt_depth = gt_depth[:h, :w]
                    valid_mask = gt_depth > 0.0
                    if mask is not None:
                        m = mask
                        if m.dim() == 4:
                            m = m[0]
                        m = m.squeeze(-1) > 0.5
                        if m.shape != valid_mask.shape:
                            mh = min(m.shape[0], valid_mask.shape[0])
                            mw = min(m.shape[1], valid_mask.shape[1])
                            m = m[:mh, :mw]
                            valid_mask = valid_mask[:mh, :mw]
                            pred_dist_2d = pred_dist_2d[:mh, :mw]
                            gt_depth = gt_depth[:mh, :mw]
                        valid_mask = valid_mask & m
                    loss_dense_depth = dense_depth_l1_loss(
                        pred_dist_2d.unsqueeze(0),
                        gt_depth.unsqueeze(0),
                        valid_mask.unsqueeze(0),
                    )
                lambda_dense_depth = self.conf.loss.lambda_dense_depth

        # Dense depth gradient supervision (Phase 3c-redesign)
        loss_dense_depth_gradient = torch.zeros(1, device=self.device)
        lambda_dense_depth_gradient = 0.0
        if self.conf.loss.use_dense_depth_gradient:
            dense_depth_dir = self.conf.loss.dense_depth_dir
            if not dense_depth_dir:
                raise RuntimeError(
                    "loss.use_dense_depth_gradient requires loss.dense_depth_dir."
                )
            with torch.cuda.nvtx.range("loss-dense-depth-gradient"):
                image_path = getattr(gpu_batch, "image_path", "")
                stem = (
                    os.path.splitext(os.path.basename(image_path))[0]
                    if image_path
                    else ""
                )
                cache = getattr(self, "_dense_depth_cache", None)
                if cache is None:
                    cache = {}
                    self._dense_depth_cache = cache
                gt_depth = cache.get(stem) if stem else None
                if gt_depth is None and stem:
                    npy_path = os.path.join(dense_depth_dir, f"{stem}.npy")
                    if os.path.exists(npy_path):
                        gt_arr = np.load(npy_path).astype("float32")
                        gt_depth = torch.from_numpy(gt_arr).to(self.device)
                        cache[stem] = gt_depth
                if gt_depth is not None:
                    pred_dist = outputs["pred_dist"]
                    if pred_dist.dim() == 4 and pred_dist.shape[-1] == 1:
                        pred_dist_2d = pred_dist[0, ..., 0]
                    else:
                        pred_dist_2d = pred_dist[0]
                    # `pred_dist` is hit distance ALONG the ray; `gt_depth`
                    # is z-depth (DA3 inferred at the pinhole proxy then
                    # remapped to fisheye coords). For ~180° fisheyes those
                    # differ by a per-pixel `cos(theta)` factor that ranges
                    # from 1.0 at the optical axis to ~0.17 near the rim;
                    # without this conversion the loss compares apples to
                    # oranges and silently fights the cosine rather than
                    # the geometry.
                    rays_dir = getattr(gpu_batch, "rays_dir", None)
                    if (
                        rays_dir is not None
                        and not getattr(
                            gpu_batch, "rays_in_world_space", False
                        )
                        and rays_dir.dim() == 4
                        and rays_dir.shape[-1] == 3
                    ):
                        cos_theta = rays_dir[0, ..., 2]
                        if cos_theta.shape == pred_dist_2d.shape:
                            pred_dist_2d = pred_dist_2d * cos_theta
                    if pred_dist_2d.shape != gt_depth.shape:
                        h = min(pred_dist_2d.shape[0], gt_depth.shape[0])
                        w = min(pred_dist_2d.shape[1], gt_depth.shape[1])
                        pred_dist_2d = pred_dist_2d[:h, :w]
                        gt_depth = gt_depth[:h, :w]
                    valid_mask = gt_depth > 0.0
                    if mask is not None:
                        m = mask
                        if m.dim() == 4:
                            m = m[0]
                        m = m.squeeze(-1) > 0.5
                        if m.shape != valid_mask.shape:
                            mh = min(m.shape[0], valid_mask.shape[0])
                            mw = min(m.shape[1], valid_mask.shape[1])
                            m = m[:mh, :mw]
                            valid_mask = valid_mask[:mh, :mw]
                            pred_dist_2d = pred_dist_2d[:mh, :mw]
                            gt_depth = gt_depth[:mh, :mw]
                        valid_mask = valid_mask & m
                    loss_dense_depth_gradient = dense_depth_gradient_l1_loss(
                        pred_dist_2d.unsqueeze(0),
                        gt_depth.unsqueeze(0),
                        valid_mask.unsqueeze(0),
                    )
                lambda_dense_depth_gradient = (
                    self.conf.loss.lambda_dense_depth_gradient
                )

        # Equirect consistency loss (Phase 3b: RoMa fisheye→equirect warps)
        loss_equirect_consistency = torch.zeros(1, device=self.device)
        lambda_equirect_consistency = 0.0
        if self.conf.loss.use_equirect_consistency:
            warp_dir = self.conf.loss.equirect_warp_dir
            image_dir = self.conf.loss.equirect_image_dir
            if not warp_dir or not image_dir:
                raise RuntimeError(
                    "loss.use_equirect_consistency requires "
                    "loss.equirect_warp_dir and loss.equirect_image_dir."
                )
            with torch.cuda.nvtx.range("loss-equirect-consistency"):
                image_path = getattr(gpu_batch, "image_path", "")
                stem = (
                    os.path.splitext(os.path.basename(image_path))[0]
                    if image_path
                    else ""
                )
                warp_cache = getattr(self, "_equirect_warp_cache", None)
                if warp_cache is None:
                    warp_cache = {}
                    self._equirect_warp_cache = warp_cache
                eq_image_cache = getattr(self, "_equirect_image_cache", None)
                if eq_image_cache is None:
                    eq_image_cache = {}
                    self._equirect_image_cache = eq_image_cache
                warp_overlap = warp_cache.get(stem) if stem else None
                if warp_overlap is None and stem:
                    npz_path = os.path.join(warp_dir, f"{stem}.npz")
                    if os.path.exists(npz_path):
                        payload = np.load(npz_path)
                        warp_np = payload["warp_AB"].astype("float32")
                        overlap_np = payload["overlap_AB"].astype("float32")
                        warp_t = torch.from_numpy(warp_np).to(self.device)
                        overlap_t = torch.from_numpy(overlap_np).to(
                            self.device
                        )
                        warp_cache[stem] = (warp_t, overlap_t)
                        warp_overlap = (warp_t, overlap_t)
                # Frame index from stem (e.g. "front_0009" -> "0009").
                eq_image_t = None
                if stem:
                    parts = stem.split("_")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        frame_key = parts[-1]
                        eq_image_t = eq_image_cache.get(frame_key)
                        if eq_image_t is None:
                            eq_path = os.path.join(
                                image_dir, f"{frame_key}.png"
                            )
                            if os.path.exists(eq_path):
                                from PIL import Image

                                arr = (
                                    np.asarray(
                                        Image.open(eq_path).convert("RGB")
                                    ).astype("float32")
                                    / 255.0
                                )
                                eq_image_t = (
                                    torch.from_numpy(arr)
                                    .permute(2, 0, 1)
                                    .unsqueeze(0)
                                    .to(self.device)
                                )
                                eq_image_cache[frame_key] = eq_image_t
                if warp_overlap is not None and eq_image_t is not None:
                    warp_t, overlap_t = warp_overlap
                    h_p, w_p = rgb_pred.shape[1], rgb_pred.shape[2]
                    if warp_t.shape[0] != h_p or warp_t.shape[1] != w_p:
                        # Resize warp + overlap to render resolution if needed.
                        warp_b = warp_t.permute(2, 0, 1).unsqueeze(0)
                        warp_b = torch.nn.functional.interpolate(
                            warp_b,
                            size=(h_p, w_p),
                            mode="bilinear",
                            align_corners=False,
                        )
                        warp_t = warp_b[0].permute(1, 2, 0).contiguous()
                        overlap_b = overlap_t.unsqueeze(0).unsqueeze(0)
                        overlap_b = torch.nn.functional.interpolate(
                            overlap_b,
                            size=(h_p, w_p),
                            mode="bilinear",
                            align_corners=False,
                        )
                        overlap_t = overlap_b[0, 0].contiguous()
                    # Sample equirect GT into fisheye coords.
                    sampled = torch.nn.functional.grid_sample(
                        eq_image_t,
                        warp_t.unsqueeze(0),
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    )
                    sampled = sampled[0].permute(1, 2, 0).unsqueeze(0)
                    threshold = float(
                        self.conf.loss.equirect_consistency_overlap_threshold
                    )
                    loss_equirect_consistency = equirect_consistency_l1_loss(
                        rgb_pred,
                        sampled,
                        overlap_t.unsqueeze(0),
                        threshold,
                    )
                lambda_equirect_consistency = (
                    self.conf.loss.lambda_equirect_consistency
                )

        # Total loss
        camera_loss_weight = torch.ones(1, device=self.device)
        if self.conf.loss.use_camera_loss_weights:
            camera_idx = int(gpu_batch.camera_idx)
            configured_weights = list(self.conf.loss.camera_loss_weights)
            if camera_idx >= len(configured_weights):
                msg = (
                    "loss.camera_loss_weights must include one weight per "
                    f"camera. Missing index {camera_idx}."
                )
                raise RuntimeError(msg)
            camera_loss_weight = torch.tensor(
                float(configured_weights[camera_idx]),
                device=self.device,
                dtype=rgb_pred.dtype,
            )
        loss = (
            lambda_l1 * loss_l1
            + lambda_ssim * loss_ssim
            + lambda_foundation_feature * loss_foundation_feature
            + lambda_opacity * loss_opacity
            + lambda_scale * loss_scale
            + lambda_sky_opacity * loss_sky_opacity
            + lambda_dense_depth * loss_dense_depth
            + lambda_dense_depth_gradient * loss_dense_depth_gradient
            + lambda_equirect_consistency * loss_equirect_consistency
        )
        loss = loss * camera_loss_weight
        return dict(
            total_loss=loss,
            l1_loss=lambda_l1 * loss_l1,
            l2_loss=lambda_l2 * loss_l2,
            ssim_loss=lambda_ssim * loss_ssim,
            camera_loss_weight=camera_loss_weight,
            dense_depth_loss=lambda_dense_depth * loss_dense_depth,
            dense_depth_loss_raw=loss_dense_depth,
            dense_depth_gradient_loss=(
                lambda_dense_depth_gradient * loss_dense_depth_gradient
            ),
            dense_depth_gradient_loss_raw=loss_dense_depth_gradient,
            equirect_consistency_loss=(
                lambda_equirect_consistency * loss_equirect_consistency
            ),
            equirect_consistency_loss_raw=loss_equirect_consistency,
            foundation_feature_loss=(
                lambda_foundation_feature * loss_foundation_feature
            ),
            foundation_feature_loss_raw=loss_foundation_feature,
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
        iteration: int | None = None,
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
            iteration=f"{iteration!s}",
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
                "foundation_feature_error",
                "depth_gt",
                "predicted_depth",
                "opacity",
                "ray_hits",
                "hits_sparse",
                "opacity_low",
            ]
            tile_groups = metrics["img_eval_tiles"]
            grid_columns = [
                column for column in grid_columns if column in tile_groups
            ]
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
                caption = f"rows=image path, columns={', '.join(grid_columns)}"
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
                writer.add_scalar(
                    f"time/val/{time_key}", mean_timings[time_key], global_step
                )

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
        feature_metric_names = (
            (
                "foundation_feature_cosine",
                "metrics/feature/foundation_cosine",
            ),
            (
                "foundation_feature_distance",
                "metrics/feature/foundation_distance",
            ),
            (
                "foundation_feature_patch_cosine",
                "metrics/feature/foundation_patch_cosine",
            ),
            (
                "foundation_feature_patch_error",
                "metrics/feature/foundation_patch_error",
            ),
        )
        for metric_name, writer_name in feature_metric_names:
            if metric_name in metrics:
                writer.add_scalar(
                    writer_name,
                    np.mean(metrics[metric_name]),
                    global_step,
                )
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
        frequency_error_names = (
            ("fft_error_ratio_low", "diagnostics/frequency_error/ratio_low"),
            ("fft_error_ratio_mid", "diagnostics/frequency_error/ratio_mid"),
            ("fft_error_ratio_high", "diagnostics/frequency_error/ratio_high"),
            (
                "fft_error_ratio_ultra",
                "diagnostics/frequency_error/ratio_ultra",
            ),
        )
        for metric_name, wandb_name in frequency_error_names:
            writer.add_scalar(
                wandb_name,
                np.mean(metrics[metric_name]),
                global_step,
            )
        edge_metric_names = (
            ("edge_top15_precision", "diagnostics/edge/precision_top15"),
            ("edge_top15_recall", "diagnostics/edge/recall_top15"),
            ("edge_top15_f1", "diagnostics/edge/f1_top15"),
            ("edge_rgb_l1", "diagnostics/edge/rgb_l1"),
            ("nonedge_rgb_l1", "diagnostics/edge/nonedge_rgb_l1"),
            ("edge_high_freq_l1", "diagnostics/edge/high_freq_l1"),
            (
                "nonedge_high_freq_l1",
                "diagnostics/edge/nonedge_high_freq_l1",
            ),
        )
        for metric_name, wandb_name in edge_metric_names:
            writer.add_scalar(
                wandb_name,
                np.mean(metrics[metric_name]),
                global_step,
            )
        radial_bands = ("center", "mid", "outer", "rim")
        radial_metric_names = (
            "rgb_l1",
            "high_freq_l1",
            "gradient_l1",
            "edge_energy_ratio",
        )
        for band_name in radial_bands:
            for metric_name in radial_metric_names:
                source_name = f"radial_{band_name}_{metric_name}"
                if source_name in metrics:
                    writer.add_scalar(
                        f"diagnostics/radial/{band_name}/{metric_name}",
                        np.mean(metrics[source_name]),
                        global_step,
                    )
        if "camera_idx" in metrics:
            camera_indices = np.asarray(metrics["camera_idx"], dtype=np.int64)
            camera_metric_names = (
                ("masked_psnr", "masked_psnr"),
                ("psnr", "psnr"),
                ("gradient_l1", "gradient_l1"),
                ("fft_energy_ratio_high", "frequency_ratio_high"),
                ("fft_error_ratio_high", "frequency_error_ratio_high"),
                ("edge_top15_f1", "edge_f1_top15"),
                ("radial_center_rgb_l1", "center_rgb_l1"),
                ("radial_rim_rgb_l1", "rim_rgb_l1"),
                (
                    "radial_center_edge_energy_ratio",
                    "center_edge_energy_ratio",
                ),
                ("radial_rim_edge_energy_ratio", "rim_edge_energy_ratio"),
            )
            for camera_idx in sorted(set(camera_indices.tolist())):
                camera_mask = camera_indices == camera_idx
                for source_name, metric_name in camera_metric_names:
                    if source_name in metrics:
                        values = np.asarray(
                            metrics[source_name], dtype=np.float32
                        )
                        writer.add_scalar(
                            f"diagnostics/camera/{camera_idx}/{metric_name}",
                            values[camera_mask].mean(),
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
                writer.add_scalar(
                    wandb_name, geometry_metrics[metric_name], global_step
                )
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
        render_stat_metric_names = (
            ("pred_opacity_finite_fraction", "diagnostics/render/opacity/finite_fraction"),
            ("pred_opacity_nonzero_fraction", "diagnostics/render/opacity/nonzero_fraction"),
            ("pred_opacity_min", "diagnostics/render/opacity/min"),
            ("pred_opacity_mean", "diagnostics/render/opacity/mean"),
            ("pred_opacity_p50", "diagnostics/render/opacity/p50"),
            ("pred_opacity_p95", "diagnostics/render/opacity/p95"),
            ("pred_opacity_p99", "diagnostics/render/opacity/p99"),
            ("pred_opacity_max", "diagnostics/render/opacity/max"),
            ("pred_dist_finite_fraction", "diagnostics/render/depth/finite_fraction"),
            ("pred_dist_nonzero_fraction", "diagnostics/render/depth/nonzero_fraction"),
            ("pred_dist_min", "diagnostics/render/depth/min"),
            ("pred_dist_mean", "diagnostics/render/depth/mean"),
            ("pred_dist_p50", "diagnostics/render/depth/p50"),
            ("pred_dist_p95", "diagnostics/render/depth/p95"),
            ("pred_dist_p99", "diagnostics/render/depth/p99"),
            ("pred_dist_max", "diagnostics/render/depth/max"),
        )
        for metric_name, writer_name in render_stat_metric_names:
            if metric_name in metrics:
                writer.add_scalar(
                    writer_name,
                    np.mean(metrics[metric_name]),
                    global_step,
                )
        footprint_metric_names = (
            (
                "footprint_front_fraction",
                "diagnostics/footprint/front_fraction",
            ),
            (
                "footprint_radius_px_mean",
                "diagnostics/footprint/radius_px_mean",
            ),
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
                writer.add_scalar(
                    wandb_name, np.mean(metrics[metric_name]), global_step
                )

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
        if self.conf.loss.use_dense_depth:
            dense_depth_loss = np.mean(metrics["losses"]["dense_depth_loss"])
            dense_depth_loss_raw = np.mean(
                metrics["losses"]["dense_depth_loss_raw"]
            )
            writer.add_scalar(
                "val/loss/dense_depth", dense_depth_loss, global_step
            )
            writer.add_scalar(
                "val/loss/dense_depth_raw",
                dense_depth_loss_raw,
                global_step,
            )
            writer.add_scalar(
                "val/lambda/dense_depth",
                self.conf.loss.lambda_dense_depth,
                global_step,
            )
        if self.conf.loss.use_dense_depth_gradient:
            dense_depth_gradient_loss = np.mean(
                metrics["losses"]["dense_depth_gradient_loss"]
            )
            dense_depth_gradient_loss_raw = np.mean(
                metrics["losses"]["dense_depth_gradient_loss_raw"]
            )
            writer.add_scalar(
                "val/loss/dense_depth_gradient",
                dense_depth_gradient_loss,
                global_step,
            )
            writer.add_scalar(
                "val/loss/dense_depth_gradient_raw",
                dense_depth_gradient_loss_raw,
                global_step,
            )
            writer.add_scalar(
                "val/lambda/dense_depth_gradient",
                self.conf.loss.lambda_dense_depth_gradient,
                global_step,
            )
        if self.conf.loss.use_foundation_feature:
            foundation_feature_loss = np.mean(
                metrics["losses"]["foundation_feature_loss"]
            )
            foundation_feature_loss_raw = np.mean(
                metrics["losses"]["foundation_feature_loss_raw"]
            )
            writer.add_scalar(
                "val/loss/foundation_feature",
                foundation_feature_loss,
                global_step,
            )
            writer.add_scalar(
                "val/loss/foundation_feature_raw",
                foundation_feature_loss_raw,
                global_step,
            )
        if self.conf.loss.use_sky_opacity:
            sky_opacity_loss = np.mean(metrics["losses"]["sky_opacity_loss"])
            sky_opacity_loss_raw = np.mean(
                metrics["losses"]["sky_opacity_loss_raw"]
            )
            writer.add_scalar(
                "val/loss/sky_opacity", sky_opacity_loss, global_step
            )
            writer.add_scalar(
                "val/loss/sky_opacity_raw", sky_opacity_loss_raw, global_step
            )

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
                "fft_error_ratio_low",
                "fft_error_ratio_mid",
                "fft_error_ratio_high",
                "fft_error_ratio_ultra",
                "edge_top15_precision",
                "edge_top15_recall",
                "edge_top15_f1",
                "edge_rgb_l1",
                "nonedge_rgb_l1",
                "edge_high_freq_l1",
                "nonedge_high_freq_l1",
                "radial_center_gradient_l1",
                "radial_mid_gradient_l1",
                "radial_outer_gradient_l1",
                "radial_rim_gradient_l1",
                "radial_center_edge_energy_ratio",
                "radial_mid_edge_energy_ratio",
                "radial_outer_edge_energy_ratio",
                "radial_rim_edge_energy_ratio",
                "foundation_feature_cosine",
                "foundation_feature_distance",
                "foundation_feature_patch_cosine",
                "foundation_feature_patch_error",
            )
        }
        for time_key in mean_timings:
            table[time_key] = f"{f'{mean_timings[time_key]:.2f}'}" + " ms/it"
        summary_path = os.path.join(
            self.tracking.output_dir,
            f"validation_metrics_step_{global_step:06d}.json",
        )
        validation_summary: dict[str, float | str] = {
            "global_step": float(global_step),
            "num_gaussians": float(self.model.num_gaussians),
        }
        for key, value in table.items():
            if isinstance(value, str):
                validation_summary[key] = value
            else:
                validation_summary[key] = float(value)
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(validation_summary, file, indent=2, sort_keys=True)
        logger.log_table(
            f"📊 Validation Metrics - Step {global_step}", record=table
        )

    @torch.cuda.nvtx.range("log_training_iter")
    def log_training_iter(
        self,
        gpu_batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        batch_metrics: dict[str, Any],
        iteration: int | None = None,
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

        if (
            self.conf.enable_writer
            and global_step > 0
            and global_step % self.conf.log_frequency == 0
        ):
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
            if self.conf.loss.use_dense_depth:
                dense_depth_loss = np.mean(
                    batch_metrics["losses"]["dense_depth_loss"]
                )
                dense_depth_loss_raw = np.mean(
                    batch_metrics["losses"]["dense_depth_loss_raw"]
                )
                writer.add_scalar(
                    "train/loss/dense_depth",
                    dense_depth_loss,
                    global_step,
                )
                writer.add_scalar(
                    "train/loss/dense_depth_raw",
                    dense_depth_loss_raw,
                    global_step,
                )
                writer.add_scalar(
                    "train/lambda/dense_depth",
                    self.conf.loss.lambda_dense_depth,
                    global_step,
                )
            if self.conf.loss.use_dense_depth_gradient:
                dense_depth_gradient_loss = np.mean(
                    batch_metrics["losses"]["dense_depth_gradient_loss"]
                )
                dense_depth_gradient_loss_raw = np.mean(
                    batch_metrics["losses"]["dense_depth_gradient_loss_raw"]
                )
                writer.add_scalar(
                    "train/loss/dense_depth_gradient",
                    dense_depth_gradient_loss,
                    global_step,
                )
                writer.add_scalar(
                    "train/loss/dense_depth_gradient_raw",
                    dense_depth_gradient_loss_raw,
                    global_step,
                )
                writer.add_scalar(
                    "train/lambda/dense_depth_gradient",
                    self.conf.loss.lambda_dense_depth_gradient,
                    global_step,
                )
            if self.conf.loss.use_camera_loss_weights:
                camera_loss_weight = np.mean(
                    batch_metrics["losses"]["camera_loss_weight"]
                )
                writer.add_scalar(
                    "train/loss/camera_weight",
                    camera_loss_weight,
                    global_step,
                )
            if self.conf.loss.use_foundation_feature:
                foundation_feature_loss = np.mean(
                    batch_metrics["losses"]["foundation_feature_loss"]
                )
                foundation_feature_loss_raw = np.mean(
                    batch_metrics["losses"]["foundation_feature_loss_raw"]
                )
                writer.add_scalar(
                    "train/loss/foundation_feature",
                    foundation_feature_loss,
                    global_step,
                )
                writer.add_scalar(
                    "train/loss/foundation_feature_raw",
                    foundation_feature_loss_raw,
                    global_step,
                )
            if self.conf.loss.use_opacity:
                opacity_loss = np.mean(batch_metrics["losses"]["opacity_loss"])
                writer.add_scalar(
                    "train/loss/opacity", opacity_loss, global_step
                )
            if self.conf.loss.use_scale:
                scale_loss = np.mean(batch_metrics["losses"]["scale_loss"])
                writer.add_scalar("train/loss/scale", scale_loss, global_step)
            if self.conf.loss.use_sky_opacity:
                sky_opacity_loss = np.mean(
                    batch_metrics["losses"]["sky_opacity_loss"]
                )
                sky_opacity_loss_raw = np.mean(
                    batch_metrics["losses"]["sky_opacity_loss_raw"]
                )
                writer.add_scalar(
                    "train/loss/sky_opacity", sky_opacity_loss, global_step
                )
                writer.add_scalar(
                    "train/loss/sky_opacity_raw",
                    sky_opacity_loss_raw,
                    global_step,
                )
            if (
                self.post_processing is not None
                and "post_processing_reg_loss" in batch_metrics["losses"]
            ):
                post_processing_reg_loss = np.mean(
                    batch_metrics["losses"]["post_processing_reg_loss"]
                )
                writer.add_scalar(
                    "train/loss/post_processing_reg",
                    post_processing_reg_loss,
                    global_step,
                )
            if (
                self.camera_residual is not None
                and "camera_residual_reg_loss" in batch_metrics["losses"]
            ):
                camera_residual_reg_loss = np.mean(
                    batch_metrics["losses"]["camera_residual_reg_loss"]
                )
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
                writer.add_scalar(
                    "train/psnr", batch_metrics["psnr"], self.global_step
                )
            if "ssim" in batch_metrics:
                writer.add_scalar(
                    "train/ssim", batch_metrics["ssim"], self.global_step
                )
            if "lpips" in batch_metrics:
                writer.add_scalar(
                    "train/lpips", batch_metrics["lpips"], self.global_step
                )
            if "hits_mean" in batch_metrics:
                writer.add_scalar(
                    "train/hits/mean",
                    batch_metrics["hits_mean"],
                    self.global_step,
                )
            if "hits_std" in batch_metrics:
                writer.add_scalar(
                    "train/hits/std",
                    batch_metrics["hits_std"],
                    self.global_step,
                )
            if "hits_min" in batch_metrics:
                writer.add_scalar(
                    "train/hits/min",
                    batch_metrics["hits_min"],
                    self.global_step,
                )
            if "hits_max" in batch_metrics:
                writer.add_scalar(
                    "train/hits/max",
                    batch_metrics["hits_max"],
                    self.global_step,
                )
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
            render_stat_metric_names = (
                (
                    "pred_opacity_finite_fraction",
                    "train/render/opacity/finite_fraction",
                ),
                (
                    "pred_opacity_nonzero_fraction",
                    "train/render/opacity/nonzero_fraction",
                ),
                ("pred_opacity_min", "train/render/opacity/min"),
                ("pred_opacity_mean", "train/render/opacity/mean"),
                ("pred_opacity_p50", "train/render/opacity/p50"),
                ("pred_opacity_p95", "train/render/opacity/p95"),
                ("pred_opacity_p99", "train/render/opacity/p99"),
                ("pred_opacity_max", "train/render/opacity/max"),
                ("pred_dist_finite_fraction", "train/render/depth/finite_fraction"),
                ("pred_dist_nonzero_fraction", "train/render/depth/nonzero_fraction"),
                ("pred_dist_min", "train/render/depth/min"),
                ("pred_dist_mean", "train/render/depth/mean"),
                ("pred_dist_p50", "train/render/depth/p50"),
                ("pred_dist_p95", "train/render/depth/p95"),
                ("pred_dist_p99", "train/render/depth/p99"),
                ("pred_dist_max", "train/render/depth/max"),
            )
            for metric_name, writer_name in render_stat_metric_names:
                if metric_name in batch_metrics:
                    writer.add_scalar(
                        writer_name,
                        batch_metrics[metric_name],
                        self.global_step,
                    )

            if "timings" in batch_metrics:
                for time_key in batch_metrics["timings"]:
                    writer.add_scalar(
                        f"time/train/{time_key}",
                        batch_metrics["timings"][time_key],
                        self.global_step,
                    )

            writer.add_scalar(
                "train/iteration", self.global_step, self.global_step
            )
            if hasattr(self, "_training_start_time"):
                elapsed_seconds = (
                    time.perf_counter() - self._training_start_time
                )
                writer.add_scalar(
                    "time/train/elapsed_seconds",
                    elapsed_seconds,
                    self.global_step,
                )

            writer.add_scalar(
                "geometry/num_gaussians",
                self.model.num_gaussians,
                self.global_step,
            )

            # # NOTE: hack to easily compare with 3DGS
            # writer.add_scalar("train_loss_patches/total_loss", loss, global_step)
            # writer.add_scalar("gaussians/count", self.model.num_gaussians, self.global_step)

        logger.log_progress(
            task_name="Training",
            advance=1,
            step=f"{self.global_step!s}",
            loss=batch_metrics["losses"]["total_loss"],
        )

    @torch.cuda.nvtx.range("log_training_pass")
    def log_training_pass(self, metrics):
        """Log information after a single training pass.

        Args:
            metrics: dictionary of aggregated metrics for all batches in current pass.

        """

    @torch.cuda.nvtx.range("on_training_end")
    def on_training_end(self):
        """Callback that prompts at the end of training."""
        conf = self.conf
        out_dir = self.tracking.output_dir

        # Export the mixture-of-3d-gaussians
        logger.log_rule("Exporting Models")

        if conf.export_ply.enabled:
            from threedgrut.export import PLYExporter

            ply_path = (
                conf.export_ply.path
                if conf.export_ply.path
                else os.path.join(out_dir, "export_last.ply")
            )
            exporter = PLYExporter()
            exporter.export(
                self.model,
                Path(ply_path),
                dataset=self.train_dataset,
                conf=conf,
            )

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
                usdz_path = os.path.join(
                    out_dir, f"export_last_{format_suffix}.usdz"
                )

            exporter.export(
                self.model,
                Path(usdz_path),
                dataset=self.train_dataset,
                conf=conf,
                background=getattr(self, "background", None),
            )

        # Export post-processing report (PPISP-based)
        if (
            self.post_processing is not None
            and conf.post_processing.method == "ppisp"
        ):
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

    @torch.cuda.nvtx.range("save_checkpoint")
    def save_checkpoint(self, last_checkpoint: bool = False):
        """Saves checkpoint to a path under {conf.out_dir}/{conf.experiment_name}.

        Args:
            last_checkpoint: If true, will update checkpoint title to 'last'.
                             Otherwise uses global step

        """
        global_step = self.global_step
        out_dir = self.tracking.output_dir
        parameters = self.model.get_model_parameters()
        parameters |= {
            "global_step": self.global_step,
            "epoch": self.n_epochs - 1,
        }

        strategy_parameters = self.strategy.get_strategy_parameters()
        parameters = {**parameters, **strategy_parameters}

        # Add post-processing state to checkpoint (module + optimizers + schedulers)
        if self.post_processing is not None:
            parameters["post_processing"] = {
                "module": self.post_processing.state_dict(),
                "optimizers": [
                    opt.state_dict() for opt in self.post_processing_optimizers
                ],
                "schedulers": [
                    sched.state_dict()
                    for sched in self.post_processing_schedulers
                ],
            }
        if self.camera_residual is not None:
            parameters["camera_residual"] = {
                "module": self.camera_residual.state_dict(),
                "optimizer": self.camera_residual_optimizer.state_dict(),
            }

        os.makedirs(
            os.path.join(out_dir, f"ours_{int(global_step)}"), exist_ok=True
        )
        if not last_checkpoint:
            ckpt_path = os.path.join(
                out_dir, f"ours_{int(global_step)}", f"ckpt_{global_step}.pt"
            )
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

    def _log_gradient_diagnostics(self, global_step: int) -> None:
        """Log whether supervision reaches the Gaussian parameters."""
        if (
            not self.conf.enable_writer
            or global_step <= 0
            or global_step % self.conf.log_frequency != 0
        ):
            return

        writer = self.tracking.writer
        parameters: tuple[tuple[str, torch.nn.Parameter], ...] = (
            ("positions", self.model.positions),
            ("rotation", self.model.rotation),
            ("scale", self.model.scale),
            ("density", self.model.density),
            ("features_albedo", self.model.features_albedo),
            ("features_specular", self.model.features_specular),
        )
        for parameter_name, parameter in parameters:
            stats = _gradient_tensor_stats(
                parameter,
                f"train/grad/{parameter_name}",
            )
            for metric_name, value in stats.items():
                writer.add_scalar(metric_name, value, global_step)

    @torch.cuda.nvtx.range("run_train_iter")
    def run_train_iter(
        self,
        global_step: int,
        batch: dict,
        profilers: dict,
        metrics: list,
        conf: DictConfig,
    ):
        # Freeze Gaussians and suspend strategy when distillation starts
        if (
            self._distillation_start_step >= 0
            and global_step >= self._distillation_start_step
        ):
            self.model.freeze_gaussians()
            self.strategy.suspend()

        # Access the GPU-cache batch data
        with torch.cuda.nvtx.range(f"train_iter{global_step}_get_gpu_batch"):
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
            gpu_batch = self._apply_camera_residual(gpu_batch)

        # Perform validation if required
        is_time_to_validate = (global_step > 0 or conf.validate_first) and (
            global_step % self.val_frequency == 0
        )
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
                outputs = apply_post_processing(
                    self.post_processing, outputs, gpu_batch, training=True
                )

        # Compute the losses of a single batch
        with torch.cuda.nvtx.range(f"train_{global_step}_loss"):
            batch_losses = self.get_losses(gpu_batch, outputs)
            # Add post-processing regularization loss
            if self.post_processing is not None:
                post_processing_reg_loss = (
                    self.post_processing.get_regularization_loss()
                )
                batch_losses["total_loss"] = (
                    batch_losses["total_loss"] + post_processing_reg_loss
                )
                batch_losses["post_processing_reg_loss"] = (
                    post_processing_reg_loss
                )
            if self.camera_residual is not None:
                camera_residual_reg_loss = (
                    self.camera_residual.get_regularization_loss()
                )
                batch_losses["total_loss"] = (
                    batch_losses["total_loss"] + camera_residual_reg_loss
                )
                batch_losses["camera_residual_reg_loss"] = (
                    camera_residual_reg_loss
                )

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
        self._log_gradient_diagnostics(global_step)
        self._validate_camera_residual_gradient(global_step)

        # Post backward strategy step
        with torch.cuda.nvtx.range(f"train_{global_step}_post_bwd"):
            scene_updated = self.strategy.post_backward(
                step=global_step,
                scene_extent=self.scene_extent,
                train_dataset=self.train_dataset,
                batch=gpu_batch,
                writer=self.tracking.writer,
                outputs=outputs,
            )

        # Optimizer step
        with torch.cuda.nvtx.range(f"train_{global_step}_backprop"):
            if isinstance(self.model.optimizer, SelectiveAdam):
                assert (
                    outputs["mog_visibility"].shape == self.model.density.shape
                ), (
                    f"Visibility shape {outputs['mog_visibility'].shape} does not match density shape {self.model.density.shape}"
                )
                self.model.optimizer.step(outputs["mog_visibility"])
            else:
                self.model.optimizer.step()
            self.model.optimizer.zero_grad()

        # Scheduler step
        with torch.cuda.nvtx.range(f"train_{global_step}_scheduler"):
            self.model.scheduler_step(global_step)

        # Post-processing optimizer/scheduler step
        if self.post_processing_optimizers is not None:
            with torch.cuda.nvtx.range(
                f"train_{global_step}_post_processing_opt"
            ):
                for opt in self.post_processing_optimizers:
                    opt.step()
                    opt.zero_grad()
                for sched in self.post_processing_schedulers:
                    sched.step()
        if self.camera_residual_optimizer is not None:
            with torch.cuda.nvtx.range(
                f"train_{global_step}_camera_residual_opt"
            ):
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
            conf.model.bvh_update_frequency > 0
            and global_step % conf.model.bvh_update_frequency == 0
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
            batch_metrics["timings"]["forward_render_cuda"] = (
                self.model.renderer.timings["forward_render"]
            )
        if "backward_render" in self.model.renderer.timings:
            batch_metrics["timings"]["backward_render_cuda"] = (
                self.model.renderer.timings["backward_render"]
            )
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

    @torch.cuda.nvtx.range("run_train_pass")
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
            self.run_train_iter(
                self.global_step, batch, profilers, metrics, conf
            )

        self.log_training_pass(metrics)

    @torch.cuda.nvtx.range("run_validation_pass")
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
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(
                batch_idx
            )
            gpu_batch = self._apply_camera_residual(gpu_batch)

            # Compute the outputs of a single batch
            with torch.cuda.nvtx.range(
                f"train.validation_step_{self.global_step}"
            ):
                profilers["inference"].start()
                outputs = self.model(gpu_batch, train=False)
                # Apply post-processing for validation (novel view mode)
                if self.post_processing is not None:
                    outputs = apply_post_processing(
                        self.post_processing,
                        outputs,
                        gpu_batch,
                        training=False,
                    )
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

                self.log_validation_iter(
                    gpu_batch, outputs, batch_metrics, iteration=val_iteration
                )
                metrics.append(batch_metrics)

        logger.end_progress(task_name="Validation")

        metrics = self._flatten_list_of_dicts(metrics)
        self.log_validation_pass(metrics)
        return metrics

    @staticmethod
    def _flatten_list_of_dicts(list_of_dicts):
        """Converts list of dicts -> dict of lists.
        Supports flattening of up to 2 levels of dict hierarchies
        """
        flat_dict = defaultdict(list)
        for d in list_of_dicts:
            for k, v in d.items():
                if isinstance(v, dict):
                    flat_dict[k] = (
                        defaultdict(list)
                        if k not in flat_dict
                        else flat_dict[k]
                    )
                    for inner_k, inner_v in v.items():
                        flat_dict[k][inner_k].append(inner_v)
                else:
                    flat_dict[k].append(v)
        return flat_dict

    def run_training(self):
        """Initiate training logic for n_epochs.
        Training and validation are controlled by the config.
        """
        assert self.model.optimizer is not None, (
            "Optimizer needs to be initialized before the training can start!"
        )
        conf = self.conf

        logger.log_rule(f"Training {conf.render.method.upper()}")
        if self.camera_residual is not None and bool(
            conf.camera_residual.finite_difference_audit.enabled
        ):
            self.run_camera_residual_finite_difference_audit()
            if bool(conf.camera_residual.finite_difference_audit.exit_after):
                logger.info(
                    "Camera residual finite-difference audit complete."
                )
                return

        if bool(conf.camera_intrinsics_audit.enabled):
            self.run_camera_intrinsics_finite_difference_audit()
            if bool(conf.camera_intrinsics_audit.exit_after):
                logger.info(
                    "Camera intrinsics finite-difference audit complete."
                )
                return

        if bool(conf.get("validate_only", False)):
            logger.info("Running validation only; training loop is disabled.")
            self.run_validation_pass(conf)
            return

        # Training loop
        self._training_start_time = time.perf_counter()
        logger.start_progress(
            task_name="Training",
            total_steps=conf.n_iterations,
            color="spring_green1",
        )

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
        logger.log_table("🎊 Training Statistics", record=table)

        if bool(conf.get("validate_final", True)):
            logger.log_rule("Final Validation")
            self.run_validation_pass(conf)

        # Perform testing
        self.on_training_end()
        logger.info("🥳 Training Complete.")

        # Updating the GUI
        if self.gui is not None:
            self.gui.training_done = True
            logger.info("🎨 GUI Blocking... Terminate GUI to Stop.")
            self.gui.block_in_rendering_loop(fps=60)
