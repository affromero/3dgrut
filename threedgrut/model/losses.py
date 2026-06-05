# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
from fused_ssim import fused_ssim


@torch.cuda.nvtx.range("l1_loss")
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


@torch.cuda.nvtx.range("l2_loss")
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="valid")


@torch.cuda.nvtx.range("equirect_consistency_l1_loss")
def equirect_consistency_l1_loss(
    pred_rgb,
    equirect_at_fisheye,
    overlap,
    threshold,
    pixel_weight=None,
):
    """L1 loss between fisheye render and equirect-sampled colors at overlap.

    Args:
        pred_rgb: [B, H, W, 3] rendered fisheye RGB in [0, 1].
        equirect_at_fisheye: [B, H, W, 3] equirect GT sampled into fisheye
            coordinates via `F.grid_sample(equirect_gt_NCHW, warp_AB)`. Must be
            in [0, 1] and aligned to `pred_rgb`.
        overlap: [B, H, W] RoMa fisheye-equirect overlap confidence in [0, 1].
        threshold: Pixels with `overlap >= threshold` contribute to the loss;
            others are masked out so non-overlapping rim pixels don't dominate.
        pixel_weight: Optional [B, H, W] non-negative per-pixel weights applied
            inside the valid overlap mask.
    Returns:
        Scalar masked-mean L1; zero tensor on the predicted device when no
        pixels exceed threshold.
    """
    valid = overlap >= threshold
    if not valid.any():
        return torch.zeros(1, device=pred_rgb.device, dtype=pred_rgb.dtype)
    diff = (pred_rgb - equirect_at_fisheye).abs()
    weight = valid.unsqueeze(-1).to(diff.dtype)
    if pixel_weight is not None:
        if pixel_weight.dim() == 3:
            pixel_weight = pixel_weight.unsqueeze(-1)
        weight = weight * pixel_weight.to(diff.dtype).clamp_min(0.0)
    denominator = weight.sum().clamp_min(1.0).mul(diff.shape[-1])
    return (diff * weight).sum() / denominator
