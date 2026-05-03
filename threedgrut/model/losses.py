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

@torch.cuda.nvtx.range("dense_depth_l1_loss")
def dense_depth_l1_loss(pred_dist, gt_depth, valid_mask):
    """L1 loss on per-pixel dense depth, masked to pixels with valid GT.

    Args:
        pred_dist: [B, H, W, 1] or [B, H, W] predicted distance (metres) from
            the renderer (`outputs["pred_dist"]`).
        gt_depth: [B, H, W] ground-truth depth (metres) loaded from the per-image
            DA3 z-extended sidecar.
        valid_mask: [B, H, W] boolean mask where `gt_depth > 0` (and any other
            constraints from the dataset, e.g. training mask).
    Returns:
        Scalar masked-mean L1 loss; returns a zero-tensor on the predicted
        device if no valid pixels exist (dataset has no DA3 cache for this
        frame).
    """
    if pred_dist.dim() == 4 and pred_dist.shape[-1] == 1:
        pred_dist = pred_dist[..., 0]
    if not valid_mask.any():
        return torch.zeros(1, device=pred_dist.device, dtype=pred_dist.dtype)
    diff = (pred_dist - gt_depth).abs()
    return (diff * valid_mask.to(diff.dtype)).sum() / valid_mask.sum().clamp_min(1.0)
