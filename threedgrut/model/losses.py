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
import torch.nn.functional as F
from beartype import beartype
from fused_ssim import fused_ssim
from jaxtyping import Float, jaxtyped


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


@jaxtyped(typechecker=beartype)
def _resize_like(
    image: Float[torch.Tensor, "batch image_height image_width channel"],
    target: Float[
        torch.Tensor, "target_batch target_height target_width target_channel"
    ],
) -> Float[torch.Tensor, "batch target_height target_width channel"]:
    if image.shape[1:3] == target.shape[1:3]:
        return image
    return F.interpolate(
        image.permute(0, 3, 1, 2),
        size=target.shape[1:3],
        mode="nearest",
    ).permute(0, 2, 3, 1)


@jaxtyped(typechecker=beartype)
def _gaussian_kernel_1d(
    *,
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Float[torch.Tensor, "kernel"]:
    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError(
            "rim_hf_kernel_size must be a positive odd integer; "
            f"got {kernel_size}."
        )
    if sigma <= 0.0:
        raise ValueError(f"rim_hf_sigma must be positive; got {sigma}.")
    half = kernel_size // 2
    coords = torch.arange(
        -half,
        half + 1,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return kernel / torch.clamp(kernel.sum(), min=torch.finfo(dtype).eps)


@jaxtyped(typechecker=beartype)
def _high_pass_rgb(
    image: Float[torch.Tensor, "batch height width channel"],
    *,
    kernel_size: int,
    sigma: float,
) -> Float[torch.Tensor, "batch height width channel"]:
    channels = image.shape[-1]
    kernel = _gaussian_kernel_1d(
        kernel_size=kernel_size,
        sigma=sigma,
        device=image.device,
        dtype=image.dtype,
    )
    horizontal = kernel.view(1, 1, 1, kernel_size).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, kernel_size, 1).expand(channels, 1, -1, 1)
    nchw = image.permute(0, 3, 1, 2)
    pad = kernel_size // 2
    low = F.conv2d(
        F.pad(nchw, (pad, pad, 0, 0), mode="replicate"),
        horizontal,
        groups=channels,
    )
    low = F.conv2d(
        F.pad(low, (0, 0, pad, pad), mode="replicate"),
        vertical,
        groups=channels,
    )
    return (nchw - low).permute(0, 2, 3, 1)


@jaxtyped(typechecker=beartype)
def rim_high_frequency_loss(
    *,
    rgb_pred: Float[torch.Tensor, "batch height width channel"],
    rgb_gt: Float[torch.Tensor, "batch height width channel"],
    depth_ray_z: Float[torch.Tensor, "depth_batch depth_height depth_width 1"],
    mask: Float[torch.Tensor, "mask_batch mask_height mask_width 1"] | None,
    theta_min_deg: float,
    theta_max_deg: float,
    kernel_size: int,
    sigma: float,
    loss_type: str,
    charbonnier_epsilon: float,
) -> Float[torch.Tensor, ""]:
    cos_t = _resize_like(
        depth_ray_z.to(device=rgb_pred.device, dtype=rgb_pred.dtype),
        rgb_pred,
    )
    if cos_t.shape[0] != rgb_pred.shape[0]:
        cos_t = cos_t.expand(rgb_pred.shape[0], -1, -1, -1)

    if theta_min_deg >= theta_max_deg:
        raise ValueError(
            "rim_hf_theta_min_deg must be smaller than "
            "rim_hf_theta_max_deg; "
            f"got {theta_min_deg} >= {theta_max_deg}."
        )
    theta_min = torch.tensor(
        theta_min_deg,
        device=rgb_pred.device,
        dtype=rgb_pred.dtype,
    )
    theta_max = torch.tensor(
        theta_max_deg,
        device=rgb_pred.device,
        dtype=rgb_pred.dtype,
    )
    cos_min = torch.cos(torch.deg2rad(theta_min))
    cos_max = torch.cos(torch.deg2rad(theta_max))
    rim_mask = (cos_t <= cos_min) & (cos_t >= cos_max)
    rim_mask = rim_mask & torch.isfinite(rgb_gt).all(dim=-1, keepdim=True)
    rim_mask = rim_mask & torch.isfinite(rgb_pred).all(dim=-1, keepdim=True)
    if mask is not None:
        mask = _resize_like(mask.to(device=rgb_pred.device), rgb_pred)
        if mask.shape[0] != rgb_pred.shape[0]:
            mask = mask.expand(rgb_pred.shape[0], -1, -1, -1)
        rim_mask = rim_mask & (mask > 0.5)

    weights = rim_mask.to(dtype=rgb_pred.dtype)
    denom = torch.clamp(
        weights.sum() * rgb_pred.shape[-1],
        min=torch.tensor(1.0, device=rgb_pred.device, dtype=rgb_pred.dtype),
    )

    pred_hp = _high_pass_rgb(
        rgb_pred,
        kernel_size=kernel_size,
        sigma=sigma,
    )
    gt_hp = _high_pass_rgb(rgb_gt, kernel_size=kernel_size, sigma=sigma)

    if loss_type == "l1":
        return (torch.abs(pred_hp - gt_hp) * weights).sum() / denom
    if loss_type == "charbonnier":
        if charbonnier_epsilon <= 0.0:
            raise ValueError(
                "rim_hf_charbonnier_epsilon must be positive; "
                f"got {charbonnier_epsilon}."
            )
        epsilon = torch.tensor(
            charbonnier_epsilon,
            device=rgb_pred.device,
            dtype=rgb_pred.dtype,
        )
        residual = pred_hp - gt_hp
        return (
            (torch.sqrt(residual * residual + epsilon * epsilon) - epsilon)
            * weights
        ).sum() / denom
    if loss_type == "ncc":
        pred_weighted = pred_hp * weights
        gt_weighted = gt_hp * weights
        pred_mean = pred_weighted.sum() / denom
        gt_mean = gt_weighted.sum() / denom
        pred_centered = (pred_hp - pred_mean) * weights
        gt_centered = (gt_hp - gt_mean) * weights
        numerator = (pred_centered * gt_centered).sum()
        pred_energy = torch.sqrt(
            torch.clamp((pred_centered * pred_centered).sum(), min=1e-12)
        )
        gt_energy = torch.sqrt(
            torch.clamp((gt_centered * gt_centered).sum(), min=1e-12)
        )
        return 1.0 - numerator / (pred_energy * gt_energy)
    raise ValueError(
        "Unsupported loss.rim_hf_loss_type "
        f"{loss_type!r}. Supported values: l1, charbonnier, ncc."
    )
