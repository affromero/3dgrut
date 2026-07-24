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

import math

import torch
import torch.nn.functional as F
from beartype import beartype
from fused_ssim import FusedSSIMMap, fused_ssim
from jaxtyping import Float, jaxtyped

FIXED_IMAGE_LOSS_MIN_VALID_FRACTION = 0.8


@torch.cuda.nvtx.range("l1_loss")
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


@torch.cuda.nvtx.range("l2_loss")
def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def native_huber_loss(
    residual: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    """Return a per-pixel Huber loss without reducing the tensor."""
    if delta <= 0.0:
        raise ValueError(f"Huber delta must be positive, got {delta}.")
    absolute = torch.abs(residual)
    return torch.where(
        absolute <= delta,
        residual.square() / (2.0 * delta),
        absolute - 0.5 * delta,
    )


def fixed_image_loss_denominator(
    *,
    rgb: torch.Tensor,
    mask: torch.Tensor | None,
    image_scale: float | torch.Tensor = 1.0,
    min_valid_fraction: float = FIXED_IMAGE_LOSS_MIN_VALID_FRACTION,
) -> torch.Tensor:
    """Return a full-image RGB reduction denominator with a validity floor."""
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3:
        raise ValueError(
            "Fixed image loss requires one HWC image with three channels."
        )
    image_scale_tensor = torch.as_tensor(
        image_scale,
        dtype=rgb.dtype,
        device=rgb.device,
    )
    if image_scale_tensor.shape != ():
        raise ValueError("Fixed image loss scale must be scalar.")
    if (
        not torch.isfinite(image_scale_tensor).item()
        or image_scale_tensor.item() <= 0.0
    ):
        raise ValueError(
            "Fixed image loss image scale must be finite and positive."
        )
    if (
        not math.isfinite(min_valid_fraction)
        or not 0.0 < min_valid_fraction <= 1.0
    ):
        raise ValueError(
            "Fixed image loss minimum valid fraction must be in (0, 1]."
        )
    denominator = torch.tensor(
        rgb.numel(),
        dtype=rgb.dtype,
        device=rgb.device,
    ) * image_scale_tensor
    if mask is None:
        return denominator
    expected_shape = (*rgb.shape[:-1], 1)
    if mask.shape != expected_shape:
        raise ValueError(
            f"Fixed image loss mask must have shape {expected_shape}."
        )
    valid_fraction = mask.to(dtype=rgb.dtype).mean()
    return denominator * torch.clamp(
        valid_fraction,
        min=min_valid_fraction,
    )


def indexed_camera_loss_weight(
    *,
    camera_index: int,
    configured_weights: list[float],
) -> float:
    """Return a validated camera weight for an explicit camera index."""
    if camera_index < 0 or camera_index >= len(configured_weights):
        raise ValueError(
            "Camera loss weights must include the requested camera index "
            f"{camera_index}."
        )
    weight = float(configured_weights[camera_index])
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("Camera loss weights must be finite and positive.")
    return weight


def _layered_depth_neighbor_sign_sum(
    transparency: torch.Tensor,
) -> torch.Tensor:
    """Return the four-neighbour transparency sign term."""
    result = torch.zeros_like(transparency)
    horizontal = transparency[:, :, 1:, :] - transparency[:, :, :-1, :]
    horizontal_sign = torch.copysign(
        torch.ones_like(horizontal),
        horizontal,
    )
    result[:, :, 1:, :] += horizontal_sign
    reverse_horizontal = transparency[:, :, :-1, :] - transparency[:, :, 1:, :]
    result[:, :, :-1, :] += torch.copysign(
        torch.ones_like(reverse_horizontal),
        reverse_horizontal,
    )
    vertical = transparency[:, 1:, :, :] - transparency[:, :-1, :, :]
    vertical_sign = torch.copysign(torch.ones_like(vertical), vertical)
    result[:, 1:, :, :] += vertical_sign
    reverse_vertical = transparency[:, :-1, :, :] - transparency[:, 1:, :, :]
    result[:, :-1, :, :] += torch.copysign(
        torch.ones_like(reverse_vertical),
        reverse_vertical,
    )
    return result


def _validate_semantic_label(
    *,
    label: int | None,
    name: str,
) -> None:
    """Validate an optional uint8 semantic label selector."""
    if label is None:
        return
    if not isinstance(label, int) or not 0 <= label <= 255:
        raise ValueError(f"{name} must be a uint8 label or None, got {label!r}.")


@torch.no_grad()
def layered_depth_adjoint_gradients(
    *,
    raw_depth: torch.Tensor,
    layer_transparency: torch.Tensor,
    median_depth: torch.Tensor,
    semantic_mask: torch.Tensor,
    normalized_weight: float,
    semantic_label_remap_from: int | None = None,
    semantic_label_remap_to: int | None = None,
    primary_semantic_label: int | None = None,
    primary_transparency_bias: float = 0.0,
    secondary_semantic_label: int | None = None,
    secondary_transparency_bias: float = 0.0,
    transparency_target: float = 0.5,
    transparency_gradient_scale: float = 0.2,
    neighbor_gradient_scale: float = 0.1,
    depth_transparency_threshold: float = 0.1,
    depth_gradient_scale: float = 0.01,
    depth_residual_delta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a configurable injected adjoint for layered depth buffers."""
    if raw_depth.ndim != 4 or raw_depth.shape[-1] != 1:
        raise ValueError("Layered raw depth must be [B, H, W, 1].")
    if layer_transparency.shape != raw_depth.shape:
        raise ValueError("Layer transparency must match raw depth.")
    if median_depth.shape != raw_depth.shape:
        raise ValueError("Median depth must match raw depth.")
    if semantic_mask.shape != raw_depth.shape:
        raise ValueError("Semantic mask must match raw depth.")
    if semantic_mask.dtype != torch.uint8:
        raise ValueError("Semantic mask must use uint8 labels.")
    if not math.isfinite(normalized_weight) or normalized_weight <= 0.0:
        raise ValueError(
            "Layered depth normalized weight must be finite and positive."
        )
    if (semantic_label_remap_from is None) != (
        semantic_label_remap_to is None
    ):
        raise ValueError(
            "Semantic label remapping requires both source and target labels."
        )
    for name, label in (
        ("semantic_label_remap_from", semantic_label_remap_from),
        ("semantic_label_remap_to", semantic_label_remap_to),
        ("primary_semantic_label", primary_semantic_label),
        ("secondary_semantic_label", secondary_semantic_label),
    ):
        _validate_semantic_label(label=label, name=name)
    if (
        primary_semantic_label is not None
        and primary_semantic_label == secondary_semantic_label
    ):
        raise ValueError("Primary and secondary semantic labels must differ.")
    coefficients = (
        primary_transparency_bias,
        secondary_transparency_bias,
        transparency_target,
        transparency_gradient_scale,
        neighbor_gradient_scale,
        depth_transparency_threshold,
        depth_gradient_scale,
        depth_residual_delta,
    )
    if not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Layered depth adjoint coefficients must be finite.")
    if depth_transparency_threshold <= 0.0:
        raise ValueError("Depth transparency threshold must be positive.")
    if depth_residual_delta <= 0.0:
        raise ValueError("Depth residual delta must be positive.")

    labels = semantic_mask
    if semantic_label_remap_from is not None:
        labels = torch.where(
            semantic_mask == semantic_label_remap_from,
            torch.full_like(semantic_mask, semantic_label_remap_to),
            semantic_mask,
        )
    height = raw_depth.shape[1]
    width = raw_depth.shape[2]
    weight = raw_depth.new_tensor(normalized_weight / (height * width))
    valid = labels != 0
    class_term = torch.zeros_like(raw_depth)
    if primary_semantic_label is not None:
        class_term = torch.where(
            labels == primary_semantic_label,
            torch.full_like(raw_depth, primary_transparency_bias) * weight,
            class_term,
        )
    if secondary_semantic_label is not None:
        class_term = torch.where(
            labels == secondary_semantic_label,
            torch.full_like(raw_depth, secondary_transparency_bias) * weight,
            class_term,
        )
    transparency_gradient = (
        weight
        * transparency_gradient_scale
        * (transparency_target - layer_transparency)
        + class_term
        + weight
        * neighbor_gradient_scale
        * _layered_depth_neighbor_sign_sum(layer_transparency)
    )
    transparency_gradient = torch.where(
        valid,
        transparency_gradient,
        torch.zeros_like(transparency_gradient),
    )

    residual = raw_depth - median_depth * (1.0 - layer_transparency)
    huber_gradient = torch.clamp(
        residual / depth_residual_delta,
        min=-1.0,
        max=1.0,
    )
    raw_depth_gradient = (
        weight
        * depth_gradient_scale
        * (1.0 - layer_transparency / depth_transparency_threshold)
        * huber_gradient
    )
    depth_eligible = (
        valid
        & (median_depth > 0.0)
        & (layer_transparency < depth_transparency_threshold)
    )
    raw_depth_gradient = torch.where(
        depth_eligible,
        raw_depth_gradient,
        torch.zeros_like(raw_depth_gradient),
    )
    return raw_depth_gradient, transparency_gradient


def layered_depth_adjoint_injection(
    *,
    raw_depth: torch.Tensor,
    layer_transparency: torch.Tensor,
    median_depth: torch.Tensor,
    semantic_mask: torch.Tensor,
    normalized_weight: float,
    semantic_label_remap_from: int | None = None,
    semantic_label_remap_to: int | None = None,
    primary_semantic_label: int | None = None,
    primary_transparency_bias: float = 0.0,
    secondary_semantic_label: int | None = None,
    secondary_transparency_bias: float = 0.0,
    transparency_target: float = 0.5,
    transparency_gradient_scale: float = 0.2,
    neighbor_gradient_scale: float = 0.1,
    depth_transparency_threshold: float = 0.1,
    depth_gradient_scale: float = 0.01,
    depth_residual_delta: float = 0.1,
) -> torch.Tensor:
    """Inject layered-depth gradients without differentiating them twice."""
    raw_depth_gradient, transparency_gradient = (
        layered_depth_adjoint_gradients(
            raw_depth=raw_depth,
            layer_transparency=layer_transparency,
            median_depth=median_depth,
            semantic_mask=semantic_mask,
            normalized_weight=normalized_weight,
            semantic_label_remap_from=semantic_label_remap_from,
            semantic_label_remap_to=semantic_label_remap_to,
            primary_semantic_label=primary_semantic_label,
            primary_transparency_bias=primary_transparency_bias,
            secondary_semantic_label=secondary_semantic_label,
            secondary_transparency_bias=secondary_transparency_bias,
            transparency_target=transparency_target,
            transparency_gradient_scale=transparency_gradient_scale,
            neighbor_gradient_scale=neighbor_gradient_scale,
            depth_transparency_threshold=depth_transparency_threshold,
            depth_gradient_scale=depth_gradient_scale,
            depth_residual_delta=depth_residual_delta,
        )
    )
    return (
        raw_depth * raw_depth_gradient.detach()
        + layer_transparency * transparency_gradient.detach()
    ).sum()


@torch.cuda.nvtx.range("ssim")
def ssim(img1, img2, window_size=11, size_average=True):
    # predicted_image, gt_image: [BS, CH, H, W], predicted_image is differentiable
    return fused_ssim(img1, img2, padding="valid")


@torch.cuda.nvtx.range("masked_ssim_same")
def masked_ssim_same_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    denominator: torch.Tensor,
    require_full_valid_window: bool = False,
) -> torch.Tensor:
    """Return zero-padded SSIM reduced over selected valid map centers."""
    if prediction.ndim != 4 or prediction.shape != target.shape:
        raise ValueError(
            "Masked SSIM requires matching NCHW prediction and target tensors."
        )
    expected_mask_shape = (
        prediction.shape[0],
        1,
        prediction.shape[2],
        prediction.shape[3],
    )
    if mask.shape != expected_mask_shape:
        raise ValueError(
            "Masked SSIM mask must have shape [N, 1, H, W]."
        )
    if denominator.shape != ():
        raise ValueError("Masked SSIM denominator must be scalar.")
    if not torch.isfinite(denominator).item() or denominator.item() <= 0.0:
        raise ValueError(
            "Masked SSIM denominator must be finite and positive."
        )
    masked_prediction = prediction * mask
    masked_target = target * mask
    ssim_map = FusedSSIMMap.apply(
        0.01**2,
        0.03**2,
        masked_prediction,
        masked_target,
        "same",
        True,
    )
    center_mask = mask
    if require_full_valid_window:
        invalid_mask = F.pad(1.0 - mask, (5, 5, 5, 5), value=0.0)
        invalid_window = F.max_pool2d(
            invalid_mask,
            kernel_size=11,
            stride=1,
        )
        center_mask = mask * (invalid_window == 0.0).to(dtype=mask.dtype)
    return ((1.0 - ssim_map) * center_mask).sum() / denominator


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
