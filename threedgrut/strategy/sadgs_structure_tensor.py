# SPDX-FileCopyrightText: Copyright (c) 2026 Hax AI
# SPDX-License-Identifier: Apache-2.0
#
# This file ports the structure-tensor implementation from
#   SAD-GS: Structure-Aware Densification for 3D Gaussian Splatting
#   Lyu, Tewari, Chen, Leimkühler, Theobalt -- SIGGRAPH 2026
#   https://arxiv.org/abs/2604.28016
#   https://github.com/LinjieLyu/SADGS  (MIT License)
#
# Original SAD-GS code is MIT-licensed. The functions below are faithful
# ports of:
#   - SADGS/utils/loss_utils.py::fast_gaussian_blur
#   - SADGS/utils/loss_utils.py::get_structure_tensor_torch
#   - SADGS/utils/loss_utils.py::get_multiscale_structure_tensor_v2
# with renaming for clarity and docstrings expanded.
#
# Wavelength-from-structure-tensor extraction is added here; SAD-GS computes
# it inline inside update_freq_stats_online and we expose it as a standalone
# helper so the validation harness can compare against the in-house
# scale_px map.

"""Multi-channel multi-scale structure tensor for SAD-GS-style densification.

The structure tensor of a multi-channel image is the smoothed outer product
of intensity gradients summed across channels (Di Zenzo's method). Its
principal eigenvalue lambda_1 measures local high-frequency energy; its
inverse square-root gives a dominant local wavelength in pixels. SAD-GS
uses this wavelength as the "speed limit" for projected Gaussian extent.

Public functions:

* :func:`fast_gaussian_blur` -- torchvision wrapper with downsample fallback
  for large sigma. Verbatim from SAD-GS apart from rename.
* :func:`multi_channel_structure_tensor` -- single-scale Di Zenzo structure
  tensor returning :math:`(S_{xx}, S_{xy}, S_{yy})` as a ``(B, 3, H, W)``
  tensor.
* :func:`multiscale_structure_tensor_v2` -- per-level structure tensor
  aggregation across an octave pyramid weighted by DoG band response.
  Matches SAD-GS v2; v1 (orientation-shared) is not ported because v2 is the
  principled formulation per the SAD-GS source comments.
* :func:`wavelength_min_from_structure_tensor` -- extracts
  :math:`\\text{wavelength}_\\min = 1 / (\\sqrt{\\lambda_1} + \\epsilon)`
  in pixels from a structure tensor.

All functions are pure-PyTorch and run on CPU or CUDA. None mutate inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
).view(1, 1, 3, 3)


def fast_gaussian_blur(
    image: torch.Tensor,
    kernel_size: int | None,
    sigma: float,
) -> torch.Tensor:
    """Gaussian blur with downsample-blur-upsample fallback for large sigma.

    Args:
        image: ``(B, C, H, W)`` float tensor.
        kernel_size: Odd kernel size. ``None`` derives ``int(2*4*sigma+1) | 1``.
        sigma: Gaussian standard deviation in pixels.

    Returns:
        Blurred image, same shape and dtype as ``image``.

    Notes:
        Sigma below ~0.01 is treated as a no-op. Sigma above ``max(H, W)``
        collapses to the spatial mean (full smoothing). For sigma > 2 the
        image is downsampled by ``int(sigma)`` (capped to ``H // 4, W // 4``)
        before blurring, which keeps the kernel small and the operation
        roughly O(B C H W) rather than O(B C H W sigma**2).
    """

    if sigma < 0.01:
        return image

    _, _, height, width = image.shape

    if sigma > max(height, width):
        return image.mean(dim=(2, 3), keepdim=True).expand(-1, -1, height, width)

    if kernel_size is None:
        kernel_size = int(2 * 4 * sigma + 1) | 1

    if sigma <= 2.0:
        kernel_size = min(kernel_size, (height - 1) * 2 - 1, (width - 1) * 2 - 1)
        if kernel_size < 3:
            return image
        return TF.gaussian_blur(image, kernel_size, sigma)

    downsample = int(sigma)
    downsample = min(downsample, height // 4, width // 4)

    if downsample <= 1:
        kernel_size = min(kernel_size, (height - 1) * 2 - 1, (width - 1) * 2 - 1)
        if kernel_size < 3:
            return image
        return TF.gaussian_blur(image, kernel_size, sigma)

    target_h = max(2, height // downsample)
    target_w = max(2, width // downsample)

    down = F.interpolate(
        image, size=(target_h, target_w), mode="bilinear", align_corners=False
    )

    sigma_down = sigma / (height / target_h)
    kernel_down = int(2 * 4 * sigma_down + 1) | 1
    kernel_down = min(kernel_down, (target_h - 1) * 2 - 1, (target_w - 1) * 2 - 1)

    if kernel_down >= 3:
        blurred_down = TF.gaussian_blur(down, kernel_down, sigma_down)
    else:
        blurred_down = down

    return F.interpolate(
        blurred_down, size=(height, width), mode="bilinear", align_corners=False
    )


def multi_channel_structure_tensor(
    image: torch.Tensor,
    sigma: float = 1.0,
    rho: float = 1.0,
) -> torch.Tensor:
    """Compute the Di Zenzo multi-channel structure tensor.

    Args:
        image: ``(B, C, H, W)`` float tensor in any range; gradients are
            normalised by the batch maximum at the end so absolute scale
            does not propagate.
        sigma: Pre-smoothing Gaussian sigma applied to the image before
            taking derivatives. Suppresses noise.
        rho: Window-integration Gaussian sigma applied to the gradient
            products. Sets the spatial extent over which the structure
            tensor is averaged.

    Returns:
        ``(B, 3, H, W)`` tensor containing ``(S_xx, S_xy, S_yy)`` along the
        channel dimension. Normalised so the batch-wise maximum of
        ``S_xx + S_yy`` is 1.

    Notes:
        Di Zenzo's method sums gradient products across the C channels
        before window integration. This catches iso-luminant edges
        (e.g. red->blue with equal luminance) that a single-channel luma
        gradient would miss.
    """

    channels = image.shape[1]
    device = image.device
    dtype = image.dtype

    kernel_pre = int(2 * 4 * sigma + 1) | 1
    smoothed = fast_gaussian_blur(image, kernel_pre, sigma)

    sobel_x = _SOBEL_X.to(device=device, dtype=dtype).repeat(channels, 1, 1, 1)
    sobel_y = _SOBEL_Y.to(device=device, dtype=dtype).repeat(channels, 1, 1, 1)

    grad_x = F.conv2d(smoothed, sobel_x, padding=1, groups=channels)
    grad_y = F.conv2d(smoothed, sobel_y, padding=1, groups=channels)

    ixx_c = grad_x.square()
    ixy_c = grad_x * grad_y
    iyy_c = grad_y.square()

    sxx_raw = ixx_c.sum(dim=1, keepdim=True)
    sxy_raw = ixy_c.sum(dim=1, keepdim=True)
    syy_raw = iyy_c.sum(dim=1, keepdim=True)

    kernel_rho = int(2 * 4 * rho + 1) | 1
    sxx = fast_gaussian_blur(sxx_raw, kernel_rho, rho)
    sxy = fast_gaussian_blur(sxy_raw, kernel_rho, rho)
    syy = fast_gaussian_blur(syy_raw, kernel_rho, rho)

    magnitude = sxx + syy
    max_val = torch.amax(magnitude, dim=(1, 2, 3), keepdim=True) + 1e-6
    sxx = sxx / max_val
    sxy = sxy / max_val
    syy = syy / max_val

    return torch.cat([sxx, sxy, syy], dim=1)


def multiscale_structure_tensor_v2(
    image: torch.Tensor,
    levels: int = 3,
    base_sigma: float = 1.0,
    octave_step: float = 1.5,
    smoothing_factor: float = 1.0,
    power_factor: float = 3.0,
) -> torch.Tensor:
    """Compute SAD-GS's "true" multi-scale structure tensor (v2).

    At each level :math:`l \\in [0, L)`, the image is progressively blurred
    to ``target_sigma_l = base_sigma * octave_step^l``. A structure tensor
    is computed on the blurred image at that scale with ``rho = 3 *
    target_sigma_l``. Orientation components are normalised by their own
    trace, then aggregated with a Difference-of-Gaussians band weight
    (raised to ``power_factor``) and a frequency-squared factor so the
    final trace approximates :math:`\\sum_l f_l^2 \\, w_l / \\sum_l w_l`.

    Args:
        image: ``(B, C, H, W)`` float tensor.
        levels: Number of octave levels in the pyramid.
        base_sigma: Smallest scale (level 0 sigma).
        octave_step: Multiplicative step between scales. SAD-GS default 1.5.
        smoothing_factor: If > 0, smooths the DoG band response at each
            level by ``2 * target_sigma`` to give low-frequency context.
        power_factor: Exponent on the DoG band response used as the level
            weight. Higher values concentrate weight on the dominant scale.

    Returns:
        ``(B, 3, H, W)`` tensor ``(S_xx, S_xy, S_yy)``. The principal
        eigenvalue of this tensor approximates a per-pixel
        :math:`f_\\text{dominant}^2`; ``wavelength_min`` derived from it is
        in pixels.
    """

    batch, _, height, width = image.shape
    device = image.device
    dtype = image.dtype

    accum_sxx = torch.zeros((batch, 1, height, width), device=device, dtype=dtype)
    accum_sxy = torch.zeros((batch, 1, height, width), device=device, dtype=dtype)
    accum_syy = torch.zeros((batch, 1, height, width), device=device, dtype=dtype)
    accum_weight = torch.zeros((batch, 1, height, width), device=device, dtype=dtype)

    current_smoothed = image
    sigma_accum = 0.0

    for level in range(levels):
        band_freq = 1.0 / (octave_step**level)
        target_sigma = base_sigma * (octave_step**level)

        sigma_inc = math.sqrt(max(1e-6, target_sigma**2 - sigma_accum**2))
        next_smoothed = fast_gaussian_blur(current_smoothed, None, sigma_inc)

        band_response = (
            (current_smoothed - next_smoothed).pow(2).sum(dim=1, keepdim=True).sqrt()
        )

        if level > 0 and smoothing_factor > 0:
            smoothing_sigma = target_sigma * 2.0
            band_response = fast_gaussian_blur(band_response, None, smoothing_sigma)

        integration_rho = target_sigma * 3.0
        level_st = multi_channel_structure_tensor(
            next_smoothed, sigma=target_sigma, rho=integration_rho
        )
        sxx_i = level_st[:, 0:1]
        sxy_i = level_st[:, 1:2]
        syy_i = level_st[:, 2:3]

        trace_i = sxx_i + syy_i + 1e-6
        sxx_norm = sxx_i / trace_i
        sxy_norm = sxy_i / trace_i
        syy_norm = syy_i / trace_i

        weight_i = band_response.pow(power_factor)
        target_wavelength_sq_inv = band_freq**2

        accum_sxx += sxx_norm * weight_i * target_wavelength_sq_inv
        accum_sxy += sxy_norm * weight_i * target_wavelength_sq_inv
        accum_syy += syy_norm * weight_i * target_wavelength_sq_inv
        accum_weight += weight_i

        current_smoothed = next_smoothed
        sigma_accum = target_sigma

    final_sxx = accum_sxx / (accum_weight + 1e-6)
    final_sxy = accum_sxy / (accum_weight + 1e-6)
    final_syy = accum_syy / (accum_weight + 1e-6)

    return torch.cat([final_sxx, final_sxy, final_syy], dim=1)


def wavelength_min_from_structure_tensor(
    structure_tensor: torch.Tensor,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """Extract the SAD-GS dominant wavelength in pixels from a structure tensor.

    Args:
        structure_tensor: ``(B, 3, H, W)`` containing ``(S_xx, S_xy, S_yy)``.
        epsilon: Added to ``sqrt(lambda_1)`` before inversion to keep flat
            regions from producing infinite wavelength.

    Returns:
        ``(B, 1, H, W)`` tensor of ``wavelength_min`` in pixels. Sharp
        single-pixel edges produce values near 1; smooth regions produce
        large values bounded by ``1 / epsilon``.

    Notes:
        Principal eigenvalue of a 2x2 symmetric matrix
        :math:`[[a, b], [b, c]]` is
        :math:`\\lambda_1 = (a + c)/2 + \\sqrt{((a-c)/2)^2 + b^2}`. The
        wavelength is :math:`1 / (\\sqrt{\\lambda_1} + \\epsilon)`.
    """

    sxx = structure_tensor[:, 0:1]
    sxy = structure_tensor[:, 1:2]
    syy = structure_tensor[:, 2:3]

    half_trace = 0.5 * (sxx + syy)
    half_diff = 0.5 * (sxx - syy)
    discriminant = (half_diff.square() + sxy.square()).clamp_min(0.0).sqrt()
    lambda_1 = half_trace + discriminant

    return 1.0 / (lambda_1.clamp_min(0.0).sqrt() + epsilon)
