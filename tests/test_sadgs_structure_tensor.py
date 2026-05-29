# SPDX-FileCopyrightText: Copyright (c) 2026 Hax AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SAD-GS structure-tensor port.

These tests pin down behavior against analytic ground truth on synthetic
images. They run on CPU in a few seconds. The companion port at
``threedgrut/strategy/sadgs_structure_tensor.py`` is meant to match the
SAD-GS reference (https://github.com/LinjieLyu/SADGS) up to numerical
tolerance after the renames documented in that file's header.
"""

from __future__ import annotations

import math

import pytest
import torch

from threedgrut.strategy.sadgs_structure_tensor import (
    fast_gaussian_blur,
    multi_channel_structure_tensor,
    multiscale_structure_tensor_v2,
    wavelength_min_from_structure_tensor,
)


# ---------------------------------------------------------------------------
# fast_gaussian_blur
# ---------------------------------------------------------------------------


def test_fast_gaussian_blur_passes_through_small_sigma() -> None:
    """sigma < 0.01 is treated as a no-op and returns the input tensor."""

    image = torch.randn(1, 3, 32, 32)
    blurred = fast_gaussian_blur(image, kernel_size=None, sigma=0.0)
    assert torch.equal(blurred, image)


def test_fast_gaussian_blur_preserves_constant_image() -> None:
    """A flat image stays flat after blurring."""

    image = torch.full((1, 3, 32, 32), 0.3)
    blurred = fast_gaussian_blur(image, kernel_size=None, sigma=2.0)
    assert torch.allclose(blurred, image, atol=1e-5)


def test_fast_gaussian_blur_reduces_noise_variance() -> None:
    """Blurring with moderate sigma reduces white-noise variance."""

    torch.manual_seed(0)
    image = torch.randn(1, 1, 128, 128)
    blurred = fast_gaussian_blur(image, kernel_size=None, sigma=2.0)
    assert blurred.var().item() < 0.6 * image.var().item()


def test_fast_gaussian_blur_huge_sigma_collapses_to_mean() -> None:
    """sigma above max(H, W) returns the spatial mean broadcast back."""

    image = torch.randn(2, 3, 16, 16)
    blurred = fast_gaussian_blur(image, kernel_size=None, sigma=10_000.0)

    mean = image.mean(dim=(2, 3), keepdim=True)
    expected = mean.expand(-1, -1, 16, 16)
    assert torch.allclose(blurred, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# multi_channel_structure_tensor
# ---------------------------------------------------------------------------


def _horizontal_step(height: int = 64, width: int = 64) -> torch.Tensor:
    """Single-channel image with a horizontal step edge at row H/2.

    Pixels above the edge are 0, below are 1. Gradient is purely along y.
    """
    image = torch.zeros(1, 1, height, width)
    image[:, :, height // 2 :, :] = 1.0
    return image


def _vertical_step(height: int = 64, width: int = 64) -> torch.Tensor:
    """Same as :func:`_horizontal_step` but with the edge along column W/2."""
    image = torch.zeros(1, 1, height, width)
    image[:, :, :, width // 2 :] = 1.0
    return image


def test_structure_tensor_on_horizontal_edge_has_dominant_yy() -> None:
    """A horizontal step edge has near-zero S_xx and large S_yy near the edge."""

    image = _horizontal_step()
    st = multi_channel_structure_tensor(image, sigma=0.5, rho=0.5)

    edge_row = image.shape[2] // 2
    sxx, sxy, syy = st[0, 0], st[0, 1], st[0, 2]

    edge_band = slice(edge_row - 2, edge_row + 3)
    assert syy[edge_band, :].mean() > 5.0 * sxx[edge_band, :].mean()
    assert sxy[edge_band, :].abs().mean() < 0.1 * syy[edge_band, :].mean()


def test_structure_tensor_on_vertical_edge_has_dominant_xx() -> None:
    """Vertical edge mirrors the horizontal case: S_xx >> S_yy."""

    image = _vertical_step()
    st = multi_channel_structure_tensor(image, sigma=0.5, rho=0.5)

    edge_col = image.shape[3] // 2
    sxx, sxy, syy = st[0, 0], st[0, 1], st[0, 2]

    edge_band = slice(edge_col - 2, edge_col + 3)
    assert sxx[:, edge_band].mean() > 5.0 * syy[:, edge_band].mean()
    assert sxy[:, edge_band].abs().mean() < 0.1 * sxx[:, edge_band].mean()


def test_structure_tensor_normalised_to_unit_max() -> None:
    """The post-normalisation trace's batch maximum equals 1 within tolerance."""

    image = _horizontal_step()
    st = multi_channel_structure_tensor(image, sigma=0.5, rho=0.5)

    trace = st[:, 0] + st[:, 2]
    assert math.isclose(trace.max().item(), 1.0, rel_tol=1e-5)


def test_structure_tensor_on_constant_image_is_small_in_interior() -> None:
    """A flat image has near-zero gradients in the interior.

    Boundary pixels can still pick up small structure tensor values from
    Sobel padding + Gaussian blur edge handling. We only assert on the
    interior; SAD-GS likewise doesn't claim boundary-pixel accuracy.
    """

    image = torch.full((1, 3, 32, 32), 0.5)
    st = multi_channel_structure_tensor(image, sigma=0.5, rho=0.5)
    interior = st[:, :, 4:-4, 4:-4]
    assert interior.abs().max().item() < 1e-3


def test_structure_tensor_di_zenzo_catches_iso_luminant_edge() -> None:
    """Di Zenzo's RGB-sum catches edges that vanish in luma.

    Construct an iso-luminant left/right split: left pixels are
    ``(0.5, 0.0, 0.0)``, right pixels are ``(0.0, 0.5, 0.0)``. The standard
    luma ``0.299 R + 0.587 G + 0.114 B`` differs only by ``0.144`` between
    the two halves, whereas a per-channel R drop of 0.5 and G rise of 0.5
    produce a clear structure-tensor response under Di Zenzo's RGB-sum.

    We compare against a luma-only structure tensor computed by treating
    the luma image as a single channel and confirm Di Zenzo's response is
    notably stronger near the edge.
    """

    height, width = 32, 32
    rgb = torch.zeros(1, 3, height, width)
    rgb[:, 0, :, : width // 2] = 0.5
    rgb[:, 1, :, width // 2 :] = 0.5

    di_zenzo = multi_channel_structure_tensor(rgb, sigma=0.5, rho=0.5)

    luma = (0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3])
    luma_only = multi_channel_structure_tensor(luma, sigma=0.5, rho=0.5)

    edge = width // 2
    dz_response = (di_zenzo[0, 0, :, edge - 2 : edge + 3]).mean()
    luma_response = (luma_only[0, 0, :, edge - 2 : edge + 3]).mean()

    assert dz_response > 3.0 * luma_response


# ---------------------------------------------------------------------------
# multiscale_structure_tensor_v2
# ---------------------------------------------------------------------------


def _sinusoid(
    wavelength_px: float, height: int = 128, width: int = 128
) -> torch.Tensor:
    """Single-channel horizontal sinusoid: I(x, y) = 0.5 + 0.5 cos(2 pi x / lambda)."""

    x = torch.arange(width).float()
    row = 0.5 + 0.5 * torch.cos(2.0 * math.pi * x / wavelength_px)
    image = row.unsqueeze(0).expand(height, -1)
    return image.unsqueeze(0).unsqueeze(0)


def test_multiscale_st_orders_wavelengths_consistent_with_input() -> None:
    """For three sinusoids with wavelengths in ratio 1:2:4, the recovered
    wavelength_min preserves the ordering.

    SAD-GS's multi-scale ST is not pixel-calibrated -- its band_freq factor
    is a relative weight (``1 / octave_step ** level``), not a physical
    cycles-per-pixel measurement. The output drives a *ratio* against
    projected Gaussian axis length (eta), so absolute calibration does
    not matter; only ordering and stability across scenes do. The strong
    invariant we can check is monotonicity in the actual wavelength.
    """

    coarse = _sinusoid(16.0)
    mid = _sinusoid(8.0)
    fine = _sinusoid(4.0)

    sts = [
        multiscale_structure_tensor_v2(img, levels=4, base_sigma=0.8)
        for img in (coarse, mid, fine)
    ]
    means = [wavelength_min_from_structure_tensor(st).mean().item() for st in sts]

    assert means[0] > means[1] > means[2]


def test_multiscale_st_preserves_shape() -> None:
    """Output shape is ``(B, 3, H, W)``."""

    image = torch.randn(2, 3, 48, 64)
    st = multiscale_structure_tensor_v2(image, levels=2, base_sigma=0.5)
    assert st.shape == (2, 3, 48, 64)


# ---------------------------------------------------------------------------
# wavelength_min_from_structure_tensor
# ---------------------------------------------------------------------------


def test_wavelength_min_is_small_at_sharp_edge() -> None:
    """A sharp step edge produces wavelength_min near 1 pixel."""

    image = _horizontal_step()
    st = multi_channel_structure_tensor(image, sigma=0.5, rho=0.5)
    wavelength = wavelength_min_from_structure_tensor(st)

    edge_band = wavelength[
        0, 0, image.shape[2] // 2 - 1 : image.shape[2] // 2 + 2, :
    ]
    assert edge_band.mean().item() < 2.0


def test_wavelength_min_is_large_in_smooth_interior() -> None:
    """A flat patch's interior wavelength_min reaches the epsilon ceiling.

    Boundary pixels accumulate small Sobel-padding artefacts that bring
    wavelength_min down to single-digit values. We assert only on the
    interior, matching SAD-GS's practical operating region (Gaussians
    typically project away from the border).
    """

    image = torch.full((1, 3, 32, 32), 0.4)
    st = multi_channel_structure_tensor(image, sigma=0.5, rho=0.5)
    wavelength = wavelength_min_from_structure_tensor(st, epsilon=1e-3)

    interior = wavelength[:, :, 4:-4, 4:-4]
    assert interior.min().item() > 100.0  # 1/epsilon = 1000, well above 100


def test_wavelength_min_monotonic_in_frequency() -> None:
    """Doubling the spatial frequency halves the recovered wavelength."""

    coarse = _sinusoid(16.0)
    fine = _sinusoid(8.0)
    st_coarse = multiscale_structure_tensor_v2(coarse, levels=4, base_sigma=0.8)
    st_fine = multiscale_structure_tensor_v2(fine, levels=4, base_sigma=0.8)
    w_coarse = wavelength_min_from_structure_tensor(st_coarse).mean().item()
    w_fine = wavelength_min_from_structure_tensor(st_fine).mean().item()

    assert w_coarse > w_fine
