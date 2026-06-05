# SPDX-FileCopyrightText: Copyright (c) 2026 Hax AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SAD-GS eta scoring and pinhole-reference projection.

These tests cover:

* Both eta modes (wavelength, projection) against analytic ground truth.
* Bilinear structure-tensor sampling against numpy reference.
* Pinhole projection against a known forward case where the math is
  hand-computable.

A separate numerical-equivalence script outside the test suite compares
each function against the SAD-GS reference implementation. We deliberately
do not import that reference inside pytest because it is a developer-only
artifact.
"""

from __future__ import annotations

import math

import pytest
import torch

from threedgrut.strategy.sadgs_eta import (
    compute_projected_axes_pinhole,
    eta_3ch_projection,
    eta_3ch_wavelength,
    sample_structure_tensor_at_pixels,
)


# ---------------------------------------------------------------------------
# eta_3ch_wavelength
# ---------------------------------------------------------------------------


def test_eta_wavelength_axis_smaller_than_wavelength_is_below_one() -> None:
    """A Gaussian axis shorter than wavelength_min produces eta < 1.

    Set Sxx so the principal eigenvalue is 1 (Syy=0, Sxy=0 -> lambda_1=1,
    wavelength_min = 1/(1+eps) ~ 1 px). An axis of length 0.5 px should
    give eta = 0.5 / 1 = 0.5.
    """

    axes = torch.tensor([[[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    sampled_st = torch.tensor([[1.0, 0.0, 0.0]])
    eta = eta_3ch_wavelength(axes, sampled_st)
    expected = 0.5 / (1.0 + 1e-5)
    assert math.isclose(eta[0, 0].item(), expected, rel_tol=1e-4)
    assert eta[0, 1].item() < 1e-3
    assert eta[0, 2].item() < 1e-3


def test_eta_wavelength_long_axis_exceeds_threshold() -> None:
    """A 4-pixel axis on a Sxx=1 (wavelength~1) region gives eta ~= 4."""

    axes = torch.tensor([[[4.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    sampled_st = torch.tensor([[1.0, 0.0, 0.0]])
    eta = eta_3ch_wavelength(axes, sampled_st)
    assert math.isclose(eta[0, 0].item(), 4.0, rel_tol=1e-3)


def test_eta_wavelength_is_orientation_agnostic() -> None:
    """Wavelength mode depends only on axis length, not direction."""

    axes_horizontal = torch.tensor([[[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    axes_vertical = torch.tensor([[[0.0, 3.0], [0.0, 0.0], [0.0, 0.0]]])
    axes_diagonal = torch.tensor(
        [[[3.0 / math.sqrt(2), 3.0 / math.sqrt(2)], [0.0, 0.0], [0.0, 0.0]]]
    )
    sampled_st = torch.tensor([[1.0, 0.0, 0.0]])

    eta_h = eta_3ch_wavelength(axes_horizontal, sampled_st)[0, 0].item()
    eta_v = eta_3ch_wavelength(axes_vertical, sampled_st)[0, 0].item()
    eta_d = eta_3ch_wavelength(axes_diagonal, sampled_st)[0, 0].item()

    assert math.isclose(eta_h, eta_v, rel_tol=1e-4)
    assert math.isclose(eta_h, eta_d, rel_tol=1e-4)


# ---------------------------------------------------------------------------
# eta_3ch_projection
# ---------------------------------------------------------------------------


def test_eta_projection_aligned_axis_picks_up_full_energy() -> None:
    """For Sxx = 1, Syy = 0, an axis along +x gives eta = sqrt(u^2) = |u|."""

    axes = torch.tensor([[[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    sampled_st = torch.tensor([[1.0, 0.0, 0.0]])
    eta = eta_3ch_projection(axes, sampled_st)
    assert math.isclose(eta[0, 0].item(), 2.0, rel_tol=1e-4)


def test_eta_projection_orthogonal_axis_is_zero() -> None:
    """For Sxx = 1, Syy = 0, an axis along +y gives eta = 0."""

    axes = torch.tensor([[[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
    sampled_st = torch.tensor([[1.0, 0.0, 0.0]])
    eta = eta_3ch_projection(axes, sampled_st)
    assert eta[0, 0].item() < 1e-4


def test_eta_projection_direction_aware_unlike_wavelength() -> None:
    """Projection mode reads zero for orthogonal direction; wavelength does not."""

    axes_perpendicular = torch.tensor([[[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])
    sampled_st = torch.tensor([[1.0, 0.0, 0.0]])

    eta_wl = eta_3ch_wavelength(axes_perpendicular, sampled_st)[0, 0].item()
    eta_pr = eta_3ch_projection(axes_perpendicular, sampled_st)[0, 0].item()

    assert eta_wl > 1.0
    assert eta_pr < 1e-3


# ---------------------------------------------------------------------------
# sample_structure_tensor_at_pixels
# ---------------------------------------------------------------------------


def test_sample_structure_tensor_returns_exact_value_at_integer_pixels() -> None:
    """Sampling at integer pixel coordinates returns the stored value."""

    st_map = torch.zeros(1, 3, 8, 8)
    st_map[0, 0, 4, 5] = 0.75
    st_map[0, 1, 4, 5] = 0.10
    st_map[0, 2, 4, 5] = 0.25

    pixel_x = torch.tensor([5.0])
    pixel_y = torch.tensor([4.0])

    sampled = sample_structure_tensor_at_pixels(st_map, pixel_x, pixel_y)
    assert math.isclose(sampled[0, 0].item(), 0.75, abs_tol=1e-5)
    assert math.isclose(sampled[0, 1].item(), 0.10, abs_tol=1e-5)
    assert math.isclose(sampled[0, 2].item(), 0.25, abs_tol=1e-5)


def test_sample_structure_tensor_bilinear_interpolation() -> None:
    """Sampling at half-pixel coordinates returns the bilinear average."""

    st_map = torch.zeros(1, 3, 4, 4)
    st_map[0, 0, 0, 0] = 1.0
    st_map[0, 0, 0, 1] = 0.0
    st_map[0, 0, 1, 0] = 0.0
    st_map[0, 0, 1, 1] = 0.0

    sampled = sample_structure_tensor_at_pixels(
        st_map, torch.tensor([0.5]), torch.tensor([0.5])
    )
    assert math.isclose(sampled[0, 0].item(), 0.25, rel_tol=1e-4)


def test_sample_structure_tensor_out_of_bounds_returns_zero() -> None:
    """Out-of-bounds samples are zero-padded."""

    st_map = torch.ones(1, 3, 4, 4)
    sampled = sample_structure_tensor_at_pixels(
        st_map, torch.tensor([-10.0]), torch.tensor([-10.0])
    )
    assert sampled.abs().max().item() < 1e-4


def test_sample_structure_tensor_rejects_multi_batch() -> None:
    """Multi-batch input raises ValueError."""

    st_map = torch.zeros(2, 3, 4, 4)
    with pytest.raises(ValueError):
        sample_structure_tensor_at_pixels(
            st_map, torch.tensor([0.0]), torch.tensor([0.0])
        )


# ---------------------------------------------------------------------------
# compute_projected_axes_pinhole
# ---------------------------------------------------------------------------


def test_pinhole_projection_returns_expected_shape() -> None:
    """Output has shape ``(N, 3, 2)``."""

    n_gaussians = 5
    torch.manual_seed(0)
    means_2d = torch.rand(n_gaussians, 2) * 64
    depths = torch.rand(n_gaussians) * 5 + 1.0
    scales = torch.rand(n_gaussians, 3) * 0.1 + 0.01
    rotations = torch.randn(n_gaussians, 4)
    world_view = torch.eye(4)

    projected = compute_projected_axes_pinhole(
        means_2d, depths, scales, rotations, world_view,
        focal_x=100.0, focal_y=100.0,
        image_width=64, image_height=64,
    )
    assert projected.shape == (n_gaussians, 3, 2)


def test_pinhole_projection_identity_rotation_zero_translation_known_case() -> None:
    """Hand-computed case: 1 Gaussian at image centre, identity rotation, depth 1.

    With identity quaternion ``(1, 0, 0, 0)`` (no rotation), identity
    world->view, focal 100, the camera-space local axes are
    ``rotation_total @ diag(scales) = diag(scales)``. The Jacobian at
    centre-depth z=1 with mean_2d = (W/2, H/2) is

    J = [[fx/z, 0, 0], [0, fy/z, 0]] (no perspective term because vec_x = vec_y = 0)

    so projected axes are ``(fx, 0)``, ``(0, fy)``, ``(0, 0)`` times
    ``scales``.
    """

    means_2d = torch.tensor([[32.0, 32.0]])
    depths = torch.tensor([1.0])
    scales = torch.tensor([[0.05, 0.07, 0.09]])
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    world_view = torch.eye(4)

    projected = compute_projected_axes_pinhole(
        means_2d, depths, scales, rotations, world_view,
        focal_x=100.0, focal_y=100.0,
        image_width=64, image_height=64,
    )

    expected = torch.tensor(
        [[[100.0 * 0.05, 0.0],
          [0.0, 100.0 * 0.07],
          [0.0, 0.0]]],
    )
    torch.testing.assert_close(projected, expected, atol=1e-4, rtol=1e-4)
