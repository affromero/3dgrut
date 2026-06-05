# SPDX-FileCopyrightText: Copyright (c) 2026 Hax AI
# SPDX-License-Identifier: Apache-2.0
#
# This file ports the per-Gaussian frequency-violation (eta) scoring from
#   SAD-GS: Structure-Aware Densification for 3D Gaussian Splatting
#   Lyu, Tewari, Chen, Leimkühler, Theobalt -- SIGGRAPH 2026
#   https://arxiv.org/abs/2604.28016
#   https://github.com/LinjieLyu/SADGS  (MIT License)
#
# The two scoring modes (wavelength and projection) are extracted verbatim
# from SADGS/utils/freq_utils.py::update_freq_stats_online lines 313-376.
# A reference pinhole projection is included for testing against the
# SADGS/utils/freq_utils.py::compute_projected_axes_subset reference.
# Production code uses the existing in-house `project_local_axis_vectors`
# helper in `threedgrut/strategy/gs.py` because it implements full
# RATIONAL/distorted-camera projection rather than SAD-GS's first-order
# pinhole Jacobian linearisation.

"""Per-Gaussian frequency-violation (eta) scoring for SAD-GS densification.

Given the multi-scale structure tensor at each pixel and the projected
2D axis vectors of each Gaussian, eta_3ch measures how much each
Gaussian's projected axis exceeds the local dominant wavelength. eta > 1
indicates the Gaussian's axis is longer than the local feature size and
should be split.

Two scoring modes are provided, both verbatim from SAD-GS:

* :func:`eta_3ch_wavelength` -- per-axis projected length divided by a
  scalar wavelength_min derived from the principal eigenvalue. Direction-
  agnostic: a Gaussian axis of given length gets the same eta regardless
  of its orientation relative to the local edge.
* :func:`eta_3ch_projection` -- bilinear form ``axis^T S axis``, then a
  square root. Direction-aware: axes aligned with the local edge get
  higher eta than axes parallel to it.

SAD-GS's default is wavelength. Hax-CV's port exposes both for diagnostics.

The pinhole reference projection :func:`compute_projected_axes_pinhole`
is used only for cross-validation against the SAD-GS reference; production
calls should reuse the in-house RATIONAL projection in `gs.py`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# eta computation (camera-model-agnostic)
# ---------------------------------------------------------------------------


def eta_3ch_wavelength(
    projected_axes_2d: torch.Tensor,
    sampled_structure_tensor: torch.Tensor,
    epsilon: float = 1e-5,
    axis_length_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute eta per-axis using SAD-GS's wavelength formulation.

    Args:
        projected_axes_2d: ``(N, 3, 2)`` tensor of each Gaussian's three
            projected axis vectors in pixels. Axis dimension is ordered as
            the Gaussian's local axes; the (u, v) components are pixel
            displacements from the Gaussian's projected centre.
        sampled_structure_tensor: ``(N, 3)`` tensor of structure-tensor
            components ``(S_xx, S_xy, S_yy)`` sampled at each Gaussian's
            projected centre.
        epsilon: Added to ``sqrt(lambda_1)`` to keep flat regions finite.
        axis_length_epsilon: Added inside the axis-length square-root to
            keep the gradient finite at degenerate axes.

    Returns:
        ``(N, 3)`` tensor of eta values per Gaussian per axis. eta > 1.0
        means the projected axis is longer than the local wavelength.

    Notes:
        Verbatim from
        ``SADGS/utils/freq_utils.py::update_freq_stats_online`` lines
        313-348.
    """

    sxx = sampled_structure_tensor[:, 0]
    sxy = sampled_structure_tensor[:, 1]
    syy = sampled_structure_tensor[:, 2]

    trace = sxx + syy
    det = sxx * syy - sxy.square()
    discriminant = ((trace / 2.0).square() - det).clamp_min(0.0).sqrt()
    lambda_1 = (trace / 2.0) + discriminant

    wavelength_min = 1.0 / (lambda_1.sqrt() + epsilon)

    axis_lengths = (
        projected_axes_2d.square().sum(dim=2).clamp_min(axis_length_epsilon).sqrt()
    )

    return axis_lengths / wavelength_min.unsqueeze(1)


def eta_3ch_projection(
    projected_axes_2d: torch.Tensor,
    sampled_structure_tensor: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compute eta per-axis using SAD-GS's direction-aware projection formulation.

    Args:
        projected_axes_2d: ``(N, 3, 2)`` Gaussian projected axes in pixels.
        sampled_structure_tensor: ``(N, 3)`` sampled ``(S_xx, S_xy, S_yy)``.
        epsilon: Added before the outer square-root to keep gradients
            stable.

    Returns:
        ``(N, 3)`` tensor of eta values. Equal to
        ``sqrt(axis^T S axis) = sqrt(Sxx u^2 + 2 Sxy u v + Syy v^2)``
        per-Gaussian per-axis.

    Notes:
        Verbatim from
        ``SADGS/utils/freq_utils.py::update_freq_stats_online`` lines
        350-373.
    """

    sxx = sampled_structure_tensor[:, 0:1]
    sxy = sampled_structure_tensor[:, 1:2]
    syy = sampled_structure_tensor[:, 2:3]

    u = projected_axes_2d[:, :, 0]
    v = projected_axes_2d[:, :, 1]

    quadratic = sxx * u.square() + 2.0 * sxy * u * v + syy * v.square()
    return quadratic.clamp_min(epsilon).sqrt()


# ---------------------------------------------------------------------------
# structure-tensor sampling at projected positions
# ---------------------------------------------------------------------------


def sample_structure_tensor_at_pixels(
    structure_tensor_map: torch.Tensor,
    pixel_x: torch.Tensor,
    pixel_y: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Bilinear or nearest-neighbour sampling of a structure-tensor map.

    Args:
        structure_tensor_map: ``(B, 3, H, W)`` structure-tensor map, channels
            ``(S_xx, S_xy, S_yy)``. Only ``B == 1`` is supported.
        pixel_x: ``(N,)`` x-coordinates in pixels in ``[0, W-1]``.
        pixel_y: ``(N,)`` y-coordinates in pixels in ``[0, H-1]``.
        mode: ``"bilinear"`` or ``"nearest"`` (passed to ``grid_sample``).

    Returns:
        ``(N, 3)`` tensor of sampled ``(S_xx, S_xy, S_yy)``.

    Notes:
        Out-of-bounds samples are clamped to zero via ``padding_mode="zeros"``.
        Use this when the caller has already validated visibility; this
        function does not return a valid mask.
    """

    if structure_tensor_map.shape[0] != 1:
        raise ValueError(
            "sample_structure_tensor_at_pixels requires batch size 1, "
            f"got {structure_tensor_map.shape[0]}"
        )

    _, _, height, width = structure_tensor_map.shape

    norm_x = 2.0 * pixel_x / max(1, width - 1) - 1.0
    norm_y = 2.0 * pixel_y / max(1, height - 1) - 1.0

    grid = torch.stack([norm_x, norm_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        structure_tensor_map,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )

    return sampled[0, :, :, 0].permute(1, 0).contiguous()


# ---------------------------------------------------------------------------
# pinhole projection reference (testing only)
# ---------------------------------------------------------------------------


def _build_rotation_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Build a rotation matrix from a quaternion ``(qw, qx, qy, qz)``.

    Args:
        quaternion: ``(N, 4)`` un-normalised quaternions.

    Returns:
        ``(N, 3, 3)`` rotation matrices.

    Notes:
        Matches the convention in ``SADGS/utils/general_utils.py::
        build_rotation`` (also the standard 3DGS convention).
    """

    normalised = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w = normalised[:, 0]
    x = normalised[:, 1]
    y = normalised[:, 2]
    z = normalised[:, 3]

    rotation = torch.zeros(
        (normalised.shape[0], 3, 3),
        device=normalised.device,
        dtype=normalised.dtype,
    )
    rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotation[:, 0, 1] = 2.0 * (x * y - w * z)
    rotation[:, 0, 2] = 2.0 * (x * z + w * y)
    rotation[:, 1, 0] = 2.0 * (x * y + w * z)
    rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotation[:, 1, 2] = 2.0 * (y * z - w * x)
    rotation[:, 2, 0] = 2.0 * (x * z - w * y)
    rotation[:, 2, 1] = 2.0 * (y * z + w * x)
    rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)

    return rotation


def compute_projected_axes_pinhole(
    means_2d: torch.Tensor,
    depths: torch.Tensor,
    scales: torch.Tensor,
    rotations_quaternion: torch.Tensor,
    world_view_transform: torch.Tensor,
    focal_x: float,
    focal_y: float,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Compute projected 3D Gaussian axis vectors under a pinhole camera.

    This is the SAD-GS reference projection. **For testing equivalence
    against the upstream implementation only.** Production code should use
    the existing in-house ``project_local_axis_vectors`` in
    ``threedgrut/strategy/gs.py``, which handles RATIONAL distortion.

    Args:
        means_2d: ``(N, 2)`` projected centres in pixel coordinates.
        depths: ``(N,)`` depths in camera coordinates.
        scales: ``(N, 3)`` Gaussian axis scales (pre-exponential).
        rotations_quaternion: ``(N, 4)`` quaternions.
        world_view_transform: ``(4, 4)`` world-to-view transform (column
            vector convention: ``W_view[3, :3]`` is translation,
            ``W_view[:3, :3]`` is rotation).
        focal_x: Pinhole focal length in pixels (x axis).
        focal_y: Pinhole focal length in pixels (y axis).
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        ``(N, 3, 2)`` projected axis vectors in pixels.

    Notes:
        Verbatim from
        ``SADGS/utils/freq_utils.py::compute_projected_axes_subset``.
        Uses cx = W/2, cy = H/2 (the principal point hardcoded in the
        SAD-GS reference).
    """

    inv_z = 1.0 / (depths + 1e-7)
    inv_z2 = inv_z * inv_z

    vec_x = (means_2d[:, 0] - image_width * 0.5) * depths / focal_x
    vec_y = (means_2d[:, 1] - image_height * 0.5) * depths / focal_y

    jac_00 = focal_x * inv_z
    jac_02 = -focal_x * vec_x * inv_z2
    jac_11 = focal_y * inv_z
    jac_12 = -focal_y * vec_y * inv_z2

    view_transposed = world_view_transform.transpose(0, 1)
    rotation_view = view_transposed[:3, :3]
    rotation_local = _build_rotation_from_quaternion(rotations_quaternion)
    rotation_total = torch.bmm(
        rotation_view.unsqueeze(0).expand(scales.shape[0], -1, -1),
        rotation_local,
    )

    axes_cam = rotation_total * scales.unsqueeze(1)

    ax_x = axes_cam[:, 0, :]
    ax_y = axes_cam[:, 1, :]
    ax_z = axes_cam[:, 2, :]

    u_vec = jac_00.unsqueeze(1) * ax_x + jac_02.unsqueeze(1) * ax_z
    v_vec = jac_11.unsqueeze(1) * ax_y + jac_12.unsqueeze(1) * ax_z

    return torch.stack([u_vec, v_vec], dim=2)
