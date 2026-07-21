# SPDX-License-Identifier: Apache-2.0
"""Differentiable camera-pose interpolation from per-frame knot poses.

Torch counterpart of the dataset's scipy knot interpolation: given J knot
poses on fixed relative stamps and Q query stamps that DEPEND ON LEARNABLE
TIMING PARAMETERS (exposure offsets), produce camera-to-world rotations and
centers with gradients flowing from the query stamps into the poses — and
from there through the world-space rays into the tracer's backward ray
gradients. Quaternion slerp between bracketing knots; piecewise-linear
translation. Matches scipy.spatial.transform.Slerp on fixed queries.
"""

import torch


def slerp_knot_poses(knot_stamps, knot_quats_xyzw, knot_trans, query_stamps):
    """Interpolate camera-to-world poses at differentiable query stamps.

    Args:
        knot_stamps: ``[J]`` knot stamps (seconds, ascending, constants).
        knot_quats_xyzw: ``[J, 4]`` unit quaternions (scipy xyzw order),
            constants.
        knot_trans: ``[J, 3]`` camera centers, constants.
        query_stamps: ``[Q]`` query stamps inside the knot span; may carry
            gradients (e.g. functions of a learnable exposure).

    Returns:
        ``(rotations, centers)``: ``[Q, 3, 3]`` rotation matrices and
        ``[Q, 3]`` camera centers, differentiable w.r.t. query_stamps.
    """
    if bool((query_stamps < knot_stamps[0]).any()) or bool(
        (query_stamps > knot_stamps[-1]).any()
    ):
        raise ValueError(
            "query stamps fall outside the knot span "
            f"[{float(knot_stamps[0]):.4f}, {float(knot_stamps[-1]):.4f}]; "
            "re-emit pose_knots.json with a larger t_exp margin."
        )
    # Bracketing interval per query (searchsorted on constant stamps; the
    # bucket INDEX is a non-differentiable integer, the fraction inside the
    # bucket carries the gradient).
    upper = torch.searchsorted(
        knot_stamps, query_stamps.detach(), right=True
    ).clamp(1, knot_stamps.shape[0] - 1)
    lower = upper - 1
    t0 = knot_stamps[lower]
    t1 = knot_stamps[upper]
    alpha = ((query_stamps - t0) / (t1 - t0)).clamp(0.0, 1.0)

    q0 = knot_quats_xyzw[lower]
    q1 = knot_quats_xyzw[upper]
    # Shortest-path slerp with a lerp fallback for nearly-parallel pairs
    # (acos gradient blows up as dot -> 1; adjacent 1-2 ms knots are nearly
    # parallel almost always, so the nlerp branch is the common path and
    # is numerically exact there).
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    parallel = dot > 0.9995
    theta = torch.arccos(dot.clamp(max=1.0 - 1e-7))
    sin_theta = torch.sin(theta)
    w0_slerp = torch.sin((1.0 - alpha[:, None]) * theta) / sin_theta
    w1_slerp = torch.sin(alpha[:, None] * theta) / sin_theta
    w0_lerp = 1.0 - alpha[:, None]
    w1_lerp = alpha[:, None]
    w0 = torch.where(parallel, w0_lerp, w0_slerp)
    w1 = torch.where(parallel, w1_lerp, w1_slerp)
    quats = w0 * q0 + w1 * q1
    quats = quats / quats.norm(dim=-1, keepdim=True)

    centers = (1.0 - alpha[:, None]) * knot_trans[lower] + alpha[
        :, None
    ] * knot_trans[upper]
    return _quat_xyzw_to_matrix(quats), centers


def _quat_xyzw_to_matrix(quats):
    """Convert ``[Q, 4]`` xyzw unit quaternions to ``[Q, 3, 3]`` matrices."""
    x, y, z, w = quats.unbind(dim=-1)
    two = 2.0
    row0 = torch.stack(
        [
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
        ],
        dim=-1,
    )
    row1 = torch.stack(
        [
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
        ],
        dim=-1,
    )
    row2 = torch.stack(
        [
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def build_rs_world_rays_torch(
    rays_dir_cam, rotations_start, centers_start, rotations_end, centers_end
):
    """Per-row rolling-shutter world rays, differentiable in the poses.

    Torch counterpart of ``rs_rays.build_rs_world_rays`` for one (start,
    end) pose pair: row v is captured at the pose lerp/nlerp-interpolated
    at ``v/(H-1)`` between start and end (row 0 = start, row H-1 = end).

    Args:
        rays_dir_cam: ``[1, H, W, 3]`` camera-space bearings (constants).
        rotations_start / rotations_end: ``[3, 3]`` camera-to-world.
        centers_start / centers_end: ``[3]`` camera centers.

    Returns:
        ``(rays_ori, rays_dir)``, each ``[1, H, W, 3]``, differentiable
        w.r.t. the poses.
    """
    height = rays_dir_cam.shape[1]
    width = rays_dir_cam.shape[2]
    alphas = torch.linspace(
        0.0,
        1.0,
        height,
        device=rays_dir_cam.device,
        dtype=rays_dir_cam.dtype,
    )[:, None]
    # Rotation nlerp per row: adjacent start/end poses are <= a few degrees
    # apart within one readout, where normalized linear interpolation of
    # the matrices' quaternion equivalent and of the matrices themselves
    # agree to first order; interpolate matrices and re-orthogonalize via
    # double cross products to stay in SO(3) without a svd in the graph.
    rot = (1.0 - alphas[:, :, None]) * rotations_start[None] + alphas[
        :, :, None
    ] * rotations_end[None]
    col0 = rot[..., 0]
    col1 = rot[..., 1]
    col0 = col0 / col0.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    col2 = torch.cross(col0, col1, dim=-1)
    col2 = col2 / col2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    col1 = torch.cross(col2, col0, dim=-1)
    rot = torch.stack([col0, col1, col2], dim=-1)  # [H, 3, 3]

    centers = (1.0 - alphas) * centers_start[None] + alphas * centers_end[
        None
    ]  # [H, 3]
    dirs = torch.einsum("hij,hwj->hwi", rot, rays_dir_cam[0]).unsqueeze(0)
    oris = centers[None, :, None, :].expand(1, height, width, 3).contiguous()
    return oris, dirs
