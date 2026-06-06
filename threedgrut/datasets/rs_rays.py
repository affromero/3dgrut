# SPDX-License-Identifier: Apache-2.0
"""Per-pixel rolling-shutter world-space rays for the 3dgrt ray path.

Each output row ``v`` is captured at the interpolated pose
``slerp/lerp(start, end, v/(H-1))`` -- row 0 = start = T(-0.5), row H-1 =
end = T(+0.5) -- matching the analytic oracle (render_oracle.render_rs_frame)
exactly (same scipy.Slerp, same alpha ramp). Applied to the dataset's RDF
camera bearings, this yields world-space per-pixel rays that, fed to the 3dgrt
ray tracer with an IDENTITY world transform (T_to_world = eye), render TRUE
per-pixel rolling shutter -- no per-Gaussian single-pose approximation, and no
CUDA change (the tracer applies T_to_world as the rayToWorld affine, so
identity is a pure passthrough; backward treats rays as constants).
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def build_rs_world_rays(rays_dir_cam, c2w_start, c2w_end):
    """Build per-pixel rolling-shutter world rays.

    Args:
        rays_dir_cam: ``[1, H, W, 3]`` RDF camera-space bearings (from
            create_equirect_camera).
        c2w_start: ``[4, 4]`` RDF camera-to-world at shutter start (row 0).
        c2w_end: ``[4, 4]`` RDF camera-to-world at shutter end (row H-1).

    Returns:
        ``(rays_ori_world, rays_dir_world)``, each ``[1, H, W, 3]``.
    """
    device, dtype = rays_dir_cam.device, rays_dir_cam.dtype
    # The dataset stores poses as [1, 4, 4] (an unsqueeze(0) survives the
    # collate strip), so accept either [1, 4, 4] or [4, 4].
    c2w_start = c2w_start.reshape(4, 4)
    c2w_end = c2w_end.reshape(4, 4)
    h, w = int(rays_dir_cam.shape[1]), int(rays_dir_cam.shape[2])
    alphas = np.arange(h, dtype=np.float64) / max(h - 1, 1)
    r0 = c2w_start[:3, :3].detach().cpu().numpy()
    r1 = c2w_end[:3, :3].detach().cpu().numpy()
    t0 = c2w_start[:3, 3].detach().cpu().numpy()
    t1 = c2w_end[:3, 3].detach().cpu().numpy()
    rot_rows = (
        Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([r0, r1])))(alphas)
        .as_matrix()
    )  # [H, 3, 3]
    t_rows = (1.0 - alphas)[:, None] * t0 + alphas[:, None] * t1  # [H, 3]
    rot_t = torch.tensor(rot_rows, device=device, dtype=dtype)
    t_t = torch.tensor(t_rows, device=device, dtype=dtype)
    bearings = rays_dir_cam[0]  # [H, W, 3]
    dirs = torch.einsum("hij,hwj->hwi", rot_t, bearings).unsqueeze(0)
    oris = t_t[None, :, None, :].expand(1, h, w, 3).contiguous()
    return oris, dirs
