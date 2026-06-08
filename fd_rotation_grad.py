"""Definitive FD check: aggregate rotation gradient dL/dtheta vs finite diff.

Phase A de-risk. The controlled-rig recovery showed rotation barely recovers,
but that has a geometry-absorption confound. This isolates the gradient: a
fixed scene, one render, ray directions rotated by a single scalar theta about
an axis, and the AGGREGATE analytical gradient dL/dtheta (sum over all pixels,
so per-pixel ray-dir-gradient noise averages out) compared to central finite
differences. If analytical != FD, the 3DGUT CUDA backward's ray-direction
gradient is genuinely broken -> the CUDA sensor-pose gradient is the fix.

Also reports the translation directional derivative dL/d(t along an axis) as a
control (ray-origin grads are expected to be correct).
"""

from __future__ import annotations

import os

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _conf():
    cfg_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "configs")
    )
    if not os.path.isdir(cfg_dir):
        cfg_dir = os.path.abspath(
            os.path.join(
                os.path.expanduser("~"),
                "Code/Hax-CV-wt-360-splat/dependencies/3dgrut/configs",
            )
        )
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        return compose(config_name="apps/colmap_3dgut")


def _rot_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Rodrigues 3x3 for a scalar angle about a unit axis (autograd-friendly)."""
    a = axis / axis.norm()
    k = torch.tensor(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ],
        device=angle.device,
    )
    eye = torch.eye(3, device=angle.device)
    return eye + torch.sin(angle) * k + (1 - torch.cos(angle)) * (k @ k)


def main() -> None:
    device = torch.device("cuda")
    conf = _conf()
    from threedgut_tracer.setup_3dgut import setup_3dgut

    plugin = setup_3dgut(conf)
    conf_dict = OmegaConf.to_container(conf)
    raster = plugin.SplatRaster(conf_dict)

    h = w = 96
    n = 4000
    torch.manual_seed(0)
    pd = torch.zeros(n, 12, device=device)
    # Dense, semi-transparent, smoothly-overlapping field so the render is
    # smooth w.r.t. ray perturbation (reliable finite differences) without
    # saturating to fully-opaque (which would zero the gradient).
    pd[:, 0:3] = torch.randn(n, 3, device=device) * 0.5
    pd[:, 3] = 1.0
    pd[:, 4] = 1.0
    pd[:, 8:11] = 0.15
    pr = torch.randn(n, 12, device=device) * 0.3

    ray_ori = torch.zeros(1, h, w, 3, device=device)
    ray_ori[..., 2] = 2.0
    ray_dir = torch.zeros(1, h, w, 3, device=device)
    for y in range(h):
        for x in range(w):
            d = torch.tensor([(x - w / 2) / w, (y - h / 2) / h, -1.0])
            ray_dir[0, y, x] = d / d.norm()
    ray_time = torch.zeros(1, h, w, 1, dtype=torch.long, device=device)
    sp = plugin.fromOpenCVPinholeCameraModelParameters(
        [w, h],
        plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
        [w / 2.0, h / 2.0],
        [w / 2.0, w / 2.0],
        [0.0] * 6,
        [0.0, 0.0],
        [0.0] * 4,
    )
    pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)

    def render(ro, rd):
        r = plugin.SplatRaster(conf_dict)
        return r.trace(
            0, 1, pd, pr, ro.contiguous(), rd.contiguous(), ray_time,
            sp, 0, 0, pose, pose,
        )[0]

    def loss_of(ro, rd):
        return render(ro, rd)[..., :3].sum().item()

    # analytical ray-dir / ray-ori grads from one backward
    res = raster.trace(
        0, 1, pd, pr, ray_ori.contiguous(), ray_dir.contiguous(),
        ray_time, sp, 0, 0, pose, pose,
    )
    rd_density = res[0]
    g = torch.zeros_like(rd_density)
    g[..., :3] = 1.0
    hit = res[1]
    _, _, ori_grd, dir_grd = raster.trace_bwd(
        0, 1, pd, pr, ray_ori, ray_dir, ray_time, sp, 0, 0, pose, pose,
        rd_density, g, hit, torch.zeros_like(hit),
    )
    print(f"ray-dir grad max-abs: {dir_grd.abs().max().item():.4f}")
    print(f"ray-ori grad max-abs: {ori_grd.abs().max().item():.4f}")

    axis = torch.tensor([0.0, 1.0, 0.0], device=device)
    # d(ray_dir)/d(theta) at theta=0 = axis x ray_dir
    dray = torch.cross(
        axis.view(1, 1, 1, 3).expand_as(ray_dir), ray_dir, dim=-1
    )
    g_rot_analytical = (dir_grd * dray).sum().item()

    tvec = torch.tensor([1.0, 0.0, 0.0], device=device)
    g_tr_analytical = (ori_grd * tvec.view(1, 1, 1, 3)).sum().item()

    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-9)

    print("\n=== ROTATION dL/dtheta (about +y) ===")
    print(f"  analytical = {g_rot_analytical:+.2f}")
    rot_fds = []
    for eps in (1e-2, 3e-3, 1e-3):
        rp = _rot_axis(torch.tensor(eps, device=device), axis)
        rm = _rot_axis(torch.tensor(-eps, device=device), axis)
        rd_p = torch.nn.functional.normalize(ray_dir @ rp.T, dim=-1)
        rd_m = torch.nn.functional.normalize(ray_dir @ rm.T, dim=-1)
        fd = (loss_of(ray_ori, rd_p) - loss_of(ray_ori, rd_m)) / (2 * eps)
        rot_fds.append(fd)
        print(f"  fd(eps={eps:.0e}) = {fd:+.2f}  rel={rel(g_rot_analytical, fd):.3f}")

    print("=== TRANSLATION dL/dt (along +x) [CONTROL] ===")
    print(f"  analytical = {g_tr_analytical:+.2f}")
    tr_fds = []
    for eps in (1e-2, 3e-3, 1e-3):
        ro_p = ray_ori + eps * tvec.view(1, 1, 1, 3)
        ro_m = ray_ori - eps * tvec.view(1, 1, 1, 3)
        fd = (loss_of(ro_p, ray_dir) - loss_of(ro_m, ray_dir)) / (2 * eps)
        tr_fds.append(fd)
        print(f"  fd(eps={eps:.0e}) = {fd:+.2f}  rel={rel(g_tr_analytical, fd):.3f}")

    # FD oracle is trustworthy only if it is consistent across eps.
    def consistent(fds):
        return rel(fds[0], fds[-1]) < 0.2 and rel(fds[1], fds[-1]) < 0.2

    tr_consistent = consistent(tr_fds)
    rot_consistent = consistent(rot_fds)
    tr_ok = tr_consistent and rel(g_tr_analytical, tr_fds[-1]) < 0.15
    rot_ok = rot_consistent and rel(g_rot_analytical, rot_fds[-1]) < 0.15
    print(
        f"\nFD-oracle reliable? translation={tr_consistent} rotation={rot_consistent}"
    )
    print(f"VERDICT: translation_grad_ok={tr_ok}  rotation_grad_ok={rot_ok}")
    if not tr_ok:
        print("=> CONTROL FAILED: FD oracle still unreliable; result inconclusive.")
    elif rot_ok:
        print("=> rotation gradient is CORRECT (recovery failure was absorption).")
    else:
        print("=> rotation gradient is BROKEN -> CUDA sensor-pose grad is the fix.")


if __name__ == "__main__":
    main()
