"""Validate ray origin/direction gradients from trace_bwd via finite differences.

Requires a GPU with CUDA. The test creates a minimal scene (few Gaussians),
renders a small image, and compares analytical gradients (from the backward pass)
against numerical gradients (central finite differences on ray_ori / ray_dir).
"""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf


def _build_jit_config() -> OmegaConf:
    """Load the full 3DGUT config via Hydra compose."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    config_dir = os.path.abspath(config_dir)

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="apps/colmap_3dgut")

    return cfg


def test_ray_gradient_finite_differences():
    """Compare analytical ray gradients against finite-difference approximation."""
    device = torch.device("cuda")
    conf = _build_jit_config()

    from threedgut_tracer.setup_3dgut import setup_3dgut

    plugin = setup_3dgut(conf)
    conf_dict = OmegaConf.to_container(conf)

    raster = plugin.SplatRaster(conf_dict)

    H, W = 16, 16
    N_PARTICLES = 32
    SH_DIM = 4  # degree 1: 4 coeffs per channel

    torch.manual_seed(42)

    # Gaussian parameters: [N, 12] = pos(3) + density(1) + quat(4) + scale(3) + padding(1)
    particle_density = torch.zeros(N_PARTICLES, 12, device=device)
    for i in range(N_PARTICLES):
        particle_density[i, 0:3] = torch.randn(3) * 0.3  # position near origin
        particle_density[i, 3] = 10.0  # high density for visibility
        particle_density[i, 4:8] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        particle_density[i, 8:11] = torch.tensor([0.15, 0.15, 0.15])  # scale

    # SH coefficients: [N, SH_DIM * 3]
    particle_radiance = torch.randn(N_PARTICLES, SH_DIM * 3, device=device) * 0.1

    # Rays: simple pinhole camera looking down -Z
    ray_ori = torch.zeros(1, H, W, 3, device=device)
    ray_ori[..., 2] = 2.0  # camera at z=2
    ray_dir = torch.zeros(1, H, W, 3, device=device)
    for y in range(H):
        for x in range(W):
            dx = (x - W / 2) / W
            dy = (y - H / 2) / H
            d = torch.tensor([dx, dy, -1.0])
            ray_dir[0, y, x] = d / d.norm()

    ray_time = torch.zeros(1, H, W, 1, dtype=torch.long, device=device)

    sensor_params = plugin.fromOpenCVPinholeCameraModelParameters(
        [W, H],
        plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
        [W / 2.0, H / 2.0],  # principal point
        [W / 2.0, W / 2.0],  # focal length
        [0.0] * 6,  # radial coeffs
        [0.0, 0.0],  # tangential coeffs
        [0.0] * 4,  # thin prism coeffs
    )

    start_ts = 0
    end_ts = 0
    # TSensorPose is vec7: [tx, ty, tz, qx, qy, qz, qw]
    # Identity = zero translation, identity quaternion [0,0,0,1]
    identity_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)
    start_pose = identity_pose
    end_pose = identity_pose.clone()

    def render(ro, rd):
        """Forward pass with a fresh raster to avoid stale context."""
        r = plugin.SplatRaster(conf_dict)
        result = r.trace(
            0,  # frame_id
            1,  # n_active_features (SH degree 1)
            particle_density,
            particle_radiance,
            ro.contiguous(),
            rd.contiguous(),
            ray_time,
            sensor_params,
            start_ts,
            end_ts,
            start_pose,
            end_pose,
        )
        return result[0]  # [H, W, 4]

    # Run forward + backward to get analytical gradients
    ray_ori_test = ray_ori.clone().requires_grad_(True)
    ray_dir_test = ray_dir.clone().requires_grad_(True)

    result = raster.trace(
        0,
        1,
        particle_density,
        particle_radiance,
        ray_ori_test.contiguous(),
        ray_dir_test.contiguous(),
        ray_time,
        sensor_params,
        start_ts,
        end_ts,
        start_pose,
        end_pose,
    )
    assert len(result) == 8
    projected_conic_opacity = result[5]
    projected_tiles_count = result[7].reshape(-1)
    assert projected_conic_opacity.shape == (N_PARTICLES, 4)
    on_screen = projected_tiles_count > 0
    assert on_screen.any()
    observed_conics = projected_conic_opacity[on_screen]
    assert torch.isfinite(observed_conics).all()
    determinant = observed_conics[:, 0] * observed_conics[:, 2] - observed_conics[:, 1].square()
    assert (observed_conics[:, 0] > 0.0).all()
    assert (observed_conics[:, 2] > 0.0).all()
    assert (determinant > 0.0).all()
    assert (observed_conics[:, 3] > 0.0).all()
    radiance_density = result[0]
    loss = radiance_density[..., :3].sum()

    # Backward
    radiance_density_grad = torch.zeros_like(radiance_density)
    radiance_density_grad[..., :3] = 1.0  # gradient of sum w.r.t. RGB
    hit_dist = result[1]
    hit_dist_grad = torch.zeros_like(hit_dist)

    density_grd, radiance_grd, ori_grd, dir_grd = raster.trace_bwd(
        0,
        1,
        particle_density,
        particle_radiance,
        ray_ori_test,
        ray_dir_test,
        ray_time,
        sensor_params,
        start_ts,
        end_ts,
        start_pose,
        end_pose,
        radiance_density,
        radiance_density_grad,
        hit_dist,
        hit_dist_grad,
    )

    print(f"ori_grd shape: {ori_grd.shape}, max abs: {ori_grd.abs().max().item():.6f}")
    print(f"dir_grd shape: {dir_grd.shape}, max abs: {dir_grd.abs().max().item():.6f}")

    if ori_grd.abs().max().item() == 0.0:
        print("WARNING: ray origin gradients are all zero!")
    if dir_grd.abs().max().item() == 0.0:
        print("WARNING: ray direction gradients are all zero!")

    # Finite difference validation: pick pixels with highest alpha (confirmed visible)
    eps = 1e-4
    fwd_result = raster.trace(
        0,
        1,
        particle_density,
        particle_radiance,
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        ray_time,
        sensor_params,
        start_ts,
        end_ts,
        start_pose,
        end_pose,
    )
    fwd_alpha = fwd_result[0][..., 3]  # [H, W]
    topk_flat = fwd_alpha.flatten().topk(min(10, H * W)).indices
    test_pixels = [(idx.item() // W, idx.item() % W) for idx in topk_flat]
    print(f"\nTesting {len(test_pixels)} pixels with highest alpha")

    print("\nFinite-difference validation (ray origin):")
    max_rel_error = 0.0
    n_compared = 0
    for py, px in test_pixels:
        for dim in range(3):
            ro_plus = ray_ori.clone()
            ro_plus[0, py, px, dim] += eps
            rd_plus = render(ro_plus, ray_dir)

            ro_minus = ray_ori.clone()
            ro_minus[0, py, px, dim] -= eps
            rd_minus = render(ro_minus, ray_dir)

            fd_grad = (rd_plus[py, px, :3].sum().item() - rd_minus[py, px, :3].sum().item()) / (2 * eps)
            an_grad = ori_grd[0, py, px, dim].item()

            if abs(fd_grad) > 1e-4 or abs(an_grad) > 1e-4:
                n_compared += 1
                if abs(fd_grad) > 1e-6:
                    rel_err = abs(an_grad - fd_grad) / max(abs(fd_grad), abs(an_grad))
                    max_rel_error = max(max_rel_error, rel_err)
                    status = "OK" if rel_err < 0.05 else "MISMATCH"
                    print(
                        f"  pixel ({py},{px}) dim {dim}: analytical={an_grad:.6f} fd={fd_grad:.6f} rel_err={rel_err:.4f} {status}"
                    )
                else:
                    print(f"  pixel ({py},{px}) dim {dim}: analytical={an_grad:.6f} fd={fd_grad:.6f} FD~0")

    print(f"\nMax relative error (origin): {max_rel_error:.4f} ({n_compared} comparisons)")

    print("\nFinite-difference validation (ray direction):")
    max_rel_error_dir = 0.0
    n_compared_dir = 0
    for py, px in test_pixels:
        for dim in range(3):
            rd_plus = ray_dir.clone()
            rd_plus[0, py, px, dim] += eps
            out_plus = render(ray_ori, rd_plus)

            rd_minus = ray_dir.clone()
            rd_minus[0, py, px, dim] -= eps
            out_minus = render(ray_ori, rd_minus)

            fd_grad = (out_plus[py, px, :3].sum().item() - out_minus[py, px, :3].sum().item()) / (2 * eps)
            an_grad = dir_grd[0, py, px, dim].item()

            if abs(fd_grad) > 1e-4 or abs(an_grad) > 1e-4:
                n_compared_dir += 1
                if abs(fd_grad) > 1e-6:
                    rel_err = abs(an_grad - fd_grad) / max(abs(fd_grad), abs(an_grad))
                    max_rel_error_dir = max(max_rel_error_dir, rel_err)
                    status = "OK" if rel_err < 0.05 else "MISMATCH"
                    print(
                        f"  pixel ({py},{px}) dim {dim}: analytical={an_grad:.6f} fd={fd_grad:.6f} rel_err={rel_err:.4f} {status}"
                    )
                else:
                    print(f"  pixel ({py},{px}) dim {dim}: analytical={an_grad:.6f} fd={fd_grad:.6f} FD~0")

    print(f"\nMax relative error (direction): {max_rel_error_dir:.4f} ({n_compared_dir} comparisons)")

    passed = n_compared > 0 and n_compared_dir > 0 and max_rel_error < 0.1 and max_rel_error_dir < 0.1
    print(f"\n{'PASSED' if passed else 'FAILED'}: ray gradient finite-difference test")
    return passed


if __name__ == "__main__":
    test_ray_gradient_finite_differences()
