"""Finite-difference validation of Gabor/SIREN carrier parameter gradients.

Builds the 3DGUT raster with a carrier enabled, renders a small synthetic
scene, and compares the analytical radiance-buffer gradients returned by
``trace_bwd`` against central finite differences on individual carrier
slots (the parameters packed after the SH coefficients). This exercises
the slang autodiff of the carrier head end to end through the compiled
kernels; the ray-gradient FD test covers the SH/pose paths.
"""

from __future__ import annotations

import os

import pytest
import torch
from omegaconf import OmegaConf


def _compose(overrides: list[str]) -> OmegaConf:
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="apps/colmap_3dgut", overrides=overrides)


def _radiance_stride(conf) -> int:
    from threedgrut.model.carriers import carrier_specular_dim

    sh_coeffs = (conf.render.particle_radiance_sph_degree + 1) ** 2
    return sh_coeffs + carrier_specular_dim(conf) // 3


def _sh_coeffs(conf) -> int:
    return (conf.render.particle_radiance_sph_degree + 1) ** 2


def _run_fd(carrier_overrides: list[str]) -> None:
    device = torch.device("cuda")
    conf = _compose(carrier_overrides)

    from threedgut_tracer.setup_3dgut import setup_3dgut

    plugin = setup_3dgut(conf)
    conf_dict = OmegaConf.to_container(conf)

    stride = _radiance_stride(conf)
    sh_coeffs = _sh_coeffs(conf)
    n_active = conf.render.particle_radiance_sph_degree

    H, W = 16, 16
    N_PARTICLES = 8
    torch.manual_seed(7)

    particle_density = torch.zeros(N_PARTICLES, 12, device=device)
    for i in range(N_PARTICLES):
        particle_density[i, 0:3] = torch.randn(3) * 0.3
        particle_density[i, 3] = 10.0
        particle_density[i, 4:8] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        particle_density[i, 8:11] = torch.tensor([0.2, 0.2, 0.2])

    particle_radiance = torch.randn(N_PARTICLES, stride * 3, device=device) * 0.1

    ray_ori = torch.zeros(1, H, W, 3, device=device)
    ray_ori[..., 2] = 2.0
    ray_dir = torch.zeros(1, H, W, 3, device=device)
    for y in range(H):
        for x in range(W):
            d = torch.tensor([(x - W / 2) / W, (y - H / 2) / H, -1.0])
            ray_dir[0, y, x] = d / d.norm()
    ray_time = torch.zeros(1, H, W, 1, dtype=torch.long, device=device)

    sensor_params = plugin.fromOpenCVPinholeCameraModelParameters(
        [W, H],
        plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
        [W / 2.0, H / 2.0],
        [W / 2.0, W / 2.0],
        [0.0] * 6,
        [0.0, 0.0],
        [0.0] * 4,
    )
    identity_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)

    def render(radiance):
        r = plugin.SplatRaster(conf_dict)
        result = r.trace(
            0, n_active, particle_density, radiance,
            ray_ori.contiguous(), ray_dir.contiguous(), ray_time,
            sensor_params, 0, 0, identity_pose, identity_pose,
        )
        return result[0]

    raster = plugin.SplatRaster(conf_dict)
    result = raster.trace(
        0, n_active, particle_density, particle_radiance,
        ray_ori.contiguous(), ray_dir.contiguous(), ray_time,
        sensor_params, 0, 0, identity_pose, identity_pose,
    )
    radiance_density = result[0]
    radiance_density_grad = torch.zeros_like(radiance_density)
    radiance_density_grad[..., :3] = 1.0
    hit_dist = result[1]
    hit_dist_grad = torch.zeros_like(hit_dist)

    _, radiance_grd, _, _ = raster.trace_bwd(
        0, n_active, particle_density, particle_radiance,
        ray_ori, ray_dir, ray_time,
        sensor_params, 0, 0, identity_pose, identity_pose,
        radiance_density, radiance_density_grad,
        hit_dist, hit_dist_grad,
    )
    radiance_grd = radiance_grd.reshape(N_PARTICLES, stride * 3)

    # Sample carrier slots (packed after the SH coefficients) across particles.
    eps = 1e-3
    carrier_float_lo = sh_coeffs * 3
    carrier_float_hi = stride * 3
    checked = 0
    max_rel_err = 0.0
    for particle in range(N_PARTICLES):
        for flat in range(carrier_float_lo, carrier_float_hi, 5):
            an = radiance_grd[particle, flat].item()
            plus = particle_radiance.clone()
            plus[particle, flat] += eps
            minus = particle_radiance.clone()
            minus[particle, flat] -= eps
            fd = (
                render(plus)[..., :3].sum().item()
                - render(minus)[..., :3].sum().item()
            ) / (2 * eps)
            if abs(fd) < 1e-4 and abs(an) < 1e-4:
                continue
            checked += 1
            rel = abs(an - fd) / max(abs(an), abs(fd))
            max_rel_err = max(max_rel_err, rel)

    assert checked > 0, "no carrier slot produced a nonzero gradient"
    assert max_rel_err < 0.1, (
        f"carrier slot gradients diverge from finite differences: "
        f"max rel err {max_rel_err:.4f} over {checked} slots"
    )


_MODES = {
    "gabor": ["model.use_gabor_carrier=true"],
    "siren": ["model.use_siren_carrier=true", "model.siren_output_init_scale=0.2"],
}


@pytest.mark.parametrize("mode", list(_MODES), ids=list(_MODES))
def test_carrier_slot_gradients_match_finite_differences(mode):
    # Each carrier mode compiles its own kernel defines; the pybind plugin
    # cannot re-register in one process, so isolate per mode.
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, __file__, mode],
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), ".."),
    )
    assert proc.returncode == 0, f"{mode} FD check failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"


if __name__ == "__main__":
    import sys

    _run_fd(_MODES[sys.argv[1]])
    print(f"{sys.argv[1]} carrier FD check passed")
