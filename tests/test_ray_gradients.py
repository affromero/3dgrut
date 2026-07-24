"""Validate ray and camera gradients from the native 3DGUT tracer."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from threedgut_tracer.setup_3dgut import setup_3dgut
from threedgut_tracer.tracer import Tracer


_K_BUFFER_SIZE_ENV = "THREEDGRUT_TEST_K_BUFFER_SIZE"


def _build_jit_config(k_buffer_size: int = 0) -> OmegaConf:
    """Load the full 3DGUT config with the selected K-buffer size."""
    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs")
    )
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="apps/colmap_3dgut")
    config.render.splat.k_buffer_size = k_buffer_size
    return config


def _make_sensor(
    plugin: object,
    *,
    width: int,
    height: int,
    shutter: object,
) -> object:
    """Create an undistorted pinhole sensor for a small synthetic render."""
    return plugin.fromOpenCVPinholeCameraModelParameters(
        [width, height],
        shutter,
        [width / 2.0, height / 2.0],
        [width / 2.0, width / 2.0],
        [0.0] * 6,
        [0.0, 0.0],
        [0.0] * 4,
    )


def _identity_pose(device: torch.device) -> torch.Tensor:
    """Return the identity TSensorPose in the native vec7 layout."""
    return torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        device=device,
    )


def test_ray_gradient_finite_differences() -> None:
    """Match direct and autograd ray/camera gradients to finite differences."""
    device = torch.device("cuda")
    k_buffer_size = int(os.environ.get(_K_BUFFER_SIZE_ENV, "0"))
    config = _build_jit_config(k_buffer_size)
    plugin = setup_3dgut(config)
    config_dict = OmegaConf.to_container(config)
    raster = plugin.SplatRaster(config_dict)

    height = 16
    width = 16
    num_particles = 32
    sh_dim = 16
    torch.manual_seed(42)

    particle_density = torch.zeros(num_particles, 12, device=device)
    for particle_index in range(num_particles):
        particle_density[particle_index, 0:3] = torch.randn(3) * 0.3
        particle_density[particle_index, 3] = 0.5
        particle_density[particle_index, 4:8] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=device,
        )
        particle_density[particle_index, 8:11] = torch.tensor(
            [0.15, 0.15, 0.15],
            device=device,
        )
    particle_radiance = torch.randn(
        num_particles,
        sh_dim * 3,
        device=device,
    ) * 0.1

    ray_origin = torch.zeros(1, height, width, 3, device=device)
    ray_origin[..., 2] = 2.0
    ray_direction = torch.zeros(1, height, width, 3, device=device)
    for row in range(height):
        for column in range(width):
            direction = torch.tensor(
                [
                    (column - width / 2) / width,
                    (row - height / 2) / height,
                    -1.0,
                ],
                device=device,
            )
            ray_direction[0, row, column] = direction / direction.norm()
    ray_time = torch.zeros(
        1,
        height,
        width,
        1,
        dtype=torch.long,
        device=device,
    )
    sensor = _make_sensor(
        plugin,
        width=width,
        height=height,
        shutter=plugin.ShutterType.ROLLING_TOP_TO_BOTTOM,
    )
    start_pose = _identity_pose(device)
    end_pose = start_pose.clone()

    def render(
        render_ray_origin: torch.Tensor,
        render_ray_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Render with a fresh native raster to avoid stale contexts."""
        render_raster = plugin.SplatRaster(config_dict)
        result = render_raster.trace(
            0,
            1,
            particle_density,
            particle_radiance,
            render_ray_origin.contiguous(),
            render_ray_direction.contiguous(),
            ray_time,
            sensor,
            0,
            0,
            start_pose,
            end_pose,
        )
        return result[0]

    direct_ray_origin = ray_origin.clone().requires_grad_(True)
    direct_ray_direction = ray_direction.clone().requires_grad_(True)
    result = raster.trace(
        0,
        1,
        particle_density,
        particle_radiance,
        direct_ray_origin.contiguous(),
        direct_ray_direction.contiguous(),
        ray_time,
        sensor,
        0,
        0,
        start_pose,
        end_pose,
    )
    assert len(result) == 9
    projected_conic_opacity = result[5]
    projected_tile_count = result[7].reshape(-1)
    accumulated_weight = result[8]
    assert projected_conic_opacity.shape == (num_particles, 4)
    assert projected_tile_count.shape == (num_particles,)
    assert projected_tile_count.gt(0).any()
    on_screen_conics = projected_conic_opacity[projected_tile_count.gt(0)]
    determinant = (
        on_screen_conics[:, 0] * on_screen_conics[:, 2]
        - on_screen_conics[:, 1].square()
    )
    assert torch.isfinite(on_screen_conics).all()
    assert on_screen_conics[:, 0].gt(0.0).all()
    assert on_screen_conics[:, 2].gt(0.0).all()
    assert determinant.gt(0.0).all()
    assert on_screen_conics[:, 3].gt(0.0).all()
    assert accumulated_weight.shape == (num_particles, 1)
    assert accumulated_weight.ge(0.0).all()
    assert accumulated_weight.sum().gt(0.0)

    radiance_density = result[0]
    radiance_density_gradient = torch.zeros_like(radiance_density)
    radiance_density_gradient[..., :3] = 1.0
    hit_distance = result[1]
    hit_distance_gradient = torch.zeros_like(hit_distance)
    (
        density_gradient,
        radiance_gradient,
        origin_gradient,
        direction_gradient,
        camera_center_gradient,
        projected_position_gradient,
    ) = raster.trace_bwd_with_abs(
        0,
        1,
        particle_density,
        particle_radiance,
        direct_ray_origin,
        direct_ray_direction,
        ray_time,
        sensor,
        0,
        0,
        start_pose,
        end_pose,
        radiance_density,
        radiance_density_gradient,
        hit_distance,
        hit_distance_gradient,
        torch.empty((0, 3), device=device),
    )
    assert radiance_gradient.shape == particle_radiance.shape
    assert projected_position_gradient.shape == (num_particles, 2)
    assert torch.isfinite(projected_position_gradient).all()

    diagnostic_result = raster.trace(
        0,
        1,
        particle_density,
        particle_radiance,
        direct_ray_origin.contiguous(),
        direct_ray_direction.contiguous(),
        ray_time,
        sensor,
        0,
        0,
        start_pose,
        end_pose,
    )
    absolute_position_gradient = torch.zeros(
        (num_particles, 3),
        device=device,
    )
    (
        absolute_density_gradient,
        _,
        _,
        _,
        _,
        absolute_projected_position_gradient,
    ) = raster.trace_bwd_with_abs(
        0,
        1,
        particle_density,
        particle_radiance,
        direct_ray_origin,
        direct_ray_direction,
        ray_time,
        sensor,
        0,
        0,
        start_pose,
        end_pose,
        diagnostic_result[0],
        radiance_density_gradient,
        diagnostic_result[1],
        hit_distance_gradient,
        absolute_position_gradient,
    )
    torch.testing.assert_close(absolute_density_gradient, density_gradient)
    torch.testing.assert_close(
        absolute_projected_position_gradient,
        projected_position_gradient,
    )
    if k_buffer_size == 0:
        assert absolute_position_gradient.gt(0.0).any()
        assert (
            absolute_position_gradient + 1.0e-5
            >= density_gradient[:, :3].abs()
        ).all()

    autograd_ray_origin = ray_origin.clone().requires_grad_(True)
    autograd_ray_direction = ray_direction.clone().requires_grad_(True)
    autograd_camera_center = torch.zeros(
        3,
        device=device,
        requires_grad=True,
    )
    sensor_poses = SimpleNamespace(
        timestamps_us=[0, 0],
        T_world_sensors=[start_pose, end_pose],
    )
    signed_position_gradient = torch.empty((0, 3), device=device)
    absolute_position_diagnostic = torch.empty((0, 3), device=device)
    autograd_result = Tracer._Autograd.apply(
        raster,
        0,
        1,
        autograd_ray_origin,
        autograd_ray_direction,
        autograd_camera_center,
        particle_density[:, 0:3],
        particle_density[:, 4:8],
        particle_density[:, 8:11],
        particle_density[:, 3:4],
        particle_radiance,
        signed_position_gradient,
        absolute_position_diagnostic,
        sensor,
        sensor_poses,
    )
    autograd_result[0][..., :3].sum().backward()
    torch.testing.assert_close(
        autograd_result[9],
        projected_position_gradient,
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        autograd_ray_origin.grad,
        origin_gradient,
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        autograd_ray_direction.grad,
        direction_gradient,
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    torch.testing.assert_close(
        autograd_camera_center.grad,
        camera_center_gradient,
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    assert torch.isfinite(density_gradient).all()
    assert torch.isfinite(origin_gradient).all()
    assert torch.isfinite(direction_gradient).all()
    assert torch.isfinite(camera_center_gradient).all()
    assert density_gradient.abs().max().gt(0.0)
    assert origin_gradient.abs().max().gt(0.0)
    assert direction_gradient.abs().max().gt(0.0)
    assert camera_center_gradient.abs().max().gt(0.0)

    if k_buffer_size > 0:
        return

    forward_result = raster.trace(
        0,
        1,
        particle_density,
        particle_radiance,
        ray_origin.contiguous(),
        ray_direction.contiguous(),
        ray_time,
        sensor,
        0,
        0,
        start_pose,
        end_pose,
    )
    top_pixels = forward_result[0][..., 3].flatten().topk(10).indices
    epsilon = 1.0e-4
    origin_errors: list[float] = []
    direction_errors: list[float] = []
    for flat_index in top_pixels:
        row = int(flat_index.item()) // width
        column = int(flat_index.item()) % width
        for component in range(3):
            origin_plus = ray_origin.clone()
            origin_minus = ray_origin.clone()
            origin_plus[0, row, column, component] += epsilon
            origin_minus[0, row, column, component] -= epsilon
            origin_finite_difference = (
                render(origin_plus, ray_direction)[row, column, :3].sum()
                - render(origin_minus, ray_direction)[row, column, :3].sum()
            ) / (2.0 * epsilon)
            analytical_origin = origin_gradient[0, row, column, component]
            if (
                origin_finite_difference.abs().item() > 1.0e-6
                and analytical_origin.abs().item() > 1.0e-6
            ):
                origin_errors.append(
                    float(
                        (analytical_origin - origin_finite_difference)
                        .abs()
                        .div(
                            torch.maximum(
                                analytical_origin.abs(),
                                origin_finite_difference.abs(),
                            )
                        )
                        .item()
                    )
                )

            direction_plus = ray_direction.clone()
            direction_minus = ray_direction.clone()
            direction_plus[0, row, column, component] += epsilon
            direction_minus[0, row, column, component] -= epsilon
            direction_finite_difference = (
                render(ray_origin, direction_plus)[row, column, :3].sum()
                - render(ray_origin, direction_minus)[row, column, :3].sum()
            ) / (2.0 * epsilon)
            analytical_direction = direction_gradient[0, row, column, component]
            if (
                direction_finite_difference.abs().item() > 1.0e-6
                and analytical_direction.abs().item() > 1.0e-6
            ):
                direction_errors.append(
                    float(
                        (analytical_direction - direction_finite_difference)
                        .abs()
                        .div(
                            torch.maximum(
                                analytical_direction.abs(),
                                direction_finite_difference.abs(),
                            )
                        )
                        .item()
                    )
                )
    assert origin_errors
    assert direction_errors
    assert sum(error < 0.1 for error in origin_errors) / len(
        origin_errors
    ) >= 0.8
    assert sum(error < 0.1 for error in direction_errors) / len(
        direction_errors
    ) >= 0.8


def test_precomputed_sh_camera_center_gradients_match_finite_differences() -> None:
    """Validate camera-center gradients for all precomputed SH degrees."""
    device = torch.device("cuda")
    config = _build_jit_config()
    plugin = setup_3dgut(config)
    config_dict = OmegaConf.to_container(config)
    particle_density = torch.zeros(1, 12, device=device)
    particle_density[0, 0:3] = torch.tensor(
        [0.2, 0.1, 1.0],
        device=device,
    )
    particle_density[0, 3] = 10.0
    particle_density[0, 4] = 1.0
    particle_density[0, 8:11] = 0.4
    particle_radiance = torch.zeros(1, 48, device=device)
    particle_radiance[0, 0:3] = 2.0
    generator = torch.Generator().manual_seed(17)
    particle_radiance[0, 3:] = (
        torch.randn(45, generator=generator).to(device) * 0.15
    )
    base_ray_origin = torch.tensor(
        [[[[0.2, 0.1, 3.0]]]],
        device=device,
    )
    ray_direction = torch.tensor(
        [[[[0.0, 0.0, -1.0]]]],
        device=device,
    )
    ray_time = torch.zeros(
        1,
        1,
        1,
        1,
        dtype=torch.long,
        device=device,
    )
    sensor = _make_sensor(
        plugin,
        width=1,
        height=1,
        shutter=plugin.ShutterType.GLOBAL,
    )

    def sensor_pose(camera_center: torch.Tensor) -> torch.Tensor:
        quaternion = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=device,
        )
        return torch.cat((-camera_center, quaternion)).unsqueeze(0)

    def trace_at_camera_center(
        camera_center: torch.Tensor,
        active_degree: int,
        radiance: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        pose = sensor_pose(camera_center)
        local_ray_origin = base_ray_origin - camera_center.view(1, 1, 1, 3)
        raster = plugin.SplatRaster(config_dict)
        return raster.trace(
            0,
            active_degree,
            particle_density,
            radiance,
            local_ray_origin.contiguous(),
            ray_direction,
            ray_time,
            sensor,
            0,
            0,
            pose,
            pose,
        )

    def assert_camera_center_gradient(
        active_degree: int,
        radiance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        camera_center = torch.zeros(3, device=device)
        pose = sensor_pose(camera_center)
        raster = plugin.SplatRaster(config_dict)
        result = raster.trace(
            0,
            active_degree,
            particle_density,
            radiance,
            base_ray_origin,
            ray_direction,
            ray_time,
            sensor,
            0,
            0,
            pose,
            pose,
        )
        output_gradient = torch.zeros_like(result[0])
        output_gradient[..., :3] = 1.0
        backward_result = raster.trace_bwd_with_abs(
            0,
            active_degree,
            particle_density,
            radiance,
            base_ray_origin,
            ray_direction,
            ray_time,
            sensor,
            0,
            0,
            pose,
            pose,
            result[0],
            output_gradient,
            result[1],
            torch.zeros_like(result[1]),
            torch.empty((0, 3), device=device),
        )
        density_gradient = backward_result[0]
        camera_center_gradient = backward_result[4]
        projected_position_gradient = backward_result[5]
        assert projected_position_gradient.shape == (1, 2)
        assert torch.isfinite(projected_position_gradient).all()

        epsilon = 1.0e-3
        finite_difference = torch.empty_like(camera_center_gradient)
        for dimension in range(3):
            positive_center = camera_center.clone()
            negative_center = camera_center.clone()
            positive_center[dimension] += epsilon
            negative_center[dimension] -= epsilon
            positive_rgb = trace_at_camera_center(
                positive_center,
                active_degree,
                radiance,
            )[0][..., :3].sum()
            negative_rgb = trace_at_camera_center(
                negative_center,
                active_degree,
                radiance,
            )[0][..., :3].sum()
            finite_difference[dimension] = (positive_rgb - negative_rgb) / (
                2.0 * epsilon
            )
        torch.testing.assert_close(
            camera_center_gradient,
            finite_difference,
            rtol=5.0e-3,
            atol=2.0e-4,
        )
        torch.testing.assert_close(
            camera_center_gradient,
            -density_gradient[0, :3],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        return result[0][..., :3], camera_center_gradient

    for active_degree in range(4):
        rgb, camera_center_gradient = assert_camera_center_gradient(
            active_degree,
            particle_radiance,
        )
        assert torch.isfinite(rgb).all()
        if active_degree == 0:
            assert torch.count_nonzero(camera_center_gradient) == 0
        else:
            assert camera_center_gradient.abs().max().gt(0.0)

    clamped_radiance = particle_radiance.clone()
    clamped_radiance[0, 0:3] = torch.tensor(
        [-4.0, 2.0, 2.0],
        device=device,
    )
    clamped_radiance[0, 3::3] = 0.0
    rgb, camera_center_gradient = assert_camera_center_gradient(
        3,
        clamped_radiance,
    )
    assert rgb[..., 0].item() == 0.0
    assert camera_center_gradient.abs().max().gt(0.0)


def test_trace_rejects_noncontiguous_native_tensor() -> None:
    """Reject temporary-copy pointer hazards before launching CUDA kernels."""
    device = torch.device("cuda")
    config = _build_jit_config()
    plugin = setup_3dgut(config)
    raster = plugin.SplatRaster(OmegaConf.to_container(config))
    particle_density = torch.zeros((1, 12), device=device)
    particle_density[:, 3] = 10.0
    particle_density[:, 4] = 1.0
    particle_density[:, 8:11] = 0.1
    particle_radiance = torch.zeros((1, 48), device=device)
    ray_origin = torch.zeros((1, 2, 2, 3), device=device)
    ray_direction = torch.ones((1, 2, 2, 3), device=device).transpose(1, 2)
    assert not ray_direction.is_contiguous()
    ray_time = torch.zeros(
        (1, 2, 2, 1),
        dtype=torch.long,
        device=device,
    )
    sensor = _make_sensor(
        plugin,
        width=2,
        height=2,
        shutter=plugin.ShutterType.GLOBAL,
    )
    pose = _identity_pose(device)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        raster.trace(
            0,
            0,
            particle_density,
            particle_radiance,
            ray_origin,
            ray_direction,
            ray_time,
            sensor,
            0,
            0,
            pose,
            pose,
        )


def test_trace_rejects_short_compiled_radiance_buffer() -> None:
    """Reject buffers whose row stride is shorter than the compiled SH layout."""
    device = torch.device("cuda")
    config = _build_jit_config()
    plugin = setup_3dgut(config)
    raster = plugin.SplatRaster(OmegaConf.to_container(config))
    particle_density = torch.zeros((1, 12), device=device)
    particle_density[:, 3] = 10.0
    particle_density[:, 4] = 1.0
    particle_density[:, 8:11] = 0.1
    particle_radiance = torch.zeros((1, 3), device=device)
    ray_origin = torch.zeros((1, 2, 2, 3), device=device)
    ray_direction = torch.ones((1, 2, 2, 3), device=device)
    ray_time = torch.zeros(
        (1, 2, 2, 1),
        dtype=torch.long,
        device=device,
    )
    sensor = _make_sensor(
        plugin,
        width=2,
        height=2,
        shutter=plugin.ShutterType.GLOBAL,
    )
    pose = _identity_pose(device)
    with pytest.raises(RuntimeError, match="compiled radiance width 48"):
        raster.trace(
            0,
            0,
            particle_density,
            particle_radiance,
            ray_origin,
            ray_direction,
            ray_time,
            sensor,
            0,
            0,
            pose,
            pose,
        )


def test_progressive_sh_padding_preserves_active_gradients() -> None:
    """Inactive compiled SH bands are zero without severing active gradients."""
    active = torch.randn((4, 12), requires_grad=True)
    padded = Tracer._pad_particle_radiance(active, compiled_width=48)
    assert padded.shape == (4, 48)
    torch.testing.assert_close(padded[:, :12], active)
    torch.testing.assert_close(padded[:, 12:], torch.zeros((4, 36)))
    padded.sum().backward()
    torch.testing.assert_close(active.grad, torch.ones_like(active))


def test_responsibility_matches_composited_opacity() -> None:
    """The native diagnostic accumulator must use the renderer T*alpha."""
    device = torch.device("cuda")
    conf = _build_jit_config()
    plugin = setup_3dgut(conf)
    raster = plugin.SplatRaster(OmegaConf.to_container(conf))

    height, width = 16, 16
    particle_density = torch.zeros(32, 12, device=device)
    torch.manual_seed(42)
    for index in range(len(particle_density)):
        particle_density[index, :3] = torch.randn(3, device=device) * 0.3
        particle_density[index, 3] = 10.0
        particle_density[index, 4] = 1.0
        particle_density[index, 8:11] = 0.15
    particle_radiance = torch.zeros(32, 48, device=device)
    ray_ori = torch.zeros(1, height, width, 3, device=device)
    ray_ori[..., 2] = 2.0
    ray_dir = torch.zeros_like(ray_ori)
    for y in range(height):
        for x in range(width):
            direction = torch.tensor(
                [(x - width / 2) / width, (y - height / 2) / height, -1.0],
                device=device,
            )
            ray_dir[0, y, x] = direction / direction.norm()
    ray_timestamp = torch.zeros(
        1,
        height,
        width,
        1,
        dtype=torch.long,
        device=device,
    )
    sensor = plugin.fromOpenCVPinholeCameraModelParameters(
        [width, height],
        plugin.ShutterType.GLOBAL,
        [width / 2.0, height / 2.0],
        [width / 2.0, width / 2.0],
        [0.0] * 6,
        [0.0, 0.0],
        [0.0] * 4,
    )
    pose = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        device=device,
    )
    result = raster.trace_with_responsibility(
        0,
        1,
        particle_density,
        particle_radiance,
        ray_ori,
        ray_dir,
        ray_timestamp,
        sensor,
        0,
        0,
        pose,
        pose,
        torch.ones(height, width, device=device),
    )

    composited_opacity = result[0][..., 3].sum()
    responsibility = result[8].sum()
    diagnostic_responsibility = result[9].sum()
    weighted = result[10].sum()
    assert torch.isfinite(responsibility)
    assert responsibility > 0.0
    assert torch.allclose(responsibility, composited_opacity, atol=1e-5)
    assert torch.allclose(diagnostic_responsibility, responsibility, atol=1e-5)
    assert torch.allclose(weighted, responsibility, atol=1e-5)


def test_k_buffer_ray_gradients() -> None:
    """Run the compile-time K-buffer variant in an isolated process."""
    environment = os.environ.copy()
    environment[_K_BUFFER_SIZE_ENV] = "16"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"{__file__}::test_ray_gradient_finite_differences",
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    test_ray_gradient_finite_differences()
