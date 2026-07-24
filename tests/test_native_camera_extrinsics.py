"""Tests for the native absolute COLMAP camera path."""

import pytest
import torch
from threedgrut.datasets.protocols import Batch
from threedgrut.model.native_camera_extrinsics import (
    NativeCameraExtrinsics,
    colmap_qvec_to_w2c_rotation,
    colmap_w2c_to_c2w,
    colmap_w2c_to_world_rays,
)


def _unit_qvec() -> torch.Tensor:
    qvec = torch.tensor((0.7, -0.2, 0.4, 0.5), dtype=torch.float64)
    return qvec / torch.linalg.vector_norm(qvec)


def _module(qvec: torch.Tensor, tvec: torch.Tensor) -> NativeCameraExtrinsics:
    return NativeCameraExtrinsics(
        initial_qvecs=qvec.to(torch.float32).reshape(1, 4),
        initial_tvecs=tvec.to(torch.float32).reshape(1, 3),
    )


def test_absolute_colmap_pose_matches_existing_c2w_ray_transform() -> None:
    """Step-zero baked rays equal the existing C2W rendering contract."""
    qvec = _unit_qvec()
    tvec = torch.tensor((1.2, -0.4, 2.5), dtype=torch.float64)
    rays_ori = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]]]],
        dtype=torch.float64,
    )
    rays_dir = torch.tensor(
        [[[[0.0, 0.0, 1.0], [0.3, -0.1, 0.8]]]],
        dtype=torch.float64,
    )
    rotation = colmap_qvec_to_w2c_rotation(qvec)
    world_to_camera = torch.eye(4, dtype=torch.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = tvec
    camera_to_world = torch.linalg.inv(world_to_camera)
    expected_origins = (
        rays_ori @ camera_to_world[:3, :3].transpose(0, 1)
        + camera_to_world[:3, 3]
    )
    expected_directions = rays_dir @ camera_to_world[:3, :3].transpose(0, 1)

    world_origins, world_directions = colmap_w2c_to_world_rays(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        qvec=qvec,
        tvec=tvec,
    )

    torch.testing.assert_close(world_origins, expected_origins)
    torch.testing.assert_close(world_directions, expected_directions)


def test_native_module_preserves_camera_rays_and_absolute_c2w() -> None:
    """The projector receives the pose while ray values stay camera-space."""
    qvec = _unit_qvec().to(torch.float32)
    tvec = torch.tensor((1.2, -0.4, 2.5), dtype=torch.float32)
    rotation = colmap_qvec_to_w2c_rotation(qvec)
    world_to_camera = torch.eye(4)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = tvec
    camera_to_world = torch.linalg.inv(world_to_camera).unsqueeze(0)
    rays_ori = torch.tensor([[[[0.1, -0.2, 0.3]]]])
    rays_dir = torch.tensor([[[[0.3, -0.1, 0.8]]]])
    batch = Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=camera_to_world,
        source_frame_idx=0,
    )
    corrected = _module(qvec, tvec)(batch)

    assert not corrected.rays_in_world_space
    torch.testing.assert_close(corrected.T_to_world, camera_to_world)
    torch.testing.assert_close(corrected.rays_ori, rays_ori, rtol=0.0, atol=0.0)
    torch.testing.assert_close(corrected.rays_dir, rays_dir, rtol=0.0, atol=0.0)


def test_camera_ray_proxy_matches_world_ray_pose_vjp() -> None:
    """Sensor-basis tracer gradients recover the world-ray pose VJP."""
    qvec = _unit_qvec().to(torch.float32)
    tvec = torch.tensor((1.2, -0.4, 2.5), dtype=torch.float32)
    module = _module(qvec, tvec)
    rays_ori = torch.tensor([[[[0.1, -0.2, 0.3]]]])
    rays_dir = torch.tensor([[[[0.3, -0.1, 0.8]]]])
    batch = Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=torch.eye(4).unsqueeze(0),
        source_frame_idx=0,
    )
    sensor_origin_gradient = torch.tensor([[[[0.7, -0.2, 0.4]]]])
    sensor_direction_gradient = torch.tensor([[[[0.1, 0.6, -0.5]]]])

    corrected = module(batch)
    proxy_objective = (
        corrected.rays_ori * sensor_origin_gradient
    ).sum() + (corrected.rays_dir * sensor_direction_gradient).sum()
    proxy_gradients = torch.autograd.grad(
        proxy_objective,
        (module.qvecs, module.tvecs),
    )

    rotation = colmap_qvec_to_w2c_rotation(module.qvecs[0]).detach()
    world_origins, world_directions = colmap_w2c_to_world_rays(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        qvec=module.qvecs[0],
        tvec=module.tvecs[0],
    )
    world_objective = (
        world_origins * (sensor_origin_gradient @ rotation)
    ).sum() + (
        world_directions * (sensor_direction_gradient @ rotation)
    ).sum()
    world_gradients = torch.autograd.grad(
        world_objective,
        (module.qvecs, module.tvecs),
    )

    torch.testing.assert_close(proxy_gradients, world_gradients)


def test_colmap_c2w_helper_matches_matrix_inverse() -> None:
    """Differentiable C2W construction matches the dataset convention."""
    qvec = _unit_qvec()
    tvec = torch.tensor((1.2, -0.4, 2.5), dtype=torch.float64)
    world_to_camera = torch.eye(4, dtype=torch.float64)
    world_to_camera[:3, :3] = colmap_qvec_to_w2c_rotation(qvec)
    world_to_camera[:3, 3] = tvec

    actual = colmap_w2c_to_c2w(qvec=qvec, tvec=tvec)

    torch.testing.assert_close(actual, torch.linalg.inv(world_to_camera))


def test_colmap_quaternion_uses_scalar_first_w2c_convention() -> None:
    """A positive ninety-degree Z qvec matches COLMAP's W2C matrix."""
    half_angle = torch.tensor(torch.pi / 4.0, dtype=torch.float64)
    qvec = torch.tensor(
        (
            torch.cos(half_angle),
            0.0,
            0.0,
            torch.sin(half_angle),
        ),
        dtype=torch.float64,
    )
    expected = torch.tensor(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        colmap_qvec_to_w2c_rotation(qvec),
        expected,
        atol=1.0e-15,
        rtol=0.0,
    )


def test_world_ray_pose_gradients_match_finite_differences() -> None:
    """All seven absolute pose gradients match central differences."""
    qvec = _unit_qvec().requires_grad_()
    tvec = torch.tensor(
        (0.3, -0.7, 1.1),
        dtype=torch.float64,
        requires_grad=True,
    )
    rays_ori = torch.tensor(
        [[[[0.2, -0.1, 0.05], [0.0, 0.3, -0.2]]]],
        dtype=torch.float64,
    )
    rays_dir = torch.tensor(
        [[[[0.1, 0.2, 0.97], [-0.4, 0.3, 0.85]]]],
        dtype=torch.float64,
    )
    origin_weights = torch.tensor(
        [[[[0.7, -0.2, 0.4], [-0.3, 0.8, 0.1]]]],
        dtype=torch.float64,
    )
    direction_weights = torch.tensor(
        [[[[0.1, 0.6, -0.5], [0.9, -0.2, 0.3]]]],
        dtype=torch.float64,
    )

    def objective(
        test_qvec: torch.Tensor, test_tvec: torch.Tensor
    ) -> torch.Tensor:
        origins, directions = colmap_w2c_to_world_rays(
            rays_ori=rays_ori,
            rays_dir=rays_dir,
            qvec=test_qvec,
            tvec=test_tvec,
        )
        return (origins * origin_weights).sum() + (
            directions * direction_weights
        ).sum()

    objective(qvec, tvec).backward()
    analytical = torch.cat((qvec.grad, tvec.grad))
    parameters = torch.cat((qvec.detach(), tvec.detach()))
    finite_differences = torch.empty_like(parameters)
    epsilon = 1.0e-6
    for index in range(parameters.numel()):
        positive = parameters.clone()
        negative = parameters.clone()
        positive[index] += epsilon
        negative[index] -= epsilon
        finite_differences[index] = (
            objective(positive[:4], positive[4:])
            - objective(negative[:4], negative[4:])
        ) / (2.0 * epsilon)

    torch.testing.assert_close(
        analytical,
        finite_differences,
        atol=1.0e-8,
        rtol=1.0e-6,
    )


def test_world_ray_transform_preserves_nonfinite_sentinels() -> None:
    """Invalid fisheye rays remain invalid without poisoning valid rays."""
    rays_ori = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    rays_dir = torch.tensor(
        [[[[0.0, 0.0, 1.0], [float("nan"), 1.0, 0.0]]]],
        dtype=torch.float32,
    )

    _, world_directions = colmap_w2c_to_world_rays(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        qvec=torch.tensor((1.0, 0.0, 0.0, 0.0)),
        tvec=torch.zeros(3),
    )

    assert torch.equal(world_directions[..., 0, :], rays_dir[..., 0, :])
    assert torch.isnan(world_directions[..., 1, 0])
    assert world_directions[..., 1, 1].item() == pytest.approx(1.0)


def test_native_camera_stats_are_detached_and_report_center_drift() -> None:
    """Per-step diagnostics do not retain a graph for all source poses."""
    qvecs = torch.stack(
        (
            torch.tensor((1.0, 0.0, 0.0, 0.0)),
            torch.tensor((2.0**-0.5, 0.0, 0.0, 2.0**-0.5)),
        )
    )
    tvecs = torch.tensor(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
    module = NativeCameraExtrinsics(
        initial_qvecs=qvecs,
        initial_tvecs=tvecs,
    )
    with torch.no_grad():
        module.tvecs[1, 0] += 0.25

    stats = module.stats()

    assert stats["rotation_norm_rad"] == pytest.approx(0.0)
    assert stats["translation_norm_m"] == pytest.approx(0.25)
    assert module.qvecs.grad is None
    assert module.tvecs.grad is None
