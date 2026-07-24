"""Behavioral tests for visibility-adaptive LiDAR supervision."""

import struct

import numpy as np
import pytest
import torch
from threedgrut.lidar_supervision import LidarRaySampler, native_lidar_loss

_IDENTITY_TRANSFORM = (
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)


def _write_las(map_path, packets: tuple[tuple[float, tuple], ...]) -> None:
    point_count = sum(len(points) for _, points in packets)
    header = bytearray(227)
    header[:4] = b"LASF"
    struct.pack_into("<I", header, 96, 227)
    header[104] = 1
    struct.pack_into("<H", header, 105, 28)
    struct.pack_into("<I", header, 107, point_count)
    struct.pack_into("<3d", header, 131, 0.001, 0.001, 0.001)
    struct.pack_into("<3d", header, 155, 0.0, 0.0, 0.0)
    records = []
    for timestamp, points in packets:
        for point in points:
            record = bytearray(28)
            quantized = tuple(round(value * 1000.0) for value in point)
            struct.pack_into("<iii", record, 0, *quantized)
            struct.pack_into("<d", record, 20, timestamp)
            records.append(record)
    map_path.write_bytes(bytes(header) + b"".join(records))


def _write_fixture(tmp_path, *, separator: str = " "):
    map_path = tmp_path / "map.las"
    poses_path = tmp_path / "poses.csv"
    calibration_path = tmp_path / "extrinsic_imu_lidar.yaml"
    packet_a = (
        (-0.2, -0.1, 3.0),
        (-0.1, 0.0, 3.0),
        (0.0, 0.0, 3.0),
        (0.1, 0.0, 3.0),
        (0.2, 0.1, 3.0),
        (0.3, 0.1, 3.0),
    )
    packet_b = tuple((x, y, 7.0) for x, y, _ in packet_a)
    _write_las(map_path, ((1.0, packet_a), (2.0, packet_b)))
    poses_path.write_text(
        "#timestamp x y z qw qx qy qz\n"
        + separator.join(("1", "0", "0", "0", "1", "0", "0", "0"))
        + "\n"
        + separator.join(("2", "0", "0", "0", "1", "0", "0", "0"))
        + "\n"
    )
    calibration_path.write_text(
        "transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]\n"
    )
    return map_path, poses_path, calibration_path


def _make_memory_sampler(
    points: np.ndarray, *, seed: int = 0
) -> LidarRaySampler:
    sampler = LidarRaySampler.__new__(LidarRaySampler)
    sampler._rng = np.random.default_rng(seed)
    sampler._pcg_state = 0xA54C114A2B1B2EBC
    sampler._camera_rays = np.zeros((256, 256, 3), dtype=np.float64)
    sampler._next_packet_index = lambda: 0
    sampler._packet_points = lambda _packet_index: points
    sampler._lidar_pose = lambda _packet_index: (
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64),
    )
    return sampler


@pytest.mark.parametrize("separator", [pytest.param(" "), pytest.param(",")])
def test_sampler_keeps_timestamp_packets_separate(
    tmp_path,
    separator: str,
) -> None:
    """Each native step uses one packet and discards every second return."""
    map_path, poses_path, calibration_path = _write_fixture(
        tmp_path,
        separator=separator,
    )
    sampler = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=_IDENTITY_TRANSFORM,
        seed=7,
    )

    reconstructed_packets = []
    for _ in range(2):
        (
            ray_origins,
            ray_directions,
            camera_to_world,
            sample_grid,
            depths,
            ray_z,
            sample_weight,
        ) = sampler.sample(
            rays_per_step=32,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        focal_length = sampler.intrinsics[0]
        u = (sample_grid[..., 0:1] + 1.0) * 128.0 - 0.5
        v = (sample_grid[..., 1:2] + 1.0) * 128.0 - 0.5
        camera_points = torch.cat(
            (
                (u - 128.0) * depths / focal_length,
                (v - 128.0) * depths / focal_length,
                depths,
            ),
            dim=-1,
        )
        rotation = camera_to_world[0, :3, :3]
        origin = camera_to_world[0, :3, 3]
        points = camera_points @ rotation.T + origin
        reconstructed_packets.append(points)

        assert ray_origins.shape == (1, 256, 256, 3)
        assert ray_directions.shape == (1, 256, 256, 3)
        assert camera_to_world.shape == (1, 4, 4)
        assert depths.shape == (1, 1, 3, 1)
        assert torch.allclose(
            torch.linalg.vector_norm(ray_directions, dim=-1),
            torch.ones((1, 256, 256)),
        )
        assert torch.all(depths >= 0.1)
        assert torch.all(ray_z > 0.0)
        assert sample_weight == pytest.approx(1.0 / 3.0)

    packet_depths = sorted(
        float(points[..., 2].mean()) for points in reconstructed_packets
    )
    assert packet_depths == pytest.approx((3.0, 7.0))


def test_sampler_is_deterministic_for_equal_seeds(tmp_path) -> None:
    """Equal initialization seeds produce equal packet rays."""
    map_path, poses_path, calibration_path = _write_fixture(tmp_path)
    samplers = tuple(
        LidarRaySampler(
            map_path=str(map_path),
            poses_path=str(poses_path),
            calibration_path=str(calibration_path),
            map_to_world_transform=_IDENTITY_TRANSFORM,
            seed=19,
        )
        for _ in range(2)
    )

    samples = tuple(
        sampler.sample(
            rays_per_step=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for sampler in samplers
    )

    for left, right in zip(samples[0][:-1], samples[1][:-1], strict=True):
        assert torch.equal(left, right)
    assert samples[0][-1] == samples[1][-1]


def test_sampler_zero_cap_keeps_every_visible_post_stride_return() -> None:
    """Native mode sums all visible returns with packet normalization."""
    points = np.asarray(
        (
            (-0.1, 0.0, 3.0),
            (0.0, 0.0, 3.0),
            (0.1, 0.0, 3.0),
            (100.0, 0.0, 3.0),
        ),
        dtype=np.float64,
    )
    sampler = _make_memory_sampler(points)

    sample = sampler.sample(
        rays_per_step=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert sample is not None
    assert sample[4].shape == (1, 1, 3, 1)
    assert sample[-1] == pytest.approx(1.0 / len(points))


def test_sampler_uses_global_even_las_stride(tmp_path) -> None:
    """Native LAS stride retains globally even source-point rows."""
    packet_a = tuple((0.0, 0.0, 3.0 + 0.1 * index) for index in range(5))
    packet_b = tuple((0.0, 0.0, 7.0 + 0.1 * index) for index in range(5))
    map_path = tmp_path / "map.las"
    poses_path = tmp_path / "poses.csv"
    calibration_path = tmp_path / "extrinsic_imu_lidar.yaml"
    _write_las(map_path, ((1.0, packet_a), (2.0, packet_b)))
    poses_path.write_text(
        "#timestamp x y z qw qx qy qz\n"
        "1 0 0 0 1 0 0 0\n"
        "2 0 0 0 1 0 0 0\n"
    )
    calibration_path.write_text(
        "transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]\n"
    )
    sampler = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=_IDENTITY_TRANSFORM,
        seed=7,
    )

    assert sampler._packet_points(0).shape == (3, 3)
    assert sampler._packet_points(1).shape == (2, 3)


@pytest.mark.parametrize(
    ("cap", "expected_count", "expected_weight"),
    [
        pytest.param(2, 2, 3.0 / 8.0, id="subsample"),
        pytest.param(3, 3, 1.0 / 4.0, id="retain-all"),
    ],
)
def test_sampler_positive_cap_preserves_existing_weighting(
    cap: int,
    expected_count: int,
    expected_weight: float,
) -> None:
    """Positive experimental caps retain unbiased visible-ray sampling."""
    points = np.asarray(
        (
            (-0.1, 0.0, 3.0),
            (0.0, 0.0, 3.0),
            (0.1, 0.0, 3.0),
            (100.0, 0.0, 3.0),
        ),
        dtype=np.float64,
    )
    sampler = _make_memory_sampler(points)

    sample = sampler.sample(
        rays_per_step=cap,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert sample is not None
    assert sample[4].shape == (1, 1, expected_count, 1)
    assert sample[-1] == pytest.approx(expected_weight)


def test_sampler_rejects_negative_ray_cap() -> None:
    """Only zero has the native uncapped meaning."""
    sampler = _make_memory_sampler(
        np.asarray(((0.0, 0.0, 3.0),), dtype=np.float64)
    )

    with pytest.raises(ValueError, match="nonnegative"):
        sampler.sample(
            rays_per_step=-1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_sampler_repeats_native_priority_queue_epochs(tmp_path) -> None:
    """Native priority scheduling visits each packet before revisiting it."""
    map_path, poses_path, calibration_path = _write_fixture(tmp_path)
    sampler = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=_IDENTITY_TRANSFORM,
        seed=7,
    )

    samples = tuple(
        sampler.sample(
            rays_per_step=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for _ in range(4)
    )

    focal_length = sampler.intrinsics[0]
    depths = []
    for _, _, camera_to_world, sample_grid, sample_depth, _, _ in samples:
        u = (sample_grid[..., 0:1] + 1.0) * 128.0 - 0.5
        v = (sample_grid[..., 1:2] + 1.0) * 128.0 - 0.5
        camera_points = torch.cat(
            (
                (u - 128.0) * sample_depth / focal_length,
                (v - 128.0) * sample_depth / focal_length,
                sample_depth,
            ),
            dim=-1,
        )
        rotation = camera_to_world[0, :3, :3]
        origin = camera_to_world[0, :3, 3]
        world_points = camera_points @ rotation.T + origin
        depths.append(float(world_points[..., 2].mean()))
    assert sorted(depths[:2]) == pytest.approx((3.0, 7.0))
    assert sorted(depths[2:]) == pytest.approx((3.0, 7.0))


def test_sampler_uses_recovered_native_pcg_packet_order(tmp_path) -> None:
    """The native queue and target draw share one recovered PCG stream."""
    map_path, poses_path, calibration_path = _write_fixture(tmp_path)
    sampler = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=_IDENTITY_TRANSFORM,
        seed=999,
    )

    packet_order = []
    for _ in range(4):
        packet_index = sampler._next_packet_index()
        packet_order.append(packet_index)
        sampler._next_native_target_index(
            len(sampler._packet_points(packet_index))
        )

    assert packet_order == [1, 0, 1, 0]


def test_sampler_checkpoint_restores_native_queue_and_rng_state(tmp_path) -> None:
    """Resume matches uninterrupted sampling across native queue epochs."""
    map_path, poses_path, calibration_path = _write_fixture(tmp_path)

    def make_sampler(seed: int = 23) -> LidarRaySampler:
        return LidarRaySampler(
            map_path=str(map_path),
            poses_path=str(poses_path),
            calibration_path=str(calibration_path),
            map_to_world_transform=_IDENTITY_TRANSFORM,
            seed=seed,
        )

    uninterrupted = make_sampler()
    for _ in range(2):
        uninterrupted.sample(
            rays_per_step=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    checkpoint_path = tmp_path / "lidar_state.pt"
    torch.save(uninterrupted.state_dict(), checkpoint_path)
    state = torch.load(checkpoint_path, weights_only=False)
    expected_samples = tuple(
        uninterrupted.sample(
            rays_per_step=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for _ in range(3)
    )

    resumed = make_sampler(seed=999)
    resumed.load_state_dict(state)
    actual_samples = tuple(
        resumed.sample(
            rays_per_step=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for _ in range(3)
    )

    for expected, actual in zip(
        expected_samples,
        actual_samples,
        strict=True,
    ):
        for left, right in zip(expected[:-1], actual[:-1], strict=True):
            assert torch.equal(left, right)
        assert expected[-1] == actual[-1]

    mid_epoch_state = uninterrupted.state_dict()
    expected = uninterrupted.sample(
        rays_per_step=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    resumed_mid_epoch = make_sampler(seed=123)
    resumed_mid_epoch.load_state_dict(mid_epoch_state)
    actual = resumed_mid_epoch.sample(
        rays_per_step=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    for left, right in zip(expected[:-1], actual[:-1], strict=True):
        assert torch.equal(left, right)
    assert expected[-1] == actual[-1]


def test_sampler_checkpoint_rejects_changed_sources(tmp_path) -> None:
    """Resume rejects input files that differ from the saved stream."""
    map_path, poses_path, calibration_path = _write_fixture(tmp_path)
    sampler = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=_IDENTITY_TRANSFORM,
        seed=23,
    )
    state = sampler.state_dict()
    calibration_path.write_text(
        calibration_path.read_text() + "# changed after checkpoint\n"
    )
    changed = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=_IDENTITY_TRANSFORM,
        seed=23,
    )

    with pytest.raises(ValueError, match="source fingerprint"):
        changed.load_state_dict(state)


def test_sampler_rejects_checkpoint_without_exact_stream_state(tmp_path) -> None:
    """Native queue replay requires queue, PCG, RNG, and source state."""
    map_path, poses_path, calibration_path = _write_fixture(tmp_path)
    sampler = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=_IDENTITY_TRANSFORM,
        seed=23,
    )

    with pytest.raises(ValueError, match="lacks exact native queue"):
        sampler.load_state_dict({"packet_count": 2})

    for key in (
        "packet_queue",
        "pcg_state",
        "rng_state",
        "source_fingerprint",
    ):
        incomplete_state = sampler.state_dict()
        incomplete_state.pop(key)
        with pytest.raises(ValueError, match="lacks exact native queue"):
            sampler.load_state_dict(incomplete_state)


def test_sampler_applies_map_and_calibration_transforms(tmp_path) -> None:
    """Ray origins use both reconstruction alignment and LiDAR extrinsics."""
    map_path, poses_path, calibration_path = _write_fixture(tmp_path)
    calibration_path.write_text(
        "transform: [1, 0, 0, 0.5, 0, 1, 0, -0.25, "
        "0, 0, 1, 0.75, 0, 0, 0, 1]\n"
    )
    map_to_world = np.eye(4, dtype=np.float64)
    map_to_world[:3, 3] = (10.0, 20.0, 30.0)
    sampler = LidarRaySampler(
        map_path=str(map_path),
        poses_path=str(poses_path),
        calibration_path=str(calibration_path),
        map_to_world_transform=map_to_world.reshape(-1),
        seed=4,
    )

    _, _, camera_to_world, _, _, _, _ = sampler.sample(
        rays_per_step=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    expected = torch.tensor((10.5, 19.75, 30.75))
    assert torch.allclose(camera_to_world[0, :3, 3], expected)


def test_native_lidar_loss_matches_recovered_type_five_equation() -> None:
    """Native type-five values and gradients match the recovered PTX."""
    pred = torch.tensor((1.25, 3.5), requires_grad=True)
    target = torch.tensor((1.0, 4.0))

    loss = native_lidar_loss(pred, target)
    loss.sum().backward()

    delta = pred.detach() - target
    normalizer = 1.0 - 0.4 * target + 0.1 * target.square()
    expected = 0.1 * torch.log1p(10.0 * torch.abs(delta)) / normalizer
    expected_gradient = (
        torch.sign(delta) / (1.0 + 10.0 * torch.abs(delta)) / normalizer
    )
    assert torch.allclose(loss, expected)
    assert torch.allclose(pred.grad, expected_gradient)
