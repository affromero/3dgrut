"""Visibility-adaptive-style LiDAR packet supervision."""

import hashlib
import os
import struct
from collections.abc import Sequence

import numpy as np
import torch
from yaml import safe_load

_LAS_HEADER_MIN_SIZE = 227
_LAS_POINT_DATA_OFFSET = 96
_LAS_POINT_FORMAT_OFFSET = 104
_LAS_POINT_RECORD_LENGTH_OFFSET = 105
_LAS_LEGACY_POINT_COUNT_OFFSET = 107
_LAS_SCALE_OFFSET = 131
_LAS_OFFSET_OFFSET = 155
_LAS_FORMAT_WITH_GPS_TIME = 1
_LAS_GPS_TIME_OFFSET = 20
_NATIVE_IMAGE_SIZE = 256.0
_NATIVE_PRINCIPAL_POINT = 128.0
_NATIVE_HALF_FOV_RADIANS = np.pi / 3.0
_NATIVE_FOCAL_LENGTH = _NATIVE_IMAGE_SIZE / (
    2.0 * np.tan(_NATIVE_HALF_FOV_RADIANS)
)
_NATIVE_MIN_AXIAL_DEPTH = 0.1
_TIMESTAMP_VALIDATION_CHUNK_SIZE = 1_000_000
_NATIVE_PCG_MULTIPLIER = 0x5851F42D4C957F2D
_NATIVE_PCG_INCREMENT = 3
_NATIVE_PCG_MASK_64 = (1 << 64) - 1
_NATIVE_PCG_MASK_32 = (1 << 32) - 1
_NATIVE_LIDAR_PCG_SEED = 0xA54C114A2B1B2EBC


def native_lidar_loss(
    pred_depth: torch.Tensor,
    target_depth: torch.Tensor,
) -> torch.Tensor:
    """Return native ELossType 5 with its target-depth normalization."""
    delta = pred_depth - target_depth
    robust_loss = 0.1 * torch.log1p(10.0 * torch.abs(delta))
    depth_normalizer = 1.0 - 0.4 * target_depth + 0.1 * target_depth.square()
    return robust_loss / depth_normalizer


def _native_pcg_step(state: int) -> tuple[int, int]:
    """Advance the recovered native PCG stream and return one u32 value."""
    xorshifted = ((state >> 45) ^ (state >> 27)) & _NATIVE_PCG_MASK_32
    rotation = (state >> 59) & 31
    value = (
        (xorshifted >> rotation)
        | (xorshifted << ((-rotation) & 31))
    ) & _NATIVE_PCG_MASK_32
    next_state = (
        state * _NATIVE_PCG_MULTIPLIER + _NATIVE_PCG_INCREMENT
    ) & _NATIVE_PCG_MASK_64
    return value, next_state


def _quaternion_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("LiDAR pose contains an invalid quaternion.")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-10:
        axis_candidates = np.eye(3, dtype=np.float64)
        axis = axis_candidates[np.argmin(np.abs(axis_candidates @ source))]
        axis = np.cross(source, axis)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    cross = np.cross(source, target)
    skew = np.asarray(
        (
            (0.0, -cross[2], cross[1]),
            (cross[2], 0.0, -cross[0]),
            (-cross[1], cross[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew / (1.0 + dot)


class LidarRaySampler:
    """Sample native virtual-camera rays from timestamped LAS packets."""

    def __init__(
        self,
        map_path: str,
        poses_path: str,
        calibration_path: str,
        map_to_world_transform: Sequence[float],
        seed: int,
    ) -> None:
        for label, path in (
            ("map", map_path),
            ("poses", poses_path),
            ("calibration", calibration_path),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"LiDAR {label} does not exist: {path}"
                )

        (
            point_offset,
            point_format,
            point_size,
            point_count,
            scale,
            offset,
        ) = self._read_las_header(map_path)
        if point_format != _LAS_FORMAT_WITH_GPS_TIME:
            raise ValueError(
                "Native LiDAR supervision requires LAS point format 1 "
                f"with GPS time, got format {point_format}."
            )
        if point_size < _LAS_GPS_TIME_OFFSET + 8:
            raise ValueError(
                "Native LiDAR supervision requires GPS time at LAS "
                f"offset {_LAS_GPS_TIME_OFFSET}, got point size {point_size}."
            )
        if point_count <= 0:
            raise ValueError("LiDAR map contains no points.")

        point_dtype = np.dtype(
            {
                "names": ("x", "y", "z", "timestamp"),
                "formats": ("<i4", "<i4", "<i4", "<f8"),
                "offsets": (0, 4, 8, _LAS_GPS_TIME_OFFSET),
                "itemsize": point_size,
            }
        )
        self._points = np.memmap(
            map_path,
            dtype=point_dtype,
            mode="r",
            offset=point_offset,
            shape=(point_count,),
        )
        self._scale = np.asarray(scale, dtype=np.float64)
        self._offset = np.asarray(offset, dtype=np.float64)
        self._rng = np.random.default_rng(seed)

        pose_delimiter = self._pose_delimiter(poses_path)
        pose_data = np.loadtxt(
            poses_path,
            delimiter=pose_delimiter,
            comments="#",
            usecols=tuple(range(8)),
            dtype=np.float64,
        )
        if pose_data.ndim == 1:
            pose_data = pose_data.reshape(1, -1)
        if pose_data.shape[0] < 2:
            raise ValueError(
                "Native LiDAR supervision requires at least two packets."
            )
        if not np.all(np.diff(pose_data[:, 0]) > 0.0):
            raise ValueError(
                "LiDAR pose timestamps must be strictly increasing."
            )
        self._pose_data = pose_data

        self._validate_sorted_timestamps()
        point_times = self._points["timestamp"]
        self._packet_starts = np.searchsorted(
            point_times,
            pose_data[:, 0],
            side="left",
        )
        self._packet_ends = np.searchsorted(
            point_times,
            pose_data[:, 0],
            side="right",
        )
        missing = np.flatnonzero(self._packet_starts == self._packet_ends)
        if missing.size:
            timestamp = pose_data[int(missing[0]), 0]
            raise ValueError(
                "LAS packets must match poses.csv timestamps exactly; "
                f"no packet exists for {timestamp:.9f}."
            )

        transform_values = np.asarray(
            tuple(map_to_world_transform),
            dtype=np.float64,
        )
        if transform_values.size != 16:
            raise ValueError(
                "map_to_world_transform must contain 16 row-major values."
            )
        self._map_to_world = transform_values.reshape(4, 4)
        if not np.allclose(
            self._map_to_world[3],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        ):
            raise ValueError("map_to_world_transform must be affine.")
        self._imu_to_lidar = self._read_calibration(calibration_path)
        fingerprint = hashlib.sha256()
        for path in (map_path, poses_path, calibration_path):
            fingerprint.update(bytes.fromhex(self._file_sha256(path)))
        fingerprint.update(self._map_to_world.tobytes())
        self._source_fingerprint = fingerprint.hexdigest()

        self._pcg_state = _NATIVE_LIDAR_PCG_SEED
        self._packet_queue = self._initialize_packet_queue(len(pose_data))
        pixel_axis = np.arange(int(_NATIVE_IMAGE_SIZE), dtype=np.float64)
        pixel_x, pixel_y = np.meshgrid(pixel_axis, pixel_axis)
        camera_rays = np.stack(
            (
                (pixel_x - _NATIVE_PRINCIPAL_POINT) / _NATIVE_FOCAL_LENGTH,
                (pixel_y - _NATIVE_PRINCIPAL_POINT) / _NATIVE_FOCAL_LENGTH,
                np.ones_like(pixel_x),
            ),
            axis=-1,
        )
        self._camera_rays = camera_rays / np.linalg.norm(
            camera_rays,
            axis=-1,
            keepdims=True,
        )

    @property
    def intrinsics(self) -> list[float]:
        """Return the recovered native virtual-camera intrinsics."""
        return [
            float(_NATIVE_FOCAL_LENGTH),
            float(_NATIVE_FOCAL_LENGTH),
            _NATIVE_PRINCIPAL_POINT,
            _NATIVE_PRINCIPAL_POINT,
        ]

    @property
    def packet_count(self) -> int:
        """Return the number of native LiDAR packet queue entries."""
        return len(self._packet_queue)

    def state_dict(self) -> dict[str, object]:
        """Return the exact native queue and random-sampling state."""
        return {
            "packet_count": self.packet_count,
            "packet_queue": list(self._packet_queue),
            "pcg_state": self._pcg_state,
            "rng_state": self._rng.bit_generator.state,
            "source_fingerprint": self._source_fingerprint,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the exact native queue and random-sampling position."""
        packet_count = int(state["packet_count"])
        required_state = (
            "packet_queue",
            "pcg_state",
            "rng_state",
            "source_fingerprint",
        )
        if any(key not in state for key in required_state):
            raise ValueError(
                "LiDAR checkpoint lacks exact native queue, PCG, RNG, or "
                "source state."
            )
        packet_queue_state = state["packet_queue"]
        if not isinstance(packet_queue_state, list):
            raise TypeError("LiDAR checkpoint packet queue must be a list.")
        packet_queue = []
        for entry in packet_queue_state:
            if not isinstance(entry, tuple) or len(entry) != 3:
                raise ValueError("LiDAR checkpoint packet queue is invalid.")
            visit_count, priority, packet_index = entry
            if not all(
                isinstance(value, (int, np.integer))
                for value in (visit_count, priority, packet_index)
            ):
                raise TypeError(
                    "LiDAR checkpoint packet queue values must be integers."
                )
            packet_queue.append(
                (int(visit_count), int(priority), int(packet_index))
            )
        pcg_state = state["pcg_state"]
        rng_state = state["rng_state"]
        source_fingerprint = str(state["source_fingerprint"])
        if source_fingerprint != self._source_fingerprint:
            raise ValueError(
                "LiDAR checkpoint source fingerprint does not match."
            )
        if packet_count != self.packet_count:
            raise ValueError(
                "LiDAR checkpoint packet count does not match the dataset."
            )
        if len(packet_queue) != packet_count:
            raise ValueError("LiDAR checkpoint packet queue size is invalid.")
        if sorted(packet_index for _, _, packet_index in packet_queue) != list(
            range(packet_count)
        ):
            raise ValueError("LiDAR checkpoint packet queue indexes are invalid.")
        if any(
            visit_count < 0
            or not 0 <= priority <= _NATIVE_PCG_MASK_32
            for visit_count, priority, _ in packet_queue
        ):
            raise ValueError("LiDAR checkpoint packet queue values are invalid.")
        if packet_queue != sorted(
            packet_queue,
            key=lambda entry: (entry[0], entry[1]),
        ):
            raise ValueError("LiDAR checkpoint packet queue ordering is invalid.")
        if not isinstance(pcg_state, (int, np.integer)):
            raise TypeError("LiDAR checkpoint PCG state must be an integer.")
        if not 0 <= int(pcg_state) <= _NATIVE_PCG_MASK_64:
            raise ValueError("LiDAR checkpoint PCG state is invalid.")
        if not isinstance(rng_state, dict):
            raise TypeError("LiDAR checkpoint RNG state must be a mapping.")
        self._packet_queue = packet_queue
        self._pcg_state = int(pcg_state)
        self._rng.bit_generator.state = rng_state

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _pose_delimiter(poses_path: str) -> str | None:
        with open(poses_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    return "," if "," in stripped else None
        raise ValueError(f"LiDAR pose file has no data rows: {poses_path}")

    @staticmethod
    def _read_calibration(calibration_path: str) -> np.ndarray:
        with open(calibration_path, "r", encoding="utf-8") as handle:
            calibration = safe_load(handle)
        if not isinstance(calibration, dict):
            raise ValueError(
                f"Invalid LiDAR calibration document: {calibration_path}"
            )
        transform_values = np.asarray(
            calibration.get("transform", ()),
            dtype=np.float64,
        )
        if transform_values.size != 16:
            raise ValueError(
                "LiDAR calibration transform must contain 16 row-major values."
            )
        transform = transform_values.reshape(4, 4)
        if not np.allclose(
            transform[3],
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        ):
            raise ValueError("LiDAR calibration transform must be affine.")
        return transform

    @staticmethod
    def _read_las_header(
        map_path: str,
    ) -> tuple[
        int,
        int,
        int,
        int,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        with open(map_path, "rb") as handle:
            header = handle.read(_LAS_HEADER_MIN_SIZE)
        if len(header) < _LAS_HEADER_MIN_SIZE or header[:4] != b"LASF":
            raise ValueError(f"Invalid LAS header: {map_path}")
        point_offset = struct.unpack_from(
            "<I", header, _LAS_POINT_DATA_OFFSET
        )[0]
        point_format = header[_LAS_POINT_FORMAT_OFFSET] & 0x3F
        point_size = struct.unpack_from(
            "<H", header, _LAS_POINT_RECORD_LENGTH_OFFSET
        )[0]
        point_count = struct.unpack_from(
            "<I", header, _LAS_LEGACY_POINT_COUNT_OFFSET
        )[0]
        scale = struct.unpack_from("<3d", header, _LAS_SCALE_OFFSET)
        offset = struct.unpack_from("<3d", header, _LAS_OFFSET_OFFSET)
        return (
            point_offset,
            point_format,
            point_size,
            point_count,
            scale,
            offset,
        )

    def _validate_sorted_timestamps(self) -> None:
        point_times = self._points["timestamp"]
        previous = -np.inf
        for start in range(
            0, len(point_times), _TIMESTAMP_VALIDATION_CHUNK_SIZE
        ):
            chunk = np.asarray(
                point_times[start : start + _TIMESTAMP_VALIDATION_CHUNK_SIZE]
            )
            if chunk.size == 0:
                continue
            if not np.all(np.isfinite(chunk)):
                raise ValueError("LAS packet timestamps must be finite.")
            if chunk[0] < previous or np.any(np.diff(chunk) < 0.0):
                raise ValueError("LAS packet timestamps must be sorted.")
            previous = float(chunk[-1])

    def _initialize_packet_queue(
        self,
        packet_count: int,
    ) -> list[tuple[int, int, int]]:
        packet_queue = []
        for packet_index in range(packet_count):
            priority = self._next_native_pcg_value()
            packet_queue.append((0, priority, packet_index))
        packet_queue.sort(key=lambda entry: (entry[0], entry[1]))
        return packet_queue

    def _next_native_pcg_value(self) -> int:
        value, self._pcg_state = _native_pcg_step(self._pcg_state)
        return value

    def _next_packet_index(self) -> int:
        visit_count, priority, packet_index = self._packet_queue.pop(0)
        priority = (
            priority + self._next_native_pcg_value()
        ) & _NATIVE_PCG_MASK_32
        self._packet_queue.append(
            (visit_count + 1, priority, packet_index)
        )
        self._packet_queue.sort(key=lambda entry: (entry[0], entry[1]))
        return packet_index

    def _next_native_target_index(self, point_count: int) -> int:
        if point_count <= 0:
            raise ValueError("Native LiDAR packet must contain points.")
        rejection_threshold = (
            (1 << 32) - point_count
        ) % point_count
        while True:
            candidate = self._next_native_pcg_value()
            if candidate >= rejection_threshold:
                return candidate % point_count

    def _packet_points(self, packet_index: int) -> np.ndarray:
        start = int(self._packet_starts[packet_index])
        end = int(self._packet_ends[packet_index])
        first_native_index = start + (start & 1)
        native_indices = np.arange(
            first_native_index,
            end,
            2,
            dtype=np.int64,
        )
        records = self._points[native_indices]
        raw_points = np.column_stack(
            (
                records["x"] * self._scale[0] + self._offset[0],
                records["y"] * self._scale[1] + self._offset[1],
                records["z"] * self._scale[2] + self._offset[2],
            )
        )
        homogeneous = np.column_stack(
            (raw_points, np.ones(len(raw_points), dtype=np.float64))
        )
        return (homogeneous @ self._map_to_world.T)[:, :3]

    def _lidar_pose(self, packet_index: int) -> tuple[np.ndarray, np.ndarray]:
        pose = self._pose_data[packet_index]
        imu_pose = np.eye(4, dtype=np.float64)
        imu_pose[:3, :3] = _quaternion_rotation(pose[4:8])
        imu_pose[:3, 3] = pose[1:4]
        lidar_pose = self._map_to_world @ imu_pose @ self._imu_to_lidar
        return lidar_pose[:3, 3], lidar_pose[:3, :3]

    def sample(
        self,
        *,
        rays_per_step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            float,
        ]
    ):
        """Return one native virtual camera and its projected LiDAR targets."""
        if rays_per_step < 0:
            raise ValueError("rays_per_step must be nonnegative.")
        packet_index = self._next_packet_index()
        points = self._packet_points(packet_index)
        origin, base_rotation = self._lidar_pose(packet_index)

        target_index = self._next_native_target_index(len(points))
        target_vector = points[target_index] - origin
        target_range = np.linalg.norm(target_vector)
        if not np.isfinite(target_range) or target_range <= 1e-4:
            raise RuntimeError(
                "Native LiDAR packet selected an invalid target."
            )
        forward = target_vector / target_range
        reference_forward = base_rotation[:, 2]
        camera_rotation = (
            _rotation_between(reference_forward, forward) @ base_rotation
        )

        camera_points = (points - origin) @ camera_rotation
        axial_depth = camera_points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (
                _NATIVE_FOCAL_LENGTH * camera_points[:, 0] / axial_depth
                + _NATIVE_PRINCIPAL_POINT
            )
            v = (
                _NATIVE_FOCAL_LENGTH * camera_points[:, 1] / axial_depth
                + _NATIVE_PRINCIPAL_POINT
            )
        visible = (
            np.isfinite(u)
            & np.isfinite(v)
            & np.isfinite(axial_depth)
            & (axial_depth >= _NATIVE_MIN_AXIAL_DEPTH)
            & (u >= 0.0)
            & (v >= 0.0)
            & (u < _NATIVE_IMAGE_SIZE)
            & (v < _NATIVE_IMAGE_SIZE)
        )
        visible_indices = np.flatnonzero(visible)
        if visible_indices.size == 0:
            raise RuntimeError("Native LiDAR packet has no visible points.")
        visible_count = int(visible_indices.size)
        if rays_per_step > 0 and visible_count > rays_per_step:
            visible_indices = self._rng.choice(
                visible_indices,
                size=rays_per_step,
                replace=False,
            )

        selected_u = u[visible_indices]
        selected_v = v[visible_indices]
        selected_depth = axial_depth[visible_indices]
        ray_x = (selected_u - _NATIVE_PRINCIPAL_POINT) / _NATIVE_FOCAL_LENGTH
        ray_y = (selected_v - _NATIVE_PRINCIPAL_POINT) / _NATIVE_FOCAL_LENGTH
        ray_z = 1.0 / np.sqrt(ray_x * ray_x + ray_y * ray_y + 1.0)
        sample_grid = np.column_stack(
            (
                (selected_u + 0.5) * 2.0 / _NATIVE_IMAGE_SIZE - 1.0,
                (selected_v + 0.5) * 2.0 / _NATIVE_IMAGE_SIZE - 1.0,
            )
        )
        sample_weight = visible_count / (len(points) * len(visible_indices))
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = camera_rotation
        camera_to_world[:3, 3] = origin
        ray_origins = np.zeros_like(self._camera_rays)

        def image_tensor(values: np.ndarray) -> torch.Tensor:
            return (
                torch.from_numpy(values.astype(np.float32, copy=False))
                .to(device=device, dtype=dtype)
                .unsqueeze(0)
            )

        def sample_tensor(values: np.ndarray, channels: int) -> torch.Tensor:
            return (
                torch.from_numpy(values.astype(np.float32, copy=False))
                .to(device=device, dtype=dtype)
                .view(1, 1, -1, channels)
            )

        return (
            image_tensor(ray_origins),
            image_tensor(self._camera_rays),
            image_tensor(camera_to_world),
            sample_tensor(sample_grid, 2),
            sample_tensor(selected_depth[:, None], 1),
            sample_tensor(ray_z[:, None], 1),
            sample_weight,
        )
