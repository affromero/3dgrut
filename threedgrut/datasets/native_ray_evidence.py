"""Authenticated native range-ray evidence for geometry-only training."""

import hashlib
import json
import os

import numpy as np
import torch

from threedgrut.datasets.protocols import Batch


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class NativeRayEvidence:
    """Load deterministic train-only rays in a coherent spherical raster."""

    def __init__(
        self,
        *,
        folder: str,
        manifest_sha256: str,
        sample_count: int,
        seed: int,
        equirect_width: int = 256,
        equirect_height: int = 128,
    ) -> None:
        if len(manifest_sha256) != 64:
            raise ValueError(
                "Native-ray evidence requires a manifest SHA-256."
            )
        if sample_count <= 0:
            raise ValueError(
                "Native-ray sample count must be positive."
            )
        if (
            equirect_width <= 0
            or equirect_height <= 0
            or equirect_width != 2 * equirect_height
        ):
            raise ValueError(
                "Native-ray equirect raster must have positive 2:1 "
                "dimensions."
            )
        manifest_path = os.path.join(folder, "manifest.json")
        observed_manifest_sha256 = _sha256_file(manifest_path)
        if observed_manifest_sha256 != manifest_sha256:
            raise ValueError(
                "Native-ray manifest SHA-256 mismatch: "
                f"{observed_manifest_sha256} != {manifest_sha256}."
            )
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self._validate_manifest(manifest)
        rows = manifest["rows"]
        self.folder = folder
        self.sample_count = sample_count
        self.seed = seed
        self.equirect_width = equirect_width
        self.equirect_height = equirect_height
        self._canonical_directions = (
            self._build_canonical_directions()
        )
        self._rows = {
            int(row["exposure_index"]): {
                "file": str(row["file"]),
                "sha256": str(row["sha256"]),
                "ray_count": int(row["ray_count"]),
            }
            for row in rows
        }
        self._cache: dict[
            int,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    @staticmethod
    def _validate_manifest(manifest: dict[str, object]) -> None:
        if int(manifest.get("schema_version", -1)) != 3:
            raise ValueError(
                "Unsupported native-ray manifest schema."
            )
        if (
            manifest.get("return_weight_semantics")
            != "normalized_leica_device_w_amplitude"
        ):
            raise ValueError(
                "Native-ray evidence requires Leica amplitude weights."
            )
        if bool(manifest.get("sealed_test_used", True)):
            raise ValueError(
                "Native-ray evidence must not use the sealed test."
            )
        if bool(manifest.get("development_data_used", True)):
            raise ValueError(
                "Native-ray evidence must be train-only."
            )
        if manifest.get("trajectory_fit_parity") != "even":
            raise ValueError(
                "Native-ray trajectory must be fit on even records."
            )
        if manifest.get("supervised_return_parity") != "odd":
            raise ValueError(
                "Native-ray supervision must use held-out odd records."
            )
        rows = manifest.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                "Native-ray manifest must contain exposure rows."
            )

    def _load_exposure(
        self,
        exposure_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(exposure_index)
        if cached is not None:
            return cached
        row = self._rows.get(exposure_index)
        if row is None:
            raise KeyError(
                "Native-ray evidence has no train exposure "
                f"{exposure_index}."
            )
        path = os.path.join(self.folder, row["file"])
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != row["sha256"]:
            raise ValueError(
                "Native-ray exposure SHA-256 mismatch for "
                f"{exposure_index}: {observed_sha256} != "
                f"{row['sha256']}."
            )
        with np.load(path, allow_pickle=False) as payload:
            origins = np.asarray(
                payload["origins"],
                dtype=np.float32,
            )
            directions = np.asarray(
                payload["directions"],
                dtype=np.float32,
            )
            depths = np.asarray(
                payload["depths"],
                dtype=np.float32,
            )
            return_weights = np.asarray(
                payload["return_weights"],
                dtype=np.float32,
            )
        expected_count = int(row["ray_count"])
        if (
            origins.shape != (expected_count, 3)
            or directions.shape != (expected_count, 3)
            or depths.shape != (expected_count,)
            or return_weights.shape != (expected_count,)
        ):
            raise ValueError(
                "Native-ray exposure shape mismatch for "
                f"{exposure_index}."
            )
        if (
            not np.isfinite(origins).all()
            or not np.isfinite(directions).all()
            or not np.isfinite(depths).all()
            or not np.isfinite(return_weights).all()
            or np.any(depths <= 0.0)
            or np.any(return_weights < 0.0)
            or np.any(return_weights > 1.0)
        ):
            raise ValueError(
                "Native-ray exposure contains invalid values for "
                f"{exposure_index}."
            )
        direction_norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(
            direction_norms,
            1.0,
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError(
                "Native-ray directions are not unit length for "
                f"{exposure_index}."
            )
        result = (origins, directions, depths, return_weights)
        self._cache[exposure_index] = result
        return result

    def _build_canonical_directions(self) -> np.ndarray:
        """Build the renderer's right-down-forward equirect ray raster."""
        width = self.equirect_width
        height = self.equirect_height
        columns = np.tile(np.arange(width) + 0.5, height)
        rows = (np.arange(height) + 0.5).repeat(width)
        phi = (1.0 - 2.0 * columns / width) * np.pi
        theta = (
            (1.0 - 2.0 * rows / height)
            * (np.pi / 2.0)
        )
        cos_theta = np.cos(theta)
        directions = np.stack(
            (
                -cos_theta * np.sin(phi),
                -np.sin(theta),
                cos_theta * np.cos(phi),
            ),
            axis=-1,
        )
        return directions.astype(np.float32).reshape(
            height,
            width,
            3,
        )

    def _equirect_indices(
        self,
        directions: np.ndarray,
    ) -> np.ndarray:
        """Map right-down-forward bearings to renderer pixel indices."""
        x = directions[:, 0]
        y = np.clip(directions[:, 1], -1.0, 1.0)
        z = directions[:, 2]
        phi = np.arctan2(-x, z)
        theta = -np.arcsin(y)
        columns = np.floor(
            (1.0 - phi / np.pi)
            * self.equirect_width
            / 2.0
        ).astype(np.int64)
        rows = np.floor(
            (1.0 - 2.0 * theta / np.pi)
            * self.equirect_height
            / 2.0
        ).astype(np.int64)
        columns %= self.equirect_width
        rows = np.clip(rows, 0, self.equirect_height - 1)
        return rows * self.equirect_width + columns

    def sample(
        self,
        *,
        exposure_index: int,
        global_step: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Batch:
        """Return one deterministic world-space ray batch."""
        origins, directions, depths, return_weights = self._load_exposure(
            exposure_index
        )
        generator = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, global_step, exposure_index]
            )
        )
        flat_pixels = self._equirect_indices(directions)
        permutation = generator.permutation(len(depths))
        _, first_occurrences = np.unique(
            flat_pixels[permutation],
            return_index=True,
        )
        selected = permutation[first_occurrences]
        if len(selected) > self.sample_count:
            selected = generator.choice(
                selected,
                size=self.sample_count,
                replace=False,
            )
        selected_pixels = flat_pixels[selected]

        height = self.equirect_height
        width = self.equirect_width
        representative_origin = np.median(origins, axis=0).astype(
            np.float32
        )
        ray_origins = np.broadcast_to(
            np.zeros(3, dtype=np.float32),
            (height, width, 3),
        ).copy()
        ray_directions = self._canonical_directions.copy()
        ray_depths = np.ones((height, width, 1), dtype=np.float32)
        ray_return_weights = np.zeros(
            (height, width, 1),
            dtype=np.float32,
        )
        mask = np.zeros((height, width, 1), dtype=np.float32)
        flat_origins = ray_origins.reshape(-1, 3)
        flat_directions = ray_directions.reshape(-1, 3)
        flat_depths = ray_depths.reshape(-1, 1)
        flat_return_weights = ray_return_weights.reshape(-1, 1)
        flat_mask = mask.reshape(-1, 1)
        flat_origins[selected_pixels] = (
            origins[selected] - representative_origin
        )
        flat_directions[selected_pixels] = directions[selected]
        flat_depths[selected_pixels, 0] = depths[selected]
        flat_return_weights[selected_pixels, 0] = return_weights[selected]
        flat_mask[selected_pixels, 0] = 1.0

        tensor_origins = torch.from_numpy(
            ray_origins
        ).to(device=device, dtype=dtype).unsqueeze(0)
        tensor_directions = torch.from_numpy(
            ray_directions
        ).to(device=device, dtype=dtype).unsqueeze(0)
        tensor_depths = torch.from_numpy(
            ray_depths
        ).to(device=device, dtype=dtype).unsqueeze(0)
        tensor_return_weights = torch.from_numpy(
            ray_return_weights
        ).to(device=device, dtype=dtype).unsqueeze(0)
        tensor_mask = torch.from_numpy(
            mask
        ).to(device=device, dtype=dtype).unsqueeze(0)
        camera_to_world = torch.eye(
            4,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        camera_to_world[0, :3, 3] = torch.from_numpy(
            representative_origin
        ).to(device=device, dtype=dtype)
        return Batch(
            rays_ori=tensor_origins,
            rays_dir=tensor_directions,
            T_to_world=camera_to_world,
            T_to_world_end=camera_to_world,
            rays_in_world_space=False,
            rgb_gt=torch.zeros(
                (1, height, width, 3),
                device=device,
                dtype=dtype,
            ),
            depth_gt=tensor_depths,
            depth_ray_z=torch.ones_like(tensor_depths),
            range_return_weight=tensor_return_weights,
            mask=tensor_mask,
            intrinsics_EquirectCameraModelParameters={
                "resolution": np.asarray(
                    [width, height],
                    dtype=np.uint64,
                ),
                "shutter_type": "GLOBAL",
            },
            sequence_idx=exposure_index,
            image_path=f"native_ray_{exposure_index:04d}.npz",
        )
