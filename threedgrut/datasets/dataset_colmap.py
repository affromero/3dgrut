# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
from typing import Optional

import ncore.sensors
import numpy as np
import torch
from ncore.data import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .colmap_gsplat import normalize_world_space, scene_scale
from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .rs_rays import build_rs_world_rays
from .utils import (
    compute_fisheye_max_angle,
    create_camera_visualization,
    create_pixel_coords,
    get_center_and_diag,
    get_worker_id,
    pinhole_camera_rays,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
    read_colmap_intrinsics_binary,
    read_colmap_intrinsics_text,
    read_colmap_points3D_binary,
    read_colmap_points3D_text,
)


_CANONICAL_FLAT_PHYSICAL_CAMERA_NAMES = ("front", "left", "right")
_SOFT_MASK_CONTRACT_FILENAME = "soft_training_masks.json"
_SOFT_MASK_SEMANTICS = "solid_angle_partition_of_unity_loss_weight_v1"
_MASK_VALIDITY_MODES = {"binary_threshold", "nonzero"}


def _read_rgb_image_array(image_path: str) -> np.ndarray:
    """Read an image as uint8 RGB, dropping alpha channels if present."""
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image, copy=True)


def _interp_c2w_knots(slerp, rel_stamps, translations, query_stamps):
    """Interpolate camera-to-world matrices from knot poses.

    Args:
        slerp: scipy Slerp over the knot rotations.
        rel_stamps: [J] knot stamps (seconds, relative to the frame stamp).
        translations: [J, 3] knot camera centers.
        query_stamps: [Q] query stamps within the knot span.

    Returns:
        [Q, 4, 4] camera-to-world matrices.
    """
    matrices = np.repeat(np.eye(4)[None], len(query_stamps), axis=0)
    matrices[:, :3, :3] = slerp(query_stamps).as_matrix()
    for axis in range(3):
        matrices[:, axis, 3] = np.interp(query_stamps, rel_stamps, translations[:, axis])
    return matrices


def _get_relative_paths(path_dir: str) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(path_dir):
        for filename in filenames:
            paths.append(os.path.relpath(os.path.join(dirpath, filename), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    logger.info(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    for image_file in _get_relative_paths(image_dir):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(resized_dir, os.path.splitext(image_file)[0] + ".png")
        if os.path.isfile(resized_path):
            continue

        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            resized_size = (
                int(round(image.width / factor)),
                int(round(image.height / factor)),
            )
            image.resize(resized_size, Image.Resampling.BICUBIC).save(resized_path)

    return resized_dir


def _opencv_pinhole_intrinsics_from_colmap(intr_model: str, intr_params: np.ndarray, scaling_factor: float):
    """Convert COLMAP distorted pinhole parameters to OpenCVPinholeCameraModelParameters fields."""
    intr_params = np.asarray(intr_params, dtype=np.float32)
    radial_coeffs = np.zeros((6,), dtype=np.float32)
    tangential_coeffs = np.zeros((2,), dtype=np.float32)
    thin_prism_coeffs = np.zeros((4,), dtype=np.float32)

    if intr_model == "SIMPLE_RADIAL":
        focal_length = np.array([intr_params[0], intr_params[0]], dtype=np.float32) / scaling_factor
        principal_point = intr_params[1:3] / scaling_factor
        radial_coeffs[0] = intr_params[3]
    elif intr_model == "RADIAL":
        focal_length = np.array([intr_params[0], intr_params[0]], dtype=np.float32) / scaling_factor
        principal_point = intr_params[1:3] / scaling_factor
        radial_coeffs[:2] = intr_params[3:5]
    elif intr_model == "OPENCV":
        focal_length = intr_params[:2] / scaling_factor
        principal_point = intr_params[2:4] / scaling_factor
        radial_coeffs[:2] = intr_params[4:6]
        tangential_coeffs[:] = intr_params[6:8]
    elif intr_model == "FULL_OPENCV":
        focal_length = intr_params[:2] / scaling_factor
        principal_point = intr_params[2:4] / scaling_factor
        radial_coeffs[:] = intr_params[[4, 5, 8, 9, 10, 11]]
        tangential_coeffs[:] = intr_params[6:8]
    else:
        raise ValueError(f"Unsupported distorted pinhole camera model: {intr_model}")

    return focal_length, principal_point, radial_coeffs, tangential_coeffs, thin_prism_coeffs


class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    def __init__(
        self,
        path,
        device="cuda",
        split="train",
        downsample_factor=1,
        test_split_interval=8,
        ray_jitter=None,
        exif_exposures: Optional[list[Optional[float]]] = None,
        camera_names: Optional[list[str]] = None,
        camera_ids: Optional[list[int]] = None,
        normalize_world_space: bool = False,
        gsplat_image_downscale: bool = False,
        sky_mask_folder: Optional[str] = None,
        train_exclude_image_list_path=None,
        train_focus_image_list_path: Optional[str] = None,
        train_focus_image_weight: float = 1.0,
        holdout_image_list_path=None,
        depth_folder: Optional[str] = None,
        mask_validity_mode: str = "binary_threshold",
        shutter_type: str = "GLOBAL",
        rs_ray_injection: bool = False,
        blur_samples: int = 1,
    ):
        self.path = path
        self.device = device
        self.split = split
        self.downsample_factor = downsample_factor
        self.ray_jitter = ray_jitter
        self.test_split_interval = test_split_interval
        self._all_exif_exposures = exif_exposures  # Exposure values for all frames (pre-split)
        self.camera_names = list(camera_names) if camera_names is not None else None
        self.camera_ids = [int(camera_id) for camera_id in camera_ids] if camera_ids is not None else None
        self.normalize_world_space = bool(normalize_world_space)
        self.gsplat_image_downscale = gsplat_image_downscale
        self.world_normalization_transform = np.eye(4, dtype=np.float32)
        self.sky_mask_folder = sky_mask_folder
        self.depth_folder = depth_folder
        self.mask_validity_mode = str(mask_validity_mode)
        if self.mask_validity_mode not in _MASK_VALIDITY_MODES:
            raise ValueError(
                "mask_validity_mode must be one of "
                f"{sorted(_MASK_VALIDITY_MODES)}, got "
                f"{self.mask_validity_mode!r}."
            )
        self.depth_paths = None
        self.train_exclude_image_list_path = train_exclude_image_list_path
        self.train_focus_image_list_path = train_focus_image_list_path
        self.holdout_image_list_path = holdout_image_list_path
        self.shutter_type = ShutterType[shutter_type].name
        self.rs_ray_injection = rs_ray_injection
        self.blur_samples = int(blur_samples)
        self.preserve_soft_training_masks = self._validate_soft_training_mask_contract()
        if self.blur_samples < 1:
            raise ValueError(f"blur_samples must be >= 1, got {self.blur_samples}.")
        if self.blur_samples > 1 and not rs_ray_injection:
            raise ValueError(
                "blur_samples > 1 requires the 3dgrt ray-injection path "
                "(rs_ray_injection=True); the UT splat path has no "
                "exposure-time sampling."
            )
        self.train_focus_image_weight = float(train_focus_image_weight)
        if self.train_focus_image_weight <= 0.0:
            raise ValueError(f"train_focus_image_weight must be positive, got {self.train_focus_image_weight}")

        # Worker-based GPU cache for multiprocessing compatibility
        self._worker_gpu_cache = {}

        # (Re)load intrinsics and extrinsics
        self.reload()

    def reload(self):
        # GPU cache of processed camera intrinsics - now per camera ID
        self.intrinsics = {}
        self.world_normalization_transform = np.eye(4, dtype=np.float32)

        # Get the scene data
        self.load_intrinsics_and_extrinsics()
        frame_indices_before_split = self._filter_cameras()

        # Build mapping from COLMAP camera_id to 0-based contiguous index
        # This is needed for post-processing which expects 0-based camera indices
        sorted_camera_ids = sorted(self.cam_intrinsics.keys())
        self._camera_id_to_idx = {cam_id: idx for idx, cam_id in enumerate(sorted_camera_ids)}
        physical_camera_keys = sorted({self._post_processing_camera_key(extr) for extr in self.cam_extrinsics})
        self._post_processing_camera_key_to_idx = {key: idx for idx, key in enumerate(physical_camera_keys)}

        self.n_frames = len(self.cam_extrinsics)
        self.load_camera_data()
        if self.normalize_world_space:
            self._apply_world_space_normalization()
        indices = np.arange(self.n_frames)

        # A name-based holdout list takes precedence over the positional
        # test_split_interval. The COLMAP reader sorts frames by name, so a
        # positional (index % interval) split cannot reproduce a designated
        # train.txt/test.txt holdout; matching by name is order-independent.
        # val == exactly the held-out frames, train == everything else.
        holdout_names = self.load_holdout_image_names()
        if holdout_names:
            in_holdout = np.array(
                [extr.name in holdout_names for extr in self.cam_extrinsics],
                dtype=bool,
            )
            n_held = int(in_holdout.sum())
            # Fail loud on a silent leak: the holdout must match EXACTLY (every
            # listed name present in the model). A partial match (e.g. a
            # .png-suffix / basename mismatch on some frames) would otherwise
            # silently leak one side into the other and shrink the eval set.
            if n_held != len(holdout_names):
                raise ValueError(
                    f"holdout list has {len(holdout_names)} names but matched "
                    f"{n_held} of {len(self.cam_extrinsics)} COLMAP frames -- "
                    "a name mismatch would silently leak/shrink the eval "
                    "(check the .png suffix / basename)."
                )
            logger.info(
                f"[holdout] split={self.split}: {n_held}/" f"{len(self.cam_extrinsics)} frames held out by name"
            )
            indices = in_holdout if self.split != "train" else ~in_holdout
        # If test_split_interval is set, every test_split_interval frame will be excluded from the training set
        # If test_split_interval is non-positive, all images will be used for training and testing
        elif self.test_split_interval > 0:
            if self.split == "train":
                indices = np.mod(indices, self.test_split_interval) != 0
            else:
                indices = np.mod(indices, self.test_split_interval) == 0

        self.cam_extrinsics = [self.cam_extrinsics[i] for i in np.where(indices)[0]]
        self.poses = self.poses[indices].astype(np.float32)
        if self.poses_end is not None:
            self.poses_end = self.poses_end[indices].astype(np.float32)

        # numpy str array of image paths and mask paths
        self.image_paths = self.image_paths[indices]
        self.mask_paths = self.mask_paths[indices]
        if self.sky_mask_paths is not None:
            self.sky_mask_paths = self.sky_mask_paths[indices]
        if self.depth_paths is not None:
            self.depth_paths = self.depth_paths[indices]

        self.camera_centers = self.camera_centers[indices]

        # Apply split indices to EXIF exposures
        if self._all_exif_exposures is not None:
            frame_exif_exposures = [self._all_exif_exposures[i] for i in frame_indices_before_split]
            self.exif_exposures: Optional[list[Optional[float]]] = [
                frame_exif_exposures[i] for i in np.where(indices)[0]
            ]
        else:
            self.exif_exposures = None

        if self.split == "train":
            self.apply_train_exclude_image_list()

        # Train-only exclusions must be applied before scene normalization.
        # Otherwise embargoed or outer-holdout camera poses influence the
        # scene extent and position learning-rate scale of an inner-fold run.
        self.center, self.length_scale, self.scene_bbox = (
            self.compute_spatial_extents()
        )
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

        # Clear existing worker caches to force recreation with new intrinsics
        self._worker_gpu_cache.clear()

    def load_holdout_image_names(self):
        if not self.holdout_image_list_path:
            return set()
        if not os.path.exists(self.holdout_image_list_path):
            raise FileNotFoundError(f"Holdout image list not found: {self.holdout_image_list_path}")
        holdout = set()
        with open(self.holdout_image_list_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    holdout.add(stripped)
        return holdout

    def load_train_exclude_image_names(self):
        if not self.train_exclude_image_list_path:
            return set()
        if not os.path.exists(self.train_exclude_image_list_path):
            raise FileNotFoundError(f"Train exclude image list not found: {self.train_exclude_image_list_path}")
        excluded = set()
        with open(self.train_exclude_image_list_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    excluded.add(stripped)
        return excluded

    def apply_train_exclude_image_list(self):
        excluded = self.load_train_exclude_image_names()
        if not excluded:
            return
        image_names = np.array(
            [os.path.basename(path) for path in self.image_paths],
            dtype=object,
        )
        keep = np.array([name not in excluded for name in image_names], dtype=bool)
        if np.all(keep):
            logger.warning(f"Train exclude image list matched no images: {self.train_exclude_image_list_path}")
            return
        dropped = int(np.count_nonzero(~keep))
        logger.info(f"Excluded {dropped} train images from {self.train_exclude_image_list_path}")
        self.cam_extrinsics = [extrinsic for extrinsic, keep_item in zip(self.cam_extrinsics, keep) if keep_item]
        self.poses = self.poses[keep].astype(np.float32)
        if self.poses_end is not None:
            self.poses_end = self.poses_end[keep].astype(np.float32)
        self.image_paths = self.image_paths[keep]
        self.mask_paths = self.mask_paths[keep]
        if self.sky_mask_paths is not None:
            self.sky_mask_paths = self.sky_mask_paths[keep]
        if self.depth_paths is not None:
            self.depth_paths = self.depth_paths[keep]
        self.camera_centers = self.camera_centers[keep]
        if self.exif_exposures is not None:
            self.exif_exposures = [exposure for exposure, keep_item in zip(self.exif_exposures, keep) if keep_item]

    def _load_points_for_world_normalization(self) -> np.ndarray:
        points_candidates = [
            (os.path.join(self.path, "sparse/0", "points3D.bin"), read_colmap_points3D_binary),
            (os.path.join(self.path, "sparse/0", "points3D.txt"), read_colmap_points3D_text),
            (os.path.join(self.path, "colmap", "points3D.txt"), read_colmap_points3D_text),
        ]
        for points_file, reader in points_candidates:
            if os.path.isfile(points_file):
                points, _, _ = reader(points_file)
                if points.shape[0] == 0:
                    raise ValueError(f"COLMAP point file is empty: {points_file}")
                return points.astype(np.float64)
        raise FileNotFoundError(f"Could not find COLMAP points3D file under {self.path}")

    def _apply_world_space_normalization(self) -> None:
        points = self._load_points_for_world_normalization()
        camtoworlds, _, transform = normalize_world_space(self.poses, points)

        self.poses = camtoworlds.astype(np.float32)
        self.camera_centers = self.poses[:, :3, 3].copy()
        self.cameras_extent = scene_scale(self.poses) * 1.1
        self.world_normalization_transform = transform.astype(np.float32)
        logger.info(
            "Applied GSplat-style COLMAP world normalization "
            f"(scene_extent={self.cameras_extent:.6f}, split={self.split})"
        )

    def load_intrinsics_and_extrinsics(self):
        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

        # Optional per-frame shutter END poses for rolling-shutter studies.
        # sparse/0/images_end.txt uses the images.txt format and holds the
        # END (t=+0.5) pose per frame, keyed by image name; absent -> global.
        end_file = os.path.join(self.path, "sparse/0", "images_end.txt")
        if os.path.exists(end_file):
            self._end_pose_by_name = {e.name: e for e in read_colmap_extrinsics_text(end_file)}
        else:
            self._end_pose_by_name = None

        # Exposure-trajectory blur sampling: sparse/0/pose_knots.json holds
        # per-frame camera-to-world knot poses on stamps relative to each
        # frame's capture stamp. K (start, end) pose pairs per frame are
        # precomputed here — sample k starts at offset (k+0.5)/K * t_exp and,
        # under a ROLLING shutter, ends one readout later — and the fetch
        # path turns each pair into one RS ray bundle ([K, H, W, 3] total);
        # the trainer averages the K renders before any loss.
        self._blur_pose_pairs = None
        if self.blur_samples > 1:
            knots_file = os.path.join(self.path, "sparse/0", "pose_knots.json")
            if not os.path.exists(knots_file):
                raise FileNotFoundError(
                    f"blur_samples={self.blur_samples} requires {knots_file}; "
                    "emit it with `hilti_ingest colmapize` (pose-knot sidecar)."
                )
            with open(knots_file, "r", encoding="utf-8") as f:
                knots = json.load(f)
            readout_s = float(knots["readout_s"])
            rel_stamps = np.asarray(knots["knot_stamps_rel_s"], dtype=np.float64)
            rolling = self.shutter_type != ShutterType.GLOBAL.name
            self._blur_pose_pairs = {}
            for name, frame in knots["frames"].items():
                t_exp = float(frame["t_exp_s"])
                offsets = (np.arange(self.blur_samples, dtype=np.float64) + 0.5) / self.blur_samples * t_exp
                last_query = offsets[-1] + (readout_s if rolling else 0.0)
                if last_query > rel_stamps[-1]:
                    raise ValueError(
                        f"pose_knots.json span [0, {rel_stamps[-1]:.4f}]s does "
                        f"not cover sample offset {last_query:.4f}s for "
                        f"'{name}'; re-emit knots with a larger t_exp margin."
                    )
                slerp = Slerp(
                    rel_stamps,
                    Rotation.from_quat(np.asarray(frame["c2w_q_xyzw"], dtype=np.float64)),
                )
                translations = np.asarray(frame["c2w_t"], dtype=np.float64)
                starts = _interp_c2w_knots(slerp, rel_stamps, translations, offsets)
                ends = _interp_c2w_knots(slerp, rel_stamps, translations, offsets + readout_s) if rolling else starts
                self._blur_pose_pairs[name] = torch.tensor(np.stack([starts, ends], axis=1), dtype=torch.float32)

    def _camera_names_by_id(self) -> dict[int, str]:
        sorted_camera_ids = sorted(self.cam_intrinsics.keys())
        camera_id_to_idx = {camera_id: idx for idx, camera_id in enumerate(sorted_camera_ids)}
        names: dict[int, str] = {camera_id: f"camera_{camera_id_to_idx[camera_id]}" for camera_id in sorted_camera_ids}

        for extr in self.cam_extrinsics:
            parent_folder = os.path.dirname(extr.name)
            if parent_folder:
                names[extr.camera_id] = parent_folder

        return names

    def _filter_cameras(self) -> list[int]:
        selected_camera_ids = set(self.cam_intrinsics.keys())
        names_by_id = self._camera_names_by_id()

        if self.camera_ids is not None:
            requested_ids = set(self.camera_ids)
            missing_ids = sorted(requested_ids - selected_camera_ids)
            if missing_ids:
                available_ids = sorted(selected_camera_ids)
                raise ValueError(f"COLMAP camera_ids {missing_ids} not found. Available camera_ids: {available_ids}")
            selected_camera_ids &= requested_ids

        if self.camera_names is not None:
            name_to_ids: dict[str, set[int]] = {}
            for camera_id, camera_name in names_by_id.items():
                name_to_ids.setdefault(camera_name, set()).add(camera_id)

            requested_names = set(self.camera_names)
            missing_names = sorted(requested_names - set(name_to_ids))
            if missing_names:
                available_names = sorted(name_to_ids)
                raise ValueError(
                    f"COLMAP camera_names {missing_names} not found. Available camera_names: {available_names}"
                )

            selected_camera_ids &= set().union(*(name_to_ids[camera_name] for camera_name in requested_names))

        if not selected_camera_ids:
            raise ValueError("COLMAP camera selection is empty.")

        frame_indices = [
            frame_idx for frame_idx, extr in enumerate(self.cam_extrinsics) if extr.camera_id in selected_camera_ids
        ]
        if not frame_indices:
            selected_names = [names_by_id[camera_id] for camera_id in sorted(selected_camera_ids)]
            raise ValueError(f"COLMAP camera selection {selected_names} has no frames.")

        if self.camera_names is not None or self.camera_ids is not None:
            selected_names = [names_by_id[camera_id] for camera_id in sorted(selected_camera_ids)]
            logger.info(
                f"Using COLMAP cameras: {selected_names} "
                f"(camera_ids={sorted(selected_camera_ids)}, frames={len(frame_indices)})"
            )

        self.cam_extrinsics = [self.cam_extrinsics[i] for i in frame_indices]
        self.cam_intrinsics = {
            camera_id: intr for camera_id, intr in self.cam_intrinsics.items() if camera_id in selected_camera_ids
        }
        return frame_indices

    def get_images_folder(self):
        downsample_suffix = "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        return f"images{downsample_suffix}"

    def get_masks_folder(self):
        downsample_suffix = "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        return f"masks{downsample_suffix}"

    def get_sky_masks_folder(self):
        if self.sky_mask_folder is not None:
            return self.sky_mask_folder
        downsample_suffix = "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        return f"sky_masks{downsample_suffix}"

    def resolve_mask_path(self, image_path, image_name):
        colmap_mask_path = os.path.join(self.path, self.get_masks_folder(), image_name)
        if os.path.exists(colmap_mask_path):
            return colmap_mask_path
        return os.path.splitext(image_path)[0] + "_mask.png"

    def _validate_soft_training_mask_contract(self) -> bool:
        contract_path = os.path.join(
            self.path,
            _SOFT_MASK_CONTRACT_FILENAME,
        )
        if not os.path.isfile(contract_path):
            return False
        with open(contract_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("mask_semantics") != _SOFT_MASK_SEMANTICS
            or payload.get("intended_split") != "training_only"
        ):
            raise ValueError(
                "Invalid soft training-mask contract: "
                f"{contract_path}"
            )
        return True

    @staticmethod
    def _normalize_training_mask(
        mask: torch.Tensor,
        *,
        preserve_soft_weights: bool,
        validity_mode: str,
    ) -> torch.Tensor:
        normalized = mask / 255.0
        if preserve_soft_weights:
            return normalized.to(torch.float32)
        if validity_mode == "nonzero":
            return (mask > 0).to(torch.float32)
        return (normalized > 0.5).to(torch.float32)

    def resolve_sky_mask_path(self, image_name):
        return os.path.join(self.path, self.get_sky_masks_folder(), image_name)

    def resolve_depth_path(self, image_name):
        if self.depth_folder is None:
            return ""
        depth_stem = os.path.splitext(image_name)[0] + ".npy"
        direct = os.path.join(self.path, self.depth_folder, depth_stem)
        if os.path.exists(direct):
            return direct
        flat_stem = os.path.splitext(os.path.basename(image_name))[0]
        camera_name, separator, frame_name = flat_stem.rpartition("_")
        if separator and camera_name and frame_name.isdigit():
            candidate = os.path.join(
                self.path,
                self.depth_folder,
                camera_name,
                "depth",
                f"{frame_name}.npy",
            )
            if os.path.exists(candidate):
                return candidate
        parts = image_name.replace("\\", "/").split("/")
        if len(parts) >= 2:
            camera_name = parts[-2]
            frame_name = os.path.splitext(parts[-1])[0] + ".npy"
            candidate = os.path.join(
                self.path,
                self.depth_folder,
                camera_name,
                "depth_zext",
                frame_name,
            )
            if os.path.exists(candidate):
                return candidate
        return direct

    def load_b2g_camera_rotations(self):
        metadata_path = os.path.join(self.path, "b2g_camera_models.json")
        if not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        cameras = metadata.get("cameras", {})
        return {
            int(camera_id): int(camera.get("image_rotation_quadrants_cw", 0)) % 4
            for camera_id, camera in cameras.items()
        }

    def load_camera_data(self):
        """
        Load the camera data and generate rays for each camera.
        This function is called on CPU for multiprocessing compatibility
        GPU tensors will be created per-worker as needed
        """
        self._camera_data_params = {}
        self._store_camera_params_cpu()

    def _store_camera_params_cpu(self):
        """Store camera parameters on CPU for multiprocessing compatibility."""

        image_dir = os.path.join(self.path, self.get_images_folder())
        colmap_to_image = {}
        if self.gsplat_image_downscale:
            colmap_image_dir = os.path.join(self.path, "images")
            for directory in [image_dir, colmap_image_dir]:
                if not os.path.exists(directory):
                    raise ValueError(f"Image folder {directory} does not exist.")

            colmap_files = sorted(_get_relative_paths(colmap_image_dir))
            image_files = sorted(_get_relative_paths(image_dir))
            if self.downsample_factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
                image_dir = _resize_image_folder(
                    colmap_image_dir,
                    image_dir + "_png",
                    factor=self.downsample_factor,
                )
                image_files = sorted(_get_relative_paths(image_dir))
            colmap_to_image = dict(zip(colmap_files, image_files))

        def create_pinhole_camera(focalx, focaly, w, h, cx=None, cy=None):
            cx = cx if cx is not None else w / 2.0
            cy = cy if cy is not None else h / 2.0
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([cx, cy], dtype=np.float32),
                focal_length=np.array([focalx, focaly], dtype=np.float32),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, focalx, focaly, w, h, self.ray_jitter, cx=cx, cy=cy)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        def create_opencv_pinhole_camera(
            focal_length,
            principal_point,
            w,
            h,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
        ):
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([w, h], dtype=np.uint64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.asarray(principal_point, dtype=np.float32),
                focal_length=np.asarray(focal_length, dtype=np.float32),
                radial_coeffs=np.asarray(radial_coeffs, dtype=np.float32),
                tangential_coeffs=np.asarray(tangential_coeffs, dtype=np.float32),
                thin_prism_coeffs=np.asarray(thin_prism_coeffs, dtype=np.float32),
            )
            camera_model = ncore.sensors.CameraModel.from_parameters(params, device="cpu", dtype=torch.float32)
            int_pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
            image_points = camera_model.pixels_to_image_points(int_pixel_coords)
            rays_d_cam = camera_model.image_points_to_camera_rays(image_points)
            rays_o_cam = torch.zeros_like(rays_d_cam)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        def create_fisheye_camera(params, w, h):
            # Generate UV coordinates
            u = np.tile(np.arange(w), h)
            v = np.arange(h).repeat(w)
            out_shape = (1, h, w, 3)
            resolution = np.array([w, h]).astype(np.uint64)
            principal_point = params[2:4].astype(np.float32)
            focal_length = params[0:2].astype(np.float32)
            radial_coeffs = params[4:].astype(np.float32)
            # Largest valid view angle, kept inside the KB4 forward map's
            # monotonic (invertible) domain. The legacy `max_radius / focal`
            # estimate inverts the image-corner radius with an equidistant
            # approximation and lands far past the polynomial's turning point
            # for wide 360 lenses, poisoning ncore's backward-poly fit and
            # shrinking/veiling the rendered fisheye periphery.
            max_angle = compute_fisheye_max_angle(
                resolution.astype(np.float64),
                principal_point.astype(np.float64),
                focal_length.astype(np.float64),
                radial_coeffs.astype(np.float64),
            )

            params = OpenCVFisheyeCameraModelParameters(
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                resolution=resolution,
                max_angle=max_angle,
                # Inherit the configured shutter (dataset.shutter_type) so a
                # ROLLING fisheye scene consumes images_end.txt; the other
                # camera creators keep GLOBAL until they need RS support.
                shutter_type=ShutterType[self.shutter_type],
            )
            camera_model = ncore.sensors.CameraModel.from_parameters(params, device="cpu", dtype=torch.float32)
            int_pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32)
            image_points = camera_model.pixels_to_image_points(int_pixel_coords)
            rays_d_cam = camera_model.image_points_to_camera_rays(image_points)
            rays_o_cam = torch.zeros_like(rays_d_cam)
            pixel_coords = create_pixel_coords(w, h)
            return (
                params.to_dict(),
                rays_o_cam.to(torch.float32).reshape(out_shape),
                rays_d_cam.to(torch.float32).reshape(out_shape),
                type(params).__name__,
                pixel_coords,
            )

        def rational_project_xy(x, y, params):
            fx, fy, cx, cy = params[:4]
            b1, b2, b3 = params[4:7]
            d1, d2, d3 = params[7:10]
            a1, a2 = params[10:12]
            p1, p2 = params[12:14]
            skew = params[14]
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r4 * r2
            numerator = 1.0 + b1 * r2 + b2 * r4 + b3 * r6
            denominator = 1.0 + d1 * r2 + d2 * r4 + d3 * r6
            denominator = np.where(np.abs(denominator) > 1e-12, denominator, 1.0)
            radial = numerator / denominator
            affine = 1.0 + a1 * r2 + a2 * r4
            x_distorted = x * radial + affine * (p2 * (r2 + 2.0 * x * x) + 2.0 * p1 * x * y)
            y_distorted = y * radial + affine * (p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)
            u = fx * x_distorted + skew * y_distorted + cx
            v = fy * y_distorted + cy
            return u, v

        def invert_rational_pixels(u, v, params, num_iterations=8):
            fx, fy, cx, cy = params[:4]
            skew = params[14]
            y_distorted = (v - cy) / fy
            x_distorted = (u - cx - skew * y_distorted) / fx
            x = x_distorted.copy()
            y = y_distorted.copy()
            eps = 1e-5

            for _ in range(num_iterations):
                pred_u, pred_v = rational_project_xy(x, y, params)
                residual_u = pred_u - u
                residual_v = pred_v - v

                pred_u_dx, pred_v_dx = rational_project_xy(x + eps, y, params)
                pred_u_dy, pred_v_dy = rational_project_xy(x, y + eps, params)
                j00 = (pred_u_dx - pred_u) / eps
                j10 = (pred_v_dx - pred_v) / eps
                j01 = (pred_u_dy - pred_u) / eps
                j11 = (pred_v_dy - pred_v) / eps
                determinant = j00 * j11 - j01 * j10
                valid = np.abs(determinant) > 1e-12
                delta_x = np.zeros_like(x)
                delta_y = np.zeros_like(y)
                delta_x[valid] = (j11[valid] * residual_u[valid] - j01[valid] * residual_v[valid]) / determinant[valid]
                delta_y[valid] = (-j10[valid] * residual_u[valid] + j00[valid] * residual_v[valid]) / determinant[valid]
                x -= delta_x
                y -= delta_y
            return x, y

        def stored_to_native_pixels(u_stored, v_stored, w, h, rotation):
            if rotation == 1:
                return v_stored, (w - 1.0) - u_stored
            if rotation == 2:
                return (w - 1.0) - u_stored, (h - 1.0) - v_stored
            if rotation == 3:
                return (h - 1.0) - v_stored, u_stored
            return u_stored, v_stored

        def native_resolution(w, h, rotation):
            if rotation % 2 == 1:
                return h, w
            return w, h

        def create_rational_camera(params, w, h, rotation):
            u_stored = np.tile(np.arange(w) + 0.5, h)
            v_stored = (np.arange(h) + 0.5).repeat(w)
            u_native, v_native = stored_to_native_pixels(u_stored, v_stored, w, h, rotation)
            native_w, native_h = native_resolution(w, h, rotation)
            out_shape = (1, h, w, 3)
            x, y = invert_rational_pixels(u_native, v_native, params)
            ray_lookat = np.stack((x, y, np.ones_like(x)), axis=-1)
            rays_d_cam = ray_lookat / np.linalg.norm(ray_lookat, axis=-1, keepdims=True)
            rays_o_cam = np.zeros_like(rays_d_cam)
            pixel_coords = create_pixel_coords(w, h)
            params_dict = {
                "resolution": np.array([w, h], dtype=np.uint64),
                "native_resolution": np.array([native_w, native_h], dtype=np.uint64),
                "image_rotation_quadrants_cw": int(rotation),
                "shutter_type": ShutterType.GLOBAL.name,
                "principal_point": params[2:4].astype(np.float32),
                "focal_length": params[:2].astype(np.float32),
                "numerator_coeffs": params[4:7].astype(np.float32),
                "denominator_coeffs": params[7:10].astype(np.float32),
                "affine_coeffs": params[10:12].astype(np.float32),
                "tangential_coeffs": params[12:14].astype(np.float32),
                "skew": float(params[14]),
            }
            return (
                params_dict,
                torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32).reshape(out_shape),
                "RationalCameraModelParameters",
                pixel_coords,
            )

        def create_equirect_camera(w, h):
            # Full-sphere equirectangular rays in the [right, down, forward]
            # camera frame (matches the CUDA projectPoint and the COLMAP poses
            # loaded without a coordinate flip). Parameter-free beyond (w, h).
            out_shape = (1, h, w, 3)
            cols = np.tile(np.arange(w) + 0.5, h)
            rows = (np.arange(h) + 0.5).repeat(w)
            phi = (1.0 - 2.0 * cols / w) * np.pi
            theta = (1.0 - 2.0 * rows / h) * (np.pi / 2.0)
            cos_theta = np.cos(theta)
            rays = np.stack(
                [
                    -cos_theta * np.sin(phi),
                    -np.sin(theta),
                    cos_theta * np.cos(phi),
                ],
                axis=-1,
            )
            rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
            rays_o_cam = np.zeros_like(rays)
            pixel_coords = create_pixel_coords(w, h)
            params_dict = {
                "resolution": np.array([w, h], dtype=np.uint64),
                "shutter_type": self.shutter_type,
            }
            return (
                params_dict,
                torch.tensor(rays_o_cam, dtype=torch.float32).reshape(out_shape),
                torch.tensor(rays, dtype=torch.float32).reshape(out_shape),
                "EquirectCameraModelParameters",
                pixel_coords,
            )

        cam_id_to_image_name = {extr.camera_id: extr.name for extr in self.cam_extrinsics}
        b2g_camera_rotations = self.load_b2g_camera_rotations()

        for intr in self.cam_intrinsics.values():
            full_width = intr.width
            full_height = intr.height

            image_name = cam_id_to_image_name[intr.id]
            if self.gsplat_image_downscale:
                image_path = os.path.join(image_dir, colmap_to_image[image_name])
            else:
                image_name = (
                    os.path.join(os.path.split(image_name)[1], "")
                    if self.get_images_folder() in image_name
                    else image_name
                )
                image_path = os.path.join(self.path, self.get_images_folder(), image_name)

            try:
                # Load the image to get its actual dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
            except FileNotFoundError:
                logger.error(f"Image {image_path} not found. Cannot determine dimensions for intrinsic ID {intr.id}.")
                continue

            if self.gsplat_image_downscale:
                scaling_factor = self.downsample_factor
                image_scale = np.array(
                    [width / (full_width // scaling_factor), height / (full_height // scaling_factor)],
                    dtype=np.float32,
                )
            else:
                # Calculate scaling factor to match the image dimensions to the intrinsic dimensions
                scaling_factor = int(round(intr.height / height))
                expected_size = f"{full_width / scaling_factor}x{full_height / scaling_factor}"
                assert (
                    abs(full_width / scaling_factor - width) <= 1
                ), f"Scaled image dimension {expected_size} (factor {scaling_factor}x) does not match the actual image dimensions {width}x{height}"
                assert (
                    abs(full_height / scaling_factor - height) <= 1
                ), f"Scaled image dimension {expected_size} (factor {scaling_factor}x) does not match the actual image dimensions {width}x{height}"

            if intr.model == "SIMPLE_PINHOLE":
                focal_length = intr.params[0] / scaling_factor
                cx = intr.params[1] / scaling_factor
                cy = intr.params[2] / scaling_factor
                if self.gsplat_image_downscale:
                    self.intrinsics[intr.id] = create_pinhole_camera(
                        focal_length * image_scale[0],
                        focal_length * image_scale[1],
                        width,
                        height,
                        cx=cx * image_scale[0],
                        cy=cy * image_scale[1],
                    )
                else:
                    self.intrinsics[intr.id] = create_pinhole_camera(
                        focal_length, focal_length, width, height, cx=cx, cy=cy
                    )

            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0] / scaling_factor
                focal_length_y = intr.params[1] / scaling_factor
                cx = intr.params[2] / scaling_factor
                cy = intr.params[3] / scaling_factor
                if self.gsplat_image_downscale:
                    focal_length_x *= image_scale[0]
                    focal_length_y *= image_scale[1]
                    cx *= image_scale[0]
                    cy *= image_scale[1]
                self.intrinsics[intr.id] = create_pinhole_camera(
                    focal_length_x, focal_length_y, width, height, cx=cx, cy=cy
                )

            elif intr.model in {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"}:
                focal_length, principal_point, radial_coeffs, tangential_coeffs, thin_prism_coeffs = (
                    _opencv_pinhole_intrinsics_from_colmap(intr.model, intr.params, scaling_factor)
                )
                if self.gsplat_image_downscale:
                    focal_length *= image_scale
                    principal_point *= image_scale
                self.intrinsics[intr.id] = create_opencv_pinhole_camera(
                    focal_length,
                    principal_point,
                    width,
                    height,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs,
                )

            elif intr.model == "OPENCV_FISHEYE":
                params = copy.deepcopy(intr.params)
                params[:4] = params[:4] / scaling_factor
                if self.gsplat_image_downscale:
                    params[:4] *= image_scale[[0, 1, 0, 1]]
                self.intrinsics[intr.id] = create_fisheye_camera(params, width, height)

            elif intr.model == "RATIONAL":
                params = copy.deepcopy(intr.params)
                params[:4] = params[:4] / scaling_factor
                rotation = b2g_camera_rotations.get(intr.id, 0)
                self.intrinsics[intr.id] = create_rational_camera(params, width, height, rotation)

            elif intr.model in (
                "EQUIRECTANGULAR",
                "SPHERICAL",
                "OPENCV_SPHERICAL",
            ):
                self.intrinsics[intr.id] = create_equirect_camera(width, height)

            else:
                assert False, (
                    f"Colmap camera model '{intr.model}' not handled: supported camera models are "
                    "PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, OPENCV_FISHEYE, "
                    "RATIONAL, EQUIRECTANGULAR, SPHERICAL, and OPENCV_SPHERICAL."
                )

        # Load poses and paths
        self.poses = []
        self.image_paths = []
        self.mask_paths = []
        self.sky_mask_paths = [] if self.sky_mask_folder is not None else None
        self.depth_paths = [] if self.depth_folder is not None else None
        self.poses_end = [] if self._end_pose_by_name is not None else None

        cam_centers = []
        for extr in logger.track(
            self.cam_extrinsics,
            description=f"Load Dataset ({self.split})",
            color="salmon1",
        ):
            R = qvec_to_so3(extr.qvec)
            T = np.array(extr.tvec)
            W2C = np.zeros((4, 4), dtype=np.float32)
            W2C[:3, 3] = T
            W2C[:3, :3] = R
            W2C[3, 3] = 1.0
            C2W = np.linalg.inv(W2C)
            self.poses.append(C2W)
            cam_centers.append(C2W[:3, 3])

            if self.poses_end is not None:
                end_extr = self._end_pose_by_name.get(extr.name)
                if end_extr is None:
                    raise ValueError(f"images_end.txt has no END pose for '{extr.name}'.")
                W2C_end = np.zeros((4, 4), dtype=np.float32)
                W2C_end[:3, 3] = np.array(end_extr.tvec)
                W2C_end[:3, :3] = qvec_to_so3(end_extr.qvec)
                W2C_end[3, 3] = 1.0
                self.poses_end.append(np.linalg.inv(W2C_end))

            if self.gsplat_image_downscale:
                image_path = os.path.join(image_dir, colmap_to_image[extr.name])
            else:
                image_path = os.path.join(self.path, self.get_images_folder(), extr.name)
            self.image_paths.append(image_path)
            self.mask_paths.append(self.resolve_mask_path(image_path, extr.name))
            if self.sky_mask_paths is not None:
                self.sky_mask_paths.append(self.resolve_sky_mask_path(extr.name))
            if self.depth_paths is not None:
                self.depth_paths.append(self.resolve_depth_path(extr.name))

        self.camera_centers = np.array(cam_centers)
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.poses = np.stack(self.poses)
        if self.poses_end is not None:
            self.poses_end = np.stack(self.poses_end)

        self.image_paths = np.stack(self.image_paths, dtype=str)
        self.mask_paths = np.stack(self.mask_paths, dtype=str)
        if self.sky_mask_paths is not None:
            self.sky_mask_paths = np.stack(self.sky_mask_paths, dtype=str)
        if self.depth_paths is not None:
            self.depth_paths = np.stack(self.depth_paths, dtype=str)

    def _lazy_worker_intrinsics_cache(self):
        """Create intrinsics cache for a specific worker."""
        worker_id = get_worker_id()

        # Check if this worker already has cached tensors
        if worker_id not in self._worker_gpu_cache:
            # For now, fall back to the original approach for each worker
            # This ensures each worker creates its own GPU tensors
            worker_intrinsics = {}
            for intr_id, (
                params_dict,
                rays_ori,
                rays_dir,
                camera_name,
                pixel_coords,
            ) in self.intrinsics.items():
                # Create new GPU tensors for this worker
                worker_rays_ori = rays_ori.to(self.device, non_blocking=True)
                worker_rays_dir = rays_dir.to(self.device, non_blocking=True)
                worker_pixel_coords = pixel_coords.to(self.device, non_blocking=True)
                worker_intrinsics[intr_id] = (
                    params_dict,
                    worker_rays_ori,
                    worker_rays_dir,
                    camera_name,
                    worker_pixel_coords,
                )
            self._worker_gpu_cache[worker_id] = worker_intrinsics

        return self._worker_gpu_cache[worker_id]

    @torch.no_grad()
    def compute_spatial_extents(self):
        camera_origins = torch.FloatTensor(self.poses[:, :3, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)  # mean distance between of cameras from center
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale

    def get_center(self):
        return self.center

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self):
        return self.cameras_extent

    def get_observer_points(self):
        return self.camera_centers

    def get_world_normalization_transform(self) -> np.ndarray:
        return self.world_normalization_transform.copy()

    def get_image_names(self) -> list[str]:
        """Return image identities after holdout and train exclusions."""
        return [extrinsic.name for extrinsic in self.cam_extrinsics]

    def get_poses(self) -> np.ndarray:
        """Get camera poses as 4x4 transformation matrices.

        COLMAP Dataset Implementation:
        COLMAP naturally provides poses in a coordinate system compatible with
        3DGRUT's "right down front" convention, so no coordinate conversion is needed.

        The poses are constructed from COLMAP's world-to-camera matrices by:
        1. Building W2C from rotation (qvec_to_so3) and translation (tvec)
        2. Inverting to get camera-to-world: C2W = inv(W2C)

        Returns:
            np.ndarray: Camera poses with shape (N, 4, 4) in "right down front" convention
        """
        return self.poses

    def get_intrinsics_idx(self, extr_idx: int):
        return self.cam_extrinsics[extr_idx].camera_id

    def get_camera_idx(self, frame_idx: int) -> int:
        """Return 0-based camera index for a given frame index.

        Maps from COLMAP's potentially non-contiguous camera_id to a
        0-based contiguous index.
        """
        colmap_camera_id = self.cam_extrinsics[frame_idx].camera_id
        return self._camera_id_to_idx[colmap_camera_id]

    @staticmethod
    def _post_processing_camera_key(extrinsic) -> str:
        """Return the physical camera key for appearance post-processing."""
        normalized_name = extrinsic.name.replace("\\", "/")
        parent, path_separator, basename = normalized_name.rpartition("/")
        stem = basename.rsplit(".", 1)[0]
        camera_name, separator, frame_token = stem.rpartition("_")
        if separator and camera_name in _CANONICAL_FLAT_PHYSICAL_CAMERA_NAMES and frame_token.isdigit():
            return camera_name
        if path_separator and parent:
            parent_camera_name = parent.rsplit("/", 1)[-1]
            if parent_camera_name in _CANONICAL_FLAT_PHYSICAL_CAMERA_NAMES:
                return parent_camera_name
            return parent
        return f"camera_id_{extrinsic.camera_id}"

    def get_post_processing_camera_idx(self, frame_idx: int) -> int:
        """Return the physical-camera index for PPISP-style corrections."""
        key = self._post_processing_camera_key(self.cam_extrinsics[frame_idx])
        return self._post_processing_camera_key_to_idx[key]

    def get_post_processing_frames_per_camera(self) -> list[int]:
        """Return split frame counts grouped by physical camera."""
        counts = [0] * len(self._post_processing_camera_key_to_idx)
        for extr in self.cam_extrinsics:
            key = self._post_processing_camera_key(extr)
            counts[self._post_processing_camera_key_to_idx[key]] += 1
        return counts

    def get_post_processing_camera_names(self) -> list[str]:
        """Return physical-camera names in post-processing index order."""
        ordered_keys = sorted(
            self._post_processing_camera_key_to_idx.items(),
            key=lambda item: item[1],
        )
        return [key for key, _ in ordered_keys]

    def get_frames_per_camera(self) -> list[int]:
        """Return list of frame counts per camera.

        Returns a list where index i contains the number of frames captured
        by camera i (using 0-based camera indices). Derived values:
        - num_cameras = len(frames_per_camera)
        - num_frames = sum(frames_per_camera)
        """
        num_cameras = len(self.cam_intrinsics)
        counts = [0] * num_cameras
        for extr in self.cam_extrinsics:
            camera_idx = self._camera_id_to_idx[extr.camera_id]
            counts[camera_idx] += 1
        return counts

    def get_camera_names(self) -> list[str]:
        """Return list of camera names.

        For multi-camera setups where images are organized in subfolders by camera,
        returns the folder names. For single-camera setups (images directly in images
        folder), returns default names like "camera_0".
        """
        num_cameras = len(self.cam_intrinsics)
        names: list[str | None] = [None] * num_cameras

        # Find one image path for each camera to determine folder name
        for extr in self.cam_extrinsics:
            camera_idx = self._camera_id_to_idx[extr.camera_id]
            if names[camera_idx] is not None:
                continue  # Already have a name for this camera

            # extr.name is relative path from images folder
            # e.g., "cam_front/image001.jpg" or just "image001.jpg"
            parent_folder = os.path.dirname(extr.name)
            if parent_folder:
                names[camera_idx] = parent_folder
            else:
                names[camera_idx] = f"camera_{camera_idx}"

        return names

    def __len__(self) -> int:
        return self.n_frames

    @staticmethod
    def _first_scalar(value):
        if torch.is_tensor(value):
            return value.reshape(-1)[0].item()
        if isinstance(value, np.ndarray):
            return value.reshape(-1)[0].item()
        if isinstance(value, (list, tuple)):
            return ColmapDataset._first_scalar(value[0])
        return value

    @staticmethod
    def _sequence_idx_from_path(image_path: str) -> int:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        digits = ""
        for char in reversed(stem):
            if not char.isdigit():
                break
            digits = char + digits
        if not digits:
            return -1
        return int(digits)

    @torch.cuda.nvtx.range("colmap_dataset::_getitem")
    def __getitem__(self, idx) -> dict:
        # Load image and get its actual dimensions
        image_data = _read_rgb_image_array(self.image_paths[idx])
        actual_h, actual_w = image_data.shape[:2]

        assert image_data.dtype == np.uint8, "Image data must be of type uint8"

        output_dict = {
            "data": torch.tensor(image_data).unsqueeze(0),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
            "intr": self.get_intrinsics_idx(idx),
            "camera_idx": self.get_camera_idx(idx),
            "post_processing_camera_idx": self.get_post_processing_camera_idx(idx),
            "frame_idx": idx,
            "sequence_idx": self._sequence_idx_from_path(self.image_paths[idx]),
            "image_path": self.image_paths[idx],
        }

        if self.poses_end is not None:
            output_dict["pose_end"] = torch.tensor(self.poses_end[idx]).unsqueeze(0)

        # Only add mask to dictionary if it exists
        if os.path.exists(mask_path := self.mask_paths[idx]):
            mask = torch.from_numpy(np.array(Image.open(mask_path).convert("L"))).reshape(1, actual_h, actual_w, 1)
            output_dict["mask"] = mask

        if self.sky_mask_paths is not None:
            sky_mask_path = self.sky_mask_paths[idx]
            if not os.path.exists(sky_mask_path):
                raise FileNotFoundError(f"Sky mask sidecar not found: {sky_mask_path}")
            sky_mask = torch.from_numpy(np.array(Image.open(sky_mask_path).convert("L"))).reshape(
                1, actual_h, actual_w, 1
            )
            output_dict["sky_mask"] = sky_mask

        # Metric-depth sidecar (must match the training-image resolution)
        if self.depth_paths is not None:
            depth_path = self.depth_paths[idx]
            if not os.path.exists(depth_path):
                raise FileNotFoundError(f"Depth sidecar not found: {depth_path}")
            depth = np.asarray(np.load(depth_path), dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2:
                raise ValueError(f"Depth sidecar must be HxW or HxWx1: {depth_path}")
            if depth.shape != (actual_h, actual_w):
                raise ValueError(
                    "Depth sidecar resolution must match the training image: "
                    f"{depth_path} has {depth.shape}, expected "
                    f"{(actual_h, actual_w)} for {self.image_paths[idx]}"
                )
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            output_dict["depth_gt"] = torch.from_numpy(depth).reshape(1, depth.shape[0], depth.shape[1], 1)

        # Add EXIF exposure if available for this frame
        if self.exif_exposures is not None and self.exif_exposures[idx] is not None:
            output_dict["exposure"] = torch.tensor(self.exif_exposures[idx], dtype=torch.float32)

        return output_dict

    def build_blur_bundles(self, image_name, rays_dir):
        """Build the K exposure-time ray bundles for one frame.

        Used by the train fetch path and by post-hoc parity evaluation
        (rendering a checkpoint WITH the blur forward model applied).

        Args:
            image_name: Frame key as in images.txt (e.g. ``cam0/x.jpg``).
            rays_dir: ``[1, H, W, 3]`` camera-space bearings for the
                frame's camera model.

        Returns:
            ``(rays_ori, rays_dir)`` world-space bundles, each
            ``[K, H, W, 3]``.
        """
        if self._blur_pose_pairs is None:
            raise RuntimeError(
                "build_blur_bundles needs blur_samples > 1 (no pose-knot " "pairs were loaded for this dataset)."
            )
        pairs = self._blur_pose_pairs.get(image_name)
        if pairs is None:
            raise KeyError(
                f"pose_knots.json has no entry for '{image_name}'; re-emit " "the knot sidecar for this scene."
            )
        bundle_oris = []
        bundle_dirs = []
        for k in range(pairs.shape[0]):
            ori_k, dir_k = build_rs_world_rays(
                rays_dir,
                pairs[k, 0].to(self.device),
                pairs[k, 1].to(self.device),
            )
            bundle_oris.append(ori_k)
            bundle_dirs.append(dir_k)
        return torch.cat(bundle_oris, dim=0), torch.cat(bundle_dirs, dim=0)

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU."""

        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        intr = batch["intr"][0].item()

        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        # Get intrinsics for current worker
        worker_intrinsics = self._lazy_worker_intrinsics_cache()

        camera_params_dict, rays_ori, rays_dir, camera_name, pixel_coords = worker_intrinsics[intr]

        sample = {
            "rgb_gt": data,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            f"intrinsics_{camera_name}": camera_params_dict,
            "camera_idx": int(self._first_scalar(batch["camera_idx"])),
            "post_processing_camera_idx": int(self._first_scalar(batch["post_processing_camera_idx"])),
            "frame_idx": int(self._first_scalar(batch["frame_idx"])),
            "sequence_idx": int(self._first_scalar(batch["sequence_idx"])),
            "image_path": batch["image_path"][0],
            "pixel_coords": pixel_coords,
        }

        sample["depth_ray_z"] = torch.abs(rays_dir[..., 2:3])

        if "depth_gt" in batch:
            depth_gt = batch["depth_gt"][0].to(self.device, non_blocking=True)
            sample["depth_gt"] = depth_gt

        if "pose_end" in batch:
            sample["T_to_world_end"] = batch["pose_end"][0].to(self.device, non_blocking=True)
        # A ROLLING shutter MUST carry a distinct END pose; otherwise the
        # kernel collapses to global shutter (RS1 == RS0) with no warning.
        shutter = camera_params_dict.get("shutter_type", ShutterType.GLOBAL.name)
        if shutter != ShutterType.GLOBAL.name:
            end = sample.get("T_to_world_end")
            if end is None or torch.allclose(end, pose, atol=1e-6):
                raise ValueError(
                    f"shutter_type={shutter} but T_to_world_end is missing or "
                    "equal to T_to_world; rolling-shutter training would "
                    "silently no-op to global shutter. Provide distinct "
                    "per-frame END poses via images_end.txt."
                )

        # 3dgrt RAY path: the equirect camera model lives only in the UT tracer,
        # so the ray path needs EXPLICIT per-pixel world rays. GLOBAL = single
        # pose; ROLLING = per-row interpolation (true per-pixel rolling shutter,
        # no per-Gaussian single-pose approximation). Identity world transform
        # makes the tracer's rayToWorld a pure passthrough. The UT splat path
        # keeps the camera model + shutter_type instead.
        if self.rs_ray_injection:
            if self._blur_pose_pairs is not None and self.split == "train":
                # Exposure-trajectory blur: K time-sampled RS bundles per
                # frame, stacked on dim 0 (the OptiX launch treats dim 0 as
                # depth, so the K bundles trace in one launch). The trainer
                # averages the K renders BEFORE the loss.
                name = "/".join(batch["image_path"][0].replace("\\", "/").split("/")[-2:])
                rays_ori_k, rays_dir_k = self.build_blur_bundles(name, rays_dir)
                sample["rays_ori"] = rays_ori_k
                sample["rays_dir"] = rays_dir_k
            else:
                # GLOBAL must stay global-naive: only a ROLLING shutter
                # consumes the END pose. rs_ray_injection is gated on
                # method==3dgrt (not on shutter), so without this guard an
                # images_end.txt present in scene_rs would silently feed
                # per-row END poses into a GLOBAL run too -- making RS0
                # (global-naive) identical to RS1 (rolling) and contaminating
                # the whole RS0-vs-RS1 ablation.
                rs_end = sample.get("T_to_world_end") if shutter != ShutterType.GLOBAL.name else None
                if rs_end is None:
                    rs_end = pose
                rays_ori_w, rays_dir_w = build_rs_world_rays(rays_dir, pose, rs_end)
                sample["rays_ori"] = rays_ori_w
                sample["rays_dir"] = rays_dir_w
            # Identity world transform = pure passthrough (rays are already
            # world-space). Keep the [1, 4, 4] shape the non-injected path
            # uses so downstream rayToWorld handling is unchanged.
            sample["T_to_world"] = torch.eye(4, device=self.device, dtype=pose.dtype).unsqueeze(0)
            sample["rays_in_world_space"] = True

        if "mask" in batch:
            mask = self._normalize_training_mask(
                batch["mask"][0].to(self.device, non_blocking=True),
                preserve_soft_weights=(
                    self.preserve_soft_training_masks
                    and self.split == "train"
                ),
                validity_mode=self.mask_validity_mode,
            )
            sample["mask"] = mask

        if "sky_mask" in batch:
            sky_mask = batch["sky_mask"][0].to(self.device, non_blocking=True) / 255.0
            sky_mask = (sky_mask > 0.5).to(torch.float32)
            sample["sky_mask"] = sky_mask

        # Add exposure prior from EXIF if available (move to GPU)
        if "exposure" in batch and batch["exposure"][0] is not None:
            sample["exposure"] = batch["exposure"].to(self.device)

        return Batch(**sample)

    def create_dataset_camera_visualization(self):
        """Create a visualization of the dataset cameras."""

        cam_list = []

        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # Camera convention rotation
            camera_convention_rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            # Get camera ID and corresponding intrinsics
            camera_id = self.get_intrinsics_idx(i_cam)
            intr, _, _, _, _ = self.intrinsics[camera_id]

            # Load actual image to get dimensions
            image_data = _read_rgb_image_array(self.image_paths[i_cam])
            h, w = image_data.shape[:2]

            f_w = intr["focal_length"][0]
            f_h = intr["focal_length"][1]

            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)

            assert image_data.dtype == np.uint8, "Image data must be of type uint8"
            rgb = image_data.reshape(h, w, 3) / np.float32(255.0)
            assert rgb.dtype == np.float32, f"RGB image must be float32, got {rgb.dtype}"

            cam_list.append(
                {
                    "ext_mat": trans_mat_world_to_camera,
                    "w": w,
                    "h": h,
                    "fov_w": fov_w,
                    "fov_h": fov_h,
                    "rgb_img": rgb,
                    "split": self.split,
                }
            )

        create_camera_visualization(cam_list)
