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

import hashlib
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from plyfile import PlyData

import threedgrt_tracer
import threedgrut.model.background as background
import threedgut_tracer
from threedgrut.datasets.protocols import Batch
from threedgrut.datasets.utils import read_colmap_points3D_text, read_next_bytes
from threedgrut.export import PLYExporter
from threedgrut.export.base import ExportableModel
from threedgrut.model.acquisition_appearance import (
    SH_DC_NORMALIZATION,
    AcquisitionAppearance,
    AcquisitionGaussianView,
)
from threedgrut.model.acquisition_visibility import AcquisitionVisibility
from threedgrut.model.carriers import carrier_specular_dim, initial_carrier_tail
from threedgrut.model.features import Features
from threedgrut.model.gaussian_track_acquisition import (
    GaussianTrackAcquisition,
    load_gaussian_track_ids,
)
from threedgrut.model.geometry import (
    SurfaceAlignedPCAConfig,
    k_nearest_neighbors,
    nearest_neighbor_dist_cpuKD,
    surface_aligned_pca_initialize,
)
from threedgrut.model.surface_acquisition_spline import (
    SurfaceAcquisitionSpline,
)
from threedgrut.optimizers import SelectiveAdam, VisibilityDecayedAdam
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    get_activation_function,
    get_scheduler,
    quaternion_to_so3,
    sh_degree_to_num_features,
    sh_degree_to_specular_dim,
    to_np,
    to_torch,
)
from threedgrut.utils.render import RGB2SH

SCENE_EXTENT_MIN = 1e-6
SCENE_EXTENT_MAX_SAMPLES = 1_000_000
PROTECTED_GAUSSIAN_PREFIX_VERSION = 1
PROTECTED_GEOMETRY_NAMES = ("positions", "rotation", "scale")


def tensor_prefix_sha256(
    tensor: torch.Tensor,
    prefix_count: int,
) -> str:
    """Hash a contiguous tensor prefix for immutable-geometry checks."""
    if prefix_count < 0 or prefix_count > tensor.shape[0]:
        raise ValueError(
            "Protected prefix count is outside the tensor: "
            f"{prefix_count} for shape {tuple(tensor.shape)}."
        )
    prefix = tensor[:prefix_count].detach().contiguous().cpu().numpy()
    return hashlib.sha256(prefix.tobytes()).hexdigest()


def zero_tensor_prefix_gradient(
    gradient: torch.Tensor,
    *,
    prefix_count: int,
) -> torch.Tensor:
    """Return a gradient with the protected row prefix zeroed."""
    if prefix_count == 0:
        return gradient
    if prefix_count < 0 or prefix_count > gradient.shape[0]:
        raise ValueError(
            "Protected prefix count is outside the gradient: "
            f"{prefix_count} for shape {tuple(gradient.shape)}."
        )
    masked = gradient.clone()
    masked[:prefix_count] = 0
    return masked


def validated_surface_aligned_pca_config(
    initialization_conf: DictConfig,
    points: torch.Tensor,
    observer_points: torch.Tensor,
) -> SurfaceAlignedPCAConfig | None:
    surface_conf = initialization_conf.get("surface_aligned_pca")
    if surface_conf is None or not bool(surface_conf.get("enabled", False)):
        return None

    method = str(initialization_conf.get("method", ""))
    if method != "colmap":
        raise ValueError(
            "initialization.surface_aligned_pca requires " "initialization.method=colmap; " f"got {method!r}"
        )
    if not bool(initialization_conf.get("use_observation_points", False)):
        raise ValueError(
            "initialization.surface_aligned_pca requires "
            "use_observation_points=true; local-neighbor incumbent scaling "
            "is not supported"
        )
    if (
        observer_points.ndim != 2
        or observer_points.shape[1] != 3
        or observer_points.shape[0] == 0
        or not bool(torch.isfinite(observer_points).all())
    ):
        raise ValueError(
            "initialization.surface_aligned_pca requires nonempty finite " "observer points with shape [N, 3]"
        )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "initialization.surface_aligned_pca requires COLMAP points " f"with shape [N, 3]; got {tuple(points.shape)}"
        )

    num_support_points_value = float(surface_conf.get("num_support_points", 0))
    query_chunk_size_value = float(surface_conf.get("query_chunk_size", 0))
    if not np.isfinite(num_support_points_value) or not num_support_points_value.is_integer():
        raise ValueError("surface_aligned_pca.num_support_points must be a finite integer")
    if not np.isfinite(query_chunk_size_value) or not query_chunk_size_value.is_integer():
        raise ValueError("surface_aligned_pca.query_chunk_size must be a finite integer")
    num_support_points = int(num_support_points_value)
    query_chunk_size = int(query_chunk_size_value)
    max_neighbor_radius_m = float(surface_conf.get("max_neighbor_radius_m", 0.0))
    max_normal_to_mid_ratio = float(surface_conf.get("max_normal_to_mid_ratio", 0.0))
    min_mid_to_max_ratio = float(surface_conf.get("min_mid_to_max_ratio", 0.0))
    min_mid_eigenvalue_m2 = float(surface_conf.get("min_mid_eigenvalue_m2", 0.0))
    min_thickness_ratio = float(surface_conf.get("min_thickness_ratio", 0.0))
    if num_support_points < 3:
        raise ValueError("surface_aligned_pca.num_support_points must be at least 3")
    if points.shape[0] < num_support_points:
        raise ValueError(
            "surface_aligned_pca requires at least " f"{num_support_points} COLMAP points; got {points.shape[0]}"
        )
    if query_chunk_size <= 0:
        raise ValueError("surface_aligned_pca.query_chunk_size must be positive")
    positive_parameters = (
        ("max_neighbor_radius_m", max_neighbor_radius_m),
        ("max_normal_to_mid_ratio", max_normal_to_mid_ratio),
        ("min_mid_to_max_ratio", min_mid_to_max_ratio),
        ("min_mid_eigenvalue_m2", min_mid_eigenvalue_m2),
    )
    for parameter_name, parameter_value in positive_parameters:
        if not np.isfinite(parameter_value) or parameter_value <= 0.0:
            raise ValueError(
                f"surface_aligned_pca.{parameter_name} must be finite and " f"positive; got {parameter_value}"
            )
    if max_normal_to_mid_ratio > 1.0:
        raise ValueError("surface_aligned_pca.max_normal_to_mid_ratio must be at most 1")
    if min_mid_to_max_ratio > 1.0:
        raise ValueError("surface_aligned_pca.min_mid_to_max_ratio must be at most 1")
    if not np.isfinite(min_thickness_ratio) or min_thickness_ratio <= 0.0 or min_thickness_ratio > 1.0:
        raise ValueError(
            "surface_aligned_pca.min_thickness_ratio must be finite and in " f"(0, 1]; got {min_thickness_ratio}"
        )

    expected_counts = (
        (
            "expected_point_count",
            surface_conf.get("expected_point_count"),
            points.shape[0],
        ),
        (
            "expected_observer_count",
            surface_conf.get("expected_observer_count"),
            observer_points.shape[0],
        ),
    )
    for count_name, configured_count, actual_count in expected_counts:
        if configured_count is None:
            continue
        expected_count_value = float(configured_count)
        if not np.isfinite(expected_count_value) or not expected_count_value.is_integer() or expected_count_value < 0.0:
            raise ValueError(f"surface_aligned_pca.{count_name} must be null or a " "non-negative integer")
        expected_count = int(expected_count_value)
        if expected_count != actual_count:
            raise ValueError(
                f"surface_aligned_pca {count_name} mismatch: expected " f"{expected_count}, got {actual_count}"
            )

    return SurfaceAlignedPCAConfig(
        num_support_points=num_support_points,
        max_neighbor_radius_m=max_neighbor_radius_m,
        max_normal_to_mid_ratio=max_normal_to_mid_ratio,
        min_mid_to_max_ratio=min_mid_to_max_ratio,
        min_mid_eigenvalue_m2=min_mid_eigenvalue_m2,
        min_thickness_ratio=min_thickness_ratio,
        query_chunk_size=query_chunk_size,
    )


def _acquisition_appearance_enabled(conf) -> bool:
    appearance_conf = conf.model.get("acquisition_appearance", {})
    return bool(appearance_conf.get("enabled", False))


def _acquisition_visibility_enabled(conf) -> bool:
    visibility_conf = conf.model.get("acquisition_visibility", {})
    return bool(visibility_conf.get("enabled", False))


def _surface_acquisition_spline_enabled(conf) -> bool:
    spline_conf = conf.model.get("surface_acquisition_spline", {})
    return bool(spline_conf.get("enabled", False))


def _gaussian_track_acquisition_enabled(conf) -> bool:
    track_conf = conf.model.get("gaussian_track_acquisition", {})
    return bool(track_conf.get("enabled", False))


def _sample_point_rows(points: torch.Tensor) -> torch.Tensor:
    """Return a bounded deterministic row sample for extent estimates."""
    if points.shape[0] <= SCENE_EXTENT_MAX_SAMPLES:
        return points
    step = (points.shape[0] + SCENE_EXTENT_MAX_SAMPLES - 1) // SCENE_EXTENT_MAX_SAMPLES
    return points[::step]


def _estimate_scene_extent_from_points(points: torch.Tensor) -> float:
    """Estimate a nonzero scene scale from point geometry."""
    sampled = _sample_point_rows(points.detach().float())
    finite_mask = torch.isfinite(sampled).all(dim=1)
    if not finite_mask.any():
        return 1.0

    sampled = sampled[finite_mask]
    center = sampled.median(dim=0).values
    radius = torch.linalg.norm(sampled - center[None, :], dim=1).median()
    if torch.isfinite(radius) and radius.item() > SCENE_EXTENT_MIN:
        return float(radius.item() * 1.1)

    bbox_diagonal = torch.linalg.norm(sampled.max(dim=0).values - sampled.min(dim=0).values)
    if torch.isfinite(bbox_diagonal) and bbox_diagonal.item() > SCENE_EXTENT_MIN:
        return float(bbox_diagonal.item() * 0.1)
    return 1.0


def _subsample_initial_points(
    pts: torch.Tensor,
    rgb: torch.Tensor,
    *,
    max_points: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic subset for bounded point-cloud initialization."""
    if max_points <= 0 or pts.shape[0] <= max_points:
        return pts, rgb
    rng = torch.Generator(device=pts.device).manual_seed(seed)
    idxs = torch.randperm(pts.shape[0], device=pts.device, generator=rng)[:max_points]
    return pts[idxs], rgb[idxs]


class MixtureOfGaussians(torch.nn.Module, ExportableModel):
    """ """

    @property
    def supports_static_export(self) -> bool:
        return (
            self.acquisition_appearance is None
            and self.acquisition_visibility is None
            and self.surface_acquisition_spline is None
            and self.gaussian_track_acquisition is None
        )

    @property
    def num_gaussians(self):
        return self.positions.shape[0]

    def feature_fields(self) -> list[str]:
        """Returns a list of feature field names - subclasses can override"""
        if self.feature_type == Features.Type.SH:
            return [
                "features_albedo",
                "features_specular",
            ]
        elif self.feature_type == Features.Type.NHT:
            return ["features"]
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_max_n_features(self) -> int:
        return self.max_n_features

    def get_n_active_features(self) -> int:
        return self.n_active_features

    def get_features_albedo(self) -> torch.Tensor:
        if self.feature_type == Features.Type.SH:
            return self.features_albedo
        else:
            raise AttributeError(
                f"features_albedo not available in feature_type='{self.feature_type.name.lower()}' mode"
            )

    def get_features_specular(self) -> torch.Tensor:
        if self.feature_type == Features.Type.SH:
            return self.features_specular
        else:
            raise AttributeError(
                f"features_specular not available in feature_type='{self.feature_type.name.lower()}' mode"
            )

    def get_features(self):
        if self.feature_type == Features.Type.SH:
            return torch.cat((self.features_albedo, self.features_specular), dim=1)
        elif self.feature_type == Features.Type.NHT:
            return self.features  # [N, K]
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def _specular_feature_dim(self) -> int:
        return sh_degree_to_specular_dim(self.max_n_features) + carrier_specular_dim(self.conf)

    def _with_carrier_tail(self, features_specular: torch.Tensor) -> torch.Tensor:
        """Widen SH-only specular features with freshly initialized carrier slots."""
        expected_dim = self._specular_feature_dim()
        if features_specular.shape[1] == expected_dim:
            return features_specular
        sh_dim = sh_degree_to_specular_dim(self.max_n_features)
        if carrier_specular_dim(self.conf) > 0 and features_specular.shape[1] == sh_dim:
            tail = initial_carrier_tail(
                num_gaussians=features_specular.shape[0],
                device=features_specular.device,
                dtype=features_specular.dtype,
                conf=self.conf,
            )
            return torch.cat((features_specular, tail), dim=1)
        raise ValueError(
            f"Unexpected features_specular width: got {features_specular.shape[1]}, expected {expected_dim}."
        )

    def get_scale(self, preactivation=False):
        if preactivation:
            return self.scale
        else:
            return self.scale_activation(self.scale)

    def get_rotation(self, preactivation=False):
        if preactivation:
            return self.rotation
        else:
            return self.rotation_activation(self.rotation)

    def get_density(self, preactivation=False):
        if preactivation:
            return self.density
        else:
            return self.density_activation(self.density)

    def get_covariance(self) -> torch.Tensor:
        scales = self.get_scale()

        S = torch.zeros((self.num_gaussians, 3, 3), dtype=scales.dtype, device=self.device)
        R = quaternion_to_so3(self.get_rotation())

        S[:, 0, 0] = scales[:, 0]
        S[:, 1, 1] = scales[:, 1]
        S[:, 2, 2] = scales[:, 2]

        return R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)

    def get_model_parameters(self) -> dict:
        assert self.optimizer is not None, "Optimizer need to be initialized when storing the checkpoint"

        model_params = {
            "positions": self.positions,
            "rotation": self.rotation,
            "scale": self.scale,
            "density": self.density,
            "background": self.background.state_dict(),
            # Add other attributes that we need at restore
            "n_active_features": self.n_active_features,
            "max_n_features": self.max_n_features,
            "progressive_training": self.progressive_training,
            "scene_extent": self.scene_extent,
            # Add optimizer state dict
            "optimizer": self.optimizer.state_dict(),
            "config": self.conf,
            # Feature type and dimensions
            "feature_type": self.feature_type.name.lower(),  # Store as string for serialization
            "particle_feature_dim": self.particle_feature_dim,
            "ray_feature_dim": self.ray_feature_dim,
        }

        if self.progressive_training:
            model_params["feature_dim_increase_interval"] = self.feature_dim_increase_interval
            model_params["feature_dim_increase_step"] = self.feature_dim_increase_step

        if self.feature_type == Features.Type.SH:
            model_params["features_albedo"] = self.features_albedo
            model_params["features_specular"] = self.features_specular
        elif self.feature_type == Features.Type.NHT:
            model_params["features"] = self.features

        if self.acquisition_appearance is not None:
            model_params["acquisition_appearance"] = self.acquisition_appearance.state_dict()
        if self.acquisition_visibility is not None:
            model_params["acquisition_visibility"] = self.acquisition_visibility.state_dict()
        if self.surface_acquisition_spline is not None:
            model_params["surface_acquisition_spline"] = self.surface_acquisition_spline.state_dict()
        if self.gaussian_track_acquisition is not None:
            model_params["gaussian_track_acquisition"] = self.gaussian_track_acquisition.state_dict()
        if self.protected_gaussian_count:
            actual_hashes = self._protected_geometry_hashes()
            expected_hashes = self._protected_prefix_metadata.get("geometry_sha256")
            if expected_hashes != actual_hashes:
                raise RuntimeError("Protected Gaussian prefix geometry changed before checkpoint save.")
            model_params["protected_gaussian_prefix"] = dict(self._protected_prefix_metadata)

        return model_params

    def __init__(self, conf, scene_extent=None):
        super().__init__()

        # Store config early - needed for feature type detection
        self.conf = conf
        self.scene_extent = scene_extent
        self.acquisition_appearance: AcquisitionAppearance | None = None
        self.acquisition_visibility: AcquisitionVisibility | None = None
        self.surface_acquisition_spline: SurfaceAcquisitionSpline | None = None
        self.gaussian_track_acquisition: GaussianTrackAcquisition | None = None
        # Per-attribute per-gaussian gradient L2 norms, populated each
        # training step by Trainer3DGRUT._compute_per_gaussian_grad_norms.
        # Consumed by the live GUI for grad-mode visualization. Empty in
        # inference-only loading.
        self._last_grad_norms: dict[str, torch.Tensor] = {}

        sh_degree = conf.model.progressive_training.max_n_features
        render_sph_degree = conf.render.particle_radiance_sph_degree
        if sh_degree > render_sph_degree:
            logger.warning(
                f"model.progressive_training.max_n_features ({sh_degree}) is greater than "
                f"render.particle_radiance_sph_degree ({render_sph_degree}). "
                f"Clamping max_n_features to {render_sph_degree}."
            )
            sh_degree = render_sph_degree
        specular_dim = sh_degree_to_specular_dim(sh_degree) + carrier_specular_dim(conf)
        self.positions = torch.nn.Parameter(
            torch.empty([0, 3])
        )  # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
        self.rotation = torch.nn.Parameter(
            torch.empty([0, 4])
        )  # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
        self.scale = torch.nn.Parameter(torch.empty([0, 3]))  # Anisotropic scale of each Gaussian [n_gaussians, 3]
        self.density = torch.nn.Parameter(torch.empty([0, 1]))  # Density of each Gaussian [n_gaussians, 1]

        # Feature type configuration - determine feature storage mode
        self.feature_type = Features.Type.from_string(self.conf.model.feature_type)

        primitive_type = (getattr(conf.render, "primitive_type", None) or "").lower()
        if self.feature_type == Features.Type.NHT and primitive_type == "trisurfel":
            raise ValueError(
                "Trisurfels are not supported in NHT mode. Use primitive_type 'instances' or 'icosahedron'."
            )

        if self.feature_type == Features.Type.SH:
            # Spherical harmonics mode: separate albedo and specular features
            self.features_albedo = torch.nn.Parameter(
                torch.empty([0, 3])
            )  # Feature vector of the 0th order SH coefficients [n_gaussians, 3]
            self.features_specular = torch.nn.Parameter(
                torch.empty([0, specular_dim])
            )  # Features of the higher order SH coefficients [n_gaussians, specular_dim]
            self.particle_feature_dim = 3 + specular_dim  # SH coeffs (input to tracer)
            self.ray_feature_dim = 3  # RGB output from tracer
        elif self.feature_type == Features.Type.NHT:
            # NHT: per-particle feature vector, decoder maps rendered features -> RGB
            feat = Features(conf)
            num_points = feat.num_interpolation_points
            nht_dim = int(conf.model.nht_features.dim)
            if nht_dim % num_points != 0:
                raise ValueError(
                    f"nht_features.dim={nht_dim} must be divisible by num_interpolation_points={num_points} "
                    f"(interpolation_type + primitive)"
                )
            self.nht_num_interpolation_points = num_points
            self.particle_feature_dim = feat.particle_feature_dim
            self.ray_feature_dim = feat.ray_feature_dim
            self.features = torch.nn.Parameter(
                torch.empty([0, self.particle_feature_dim])
            )  # NHT features [n_gaussians, particle_feature_dim]
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}. Must be 'sh' or 'nht'.")

        self.max_sh_degree = sh_degree

        self.positions_gradient_norm = None

        self.device = "cuda"
        self.optimizer = None
        self.density_activation = get_activation_function(self.conf.model.density_activation)
        self.density_activation_inv = get_activation_function(self.conf.model.density_activation, inverse=True)
        self.scale_activation = get_activation_function(self.conf.model.scale_activation)
        self.scale_activation_inv = get_activation_function(self.conf.model.scale_activation, inverse=True)
        self.rotation_activation = get_activation_function("normalize")  # The default value of the dim parameter is 1

        self.background = background.make(self.conf.model.background.name, self.conf.model.background)

        # Progressive feature training is SH-specific. NHT renders the full learned
        # harmonic feature vector from the first step, matching the reference.
        if self.feature_type == Features.Type.NHT:
            self.n_active_features = self.ray_feature_dim
            self.max_n_features = self.ray_feature_dim
            self.progressive_training = False
        else:
            self.n_active_features = min(self.conf.model.progressive_training.init_n_features, sh_degree)
            self.max_n_features = (
                sh_degree  # For SH, this is the SH degree (clamped if > render.particle_radiance_sph_degree)
            )
            self.progressive_training = False
            if self.n_active_features < self.max_n_features:
                self.feature_dim_increase_interval = self.conf.model.progressive_training.increase_frequency
                self.feature_dim_increase_step = self.conf.model.progressive_training.increase_step
                self.progressive_training = True

        if conf.render.method == "3dgrt":
            self.renderer = threedgrt_tracer.Tracer(conf)
        elif conf.render.method == "3dgut":
            self.renderer = threedgut_tracer.Tracer(conf)
        else:
            raise ValueError(f"Unknown rendering method: {conf.render.method}")

        # State of gradients of Gaussian parameters
        self._gaussians_frozen = False
        self._resume_lr_scale = 1.0
        self.protected_gaussian_count = 0
        self._protected_prefix_metadata: dict[str, object] = {}
        self._protected_gradient_handles: list[torch.utils.hooks.RemovableHandle] = []

    def _protected_geometry_hashes(self) -> dict[str, str]:
        return {
            name: tensor_prefix_sha256(
                getattr(self, name),
                self.protected_gaussian_count,
            )
            for name in PROTECTED_GEOMETRY_NAMES
        }

    def _load_protected_prefix_metadata(
        self,
        checkpoint: dict[str, object],
    ) -> None:
        metadata = checkpoint.get("protected_gaussian_prefix")
        if metadata is None:
            self.protected_gaussian_count = 0
            self._protected_prefix_metadata = {}
            self.refresh_protected_gradient_hooks()
            return
        if not isinstance(metadata, dict):
            raise ValueError("Protected Gaussian prefix metadata must be a mapping.")
        if metadata.get("version") != PROTECTED_GAUSSIAN_PREFIX_VERSION:
            raise ValueError(f"Unsupported protected Gaussian prefix version: {metadata.get('version')!r}.")
        count = metadata.get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError("Protected Gaussian prefix count must be an integer.")
        if count <= 0 or count > self.num_gaussians:
            raise ValueError(
                "Protected Gaussian prefix count is outside the model: "
                f"{count} for {self.num_gaussians} Gaussians."
            )
        expected_hashes = metadata.get("geometry_sha256")
        if not isinstance(expected_hashes, dict):
            raise ValueError("Protected Gaussian prefix geometry hashes are missing.")
        self.protected_gaussian_count = count
        actual_hashes = self._protected_geometry_hashes()
        for name in PROTECTED_GEOMETRY_NAMES:
            expected = expected_hashes.get(name)
            if not isinstance(expected, str) or expected != actual_hashes[name]:
                raise ValueError(
                    "Protected Gaussian prefix geometry hash mismatch for "
                    f"{name}: expected {expected!r}, got {actual_hashes[name]!r}."
                )
        self._protected_prefix_metadata = dict(metadata)
        self.refresh_protected_gradient_hooks()

    def refresh_protected_gradient_hooks(self) -> None:
        for handle in getattr(self, "_protected_gradient_handles", []):
            handle.remove()
        self._protected_gradient_handles = []
        protected_gaussian_count = int(getattr(self, "protected_gaussian_count", 0))
        if protected_gaussian_count == 0:
            return
        if protected_gaussian_count > self.num_gaussians:
            raise RuntimeError("Protected Gaussian prefix exceeds current topology.")
        for name in PROTECTED_GEOMETRY_NAMES:
            parameter = getattr(self, name)
            if not parameter.requires_grad:
                continue
            handle = parameter.register_hook(
                partial(
                    zero_tensor_prefix_gradient,
                    prefix_count=protected_gaussian_count,
                )
            )
            self._protected_gradient_handles.append(handle)

    def validate_protected_optimizer_state(self) -> None:
        if self.protected_gaussian_count == 0 or self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group["name"] not in PROTECTED_GEOMETRY_NAMES:
                continue
            parameter = group["params"][0]
            state = self.optimizer.state.get(parameter, {})
            for key, value in state.items():
                if key == "step" or not torch.is_tensor(value):
                    continue
                if value.ndim == 0 or value.shape[0] != self.num_gaussians:
                    continue
                prefix = value[: self.protected_gaussian_count]
                if bool((prefix != 0).any()):
                    raise ValueError(
                        "Protected Gaussian optimizer state is nonzero for "
                        f"{group['name']}.{key}."
                    )

    @torch.no_grad()
    def build_acc(self, rebuild=True):
        self.renderer.build_acc(self, rebuild)

    def freeze_gaussians(self) -> None:
        """Freeze all Gaussian parameters for PPISP controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        self.positions.requires_grad = False
        self.rotation.requires_grad = False
        self.scale.requires_grad = False
        self.density.requires_grad = False

        if self.feature_type == Features.Type.SH:
            self.features_albedo.requires_grad = False
            self.features_specular.requires_grad = False
        elif self.feature_type == Features.Type.NHT:
            self.features.requires_grad = False

        self._gaussians_frozen = True
        logger.info("❄️ [Distillation] Gaussian parameters frozen")

    def _initialize_acquisition_appearance(self) -> None:
        if not _acquisition_appearance_enabled(self.conf):
            self.acquisition_appearance = None
            return
        if str(self.conf.optimizer.type) != "adam":
            raise ValueError("Acquisition appearance requires optimizer.type=adam.")
        if bool(self.conf.with_gui) or bool(self.conf.with_viser_gui):
            raise ValueError("Acquisition appearance screening requires both GUI modes to be disabled.")
        if self.num_gaussians <= 0:
            raise ValueError("Acquisition appearance requires initialized Gaussians.")
        existing = self.acquisition_appearance
        if existing is not None and existing.direction_raw.shape[0] == self.num_gaussians:
            return
        appearance_conf = self.conf.model.acquisition_appearance
        self.acquisition_appearance = AcquisitionAppearance(
            num_gaussians=self.num_gaussians,
            num_cameras=int(appearance_conf.num_cameras),
            num_knots=int(appearance_conf.num_knots),
            min_sequence_idx=int(appearance_conf.min_sequence_idx),
            max_sequence_idx=int(appearance_conf.max_sequence_idx),
            max_rgb_delta=float(appearance_conf.max_rgb_delta),
            magnitude_regularization=float(appearance_conf.magnitude_regularization),
            curvature_regularization=float(appearance_conf.curvature_regularization),
            device=self.positions.device,
            dtype=self.positions.dtype,
            rank=int(appearance_conf.rank),
        )

    def _initialize_acquisition_visibility(self) -> None:
        if not _acquisition_visibility_enabled(self.conf):
            self.acquisition_visibility = None
            return
        if str(self.conf.optimizer.type) != "adam":
            raise ValueError("Acquisition visibility requires optimizer.type=adam.")
        if str(self.conf.model.density_activation) != "sigmoid":
            raise ValueError(
                "Acquisition visibility requires sigmoid density activation "
                "so its bounded correction is an opacity-logit delta."
            )
        if bool(self.conf.with_gui) or bool(self.conf.with_viser_gui):
            raise ValueError("Acquisition visibility screening requires both GUI modes to be disabled.")
        if self.num_gaussians <= 0:
            raise ValueError("Acquisition visibility requires initialized Gaussians.")
        existing = self.acquisition_visibility
        if existing is not None and existing.response_raw.shape[0] == self.num_gaussians:
            return
        visibility_conf = self.conf.model.acquisition_visibility
        self.acquisition_visibility = AcquisitionVisibility(
            num_gaussians=self.num_gaussians,
            num_cameras=int(visibility_conf.num_cameras),
            num_knots=int(visibility_conf.num_knots),
            min_sequence_idx=int(visibility_conf.min_sequence_idx),
            max_sequence_idx=int(visibility_conf.max_sequence_idx),
            max_logit_delta=float(visibility_conf.max_logit_delta),
            response_sparsity_regularization=float(visibility_conf.response_sparsity_regularization),
            magnitude_regularization=float(visibility_conf.magnitude_regularization),
            curvature_regularization=float(visibility_conf.curvature_regularization),
            device=self.positions.device,
            dtype=self.positions.dtype,
            rank=int(visibility_conf.rank),
        )

    def _initialize_surface_acquisition_spline(self) -> None:
        if not _surface_acquisition_spline_enabled(self.conf):
            self.surface_acquisition_spline = None
            return
        if _acquisition_appearance_enabled(self.conf) or _acquisition_visibility_enabled(self.conf):
            raise ValueError(
                "Surface acquisition spline is mutually exclusive with the scene-global acquisition treatments."
            )
        if str(self.conf.optimizer.type) != "adam":
            raise ValueError("Surface acquisition spline requires optimizer.type=adam.")
        if bool(self.conf.with_gui) or bool(self.conf.with_viser_gui):
            raise ValueError("Surface acquisition spline screening requires both GUI modes to be disabled.")
        if self.num_gaussians <= 0:
            raise ValueError("Surface acquisition spline requires initialized Gaussians.")
        existing = self.surface_acquisition_spline
        if existing is not None and existing.direction_raw.shape[0] == self.num_gaussians:
            return
        spline_conf = self.conf.model.surface_acquisition_spline
        self.surface_acquisition_spline = SurfaceAcquisitionSpline(
            num_gaussians=self.num_gaussians,
            num_cameras=int(spline_conf.num_cameras),
            num_knots=int(spline_conf.num_knots),
            min_sequence_idx=int(spline_conf.min_sequence_idx),
            max_sequence_idx=int(spline_conf.max_sequence_idx),
            max_rgb_delta=float(spline_conf.max_rgb_delta),
            magnitude_regularization=float(spline_conf.magnitude_regularization),
            curvature_regularization=float(spline_conf.curvature_regularization),
            device=self.positions.device,
            dtype=self.positions.dtype,
            rank=int(spline_conf.rank),
        )

    def _initialize_gaussian_track_acquisition(self) -> None:
        if not _gaussian_track_acquisition_enabled(self.conf):
            self.gaussian_track_acquisition = None
            return
        if _surface_acquisition_spline_enabled(self.conf):
            raise ValueError(
                "Gaussian-track acquisition is mutually exclusive with "
                "the per-Gaussian surface acquisition spline."
            )
        if str(self.conf.optimizer.type) != "adam":
            raise ValueError("Gaussian-track acquisition requires optimizer.type=adam.")
        if bool(self.conf.with_gui) or bool(self.conf.with_viser_gui):
            raise ValueError("Gaussian-track acquisition screening requires both GUI modes to be disabled.")
        if self.num_gaussians <= 0:
            raise ValueError("Gaussian-track acquisition requires initialized Gaussians.")
        existing = self.gaussian_track_acquisition
        if existing is not None and existing.gaussian_track_ids.shape[0] == self.num_gaussians:
            return
        track_conf = self.conf.model.gaussian_track_acquisition
        track_ids = load_gaussian_track_ids(
            path=str(track_conf.track_ids_path),
            expected_sha256=str(track_conf.track_ids_sha256),
            num_gaussians=self.num_gaussians,
            device=self.positions.device,
        )
        self.gaussian_track_acquisition = GaussianTrackAcquisition(
            gaussian_track_ids=track_ids,
            num_cameras=int(track_conf.num_cameras),
            num_knots=int(track_conf.num_knots),
            min_sequence_idx=int(track_conf.min_sequence_idx),
            max_sequence_idx=int(track_conf.max_sequence_idx),
            max_rgb_delta=float(track_conf.max_rgb_delta),
            magnitude_regularization=float(track_conf.magnitude_regularization),
            curvature_regularization=float(track_conf.curvature_regularization),
            device=self.positions.device,
            dtype=self.positions.dtype,
        )

    def get_regularization_loss(self) -> torch.Tensor:
        loss = self.positions.sum() * 0.0
        if self.acquisition_appearance is not None:
            loss = loss + self.acquisition_appearance.get_regularization_loss()
        if self.acquisition_visibility is not None:
            loss = loss + self.acquisition_visibility.get_regularization_loss()
        return loss

    def get_contextual_regularization_loss(
        self,
        batch: Batch,
    ) -> torch.Tensor:
        loss = self.positions.sum() * 0.0
        camera_idx = getattr(batch, "post_processing_camera_idx", -1)
        if int(camera_idx) < 0:
            camera_idx = batch.camera_idx
        if self.surface_acquisition_spline is not None:
            loss = loss + self.surface_acquisition_spline.get_local_regularization_loss(
                camera_idx=camera_idx,
                sequence_idx=batch.sequence_idx,
            )
        if self.gaussian_track_acquisition is not None:
            loss = loss + self.gaussian_track_acquisition.get_local_regularization_loss(
                camera_idx=camera_idx,
                sequence_idx=batch.sequence_idx,
            )
        return loss

    def validate_fields(self):
        num_gaussians = self.num_gaussians
        assert self.positions.shape == (num_gaussians, 3)
        assert self.density.shape == (num_gaussians, 1)
        assert self.rotation.shape == (num_gaussians, 4)
        assert self.scale.shape == (num_gaussians, 3)

        if self.feature_type == Features.Type.SH:
            assert self.features_albedo.shape == (num_gaussians, 3)
            assert self.features_specular.shape == (num_gaussians, self._specular_feature_dim())
        elif self.feature_type == Features.Type.NHT:
            assert self.features.shape == (num_gaussians, self.particle_feature_dim)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def init_from_colmap(
        self,
        root_path: str,
        observer_pts,
    ):
        # Special case for scannetpp dataset
        if self.conf.dataset.type == "scannetpp":
            points_file = os.path.join(root_path, "colmap", "points3D.txt")
            pts, rgb, _ = read_colmap_points3D_text(points_file)
            file_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
            file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)

        else:
            points_file = os.path.join(root_path, "sparse/0", "points3D.bin")
            # also handle nonbinary points files
            if not os.path.isfile(points_file):
                points_file = os.path.join(root_path, "sparse/0", "points3D.txt")
                pts, rgb, _ = read_colmap_points3D_text(points_file)
                file_pts = torch.tensor(pts, dtype=torch.float32, device=self.device)
                file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)
            else:
                with open(points_file, "rb") as file:
                    n_pts = read_next_bytes(file, 8, "Q")[0]
                    logger.info(f"Found {n_pts} colmap points")

                    file_pts = np.zeros((n_pts, 3), dtype=np.float32)
                    file_rgb = np.zeros((n_pts, 3), dtype=np.float32)

                    for i_pt in range(n_pts):
                        # read the points
                        pt_data = read_next_bytes(file, 43, "QdddBBBd")
                        file_pts[i_pt, :] = np.array(pt_data[1:4])
                        file_rgb[i_pt, :] = np.array(pt_data[4:7])
                        # NOTE: error stored in last element of file, currently not used

                        # skip the track data
                        t_len = read_next_bytes(file, num_bytes=8, format_char_sequence="Q")[0]
                        read_next_bytes(file, num_bytes=8 * t_len, format_char_sequence="ii" * t_len)

                file_pts = torch.tensor(file_pts, dtype=torch.float32, device=self.device)
                file_rgb = torch.tensor(file_rgb, dtype=torch.uint8, device=self.device)

        assert file_rgb.dtype == torch.uint8, "Expecting RGB values to be in [0, 255] range"
        max_points = int(self.conf.initialization.get("num_points", 0))
        original_points = file_pts.shape[0]
        file_pts, file_rgb = _subsample_initial_points(
            file_pts,
            file_rgb,
            max_points=max_points,
            seed=int(self.conf.seed_initialization),
        )
        if file_pts.shape[0] != original_points:
            logger.info(f"Subsampled COLMAP initialization points from {original_points} to {file_pts.shape[0]}")

        surface_aligned_pca_config = validated_surface_aligned_pca_config(
            self.conf.initialization,
            file_pts,
            observer_pts,
        )
        if surface_aligned_pca_config is None:
            self.default_initialize_from_points(
                file_pts,
                observer_pts,
                file_rgb,
                use_observer_pts=self.conf.initialization.use_observation_points,
            )
        else:
            self.default_initialize_from_points(
                file_pts,
                observer_pts,
                file_rgb,
                use_observer_pts=True,
                surface_aligned_pca_config=surface_aligned_pca_config,
            )

    def init_from_fused_point_cloud(self, pc_path: str, observer_pts):
        """
        Initialize gaussians from an fused point cloud PLY file.
        Similar to init_from_colmap but loads from a given PLY file instead of sparse/0/points3D.txt

        Args:
            pc_path: Path to the PLY point cloud file
            observer_pts: Observer points tensor for scale initialization
        """
        logger.info(f"Loading fused point cloud from {pc_path}...")

        # Read PLY file
        plydata = PlyData.read(pc_path)
        vertices = plydata["vertex"]

        # Extract XYZ coordinates
        xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

        # Extract RGB colors (check if they exist)
        if "red" in vertices and "green" in vertices and "blue" in vertices:
            rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.uint8)
        else:
            # If no colors, initialize with random colors
            logger.warning("No RGB data found in point cloud, using random colors")
            rgb = np.random.randint(0, 256, size=(len(vertices), 3), dtype=np.uint8)

        # Convert to torch tensors
        file_pts = torch.tensor(xyz, dtype=torch.float32, device=self.device)
        file_rgb = torch.tensor(rgb, dtype=torch.uint8, device=self.device)

        logger.info(f"Loaded {len(file_pts)} points from accumulated point cloud")

        # Initialize using the same method as COLMAP
        assert file_rgb.dtype == torch.uint8, "Expecting RGB values to be in [0, 255] range"
        self.default_initialize_from_points(
            file_pts,
            observer_pts,
            file_rgb,
            use_observer_pts=self.conf.initialization.use_observation_points,
        )

    def init_from_pretrained_point_cloud(self, pc_path: str, set_optimizable_parameters: bool = True):
        if self.feature_type != Features.Type.SH:
            raise NotImplementedError(
                f"init_from_pretrained_point_cloud only supports feature_type='sh', got '{self.feature_type.name.lower()}'"
            )
        data = PlyData.read(pc_path)
        num_gaussians = len(data["vertex"])
        self.positions = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (data["vertex"]["x"], data["vertex"]["y"], data["vertex"]["z"]),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore
        self.rotation = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (
                            data["vertex"]["rot_0"],
                            data["vertex"]["rot_1"],
                            data["vertex"]["rot_2"],
                            data["vertex"]["rot_3"],
                        ),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore
        self.scale = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (
                            data["vertex"]["scale_0"],
                            data["vertex"]["scale_1"],
                            data["vertex"]["scale_2"],
                        ),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore
        self.density = torch.nn.Parameter(
            to_torch(
                data["vertex"]["opacity"].astype(np.float32).reshape(num_gaussians, 1),
                device=self.device,
            )
        )
        self.features_albedo = torch.nn.Parameter(
            to_torch(
                np.transpose(
                    np.stack(
                        (
                            data["vertex"]["f_dc_0"],
                            data["vertex"]["f_dc_1"],
                            data["vertex"]["f_dc_2"],
                        ),
                        dtype=np.float32,
                    )
                ),
                device=self.device,
            )
        )  # type: ignore

        feats_sph = to_torch(
            np.transpose(
                np.stack(
                    (
                        data["vertex"]["f_rest_0"],
                        data["vertex"]["f_rest_1"],
                        data["vertex"]["f_rest_2"],
                        data["vertex"]["f_rest_3"],
                        data["vertex"]["f_rest_4"],
                        data["vertex"]["f_rest_5"],
                        data["vertex"]["f_rest_6"],
                        data["vertex"]["f_rest_7"],
                        data["vertex"]["f_rest_8"],
                        data["vertex"]["f_rest_9"],
                        data["vertex"]["f_rest_10"],
                        data["vertex"]["f_rest_11"],
                        data["vertex"]["f_rest_12"],
                        data["vertex"]["f_rest_13"],
                        data["vertex"]["f_rest_14"],
                        data["vertex"]["f_rest_15"],
                        data["vertex"]["f_rest_16"],
                        data["vertex"]["f_rest_17"],
                        data["vertex"]["f_rest_18"],
                        data["vertex"]["f_rest_19"],
                        data["vertex"]["f_rest_20"],
                        data["vertex"]["f_rest_21"],
                        data["vertex"]["f_rest_22"],
                        data["vertex"]["f_rest_23"],
                        data["vertex"]["f_rest_24"],
                        data["vertex"]["f_rest_25"],
                        data["vertex"]["f_rest_26"],
                        data["vertex"]["f_rest_27"],
                        data["vertex"]["f_rest_28"],
                        data["vertex"]["f_rest_29"],
                        data["vertex"]["f_rest_30"],
                        data["vertex"]["f_rest_31"],
                        data["vertex"]["f_rest_32"],
                        data["vertex"]["f_rest_33"],
                        data["vertex"]["f_rest_34"],
                        data["vertex"]["f_rest_35"],
                        data["vertex"]["f_rest_36"],
                        data["vertex"]["f_rest_37"],
                        data["vertex"]["f_rest_38"],
                        data["vertex"]["f_rest_39"],
                        data["vertex"]["f_rest_40"],
                        data["vertex"]["f_rest_41"],
                        data["vertex"]["f_rest_42"],
                        data["vertex"]["f_rest_43"],
                        data["vertex"]["f_rest_44"],
                    ),
                    dtype=np.float32,
                )
            ),
            device=self.device,
        )

        # reinterpret from C-style to F-style layout
        feats_sph = feats_sph.reshape(num_gaussians, 3, -1).transpose(-1, -2).reshape(num_gaussians, -1)

        self.features_specular = torch.nn.Parameter(feats_sph)

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    @torch.no_grad()
    def init_from_random_point_cloud(
        self,
        num_gaussians: int = 100_000,
        dtype=torch.float32,
        set_optimizable_parameters: bool = True,
        xyz_max=1.5,
        xyz_min=-1.5,
    ):
        logger.info(f"Generating random point cloud ({num_gaussians})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz in [-1.5, 1.5] -> standard NeRF convention, people often scale with 0.33 to get it to [-0.5, 0.5]
        fused_point_cloud = (
            torch.rand((num_gaussians, 3), dtype=dtype, device=self.device) * (xyz_max - xyz_min) + xyz_min
        )
        # sh albedo in [0, 0.0039]
        fused_color = torch.rand((num_gaussians, 3), dtype=dtype, device=self.device) / 255.0

        # Initialize features based on feature_type
        if self.feature_type == Features.Type.SH:
            features_albedo = fused_color.contiguous()
            num_specular_features = self._specular_feature_dim()
            features_specular = torch.zeros(
                (num_gaussians, num_specular_features), dtype=dtype, device=self.device
            ).contiguous()
            if carrier_specular_dim(self.conf) > 0:
                sh_dim = sh_degree_to_specular_dim(self.max_n_features)
                features_specular[:, sh_dim:] = initial_carrier_tail(
                    num_gaussians=num_gaussians, device=self.device, dtype=dtype, conf=self.conf
                )
        elif self.feature_type == Features.Type.NHT:
            init_min = float(getattr(self.conf.model.nht_features, "init_min", -5.0))
            init_max = float(getattr(self.conf.model.nht_features, "init_max", 5.0))
            features = (
                torch.rand((num_gaussians, self.particle_feature_dim), dtype=dtype, device=self.device)
                * (init_max - init_min)
                + init_min
            )

        dist = torch.clamp_min(nearest_neighbor_dist_cpuKD(fused_point_cloud), 1e-3)
        scales = torch.log(dist * self.conf.model.default_scale_factor)[..., None].repeat(1, 3)

        rots = torch.rand((num_gaussians, 4), device=self.device)
        rots[:, 0] = 1

        opacities = self.density_activation_inv(
            self.conf.model.default_density * torch.ones((num_gaussians, 1), dtype=dtype, device=self.device)
        )

        self.positions = torch.nn.Parameter(fused_point_cloud)  # type: ignore
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))

        if self.feature_type == Features.Type.SH:
            self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
            self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))
        elif self.feature_type == Features.Type.NHT:
            self.features = torch.nn.Parameter(features.to(dtype=dtype, device=self.device))

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer=True):
        # Backward compatibility: detect legacy checkpoints without feature_type
        if "feature_type" not in checkpoint and "features_albedo" in checkpoint:
            logger.info("Loading legacy checkpoint - auto-detecting feature_type='sh'")
            checkpoint["feature_type"] = "sh"
            checkpoint["particle_feature_dim"] = (
                checkpoint["features_albedo"].shape[1] + checkpoint["features_specular"].shape[1]
            )
            checkpoint["ray_feature_dim"] = 3

        # Load features based on feature_type (convert string to enum)
        checkpoint_feature_type_str = checkpoint.get("feature_type", "sh")
        checkpoint_feature_type = Features.Type.from_string(checkpoint_feature_type_str)

        # NHT: 3DGUT is compiled with PARTICLE_FEATURE_DIM / RAY_FEATURE_DIM from current config.
        # Checkpoints must match those compile-time constants or CUDA will read past feature buffers.
        if checkpoint_feature_type == Features.Type.NHT:
            if "features" not in checkpoint:
                raise ValueError("NHT checkpoint missing 'features' tensor")
            feat = checkpoint["features"]
            ck_pf = int(feat.shape[1])
            ck_rf = checkpoint.get("ray_feature_dim")
            if ck_pf != self.particle_feature_dim:
                raise ValueError(
                    f"NHT checkpoint features width is {ck_pf} but this build expects "
                    f"particle_feature_dim={self.particle_feature_dim} from config "
                    f"(model.nht_features.dim / interpolation). The 3DGUT CUDA extension was compiled for "
                    f"the config value; use the same nht_features (and render.primitive_type) as the run "
                    f"that produced the checkpoint, or train from scratch."
                )
            if ck_rf is not None and int(ck_rf) != self.ray_feature_dim:
                raise ValueError(
                    f"NHT checkpoint ray_feature_dim={ck_rf} does not match config ray_feature_dim="
                    f"{self.ray_feature_dim}. Align model.nht_features.activation with the checkpoint run."
                )
            if "particle_feature_dim" in checkpoint and int(checkpoint["particle_feature_dim"]) != ck_pf:
                logger.warning(
                    f"Checkpoint particle_feature_dim={checkpoint['particle_feature_dim']} disagrees with "
                    f"features.shape[1]={ck_pf}; using tensor shape."
                )

        # Load basic parameters
        self.positions = checkpoint["positions"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.n_active_features = checkpoint["n_active_features"]
        self.max_n_features = checkpoint["max_n_features"]
        self.scene_extent = checkpoint["scene_extent"]

        # Load feature dimensions. For NHT, keep config-derived dims (validated above vs checkpoint tensors);
        # stale metadata keys must not override after a successful shape check.
        if checkpoint_feature_type != Features.Type.NHT:
            if "particle_feature_dim" in checkpoint:
                self.particle_feature_dim = checkpoint["particle_feature_dim"]
            if "ray_feature_dim" in checkpoint:
                self.ray_feature_dim = checkpoint["ray_feature_dim"]

        if checkpoint_feature_type == Features.Type.SH:
            self.features_albedo = checkpoint["features_albedo"]
            self.features_specular = torch.nn.Parameter(self._with_carrier_tail(checkpoint["features_specular"]))
        elif checkpoint_feature_type == Features.Type.NHT:
            self.features = checkpoint["features"]
            self.nht_num_interpolation_points = Features(self.conf).num_interpolation_points
            self.n_active_features = self.ray_feature_dim
            self.max_n_features = self.ray_feature_dim
            self.progressive_training = False
        else:
            raise ValueError(f"Unknown feature_type in checkpoint: {checkpoint_feature_type}")

        # Acquisition arms are per-Gaussian; initialize against the just-loaded
        # topology before restoring their optional state dicts.
        self._initialize_acquisition_appearance()
        self._initialize_acquisition_visibility()
        self._initialize_surface_acquisition_spline()
        self._initialize_gaussian_track_acquisition()
        appearance_state = checkpoint.get("acquisition_appearance")
        if appearance_state is not None:
            if self.acquisition_appearance is None:
                raise ValueError(
                    "Checkpoint contains acquisition appearance state but the representation is disabled."
                )
            self.acquisition_appearance.load_state_dict(appearance_state, strict=True)
        elif self.acquisition_appearance is not None:
            logger.info("Initializing zero-output acquisition appearance from a legacy Gaussian checkpoint.")
        visibility_state = checkpoint.get("acquisition_visibility")
        if visibility_state is not None:
            if self.acquisition_visibility is None:
                raise ValueError(
                    "Checkpoint contains acquisition visibility state but the representation is disabled."
                )
            self.acquisition_visibility.load_state_dict(visibility_state, strict=True)
        elif self.acquisition_visibility is not None:
            logger.info("Initializing zero-output acquisition visibility from a legacy Gaussian checkpoint.")
        spline_state = checkpoint.get("surface_acquisition_spline")
        if spline_state is not None:
            if self.surface_acquisition_spline is None:
                raise ValueError(
                    "Checkpoint contains surface acquisition spline state but the representation is disabled."
                )
            self.surface_acquisition_spline.load_state_dict(spline_state, strict=True)
        elif self.surface_acquisition_spline is not None:
            logger.info("Initializing zero-output surface acquisition spline from a legacy Gaussian checkpoint.")
        track_state = checkpoint.get("gaussian_track_acquisition")
        if track_state is not None:
            if self.gaussian_track_acquisition is None:
                raise ValueError(
                    "Checkpoint contains Gaussian-track acquisition state but the representation is disabled."
                )
            self.gaussian_track_acquisition.load_state_dict(track_state, strict=True)
        elif self.gaussian_track_acquisition is not None:
            logger.info("Initializing zero-output Gaussian-track acquisition from a parent checkpoint.")

        if self.progressive_training:
            self.feature_dim_increase_interval = checkpoint.get(
                "feature_dim_increase_interval",
                self.feature_dim_increase_interval,
            )
            self.feature_dim_increase_step = checkpoint.get(
                "feature_dim_increase_step",
                self.feature_dim_increase_step,
            )

        self.background.load_state_dict(checkpoint["background"])
        self._load_protected_prefix_metadata(checkpoint)
        if setup_optimizer:
            self.set_optimizable_parameters()
            self.setup_optimizer(state_dict=checkpoint["optimizer"])
            self.validate_protected_optimizer_state()
        self.validate_fields()

    def init_from_lidar(self, point_cloud, observer_pts):
        """
        Initialize from lidar point cloud.
        Observer points can be any set locations that observation came from.
        Camera centers, ray source points, etc. They are used to estimate initial scales.
        """
        logger.info("Initializing based on lidar point cloud ...")

        self.default_initialize_from_points(
            point_cloud.xyz_end.to(device=self.device),
            observer_pts,
            point_cloud.color,
            use_observer_pts=self.conf.initialization.use_observation_points,
        )

    def default_initialize_from_points(
        self,
        pts,
        observer_pts,
        colors=None,
        use_observer_pts=True,
        surface_aligned_pca_config: SurfaceAlignedPCAConfig | None = None,
    ):
        """
        Given an Nx3 array of points (and optionally Nx3 rgb colors),
        initialize default values for the other parameters of the model
        """

        dtype = torch.float32

        # Local generator for deterministic random initialization (does not affect global RNG)
        rng = torch.Generator(device=self.device).manual_seed(self.conf.seed_initialization)

        N = pts.shape[0]
        positions = pts
        self.ensure_scene_extent_from_points(positions)

        # Random rotations
        rots = torch.rand((N, 4), dtype=dtype, device=self.device, generator=rng)

        if use_observer_pts:
            # NOTE: it seems we get different scales compared to the original 3DGS implementation
            # estimate scales based on distances to observers
            dist_to_observers = torch.clamp_min(nearest_neighbor_dist_cpuKD(pts, observer_pts), 1e-7)
            observation_scale = dist_to_observers * self.conf.initialization.observation_scale_factor
        else:
            # Initialize the GS size to be the average dist of the 3 nearest neighbors
            dist2_avg = (k_nearest_neighbors(pts, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
            observation_scale = torch.sqrt(dist2_avg)

        observation_scale = observation_scale * self.conf.model.default_scale_factor

        if surface_aligned_pca_config is None:
            scales = self.scale_activation_inv(observation_scale)[:, None].repeat(1, 3)
        else:
            incumbent_physical_scales = observation_scale[:, None].repeat(1, 3)
            surface_result = surface_aligned_pca_initialize(
                pts,
                incumbent_physical_scales,
                rots,
                surface_aligned_pca_config,
            )
            rots = surface_result.raw_rotations_wxyz
            scales = self.scale_activation_inv(surface_result.physical_scales)
            audit = surface_result.audit
            logger.info(f"Surface-aligned PCA initialization audit: {audit}")

        # set colors, random if they weren't given
        if colors is None:
            colors = torch.randint(0, 256, (N, 3), dtype=torch.uint8, device=self.device, generator=rng)

        # set density as a constant
        opacities = self.density_activation_inv(
            torch.full(
                (N, 1),
                fill_value=self.conf.model.default_density,
                dtype=dtype,
                device=self.device,
            )
        )

        # Initialize features based on feature_type. Default to SH for
        # partially-constructed callers that predate the NHT feature split.
        feature_type = getattr(self, "feature_type", Features.Type.SH)
        if feature_type == Features.Type.SH:
            features_albedo = to_torch(RGB2SH(to_np(colors.float() / 255.0)), device=self.device)
            num_specular_dims = self._specular_feature_dim()
            features_specular = torch.zeros((N, num_specular_dims))
            if carrier_specular_dim(self.conf) > 0:
                sh_dim = sh_degree_to_specular_dim(self.max_n_features)
                features_specular[:, sh_dim:] = initial_carrier_tail(
                    num_gaussians=N,
                    device=features_specular.device,
                    dtype=features_specular.dtype,
                    conf=self.conf,
                )
        elif feature_type == Features.Type.NHT:
            init_min = float(getattr(self.conf.model.nht_features, "init_min", -5.0))
            init_max = float(getattr(self.conf.model.nht_features, "init_max", 5.0))
            features = (
                torch.rand((N, self.particle_feature_dim), dtype=dtype, device=self.device, generator=rng)
                * (init_max - init_min)
                + init_min
            )

        self.positions = torch.nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))

        if feature_type == Features.Type.SH:
            self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
            self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))
        elif feature_type == Features.Type.NHT:
            self.features = torch.nn.Parameter(features.to(dtype=dtype, device=self.device))

        self.set_optimizable_parameters()
        self.setup_optimizer()
        self.validate_fields()

    def setup_optimizer(self, state_dict=None):
        params = []
        for name, args in self.conf.optimizer.params.items():
            # Skip parameters that don't exist (e.g., 'features' in SH mode or 'features_albedo' in learned mode)
            if not hasattr(self, name):
                logger.info(
                    f"Skipping optimizer parameter '{name}' - not present in {self.feature_type.name.lower()} mode"
                )
                continue

            module = getattr(self, name)

            # If the module is a torch.nn.Module, we can add all of its trainable parameters to the optimizer
            if isinstance(module, torch.nn.Module):
                module_parameters = filter(lambda p: p.requires_grad and len(p) > 0, module.parameters())
                n_params = sum([np.prod(p.size(), dtype=int) for p in module_parameters])

                if n_params > 0:
                    params.append({"params": module.parameters(), "name": name, **args})

            # If the module is a torch.nn.Parameter, we can add it to the optimizer
            elif isinstance(module, torch.nn.Parameter):
                if module.requires_grad:
                    params.append({"params": [module], "name": name, **args})

        optimizer_betas = tuple(float(beta) for beta in self.conf.optimizer.get("betas", (0.9, 0.999)))
        if len(optimizer_betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two values")

        if self.conf.optimizer.type == "adam":
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.conf.optimizer.lr,
                betas=optimizer_betas,
                eps=self.conf.optimizer.eps,
            )
            logger.info("🔆 Using Adam optimizer")
        elif self.conf.optimizer.type == "selective_adam":
            self.optimizer = SelectiveAdam(params, lr=self.conf.optimizer.lr, eps=self.conf.optimizer.eps)
            logger.info("🔆 Using Selective Adam optimizer")
        elif self.conf.optimizer.type == "visibility_decayed_adam":
            self.optimizer = VisibilityDecayedAdam(
                params,
                lr=self.conf.optimizer.lr,
                betas=optimizer_betas,
                eps=self.conf.optimizer.eps,
            )
            logger.info("Using visibility-decayed Adam optimizer")
        else:
            raise ValueError(f"Unknown optimizer type: {self.conf.optimizer.type}")

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "positions" and bool(
                self.conf.optimizer.get("scale_position_lr_by_scene_extent", True)
            ):
                param_group["lr"] *= self.scene_extent  # Multiply the position lr by the scene scale

        self.setup_scheduler()

        # When loading from the checkpoint also load the state dict
        if state_dict is not None:
            try:
                self.optimizer.load_state_dict(state_dict)
            except ValueError as exc:
                raise ValueError(
                    "Cannot restore Gaussian optimizer state without changing the resume contract."
                ) from exc
            self._apply_resume_lr_scale()

    def _apply_resume_lr_scale(self) -> None:
        scale = float(self.conf.optimizer.get("resume_lr_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"optimizer.resume_lr_scale must be finite and positive; got {scale}")
        self._resume_lr_scale = scale
        if scale == 1.0:
            return
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(param_group["lr"]) * scale
        logger.info(f"🔆 Scaled resumed optimizer learning rates by {scale:g}")

    def setup_scheduler(self):
        self.schedulers = {}
        for name, args in self.conf.scheduler.items():
            if not hasattr(self, name):
                continue
            attr = getattr(self, name)
            if not (hasattr(attr, "requires_grad") and attr.requires_grad):
                continue
            if args.type is None:
                continue
            if args.type == "skip":
                self.schedulers[name] = get_scheduler(args.type)()
            elif name == "positions":
                position_lr_scale = (
                    self.scene_extent
                    if bool(self.conf.optimizer.get("scale_position_lr_by_scene_extent", True))
                    else 1.0
                )
                self.schedulers[name] = get_scheduler(args.type)(
                    lr_init=args.lr_init * position_lr_scale,
                    lr_final=args.lr_final * position_lr_scale,
                    max_steps=args.max_steps,
                )
            elif name == "features":
                lr_init = getattr(self.conf.optimizer.params.features, "lr", 0.07)
                decay_final = getattr(args, "decay_final", 0.001)
                lr_final = lr_init * decay_final
                self.schedulers[name] = get_scheduler(args.type)(
                    lr_init=lr_init, lr_final=lr_final, max_steps=args.max_steps
                )
            else:
                self.schedulers[name] = get_scheduler(args.type)(**args)

    def scheduler_step(self, step):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.schedulers:
                lr = self.schedulers[param_group["name"]](step)
                if lr is not None:
                    lr = float(lr) * self._resume_lr_scale
                    if not np.isfinite(float(lr)):
                        raise ValueError(f"Non-finite scheduler LR for {param_group['name']} at step {step}: {lr}")
                    param_group["lr"] = lr

    def ensure_scene_extent_from_points(self, points: torch.Tensor) -> None:
        """Use point geometry when camera-center extent is degenerate.

        A centered / near-stationary rig (e.g. a 360 panorama capture) yields
        a camera-derived extent ~0 even though the scene has real scale. An
        extent that is tiny *relative to* the point-cloud radius is degenerate
        and would freeze the position LR (``lr * scene_extent``), so fall back
        to the geometry-derived radius. The check is scale-invariant, so a
        genuine multi-view rig (camera extent ~ scene radius) keeps its
        camera-derived extent; only a stationary/centered rig triggers the
        fallback.
        """
        point_extent = _estimate_scene_extent_from_points(points)
        if self.scene_extent is not None:
            scene_extent = float(self.scene_extent)
            if np.isfinite(scene_extent) and scene_extent > SCENE_EXTENT_MIN and scene_extent >= 0.05 * point_extent:
                return

        self.scene_extent = point_extent
        logger.warning(
            "Camera-derived scene extent is degenerate; using point-cloud "
            f"fallback scene_extent={self.scene_extent:.6f}"
        )

    def set_optimizable_parameters(self):
        self._initialize_acquisition_appearance()
        self._initialize_acquisition_visibility()
        self._initialize_surface_acquisition_spline()
        self._initialize_gaussian_track_acquisition()
        freeze_for_appearance = bool(
            _acquisition_appearance_enabled(self.conf)
            and self.conf.model.acquisition_appearance.get("freeze_base_gaussians", True)
        )
        freeze_for_visibility = bool(
            _acquisition_visibility_enabled(self.conf)
            and self.conf.model.acquisition_visibility.get("freeze_base_gaussians", True)
        )
        freeze_for_surface_spline = bool(
            _surface_acquisition_spline_enabled(self.conf)
            and self.conf.model.surface_acquisition_spline.get("freeze_base_gaussians", True)
        )
        freeze_for_gaussian_tracks = bool(
            _gaussian_track_acquisition_enabled(self.conf)
            and self.conf.model.gaussian_track_acquisition.get("freeze_parent", True)
        )
        freeze_base = (
            freeze_for_appearance
            or freeze_for_visibility
            or freeze_for_surface_spline
            or freeze_for_gaussian_tracks
        )
        if freeze_for_gaussian_tracks:
            if self.acquisition_appearance is not None:
                self.acquisition_appearance.requires_grad_(False)
            if self.acquisition_visibility is not None:
                self.acquisition_visibility.requires_grad_(False)

        self.density.requires_grad = bool(self.conf.model.optimize_density and not freeze_base)
        self.rotation.requires_grad = bool(self.conf.model.optimize_rotation and not freeze_base)
        self.scale.requires_grad = bool(self.conf.model.optimize_scale and not freeze_base)
        self.positions.requires_grad = bool(self.conf.model.optimize_position and not freeze_base)

        # Handle feature optimization based on feature_type. Default to SH for
        # partially-constructed callers that predate the NHT feature split.
        feature_type = getattr(self, "feature_type", Features.Type.SH)
        if feature_type == Features.Type.SH:
            self.features_albedo.requires_grad = bool(
                self.conf.model.optimize_features_albedo and not freeze_base
            )
            self.features_specular.requires_grad = bool(
                self.conf.model.optimize_features_specular and not freeze_base
            )
        elif feature_type == Features.Type.NHT:
            # For learned features, check if optimize_features config exists
            self.features.requires_grad = bool(self.conf.model.optimize_features and not freeze_base)

        self.refresh_protected_gradient_hooks()

    def update_optimizable_parameters(self, optimizable_tensors: dict[str, torch.Tensor]):
        for name, value in optimizable_tensors.items():
            setattr(self, name, value)
        self.refresh_protected_gradient_hooks()

    def increase_num_active_features(self) -> None:
        self.n_active_features = min(self.max_n_features, self.n_active_features + self.feature_dim_increase_step)

    def get_active_feature_mask(self) -> torch.Tensor:
        if self.feature_type == Features.Type.SH:
            current_sh_degree = self.n_active_features
            max_sh_degree = self.max_n_features
            active_features = sh_degree_to_num_features(current_sh_degree)
            num_features = sh_degree_to_num_features(max_sh_degree)
        else:
            active_features = self.n_active_features
            num_features = self.max_n_features
        mask = torch.zeros((1, num_features), device=self.device, dtype=self.get_features().dtype)
        mask[0, :active_features] = 1.0
        return mask

    def clamp_density(self):
        updated_densities = torch.clamp(self.get_density(), min=1e-4, max=1.0 - 1e-4)
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self.density = optimizable_tensors["density"]

    def forward(self, batch: Batch, train=False, frame_id=0) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: a Batch structure containing the input data
            train: a boolean indicating whether the model is in training mode
            frame_id: an integer indicating the frame id (default is 0)
        Returns:
            A dictionary containing the output of the model
        """
        if (
            self.acquisition_appearance is None
            and self.acquisition_visibility is None
            and self.surface_acquisition_spline is None
            and self.gaussian_track_acquisition is None
        ):
            return self.renderer.render(self, batch, train, frame_id)
        camera_idx = getattr(batch, "post_processing_camera_idx", -1)
        if int(camera_idx) < 0:
            camera_idx = batch.camera_idx
        density = self.get_density()
        if self.acquisition_visibility is not None:
            density_logit_delta = self.acquisition_visibility.logit_delta(
                camera_idx=camera_idx,
                sequence_idx=batch.sequence_idx,
            )
            density = self.density_activation(self.density + density_logit_delta)
        if self.surface_acquisition_spline is not None:
            view = self.surface_acquisition_spline.materialize(
                positions=self.positions,
                rotation=self.get_rotation(),
                scale=self.get_scale(),
                density=density,
                features_albedo=self.features_albedo,
                features_specular=self.features_specular,
                background=self.background,
                n_active_features=self.n_active_features,
                camera_idx=camera_idx,
                sequence_idx=batch.sequence_idx,
            )
        else:
            albedo = self.features_albedo
            if self.acquisition_appearance is not None:
                albedo = (
                    albedo
                    + self.acquisition_appearance.rgb_delta(
                        camera_idx=camera_idx,
                        sequence_idx=batch.sequence_idx,
                    )
                    / SH_DC_NORMALIZATION
                )
            if self.gaussian_track_acquisition is not None:
                albedo = (
                    albedo
                    + self.gaussian_track_acquisition.rgb_delta(
                        camera_idx=camera_idx,
                        sequence_idx=batch.sequence_idx,
                    )
                    / SH_DC_NORMALIZATION
                )
            features = torch.cat((albedo, self.features_specular), dim=1)
            view = AcquisitionGaussianView(
                positions=self.positions,
                rotation=self.get_rotation(),
                scale=self.get_scale(),
                density=density,
                features=features.contiguous(),
                background=self.background,
                n_active_features=self.n_active_features,
            )
        return self.renderer.render(view, batch, train, frame_id)

    def trace(self, rays_o, rays_d, T_to_world=None):
        """Traces the model with the given rays. This method is a convenience method for ray-traced inference mode.
        If T_to_world is None, the rays are assumed to be in world space.
        Otherwise, the rays are assumed to be in camera space.
        rays_ori: torch.Tensor  # [B, H, W, 3] ray origins in arbitrary space
        rays_dir: torch.Tensor  # [B, H, W, 3] ray directions in arbitrary space
        T_to_world: torch.Tensor  # [B, 4, 4] transformation matrix from the ray space to the world space
        """
        if T_to_world is None:
            T_to_world = torch.eye(4, dtype=rays_o.dtype, device=rays_o.device)[None]
        inputs = Batch(T_to_world=T_to_world, rays_ori=rays_o, rays_dir=rays_d)
        return self.renderer.render(self, inputs)

    def get_bvh_stats(self) -> dict:
        """Forward to the configured backend renderer; empty dict if unavailable."""
        try:
            return self.renderer.get_bvh_stats()
        except AttributeError:
            return {}

    @torch.no_grad()
    def render_diagnostic(
        self,
        gpu_batch: Batch,
        features_override=None,
        sph_degree_override=None,
    ) -> dict[str, torch.Tensor]:
        """No-grad diagnostic render dispatching to the configured backend.

        Used by GUIs for visualization modes (gradient heatmaps etc.) where
        per-particle SH features are overridden by a scalar-derived color and
        the SH degree is forced to 0 so only band-0 evaluates. Bypasses the
        training autograd graph entirely; no training math is affected.
        """
        return self.renderer.render_diagnostic(
            self,
            gpu_batch,
            features_override=features_override,
            sph_degree_override=sph_degree_override,
        )

    @torch.no_grad()
    def export_ply(self, mogt_path: str):
        exporter = PLYExporter()
        exporter.export(self, Path(mogt_path))

    @torch.no_grad()
    def init_from_ply(self, mogt_path: str, init_model=True):
        if self.feature_type != Features.Type.SH:
            raise NotImplementedError(
                f"init_from_ply only supports feature_type='sh', got '{self.feature_type.name.lower()}'"
            )
        plydata = PlyData.read(mogt_path)

        mogt_pos = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        mogt_densities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        num_gaussians = mogt_pos.shape[0]
        mogt_albedo = np.zeros((num_gaussians, 3))
        mogt_albedo[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        mogt_albedo[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        mogt_albedo[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        sh_speculars = (self.max_n_features + 1) ** 2 - 1
        sh_extra_f_count = 3 * sh_speculars
        expected_extra_f_count = self._specular_feature_dim()
        if len(extra_f_names) % 3 != 0:
            raise ValueError(
                f"PLY f_rest_* properties must be packed as float3 slots; found {len(extra_f_names)} fields."
            )

        if len(extra_f_names) in {sh_extra_f_count, expected_extra_f_count}:
            # Full spherical harmonics data available. Carrier-enabled PLYs
            # append carrier slots after the SH coefficients.
            mogt_specular = np.zeros((num_gaussians, len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                mogt_specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
            num_speculars = len(extra_f_names) // 3
            mogt_specular = mogt_specular.reshape((num_gaussians, 3, num_speculars))
            mogt_specular = mogt_specular.transpose(0, 2, 1).reshape((num_gaussians, num_speculars * 3))
        elif len(extra_f_names) == 0:
            # Only DC components available, create zero-filled higher-order harmonics
            mogt_specular = np.zeros((num_gaussians, sh_extra_f_count))
            logger.info(f"PLY file only contains DC components, initializing higher-order spherical harmonics to zero")
        else:
            # Partial data - this is unexpected
            raise ValueError(
                f"Unexpected number of f_rest_ properties: found {len(extra_f_names)}, "
                f"expected {sh_extra_f_count}, {expected_extra_f_count}, or 0"
            )

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        mogt_scales = np.zeros((num_gaussians, len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            mogt_scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        mogt_rotation = np.zeros((num_gaussians, len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            mogt_rotation[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.positions = torch.nn.Parameter(torch.tensor(mogt_pos, dtype=self.positions.dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(
            torch.tensor(mogt_albedo, dtype=self.features_albedo.dtype, device=self.device)
        )
        self.features_specular = torch.nn.Parameter(
            self._with_carrier_tail(torch.tensor(mogt_specular, dtype=self.features_specular.dtype, device=self.device))
        )
        self.density = torch.nn.Parameter(torch.tensor(mogt_densities, dtype=self.density.dtype, device=self.device))
        self.scale = torch.nn.Parameter(torch.tensor(mogt_scales, dtype=self.scale.dtype, device=self.device))
        self.rotation = torch.nn.Parameter(torch.tensor(mogt_rotation, dtype=self.rotation.dtype, device=self.device))

        self.n_active_features = self.max_n_features

        if init_model:
            self.set_optimizable_parameters()
            self.setup_optimizer()
            self.validate_fields()

    def copy_fields(self, other, deepcopy=False):
        """Copies fields from other onto self"""
        if self.optimizer is not None:
            raise NotImplementedError(
                "Operations that create copies of the model during training " "are currently not supported."
            )

        if deepcopy:
            self.positions = torch.nn.Parameter(other.positions.clone())
            self.rotation = torch.nn.Parameter(other.rotation.clone())
            self.scale = torch.nn.Parameter(other.scale.clone())
            self.density = torch.nn.Parameter(other.density.clone())
            if other.feature_type == Features.Type.SH:
                self.features_albedo = torch.nn.Parameter(other.features_albedo.clone())
                self.features_specular = torch.nn.Parameter(other.features_specular.clone())
            elif other.feature_type == Features.Type.NHT:
                self.features = torch.nn.Parameter(other.features.clone())
        else:
            self.positions = torch.nn.Parameter(other.positions)
            self.rotation = torch.nn.Parameter(other.rotation)
            self.scale = torch.nn.Parameter(other.scale)
            self.density = torch.nn.Parameter(other.density)
            if other.feature_type == Features.Type.SH:
                self.features_albedo = torch.nn.Parameter(other.features_albedo)
                self.features_specular = torch.nn.Parameter(other.features_specular)
            elif other.feature_type == Features.Type.NHT:
                self.features = torch.nn.Parameter(other.features)
        self.max_sh_degree = other.max_sh_degree
        self.n_active_features = other.n_active_features
        self.scene_extent = other.scene_extent
        self.progressive_training = other.progressive_training
        self.feature_dim_increase_interval = other.feature_dim_increase_interval
        self.feature_dim_increase_step = other.feature_dim_increase_step
        self.background = other.background
        self.feature_type = other.feature_type
        self.particle_feature_dim = other.particle_feature_dim
        self.ray_feature_dim = other.ray_feature_dim
        if hasattr(other, "nht_num_interpolation_points"):
            self.nht_num_interpolation_points = other.nht_num_interpolation_points
        self.validate_fields()

    def clone(self):
        other = MixtureOfGaussians(conf=self.conf, scene_extent=self.scene_extent)
        other.copy_fields(self, deepcopy=True)
        return other

    def __getitem__(self, idx):
        sliced = MixtureOfGaussians(conf=self.conf, scene_extent=self.scene_extent)
        sliced.copy_fields(self, deepcopy=False)
        sliced.positions = torch.nn.Parameter(sliced.positions[idx])
        sliced.rotation = torch.nn.Parameter(sliced.rotation[idx])
        sliced.scale = torch.nn.Parameter(sliced.scale[idx])
        sliced.density = torch.nn.Parameter(sliced.density[idx])
        if self.feature_type == Features.Type.SH:
            sliced.features_albedo = torch.nn.Parameter(sliced.features_albedo[idx])
            sliced.features_specular = torch.nn.Parameter(sliced.features_specular[idx])
        elif self.feature_type == Features.Type.NHT:
            sliced.features = torch.nn.Parameter(sliced.features[idx])
        return sliced

    def __len__(self):
        return self.positions.shape[0] if self.positions is not None else 0
