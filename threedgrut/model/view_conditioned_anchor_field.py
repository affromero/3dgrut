# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import math
import os
from enum import Enum

import numpy as np
import torch
from omegaconf import DictConfig

import threedgut_tracer
import threedgrut.model.background as background
from threedgrut.datasets.protocols import Batch
from threedgrut.model.representation import (
    GaussianRepresentation,
    REPRESENTATION_VERSION,
)
from threedgrut.utils.logger import logger
from threedgrut.utils.render import RGB2SH


ANCHOR_SOURCE_SCHEMA_VERSION = 4
ANCHOR_FIELD_CONTRACT_VERSION = 3
ANCHOR_POINTS_FORMAT = (
    "npy_float32_xyz_train_rgb_normal_layer_component_v1"
)
ANCHOR_PAYLOAD_WIDTH = 11
WORLD_PLANE_LAYER = 0
VIEW_LOCAL_RESIDUAL_LAYER = 1
VIEW_LOCAL_CLEARANCE_RADIUS_M = 1.0
GEOMETRY_VALUES_PER_OFFSET = 11
SUPPORTED_SURFACE_RESULT_STATUS = "supported"
SUPPORTED_SURFACE_RESULT_SCHEMA_VERSION = 1
EXPECTED_SURFACE_GATES_SHA256 = (
    "2473b66c7abe63411b9f4983e39e71f547aec126ba4e93983933c57931f272bf"
)


class AnchorDCMode(str, Enum):
    """View-independent anchor radiance parameterization."""

    FROZEN_SOURCE = "frozen_source"
    BOUNDED_DELTA = "bounded_delta"


class AnchorCovarianceMode(str, Enum):
    """Structural covariance treatment applied to anchor children."""

    FREE = "free"
    ALL_SURFACE = "all_surface"
    HYBRID_SURFACE = "hybrid_surface"


class AnchorOptimizationMode(str, Enum):
    """Parameter groups permitted to change during one training run."""

    FULL = "full"
    DC_ONLY = "dc_only"


def deterministic_voxel_anchors(
    points: torch.Tensor,
    colors: torch.Tensor,
    *,
    voxel_size_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce points to input-order-invariant voxel means on the CPU."""
    if voxel_size_m <= 0.0 or not math.isfinite(voxel_size_m):
        raise ValueError("voxel_size_m must be finite and positive.")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Expected points with shape [N, 3], got {tuple(points.shape)}."
        )
    if colors.shape != points.shape:
        raise ValueError(
            "Anchor colors must match point shape; "
            f"got {tuple(colors.shape)} versus {tuple(points.shape)}."
        )
    if points.shape[0] == 0:
        raise ValueError("Anchor initialization requires at least one point.")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("Anchor initialization points must be finite.")
    if not bool(torch.isfinite(colors.float()).all()):
        raise ValueError("Anchor initialization colors must be finite.")

    points_np = points.detach().cpu().double().numpy()
    colors_np = colors.detach().cpu().double().numpy()
    voxel_indices = np.floor(points_np / voxel_size_m).astype(np.int64)
    order = np.lexsort(
        (
            colors_np[:, 2],
            colors_np[:, 1],
            colors_np[:, 0],
            points_np[:, 2],
            points_np[:, 1],
            points_np[:, 0],
            voxel_indices[:, 2],
            voxel_indices[:, 1],
            voxel_indices[:, 0],
        )
    )
    sorted_voxels = voxel_indices[order]
    sorted_points = points_np[order]
    sorted_colors = colors_np[order]
    starts = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.flatnonzero(np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1))
            + 1,
        )
    )
    counts = np.diff(
        np.concatenate((starts, np.array([points_np.shape[0]], dtype=np.int64)))
    ).reshape(-1, 1)
    anchor_points = np.add.reduceat(sorted_points, starts, axis=0) / counts
    anchor_colors = np.add.reduceat(sorted_colors, starts, axis=0) / counts
    return (
        torch.from_numpy(anchor_points.astype(np.float32)),
        torch.from_numpy(anchor_colors.astype(np.float32)),
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_anchor_source_manifest_contract(
    manifest: dict[str, object],
) -> None:
    if manifest.get("schema_version") != ANCHOR_SOURCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported anchor source schema_version: "
            f"{manifest.get('schema_version')!r}."
        )
    if manifest.get("sealed_test_used") is not False:
        raise ValueError(
            "Anchor source manifest must explicitly record "
            "sealed_test_used=false."
        )
    if manifest.get("geometry_input_contract") != (
        "train_piecewise_surface_plane_vertices_plus_"
        "residual_surfel_centers"
    ):
        raise ValueError(
            "Anchor source must contain authenticated plane and residual "
            "surface layers."
        )
    if (
        manifest.get("surface_result_schema_version")
        != SUPPORTED_SURFACE_RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "Anchor source requires the supported surface-result schema."
        )
    if (
        manifest.get("surface_result_status")
        != SUPPORTED_SURFACE_RESULT_STATUS
    ):
        raise ValueError(
            "Anchor source requires a supported surface result."
        )
    if manifest.get("surface_development_rgb_used") is not False:
        raise ValueError(
            "Anchor source surface selection must not use development RGB."
        )
    if (
        manifest.get(
            "surface_development_depth_used_for_scoring_only"
        )
        is not True
    ):
        raise ValueError(
            "Anchor source requires development depth to be scoring-only."
        )
    if (
        manifest.get("surface_gates_sha256")
        != EXPECTED_SURFACE_GATES_SHA256
    ):
        raise ValueError(
            "Anchor source surface-gate digest differs from the frozen "
            "all-camera contract."
        )
    required_hashes = (
        "fold_contract_sha256",
        "training_image_names_sha256",
        "source_image_names_sha256",
        "images_txt_sha256",
        "anchor_materializer_sha256",
        "surface_mesh_sha256",
        "surface_result_sha256",
        "surface_driver_sha256",
        "surface_gates_sha256",
        "visibility_result_sha256",
        "visibility_preregistration_sha256",
        "visibility_driver_sha256",
    )
    for key in required_hashes:
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(
                f"Anchor source manifest requires a SHA-256 {key}."
            )
    color_fusion = manifest.get("color_fusion")
    if not isinstance(color_fusion, dict):
        raise ValueError("Anchor source requires color_fusion provenance.")
    if color_fusion.get("sealed_test_used") is not False:
        raise ValueError(
            "Anchor color fusion must record sealed_test_used=false."
        )
    if (
        color_fusion.get("training_image_names_sha256")
        != manifest.get("training_image_names_sha256")
    ):
        raise ValueError(
            "Anchor geometry and color provenance use different training "
            "sets."
        )
    depth_digest = color_fusion.get("source_depth_provenance_sha256")
    payload_digest = color_fusion.get("training_payload_sha256")
    if (
        not isinstance(depth_digest, str)
        or len(depth_digest) != 64
        or not isinstance(payload_digest, str)
        or len(payload_digest) != 64
    ):
        raise ValueError(
            "Anchor color fusion requires depth and training-payload "
            "digests."
        )
    if manifest.get("view_local_clearance_radius_m") != (
        VIEW_LOCAL_CLEARANCE_RADIUS_M
    ):
        raise ValueError(
            "Anchor source must preserve the supported 1 m visibility "
            "clearance."
        )
    if manifest.get("component_separated_voxelization") is not True:
        raise ValueError(
            "Anchor source must use component-separated voxelization."
        )


def _load_fold_safe_surface_source(
    manifest_path: str,
    *,
    expected_training_image_names: list[str] | None = None,
    expected_fold_contract_sha256: str | None = None,
) -> tuple[
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if not manifest_path:
        raise ValueError(
            "View-conditioned anchors require "
            "model.anchor_field.source_manifest_path."
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Anchor source manifest must be a JSON object.")
    _validate_anchor_source_manifest_contract(manifest)
    if expected_training_image_names is not None:
        actual_training_digest = _sha256_json(
            sorted(expected_training_image_names)
        )
        if (
            manifest.get("training_image_names_sha256")
            != actual_training_digest
        ):
            raise ValueError(
                "Anchor source training-image digest differs from the "
                "actual filtered training dataset."
            )
    if (
        expected_fold_contract_sha256 is not None
        and manifest.get("fold_contract_sha256")
        != expected_fold_contract_sha256
    ):
        raise ValueError(
            "Anchor source grouped-fold contract differs from the "
            "configured experiment."
        )
    points_name = manifest.get("points_file")
    points_format = manifest.get("points_format")
    points_sha256 = manifest.get("points_sha256")
    if not isinstance(points_name, str) or not points_name:
        raise ValueError("Anchor source manifest requires points_file.")
    if not isinstance(points_sha256, str) or len(points_sha256) != 64:
        raise ValueError("Anchor source manifest requires points_sha256.")
    if points_format != ANCHOR_POINTS_FORMAT:
        raise ValueError(
            f"Anchor source must use points_format={ANCHOR_POINTS_FORMAT!r}."
        )
    points_path = os.path.join(os.path.dirname(manifest_path), points_name)
    actual_sha256 = _sha256_file(points_path)
    if actual_sha256 != points_sha256:
        raise ValueError(
            "Anchor source digest mismatch: "
            f"expected {points_sha256}, got {actual_sha256}."
        )

    payload = np.load(points_path, allow_pickle=False)
    if payload.dtype != np.float32 or payload.ndim != 2:
        raise ValueError(
            "Anchor source payload must be a rank-2 float32 array."
        )
    if payload.shape[1] != ANCHOR_PAYLOAD_WIDTH:
        raise ValueError(
            "Anchor source payload must have "
            "XYZRGB-normal-layer-component shape [N, 11]."
        )
    positions = torch.from_numpy(payload[:, :3])
    colors = torch.from_numpy(payload[:, 3:6])
    normals = torch.from_numpy(payload[:, 6:9])
    layers_float = torch.from_numpy(payload[:, 9])
    components_float = torch.from_numpy(payload[:, 10])
    if int(manifest.get("point_count", -1)) != positions.shape[0]:
        raise ValueError(
            "Anchor source point_count does not match the NPZ payload."
        )
    if not bool(torch.isfinite(torch.from_numpy(payload)).all()):
        raise ValueError("Anchor source payload must contain finite values.")
    normal_norm = torch.linalg.norm(normals, dim=1)
    if not bool(torch.allclose(
        normal_norm,
        torch.ones_like(normal_norm),
        atol=1.0e-4,
        rtol=1.0e-4,
    )):
        raise ValueError("Anchor source normals must be unit length.")
    if not bool(torch.equal(layers_float, layers_float.round())):
        raise ValueError("Anchor source layers must be integer codes.")
    layers = layers_float.to(torch.uint8)
    valid_layers = (layers == WORLD_PLANE_LAYER) | (
        layers == VIEW_LOCAL_RESIDUAL_LAYER
    )
    if not bool(valid_layers.all()):
        raise ValueError("Anchor source contains an unsupported layer code.")
    if not bool(torch.equal(components_float, components_float.round())):
        raise ValueError("Anchor source components must be integer codes.")
    components = components_float.to(torch.int32)
    if bool((components < 0).any()):
        raise ValueError("Anchor source components must be non-negative.")
    plane_count = int((layers == WORLD_PLANE_LAYER).sum())
    residual_count = int((layers == VIEW_LOCAL_RESIDUAL_LAYER).sum())
    if (
        manifest.get("plane_anchor_count") != plane_count
        or manifest.get("residual_anchor_count") != residual_count
    ):
        raise ValueError(
            "Anchor source layer counts do not match the payload."
        )
    return manifest, positions, colors, normals, layers, components


def load_fold_safe_point_source(
    manifest_path: str,
    *,
    expected_training_image_names: list[str] | None = None,
    expected_fold_contract_sha256: str | None = None,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    """Load train-safe positions and colors for ordinary initialization."""
    manifest, positions, colors, _, _, _ = (
        _load_fold_safe_surface_source(
            manifest_path,
            expected_training_image_names=expected_training_image_names,
            expected_fold_contract_sha256=expected_fold_contract_sha256,
        )
    )
    return manifest, positions, colors


class _GaussianView:
    """Ephemeral Gaussian tensors materialized for one continuous camera pose."""

    def __init__(
        self,
        *,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
        densities: torch.Tensor,
        features: torch.Tensor,
        gaussian_background: torch.nn.Module,
    ) -> None:
        self.positions = positions
        self.rotation = rotations
        self.scale = scales
        self.density = densities
        self.features_albedo = features
        self.features_specular = torch.empty(
            (positions.shape[0], 0),
            device=positions.device,
            dtype=positions.dtype,
        )
        self.background = gaussian_background
        self.n_active_features = 0

    @property
    def num_gaussians(self) -> int:
        return int(self.positions.shape[0])

    def get_rotation(self) -> torch.Tensor:
        return self.rotation

    def get_scale(self) -> torch.Tensor:
        return self.scale

    def get_density(self) -> torch.Tensor:
        return self.density

    def get_features(self) -> torch.Tensor:
        return self.features_albedo


class ViewConditionedAnchorField(torch.nn.Module):
    """Held-out-safe view-conditioned Gaussian field over fixed voxel anchors."""

    representation_name = GaussianRepresentation.VIEW_CONDITIONED_ANCHOR.value
    supports_static_export = False
    supports_geometry_only_optimizer = False
    supports_visibility_optimizer = False
    supports_progressive_features = False

    def __init__(
        self,
        conf: DictConfig,
        *,
        scene_extent: float | None = None,
    ) -> None:
        super().__init__()
        if str(conf.render.method) != "3dgut":
            raise ValueError(
                "View-conditioned anchors currently require render.method=3dgut."
            )
        if str(conf.optimizer.type) != "adam":
            raise ValueError(
                "View-conditioned anchors require optimizer.type=adam."
            )
        if bool(conf.with_gui) or bool(conf.with_viser_gui):
            raise ValueError(
                "View-conditioned anchor screening requires both GUI modes "
                "to be disabled."
            )
        if bool(conf.camera_residual.enabled):
            raise ValueError(
                "The controlled anchor screen requires camera_residual.enabled=false."
            )
        if str(conf.dataset.get("shutter_type", "GLOBAL")) != "GLOBAL":
            raise ValueError(
                "The controlled anchor screen requires GLOBAL shutter."
            )
        if bool(conf.loss.get("use_depth", False)):
            raise ValueError(
                "The first anchor screen is image-only; loss.use_depth "
                "must remain false."
            )
        geometry_only = conf.loss.get("geometry_only_pass", {})
        if bool(geometry_only.get("enabled", False)):
            raise ValueError(
                "The controlled anchor screen does not support the ordinary "
                "geometry-only sparse optimizer."
            )
        if bool(conf.export_ply.enabled) or bool(conf.export_usd.enabled):
            raise ValueError(
                "View-conditioned anchor screening is checkpoint-native; "
                "static PLY/USD export must remain disabled until baking."
            )
        if int(conf.dataset.get("blur_samples", 1)) != 1:
            raise ValueError(
                "The controlled anchor screen requires dataset.blur_samples=1."
            )
        if bool(conf.dataset.get("rs_ray_injection", False)):
            raise ValueError(
                "The controlled pinhole anchor screen requires "
                "dataset.rs_ray_injection=false so projection and ray "
                "generation share the physical camera pose."
            )
        carrier_enabled = (
            bool(conf.model.get("use_gabor_carrier", False))
            or bool(conf.model.get("use_hermite_carrier", False))
            or bool(conf.model.get("use_siren_carrier", False))
        )
        if carrier_enabled:
            raise ValueError(
                "The controlled anchor screen does not support radiance "
                "carrier coefficients."
            )
        anchor_conf = conf.model.anchor_field
        self.conf = conf
        self.scene_extent = scene_extent
        self.device = "cuda"
        self.voxel_size_m = float(anchor_conf.voxel_size_m)
        self.offsets_per_anchor = int(anchor_conf.offsets_per_anchor)
        self.feature_dim = int(anchor_conf.feature_dim)
        self.hidden_dim = int(anchor_conf.hidden_dim)
        self.max_offset_fraction = float(anchor_conf.max_offset_fraction)
        self.initial_scale_fraction = float(
            anchor_conf.initial_scale_fraction
        )
        self.min_scale_fraction = float(anchor_conf.min_scale_fraction)
        self.max_scale_fraction = float(anchor_conf.max_scale_fraction)
        self.coverage_scale_fraction = float(
            anchor_conf.coverage_scale_fraction
        )
        self.coverage_max_scale_fraction = float(
            anchor_conf.coverage_max_scale_fraction
        )
        self.coverage_initial_density = float(
            anchor_conf.coverage_initial_density
        )
        self.coverage_max_density = float(
            anchor_conf.coverage_max_density
        )
        self.detail_initial_density = float(
            anchor_conf.detail_initial_density
        )
        self.use_view_local_residual_gate = bool(
            anchor_conf.use_view_local_residual_gate
        )
        self.view_local_clearance_radius_m = float(
            anchor_conf.view_local_clearance_radius_m
        )
        self.dc_mode = AnchorDCMode(str(anchor_conf.dc_mode))
        self.dc_delta_max_sh = float(anchor_conf.dc_delta_max_sh)
        self.covariance_mode = AnchorCovarianceMode(
            str(anchor_conf.covariance_mode)
        )
        self.optimization_mode = AnchorOptimizationMode(
            str(anchor_conf.optimization_mode)
        )
        self.normal_scale_fraction = float(
            anchor_conf.normal_scale_fraction
        )
        self.radiance_coefficient_count = (
            int(conf.render.particle_radiance_sph_degree) + 1
        ) ** 2
        if self.offsets_per_anchor != 4:
            raise ValueError(
                "The preregistered S085 anchor treatment requires "
                "offsets_per_anchor=4."
            )
        if self.feature_dim < 3 or self.hidden_dim < 4:
            raise ValueError(
                "Anchor feature_dim must be at least 3 and hidden_dim at least 4."
            )
        if not 0.0 < self.max_offset_fraction < 1.0:
            raise ValueError(
                "Anchor max_offset_fraction must be in the open interval (0, 1)."
            )
        if not (
            0.0
            < self.min_scale_fraction
            <= self.initial_scale_fraction
            <= self.max_scale_fraction
            < 1.0
        ):
            raise ValueError(
                "Anchor scale fractions must satisfy "
                "0 < min <= initial <= max < 1."
            )
        if not (
            self.initial_scale_fraction
            <= self.coverage_scale_fraction
            <= self.coverage_max_scale_fraction
            < 1.0
        ):
            raise ValueError(
                "Coverage scale fractions must satisfy "
                "detail initial <= coverage initial <= coverage max < 1."
            )
        if not (
            0.0
            < self.coverage_initial_density
            < self.coverage_max_density
            < 1.0
        ):
            raise ValueError(
                "Coverage densities must satisfy "
                "0 < initial < max < 1."
            )
        if not 0.0 < self.detail_initial_density < 1.0:
            raise ValueError(
                "detail_initial_density must be in the open interval (0, 1)."
            )
        if (
            not math.isfinite(self.view_local_clearance_radius_m)
            or self.view_local_clearance_radius_m <= 0.0
        ):
            raise ValueError(
                "view_local_clearance_radius_m must be finite and positive."
            )
        if (
            self.use_view_local_residual_gate
            and self.view_local_clearance_radius_m
            != VIEW_LOCAL_CLEARANCE_RADIUS_M
        ):
            raise ValueError(
                "The supported residual visibility treatment requires the "
                "frozen 1 m clearance radius."
            )
        if (
            not math.isfinite(self.dc_delta_max_sh)
            or self.dc_delta_max_sh <= 0.0
        ):
            raise ValueError(
                "dc_delta_max_sh must be finite and positive."
            )
        if (
            self.optimization_mode == AnchorOptimizationMode.DC_ONLY
            and self.dc_mode != AnchorDCMode.BOUNDED_DELTA
        ):
            raise ValueError(
                "dc_only optimization requires dc_mode=bounded_delta."
            )
        if (
            not math.isfinite(self.normal_scale_fraction)
            or not 0.0
            < self.normal_scale_fraction
            <= self.min_scale_fraction
        ):
            raise ValueError(
                "normal_scale_fraction must satisfy "
                "0 < normal <= min_scale_fraction."
            )

        self.register_buffer("anchor_positions", torch.empty((0, 3)))
        self.register_buffer("anchor_base_dc", torch.empty((0, 3)))
        self.register_buffer("anchor_normals", torch.empty((0, 3)))
        self.register_buffer("anchor_layers", torch.empty((0,), dtype=torch.uint8))
        self.register_buffer(
            "anchor_components",
            torch.empty((0,), dtype=torch.int32),
        )
        self.register_buffer(
            "anchor_base_rotations",
            torch.empty((0, 4)),
        )
        self.anchor_features = torch.nn.Parameter(
            torch.empty((0, self.feature_dim))
        )
        self.anchor_dc_delta_raw = torch.nn.Parameter(
            torch.empty((0, 3))
        )
        self.geometry_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(
                self.hidden_dim,
                self.offsets_per_anchor * GEOMETRY_VALUES_PER_OFFSET,
            ),
        )
        self.appearance_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim + 4, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(
                self.hidden_dim,
                self.offsets_per_anchor * 3,
            ),
        )
        self.geometry_decoder.to(self.device)
        self.appearance_decoder.to(self.device)
        self.background = background.make(
            self.conf.model.background.name,
            self.conf.model.background,
        )
        self.renderer = threedgut_tracer.Tracer(conf)
        self.optimizer: torch.optim.Adam | None = None
        self.n_active_features = 0
        self.max_n_features = 0
        self.progressive_training = False
        self._last_grad_norms: dict[str, torch.Tensor] = {}
        self.anchor_source_manifest: dict[str, object] | None = None
        self._initialize_decoder_parameters()

    @property
    def num_anchors(self) -> int:
        return int(self.anchor_positions.shape[0])

    @property
    def num_gaussians(self) -> int:
        return self.num_anchors * self.offsets_per_anchor

    def _initialize_decoder_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.geometry_decoder[0].weight)
        torch.nn.init.zeros_(self.geometry_decoder[0].bias)
        torch.nn.init.normal_(
            self.geometry_decoder[2].weight,
            mean=0.0,
            std=1.0e-4,
        )
        geometry_bias = torch.zeros(
            self.offsets_per_anchor,
            GEOMETRY_VALUES_PER_OFFSET,
        )
        tetrahedron = torch.tensor(
            (
                (1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
            )
        ) / math.sqrt(3.0)
        offset_ratio = 0.35
        geometry_bias[:, :3] = torch.atanh(
            tetrahedron
            * (offset_ratio / self.max_offset_fraction)
        )
        geometry_bias[0, 3:6] = math.log(
            self.voxel_size_m * self.coverage_scale_fraction
        )
        geometry_bias[1:, 3:6] = math.log(
            self.voxel_size_m * self.initial_scale_fraction
        )
        if self.covariance_mode == AnchorCovarianceMode.ALL_SURFACE:
            geometry_bias[:, 5] = math.log(
                self.voxel_size_m * self.normal_scale_fraction
            )
        elif self.covariance_mode == AnchorCovarianceMode.HYBRID_SURFACE:
            geometry_bias[1:, 5] = math.log(
                self.voxel_size_m * self.normal_scale_fraction
            )
        geometry_bias[:, 6] = 1.0
        coverage_density_ratio = (
            self.coverage_initial_density / self.coverage_max_density
        )
        geometry_bias[0, 10] = math.log(
            coverage_density_ratio / (1.0 - coverage_density_ratio)
        )
        geometry_bias[1:, 10] = math.log(
            self.detail_initial_density
            / (1.0 - self.detail_initial_density)
        )
        with torch.no_grad():
            self.geometry_decoder[2].bias.copy_(geometry_bias.reshape(-1))

        torch.nn.init.xavier_uniform_(self.appearance_decoder[0].weight)
        torch.nn.init.zeros_(self.appearance_decoder[0].bias)
        torch.nn.init.normal_(
            self.appearance_decoder[2].weight,
            mean=0.0,
            std=1.0e-4,
        )
        torch.nn.init.zeros_(self.appearance_decoder[2].bias)

    def initialize_from_source_manifest(
        self,
        *,
        training_image_names: list[str] | None = None,
        fold_contract_sha256: str | None = None,
    ) -> None:
        """Initialize fixed anchors from a fold-bound train-only source."""
        anchor_conf = self.conf.model.anchor_field
        (
            manifest,
            points,
            colors,
            normals,
            layers,
            components,
        ) = _load_fold_safe_surface_source(
            str(anchor_conf.source_manifest_path),
            expected_training_image_names=training_image_names,
            expected_fold_contract_sha256=fold_contract_sha256,
        )
        source_voxel_size = manifest.get("voxel_size_m")
        if (
            not isinstance(source_voxel_size, (float, int))
            or float(source_voxel_size) != self.voxel_size_m
        ):
            raise ValueError(
                "Anchor source voxel_size_m differs from model configuration."
            )
        anchors = points
        anchor_colors = colors
        anchor_colors = anchor_colors.clamp(0.0, 255.0) / 255.0
        anchor_dc = RGB2SH(anchor_colors)
        features = torch.zeros(
            (anchors.shape[0], self.feature_dim),
            dtype=torch.float32,
        )
        features[:, :3] = anchor_dc
        generator = torch.Generator(device="cpu").manual_seed(
            int(self.conf.seed_initialization)
        )
        if self.feature_dim > 3:
            features[:, 3:] = (
                torch.randn(
                    (anchors.shape[0], self.feature_dim - 3),
                    generator=generator,
                )
                * 0.01
            )
        self.anchor_positions = anchors.to(self.device)
        self.anchor_base_dc = anchor_dc.to(self.device)
        self.anchor_normals = normals.to(self.device)
        self.anchor_layers = layers.to(self.device)
        self.anchor_components = components.to(self.device)
        self.anchor_base_rotations = self._normal_aligned_rotations(
            self.anchor_normals
        )
        self.anchor_features = torch.nn.Parameter(features.to(self.device))
        self.anchor_dc_delta_raw = torch.nn.Parameter(
            torch.zeros(
                (anchors.shape[0], 3),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.anchor_source_manifest = manifest
        logger.info(
            "Initialized view-conditioned anchor field: "
            f"{points.shape[0]} source points -> {self.num_anchors} anchors -> "
            f"{self.num_gaussians} Gaussians."
        )

    @staticmethod
    def _normal_aligned_rotations(normals: torch.Tensor) -> torch.Tensor:
        """Return WXYZ rotations mapping local Z onto each unit normal."""
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(
                "Anchor normals must have shape [N, 3]."
            )
        normal_norm = torch.linalg.norm(normals, dim=1, keepdim=True)
        if not bool(torch.isfinite(normal_norm).all()) or bool(
            (normal_norm <= 1.0e-8).any()
        ):
            raise ValueError("Anchor normals must be finite and non-zero.")
        unit = normals / normal_norm
        local_z = torch.zeros_like(unit)
        local_z[:, 2] = 1.0
        vector = torch.linalg.cross(local_z, unit, dim=1)
        scalar = 1.0 + unit[:, 2:3]
        quaternion = torch.cat((scalar, vector), dim=1)
        opposite = scalar[:, 0] <= 1.0e-7
        quaternion[opposite] = torch.tensor(
            (0.0, 1.0, 0.0, 0.0),
            dtype=quaternion.dtype,
            device=quaternion.device,
        )
        return torch.nn.functional.normalize(quaternion, dim=1)

    def validate_training_contract(
        self,
        *,
        training_image_names: list[str],
        fold_contract_sha256: str,
    ) -> None:
        """Authenticate an embedded checkpoint manifest against this run."""
        manifest = self.anchor_source_manifest
        if manifest is None:
            raise ValueError("Anchor checkpoint has no source manifest.")
        if manifest.get("training_image_names_sha256") != _sha256_json(
            sorted(training_image_names)
        ):
            raise ValueError(
                "Anchor checkpoint training-image digest differs from the "
                "actual filtered training dataset."
            )
        if manifest.get("fold_contract_sha256") != fold_contract_sha256:
            raise ValueError(
                "Anchor checkpoint grouped-fold contract differs from this run."
            )

    def _geometry(self) -> tuple[torch.Tensor, ...]:
        if self.num_anchors == 0:
            raise RuntimeError("Anchor field has not been initialized.")
        raw = self.geometry_decoder(self.anchor_features).reshape(
            self.num_anchors,
            self.offsets_per_anchor,
            GEOMETRY_VALUES_PER_OFFSET,
        )
        max_offset = self.voxel_size_m * self.max_offset_fraction
        offsets = torch.tanh(raw[..., :3]) * max_offset
        positions = (
            self.anchor_positions[:, None, :] + offsets
        ).reshape(-1, 3)
        scale_maxima = torch.full(
            (1, self.offsets_per_anchor, 1),
            self.voxel_size_m * self.max_scale_fraction,
            dtype=raw.dtype,
            device=raw.device,
        )
        scale_maxima[:, 0, :] = (
            self.voxel_size_m * self.coverage_max_scale_fraction
        )
        scales_by_anchor = torch.minimum(
            torch.exp(raw[..., 3:6]).clamp_min(
                self.voxel_size_m * self.min_scale_fraction
            ),
            scale_maxima,
        )
        learned_rotations = torch.nn.functional.normalize(
            raw[..., 6:10],
            dim=-1,
        )
        surface_rotations = self.anchor_base_rotations[:, None, :].expand(
            -1,
            self.offsets_per_anchor,
            -1,
        )
        if self.covariance_mode == AnchorCovarianceMode.ALL_SURFACE:
            scales_by_anchor = scales_by_anchor.clone()
            scales_by_anchor[..., 2] = (
                self.voxel_size_m * self.normal_scale_fraction
            )
            rotations_by_anchor = surface_rotations
        elif self.covariance_mode == AnchorCovarianceMode.HYBRID_SURFACE:
            coverage_scale = torch.exp(
                raw[:, :1, 3:6].mean(dim=-1, keepdim=True)
            ).clamp(
                min=self.voxel_size_m * self.min_scale_fraction,
                max=self.voxel_size_m * self.coverage_max_scale_fraction,
            )
            coverage_scales = coverage_scale.expand(-1, -1, 3)
            detail_scales = scales_by_anchor[:, 1:].clone()
            detail_scales[..., 2] = (
                self.voxel_size_m * self.normal_scale_fraction
            )
            scales_by_anchor = torch.cat(
                (coverage_scales, detail_scales),
                dim=1,
            )
            rotations_by_anchor = torch.cat(
                (
                    learned_rotations[:, :1],
                    surface_rotations[:, 1:],
                ),
                dim=1,
            )
        else:
            rotations_by_anchor = learned_rotations
        rotations = rotations_by_anchor.reshape(-1, 4)
        scales = scales_by_anchor.reshape(-1, 3)
        density_maxima = torch.ones(
            (1, self.offsets_per_anchor, 1),
            dtype=raw.dtype,
            device=raw.device,
        )
        density_maxima[:, 0, :] = self.coverage_max_density
        densities = (
            torch.sigmoid(raw[..., 10:11]) * density_maxima
        ).reshape(-1, 1)
        return positions, rotations, scales, densities

    def _camera_center(self, batch: Batch) -> torch.Tensor:
        if batch.T_to_world.shape[0] != 1 or batch.rays_ori.shape[0] != 1:
            raise ValueError(
                "View-conditioned anchor screen requires one global-shutter "
                "camera sample per batch; blur-expanded batches are unsupported."
            )
        if batch.rays_in_world_space:
            ray_origins = batch.rays_ori.reshape(-1, 3)
            camera_center = ray_origins[0]
            max_origin_delta = (
                ray_origins - camera_center[None, :]
            ).abs().max()
            if float(max_origin_delta) > 1.0e-5:
                raise ValueError(
                    "World-space anchor conditioning requires one shared "
                    "global-shutter ray origin."
                )
            return camera_center
        if batch.T_to_world_end is not None and not torch.equal(
            batch.T_to_world,
            batch.T_to_world_end,
        ):
            raise ValueError(
                "View-conditioned anchor screen requires global-shutter poses."
            )
        return batch.T_to_world[0, :3, 3]

    def _materialize(self, camera_center: torch.Tensor) -> _GaussianView:
        positions, rotations, scales, densities = self._geometry()
        anchor_direction = camera_center[None, :] - self.anchor_positions
        distance = torch.linalg.norm(anchor_direction, dim=1, keepdim=True)
        direction = anchor_direction / distance.clamp_min(1.0e-6)
        log_distance = torch.log(distance.clamp_min(1.0e-4))
        appearance_input = torch.cat(
            (
                self.anchor_features,
                direction,
                log_distance,
            ),
            dim=1,
        )
        residual = (
            torch.tanh(self.appearance_decoder(appearance_input))
            .reshape(self.num_anchors, self.offsets_per_anchor, 3)
            * 0.25
        )
        base_dc = self.anchor_base_dc
        if self.dc_mode == AnchorDCMode.BOUNDED_DELTA:
            base_dc = base_dc + (
                torch.tanh(self.anchor_dc_delta_raw)
                * self.dc_delta_max_sh
            )
        features = (
            base_dc[:, None, :] + residual
        ).reshape(-1, 3)
        radiance = torch.zeros(
            (
                features.shape[0],
                3 * self.radiance_coefficient_count,
            ),
            dtype=features.dtype,
            device=features.device,
        )
        radiance[:, :3] = features
        if self.use_view_local_residual_gate:
            residual_layer = (
                self.anchor_layers == VIEW_LOCAL_RESIDUAL_LAYER
            )
            within_clearance = (
                distance[:, 0] <= self.view_local_clearance_radius_m
            )
            suppressed_anchor = residual_layer & within_clearance
            visible_density = (
                ~suppressed_anchor
            ).repeat_interleave(self.offsets_per_anchor)[:, None]
            densities = densities * visible_density.to(densities.dtype)
        return _GaussianView(
            positions=positions.contiguous(),
            rotations=rotations.contiguous(),
            scales=scales.contiguous(),
            densities=densities.contiguous(),
            features=radiance.contiguous(),
            gaussian_background=self.background,
        )

    def forward(
        self,
        batch: Batch,
        train: bool = False,
        frame_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        view = self._materialize(self._camera_center(batch))
        return self.renderer.render(
            view,
            batch,
            train=train,
            frame_id=frame_id,
        )

    @torch.no_grad()
    def build_acc(self, rebuild: bool = True) -> None:
        self.renderer.build_acc(self, rebuild)

    def get_positions(self) -> torch.Tensor:
        return self._geometry()[0]

    def get_rotation(self, preactivation: bool = False) -> torch.Tensor:
        if preactivation:
            raise ValueError(
                "Anchor rotations have no persistent preactivation tensor."
            )
        return self._geometry()[1]

    def get_scale(self, preactivation: bool = False) -> torch.Tensor:
        if preactivation:
            raise ValueError(
                "Anchor scales have no persistent preactivation tensor."
            )
        return self._geometry()[2]

    def get_density(self, preactivation: bool = False) -> torch.Tensor:
        if preactivation:
            raise ValueError(
                "Anchor densities have no persistent preactivation tensor."
            )
        return self._geometry()[3]

    def diagnostic_parameters(self) -> dict[str, torch.nn.Parameter]:
        """Return stable trainable tensors for model-level gradient diagnostics."""
        parameters = {
            "anchor_features": self.anchor_features,
            "anchor_dc_delta_raw": self.anchor_dc_delta_raw,
        }
        for prefix, module in (
            ("geometry_decoder", self.geometry_decoder),
            ("appearance_decoder", self.appearance_decoder),
        ):
            for name, parameter in module.named_parameters():
                parameters[f"{prefix}.{name}"] = parameter
        return parameters

    def geometry_requires_grad(self) -> bool:
        return any(
            parameter.requires_grad
            for parameter in self.geometry_decoder.parameters()
        )

    def setup_optimizer(
        self,
        state_dict: dict[str, object] | None = None,
    ) -> None:
        anchor_conf = self.conf.model.anchor_field
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if self.optimization_mode == AnchorOptimizationMode.DC_ONLY:
            self.anchor_dc_delta_raw.requires_grad_(True)
            groups = [
                {
                    "params": [self.anchor_dc_delta_raw],
                    "name": "anchor_dc_delta_raw",
                    "lr": float(anchor_conf.dc_lr),
                }
            ]
        else:
            self.anchor_features.requires_grad_(True)
            for parameter in self.geometry_decoder.parameters():
                parameter.requires_grad_(True)
            for parameter in self.appearance_decoder.parameters():
                parameter.requires_grad_(True)
            groups = [
            {
                "params": [self.anchor_features],
                "name": "anchor_features",
                "lr": float(anchor_conf.feature_lr),
            },
            {
                "params": self.geometry_decoder.parameters(),
                "name": "geometry_decoder",
                "lr": float(anchor_conf.geometry_lr),
            },
            {
                "params": self.appearance_decoder.parameters(),
                "name": "appearance_decoder",
                "lr": float(anchor_conf.appearance_lr),
            },
            ]
            if self.dc_mode == AnchorDCMode.BOUNDED_DELTA:
                self.anchor_dc_delta_raw.requires_grad_(True)
                groups.append(
                    {
                        "params": [self.anchor_dc_delta_raw],
                        "name": "anchor_dc_delta_raw",
                        "lr": float(anchor_conf.dc_lr),
                    }
                )
        self.optimizer = torch.optim.Adam(
            groups,
            lr=0.0,
            betas=tuple(float(value) for value in self.conf.optimizer.betas),
            eps=float(self.conf.optimizer.eps),
        )
        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)

    def scheduler_step(self, step: int) -> None:
        del step

    def freeze_gaussians(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def get_model_parameters(self) -> dict[str, object]:
        if self.optimizer is None:
            raise RuntimeError(
                "Anchor optimizer must be initialized before checkpointing."
            )
        if self.anchor_source_manifest is None:
            raise RuntimeError(
                "Anchor source manifest is unavailable during checkpointing."
            )
        return {
            "representation": {
                "name": self.representation_name,
                "version": REPRESENTATION_VERSION,
            },
            "model_state": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scene_extent": self.scene_extent,
            "config": self.conf,
            "anchor_count": self.num_anchors,
            "offsets_per_anchor": self.offsets_per_anchor,
            "anchor_source_manifest": self.anchor_source_manifest,
            "anchor_field_contract": self._anchor_field_contract(),
        }

    def _anchor_field_contract(self) -> dict[str, object]:
        """Return the checkpoint-critical structural treatment contract."""
        return {
            "version": ANCHOR_FIELD_CONTRACT_VERSION,
            "use_view_local_residual_gate": (
                self.use_view_local_residual_gate
            ),
            "view_local_clearance_radius_m": (
                self.view_local_clearance_radius_m
            ),
            "dc_mode": self.dc_mode.value,
            "dc_delta_max_sh": self.dc_delta_max_sh,
            "covariance_mode": self.covariance_mode.value,
            "optimization_mode": self.optimization_mode.value,
            "coverage_child_index": 0,
            "detail_child_indices": [1, 2, 3],
            "normal_scale_fraction": self.normal_scale_fraction,
        }

    def init_from_checkpoint(
        self,
        checkpoint: dict[str, object],
        setup_optimizer: bool = True,
    ) -> None:
        state = checkpoint.get("model_state")
        anchor_count = checkpoint.get("anchor_count")
        source_manifest = checkpoint.get("anchor_source_manifest")
        if not isinstance(state, dict):
            raise ValueError("Anchor checkpoint is missing model_state.")
        if not isinstance(anchor_count, int) or anchor_count <= 0:
            raise ValueError("Anchor checkpoint has invalid anchor_count.")
        if checkpoint.get("offsets_per_anchor") != self.offsets_per_anchor:
            raise ValueError(
                "Anchor checkpoint offsets_per_anchor differs from config."
            )
        if not isinstance(source_manifest, dict):
            raise ValueError(
                "Anchor checkpoint is missing anchor_source_manifest."
            )
        if checkpoint.get("anchor_field_contract") != (
            self._anchor_field_contract()
        ):
            raise ValueError(
                "Anchor checkpoint structural treatment contract differs "
                "from the configured representation."
            )
        _validate_anchor_source_manifest_contract(source_manifest)
        self.anchor_positions = torch.empty(
            (anchor_count, 3),
            device=self.device,
        )
        self.anchor_base_dc = torch.empty(
            (anchor_count, 3),
            device=self.device,
        )
        self.anchor_normals = torch.empty(
            (anchor_count, 3),
            device=self.device,
        )
        self.anchor_layers = torch.empty(
            (anchor_count,),
            dtype=torch.uint8,
            device=self.device,
        )
        self.anchor_components = torch.empty(
            (anchor_count,),
            dtype=torch.int32,
            device=self.device,
        )
        self.anchor_base_rotations = torch.empty(
            (anchor_count, 4),
            device=self.device,
        )
        self.anchor_features = torch.nn.Parameter(
            torch.empty(
                (anchor_count, self.feature_dim),
                device=self.device,
            )
        )
        self.anchor_dc_delta_raw = torch.nn.Parameter(
            torch.empty(
                (anchor_count, 3),
                device=self.device,
            )
        )
        self.load_state_dict(state, strict=True)
        self.scene_extent = checkpoint.get("scene_extent")
        self.anchor_source_manifest = source_manifest
        if setup_optimizer:
            optimizer_state = checkpoint.get("optimizer")
            if not isinstance(optimizer_state, dict):
                raise ValueError(
                    "Anchor checkpoint is missing optimizer state."
                )
            self.setup_optimizer(optimizer_state)

    def export_ply(self, path: str) -> None:
        raise RuntimeError(
            "View-conditioned anchor checkpoints cannot be exported as a "
            f"static PLY without an explicit bake step: {path}"
        )
