# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION
# & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

SH_DC_NORMALIZATION = 0.28209479177387814
SUPPORTED_APPEARANCE_RANK = 2


def _scalar_index(
    value: int | torch.Tensor,
    *,
    name: str,
    lower_bound: int,
    upper_bound: int,
) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain exactly one value.")
        numeric_value = float(value.detach().cpu().item())
    else:
        numeric_value = float(value)
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"{name} must be a finite integer.")
    index = int(numeric_value)
    if index < lower_bound or index > upper_bound:
        raise ValueError(
            f"{name}={index} is outside the registered interval "
            f"[{lower_bound}, {upper_bound}]."
        )
    return index


class AcquisitionGaussianView:
    """Ephemeral radiance view over unchanged Gaussian geometry."""

    def __init__(
        self,
        *,
        positions: torch.Tensor,
        rotation: torch.Tensor,
        scale: torch.Tensor,
        density: torch.Tensor,
        features: torch.Tensor,
        background: torch.nn.Module,
        n_active_features: int,
    ) -> None:
        self.positions = positions
        self.rotation = rotation
        self.scale = scale
        self.density = density
        self.features = features
        self.background = background
        self.n_active_features = n_active_features

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
        return self.features


class AcquisitionAppearance(torch.nn.Module):
    """Low-rank surface response driven by interpolated acquisition state."""

    def __init__(
        self,
        *,
        num_gaussians: int,
        num_cameras: int,
        num_knots: int,
        min_sequence_idx: int,
        max_sequence_idx: int,
        max_rgb_delta: float,
        magnitude_regularization: float,
        curvature_regularization: float,
        device: str | torch.device,
        dtype: torch.dtype,
        rank: int = SUPPORTED_APPEARANCE_RANK,
    ) -> None:
        super().__init__()
        if num_gaussians <= 0:
            raise ValueError("num_gaussians must be positive.")
        if num_cameras <= 0:
            raise ValueError("num_cameras must be positive.")
        if num_knots < 3:
            raise ValueError("num_knots must be at least three.")
        if min_sequence_idx < 0:
            raise ValueError("min_sequence_idx must be non-negative.")
        if max_sequence_idx <= min_sequence_idx:
            raise ValueError(
                "max_sequence_idx must exceed min_sequence_idx."
            )
        if not math.isfinite(max_rgb_delta) or max_rgb_delta <= 0.0:
            raise ValueError("max_rgb_delta must be finite and positive.")
        if rank != SUPPORTED_APPEARANCE_RANK:
            raise ValueError(
                "Acquisition appearance currently supports rank two only."
            )
        regularizers = (
            magnitude_regularization,
            curvature_regularization,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in regularizers):
            raise ValueError(
                "Acquisition regularization weights must be finite and "
                "non-negative."
            )

        self.num_cameras = num_cameras
        self.num_knots = num_knots
        self.min_sequence_idx = min_sequence_idx
        self.max_sequence_idx = max_sequence_idx
        self.max_rgb_delta = max_rgb_delta
        self.magnitude_regularization = magnitude_regularization
        self.curvature_regularization = curvature_regularization
        self.rank = rank

        initial_directions = torch.tensor(
            (
                (0.5, 0.5, 0.5),
                (0.5, 0.0, -0.5),
            ),
            device=device,
            dtype=dtype,
        )
        initial_direction_raw = torch.atanh(initial_directions)
        self.direction_raw = torch.nn.Parameter(
            initial_direction_raw[None].repeat(num_gaussians, 1, 1)
        )
        self.temporal_raw = torch.nn.Parameter(
            torch.zeros(
                (num_cameras, rank, num_knots),
                device=device,
                dtype=dtype,
            )
        )

    def sample_coefficients(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        camera = _scalar_index(
            camera_idx,
            name="camera_idx",
            lower_bound=0,
            upper_bound=self.num_cameras - 1,
        )
        sequence = _scalar_index(
            sequence_idx,
            name="sequence_idx",
            lower_bound=self.min_sequence_idx,
            upper_bound=self.max_sequence_idx,
        )
        scaled = (
            float(sequence - self.min_sequence_idx)
            / float(self.max_sequence_idx - self.min_sequence_idx)
            * float(self.num_knots - 1)
        )
        lower = int(math.floor(scaled))
        upper = min(lower + 1, self.num_knots - 1)
        weight = scaled - float(lower)
        table = torch.tanh(self.temporal_raw[camera])
        coefficients = table[:, lower] * (1.0 - weight)
        coefficients = coefficients + table[:, upper] * weight
        return coefficients

    def rgb_delta(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self.sample_coefficients(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )
        directions = torch.tanh(self.direction_raw)
        delta = torch.einsum("nrk,r->nk", directions, coefficients)
        return delta * (self.max_rgb_delta / float(self.rank))

    def materialize(
        self,
        *,
        positions: torch.Tensor,
        rotation: torch.Tensor,
        scale: torch.Tensor,
        density: torch.Tensor,
        features_albedo: torch.Tensor,
        features_specular: torch.Tensor,
        background: torch.nn.Module,
        n_active_features: int,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> AcquisitionGaussianView:
        if features_albedo.shape != (self.direction_raw.shape[0], 3):
            raise ValueError(
                "Acquisition appearance no longer matches Gaussian topology."
            )
        albedo = features_albedo + self.rgb_delta(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        ) / SH_DC_NORMALIZATION
        features = torch.cat((albedo, features_specular), dim=1)
        return AcquisitionGaussianView(
            positions=positions,
            rotation=rotation,
            scale=scale,
            density=density,
            features=features.contiguous(),
            background=background,
            n_active_features=n_active_features,
        )

    def get_regularization_loss(self) -> torch.Tensor:
        temporal = torch.tanh(self.temporal_raw)
        magnitude = temporal.square().mean()
        second_difference = (
            temporal[..., 2:]
            - 2.0 * temporal[..., 1:-1]
            + temporal[..., :-2]
        )
        curvature = second_difference.square().mean()
        return (
            self.magnitude_regularization * magnitude
            + self.curvature_regularization * curvature
        )
