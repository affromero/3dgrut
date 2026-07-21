# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION
# & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from threedgrut.model.acquisition_appearance import (
    SH_DC_NORMALIZATION,
    AcquisitionGaussianView,
    _scalar_index,
)


SUPPORTED_SURFACE_SPLINE_RANK = 2


class SurfaceAcquisitionSpline(torch.nn.Module):
    """Surface-local radiance innovations over acquisition coordinate."""

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
        rank: int = SUPPORTED_SURFACE_SPLINE_RANK,
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
            raise ValueError("max_sequence_idx must exceed min_sequence_idx.")
        if not math.isfinite(max_rgb_delta) or max_rgb_delta <= 0.0:
            raise ValueError("max_rgb_delta must be finite and positive.")
        if rank != SUPPORTED_SURFACE_SPLINE_RANK:
            raise ValueError(
                "Surface acquisition spline currently supports rank two only."
            )
        regularizers = (
            magnitude_regularization,
            curvature_regularization,
        )
        if any(
            not math.isfinite(value) or value < 0.0 for value in regularizers
        ):
            raise ValueError(
                "Surface spline regularization weights must be finite and "
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
        self.direction_raw = torch.nn.Parameter(
            torch.atanh(initial_directions)[None].repeat(
                num_gaussians,
                1,
                1,
            )
        )
        self.coefficient_knots = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(
                        (num_gaussians, rank),
                        device=device,
                        dtype=dtype,
                    )
                )
                for _ in range(num_cameras * num_knots)
            ]
        )

    def _knot(self, camera_idx: int, knot_idx: int) -> torch.Tensor:
        return self.coefficient_knots[camera_idx * self.num_knots + knot_idx]

    def _interval(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> tuple[int, int, int, float]:
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
        lower = math.floor(scaled)
        upper = min(lower + 1, self.num_knots - 1)
        return camera, lower, upper, scaled - float(lower)

    def sample_coefficients(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        camera, lower, upper, weight = self._interval(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )
        lower_value = torch.tanh(self._knot(camera, lower))
        upper_value = torch.tanh(self._knot(camera, upper))
        return lower_value * (1.0 - weight) + upper_value * weight

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
        delta = torch.einsum("nrk,nr->nk", directions, coefficients)
        return delta * (self.max_rgb_delta / float(self.rank))

    def get_local_regularization_loss(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        camera, lower, upper, _ = self._interval(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )
        active = sorted({lower, upper})
        active_values = torch.stack(
            [torch.tanh(self._knot(camera, index)) for index in active]
        )
        magnitude = active_values.square().mean()
        curvature_centers = sorted(
            {index for index in active if 0 < index < self.num_knots - 1}
        )
        if curvature_centers:
            second_differences = torch.stack(
                [
                    torch.tanh(self._knot(camera, center + 1))
                    - 2.0 * torch.tanh(self._knot(camera, center))
                    + torch.tanh(self._knot(camera, center - 1))
                    for center in curvature_centers
                ]
            )
            curvature = second_differences.square().mean()
        else:
            curvature = magnitude * 0.0
        return (
            self.magnitude_regularization * magnitude
            + self.curvature_regularization * curvature
        )

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
                "Surface acquisition spline no longer matches Gaussian "
                "topology."
            )
        albedo = (
            features_albedo
            + self.rgb_delta(
                camera_idx=camera_idx,
                sequence_idx=sequence_idx,
            )
            / SH_DC_NORMALIZATION
        )
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
