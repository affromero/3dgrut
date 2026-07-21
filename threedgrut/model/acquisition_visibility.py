# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION
# & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

from threedgrut.model.acquisition_appearance import _scalar_index


SUPPORTED_VISIBILITY_RANK = 4
INITIAL_TEMPORAL_AMPLITUDE = 0.25


class AcquisitionVisibility(torch.nn.Module):
    """Low-rank opacity-logit response over scanner acquisition state."""

    def __init__(
        self,
        *,
        num_gaussians: int,
        num_cameras: int,
        num_knots: int,
        min_sequence_idx: int,
        max_sequence_idx: int,
        max_logit_delta: float,
        response_sparsity_regularization: float,
        magnitude_regularization: float,
        curvature_regularization: float,
        device: str | torch.device,
        dtype: torch.dtype,
        rank: int = SUPPORTED_VISIBILITY_RANK,
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
        if not math.isfinite(max_logit_delta) or max_logit_delta <= 0.0:
            raise ValueError("max_logit_delta must be finite and positive.")
        if rank != SUPPORTED_VISIBILITY_RANK:
            raise ValueError(
                "Acquisition visibility currently supports rank four only."
            )
        regularizers = (
            response_sparsity_regularization,
            magnitude_regularization,
            curvature_regularization,
        )
        if any(
            not math.isfinite(value) or value < 0.0 for value in regularizers
        ):
            raise ValueError(
                "Acquisition visibility regularization weights must be "
                "finite and non-negative."
            )

        self.num_cameras = num_cameras
        self.num_knots = num_knots
        self.min_sequence_idx = min_sequence_idx
        self.max_sequence_idx = max_sequence_idx
        self.max_logit_delta = max_logit_delta
        self.response_sparsity_regularization = (
            response_sparsity_regularization
        )
        self.magnitude_regularization = magnitude_regularization
        self.curvature_regularization = curvature_regularization
        self.rank = rank

        self.response_raw = torch.nn.Parameter(
            torch.zeros(
                (num_gaussians, rank),
                device=device,
                dtype=dtype,
            )
        )
        coordinate = torch.linspace(
            0.0,
            1.0,
            num_knots,
            device=device,
            dtype=dtype,
        )
        temporal_basis = torch.stack(
            (
                torch.ones_like(coordinate),
                2.0 * coordinate - 1.0,
                torch.sin(2.0 * math.pi * coordinate),
                torch.cos(2.0 * math.pi * coordinate),
            )
        )
        temporal_basis = temporal_basis * INITIAL_TEMPORAL_AMPLITUDE
        initial_temporal_raw = torch.atanh(temporal_basis)
        self.temporal_raw = torch.nn.Parameter(
            initial_temporal_raw[None].repeat(num_cameras, 1, 1)
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

    def logit_delta(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        coefficients = self.sample_coefficients(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )
        response = torch.tanh(self.response_raw)
        delta = torch.einsum("nr,r->n", response, coefficients)
        return delta[:, None] * (self.max_logit_delta / float(self.rank))

    def get_regularization_loss(self) -> torch.Tensor:
        response = torch.tanh(self.response_raw)
        temporal = torch.tanh(self.temporal_raw)
        response_sparsity = response.abs().mean()
        temporal_magnitude = temporal.square().mean()
        second_difference = (
            temporal[..., 2:] - 2.0 * temporal[..., 1:-1] + temporal[..., :-2]
        )
        temporal_curvature = second_difference.square().mean()
        return (
            self.response_sparsity_regularization * response_sparsity
            + self.magnitude_regularization * temporal_magnitude
            + self.curvature_regularization * temporal_curvature
        )
