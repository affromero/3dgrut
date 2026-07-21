# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION
# & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import math

import numpy as np
import torch

from threedgrut.model.acquisition_appearance import _scalar_index


def _sha256_file(path: str) -> str:
    """Hash one immutable Gaussian-track artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_gaussian_track_ids(
    *,
    path: str,
    expected_sha256: str,
    num_gaussians: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Load an authenticated zero-abstaining Gaussian-track map."""
    if not path:
        raise ValueError("gaussian_track_acquisition.track_ids_path is empty.")
    if not expected_sha256:
        raise ValueError(
            "gaussian_track_acquisition.track_ids_sha256 is empty."
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Gaussian-track ID hash mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )
    track_ids = np.load(path, allow_pickle=False)
    if track_ids.shape != (num_gaussians,):
        raise ValueError(
            "Gaussian-track IDs do not match Gaussian topology: "
            f"{track_ids.shape} versus {(num_gaussians,)}."
        )
    if not np.issubdtype(track_ids.dtype, np.integer):
        raise ValueError("Gaussian-track IDs must have integer dtype.")
    if np.any(track_ids < 0):
        raise ValueError("Gaussian-track IDs must be non-negative.")
    return torch.from_numpy(track_ids.astype(np.int64, copy=False)).to(
        device=device
    )


class GaussianTrackAcquisition(torch.nn.Module):
    """Bounded acquisition residual shared by persistent Gaussian tracks."""

    def __init__(
        self,
        *,
        gaussian_track_ids: torch.Tensor,
        num_cameras: int,
        num_knots: int,
        min_sequence_idx: int,
        max_sequence_idx: int,
        max_rgb_delta: float,
        magnitude_regularization: float,
        curvature_regularization: float,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if gaussian_track_ids.ndim != 1 or gaussian_track_ids.numel() == 0:
            raise ValueError("gaussian_track_ids must be a non-empty vector.")
        if gaussian_track_ids.dtype != torch.int64:
            raise ValueError("gaussian_track_ids must have torch.int64 dtype.")
        if bool(torch.any(gaussian_track_ids < 0)):
            raise ValueError("gaussian_track_ids must be non-negative.")
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
        regularizers = (
            magnitude_regularization,
            curvature_regularization,
        )
        if any(
            not math.isfinite(value) or value < 0.0 for value in regularizers
        ):
            raise ValueError(
                "Gaussian-track regularization weights must be finite and "
                "non-negative."
            )

        track_ids = gaussian_track_ids.to(device=device)
        self.register_buffer(
            "gaussian_track_ids",
            track_ids,
            persistent=True,
        )
        self.num_tracks = int(track_ids.max().item()) + 1
        self.num_cameras = num_cameras
        self.num_knots = num_knots
        self.min_sequence_idx = min_sequence_idx
        self.max_sequence_idx = max_sequence_idx
        self.max_rgb_delta = max_rgb_delta
        self.magnitude_regularization = magnitude_regularization
        self.curvature_regularization = curvature_regularization
        self.rgb_knots = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(
                        (self.num_tracks, 3),
                        device=device,
                        dtype=dtype,
                    )
                )
                for _ in range(num_cameras * num_knots)
            ]
        )

    def _knot(self, camera_idx: int, knot_idx: int) -> torch.Tensor:
        return self.rgb_knots[camera_idx * self.num_knots + knot_idx]

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

    def rgb_delta(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """Return one shared bounded RGB residual per Gaussian."""
        camera, lower, upper, weight = self._interval(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )
        track_ids = self.gaussian_track_ids
        lower_value = torch.tanh(self._knot(camera, lower)[track_ids])
        upper_value = torch.tanh(self._knot(camera, upper)[track_ids])
        delta = lower_value * (1.0 - weight) + upper_value * weight
        supported = (track_ids > 0).to(dtype=delta.dtype)[:, None]
        return delta * supported * self.max_rgb_delta

    def get_local_regularization_loss(
        self,
        *,
        camera_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """Regularize only knots neighboring the current acquisition."""
        camera, lower, upper, _ = self._interval(
            camera_idx=camera_idx,
            sequence_idx=sequence_idx,
        )
        active = sorted({lower, upper})
        active_values = torch.stack(
            [torch.tanh(self._knot(camera, index)[1:]) for index in active]
        )
        magnitude = active_values.square().mean()
        curvature_centers = sorted(
            {index for index in active if 0 < index < self.num_knots - 1}
        )
        if curvature_centers:
            second_differences = torch.stack(
                [
                    torch.tanh(self._knot(camera, center + 1)[1:])
                    - 2.0 * torch.tanh(self._knot(camera, center)[1:])
                    + torch.tanh(self._knot(camera, center - 1)[1:])
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
