"""Trainable local image-projection fields indexed by source frame."""

from __future__ import annotations

import torch

from threedgrut.model.native_projection_distortion import (
    sample_native_distortion,
)


class LocalProjectionField(torch.nn.Module):
    """One bilinear two-axis projection field for each source frame."""

    checkpoint_format_version = 1
    checkpoint_algorithm = "local_projection_field"

    def __init__(
        self,
        *,
        num_source_frames: int,
        grid_height: int = 12,
        grid_width: int = 12,
    ) -> None:
        """Initialize an exact-zero field for every stable source frame."""
        super().__init__()
        if num_source_frames <= 0:
            raise ValueError("num_source_frames must be positive.")
        if grid_height <= 1 or grid_width <= 1:
            raise ValueError("Projection-field grid dimensions must exceed one.")
        self.values = torch.nn.Parameter(
            torch.zeros((num_source_frames, grid_height, grid_width, 2))
        )
        self.register_buffer(
            "source_frame_count",
            torch.tensor(num_source_frames, dtype=torch.int64),
        )
        self.register_buffer(
            "grid_dimensions",
            torch.tensor((grid_height, grid_width), dtype=torch.int64),
        )

    @property
    def num_source_frames(self) -> int:
        """Return the stable source-frame table length."""
        return int(self.source_frame_count.item())

    def sample(
        self,
        *,
        source_frame_idx: int,
        means2d: torch.Tensor,
        resolution: tuple[int, int],
    ) -> torch.Tensor:
        """Sample the selected field at projected pixel centers."""
        if not 0 <= source_frame_idx < self.num_source_frames:
            raise ValueError(
                f"Invalid local projection source frame: {source_frame_idx}."
            )
        if means2d.ndim != 2 or means2d.shape[1] != 2:
            raise ValueError("means2d must have shape (N, 2).")
        width, height = resolution
        if width <= 0 or height <= 0:
            raise ValueError("Projection-field resolution must be positive.")
        field = self.values[source_frame_idx].to(dtype=means2d.dtype)
        sampled = sample_native_distortion(
            field,
            means2d.reshape(1, -1, 1, 2),
            resolution=(width, height),
        )
        return sampled.reshape(-1, 2)

    def validate_state(self) -> None:
        """Reject checkpoint state with an incompatible field layout."""
        expected_shape = (
            self.num_source_frames,
            int(self.grid_dimensions[0].item()),
            int(self.grid_dimensions[1].item()),
            2,
        )
        if self.values.shape != expected_shape:
            raise ValueError(
                "Local projection field has incompatible checkpoint shape."
            )
        if self.values.dtype is not torch.float32:
            raise ValueError("Local projection field values must use float32.")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("Local projection field values must be finite.")
