import torch
from torch import nn


class LuminanceAffine(nn.Module):
    """Constrained RGB affine correction keyed by camera and optional frame."""

    def __init__(
        self,
        *,
        num_cameras: int,
        num_frames: int,
        lr: float = 1e-3,
        reg_lambda: float = 1e-2,
        use_frame_residual: bool = False,
        max_log_gain: float = 0.25,
        max_bias: float = 0.10,
        use_color_matrix: bool = False,
        max_matrix_delta: float = 0.10,
        color_matrix_reg_lambda: float = 0.25,
        use_radial_affine: bool = False,
        radial_band_count: int = 4,
        radial_max_log_gain: float = 0.08,
        radial_max_bias: float = 0.03,
        radial_reg_lambda: float = 0.50,
        use_residual_grid: bool = False,
        residual_grid_size: int = 32,
        residual_grid_max: float = 0.05,
        residual_grid_reg_lambda: float = 0.01,
        use_residual_grid_edge_gate: bool = False,
        residual_grid_gate_floor: float = 0.20,
        use_temporal_affine: bool = False,
        temporal_num_knots: int = 32,
        temporal_max_sequence_idx: int = 400,
        temporal_max_log_gain: float = 0.08,
        temporal_max_bias: float = 0.03,
        temporal_reg_lambda: float = 0.50,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.use_frame_residual = use_frame_residual
        self.max_log_gain = max_log_gain
        self.max_bias = max_bias
        self.use_color_matrix = use_color_matrix
        self.max_matrix_delta = max_matrix_delta
        self.color_matrix_reg_lambda = color_matrix_reg_lambda
        self.use_radial_affine = use_radial_affine
        self.radial_band_count = radial_band_count
        self.radial_max_log_gain = radial_max_log_gain
        self.radial_max_bias = radial_max_bias
        self.radial_reg_lambda = radial_reg_lambda
        self.use_residual_grid = use_residual_grid
        self.residual_grid_size = residual_grid_size
        self.residual_grid_max = residual_grid_max
        self.residual_grid_reg_lambda = residual_grid_reg_lambda
        self.use_residual_grid_edge_gate = use_residual_grid_edge_gate
        self.residual_grid_gate_floor = residual_grid_gate_floor
        self.use_temporal_affine = use_temporal_affine
        self.temporal_num_knots = max(2, int(temporal_num_knots))
        self.temporal_max_sequence_idx = max(1, int(temporal_max_sequence_idx))
        self.temporal_max_log_gain = temporal_max_log_gain
        self.temporal_max_bias = temporal_max_bias
        self.temporal_reg_lambda = temporal_reg_lambda

        self.camera_log_gain = nn.Parameter(torch.zeros(num_cameras, 3))
        self.camera_bias = nn.Parameter(torch.zeros(num_cameras, 3))
        color_matrix = torch.zeros(num_cameras, 3, 3)
        if use_color_matrix:
            self.color_matrix_raw = nn.Parameter(color_matrix)
        else:
            self.register_buffer("color_matrix_raw", color_matrix)
        radial_log_gain = torch.zeros(num_cameras, radial_band_count, 3)
        radial_bias = torch.zeros(num_cameras, radial_band_count, 3)
        if use_radial_affine:
            self.radial_log_gain_raw = nn.Parameter(radial_log_gain)
            self.radial_bias_raw = nn.Parameter(radial_bias)
        else:
            self.register_buffer("radial_log_gain_raw", radial_log_gain)
            self.register_buffer("radial_bias_raw", radial_bias)
        if use_residual_grid:
            self.residual_grid = nn.Parameter(
                torch.zeros(
                    num_cameras, 3, residual_grid_size, residual_grid_size
                )
            )
        else:
            self.register_buffer(
                "residual_grid",
                torch.zeros(
                    num_cameras, 3, residual_grid_size, residual_grid_size
                ),
            )
        if use_frame_residual:
            self.frame_log_gain = nn.Parameter(torch.zeros(num_frames, 3))
            self.frame_bias = nn.Parameter(torch.zeros(num_frames, 3))
        else:
            self.register_buffer("frame_log_gain", torch.zeros(num_frames, 3))
            self.register_buffer("frame_bias", torch.zeros(num_frames, 3))
        temporal_shape = (num_cameras, self.temporal_num_knots, 3)
        if use_temporal_affine:
            self.temporal_log_gain_raw = nn.Parameter(
                torch.zeros(temporal_shape)
            )
            self.temporal_bias_raw = nn.Parameter(torch.zeros(temporal_shape))
        else:
            self.register_buffer(
                "temporal_log_gain_raw",
                torch.zeros(temporal_shape),
            )
            self.register_buffer(
                "temporal_bias_raw", torch.zeros(temporal_shape)
            )

    def create_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create optimizers for the affine correction parameters."""
        return [torch.optim.Adam(self.parameters(), lr=self.lr)]

    def create_schedulers(
        self,
        optimizers: list[torch.optim.Optimizer],
        *,
        max_optimization_iters: int,
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Return schedulers matching the post-processing optimizer API."""
        return []

    def get_regularization_loss(self) -> torch.Tensor:
        """Penalize deviation from identity color correction."""
        loss = (
            self.camera_log_gain.square().mean()
            + self.camera_bias.square().mean()
        )
        if self.use_frame_residual:
            loss = loss + 0.25 * (
                self.frame_log_gain.square().mean()
                + self.frame_bias.square().mean()
            )
        if self.use_color_matrix:
            loss = (
                loss
                + self.color_matrix_reg_lambda
                * self.color_matrix_raw.square().mean()
            )
        if self.use_radial_affine:
            loss = loss + self.radial_reg_lambda * (
                self.radial_log_gain_raw.square().mean()
                + self.radial_bias_raw.square().mean()
            )
        if self.use_residual_grid:
            loss = (
                loss
                + self.residual_grid_reg_lambda
                * self.residual_grid.square().mean()
            )
        if self.use_temporal_affine:
            loss = loss + self.temporal_reg_lambda * (
                self.temporal_log_gain_raw.square().mean()
                + self.temporal_bias_raw.square().mean()
            )
        return self.reg_lambda * loss

    @staticmethod
    def _index(value: int | torch.Tensor, max_count: int) -> int:
        if torch.is_tensor(value):
            value = int(value.detach().flatten()[0].item())
        return max(0, min(int(value), max_count - 1))

    def forward(
        self,
        pred_rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        *,
        resolution: tuple[int, int],
        camera_idx: int | torch.Tensor,
        frame_idx: int | torch.Tensor,
        sequence_idx: int | torch.Tensor = -1,
        exposure_prior: torch.Tensor | None = None,
        residual_grid_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply bounded diagonal RGB affine correction to rendered RGB."""
        camera_index = self._index(camera_idx, self.camera_log_gain.shape[0])
        log_gain = self.camera_log_gain[camera_index]
        bias = self.camera_bias[camera_index]

        if self.use_frame_residual:
            frame_index = (
                int(frame_idx.detach().flatten()[0].item())
                if torch.is_tensor(frame_idx)
                else int(frame_idx)
            )
            if 0 <= frame_index < self.frame_log_gain.shape[0]:
                log_gain = log_gain + self.frame_log_gain[frame_index]
                bias = bias + self.frame_bias[frame_index]

        if self.use_temporal_affine:
            temporal_log_gain, temporal_bias = self.sample_temporal_affine(
                camera_index=camera_index,
                sequence_idx=sequence_idx,
            )
            log_gain = log_gain + temporal_log_gain
            bias = bias + temporal_bias

        if self.use_radial_affine:
            radial_log_gain, radial_bias = self.sample_radial_affine(
                pixel_coords=pixel_coords,
                resolution=resolution,
                camera_index=camera_index,
            )
            log_gain = log_gain + radial_log_gain
            bias = bias + radial_bias

        gain = torch.exp(log_gain.clamp(-self.max_log_gain, self.max_log_gain))
        bounded_bias = bias.clamp(-self.max_bias, self.max_bias)
        corrected = pred_rgb
        if self.use_color_matrix:
            identity = torch.eye(
                3,
                device=pred_rgb.device,
                dtype=pred_rgb.dtype,
            )
            matrix_delta = torch.tanh(self.color_matrix_raw[camera_index])
            matrix_delta = matrix_delta * self.max_matrix_delta
            matrix = identity + matrix_delta
            corrected = corrected @ matrix.transpose(0, 1)
        corrected = corrected * gain + bounded_bias
        if self.use_residual_grid:
            residual = self.sample_residual_grid(
                pixel_coords=pixel_coords,
                resolution=resolution,
                camera_index=camera_index,
            )
            if self.use_residual_grid_edge_gate:
                residual = residual * self._residual_gate(
                    residual,
                    residual_grid_gate,
                )
            corrected = corrected + residual
        return corrected.clamp(0.0, 1.0)

    def _residual_gate(
        self,
        residual: torch.Tensor,
        residual_grid_gate: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a bounded per-pixel multiplier for residual-grid output."""
        floor = float(self.residual_grid_gate_floor)
        if residual_grid_gate is None:
            return torch.full(
                (residual.shape[0], 1),
                floor,
                device=residual.device,
                dtype=residual.dtype,
            )
        gate = residual_grid_gate.to(
            device=residual.device, dtype=residual.dtype
        )
        gate = gate.reshape(-1, 1).clamp(0.0, 1.0)
        return floor + (1.0 - floor) * gate

    def sample_temporal_affine(
        self,
        *,
        camera_index: int,
        sequence_idx: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate bounded per-camera affine terms by scanner sequence."""
        if torch.is_tensor(sequence_idx):
            sequence_value = float(sequence_idx.detach().flatten()[0].item())
        else:
            sequence_value = float(sequence_idx)
        if sequence_value < 0.0:
            return (
                torch.zeros(
                    3,
                    device=self.camera_log_gain.device,
                    dtype=self.camera_log_gain.dtype,
                ),
                torch.zeros(
                    3,
                    device=self.camera_bias.device,
                    dtype=self.camera_bias.dtype,
                ),
            )

        sequence_value = min(
            sequence_value, float(self.temporal_max_sequence_idx)
        )
        scaled = (
            sequence_value
            / float(self.temporal_max_sequence_idx)
            * float(self.temporal_num_knots - 1)
        )
        lower = int(scaled)
        upper = min(lower + 1, self.temporal_num_knots - 1)
        weight = scaled - float(lower)

        log_gain_table = torch.tanh(self.temporal_log_gain_raw[camera_index])
        log_gain_table = log_gain_table * self.temporal_max_log_gain
        bias_table = torch.tanh(self.temporal_bias_raw[camera_index])
        bias_table = bias_table * self.temporal_max_bias

        log_gain = log_gain_table[lower] * (1.0 - weight)
        log_gain = log_gain + log_gain_table[upper] * weight
        bias = bias_table[lower] * (1.0 - weight)
        bias = bias + bias_table[upper] * weight
        return log_gain, bias

    def sample_radial_affine(
        self,
        *,
        pixel_coords: torch.Tensor,
        resolution: tuple[int, int],
        camera_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate bounded per-camera affine terms by image radius."""
        width, height = resolution
        center_x = (width - 1) * 0.5
        center_y = (height - 1) * 0.5
        radius_x = (pixel_coords[:, 0] - center_x) / max(center_x, 1.0)
        radius_y = (pixel_coords[:, 1] - center_y) / max(center_y, 1.0)
        radius = torch.sqrt(radius_x.square() + radius_y.square())
        radius = (radius / 2.0**0.5).clamp(0.0, 1.0)

        scaled = radius * (self.radial_band_count - 1)
        lower = torch.floor(scaled).to(torch.long)
        upper = torch.clamp(lower + 1, max=self.radial_band_count - 1)
        weight = (scaled - lower.to(scaled.dtype)).unsqueeze(-1)

        log_gain_table = torch.tanh(self.radial_log_gain_raw[camera_index])
        log_gain_table = log_gain_table * self.radial_max_log_gain
        bias_table = torch.tanh(self.radial_bias_raw[camera_index])
        bias_table = bias_table * self.radial_max_bias

        log_gain = log_gain_table[lower] * (1.0 - weight)
        log_gain = log_gain + log_gain_table[upper] * weight
        bias = bias_table[lower] * (1.0 - weight)
        bias = bias + bias_table[upper] * weight
        return log_gain, bias

    def sample_residual_grid(
        self,
        *,
        pixel_coords: torch.Tensor,
        resolution: tuple[int, int],
        camera_index: int,
    ) -> torch.Tensor:
        """Sample a bounded per-camera residual grid in image coordinates."""
        width, height = resolution
        grid = self.residual_grid[camera_index].unsqueeze(0)
        norm_x = 2.0 * pixel_coords[:, 0] / max(width - 1, 1) - 1.0
        norm_y = 2.0 * pixel_coords[:, 1] / max(height - 1, 1) - 1.0
        sample_grid = torch.stack((norm_x, norm_y), dim=-1).view(1, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        return sampled.clamp(-self.residual_grid_max, self.residual_grid_max)
