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
        use_residual_grid: bool = False,
        residual_grid_size: int = 32,
        residual_grid_max: float = 0.05,
        residual_grid_reg_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.use_frame_residual = use_frame_residual
        self.max_log_gain = max_log_gain
        self.max_bias = max_bias
        self.use_residual_grid = use_residual_grid
        self.residual_grid_size = residual_grid_size
        self.residual_grid_max = residual_grid_max
        self.residual_grid_reg_lambda = residual_grid_reg_lambda

        self.camera_log_gain = nn.Parameter(torch.zeros(num_cameras, 3))
        self.camera_bias = nn.Parameter(torch.zeros(num_cameras, 3))
        if use_residual_grid:
            self.residual_grid = nn.Parameter(
                torch.zeros(num_cameras, 3, residual_grid_size, residual_grid_size)
            )
        else:
            self.register_buffer(
                "residual_grid",
                torch.zeros(num_cameras, 3, residual_grid_size, residual_grid_size),
            )
        if use_frame_residual:
            self.frame_log_gain = nn.Parameter(torch.zeros(num_frames, 3))
            self.frame_bias = nn.Parameter(torch.zeros(num_frames, 3))
        else:
            self.register_buffer("frame_log_gain", torch.zeros(num_frames, 3))
            self.register_buffer("frame_bias", torch.zeros(num_frames, 3))

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
        if self.use_residual_grid:
            loss = loss + self.residual_grid_reg_lambda * self.residual_grid.square().mean()
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
        exposure_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply bounded diagonal RGB affine correction to rendered RGB."""
        camera_index = self._index(camera_idx, self.camera_log_gain.shape[0])
        log_gain = self.camera_log_gain[camera_index]
        bias = self.camera_bias[camera_index]

        if self.use_frame_residual:
            frame_index = int(frame_idx.detach().flatten()[0].item()) if torch.is_tensor(frame_idx) else int(frame_idx)
            if 0 <= frame_index < self.frame_log_gain.shape[0]:
                log_gain = log_gain + self.frame_log_gain[frame_index]
                bias = bias + self.frame_bias[frame_index]

        gain = torch.exp(log_gain.clamp(-self.max_log_gain, self.max_log_gain))
        bounded_bias = bias.clamp(-self.max_bias, self.max_bias)
        corrected = pred_rgb * gain + bounded_bias
        if self.use_residual_grid:
            corrected = corrected + self.sample_residual_grid(
                pixel_coords=pixel_coords,
                resolution=resolution,
                camera_index=camera_index,
            )
        return corrected.clamp(0.0, 1.0)

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
