"""Held-out-predictive spatial appearance correction for PPISP.

The module deliberately predicts bounded fields from the rendered image and
normalized sensor coordinates for one physical-camera slot. An opt-in context
path also consumes renderer-predicted ray distance and opacity plus a
train-normalized camera pose. It has no frame-index embedding, so novel-view
inference uses the same information contract as PPISP's global controller.
"""

from __future__ import annotations

from typing import NamedTuple, override

import torch
import torch.nn.functional as F
from ppisp import PPISP, PPISPConfig
from torch import nn


class MultiscalePPISPConfig(NamedTuple):
    """Configuration for bounded multiscale PPISP fields."""

    coarse_grid_size: int = 4
    fine_grid_size: int = 16
    coarse_max_log_gain: float = 0.04
    coarse_max_bias: float = 0.02
    fine_max_log_gain: float = 0.02
    fine_max_bias: float = 0.01
    magnitude_regularization: float = 0.01
    total_variation_regularization: float = 0.01
    init_seed: int = 20_260_717
    use_view_context: bool = False


DEFAULT_MULTISCALE_PPISP_CONFIG = MultiscalePPISPConfig()
VIEW_CONTEXT_CHANNEL_ORDER = (
    "render_rgb_r",
    "render_rgb_g",
    "render_rgb_b",
    "sensor_x",
    "sensor_y",
    "ray_distance_over_train_scale_tanh",
    "opacity",
    "camera_center_x_over_train_scale_tanh",
    "camera_center_y_over_train_scale_tanh",
    "camera_center_z_over_train_scale_tanh",
    "camera_to_world_rotation_00",
    "camera_to_world_rotation_01",
    "camera_to_world_rotation_02",
    "camera_to_world_rotation_10",
    "camera_to_world_rotation_11",
    "camera_to_world_rotation_12",
    "camera_to_world_rotation_20",
    "camera_to_world_rotation_21",
    "camera_to_world_rotation_22",
)
VIEW_CONTEXT_FEATURE_CHANNELS = len(VIEW_CONTEXT_CHANNEL_ORDER) - 5


def view_context_inference_contract() -> dict[str, object]:
    """Return the exact novel-view signal and normalization contract."""
    return {
        "schema_version": 1,
        "channel_order": list(VIEW_CONTEXT_CHANNEL_ORDER),
        "ray_distance_semantics": "renderer_ray_distance_not_axial_depth",
        "ray_distance_normalization": "tanh(distance/train_length_scale)",
        "opacity_normalization": "finite_clamp_0_1",
        "camera_center_normalization": (
            "tanh((camera_center-train_center)/train_length_scale)"
        ),
        "camera_rotation_semantics": "camera_to_world_rotation_row_major",
        "normalization_statistics": (
            "serialized_training_fold_center_and_length_scale"
        ),
    }


class MultiscaleFieldController(nn.Module):
    """Predict coarse and fine RGB affine fields from image and coordinates."""

    def __init__(
        self,
        coarse_grid_size: int,
        fine_grid_size: int,
        input_channels: int = 5,
    ) -> None:
        """Initialize zero-output coarse and fine field heads."""
        super().__init__()
        if input_channels < 5:
            message = "Multiscale controller requires at least five channels."
            raise ValueError(message)
        self.coarse_grid_size = coarse_grid_size
        self.fine_grid_size = fine_grid_size
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=2,
                dilation=2,
            ),
            nn.SiLU(),
            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=4,
                dilation=4,
            ),
            nn.SiLU(),
        )
        self.coarse_head = nn.Conv2d(32, 6, kernel_size=1)
        self.fine_head = nn.Conv2d(32, 6, kernel_size=1)
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
        nn.init.zeros_(self.fine_head.weight)
        nn.init.zeros_(self.fine_head.bias)

    def forward(
        self,
        image: torch.Tensor,
        view_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict raw coarse and fine affine fields."""
        encoder_input = F.adaptive_avg_pool2d(image, (64, 64))
        batch_size, _, height, width = encoder_input.shape
        x_coords = torch.linspace(
            -1.0,
            1.0,
            width,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, 1, width)
        y_coords = torch.linspace(
            -1.0,
            1.0,
            height,
            device=image.device,
            dtype=image.dtype,
        ).view(1, 1, height, 1)
        coordinates = torch.cat(
            (
                x_coords.expand(batch_size, -1, height, -1),
                y_coords.expand(batch_size, -1, -1, width),
            ),
            dim=1,
        )
        feature_parts = [encoder_input, coordinates]
        if self.input_channels > 5:
            if view_context is None:
                message = (
                    "View-conditioned multiscale controller requires its "
                    "serialized inference context."
                )
                raise ValueError(message)
            if view_context.ndim != 4:
                message = "View context must have shape [B, C, H, W]."
                raise ValueError(message)
            if view_context.shape[0] != batch_size:
                message = "View context batch size must match rendered RGB."
                raise ValueError(message)
            expected_channels = self.input_channels - 5
            if view_context.shape[1] != expected_channels:
                message = (
                    "View context has the wrong channel count: expected "
                    f"{expected_channels}, got {view_context.shape[1]}."
                )
                raise ValueError(message)
            feature_parts.append(
                F.adaptive_avg_pool2d(view_context, (64, 64))
            )
        features = self.encoder(torch.cat(feature_parts, dim=1))
        coarse = self.coarse_head(
            F.adaptive_avg_pool2d(
                features,
                (self.coarse_grid_size, self.coarse_grid_size),
            )
        )
        fine = self.fine_head(
            F.adaptive_avg_pool2d(
                features,
                (self.fine_grid_size, self.fine_grid_size),
            )
        )
        return coarse, fine


class PredictiveMultiscalePPISP(PPISP):
    """PPISP plus bounded, content-predictive local affine fields."""

    def __init__(
        self,
        num_cameras: int,
        num_frames: int,
        config: PPISPConfig,
        multiscale_config: MultiscalePPISPConfig = (
            DEFAULT_MULTISCALE_PPISP_CONFIG
        ),
        view_center: torch.Tensor | None = None,
        view_scale: torch.Tensor | float | None = None,
    ) -> None:
        """Initialize PPISP and per-camera predictive field controllers."""
        coarse_grid_size = multiscale_config.coarse_grid_size
        fine_grid_size = multiscale_config.fine_grid_size
        if coarse_grid_size < 1 or fine_grid_size < coarse_grid_size:
            message = (
                "Multiscale PPISP grid sizes must satisfy "
                "1 <= coarse_grid_size <= fine_grid_size."
            )
            raise ValueError(message)
        bounds = (
            multiscale_config.coarse_max_log_gain,
            multiscale_config.coarse_max_bias,
            multiscale_config.fine_max_log_gain,
            multiscale_config.fine_max_bias,
        )
        if any(value <= 0.0 for value in bounds):
            message = "Multiscale PPISP field bounds must be positive."
            raise ValueError(message)
        if multiscale_config.magnitude_regularization < 0.0:
            message = "Magnitude regularization must be non-negative."
            raise ValueError(message)
        if multiscale_config.total_variation_regularization < 0.0:
            message = "Total-variation regularization must be non-negative."
            raise ValueError(message)
        super().__init__(
            num_cameras=num_cameras,
            num_frames=num_frames,
            config=config,
        )
        if not config.use_controller:
            message = (
                "Predictive multiscale PPISP requires use_controller=true."
            )
            raise ValueError(message)
        if multiscale_config.init_seed < 0:
            message = "Multiscale PPISP init seed must be non-negative."
            raise ValueError(message)
        if not config.controller_distillation:
            message = (
                "Predictive multiscale PPISP requires frozen-geometry "
                "controller distillation."
            )
            raise ValueError(message)
        self.multiscale_config = multiscale_config
        self.coarse_grid_size = coarse_grid_size
        self.fine_grid_size = fine_grid_size
        self.coarse_max_log_gain = multiscale_config.coarse_max_log_gain
        self.coarse_max_bias = multiscale_config.coarse_max_bias
        self.fine_max_log_gain = multiscale_config.fine_max_log_gain
        self.fine_max_bias = multiscale_config.fine_max_bias
        self.magnitude_regularization = (
            multiscale_config.magnitude_regularization
        )
        self.total_variation_regularization = (
            multiscale_config.total_variation_regularization
        )
        self.use_view_context = multiscale_config.use_view_context
        input_channels = (
            len(VIEW_CONTEXT_CHANNEL_ORDER)
            if self.use_view_context
            else 5
        )
        if self.use_view_context:
            if view_center is None or view_scale is None:
                message = (
                    "View-conditioned multiscale PPISP requires train-fold "
                    "camera center and length scale."
                )
                raise ValueError(message)
            normalized_center = torch.as_tensor(
                view_center,
                device=self.exposure_params.device,
                dtype=self.exposure_params.dtype,
            ).reshape(-1)
            normalized_scale = torch.as_tensor(
                view_scale,
                device=self.exposure_params.device,
                dtype=self.exposure_params.dtype,
            ).reshape(())
            if normalized_center.shape != (3,):
                message = "View-context train center must contain 3 values."
                raise ValueError(message)
            if not torch.isfinite(normalized_center).all():
                message = "View-context train center must be finite."
                raise ValueError(message)
            if not torch.isfinite(normalized_scale) or normalized_scale <= 0:
                message = (
                    "View-context train length scale must be finite and "
                    "positive."
                )
                raise ValueError(message)
            self.register_buffer(
                "view_context_center",
                normalized_center.clone(),
            )
            self.register_buffer(
                "view_context_scale",
                normalized_scale.clone(),
            )
        with torch.random.fork_rng():
            torch.manual_seed(multiscale_config.init_seed)
            self.multiscale_controllers = nn.ModuleList(
                MultiscaleFieldController(
                    coarse_grid_size=coarse_grid_size,
                    fine_grid_size=fine_grid_size,
                    input_channels=input_channels,
                )
                for _ in range(num_cameras)
            )
        self.multiscale_controllers.to(self.exposure_params.device)
        self._multiscale_regularization = torch.zeros(
            (),
            device=self.exposure_params.device,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: PPISPConfig,
        multiscale_config: MultiscalePPISPConfig = (
            DEFAULT_MULTISCALE_PPISP_CONFIG
        ),
    ) -> PredictiveMultiscalePPISP:
        """Restore the exact architecture and learned checkpoint state."""
        view_center = None
        view_scale = None
        if multiscale_config.use_view_context:
            center_key = "view_context_center"
            scale_key = "view_context_scale"
            if center_key not in state_dict or scale_key not in state_dict:
                message = (
                    "View-conditioned checkpoint is missing serialized "
                    "train-fold normalization statistics."
                )
                raise ValueError(message)
            view_center = state_dict[center_key]
            view_scale = state_dict[scale_key]
        instance = cls(
            num_cameras=int(state_dict["crf_params"].shape[0]),
            num_frames=int(state_dict["exposure_params"].shape[0]),
            config=config,
            multiscale_config=multiscale_config,
            view_center=view_center,
            view_scale=view_scale,
        )
        instance.load_state_dict(state_dict, strict=True)
        return instance

    def _controller_ready(self) -> bool:
        if self._ppisp_scheduler is None:
            return bool(
                self.config.use_controller
                and self.config.controller_activation_ratio < 1.0
                and len(self.controllers) > 0
            )
        return bool(
            self.config.use_controller
            and self.config.controller_activation_ratio < 1.0
            and self._controller_activation_step >= 0
            and self._ppisp_scheduler.last_epoch
            >= self._controller_activation_step
        )

    @staticmethod
    def _field_total_variation(field: torch.Tensor) -> torch.Tensor:
        horizontal = field[:, :, :, 1:] - field[:, :, :, :-1]
        vertical = field[:, :, 1:, :] - field[:, :, :-1, :]
        return horizontal.abs().mean() + vertical.abs().mean()

    @staticmethod
    def _bounded_field(
        raw_field: torch.Tensor,
        *,
        max_log_gain: float,
        max_bias: float,
    ) -> torch.Tensor:
        log_gain = torch.tanh(raw_field[:, :3]) * max_log_gain
        bias = torch.tanh(raw_field[:, 3:]) * max_bias
        return torch.cat((log_gain, bias), dim=1)

    @staticmethod
    def _single_channel_image(
        value: torch.Tensor,
        *,
        label: str,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Normalize renderer scalar outputs to `[1, 1, H, W]`."""
        if value.ndim == 3:
            value = value.unsqueeze(-1)
        if value.shape != (1, height, width, 1):
            message = (
                f"{label} must have shape [1, H, W] or [1, H, W, 1], "
                f"got {tuple(value.shape)}."
            )
            raise ValueError(message)
        return value.permute(0, 3, 1, 2)

    def _normalized_view_context(
        self,
        *,
        render_ray_distance: torch.Tensor | None,
        render_opacity: torch.Tensor | None,
        camera_to_world: torch.Tensor | None,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build context using only signals available for held-out views."""
        if (
            render_ray_distance is None
            or render_opacity is None
            or camera_to_world is None
        ):
            message = (
                "View-conditioned multiscale PPISP requires renderer ray "
                "distance, opacity, and camera-to-world pose."
            )
            raise ValueError(message)
        if camera_to_world.shape != (1, 4, 4):
            message = (
                "camera_to_world must have shape [1, 4, 4], got "
                f"{tuple(camera_to_world.shape)}."
            )
            raise ValueError(message)
        ray_distance = self._single_channel_image(
            render_ray_distance,
            label="render_ray_distance",
            height=height,
            width=width,
        ).to(device=device, dtype=dtype)
        opacity = self._single_channel_image(
            render_opacity,
            label="render_opacity",
            height=height,
            width=width,
        ).to(device=device, dtype=dtype)
        pose = camera_to_world.detach().to(device=device, dtype=dtype)
        scale = self.view_context_scale.to(device=device, dtype=dtype)
        center = self.view_context_center.to(device=device, dtype=dtype)
        normalized_distance = torch.tanh(
            torch.nan_to_num(ray_distance.detach(), nan=0.0).clamp_min(0.0)
            / scale
        )
        normalized_opacity = torch.nan_to_num(
            opacity.detach(),
            nan=0.0,
        ).clamp(0.0, 1.0)
        normalized_center = torch.tanh(
            (pose[0, :3, 3] - center) / scale
        )
        rotation = pose[0, :3, :3].reshape(-1)
        pose_features = torch.cat((normalized_center, rotation)).view(
            1,
            12,
            1,
            1,
        )
        pose_features = pose_features.expand(1, -1, height, width)
        return torch.cat(
            (
                normalized_distance,
                normalized_opacity,
                pose_features,
            ),
            dim=1,
        ).detach()

    @override
    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        resolution: tuple[int, int],
        camera_idx: torch.Tensor | int | None = None,
        frame_idx: torch.Tensor | int | None = None,
        exposure_prior: torch.Tensor | None = None,
        render_ray_distance: torch.Tensor | None = None,
        render_opacity: torch.Tensor | None = None,
        camera_to_world: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply global PPISP and, when ready, bounded local fields."""
        base = super().forward(
            rgb,
            pixel_coords,
            resolution,
            camera_idx,
            frame_idx,
            exposure_prior,
        )
        camera = -1 if camera_idx is None else int(camera_idx)
        if camera < 0 or not self._controller_ready():
            self._multiscale_regularization = torch.zeros(
                (),
                device=base.device,
                dtype=base.dtype,
            )
            return base

        width, height = resolution
        image = base.view(height, width, 3).permute(2, 0, 1).unsqueeze(0)
        view_context = None
        if self.use_view_context:
            view_context = self._normalized_view_context(
                render_ray_distance=render_ray_distance,
                render_opacity=render_opacity,
                camera_to_world=camera_to_world,
                height=height,
                width=width,
                device=base.device,
                dtype=base.dtype,
            )
        coarse_raw, fine_raw = self.multiscale_controllers[camera](
            image.detach(),
            view_context=view_context,
        )
        coarse = self._bounded_field(
            coarse_raw,
            max_log_gain=self.coarse_max_log_gain,
            max_bias=self.coarse_max_bias,
        )
        fine = self._bounded_field(
            fine_raw,
            max_log_gain=self.fine_max_log_gain,
            max_bias=self.fine_max_bias,
        )
        coarse_full = F.interpolate(
            coarse,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        fine_full = F.interpolate(
            fine,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        field = coarse_full + fine_full
        log_gain = field[:, :3].permute(0, 2, 3, 1).reshape(-1, 3)
        bias = field[:, 3:].permute(0, 2, 3, 1).reshape(-1, 3)
        magnitude = coarse.square().mean() + fine.square().mean()
        total_variation = self._field_total_variation(
            coarse
        ) + self._field_total_variation(fine)
        self._multiscale_regularization = (
            self.magnitude_regularization * magnitude
            + self.total_variation_regularization * total_variation
        )
        return base * torch.exp(log_gain) + bias

    @override
    def get_regularization_loss(self) -> torch.Tensor:
        """Return PPISP and predictive-field regularization."""
        return super().get_regularization_loss() + (
            self._multiscale_regularization
        )

    @override
    def create_optimizers(self) -> list[torch.optim.Optimizer]:
        """Attach field parameters to PPISP's controller optimizer."""
        optimizers = super().create_optimizers()
        if len(optimizers) != 2:
            message = (
                "Predictive multiscale PPISP requires a controller optimizer."
            )
            raise RuntimeError(message)
        optimizers[1].add_param_group(
            {"params": self.multiscale_controllers.parameters()}
        )
        return optimizers
