from dataclasses import replace

import torch
import torch.nn as nn

from threedgrut.datasets.protocols import Batch


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vector[..., 0])
    return torch.stack(
        (
            torch.stack((zeros, -vector[..., 2], vector[..., 1]), dim=-1),
            torch.stack((vector[..., 2], zeros, -vector[..., 0]), dim=-1),
            torch.stack((-vector[..., 1], vector[..., 0], zeros), dim=-1),
        ),
        dim=-2,
    )


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / angle
    skew = _skew(axis)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.expand(axis_angle.shape[0], 3, 3)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


class CameraResidual(nn.Module):
    """Bounded local camera residual applied to ray origins/directions."""

    def __init__(
        self,
        *,
        num_cameras: int,
        lr: float,
        reg_lambda: float,
        max_rotation_rad: float,
        max_translation_m: float,
        optimize_global: bool,
        optimize_per_camera: bool,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.max_rotation_rad = max_rotation_rad
        self.max_translation_m = max_translation_m
        self.optimize_global = optimize_global
        self.optimize_per_camera = optimize_per_camera
        self.global_rotation_raw = nn.Parameter(torch.zeros(1, 3))
        self.global_translation_raw = nn.Parameter(torch.zeros(1, 3))
        self.camera_rotation_raw = nn.Parameter(torch.zeros(num_cameras, 3))
        self.camera_translation_raw = nn.Parameter(torch.zeros(num_cameras, 3))
        self.global_rotation_raw.requires_grad_(optimize_global)
        self.global_translation_raw.requires_grad_(optimize_global)
        self.camera_rotation_raw.requires_grad_(optimize_per_camera)
        self.camera_translation_raw.requires_grad_(optimize_per_camera)

    def _bounded(self, camera_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = torch.zeros(
            (1, 3),
            device=self.global_rotation_raw.device,
            dtype=self.global_rotation_raw.dtype,
        )
        translation = torch.zeros_like(rotation)
        if self.optimize_global:
            rotation = rotation + self.global_rotation_raw
            translation = translation + self.global_translation_raw
        if self.optimize_per_camera:
            rotation = rotation + self.camera_rotation_raw[camera_idx : camera_idx + 1]
            translation = translation + self.camera_translation_raw[
                camera_idx : camera_idx + 1
            ]
        rotation = torch.tanh(rotation) * self.max_rotation_rad
        translation = torch.tanh(translation) * self.max_translation_m
        return rotation, translation

    def set_global_delta(
        self,
        *,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> None:
        """Set a fixed global bounded residual for finite-difference audits."""
        if rotation.shape != (3,):
            raise ValueError(f"Expected rotation shape (3,), got {rotation.shape}.")
        if translation.shape != (3,):
            raise ValueError(
                f"Expected translation shape (3,), got {translation.shape}."
            )
        eps = 1.0e-6
        rotation_raw = torch.atanh(
            (rotation / self.max_rotation_rad).clamp(-1.0 + eps, 1.0 - eps)
        )
        translation_raw = torch.atanh(
            (translation / self.max_translation_m).clamp(
                -1.0 + eps,
                1.0 - eps,
            )
        )
        with torch.no_grad():
            self.global_rotation_raw.zero_()
            self.global_translation_raw.zero_()
            self.global_rotation_raw[0].copy_(
                rotation_raw.to(
                    device=self.global_rotation_raw.device,
                    dtype=self.global_rotation_raw.dtype,
                )
            )
            self.global_translation_raw[0].copy_(
                translation_raw.to(
                    device=self.global_translation_raw.device,
                    dtype=self.global_translation_raw.dtype,
                )
            )
            self.camera_rotation_raw.zero_()
            self.camera_translation_raw.zero_()

    def set_camera_delta(
        self,
        *,
        camera_idx: int,
        rotation: torch.Tensor,
        translation: torch.Tensor,
    ) -> None:
        """Set one fixed per-camera bounded residual for audits."""
        if camera_idx < 0 or camera_idx >= self.camera_rotation_raw.shape[0]:
            raise ValueError(f"Invalid camera_idx for CameraResidual: {camera_idx}.")
        if rotation.shape != (3,):
            raise ValueError(f"Expected rotation shape (3,), got {rotation.shape}.")
        if translation.shape != (3,):
            raise ValueError(
                f"Expected translation shape (3,), got {translation.shape}."
            )
        eps = 1.0e-6
        rotation_raw = torch.atanh(
            (rotation / self.max_rotation_rad).clamp(-1.0 + eps, 1.0 - eps)
        )
        translation_raw = torch.atanh(
            (translation / self.max_translation_m).clamp(
                -1.0 + eps,
                1.0 - eps,
            )
        )
        with torch.no_grad():
            self.global_rotation_raw.zero_()
            self.global_translation_raw.zero_()
            self.camera_rotation_raw.zero_()
            self.camera_translation_raw.zero_()
            self.camera_rotation_raw[camera_idx].copy_(
                rotation_raw.to(
                    device=self.camera_rotation_raw.device,
                    dtype=self.camera_rotation_raw.dtype,
                )
            )
            self.camera_translation_raw[camera_idx].copy_(
                translation_raw.to(
                    device=self.camera_translation_raw.device,
                    dtype=self.camera_translation_raw.dtype,
                )
            )

    def forward(self, batch: Batch) -> Batch:
        rotation, translation = self._bounded(batch.camera_idx)
        rotation_matrix = _axis_angle_to_matrix(rotation)[0]
        rays_dir = torch.nn.functional.normalize(
            batch.rays_dir @ rotation_matrix.transpose(0, 1),
            dim=-1,
        )
        rays_ori = batch.rays_ori @ rotation_matrix.transpose(0, 1)
        rays_ori = rays_ori + translation.reshape(1, 1, 1, 3)
        return replace(batch, rays_ori=rays_ori, rays_dir=rays_dir)

    def get_regularization_loss(self) -> torch.Tensor:
        return self.reg_lambda * (
            self.global_rotation_raw.square().mean()
            + self.global_translation_raw.square().mean()
            + self.camera_rotation_raw.square().mean()
            + self.camera_translation_raw.square().mean()
        )

    def create_optimizer(self) -> torch.optim.Optimizer:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("CameraResidual has no trainable parameters enabled.")
        return torch.optim.Adam(
            parameters,
            lr=self.lr,
        )

    def max_abs_grad(self) -> float:
        """Return the maximum absolute gradient over trainable residual params."""
        grad_values = [
            parameter.grad.detach().abs().max()
            for parameter in self.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not grad_values:
            return 0.0
        return torch.stack(grad_values).max().item()

    def stats(self) -> dict[str, float]:
        camera_idx = 0
        rotation, translation = self._bounded(camera_idx)
        return {
            "rotation_norm_rad": rotation.norm().item(),
            "translation_norm_m": translation.norm().item(),
            "max_abs_grad": self.max_abs_grad(),
        }
