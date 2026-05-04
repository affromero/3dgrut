# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

import numpy as np
import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import check_step_condition, quaternion_to_so3


class GSStrategy(BaseStrategy):
    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)

        # Parameters related to densification, pruning and reset
        self.split_n_gaussians = self.conf.strategy.densify.split.n_gaussians
        self.relative_size_threshold = self.conf.strategy.densify.relative_size_threshold
        self.prune_density_threshold = self.conf.strategy.prune.density_threshold
        self.clone_grad_threshold = self.conf.strategy.densify.clone_grad_threshold
        self.split_grad_threshold = self.conf.strategy.densify.split_grad_threshold
        self.new_max_density = self.conf.strategy.reset_density.new_max_density
        self.footprint_control = self.conf.strategy.get("footprint_control", {})
        self.footprint_scale_control = self.conf.strategy.get("footprint_scale_control", {})
        self.residual_density_control = self.conf.strategy.get("residual_density_control", {})

        # Accumulation of the norms of the positions gradients
        self.densify_grad_norm_accum = torch.empty([0, 1])
        self.densify_grad_norm_denom = torch.empty([0, 1])
        self.footprint_stats = {}
        self.footprint_scale_stats = {}
        self.residual_density_stats = {}
        self.edge_error_density_stats = {}
        self.frequency_error_density_stats = {}
        self.abs_gradient_density_stats = {}
        self.camera_balance_density_stats = {}
        self.structure_density_stats = {}
        self.roma_precision_density_stats: dict = {}
        self.camera_balance_residual_ema = torch.empty((0,), device=self.model.device)
        self.densify_abs_grad_norm_accum = torch.empty([0, 1])
        self.densify_signed_grad_accum = torch.empty([0, 3])
        self.structure_axis_x_accum = torch.empty([0, 1])
        self.structure_axis_y_accum = torch.empty([0, 1])
        self.structure_local_axis_accum = torch.empty([0, 3])
        self.structure_axis_denom = torch.empty([0, 1])
        # Per-image RoMa precision tensors keyed by image stem (e.g. "left_0009").
        # Lazily filled on first read; small (~3 * frame_count entries).
        self._roma_precision_cache: dict = {}

    def get_strategy_parameters(self) -> dict:
        params = {}

        params["densify_grad_norm_accum"] = (self.densify_grad_norm_accum,)
        params["densify_grad_norm_denom"] = (self.densify_grad_norm_denom,)
        params["densify_abs_grad_norm_accum"] = (
            self.densify_abs_grad_norm_accum,
        )
        params["densify_signed_grad_accum"] = (
            self.densify_signed_grad_accum,
        )
        params["structure_axis_x_accum"] = (self.structure_axis_x_accum,)
        params["structure_axis_y_accum"] = (self.structure_axis_y_accum,)
        params["structure_local_axis_accum"] = (self.structure_local_axis_accum,)
        params["structure_axis_denom"] = (self.structure_axis_denom,)

        return params

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        num_gaussians = self.model.num_gaussians
        if checkpoint is not None:
            self.densify_grad_norm_accum = checkpoint["densify_grad_norm_accum"][0].detach()
            self.densify_grad_norm_denom = checkpoint["densify_grad_norm_denom"][0].detach()
            self.densify_abs_grad_norm_accum = self._checkpoint_or_zero_buffer(
                checkpoint,
                "densify_abs_grad_norm_accum",
                num_gaussians,
            )
            self.densify_signed_grad_accum = self._checkpoint_or_zero_buffer(
                checkpoint,
                "densify_signed_grad_accum",
                num_gaussians,
                width=3,
            )
            self.structure_axis_x_accum = self._checkpoint_or_zero_buffer(
                checkpoint,
                "structure_axis_x_accum",
                num_gaussians,
            )
            self.structure_axis_y_accum = self._checkpoint_or_zero_buffer(
                checkpoint,
                "structure_axis_y_accum",
                num_gaussians,
            )
            self.structure_local_axis_accum = self._checkpoint_or_zero_buffer(
                checkpoint,
                "structure_local_axis_accum",
                num_gaussians,
                width=3,
            )
            self.structure_axis_denom = self._checkpoint_or_zero_buffer(
                checkpoint,
                "structure_axis_denom",
                num_gaussians,
                dtype=torch.int,
            )
        else:
            self.densify_grad_norm_accum = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.densify_grad_norm_denom = torch.zeros((num_gaussians, 1), dtype=torch.int, device=self.model.device)
            self.densify_abs_grad_norm_accum = torch.zeros(
                (num_gaussians, 1),
                dtype=torch.float,
                device=self.model.device,
            )
            self.densify_signed_grad_accum = torch.zeros(
                (num_gaussians, 3),
                dtype=torch.float,
                device=self.model.device,
            )
            self.structure_axis_x_accum = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.structure_axis_y_accum = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.structure_local_axis_accum = torch.zeros(
                (num_gaussians, 3),
                dtype=torch.float,
                device=self.model.device,
            )
            self.structure_axis_denom = torch.zeros((num_gaussians, 1), dtype=torch.int, device=self.model.device)

    def _checkpoint_or_zero_buffer(
        self,
        checkpoint: dict,
        key: str,
        num_gaussians: int,
        dtype: torch.dtype = torch.float,
        width: int = 1,
    ) -> torch.Tensor:
        checkpoint_value = checkpoint.get(key)
        if checkpoint_value is not None:
            return checkpoint_value[0].detach()
        return torch.zeros((num_gaussians, width), dtype=dtype, device=self.model.device)

    def _post_backward(
        self,
        step: int,
        scene_extent: float,
        train_dataset,
        batch=None,
        writer=None,
        outputs=None,
    ) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""

        # Update densification buffer:
        if check_step_condition(step, 0, self.conf.strategy.densify.end_iteration, 1):
            with torch.cuda.nvtx.range(f"train_{step}_grad_buffer"):
                self.update_gradient_buffer(
                    sensor_position=batch.T_to_world[0, :3, 3],
                    batch=batch,
                    outputs=outputs,
                )

        # Clamp density
        if check_step_condition(step, 0, -1, 1) and self.conf.model.density_activation == "none":
            with torch.cuda.nvtx.range(f"train_{step}_clamp_density"):
                self.model.clamp_density()

        return False

    def _post_optimizer_step(self, step: int, scene_extent: float, train_dataset, batch=None, writer=None) -> bool:
        """Callback function to be executed after the optimizer step."""
        scene_updated = False
        # Densify the Gaussians

        if check_step_condition(
            step,
            self.conf.strategy.densify.start_iteration,
            self.conf.strategy.densify.end_iteration,
            self.conf.strategy.densify.frequency,
        ):
            self.densify_gaussians(scene_extent=scene_extent)
            scene_updated = True

        if self.should_apply_footprint_scale_control(step):
            self.apply_footprint_scale_control(batch)

        # Prune the Gaussians based on their opacity
        if check_step_condition(
            step,
            self.conf.strategy.prune.start_iteration,
            self.conf.strategy.prune.end_iteration,
            self.conf.strategy.prune.frequency,
        ):
            self.prune_gaussians_opacity()
            scene_updated = True

        # Prune the Gaussians based on their scales
        if check_step_condition(
            step,
            self.conf.strategy.prune_scale.start_iteration,
            self.conf.strategy.prune_scale.end_iteration,
            self.conf.strategy.prune_scale.frequency,
        ):
            self.prune_gaussians_scale(train_dataset)
            scene_updated = True

        # Decay the density values
        if check_step_condition(
            step,
            self.conf.strategy.density_decay.start_iteration,
            self.conf.strategy.density_decay.end_iteration,
            self.conf.strategy.density_decay.frequency,
        ):
            self.decay_density()

        # Reset the Gaussian density
        if check_step_condition(
            step,
            self.conf.strategy.reset_density.start_iteration,
            self.conf.strategy.reset_density.end_iteration,
            self.conf.strategy.reset_density.frequency,
        ):
            self.reset_density()

        self.log_footprint_control(step, writer)
        self.log_footprint_scale_control(step, writer)
        self.log_residual_density_control(step, writer)
        self.log_structure_density_control(step, writer)

        return scene_updated

    @torch.no_grad()
    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(self, sensor_position: torch.Tensor, batch=None, outputs=None) -> None:
        params_grad = self.model.positions.grad
        assert params_grad is not None
        mask = (params_grad != 0).max(dim=1)[0]
        if self.residual_density_control.get("enabled", False) and outputs is not None:
            visibility = outputs.get("mog_visibility")
            if visibility is not None and self.residual_density_control.get("use_visibility", True):
                mask = torch.logical_and(mask, visibility.reshape(-1) > 0)

        if not mask.any():
            return

        distance_to_camera = (self.model.positions[mask] - sensor_position).norm(dim=1, keepdim=True)
        scaled_grad = params_grad[mask] * distance_to_camera / 2
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)
        density_weight = torch.ones_like(grad_norm)

        if self.footprint_control.get("enabled", False) and batch is not None:
            density_weight = density_weight * self.compute_footprint_weights(
                mask,
                batch,
                outputs,
            )

        if self.residual_density_control.get("enabled", False) and batch is not None and outputs is not None:
            density_weight = density_weight * self.compute_residual_density_weights(
                mask,
                batch,
                outputs,
            )

        weighted_grad = scaled_grad * density_weight

        self.densify_grad_norm_accum[mask] += grad_norm * density_weight
        self.densify_abs_grad_norm_accum[mask] += weighted_grad.abs().sum(
            dim=-1,
            keepdim=True,
        )
        self.densify_signed_grad_accum[mask] += weighted_grad
        self.densify_grad_norm_denom[mask] += 1

    def camera_focal_mean(self, batch) -> float:
        camera_params = (
            batch.intrinsics_OpenCVPinholeCameraModelParameters
            or batch.intrinsics_OpenCVFisheyeCameraModelParameters
            or batch.intrinsics_RationalCameraModelParameters
        )
        if camera_params is not None:
            focal_length = camera_params.get("focal_length")
            if focal_length is not None:
                return float(torch.as_tensor(focal_length).float().mean().item())
        if batch.intrinsics is not None:
            return float(torch.as_tensor(batch.intrinsics[:2]).float().mean().item())
        return 1.0

    @torch.no_grad()
    def compute_footprint_weights(self, mask: torch.Tensor, batch, outputs=None) -> torch.Tensor:
        selected_positions = self.model.positions[mask]
        if selected_positions.numel() == 0:
            return torch.ones((0, 1), device=self.model.device)

        radius_px = self.compute_projected_radius(mask, batch, outputs)
        in_front = radius_px > 0.0

        min_radius = float(self.footprint_control.get("min_radius_px", 0.75))
        max_radius = float(self.footprint_control.get("max_radius_px", 8.0))
        subpixel_boost = float(self.footprint_control.get("subpixel_grad_boost", 3.0))
        oversized_boost = float(self.footprint_control.get("oversized_grad_boost", 3.0))

        weights = torch.ones_like(radius_px)
        subpixel = torch.logical_and(in_front, radius_px < min_radius)
        oversized = torch.logical_and(in_front, radius_px > max_radius)
        weights[subpixel] = torch.clamp(min_radius / radius_px[subpixel].clamp_min(1e-6), max=subpixel_boost)
        weights[oversized] = torch.clamp(radius_px[oversized] / max_radius, max=oversized_boost)
        weights[~in_front] = 1.0

        self.footprint_stats = {
            "front_fraction": in_front.float().mean().item(),
            "radius_px_mean": radius_px[in_front].mean().item() if in_front.any() else 0.0,
            "subpixel_fraction": subpixel.float().mean().item(),
            "oversized_fraction": oversized.float().mean().item(),
            "weight_mean": weights.mean().item(),
            "weight_max": weights.max().item(),
        }
        return weights.unsqueeze(1)

    def log_footprint_control(self, step: int, writer) -> None:
        if not self.footprint_control.get("enabled", False):
            return
        if writer is None or not hasattr(writer, "add_scalar"):
            return
        log_frequency = int(self.footprint_control.get("log_frequency", 100))
        if not check_step_condition(step, 0, -1, log_frequency):
            return
        for name, value in self.footprint_stats.items():
            writer.add_scalar(f"diagnostics/footprint_control/{name}", value, step)

    def should_apply_footprint_scale_control(self, step: int) -> bool:
        if not self.footprint_scale_control.get("enabled", False):
            return False
        return check_step_condition(
            step,
            int(self.footprint_scale_control.get("start_iteration", 100)),
            int(self.footprint_scale_control.get("end_iteration", 3000)),
            int(self.footprint_scale_control.get("frequency", 100)),
        )

    @torch.no_grad()
    def apply_footprint_scale_control(self, batch) -> None:
        if batch is None:
            return
        scale_factor = self.compute_footprint_scale_factor(batch)
        if scale_factor.numel() == 0:
            return

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name == "scale", "wrong parameter passed to update_param_fn"
            scale = self.model.scale_activation(param)
            adjusted_scale = scale * scale_factor[:, None]
            adjusted_param = self.model.scale_activation_inv(adjusted_scale)
            return torch.nn.Parameter(adjusted_param, requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, value: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(value)

        self._update_param_with_optimizer(
            update_param_fn,
            update_optimizer_fn,
            names=["scale"],
        )

    @torch.no_grad()
    def compute_footprint_scale_factor(self, batch) -> torch.Tensor:
        positions = self.model.positions
        if positions.numel() == 0:
            return torch.ones((0,), device=self.model.device)

        pose = batch.T_to_world[0]
        world_to_camera = torch.linalg.inv(pose)
        homogeneous = torch.cat(
            (positions, torch.ones_like(positions[:, :1])),
            dim=1,
        )
        camera_positions = homogeneous @ world_to_camera.transpose(0, 1)
        raw_depth = camera_positions[:, 2]
        depth = raw_depth.clamp_min(1e-3)
        in_front = raw_depth > 1e-3

        scales = self.model.get_scale().float().amax(dim=1)
        focal_mean = self.camera_focal_mean(batch)
        radius_px = scales * focal_mean / depth

        min_radius = float(self.footprint_scale_control.get("min_radius_px", 0.5))
        max_radius = float(self.footprint_scale_control.get("max_radius_px", 12.0))
        grow_subpixel = bool(self.footprint_scale_control.get("grow_subpixel", True))
        max_update = float(self.footprint_scale_control.get("max_update_factor", 1.1))
        min_update = 1.0 / max_update

        scale_factor = torch.ones_like(radius_px)
        subpixel = torch.logical_and(in_front, radius_px < min_radius)
        oversized = torch.logical_and(in_front, radius_px > max_radius)
        if grow_subpixel:
            scale_factor[subpixel] = torch.clamp(
                min_radius / radius_px[subpixel].clamp_min(1e-6),
                max=max_update,
            )
        scale_factor[oversized] = torch.clamp(
            max_radius / radius_px[oversized],
            min=min_update,
        )
        scale_factor[~in_front] = 1.0

        adjusted = scale_factor != 1.0
        self.footprint_scale_stats = {
            "front_fraction": in_front.float().mean().item(),
            "radius_px_mean": radius_px[in_front].mean().item() if in_front.any() else 0.0,
            "subpixel_fraction": subpixel.float().mean().item(),
            "oversized_fraction": oversized.float().mean().item(),
            "adjusted_fraction": adjusted.float().mean().item(),
            "scale_factor_mean": scale_factor.mean().item(),
            "scale_factor_min": scale_factor.min().item(),
            "scale_factor_max": scale_factor.max().item(),
        }
        return scale_factor

    def log_footprint_scale_control(self, step: int, writer) -> None:
        if not self.footprint_scale_control.get("enabled", False):
            return
        if writer is None or not hasattr(writer, "add_scalar"):
            return
        log_frequency = int(self.footprint_scale_control.get("log_frequency", 100))
        if not check_step_condition(step, 0, -1, log_frequency):
            return
        for name, value in self.footprint_scale_stats.items():
            writer.add_scalar(f"diagnostics/footprint_scale_control/{name}", value, step)

    @torch.no_grad()
    def compute_residual_density_weights(self, mask: torch.Tensor, batch, outputs) -> torch.Tensor:
        selected_positions = self.model.positions[mask]
        if selected_positions.numel() == 0:
            return torch.ones((0, 1), device=self.model.device)

        area_px = self.compute_projected_area(mask, batch, outputs)
        area_reference = float(self.residual_density_control.get("area_reference_px", 4.0))
        area_power = float(self.residual_density_control.get("area_power", 0.5))
        area_weight = torch.pow(area_px / max(area_reference, 1e-6), area_power)
        area_weight = torch.clamp(
            area_weight,
            min=float(self.residual_density_control.get("area_min_weight", 0.25)),
            max=float(self.residual_density_control.get("area_max_weight", 4.0)),
        )

        residual_scalar = self.compute_frame_residual(batch, outputs)
        camera_balance_weight = self.compute_camera_balance_weight(
            batch,
            residual_scalar,
        )
        residual_reference = float(self.residual_density_control.get("residual_reference", 0.05))
        residual_power = float(self.residual_density_control.get("residual_power", 0.5))
        residual_weight = (residual_scalar / max(residual_reference, 1e-6)) ** residual_power
        residual_weight = min(
            max(residual_weight, float(self.residual_density_control.get("residual_min_weight", 0.5))),
            float(self.residual_density_control.get("residual_max_weight", 2.0)),
        )

        gradient_weight = self.compute_responsibility_gradient_weight(mask)
        edge_error_weight = self.compute_edge_error_weight(mask, batch, outputs)
        frequency_error_weight = self.compute_frequency_error_weight(
            mask,
            batch,
            outputs,
        )
        roma_precision_weight = self.compute_roma_precision_weight(mask, batch, outputs)
        structure_weight = self.compute_structure_density_weight(mask, batch, outputs)
        total_weight = (
            area_weight
            * residual_weight
            * gradient_weight
            * edge_error_weight
            * frequency_error_weight
            * roma_precision_weight
            * structure_weight
            * camera_balance_weight
        )
        total_weight = torch.clamp(
            total_weight,
            min=float(self.residual_density_control.get("min_total_weight", 0.25)),
            max=float(self.residual_density_control.get("max_total_weight", 6.0)),
        )

        self.residual_density_stats = {
            "selected_fraction": mask.float().mean().item(),
            "frame_residual_l1": float(residual_scalar),
            "area_px_mean": area_px.mean().item(),
            "area_px_p95": torch.quantile(area_px, 0.95).item(),
            "area_weight_mean": area_weight.mean().item(),
            "area_weight_max": area_weight.max().item(),
            "exact_projected_extent": float(outputs is not None and "mog_projected_extent" in outputs),
            "gradient_weight_mean": gradient_weight.mean().item(),
            "gradient_weight_max": gradient_weight.max().item(),
            "edge_error_weight_mean": edge_error_weight.mean().item(),
            "edge_error_weight_max": edge_error_weight.max().item(),
            "frequency_error_weight_mean": frequency_error_weight.mean().item(),
            "frequency_error_weight_max": frequency_error_weight.max().item(),
            "roma_precision_weight_mean": roma_precision_weight.mean().item(),
            "roma_precision_weight_max": roma_precision_weight.max().item(),
            "structure_weight_mean": structure_weight.mean().item(),
            "structure_weight_max": structure_weight.max().item(),
            "camera_balance_weight": float(camera_balance_weight),
            "total_weight_mean": total_weight.mean().item(),
            "total_weight_max": total_weight.max().item(),
        }
        self.residual_density_stats.update(self.edge_error_density_stats)
        self.residual_density_stats.update(self.frequency_error_density_stats)
        self.residual_density_stats.update(self.roma_precision_density_stats)
        self.residual_density_stats.update(self.camera_balance_density_stats)
        self.residual_density_stats.update(self.structure_density_stats)
        if "mog_tiles_count" in outputs:
            tiles_count = outputs["mog_tiles_count"][mask].float()
            self.residual_density_stats.update(
                {
                    "tiles_count_mean": tiles_count.mean().item(),
                    "tiles_count_p95": torch.quantile(tiles_count, 0.95).item(),
                }
            )
        return total_weight.unsqueeze(1)

    @torch.no_grad()
    def compute_structure_density_weight(self, mask: torch.Tensor, batch, outputs) -> torch.Tensor:
        selected_count = int(mask.sum().item())
        self.structure_density_stats = {}
        weights = torch.ones((selected_count,), device=self.model.device)
        if selected_count == 0:
            return weights
        if not self.residual_density_control.get("use_structure_density_weight", False):
            return weights
        if "mog_projected_position" not in outputs or "mog_projected_extent" not in outputs:
            return weights

        maps = self.compute_structure_maps(batch.rgb_gt, outputs["pred_rgb"])
        positions = outputs["mog_projected_position"][mask].float()
        x = torch.round(positions[:, 0]).long()
        y = torch.round(positions[:, 1]).long()
        height, width = maps["edge"].shape
        valid = torch.logical_and(
            torch.logical_and(x >= 0, x < width),
            torch.logical_and(y >= 0, y < height),
        )
        if "mog_visibility" in outputs:
            visible = outputs["mog_visibility"][mask].reshape(-1) > 0
            valid = torch.logical_and(valid, visible)
        if batch.mask is not None:
            frame_mask = batch.mask
            if frame_mask.dim() == 4:
                frame_mask = frame_mask[0]
            frame_mask = frame_mask.squeeze(-1) > 0.5
            valid_mask_sample = torch.zeros_like(valid)
            valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if valid_indices.numel() > 0:
                valid_mask_sample[valid_indices] = frame_mask[
                    y[valid_indices],
                    x[valid_indices],
                ]
            valid = torch.logical_and(valid, valid_mask_sample)

        edge = torch.zeros_like(weights)
        anisotropy = torch.zeros_like(weights)
        scale_px = torch.ones_like(weights)
        residual = torch.zeros_like(weights)
        grad_x = torch.zeros_like(weights)
        grad_y = torch.zeros_like(weights)
        signed_grad_x = torch.zeros_like(weights)
        signed_grad_y = torch.zeros_like(weights)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() > 0:
            sample_y = y[valid_indices]
            sample_x = x[valid_indices]
            edge[valid_indices] = maps["edge"][sample_y, sample_x]
            anisotropy[valid_indices] = maps["anisotropy"][sample_y, sample_x]
            scale_px[valid_indices] = maps["scale_px"][sample_y, sample_x]
            residual[valid_indices] = maps["residual"][sample_y, sample_x]
            grad_x[valid_indices] = maps["grad_x"][sample_y, sample_x]
            grad_y[valid_indices] = maps["grad_y"][sample_y, sample_x]
            signed_grad_x[valid_indices] = maps["grad_x_signed"][sample_y, sample_x]
            signed_grad_y[valid_indices] = maps["grad_y_signed"][sample_y, sample_x]

        extent = outputs["mog_projected_extent"][mask].float().abs()
        extent_x = extent[:, 0].clamp_min(1.0e-6)
        extent_y = extent[:, 1].clamp_min(1.0e-6)
        normalized_edge = self.normalize_signal(edge)
        normalized_residual = self.normalize_signal(residual)
        reliable_structure = normalized_edge * (1.0 + anisotropy)
        violation_x = extent_x * self.normalize_signal(grad_x) / scale_px.clamp_min(1.0)
        violation_y = extent_y * self.normalize_signal(grad_y) / scale_px.clamp_min(1.0)
        violation = torch.maximum(violation_x, violation_y)
        score = violation * reliable_structure * (1.0 + normalized_residual)
        score = torch.where(valid, score, torch.zeros_like(score))

        score_reference = float(
            self.residual_density_control.get("structure_score_reference", 1.0)
        )
        score_power = float(self.residual_density_control.get("structure_score_power", 0.5))
        weights = torch.pow(score / max(score_reference, 1e-6), score_power)
        weights = torch.clamp(
            weights,
            min=float(self.residual_density_control.get("structure_min_weight", 0.5)),
            max=float(self.residual_density_control.get("structure_max_weight", 4.0)),
        )

        axis_x = torch.logical_and(valid, violation_x >= violation_y)
        axis_y = torch.logical_and(valid, violation_y > violation_x)
        axis_score = torch.clamp(score, max=float(self.residual_density_control.get("structure_axis_max_score", 10.0)))
        self.structure_axis_x_accum[mask] += (axis_x.float() * axis_score).unsqueeze(1)
        self.structure_axis_y_accum[mask] += (axis_y.float() * axis_score).unsqueeze(1)
        self.accumulate_projected_local_axis_scores(
            mask,
            batch,
            valid,
            signed_grad_x,
            signed_grad_y,
            axis_score,
        )
        self.structure_axis_denom[mask] += valid.float().unsqueeze(1).to(self.structure_axis_denom.dtype)

        center_x = (width - 1) * 0.5
        center_y = (height - 1) * 0.5
        radius = torch.sqrt((x.float() - center_x).square() + (y.float() - center_y).square())
        max_radius = max(min(width, height) * 0.5, 1.0)
        rim_threshold = float(self.residual_density_control.get("structure_rim_radius_fraction", 0.80))
        rim = torch.logical_and(valid, radius / max_radius >= rim_threshold)

        valid_scores = score[valid]
        self.structure_density_stats = {
            "structure_valid_fraction": valid.float().mean().item(),
            "structure_score_mean": valid_scores.mean().item() if valid.any() else 0.0,
            "structure_score_p95": torch.quantile(valid_scores, 0.95).item()
            if valid.any()
            else 0.0,
            "structure_weight_mean": weights.mean().item(),
            "structure_weight_max": weights.max().item(),
            "structure_axis_x_fraction": axis_x.float().mean().item(),
            "structure_axis_y_fraction": axis_y.float().mean().item(),
            "structure_rim_fraction": rim.float().mean().item(),
            "structure_edge_mean": edge[valid].mean().item() if valid.any() else 0.0,
            "structure_anisotropy_mean": anisotropy[valid].mean().item()
            if valid.any()
            else 0.0,
            "structure_scale_px_mean": scale_px[valid].mean().item() if valid.any() else 0.0,
        }
        return weights

    @torch.no_grad()
    def accumulate_projected_local_axis_scores(
        self,
        mask: torch.Tensor,
        batch,
        valid: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        axis_score: torch.Tensor,
    ) -> None:
        if not self.residual_density_control.get("structure_project_local_axes", False):
            return
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            return
        projected_axes = self.project_local_axis_vectors(mask, batch)
        if projected_axes is None:
            return
        gradient = torch.stack([grad_x, grad_y], dim=1)
        gradient = torch.nn.functional.normalize(gradient, dim=1, eps=1e-6)
        projected_axes = torch.nn.functional.normalize(
            projected_axes,
            dim=2,
            eps=1e-6,
        )
        axis_alignment = torch.abs(
            (projected_axes * gradient[:, None, :]).sum(dim=2)
        )
        axis_scores = axis_alignment * axis_score[:, None]
        axis_scores = torch.where(valid[:, None], axis_scores, torch.zeros_like(axis_scores))
        self.structure_local_axis_accum[mask] += axis_scores
        valid_axis = axis_scores[valid_indices].argmax(dim=1)
        self.structure_density_stats.update(
            {
                "structure_local_axis_0_fraction": (
                    (valid_axis == 0).float().mean().item()
                ),
                "structure_local_axis_1_fraction": (
                    (valid_axis == 1).float().mean().item()
                ),
                "structure_local_axis_2_fraction": (
                    (valid_axis == 2).float().mean().item()
                ),
            }
        )

    @torch.no_grad()
    def project_local_axis_vectors(self, mask: torch.Tensor, batch) -> torch.Tensor | None:
        intrinsics = batch.intrinsics_RationalCameraModelParameters
        if intrinsics is None:
            return None
        positions = self.model.positions[mask].float()
        if positions.numel() == 0:
            return torch.zeros((0, 3, 2), device=self.model.device)
        scales = self.model.get_scale()[mask].float().clamp_min(1e-4)
        rotations = quaternion_to_so3(self.model.rotation[mask]).float()
        local_axes = rotations.transpose(1, 2)
        endpoints = positions[:, None, :] + local_axes * scales[:, :, None]
        center_pixels, center_valid = self.project_rational_world_points(
            positions,
            batch,
            intrinsics,
        )
        endpoint_pixels, endpoint_valid = self.project_rational_world_points(
            endpoints.reshape(-1, 3),
            batch,
            intrinsics,
        )
        endpoint_pixels = endpoint_pixels.reshape(-1, 3, 2)
        endpoint_valid = endpoint_valid.reshape(-1, 3)
        axis_vectors = endpoint_pixels - center_pixels[:, None, :]
        valid_axes = torch.logical_and(center_valid[:, None], endpoint_valid)
        axis_vectors = torch.where(
            valid_axes[:, :, None],
            axis_vectors,
            torch.zeros_like(axis_vectors),
        )
        return axis_vectors

    @torch.no_grad()
    def project_rational_world_points(
        self,
        points_world: torch.Tensor,
        batch,
        intrinsics: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transform_world_to_camera = torch.linalg.inv(batch.T_to_world[0].float())
        ones = torch.ones(
            (points_world.shape[0], 1),
            device=points_world.device,
            dtype=points_world.dtype,
        )
        points_h = torch.cat([points_world, ones], dim=1)
        points_camera = (points_h @ transform_world_to_camera.T)[:, :3]
        z = points_camera[:, 2].clamp_min(1e-6)
        x = points_camera[:, 0] / z
        y = points_camera[:, 1] / z
        pixels_native = self.rational_project_xy(x, y, intrinsics)
        pixels_stored = self.native_to_stored_rational_pixels(
            pixels_native,
            intrinsics,
        )
        valid = points_camera[:, 2] > 1e-6
        return pixels_stored, valid

    @torch.no_grad()
    def rational_project_xy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        intrinsics: dict,
    ) -> torch.Tensor:
        focal = torch.as_tensor(
            intrinsics["focal_length"],
            device=x.device,
            dtype=x.dtype,
        )
        principal = torch.as_tensor(
            intrinsics["principal_point"],
            device=x.device,
            dtype=x.dtype,
        )
        numerator = torch.as_tensor(
            intrinsics["numerator_coeffs"],
            device=x.device,
            dtype=x.dtype,
        )
        denominator = torch.as_tensor(
            intrinsics["denominator_coeffs"],
            device=x.device,
            dtype=x.dtype,
        )
        affine = torch.as_tensor(
            intrinsics["affine_coeffs"],
            device=x.device,
            dtype=x.dtype,
        )
        tangential = torch.as_tensor(
            intrinsics["tangential_coeffs"],
            device=x.device,
            dtype=x.dtype,
        )
        skew = torch.as_tensor(intrinsics["skew"], device=x.device, dtype=x.dtype)
        r2 = x.square() + y.square()
        r4 = r2.square()
        r6 = r4 * r2
        radial_num = 1.0 + numerator[0] * r2 + numerator[1] * r4 + numerator[2] * r6
        radial_den = 1.0 + denominator[0] * r2 + denominator[1] * r4 + denominator[2] * r6
        radial = radial_num / radial_den.clamp_min(1e-8)
        p1 = tangential[0]
        p2 = tangential[1]
        x_distorted = x * radial + affine[0] * (
            2.0 * p1 * x * y + p2 * (r2 + 2.0 * x.square())
        )
        y_distorted = y * radial + affine[1] * (
            p1 * (r2 + 2.0 * y.square()) + 2.0 * p2 * x * y
        )
        u = focal[0] * x_distorted + skew * y_distorted + principal[0]
        v = focal[1] * y_distorted + principal[1]
        return torch.stack([u, v], dim=1)

    @torch.no_grad()
    def native_to_stored_rational_pixels(
        self,
        pixels_native: torch.Tensor,
        intrinsics: dict,
    ) -> torch.Tensor:
        resolution = torch.as_tensor(
            intrinsics["resolution"],
            device=pixels_native.device,
            dtype=pixels_native.dtype,
        )
        width = resolution[0]
        height = resolution[1]
        rotation = int(intrinsics.get("image_rotation_quadrants_cw", 0)) % 4
        u = pixels_native[:, 0]
        v = pixels_native[:, 1]
        if rotation == 1:
            return torch.stack([(width - 1.0) - v, u], dim=1)
        if rotation == 2:
            return torch.stack([(width - 1.0) - u, (height - 1.0) - v], dim=1)
        if rotation == 3:
            return torch.stack([v, (height - 1.0) - u], dim=1)
        return pixels_native

    @torch.no_grad()
    def compute_structure_maps(
        self,
        rgb_gt: torch.Tensor,
        rgb_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if rgb_gt.dim() == 4:
            rgb_gt = rgb_gt[0]
        if rgb_pred.dim() == 4:
            rgb_pred = rgb_pred[0]
        luma_gt = self.rgb_to_luma(rgb_gt)
        luma_pred = self.rgb_to_luma(rgb_pred)
        grad_x = torch.zeros_like(luma_gt)
        grad_y = torch.zeros_like(luma_gt)
        grad_x[:, :-1] = luma_gt[:, 1:] - luma_gt[:, :-1]
        grad_y[:-1, :] = luma_gt[1:, :] - luma_gt[:-1, :]
        abs_grad_x = torch.abs(grad_x)
        abs_grad_y = torch.abs(grad_y)
        edge = torch.sqrt(torch.clamp_min(grad_x.square() + grad_y.square(), 1e-12))

        tensor_xx = self.smooth_structure_map(grad_x.square())
        tensor_yy = self.smooth_structure_map(grad_y.square())
        tensor_xy = self.smooth_structure_map(grad_x * grad_y)
        trace = tensor_xx + tensor_yy
        delta = torch.sqrt(
            torch.clamp_min((tensor_xx - tensor_yy).square() + 4.0 * tensor_xy.square(), 0.0)
        )
        anisotropy = delta / trace.clamp_min(1e-6)

        laplacian = torch.zeros_like(luma_gt)
        laplacian[1:-1, 1:-1] = torch.abs(
            -4.0 * luma_gt[1:-1, 1:-1]
            + luma_gt[1:-1, :-2]
            + luma_gt[1:-1, 2:]
            + luma_gt[:-2, 1:-1]
            + luma_gt[2:, 1:-1]
        )
        scale_px = torch.clamp(edge / laplacian.clamp_min(1e-4), min=1.0, max=16.0)
        residual = torch.abs(luma_pred - luma_gt)
        return {
            "edge": edge,
            "anisotropy": anisotropy,
            "scale_px": scale_px,
            "residual": residual,
            "grad_x_signed": grad_x,
            "grad_y_signed": grad_y,
            "grad_x": abs_grad_x,
            "grad_y": abs_grad_y,
        }

    @torch.no_grad()
    def smooth_structure_map(self, image: torch.Tensor) -> torch.Tensor:
        smoothed = torch.nn.functional.avg_pool2d(
            image[None, None],
            kernel_size=5,
            stride=1,
            padding=2,
        )
        return smoothed[0, 0]

    def log_structure_density_control(self, step: int, writer) -> None:
        if not self.residual_density_control.get("use_structure_density_weight", False):
            return
        if writer is None or not hasattr(writer, "add_scalar"):
            return
        log_frequency = int(self.residual_density_control.get("structure_log_frequency", 100))
        if not check_step_condition(step, 0, -1, log_frequency):
            return
        for name, value in self.structure_density_stats.items():
            writer.add_scalar(f"diagnostics/structure_density/{name}", value, step)

    @torch.no_grad()
    def compute_edge_error_weight(self, mask: torch.Tensor, batch, outputs) -> torch.Tensor:
        selected_count = int(mask.sum().item())
        self.edge_error_density_stats = {}
        weights = torch.ones((selected_count,), device=self.model.device)
        if selected_count == 0:
            return weights
        if not self.residual_density_control.get("use_edge_error_weight", False):
            return weights
        if "mog_projected_position" not in outputs:
            return weights

        edge_error = self.compute_edge_error_map(batch.rgb_gt, outputs["pred_rgb"])
        positions = outputs["mog_projected_position"][mask].float()
        x = torch.round(positions[:, 0]).long()
        y = torch.round(positions[:, 1]).long()
        height, width = edge_error.shape
        valid = torch.logical_and(
            torch.logical_and(x >= 0, x < width),
            torch.logical_and(y >= 0, y < height),
        )
        if "mog_visibility" in outputs:
            visible = outputs["mog_visibility"][mask].reshape(-1) > 0
            valid = torch.logical_and(valid, visible)
        if batch.mask is not None:
            frame_mask = batch.mask
            if frame_mask.dim() == 4:
                frame_mask = frame_mask[0]
            frame_mask = frame_mask.squeeze(-1) > 0.5
            valid_mask_sample = torch.zeros_like(valid)
            valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if valid_indices.numel() > 0:
                valid_mask_sample[valid_indices] = frame_mask[
                    y[valid_indices],
                    x[valid_indices],
                ]
            valid = torch.logical_and(valid, valid_mask_sample)

        sampled_error = torch.zeros_like(weights)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() > 0:
            sampled_error[valid_indices] = edge_error[
                y[valid_indices],
                x[valid_indices],
            ]

        edge_reference = float(
            self.residual_density_control.get("edge_error_reference", 0.05)
        )
        edge_power = float(self.residual_density_control.get("edge_error_power", 0.5))
        weights = torch.pow(sampled_error / max(edge_reference, 1e-6), edge_power)
        weights = torch.clamp(
            weights,
            min=float(self.residual_density_control.get("edge_error_min_weight", 0.5)),
            max=float(self.residual_density_control.get("edge_error_max_weight", 3.0)),
        )
        self.edge_error_density_stats = {
            "edge_error_valid_fraction": valid.float().mean().item(),
            "edge_error_mean": sampled_error[valid].mean().item() if valid.any() else 0.0,
            "edge_error_p95": torch.quantile(sampled_error[valid], 0.95).item()
            if valid.any()
            else 0.0,
        }
        return weights

    @torch.no_grad()
    def compute_frequency_error_weight(
        self,
        mask: torch.Tensor,
        batch,
        outputs,
    ) -> torch.Tensor:
        selected_count = int(mask.sum().item())
        self.frequency_error_density_stats = {}
        weights = torch.ones((selected_count,), device=self.model.device)
        if selected_count == 0:
            return weights
        if not self.residual_density_control.get(
            "use_frequency_error_weight",
            False,
        ):
            return weights
        if "mog_projected_position" not in outputs:
            return weights

        frequency_error, gt_edge = self.compute_frequency_error_map(
            batch.rgb_gt,
            outputs["pred_rgb"],
        )
        positions = outputs["mog_projected_position"][mask].float()
        x = torch.round(positions[:, 0]).long()
        y = torch.round(positions[:, 1]).long()
        height, width = frequency_error.shape
        valid = torch.logical_and(
            torch.logical_and(x >= 0, x < width),
            torch.logical_and(y >= 0, y < height),
        )
        if "mog_visibility" in outputs:
            visible = outputs["mog_visibility"][mask].reshape(-1) > 0
            valid = torch.logical_and(valid, visible)
        if batch.mask is not None:
            frame_mask = batch.mask
            if frame_mask.dim() == 4:
                frame_mask = frame_mask[0]
            frame_mask = frame_mask.squeeze(-1) > 0.5
            valid_mask_sample = torch.zeros_like(valid)
            valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if valid_indices.numel() > 0:
                valid_mask_sample[valid_indices] = frame_mask[
                    y[valid_indices],
                    x[valid_indices],
                ]
            valid = torch.logical_and(valid, valid_mask_sample)

        sampled_error = torch.zeros_like(weights)
        sampled_edge = torch.zeros_like(weights)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() > 0:
            sample_y = y[valid_indices]
            sample_x = x[valid_indices]
            sampled_error[valid_indices] = frequency_error[sample_y, sample_x]
            sampled_edge[valid_indices] = gt_edge[sample_y, sample_x]

        reference = float(
            self.residual_density_control.get("frequency_error_reference", 0.03)
        )
        power = float(
            self.residual_density_control.get("frequency_error_power", 0.5)
        )
        weights = torch.pow(sampled_error / max(reference, 1e-6), power)
        if self.residual_density_control.get("frequency_use_edge_gate", True):
            edge_reference = float(
                self.residual_density_control.get("frequency_edge_reference", 0.05)
            )
            edge_power = float(
                self.residual_density_control.get("frequency_edge_power", 0.5)
            )
            edge_weight = torch.pow(
                sampled_edge / max(edge_reference, 1e-6),
                edge_power,
            )
            edge_weight = torch.clamp(
                edge_weight,
                min=float(
                    self.residual_density_control.get(
                        "frequency_edge_min_weight",
                        0.75,
                    )
                ),
                max=float(
                    self.residual_density_control.get(
                        "frequency_edge_max_weight",
                        2.0,
                    )
                ),
            )
            weights = weights * edge_weight
        weights = torch.clamp(
            weights,
            min=float(
                self.residual_density_control.get(
                    "frequency_error_min_weight",
                    0.5,
                )
            ),
            max=float(
                self.residual_density_control.get(
                    "frequency_error_max_weight",
                    4.0,
                )
            ),
        )
        self.frequency_error_density_stats = {
            "frequency_error_valid_fraction": valid.float().mean().item(),
            "frequency_error_mean": sampled_error[valid].mean().item()
            if valid.any()
            else 0.0,
            "frequency_error_p95": torch.quantile(sampled_error[valid], 0.95).item()
            if valid.any()
            else 0.0,
            "frequency_edge_mean": sampled_edge[valid].mean().item()
            if valid.any()
            else 0.0,
        }
        return weights

    @torch.no_grad()
    def _load_roma_precision_for_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and cache a per-image RoMa precision map.

        The map is a precomputed [H, W] tensor at the raw-fisheye image resolution,
        produced by `blk_windows.process_b2g.priors.roma_per_image_precision`.
        Indexed by image stem (e.g. "left_0009"); returns None if missing so the
        density-control weight collapses to 1.0 for that frame.
        """
        precision_dir = self.residual_density_control.get("roma_precision_dir", "")
        if not precision_dir or not image_path:
            return None
        stem = os.path.splitext(os.path.basename(image_path))[0]
        if stem in self._roma_precision_cache:
            return self._roma_precision_cache[stem]
        candidate = os.path.join(precision_dir, f"{stem}.npy")
        if not os.path.exists(candidate):
            self._roma_precision_cache[stem] = None
            return None
        array = np.load(candidate).astype(np.float32)
        tensor = torch.from_numpy(array).to(self.model.device)
        self._roma_precision_cache[stem] = tensor
        return tensor

    @torch.no_grad()
    def compute_roma_precision_weight(self, mask: torch.Tensor, batch, outputs) -> torch.Tensor:
        """Per-splat density-control weight from per-image RoMa overlap precision.

        Mirrors `compute_edge_error_weight`: looks up a [H, W] precision map at each
        splat's projected pixel position. Splats projecting into a high-precision
        pixel (cross-view-confirmed by RoMa) get up-weighted; rim / texture-poor
        splats where RoMa says nothing matches get down-weighted. Phase 3a target.

        Returns torch.ones (neutral multiplier) when the feature is disabled, the
        sidecar is missing, or no projected positions are available.
        """
        selected_count = int(mask.sum().item())
        self.roma_precision_density_stats = {}
        weights = torch.ones((selected_count,), device=self.model.device)
        if selected_count == 0:
            return weights
        if not self.residual_density_control.get("use_roma_precision_weight", False):
            return weights
        if "mog_projected_position" not in outputs:
            return weights
        precision_map = self._load_roma_precision_for_image(getattr(batch, "image_path", ""))
        if precision_map is None:
            return weights

        positions = outputs["mog_projected_position"][mask].float()
        x = torch.round(positions[:, 0]).long()
        y = torch.round(positions[:, 1]).long()
        height, width = precision_map.shape
        valid = torch.logical_and(
            torch.logical_and(x >= 0, x < width),
            torch.logical_and(y >= 0, y < height),
        )
        if "mog_visibility" in outputs:
            visible = outputs["mog_visibility"][mask].reshape(-1) > 0
            valid = torch.logical_and(valid, visible)

        sampled = torch.zeros_like(weights)
        valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if valid_indices.numel() > 0:
            sampled[valid_indices] = precision_map[y[valid_indices], x[valid_indices]]

        reference = float(self.residual_density_control.get("roma_precision_reference", 0.5))
        power = float(self.residual_density_control.get("roma_precision_power", 1.0))
        weights = torch.pow(sampled / max(reference, 1e-6), power)
        weights = torch.clamp(
            weights,
            min=float(self.residual_density_control.get("roma_precision_min_weight", 0.5)),
            max=float(self.residual_density_control.get("roma_precision_max_weight", 3.0)),
        )
        self.roma_precision_density_stats = {
            "roma_precision_valid_fraction": valid.float().mean().item(),
            "roma_precision_mean": sampled[valid].mean().item() if valid.any() else 0.0,
            "roma_precision_p95": torch.quantile(sampled[valid], 0.95).item()
            if valid.any()
            else 0.0,
        }
        return weights

    @torch.no_grad()
    def compute_edge_error_map(
        self,
        rgb_gt: torch.Tensor,
        rgb_pred: torch.Tensor,
    ) -> torch.Tensor:
        if rgb_gt.dim() == 4:
            rgb_gt = rgb_gt[0]
        if rgb_pred.dim() == 4:
            rgb_pred = rgb_pred[0]
        luma_gt = self.rgb_to_luma(rgb_gt)
        luma_pred = self.rgb_to_luma(rgb_pred)
        return torch.abs(
            self.image_gradient_magnitude(luma_pred)
            - self.image_gradient_magnitude(luma_gt)
        )

    @torch.no_grad()
    def compute_frequency_error_map(
        self,
        rgb_gt: torch.Tensor,
        rgb_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb_gt.dim() == 4:
            rgb_gt = rgb_gt[0]
        if rgb_pred.dim() == 4:
            rgb_pred = rgb_pred[0]
        luma_gt = self.rgb_to_luma(rgb_gt)
        luma_pred = self.rgb_to_luma(rgb_pred)
        frequency_error = torch.abs(
            self.image_laplacian_abs(luma_pred)
            - self.image_laplacian_abs(luma_gt)
        )
        gt_edge = self.image_gradient_magnitude(luma_gt)
        return frequency_error, gt_edge

    @torch.no_grad()
    def rgb_to_luma(self, rgb: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(
            [0.299, 0.587, 0.114],
            device=rgb.device,
            dtype=rgb.dtype,
        )
        return torch.sum(rgb[..., :3] * weights, dim=-1)

    @torch.no_grad()
    def image_gradient_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        grad_x = torch.zeros_like(image)
        grad_y = torch.zeros_like(image)
        grad_x[:, :-1] = torch.abs(image[:, 1:] - image[:, :-1])
        grad_y[:-1, :] = torch.abs(image[1:, :] - image[:-1, :])
        return torch.sqrt(torch.clamp_min(grad_x.square() + grad_y.square(), 1e-12))

    @torch.no_grad()
    def image_laplacian_abs(self, image: torch.Tensor) -> torch.Tensor:
        laplacian = torch.zeros_like(image)
        laplacian[1:-1, 1:-1] = (
            -4.0 * image[1:-1, 1:-1]
            + image[1:-1, :-2]
            + image[1:-1, 2:]
            + image[:-2, 1:-1]
            + image[2:, 1:-1]
        )
        return torch.abs(laplacian)

    @torch.no_grad()
    def compute_projected_area(self, mask: torch.Tensor, batch, outputs=None) -> torch.Tensor:
        area_source = self.residual_density_control.get("area_source", "projected_extent")
        if (
            outputs is not None
            and area_source == "tile_count"
            and "mog_tiles_count" in outputs
        ):
            tile_count = outputs["mog_tiles_count"][mask].float().reshape(-1)
            if tile_count.numel() == 0:
                return torch.ones((0,), device=self.model.device)
            area = tile_count * float(self.residual_density_control.get("tile_area_px", 256.0))
            if "mog_visibility" in outputs:
                visible = outputs["mog_visibility"][mask].reshape(-1) > 0
                area = torch.where(visible, area, torch.zeros_like(area))
            return area.clamp_min(1e-6)

        if outputs is not None and "mog_projected_extent" in outputs:
            extent = outputs["mog_projected_extent"][mask].float().abs()
            if extent.numel() == 0:
                return torch.ones((0,), device=self.model.device)
            area = torch.pi * extent[:, 0] * extent[:, 1]
            area = torch.where(torch.isfinite(area), area, torch.zeros_like(area))
            max_area = float(self.residual_density_control.get("max_area_px", 65536.0))
            area = torch.clamp(area, max=max_area)
            if "mog_visibility" in outputs:
                visible = outputs["mog_visibility"][mask].reshape(-1) > 0
                area = torch.where(visible, area, torch.zeros_like(area))
            return area.clamp_min(1e-6)

        radius_px = self.compute_projected_radius(mask, batch, outputs)
        return (torch.pi * torch.square(radius_px)).clamp_min(1e-6)

    @torch.no_grad()
    def compute_projected_radius(self, mask: torch.Tensor, batch, outputs=None) -> torch.Tensor:
        if outputs is not None and "mog_projected_extent" in outputs:
            area = self.compute_projected_area(mask, batch, outputs)
            return torch.sqrt(area / torch.pi).clamp_min(1e-6)

        selected_positions = self.model.positions[mask]
        pose = batch.T_to_world[0]
        world_to_camera = torch.linalg.inv(pose)
        homogeneous = torch.cat(
            (selected_positions, torch.ones_like(selected_positions[:, :1])),
            dim=1,
        )
        camera_positions = homogeneous @ world_to_camera.transpose(0, 1)
        depth = camera_positions[:, 2].clamp_min(1e-3)
        in_front = camera_positions[:, 2] > 1e-3

        scales = self.model.get_scale()[mask].float().amax(dim=1)
        focal_mean = self.camera_focal_mean(batch)
        radius_px = scales * focal_mean / depth
        radius_px[~in_front] = 0.0
        return radius_px.clamp_min(1e-6)

    @torch.no_grad()
    def compute_frame_residual(self, batch, outputs) -> float:
        rgb_gt = batch.rgb_gt
        rgb_pred = outputs["pred_rgb"]
        residual = torch.abs(rgb_pred - rgb_gt).mean(dim=-1, keepdim=True)
        if self.residual_density_control.get("use_masked_residual", True) and batch.mask is not None:
            denominator = torch.clamp(batch.mask.sum(), min=1.0)
            return float((residual * batch.mask).sum().item() / denominator.item())
        return float(residual.mean().item())

    def batch_camera_index(self, batch) -> int:
        camera_idx = torch.as_tensor(batch.camera_idx).reshape(-1)[0]
        return int(camera_idx.item())

    def ensure_camera_balance_capacity(self, camera_idx: int) -> None:
        required_count = camera_idx + 1
        if self.camera_balance_residual_ema.numel() >= required_count:
            return
        resized = torch.zeros((required_count,), device=self.model.device)
        if self.camera_balance_residual_ema.numel() > 0:
            resized[: self.camera_balance_residual_ema.numel()] = (
                self.camera_balance_residual_ema
            )
        self.camera_balance_residual_ema = resized

    @torch.no_grad()
    def compute_camera_balance_weight(self, batch, residual_scalar: float) -> float:
        if not self.residual_density_control.get("use_camera_balance", False):
            return 1.0

        camera_idx = self.batch_camera_index(batch)
        self.ensure_camera_balance_capacity(camera_idx)
        residual = torch.tensor(
            float(residual_scalar),
            device=self.model.device,
            dtype=self.camera_balance_residual_ema.dtype,
        )
        decay = float(
            self.residual_density_control.get("camera_balance_ema_decay", 0.95)
        )
        previous = self.camera_balance_residual_ema[camera_idx]
        if bool((previous <= 0.0).item()):
            updated = residual
        else:
            updated = decay * previous + (1.0 - decay) * residual
        self.camera_balance_residual_ema[camera_idx] = updated

        observed = self.camera_balance_residual_ema[
            self.camera_balance_residual_ema > 0.0
        ]
        reference = observed.mean().clamp_min(1.0e-6)
        power = float(self.residual_density_control.get("camera_balance_power", 1.0))
        raw_weight = torch.pow(updated / reference, power)
        weight = torch.clamp(
            raw_weight,
            min=float(
                self.residual_density_control.get("camera_balance_min_weight", 0.75)
            ),
            max=float(
                self.residual_density_control.get("camera_balance_max_weight", 2.0)
            ),
        )
        self.camera_balance_density_stats = {
            "camera_balance_camera_idx": float(camera_idx),
            "camera_balance_current_residual": float(residual.item()),
            "camera_balance_current_ema": float(updated.item()),
            "camera_balance_reference_ema": float(reference.item()),
            "camera_balance_weight": float(weight.item()),
            "camera_balance_observed_count": float(observed.numel()),
            "camera_balance_ema_min": float(observed.min().item()),
            "camera_balance_ema_max": float(observed.max().item()),
        }
        return float(weight.item())

    @torch.no_grad()
    def compute_responsibility_gradient_weight(self, mask: torch.Tensor) -> torch.Tensor:
        weights = torch.ones((int(mask.sum()),), device=self.model.device)
        color_grad = self.model.features_albedo.grad
        if color_grad is not None:
            color_signal = torch.norm(color_grad[mask], dim=-1)
            weights = weights + float(self.residual_density_control.get("color_grad_weight", 0.5)) * self.normalize_signal(
                color_signal
            )

        density_grad = self.model.density.grad
        if density_grad is not None:
            density_signal = torch.abs(density_grad[mask].squeeze(-1))
            weights = weights + float(
                self.residual_density_control.get("density_grad_weight", 0.25)
            ) * self.normalize_signal(density_signal)

        return torch.clamp(
            weights,
            max=float(self.residual_density_control.get("gradient_max_weight", 3.0)),
        )

    @torch.no_grad()
    def normalize_signal(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.numel() == 0:
            return signal
        positive = signal[signal > 0]
        if positive.numel() == 0:
            return torch.zeros_like(signal)
        return signal / positive.mean().clamp_min(1e-12)

    def log_residual_density_control(self, step: int, writer) -> None:
        if not self.residual_density_control.get("enabled", False):
            return
        if writer is None or not hasattr(writer, "add_scalar"):
            return
        log_frequency = int(self.residual_density_control.get("log_frequency", 100))
        if not check_step_condition(step, 0, -1, log_frequency):
            return
        for name, value in self.residual_density_stats.items():
            writer.add_scalar(f"diagnostics/residual_density_control/{name}", value, step)

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, scene_extent):
        assert (
            self.model.optimizer is not None
        ), "Optimizer need to be initialized before splitting and cloning the Gaussians"
        densify_grad_norm = self.densify_grad_norm_accum / self.densify_grad_norm_denom
        densify_grad_norm[densify_grad_norm.isnan()] = 0.0
        if self.residual_density_control.get("use_abs_gradient_score", False):
            densify_grad_norm = self.compute_abs_gradient_densify_score(
                densify_grad_norm,
            )

        if self.residual_density_control.get("structure_score_only", False):
            self.log_structure_score_only_candidates(
                densify_grad_norm.squeeze(),
                scene_extent,
            )
            self.reset_densification_buffers()
            return

        self.clone_gaussians(densify_grad_norm.squeeze(), scene_extent)
        self.split_gaussians(densify_grad_norm.squeeze(), scene_extent)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def compute_abs_gradient_densify_score(
        self,
        densify_grad_norm: torch.Tensor,
    ) -> torch.Tensor:
        denom = self.densify_grad_norm_denom.float().clamp_min(1.0)
        active = self.densify_grad_norm_denom > 0
        abs_score = self.densify_abs_grad_norm_accum / denom
        signed_vector = self.densify_signed_grad_accum / denom
        signed_vector_score = torch.norm(signed_vector, dim=-1, keepdim=True)
        collision_ratio = abs_score / signed_vector_score.clamp_min(1e-12)
        collision_ratio = torch.where(
            active,
            collision_ratio,
            torch.ones_like(collision_ratio),
        )
        reference = float(
            self.residual_density_control.get(
                "abs_gradient_collision_reference",
                1.0,
            )
        )
        power = float(
            self.residual_density_control.get(
                "abs_gradient_collision_power",
                0.5,
            )
        )
        collision_weight = torch.pow(
            collision_ratio / max(reference, 1e-6),
            power,
        )
        collision_weight = torch.clamp(
            collision_weight,
            min=float(
                self.residual_density_control.get(
                    "abs_gradient_collision_min_weight",
                    1.0,
                )
            ),
            max=float(
                self.residual_density_control.get(
                    "abs_gradient_collision_max_weight",
                    3.0,
                )
            ),
        )
        score = densify_grad_norm * collision_weight
        self.abs_gradient_density_stats = {
            "abs_gradient_active_fraction": active.float().mean().item(),
            "abs_gradient_score_mean": abs_score[active].mean().item()
            if active.any()
            else 0.0,
            "abs_gradient_signed_vector_score_mean": signed_vector_score[
                active
            ].mean().item()
            if active.any()
            else 0.0,
            "abs_gradient_collision_ratio_mean": collision_ratio[
                active
            ].mean().item()
            if active.any()
            else 0.0,
            "abs_gradient_collision_ratio_p95": torch.quantile(
                collision_ratio[active],
                0.95,
            ).item()
            if active.any()
            else 0.0,
            "abs_gradient_collision_weight_mean": collision_weight[
                active
            ].mean().item()
            if active.any()
            else 0.0,
            "abs_gradient_collision_weight_max": collision_weight[
                active
            ].max().item()
            if active.any()
            else 0.0,
        }
        self.residual_density_stats.update(self.abs_gradient_density_stats)
        return score

    @torch.no_grad()
    def log_structure_score_only_candidates(
        self,
        densify_grad_norm: torch.Tensor,
        scene_extent: float,
    ) -> None:
        n_init_points = self.model.num_gaussians
        padded_grad = torch.zeros((n_init_points), device=self.model.device)
        padded_grad[: densify_grad_norm.shape[0]] = densify_grad_norm.squeeze()
        clone_mask = padded_grad >= self.clone_grad_threshold
        clone_mask = torch.logical_and(
            clone_mask,
            torch.max(self.model.get_scale(), dim=1).values
            <= self.relative_size_threshold * scene_extent,
        )
        split_mask = padded_grad >= self.split_grad_threshold
        split_mask = torch.logical_and(
            split_mask,
            torch.max(self.model.get_scale(), dim=1).values
            > self.relative_size_threshold * scene_extent,
        )
        candidate_mask = torch.logical_or(clone_mask, split_mask)
        candidate_scores = padded_grad[candidate_mask]
        axis_denom = self.structure_axis_denom.clamp_min(1)
        axis_x_score = self.structure_axis_x_accum / axis_denom
        axis_y_score = self.structure_axis_y_accum / axis_denom
        axis_x = torch.logical_and(candidate_mask[:, None], axis_x_score >= axis_y_score)
        axis_y = torch.logical_and(candidate_mask[:, None], axis_y_score > axis_x_score)
        candidate_count = candidate_mask.float().sum().clamp_min(1.0)
        self.structure_density_stats.update(
            {
                "score_only_candidate_count": float(candidate_mask.sum().item()),
                "score_only_clone_candidate_count": float(clone_mask.sum().item()),
                "score_only_split_candidate_count": float(split_mask.sum().item()),
                "score_only_score_mean": candidate_scores.mean().item()
                if candidate_scores.numel() > 0
                else 0.0,
                "score_only_score_p95": torch.quantile(candidate_scores, 0.95).item()
                if candidate_scores.numel() > 0
                else 0.0,
                "score_only_axis_x_fraction": (
                    axis_x.float().sum() / candidate_count
                ).item(),
                "score_only_axis_y_fraction": (
                    axis_y.float().sum() / candidate_count
                ).item(),
            }
        )

    @torch.cuda.nvtx.range("split_gaussians")
    def split_gaussians(self, densify_grad_norm: torch.Tensor, scene_extent: float):
        n_init_points = self.model.num_gaussians

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")

        # Here we already have the cloned points in the self.model.positions so only take the points up to size of the initial grad
        padded_grad[: densify_grad_norm.shape[0]] = densify_grad_norm.squeeze()
        mask = torch.where(padded_grad >= self.split_grad_threshold, True, False)
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values > self.relative_size_threshold * scene_extent
        )
        mask = self.cap_residual_density_candidates(mask, padded_grad, "split")

        selected_scale = self.model.get_scale()[mask]
        structure_axis_index = self.structure_split_axis_index(mask)
        if structure_axis_index is not None:
            row_index = torch.arange(
                selected_scale.shape[0],
                device=selected_scale.device,
            )
            axis_stds = selected_scale.clone()
            anisotropic_mask = structure_axis_index >= 0
            if anisotropic_mask.any():
                anisotropic_rows = row_index[anisotropic_mask]
                anisotropic_axes = structure_axis_index[anisotropic_mask]
                axis_stds[anisotropic_mask] = 0.0
                axis_stds[anisotropic_rows, anisotropic_axes] = selected_scale[
                    anisotropic_rows,
                    anisotropic_axes,
                ]
            stds = axis_stds.repeat(self.split_n_gaussians, 1)
            selected_axis_index = structure_axis_index[anisotropic_mask]
            evidence_fraction = anisotropic_mask.float().mean().item()
            self.structure_density_stats["anisotropic_split_axis_x_fraction"] = (
                (selected_axis_index == 0).float().mean().item()
                if selected_axis_index.numel() > 0
                else 0.0
            )
            self.structure_density_stats["anisotropic_split_axis_y_fraction"] = (
                (selected_axis_index == 1).float().mean().item()
                if selected_axis_index.numel() > 0
                else 0.0
            )
            self.structure_density_stats["anisotropic_split_axis_z_fraction"] = (
                (selected_axis_index == 2).float().mean().item()
                if selected_axis_index.numel() > 0
                else 0.0
            )
            self.structure_density_stats["anisotropic_split_evidence_fraction"] = (
                evidence_fraction
            )
        else:
            stds = selected_scale.repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask]).repeat(self.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        structure_axis_index_repeated = (
            structure_axis_index.repeat(self.split_n_gaussians)
            if structure_axis_index is not None
            else None
        )
        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Splitted {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            repeats = [self.split_n_gaussians] + [1] * (param.dim() - 1)
            if name == "positions":
                p_split = param[mask].repeat(repeats) + offsets  # [2N, 3]
            elif name == "scale":
                split_scale = self.model.scale_activation(
                    param[mask].repeat(repeats)
                ) / (0.8 * self.split_n_gaussians)
                if structure_axis_index_repeated is not None:
                    axis_divisor = float(
                        self.residual_density_control.get(
                            "structure_split_axis_scale_divisor",
                            1.5,
                        )
                    )
                    valid_axis = structure_axis_index_repeated >= 0
                    if valid_axis.any():
                        split_rows = torch.arange(
                            split_scale.shape[0],
                            device=split_scale.device,
                        )[valid_axis]
                        split_axes = structure_axis_index_repeated[valid_axis]
                        split_scale[split_rows, split_axes] = (
                            split_scale[split_rows, split_axes] / axis_divisor
                        )
                p_split = self.model.scale_activation_inv(split_scale)
            else:
                p_split = param[mask].repeat(repeats)

            p_new = torch.nn.Parameter(torch.cat([param[~mask], p_split]), requires_grad=param.requires_grad)

            return p_new

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros((self.split_n_gaussians * int(mask.sum()), *v.shape[1:]), device=v.device)
            return torch.cat([v[~mask], v_split])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

    @torch.no_grad()
    def structure_split_axis_index(self, mask: torch.Tensor) -> torch.Tensor | None:
        if not self.residual_density_control.get("use_structure_density_weight", False):
            return None
        if self.residual_density_control.get("structure_score_only", False):
            return None
        if not self.residual_density_control.get("structure_anisotropic_split", False):
            return None
        if not mask.any():
            return None
        if self.residual_density_control.get("structure_project_local_axes", False):
            axis_denom = self.structure_axis_denom.clamp_min(1)
            local_axis_score = self.structure_local_axis_accum / axis_denom
            selected_score = local_axis_score[mask]
            axis_index = selected_score.argmax(dim=1)
            has_evidence = selected_score.sum(dim=1) > 0
            return torch.where(
                has_evidence,
                axis_index,
                torch.full_like(axis_index, -1),
            )
        axis_denom = self.structure_axis_denom.clamp_min(1)
        axis_x_score = (self.structure_axis_x_accum / axis_denom)[mask].reshape(-1)
        axis_y_score = (self.structure_axis_y_accum / axis_denom)[mask].reshape(-1)
        axis_index = torch.where(
            axis_x_score >= axis_y_score,
            torch.zeros_like(axis_x_score, dtype=torch.long),
            torch.ones_like(axis_y_score, dtype=torch.long),
        )
        has_evidence = (axis_x_score + axis_y_score) > 0
        return torch.where(
            has_evidence,
            axis_index,
            torch.full_like(axis_index, -1),
        )

    @torch.cuda.nvtx.range("clone_gaussians")
    def clone_gaussians(self, densify_grad_norm: torch.Tensor, scene_extent: float):
        assert densify_grad_norm is not None, "Positional gradients must be available in order to clone the Gaussians"
        # Extract points that satisfy the gradient condition
        mask = torch.where(densify_grad_norm >= self.clone_grad_threshold, True, False)

        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(
            mask, torch.max(self.model.get_scale(), dim=1).values <= self.relative_size_threshold * scene_extent
        )
        mask = self.cap_residual_density_candidates(mask, densify_grad_norm, "clone")

        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Cloned {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        clone_scale_multiplier = float(
            self.residual_density_control.get("clone_scale_multiplier", 1.0)
        )
        clone_density_multiplier = float(
            self.residual_density_control.get("clone_density_multiplier", 1.0)
        )
        clone_min_scale = float(
            self.residual_density_control.get("clone_min_scale", 0.0)
        )
        if self.residual_density_control.get("enabled", False):
            self.residual_density_stats["clone_scale_multiplier"] = clone_scale_multiplier
            self.residual_density_stats["clone_density_multiplier"] = (
                clone_density_multiplier
            )
            self.residual_density_stats["clone_min_scale"] = clone_min_scale

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            cloned_param = param[mask]
            if name == "scale" and clone_scale_multiplier != 1.0:
                cloned_scale = self.model.scale_activation(cloned_param)
                cloned_scale = cloned_scale * clone_scale_multiplier
                if clone_min_scale > 0.0:
                    cloned_scale = cloned_scale.clamp_min(clone_min_scale)
                cloned_param = self.model.scale_activation_inv(cloned_scale)
            elif name == "density" and clone_density_multiplier != 1.0:
                cloned_density = self.model.density_activation(cloned_param)
                cloned_density = cloned_density * clone_density_multiplier
                cloned_param = self.model.density_activation_inv(cloned_density)
            param_new = torch.cat([param, cloned_param])
            return torch.nn.Parameter(param_new, requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat([v, torch.zeros((int(mask.sum()), *v.shape[1:]), device=v.device)])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        if self.should_preserve_structure_buffers_for_split():
            self.extend_densification_buffers_for_clones(int(mask.sum().item()))
        else:
            self.reset_densification_buffers()

    def should_preserve_structure_buffers_for_split(self) -> bool:
        return bool(
            self.residual_density_control.get("enabled", False)
            and self.residual_density_control.get("use_structure_density_weight", False)
            and self.residual_density_control.get("structure_anisotropic_split", False)
        )

    def extend_densification_buffers_for_clones(self, clone_count: int) -> None:
        buffer_shapes = (
            ("densify_grad_norm_accum", self.densify_grad_norm_accum, 1),
            ("densify_grad_norm_denom", self.densify_grad_norm_denom, 1),
            (
                "densify_abs_grad_norm_accum",
                self.densify_abs_grad_norm_accum,
                1,
            ),
            ("densify_signed_grad_accum", self.densify_signed_grad_accum, 3),
            ("structure_axis_x_accum", self.structure_axis_x_accum, 1),
            ("structure_axis_y_accum", self.structure_axis_y_accum, 1),
            ("structure_local_axis_accum", self.structure_local_axis_accum, 3),
            ("structure_axis_denom", self.structure_axis_denom, 1),
        )
        for name, buffer, width in buffer_shapes:
            clone_buffer = torch.zeros(
                (clone_count, width),
                device=self.model.device,
                dtype=buffer.dtype,
            )
            setattr(self, name, torch.cat([buffer, clone_buffer], dim=0))

    @torch.no_grad()
    def cap_residual_density_candidates(
        self,
        mask: torch.Tensor,
        scores: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        if not self.residual_density_control.get("enabled", False):
            return mask
        candidate_count = int(mask.sum().item())
        if candidate_count == 0:
            self.residual_density_stats[f"{label}_candidate_count"] = 0.0
            self.residual_density_stats[f"{label}_kept_count"] = 0.0
            return mask

        max_fraction = float(
            self.residual_density_control.get("max_candidate_fraction", 0.001)
        )
        max_by_fraction = max(1, int(mask.numel() * max_fraction))
        max_key = f"max_{label}_candidates_per_step"
        max_by_count = int(self.residual_density_control.get(max_key, 50000))
        keep_count = min(candidate_count, max_by_fraction, max_by_count)
        self.residual_density_stats[f"{label}_candidate_count"] = float(candidate_count)
        self.residual_density_stats[f"{label}_kept_count"] = float(keep_count)
        if keep_count >= candidate_count:
            return mask

        candidate_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        candidate_scores = scores[candidate_indices]
        top_indices = torch.topk(candidate_scores, keep_count).indices
        capped_mask = torch.zeros_like(mask)
        capped_mask[candidate_indices[top_indices]] = True
        return capped_mask

    def prune_gaussians_weight(self):
        # Prune the Gaussians based on their weight
        mask = self.model.rolling_weight_contrib[:, 0] >= self.conf.strategy.prune_weight.weight_threshold
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Weight-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[mask], requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def prune_gaussians_scale(self, dataset):
        cam_normals = torch.from_numpy(dataset.poses[:, :3, 2]).to(self.model.device)
        similarities = torch.matmul(self.model.positions, cam_normals.T)
        cam_dists = similarities.min(dim=1)[0].clamp(min=1e-8)
        ratio = self.model.get_scale().min(dim=1)[0] / cam_dists * dataset.intrinsic[0].max()

        # Prune the Gaussians based on their weight
        mask = ratio >= self.conf.strategy.prune_scale.threshold
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Scale-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[mask], requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def prune_gaussians_opacity(self):
        # Prune the Gaussians based on their opacity
        mask = self.model.get_density().squeeze() >= self.prune_density_threshold

        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Density-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[mask], requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def reset_densification_buffers(self) -> None:
        self.densify_grad_norm_accum = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.densify_grad_norm_accum.dtype,
        )

        self.densify_grad_norm_denom = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.densify_grad_norm_denom.dtype,
        )
        self.densify_abs_grad_norm_accum = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.densify_abs_grad_norm_accum.dtype,
        )
        self.densify_signed_grad_accum = torch.zeros(
            (self.model.get_positions().shape[0], 3),
            device=self.model.device,
            dtype=self.densify_signed_grad_accum.dtype,
        )
        self.structure_axis_x_accum = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.structure_axis_x_accum.dtype,
        )
        self.structure_axis_y_accum = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.structure_axis_y_accum.dtype,
        )
        self.structure_local_axis_accum = torch.zeros(
            (self.model.get_positions().shape[0], 3),
            device=self.model.device,
            dtype=self.structure_local_axis_accum.dtype,
        )
        self.structure_axis_denom = torch.zeros(
            (self.model.get_positions().shape[0], 1),
            device=self.model.device,
            dtype=self.structure_axis_denom.dtype,
        )

    def prune_densification_buffers(self, valid_mask: torch.Tensor) -> None:
        # Update non-optimizable buffers
        self.densify_grad_norm_accum = self.densify_grad_norm_accum[valid_mask]
        self.densify_grad_norm_denom = self.densify_grad_norm_denom[valid_mask]
        self.densify_abs_grad_norm_accum = self.densify_abs_grad_norm_accum[
            valid_mask
        ]
        self.densify_signed_grad_accum = self.densify_signed_grad_accum[
            valid_mask
        ]
        self.structure_axis_x_accum = self.structure_axis_x_accum[valid_mask]
        self.structure_axis_y_accum = self.structure_axis_y_accum[valid_mask]
        self.structure_local_axis_accum = self.structure_local_axis_accum[valid_mask]
        self.structure_axis_denom = self.structure_axis_denom[valid_mask]

    def decay_density(self):
        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name == "density", "wrong paramaeter passed to update_param_fn"

            decayed_densities = self.model.density_activation_inv(
                self.model.get_density() * self.conf.strategy.density_decay.gamma
            )

            return torch.nn.Parameter(decayed_densities, requires_grad=param.requires_grad)

        self._update_param_with_optimizer(update_param_fn, None, names=["density"])

    def reset_density(self):
        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name == "density", "wrong paramaeter passed to update_param_fn"
            densities = torch.clamp(
                param,
                max=self.model.density_activation_inv(torch.tensor(self.new_max_density)).item(),
            )
            return torch.nn.Parameter(densities)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn, names=["density"])
