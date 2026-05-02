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

from typing import Optional

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
        self.camera_balance_density_stats = {}
        self.camera_balance_residual_ema = torch.empty((0,), device=self.model.device)

    def get_strategy_parameters(self) -> dict:
        params = {}

        params["densify_grad_norm_accum"] = (self.densify_grad_norm_accum,)
        params["densify_grad_norm_denom"] = (self.densify_grad_norm_denom,)

        return params

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        if checkpoint is not None:
            self.densify_grad_norm_accum = checkpoint["densify_grad_norm_accum"][0].detach()
            self.densify_grad_norm_denom = checkpoint["densify_grad_norm_denom"][0].detach()
        else:
            num_gaussians = self.model.num_gaussians
            self.densify_grad_norm_accum = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.densify_grad_norm_denom = torch.zeros((num_gaussians, 1), dtype=torch.int, device=self.model.device)

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
        grad_norm = torch.norm(params_grad[mask] * distance_to_camera, dim=-1, keepdim=True) / 2

        if self.footprint_control.get("enabled", False) and batch is not None:
            grad_norm = grad_norm * self.compute_footprint_weights(mask, batch, outputs)

        if self.residual_density_control.get("enabled", False) and batch is not None and outputs is not None:
            grad_norm = grad_norm * self.compute_residual_density_weights(mask, batch, outputs)

        self.densify_grad_norm_accum[mask] += grad_norm
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
        total_weight = (
            area_weight
            * residual_weight
            * gradient_weight
            * edge_error_weight
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
            "camera_balance_weight": float(camera_balance_weight),
            "total_weight_mean": total_weight.mean().item(),
            "total_weight_max": total_weight.max().item(),
        }
        self.residual_density_stats.update(self.edge_error_density_stats)
        self.residual_density_stats.update(self.camera_balance_density_stats)
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

        self.clone_gaussians(densify_grad_norm.squeeze(), scene_extent)
        self.split_gaussians(densify_grad_norm.squeeze(), scene_extent)

        torch.cuda.empty_cache()

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

        stds = self.model.get_scale()[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask]).repeat(self.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
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
                p_split = self.model.scale_activation_inv(
                    self.model.scale_activation(param[mask].repeat(repeats)) / (0.8 * self.split_n_gaussians)
                )
            else:
                p_split = param[mask].repeat(repeats)

            p_new = torch.nn.Parameter(torch.cat([param[~mask], p_split]), requires_grad=param.requires_grad)

            return p_new

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros((self.split_n_gaussians * int(mask.sum()), *v.shape[1:]), device=v.device)
            return torch.cat([v[~mask], v_split])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

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
        self.reset_densification_buffers()

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

    def prune_densification_buffers(self, valid_mask: torch.Tensor) -> None:
        # Update non-optimizable buffers
        self.densify_grad_norm_accum = self.densify_grad_norm_accum[valid_mask]
        self.densify_grad_norm_denom = self.densify_grad_norm_denom[valid_mask]

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
