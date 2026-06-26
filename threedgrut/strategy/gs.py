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
from beartype import beartype
from jaxtyping import Float, jaxtyped

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    check_step_condition,
    quaternion_to_so3,
    sh_degree_to_specular_dim,
)


class GSStrategy(BaseStrategy):
    def __init__(self, config, model: MixtureOfGaussians) -> None:
        super().__init__(config=config, model=model)

        # Parameters related to densification, pruning and reset
        self.split_n_gaussians = self.conf.strategy.densify.split.n_gaussians
        self.relative_size_threshold = (
            self.conf.strategy.densify.relative_size_threshold
        )
        self.prune_density_threshold = (
            self.conf.strategy.prune.density_threshold
        )
        self.clone_grad_threshold = (
            self.conf.strategy.densify.clone_grad_threshold
        )
        self.split_grad_threshold = (
            self.conf.strategy.densify.split_grad_threshold
        )
        self.new_max_density = self.conf.strategy.reset_density.new_max_density

        # Field-angle-aware (solid-angle-weighted) densification config. When
        # enabled, the rim (high field angle theta) densifies more and keeps
        # smaller Gaussians, compensating the fisheye projection Jacobian that
        # screen-space density control is blind to. Defaults keep the upstream
        # scalar-threshold behaviour byte-identical.
        theta_cfg = getattr(self.conf.strategy.densify, "theta_aware", None)
        self.theta_aware = bool(getattr(theta_cfg, "enabled", False))
        self.theta_power = float(getattr(theta_cfg, "theta_power", 1.5))
        self.theta_size_cap_power = float(
            getattr(theta_cfg, "size_cap_power", 1.0)
        )
        self.theta_min_cos = float(getattr(theta_cfg, "min_cos", 0.1))
        feature_cfg = self.conf.strategy.densify.get("feature_grad", {})
        self.feature_grad_aware = bool(feature_cfg.get("enabled", False))
        self.feature_grad_power = float(feature_cfg.get("power", 0.5))
        self.feature_grad_quantile = float(feature_cfg.get("quantile", 0.95))
        self.feature_grad_max_boost = float(
            feature_cfg.get("max_boost", 4.0)
        )
        self.feature_grad_carrier_tail_only = bool(
            feature_cfg.get("carrier_tail_only", False)
        )
        if self.feature_grad_power <= 0:
            msg = (
                "strategy.densify.feature_grad.power must be positive; "
                f"got {self.feature_grad_power}."
            )
            raise ValueError(msg)
        if not 0 < self.feature_grad_quantile <= 1:
            msg = (
                "strategy.densify.feature_grad.quantile must be in (0, 1]; "
                f"got {self.feature_grad_quantile}."
            )
            raise ValueError(msg)
        if self.feature_grad_max_boost < 1:
            msg = (
                "strategy.densify.feature_grad.max_boost must be >= 1; "
                f"got {self.feature_grad_max_boost}."
            )
            raise ValueError(msg)

        # Accumulation of the norms of the positions gradients
        self.densify_grad_norm_accum = torch.empty([0, 1])
        self.densify_grad_norm_denom = torch.empty([0, 1])
        # Gradient-weighted accumulation of cos(theta) (and its weight) so the
        # mean field angle each Gaussian was observed-with can be recovered at
        # densify time. Only used when theta_aware is enabled.
        self.densify_cos_accum = torch.empty([0, 1])
        self.densify_cos_weight = torch.empty([0, 1])
        self.densify_feature_grad_accum = torch.empty([0, 1])
        self.densify_feature_grad_denom = torch.empty([0, 1])

    def get_strategy_parameters(self) -> dict:
        params = {}

        params["densify_grad_norm_accum"] = (self.densify_grad_norm_accum,)
        params["densify_grad_norm_denom"] = (self.densify_grad_norm_denom,)
        params["densify_cos_accum"] = (self.densify_cos_accum,)
        params["densify_cos_weight"] = (self.densify_cos_weight,)
        params["densify_feature_grad_accum"] = (
            self.densify_feature_grad_accum,
        )
        params["densify_feature_grad_denom"] = (
            self.densify_feature_grad_denom,
        )

        return params

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        if checkpoint is not None:
            self.densify_grad_norm_accum = checkpoint[
                "densify_grad_norm_accum"
            ][0].detach()
            self.densify_grad_norm_denom = checkpoint[
                "densify_grad_norm_denom"
            ][0].detach()
            cos_accum = checkpoint.get("densify_cos_accum")
            cos_weight = checkpoint.get("densify_cos_weight")
            feature_accum = checkpoint.get("densify_feature_grad_accum")
            feature_denom = checkpoint.get("densify_feature_grad_denom")
            num_gaussians = self.densify_grad_norm_accum.shape[0]
            self.densify_cos_accum = (
                cos_accum[0].detach()
                if cos_accum is not None
                else torch.zeros(
                    (num_gaussians, 1),
                    dtype=torch.float,
                    device=self.model.device,
                )
            )
            self.densify_cos_weight = (
                cos_weight[0].detach()
                if cos_weight is not None
                else torch.zeros(
                    (num_gaussians, 1),
                    dtype=torch.float,
                    device=self.model.device,
                )
            )
            self.densify_feature_grad_accum = (
                feature_accum[0].detach()
                if feature_accum is not None
                else torch.zeros(
                    (num_gaussians, 1),
                    dtype=torch.float,
                    device=self.model.device,
                )
            )
            self.densify_feature_grad_denom = (
                feature_denom[0].detach()
                if feature_denom is not None
                else torch.zeros(
                    (num_gaussians, 1),
                    dtype=torch.int,
                    device=self.model.device,
                )
            )
        else:
            num_gaussians = self.model.num_gaussians
            self.densify_grad_norm_accum = torch.zeros(
                (num_gaussians, 1), dtype=torch.float, device=self.model.device
            )
            self.densify_grad_norm_denom = torch.zeros(
                (num_gaussians, 1), dtype=torch.int, device=self.model.device
            )
            self.densify_cos_accum = torch.zeros(
                (num_gaussians, 1), dtype=torch.float, device=self.model.device
            )
            self.densify_cos_weight = torch.zeros(
                (num_gaussians, 1), dtype=torch.float, device=self.model.device
            )
            self.densify_feature_grad_accum = torch.zeros(
                (num_gaussians, 1), dtype=torch.float, device=self.model.device
            )
            self.densify_feature_grad_denom = torch.zeros(
                (num_gaussians, 1), dtype=torch.int, device=self.model.device
            )

    def _post_backward(
        self,
        step: int,
        scene_extent: float,
        train_dataset,
        batch=None,
        writer=None,
    ) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""

        # Update densification buffer:
        if check_step_condition(
            step, 0, self.conf.strategy.densify.end_iteration, 1
        ):
            with torch.cuda.nvtx.range(f"train_{step}_grad_buffer"):
                self.update_gradient_buffer(
                    sensor_position=batch.T_to_world[0, :3, 3],
                    sensor_forward=batch.T_to_world[0, :3, 2],
                )

        # Clamp density
        if (
            check_step_condition(step, 0, -1, 1)
            and self.conf.model.density_activation == "none"
        ):
            with torch.cuda.nvtx.range(f"train_{step}_clamp_density"):
                self.model.clamp_density()

        return False

    def _post_optimizer_step(
        self,
        step: int,
        scene_extent: float,
        train_dataset,
        batch=None,
        writer=None,
    ) -> bool:
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

        return scene_updated

    @torch.no_grad()
    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(
        self,
        sensor_position: torch.Tensor,
        sensor_forward: torch.Tensor | None = None,
    ) -> None:
        params_grad = self.model.positions.grad
        mask = (params_grad != 0).max(dim=1)[0]
        assert params_grad is not None
        to_camera = self.model.positions[mask] - sensor_position
        distance_to_camera = to_camera.norm(dim=1, keepdim=True)

        grad_increment = (
            torch.norm(
                params_grad[mask] * distance_to_camera, dim=-1, keepdim=True
            )
            / 2
        )
        self.densify_grad_norm_accum[mask] += grad_increment
        self.densify_grad_norm_denom[mask] += 1

        # Field-angle-aware: accumulate the gradient-weighted cos(theta) of
        # each observed Gaussian relative to this camera's optical axis. The
        # weight is the gradient increment just added, so at densify time the
        # ratio (accum / weight) is the gradient-weighted mean cos(theta) --
        # "the field angle this Gaussian's densification demand came from".
        if self.theta_aware and sensor_forward is not None:
            forward = sensor_forward / sensor_forward.norm().clamp_min(1e-8)
            cos_theta = (
                to_camera / distance_to_camera.clamp_min(1e-8)
            ) @ forward
            cos_theta = cos_theta.unsqueeze(1).clamp(-1.0, 1.0)
            self.densify_cos_accum[mask] += grad_increment * cos_theta
            self.densify_cos_weight[mask] += grad_increment

        feature_grad = self.model.features_specular.grad
        if self.feature_grad_aware and feature_grad is not None:
            feature_grad_values = feature_grad.detach()[mask]
            if self.feature_grad_carrier_tail_only:
                sh_dim = sh_degree_to_specular_dim(
                    self.model.get_max_n_features()
                )
                feature_grad_values = feature_grad_values[:, sh_dim:]
            feature_increment = (
                feature_grad_values.flatten(start_dim=1).norm(
                    dim=1, keepdim=True
                )
            )
            self.densify_feature_grad_accum[mask] += feature_increment
            self.densify_feature_grad_denom[mask] += 1

    @torch.no_grad()
    def _theta_threshold_factors(
        self,
    ) -> Optional[torch.Tensor]:
        """Per-Gaussian field-angle multiplier ``clamp(cos(theta), c)^p``.

        Returns ``None`` when field-angle-aware densification is disabled.
        Scaling the grad-threshold and the split/clone size-cap by this factor
        lowers both toward the rim (small ``cos(theta)``), so the periphery
        gets MORE, SMALLER Gaussians. Gaussians never observed in the window
        (zero accumulated weight) get factor 1.0 (unchanged behaviour).
        """
        if not self.theta_aware:
            return None
        weight = self.densify_cos_weight.squeeze()
        mean_cos = torch.where(
            weight > 0,
            (self.densify_cos_accum.squeeze() / weight.clamp_min(1e-12)),
            torch.ones_like(weight),
        )
        mean_cos = mean_cos.clamp(self.theta_min_cos, 1.0)
        return mean_cos

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def _feature_grad_boost_factors(
        self,
    ) -> Float[torch.Tensor, "gaussian"] | None:
        """Return bounded boost from accumulated features_specular gradients."""
        if not self.feature_grad_aware:
            return None
        denom = self.densify_feature_grad_denom.squeeze(-1)
        accum = self.densify_feature_grad_accum.squeeze(-1)
        mean_grad = torch.where(
            denom > 0,
            accum / denom.to(accum.dtype).clamp_min(1),
            torch.zeros_like(accum),
        )
        positive = mean_grad[torch.isfinite(mean_grad) & (mean_grad > 0)]
        if positive.numel() == 0:
            return torch.ones_like(mean_grad)
        reference = torch.quantile(
            positive,
            torch.tensor(
                self.feature_grad_quantile,
                device=positive.device,
                dtype=positive.dtype,
            ),
        ).clamp_min(1e-12)
        normalized = mean_grad.clamp_min(0) / reference
        boost = 1.0 + normalized.pow(self.feature_grad_power)
        return boost.clamp(1.0, self.feature_grad_max_boost)

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, scene_extent):
        assert self.model.optimizer is not None, (
            "Optimizer need to be initialized before splitting and cloning the Gaussians"
        )
        scene_extent = float(scene_extent)
        densify_grad_norm = (
            self.densify_grad_norm_accum / self.densify_grad_norm_denom
        )
        densify_grad_norm[densify_grad_norm.isnan()] = 0.0

        mean_cos = self._theta_threshold_factors()
        feature_boost = self._feature_grad_boost_factors()
        if feature_boost is not None:
            densify_grad_norm = densify_grad_norm * feature_boost.unsqueeze(1)

        densify_scores = densify_grad_norm.squeeze(-1)
        self.log_densify_stats(
            densify_scores, scene_extent, mean_cos, feature_boost
        )
        self.clone_gaussians(densify_scores, scene_extent, mean_cos)
        self.split_gaussians(densify_scores, scene_extent, mean_cos)

        torch.cuda.empty_cache()

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def log_densify_stats(
        self,
        densify_grad_norm: Float[torch.Tensor, "gaussian"],
        scene_extent: float,
        mean_cos: Float[torch.Tensor, "gaussian"] | None = None,
        feature_boost: Float[torch.Tensor, "gaussian"] | None = None,
    ) -> None:
        if not self.conf.strategy.print_stats:
            return
        finite_mask = torch.isfinite(densify_grad_norm)
        finite_values = densify_grad_norm[finite_mask]
        if finite_values.numel() == 0:
            logger.info("Densify grad stats: no finite gradients")
            return
        positive_values = finite_values[finite_values > 0]
        if positive_values.numel() > 200000:
            stride = max(positive_values.numel() // 200000, 1)
            positive_values = positive_values[::stride][:200000]
        max_scale = torch.max(self.model.get_scale(), dim=1).values
        n_points = densify_grad_norm.shape[0]
        clone_thresh, size_threshold = self._theta_thresholds(
            self.clone_grad_threshold, scene_extent, mean_cos, n_points
        )
        split_thresh, _ = self._theta_thresholds(
            self.split_grad_threshold, scene_extent, mean_cos, n_points
        )
        clone_mask = torch.logical_and(
            densify_grad_norm >= clone_thresh,
            max_scale <= size_threshold,
        )
        split_mask = torch.logical_and(
            densify_grad_norm >= split_thresh,
            max_scale > size_threshold,
        )
        # Field-angle breakdown: prove the rim densifies MORE than the centre.
        if mean_cos is not None:
            n = min(mean_cos.shape[0], n_points)
            cos = mean_cos[:n]
            cand = torch.logical_or(clone_mask[:n], split_mask[:n])
            rim = cos <= torch.cos(torch.deg2rad(torch.tensor(60.0)))
            ctr = cos >= torch.cos(torch.deg2rad(torch.tensor(40.0)))
            n_rim = int(rim.sum().item())
            n_ctr = int(ctr.sum().item())
            rim_rate = float(cand[rim].float().mean().item()) if n_rim else 0.0
            ctr_rate = float(cand[ctr].float().mean().item()) if n_ctr else 0.0
            logger.info(
                "Densify theta-aware: "
                f"rim(theta>60) n={n_rim} cand_rate={rim_rate:.3f}, "
                f"center(theta<40) n={n_ctr} cand_rate={ctr_rate:.3f}, "
                f"power={self.theta_power:g}/{self.theta_size_cap_power:g}, "
                f"min_cos={self.theta_min_cos:g}"
            )
        if feature_boost is not None:
            boost_values = feature_boost[torch.isfinite(feature_boost)]
            if boost_values.numel() > 0:
                if boost_values.numel() > 200000:
                    stride = max(boost_values.numel() // 200000, 1)
                    boost_values = boost_values[::stride][:200000]
                boost_quantiles = torch.quantile(
                    boost_values,
                    torch.tensor(
                        [0.5, 0.95, 0.99],
                        device=boost_values.device,
                        dtype=boost_values.dtype,
                    ),
                )
                boosted = int((feature_boost > 1.0).sum().item())
                logger.info(
                    "Densify feature-grad boost: "
                    f"boosted={boosted}/{feature_boost.numel()}, "
                    f"mean={boost_values.mean().item():.3f}, "
                    f"p50={boost_quantiles[0].item():.3f}, "
                    f"p95={boost_quantiles[1].item():.3f}, "
                    f"p99={boost_quantiles[2].item():.3f}, "
                    f"max={boost_values.max().item():.3f}, "
                    f"power={self.feature_grad_power:g}, "
                    f"q={self.feature_grad_quantile:g}, "
                    f"carrier_tail_only={self.feature_grad_carrier_tail_only}"
                )
        n_total = densify_grad_norm.numel()
        n_finite = int(finite_mask.sum().item())
        n_positive = int((finite_values > 0).sum().item())
        n_clone = int(clone_mask.sum().item())
        n_split = int(split_mask.sum().item())
        if positive_values.numel() == 0:
            logger.info(
                "Densify grad stats: "
                f"finite={n_finite}/{n_total}, nonzero=0, "
                f"clone_candidates={n_clone}, split_candidates={n_split}, "
                f"thresholds=({self.clone_grad_threshold:g}, "
                f"{self.split_grad_threshold:g})"
            )
            return
        quantiles = torch.quantile(
            positive_values,
            torch.tensor(
                [0.5, 0.95, 0.99],
                device=positive_values.device,
                dtype=positive_values.dtype,
            ),
        )
        mean_value = positive_values.mean()
        max_value = positive_values.max()
        logger.info(
            "Densify grad stats: "
            f"finite={n_finite}/{n_total}, "
            f"nonzero={n_positive}/{n_total}, "
            f"mean={mean_value.item():.3e}, "
            f"p50={quantiles[0].item():.3e}, "
            f"p95={quantiles[1].item():.3e}, "
            f"p99={quantiles[2].item():.3e}, "
            f"max={max_value.item():.3e}, "
            f"clone_candidates={n_clone}, split_candidates={n_split}, "
            f"thresholds=({self.clone_grad_threshold:g}, "
            f"{self.split_grad_threshold:g})"
        )

    @torch.no_grad()
    def _theta_thresholds(
        self,
        base_threshold: float,
        scene_extent: float,
        mean_cos: Optional[torch.Tensor],
        n_points: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-Gaussian (grad_threshold, size_cap) for ``n_points`` Gaussians.

        With ``mean_cos=None`` (field-angle-aware off) both are the scalar
        upstream values broadcast to every Gaussian. Otherwise the rim
        (small ``cos(theta)``) gets a lower threshold and a smaller size-cap.
        ``mean_cos`` is padded with 1.0 (no scaling) for any trailing
        Gaussians added since it was computed (e.g. clones, in the split pass).
        """
        device = self.model.positions.device
        base_cap = self.relative_size_threshold * scene_extent
        if mean_cos is None:
            thresh = torch.full((n_points,), base_threshold, device=device)
            cap = torch.full((n_points,), base_cap, device=device)
            return thresh, cap
        cos = torch.ones((n_points,), device=device)
        n = min(mean_cos.shape[0], n_points)
        cos[:n] = mean_cos[:n]
        thresh = base_threshold * cos.pow(self.theta_power)
        cap = base_cap * cos.pow(self.theta_size_cap_power)
        return thresh, cap

    @torch.cuda.nvtx.range("split_gaussians")
    def split_gaussians(
        self,
        densify_grad_norm: torch.Tensor,
        scene_extent: float,
        mean_cos: Optional[torch.Tensor] = None,
    ):
        n_init_points = self.model.num_gaussians

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")

        # Here we already have the cloned points in the self.model.positions so only take the points up to size of the initial grad
        padded_grad[: densify_grad_norm.shape[0]] = densify_grad_norm.squeeze()
        split_thresh, size_cap = self._theta_thresholds(
            self.split_grad_threshold, scene_extent, mean_cos, n_init_points
        )
        mask = padded_grad >= split_thresh
        mask = torch.logical_and(
            mask,
            torch.max(self.model.get_scale(), dim=1).values > size_cap,
        )

        stds = self.model.get_scale()[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask]).repeat(
            self.split_n_gaussians, 1, 1
        )
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(
                f"Splitted {n_clone} / {n_before} ({n_clone / n_before * 100:.2f}%) gaussians"
            )

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            repeats = [self.split_n_gaussians] + [1] * (param.dim() - 1)
            if name == "positions":
                p_split = param[mask].repeat(repeats) + offsets  # [2N, 3]
            elif name == "scale":
                p_split = self.model.scale_activation_inv(
                    self.model.scale_activation(param[mask].repeat(repeats))
                    / (0.8 * self.split_n_gaussians)
                )
            else:
                p_split = param[mask].repeat(repeats)

            p_new = torch.nn.Parameter(
                torch.cat([param[~mask], p_split]),
                requires_grad=param.requires_grad,
            )

            return p_new

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            v_split = torch.zeros(
                (self.split_n_gaussians * int(mask.sum()), *v.shape[1:]),
                device=v.device,
            )
            return torch.cat([v[~mask], v_split])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

    @torch.cuda.nvtx.range("clone_gaussians")
    def clone_gaussians(
        self,
        densify_grad_norm: torch.Tensor,
        scene_extent: float,
        mean_cos: Optional[torch.Tensor] = None,
    ):
        assert densify_grad_norm is not None, (
            "Positional gradients must be available in order to clone the Gaussians"
        )
        densify_grad_norm = densify_grad_norm.squeeze()
        clone_thresh, size_cap = self._theta_thresholds(
            self.clone_grad_threshold,
            scene_extent,
            mean_cos,
            densify_grad_norm.shape[0],
        )
        # Extract points that satisfy the gradient condition
        mask = densify_grad_norm >= clone_thresh

        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(
            mask,
            torch.max(self.model.get_scale(), dim=1).values <= size_cap,
        )

        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(
                f"Cloned {n_clone} / {n_before} ({n_clone / n_before * 100:.2f}%) gaussians"
            )

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            param_new = torch.cat([param, param[mask]])
            return torch.nn.Parameter(
                param_new, requires_grad=param.requires_grad
            )

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    v,
                    torch.zeros(
                        (int(mask.sum()), *v.shape[1:]), device=v.device
                    ),
                ]
            )

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

    def prune_gaussians_weight(self):
        # Prune the Gaussians based on their weight
        mask = (
            self.model.rolling_weight_contrib[:, 0]
            >= self.conf.strategy.prune_weight.weight_threshold
        )
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(
                f"Weight-pruned {n_prune} / {n_before} ({n_prune / n_before * 100:.2f}%) gaussians"
            )

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(
                param[mask], requires_grad=param.requires_grad
            )

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def prune_gaussians_scale(self, dataset):
        cam_normals = torch.from_numpy(dataset.poses[:, :3, 2]).to(
            self.model.device
        )
        similarities = torch.matmul(self.model.positions, cam_normals.T)
        cam_dists = similarities.min(dim=1)[0].clamp(min=1e-8)
        ratio = (
            self.model.get_scale().min(dim=1)[0]
            / cam_dists
            * dataset.intrinsic[0].max()
        )

        # Prune the Gaussians based on their weight
        mask = ratio >= self.conf.strategy.prune_scale.threshold
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(
                f"Scale-pruned {n_prune} / {n_before} ({n_prune / n_before * 100:.2f}%) gaussians"
            )

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(
                param[mask], requires_grad=param.requires_grad
            )

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def prune_gaussians_opacity(self):
        # Prune the Gaussians based on their opacity
        mask = (
            self.model.get_density().squeeze() >= self.prune_density_threshold
        )

        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(
                f"Density-pruned {n_prune} / {n_before} ({n_prune / n_before * 100:.2f}%) gaussians"
            )

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(
                param[mask], requires_grad=param.requires_grad
            )

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    def reset_densification_buffers(self) -> None:
        n = self.model.get_positions().shape[0]
        self.densify_grad_norm_accum = torch.zeros(
            (n, 1),
            device=self.model.device,
            dtype=self.densify_grad_norm_accum.dtype,
        )

        self.densify_grad_norm_denom = torch.zeros(
            (n, 1),
            device=self.model.device,
            dtype=self.densify_grad_norm_denom.dtype,
        )
        self.densify_cos_accum = torch.zeros(
            (n, 1), device=self.model.device, dtype=torch.float
        )
        self.densify_cos_weight = torch.zeros(
            (n, 1), device=self.model.device, dtype=torch.float
        )
        self.densify_feature_grad_accum = torch.zeros(
            (n, 1),
            device=self.model.device,
            dtype=self.densify_feature_grad_accum.dtype,
        )
        self.densify_feature_grad_denom = torch.zeros(
            (n, 1),
            device=self.model.device,
            dtype=self.densify_feature_grad_denom.dtype,
        )

    def prune_densification_buffers(self, valid_mask: torch.Tensor) -> None:
        # Update non-optimizable buffers
        self.densify_grad_norm_accum = self.densify_grad_norm_accum[valid_mask]
        self.densify_grad_norm_denom = self.densify_grad_norm_denom[valid_mask]
        if self.densify_cos_accum.shape[0] == valid_mask.shape[0]:
            self.densify_cos_accum = self.densify_cos_accum[valid_mask]
            self.densify_cos_weight = self.densify_cos_weight[valid_mask]
        if self.densify_feature_grad_accum.shape[0] == valid_mask.shape[0]:
            self.densify_feature_grad_accum = (
                self.densify_feature_grad_accum[valid_mask]
            )
            self.densify_feature_grad_denom = (
                self.densify_feature_grad_denom[valid_mask]
            )

    def decay_density(self):
        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name == "density", (
                "wrong paramaeter passed to update_param_fn"
            )

            decayed_densities = self.model.density_activation_inv(
                self.model.get_density()
                * self.conf.strategy.density_decay.gamma
            )

            return torch.nn.Parameter(
                decayed_densities, requires_grad=param.requires_grad
            )

        self._update_param_with_optimizer(
            update_param_fn, None, names=["density"]
        )

    def reset_density(self):
        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            assert name == "density", (
                "wrong paramaeter passed to update_param_fn"
            )
            densities = torch.clamp(
                param,
                max=self.model.density_activation_inv(
                    torch.tensor(self.new_max_density)
                ).item(),
            )
            return torch.nn.Parameter(densities)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(v)

        # update the parameters and the state in the optimizers
        self._update_param_with_optimizer(
            update_param_fn, update_optimizer_fn, names=["density"]
        )
