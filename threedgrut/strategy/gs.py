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

import math
from typing import Optional

import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.base import BaseStrategy
from threedgrut.strategy.scale_shape_split import (
    ScaleShapeThresholds,
    deterministic_split_children,
    scale_shape_split_mask,
    transmittance_preserving_child_opacity,
)
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import (
    check_step_condition,
    quaternion_to_so3,
    sh_degree_to_specular_dim,
)

_COVARIANCE_INVALID_REASON_NAMES = (
    "nonfinite",
    "nonpositive_diagonal",
    "nonpositive_determinant",
    "nonpositive_opacity",
)
_COVARIANCE_INVALID_EXAMPLES_PER_REASON = 8


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

        # Field-angle-aware (solid-angle-weighted) densification config. When
        # enabled, the rim (high field angle theta) densifies more and keeps
        # smaller Gaussians, compensating the fisheye projection Jacobian that
        # screen-space density control is blind to. Defaults keep the upstream
        # scalar-threshold behaviour byte-identical.
        theta_cfg = getattr(self.conf.strategy.densify, "theta_aware", None)
        self.theta_aware = bool(getattr(theta_cfg, "enabled", False))
        self.theta_power = float(getattr(theta_cfg, "theta_power", 1.5))
        self.theta_size_cap_power = float(getattr(theta_cfg, "size_cap_power", 1.0))
        self.theta_min_cos = float(getattr(theta_cfg, "min_cos", 0.1))
        feature_cfg = self.conf.strategy.densify.get("feature_grad", {})
        self.feature_grad_aware = bool(feature_cfg.get("enabled", False))
        self.feature_grad_power = float(feature_cfg.get("power", 0.5))
        self.feature_grad_quantile = float(feature_cfg.get("quantile", 0.95))
        self.feature_grad_max_boost = float(feature_cfg.get("max_boost", 4.0))
        self.feature_grad_carrier_tail_only = bool(feature_cfg.get("carrier_tail_only", False))
        projected_extent_cfg = self.conf.strategy.densify.get("projected_extent_split", {})
        self.projected_extent_split_enabled = bool(projected_extent_cfg.get("enabled", False))
        self.projected_extent_split_max_px = float(projected_extent_cfg.get("max_px", 8.0))
        covariance_gradient_cfg = self.conf.strategy.densify.get("covariance_gradient_split", {})
        self.covariance_gradient_split_enabled = bool(covariance_gradient_cfg.get("enabled", False))
        self.covariance_gradient_split_radius_px = float(covariance_gradient_cfg.get("radius_px", 8.0))
        min_large_observations = covariance_gradient_cfg.get("min_large_observations", 6)
        self.covariance_gradient_split_min_large_observations = int(min_large_observations)
        self.covariance_gradient_split_max_reroute_fraction = float(
            covariance_gradient_cfg.get(
                "max_reroute_fraction",
                0.22385751031338305,
            )
        )
        max_invalid_fraction = covariance_gradient_cfg.get(
            "max_invalid_fraction",
            1e-5,
        )
        self.covariance_gradient_split_max_invalid_fraction = float(max_invalid_fraction)
        scale_shape_split_cfg = self.conf.strategy.densify.get("scale_shape_split", {})
        self.scale_shape_split_enabled = bool(scale_shape_split_cfg.get("enabled", False))
        self.scale_shape_anisotropy_threshold = float(scale_shape_split_cfg.get("anisotropy_threshold", 8.0))
        self.scale_shape_min_largest_scale = float(scale_shape_split_cfg.get("min_largest_scale", 0.01))
        min_observations = scale_shape_split_cfg.get("min_observations", 32)
        self.scale_shape_min_observations = int(min_observations)
        if self.projected_extent_split_enabled and self.covariance_gradient_split_enabled:
            raise ValueError(
                "strategy.densify.projected_extent_split and "
                "strategy.densify.covariance_gradient_split are mutually "
                "exclusive experiment arms."
            )
        if self.projected_extent_split_enabled:
            render_method = self.conf.get("render", {}).get("method")
            if render_method != "3dgut":
                raise ValueError(
                    "strategy.densify.projected_extent_split requires " f"render.method=3dgut, got {render_method!r}."
                )
            if not math.isfinite(self.projected_extent_split_max_px) or self.projected_extent_split_max_px <= 0.0:
                raise ValueError(
                    "strategy.densify.projected_extent_split.max_px must "
                    "be finite and positive, got "
                    f"{self.projected_extent_split_max_px}."
                )
        if self.covariance_gradient_split_enabled:
            render_method = self.conf.get("render", {}).get("method")
            if render_method != "3dgut":
                raise ValueError(
                    "strategy.densify.covariance_gradient_split requires "
                    f"render.method=3dgut, got {render_method!r}."
                )
            if (
                not math.isfinite(self.covariance_gradient_split_radius_px)
                or self.covariance_gradient_split_radius_px <= 0.0
            ):
                raise ValueError(
                    "strategy.densify.covariance_gradient_split.radius_px "
                    "must be finite and positive, got "
                    f"{self.covariance_gradient_split_radius_px}."
                )
            if (
                isinstance(min_large_observations, bool)
                or self.covariance_gradient_split_min_large_observations != min_large_observations
                or self.covariance_gradient_split_min_large_observations < 1
            ):
                raise ValueError(
                    "strategy.densify.covariance_gradient_split."
                    "min_large_observations must be a positive integer, got "
                    f"{min_large_observations!r}."
                )
            if (
                not math.isfinite(self.covariance_gradient_split_max_reroute_fraction)
                or not 0.0 < self.covariance_gradient_split_max_reroute_fraction < 1.0
            ):
                raise ValueError(
                    "strategy.densify.covariance_gradient_split."
                    "max_reroute_fraction must be finite and in (0, 1), got "
                    f"{self.covariance_gradient_split_max_reroute_fraction}."
                )
            if (
                isinstance(max_invalid_fraction, bool)
                or not math.isfinite(self.covariance_gradient_split_max_invalid_fraction)
                or not 0.0 <= self.covariance_gradient_split_max_invalid_fraction < 1.0
            ):
                raise ValueError(
                    "strategy.densify.covariance_gradient_split."
                    "max_invalid_fraction must be finite and in [0, 1), got "
                    f"{max_invalid_fraction!r}."
                )
        if self.scale_shape_split_enabled:
            incompatible_arms = self.projected_extent_split_enabled or self.covariance_gradient_split_enabled
            if incompatible_arms:
                raise ValueError(
                    "strategy.densify.scale_shape_split is mutually "
                    "exclusive with projected-extent and covariance-gradient "
                    "split arms."
                )
            render_method = self.conf.get("render", {}).get("method")
            if render_method != "3dgut":
                raise ValueError(
                    "strategy.densify.scale_shape_split requires " f"render.method=3dgut, got {render_method!r}."
                )
            if self.conf.model.density_activation != "sigmoid":
                raise ValueError("strategy.densify.scale_shape_split requires " "model.density_activation=sigmoid.")
            if self.conf.model.scale_activation != "exp":
                raise ValueError("strategy.densify.scale_shape_split requires " "model.scale_activation=exp.")
            if self.split_n_gaussians != 2:
                raise ValueError("strategy.densify.scale_shape_split requires " "exactly two split children.")
            if not math.isfinite(self.scale_shape_anisotropy_threshold) or self.scale_shape_anisotropy_threshold <= 1.0:
                raise ValueError("scale_shape_split.anisotropy_threshold must be " "finite and greater than one.")
            if not math.isfinite(self.scale_shape_min_largest_scale) or self.scale_shape_min_largest_scale <= 0.0:
                raise ValueError("scale_shape_split.min_largest_scale must be " "finite and positive.")
            if (
                isinstance(min_observations, bool)
                or self.scale_shape_min_observations != min_observations
                or self.scale_shape_min_observations < 1
            ):
                raise ValueError(
                    "scale_shape_split.min_observations must be a " f"positive integer, got {min_observations!r}."
                )
        if self.feature_grad_power <= 0:
            msg = "strategy.densify.feature_grad.power must be positive; " f"got {self.feature_grad_power}."
            raise ValueError(msg)
        if not 0 < self.feature_grad_quantile <= 1:
            msg = "strategy.densify.feature_grad.quantile must be in (0, 1]; " f"got {self.feature_grad_quantile}."
            raise ValueError(msg)
        if self.feature_grad_max_boost < 1:
            msg = "strategy.densify.feature_grad.max_boost must be >= 1; " f"got {self.feature_grad_max_boost}."
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
        self.densify_projected_extent_max = torch.empty([0])
        self.densify_scale_shape_observation_count = torch.empty([0, 1], dtype=torch.int)
        self.densify_covariance_gradient_mass = torch.empty([0, 1])
        self.densify_covariance_large_observations = torch.empty([0, 1])
        self.densify_covariance_packet_observations = torch.empty([0], dtype=torch.int64)
        self.densify_covariance_invalid_packet_observations = torch.empty([0], dtype=torch.int64)
        self.densify_covariance_invalid_reason_counts = torch.empty([0], dtype=torch.int64)
        self.densify_covariance_invalid_example_indices = torch.empty([0, 0], dtype=torch.int64)
        self.densify_covariance_invalid_example_values = torch.empty([0, 0, 0], dtype=torch.float64)

    def get_strategy_parameters(self) -> dict:
        params = {}

        params["densify_grad_norm_accum"] = (self.densify_grad_norm_accum,)
        params["densify_grad_norm_denom"] = (self.densify_grad_norm_denom,)
        params["densify_cos_accum"] = (self.densify_cos_accum,)
        params["densify_cos_weight"] = (self.densify_cos_weight,)
        params["densify_feature_grad_accum"] = (self.densify_feature_grad_accum,)
        params["densify_feature_grad_denom"] = (self.densify_feature_grad_denom,)
        if self.projected_extent_split_enabled:
            params["densify_projected_extent_max"] = (self.densify_projected_extent_max,)
        if self.scale_shape_split_enabled:
            params["densify_scale_shape_observation_count"] = (self.densify_scale_shape_observation_count,)
        if self.covariance_gradient_split_enabled:
            params["densify_covariance_gradient_mass"] = (self.densify_covariance_gradient_mass,)
            params["densify_covariance_large_observations"] = (self.densify_covariance_large_observations,)
            params["densify_covariance_packet_observations"] = (self.densify_covariance_packet_observations,)
            params["densify_covariance_invalid_packet_observations"] = (
                self.densify_covariance_invalid_packet_observations,
            )
            params["densify_covariance_invalid_reason_counts"] = (self.densify_covariance_invalid_reason_counts,)
            params["densify_covariance_invalid_example_indices"] = (self.densify_covariance_invalid_example_indices,)
            params["densify_covariance_invalid_example_values"] = (self.densify_covariance_invalid_example_values,)

        return params

    def init_densification_buffer(self, checkpoint: Optional[dict] = None):
        if checkpoint is not None:
            self.densify_grad_norm_accum = checkpoint["densify_grad_norm_accum"][0].detach()
            self.densify_grad_norm_denom = checkpoint["densify_grad_norm_denom"][0].detach()
            cos_accum = checkpoint.get("densify_cos_accum")
            cos_weight = checkpoint.get("densify_cos_weight")
            feature_accum = checkpoint.get("densify_feature_grad_accum")
            feature_denom = checkpoint.get("densify_feature_grad_denom")
            projected_extent_max = checkpoint.get("densify_projected_extent_max")
            scale_shape_observation_count = checkpoint.get("densify_scale_shape_observation_count")
            covariance_gradient_mass = checkpoint.get("densify_covariance_gradient_mass")
            covariance_large_observations = checkpoint.get("densify_covariance_large_observations")
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
            self.densify_projected_extent_max = self._restored_extent_buffer(
                projected_extent_max,
                num_gaussians=num_gaussians,
            )
            self.densify_scale_shape_observation_count = self._restored_scale_shape_observation_count(
                checkpoint=checkpoint,
                checkpoint_value=scale_shape_observation_count,
                num_gaussians=num_gaussians,
            )
            (
                self.densify_covariance_gradient_mass,
                self.densify_covariance_large_observations,
            ) = self._restored_covariance_gradient_buffers(
                checkpoint=checkpoint,
                gradient_mass=covariance_gradient_mass,
                large_observations=covariance_large_observations,
                num_gaussians=num_gaussians,
            )
            (
                self.densify_covariance_packet_observations,
                self.densify_covariance_invalid_packet_observations,
                self.densify_covariance_invalid_reason_counts,
                self.densify_covariance_invalid_example_indices,
                self.densify_covariance_invalid_example_values,
            ) = self._restored_covariance_packet_telemetry(checkpoint)
        else:
            num_gaussians = self.model.num_gaussians
            self.densify_grad_norm_accum = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.densify_grad_norm_denom = torch.zeros((num_gaussians, 1), dtype=torch.int, device=self.model.device)
            self.densify_cos_accum = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.densify_cos_weight = torch.zeros((num_gaussians, 1), dtype=torch.float, device=self.model.device)
            self.densify_feature_grad_accum = torch.zeros(
                (num_gaussians, 1), dtype=torch.float, device=self.model.device
            )
            self.densify_feature_grad_denom = torch.zeros((num_gaussians, 1), dtype=torch.int, device=self.model.device)
            self.densify_projected_extent_max = torch.zeros(
                num_gaussians,
                dtype=torch.float,
                device=self.model.device,
            )
            self.densify_scale_shape_observation_count = torch.zeros(
                (num_gaussians, 1),
                dtype=torch.int,
                device=self.model.device,
            )
            if not self.scale_shape_split_enabled:
                self.densify_scale_shape_observation_count = self.densify_scale_shape_observation_count[:0]
            self.densify_covariance_gradient_mass = torch.zeros(
                (num_gaussians, 1),
                dtype=torch.float,
                device=self.model.device,
            )
            self.densify_covariance_large_observations = torch.zeros(
                (num_gaussians, 1),
                dtype=torch.int,
                device=self.model.device,
            )
            (
                self.densify_covariance_packet_observations,
                self.densify_covariance_invalid_packet_observations,
                self.densify_covariance_invalid_reason_counts,
                self.densify_covariance_invalid_example_indices,
                self.densify_covariance_invalid_example_values,
            ) = self._empty_covariance_packet_telemetry()

    def _restored_extent_buffer(
        self,
        checkpoint_value: object,
        *,
        num_gaussians: int,
    ) -> torch.Tensor:
        if not self.projected_extent_split_enabled or checkpoint_value is None:
            return torch.zeros(
                num_gaussians,
                dtype=torch.float,
                device=self.model.device,
            )
        if not isinstance(checkpoint_value, (list, tuple)):
            raise ValueError("Checkpoint densify_projected_extent_max must be a tuple.")
        if len(checkpoint_value) != 1 or not torch.is_tensor(checkpoint_value[0]):
            raise ValueError("Checkpoint densify_projected_extent_max must contain one " "tensor.")
        restored = checkpoint_value[0].detach()
        if restored.shape != (num_gaussians,):
            raise ValueError(
                "Checkpoint densify_projected_extent_max must have shape "
                f"({num_gaussians},), got {tuple(restored.shape)}."
            )
        model_device = self.model.positions.device
        if restored.device != model_device:
            raise ValueError(
                "Checkpoint densify_projected_extent_max must share the "
                f"model device {model_device}, got {restored.device}."
            )
        if not restored.is_floating_point():
            raise ValueError("Checkpoint densify_projected_extent_max must be floating " "point.")
        return restored

    def _restored_scale_shape_observation_count(
        self,
        *,
        checkpoint: dict,
        checkpoint_value: object,
        num_gaussians: int,
    ) -> torch.Tensor:
        """Restore the enabled arm's exact within-window visibility status."""
        zeros = torch.zeros(
            (num_gaussians, 1),
            dtype=torch.int,
            device=self.model.device,
        )
        if not self.scale_shape_split_enabled:
            return zeros[:0]
        if checkpoint_value is None:
            if self._checkpoint_is_mid_covariance_gradient_window(checkpoint):
                raise ValueError(
                    "An enabled scale-shape resume checkpoint is " "missing its mid-window observation-count buffer."
                )
            return zeros
        restored = self._validated_covariance_gradient_checkpoint_tensor(
            checkpoint_value,
            name="densify_scale_shape_observation_count",
            num_gaussians=num_gaussians,
            floating_point=False,
        )
        if bool((restored < 0).any()) or bool((restored > self.scale_shape_min_observations).any()):
            raise ValueError("Checkpoint densify_scale_shape_observation_count must be in " "[0, min_observations].")
        return restored

    def _checkpoint_is_mid_covariance_gradient_window(
        self,
        checkpoint: dict,
    ) -> bool:
        """Whether a resumed arm still needs accumulated observations."""
        global_step = checkpoint.get("global_step")
        if global_step is None:
            raise ValueError(
                "A covariance-gradient resume checkpoint must contain "
                "global_step so buffer lifecycle can be verified."
            )
        global_step = int(global_step)
        if global_step <= 0:
            return False
        end_iteration = int(self.conf.strategy.densify.end_iteration)
        if end_iteration >= 0 and global_step >= end_iteration:
            return False
        last_completed_step = global_step - 1
        if not self._covariance_gradient_window_is_open(last_completed_step):
            return False
        start_iteration = int(self.conf.strategy.densify.start_iteration)
        frequency = int(self.conf.strategy.densify.frequency)
        reset_at_last_step = check_step_condition(
            last_completed_step,
            start_iteration,
            end_iteration,
            frequency,
        )
        return not reset_at_last_step

    def _covariance_gradient_window_is_open(self, step: int) -> bool:
        """Whether ``step`` can still feed a reachable densification event."""
        end_iteration = int(self.conf.strategy.densify.end_iteration)
        if end_iteration < 0:
            return True
        frequency = int(self.conf.strategy.densify.frequency)
        start_iteration = int(self.conf.strategy.densify.start_iteration)
        last_densification_step = ((end_iteration - 1) // frequency) * frequency
        return last_densification_step > start_iteration and step <= last_densification_step

    def _restored_covariance_gradient_buffers(
        self,
        *,
        checkpoint: dict,
        gradient_mass: object,
        large_observations: object,
        num_gaussians: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros_mass = torch.zeros(
            (num_gaussians, 1),
            dtype=torch.float,
            device=self.model.device,
        )
        zeros_observations = torch.zeros(
            (num_gaussians, 1),
            dtype=torch.int,
            device=self.model.device,
        )
        if not self.covariance_gradient_split_enabled:
            return zeros_mass, zeros_observations
        buffers_missing = gradient_mass is None and large_observations is None
        if buffers_missing:
            if self._checkpoint_is_mid_covariance_gradient_window(checkpoint):
                raise ValueError(
                    "An enabled covariance-gradient resume checkpoint is "
                    "missing mid-window accumulation buffers. Resume from a "
                    "checkpoint that contains both buffers or from an exact "
                    "post-densification reset boundary."
                )
            return zeros_mass, zeros_observations
        if gradient_mass is None or large_observations is None:
            raise ValueError(
                "Covariance-gradient resume checkpoints must contain both "
                "gradient-mass and large-observation buffers."
            )
        mass = self._validated_covariance_gradient_checkpoint_tensor(
            gradient_mass,
            name="densify_covariance_gradient_mass",
            num_gaussians=num_gaussians,
            floating_point=True,
        )
        observations = self._validated_covariance_gradient_checkpoint_tensor(
            large_observations,
            name="densify_covariance_large_observations",
            num_gaussians=num_gaussians,
            floating_point=False,
        )
        if not torch.isfinite(mass).all() or (mass < 0.0).any():
            raise ValueError("Checkpoint densify_covariance_gradient_mass must be finite " "and non-negative.")
        if (observations < 0).any():
            raise ValueError("Checkpoint densify_covariance_large_observations must be " "non-negative.")
        return mass, observations

    def _empty_covariance_packet_telemetry(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        device = self.model.positions.device
        reason_count = len(_COVARIANCE_INVALID_REASON_NAMES)
        example_count = _COVARIANCE_INVALID_EXAMPLES_PER_REASON
        return (
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(reason_count, dtype=torch.int64, device=device),
            torch.full(
                (reason_count, example_count),
                -1,
                dtype=torch.int64,
                device=device,
            ),
            torch.full(
                (reason_count, example_count, 4),
                float("nan"),
                dtype=torch.float64,
                device=device,
            ),
        )

    def _restored_covariance_packet_telemetry(
        self,
        checkpoint: dict,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        empty = self._empty_covariance_packet_telemetry()
        if not self.covariance_gradient_split_enabled:
            return empty
        names_and_shapes = (
            ("densify_covariance_packet_observations", (1,), False),
            (
                "densify_covariance_invalid_packet_observations",
                (1,),
                False,
            ),
            (
                "densify_covariance_invalid_reason_counts",
                (len(_COVARIANCE_INVALID_REASON_NAMES),),
                False,
            ),
            (
                "densify_covariance_invalid_example_indices",
                (
                    len(_COVARIANCE_INVALID_REASON_NAMES),
                    _COVARIANCE_INVALID_EXAMPLES_PER_REASON,
                ),
                False,
            ),
            (
                "densify_covariance_invalid_example_values",
                (
                    len(_COVARIANCE_INVALID_REASON_NAMES),
                    _COVARIANCE_INVALID_EXAMPLES_PER_REASON,
                    4,
                ),
                True,
            ),
        )
        checkpoint_values = tuple(checkpoint.get(name) for name, _, _ in names_and_shapes)
        if all(value is None for value in checkpoint_values):
            if self._checkpoint_is_mid_covariance_gradient_window(checkpoint):
                raise ValueError(
                    "An enabled covariance-gradient resume checkpoint is "
                    "missing mid-window invalid-packet telemetry. Resume "
                    "from a v2 checkpoint or an exact post-densification "
                    "reset boundary."
                )
            return empty
        if any(value is None for value in checkpoint_values):
            raise ValueError(
                "Covariance-gradient resume checkpoints must contain the " "complete invalid-packet telemetry state."
            )
        restored = tuple(
            self._validated_covariance_telemetry_checkpoint_tensor(
                checkpoint_value,
                name=name,
                expected_shape=shape,
                floating_point=floating_point,
            )
            for (name, shape, floating_point), checkpoint_value in zip(
                names_and_shapes,
                checkpoint_values,
            )
        )
        (
            packet_observations,
            invalid_packet_observations,
            reason_counts,
            example_indices,
            example_values,
        ) = restored
        if packet_observations.dtype != torch.int64:
            raise ValueError("Checkpoint densify_covariance_packet_observations must use " "torch.int64.")
        if invalid_packet_observations.dtype != torch.int64:
            raise ValueError("Checkpoint " "densify_covariance_invalid_packet_observations must use " "torch.int64.")
        if reason_counts.dtype != torch.int64:
            raise ValueError("Checkpoint densify_covariance_invalid_reason_counts must use " "torch.int64.")
        if example_indices.dtype != torch.int64:
            raise ValueError("Checkpoint densify_covariance_invalid_example_indices must " "use torch.int64.")
        if example_values.dtype != torch.float64:
            raise ValueError("Checkpoint densify_covariance_invalid_example_values must " "use torch.float64.")
        if (packet_observations < 0).any() or (invalid_packet_observations < 0).any() or (reason_counts < 0).any():
            raise ValueError("Checkpoint covariance packet telemetry counts must be " "non-negative.")
        if invalid_packet_observations[0] > packet_observations[0]:
            raise ValueError(
                "Checkpoint invalid covariance packet count cannot exceed " "the on-screen packet denominator."
            )
        if reason_counts.sum() != invalid_packet_observations[0]:
            raise ValueError(
                "Checkpoint ordered invalid covariance predicate counts must " "sum to the invalid-packet union count."
            )
        stored_per_reason = reason_counts.clamp_max(_COVARIANCE_INVALID_EXAMPLES_PER_REASON)
        example_slots = torch.arange(
            _COVARIANCE_INVALID_EXAMPLES_PER_REASON,
            dtype=torch.int64,
            device=example_indices.device,
        ).unsqueeze(0)
        populated = example_slots < stored_per_reason.unsqueeze(1)
        if (example_indices[populated] < 0).any():
            raise ValueError(
                "Checkpoint invalid covariance examples must contain an " "index for every populated slot."
            )
        if (example_indices[~populated] != -1).any():
            raise ValueError("Checkpoint invalid covariance example indices must use -1 " "for empty slots.")
        if not torch.isnan(example_values[~populated]).all():
            raise ValueError("Checkpoint invalid covariance example values must use NaN " "for empty slots.")
        return (
            packet_observations,
            invalid_packet_observations,
            reason_counts,
            example_indices,
            example_values,
        )

    def _validated_covariance_telemetry_checkpoint_tensor(
        self,
        checkpoint_value: object,
        *,
        name: str,
        expected_shape: tuple[int, ...],
        floating_point: bool,
    ) -> torch.Tensor:
        if not isinstance(checkpoint_value, (list, tuple)):
            raise ValueError(f"Checkpoint {name} must be a tuple.")
        if len(checkpoint_value) != 1 or not torch.is_tensor(checkpoint_value[0]):
            raise ValueError(f"Checkpoint {name} must contain exactly one tensor.")
        restored = checkpoint_value[0].detach()
        if restored.shape != expected_shape:
            raise ValueError(f"Checkpoint {name} must have shape {expected_shape}, got " f"{tuple(restored.shape)}.")
        model_device = self.model.positions.device
        if restored.device != model_device:
            raise ValueError(
                f"Checkpoint {name} must share the model device " f"{model_device}, got {restored.device}."
            )
        if restored.is_floating_point() != floating_point:
            expected_dtype = "floating point" if floating_point else "an integer dtype"
            raise ValueError(f"Checkpoint {name} must use {expected_dtype}.")
        return restored

    def _validated_covariance_gradient_checkpoint_tensor(
        self,
        checkpoint_value: object,
        *,
        name: str,
        num_gaussians: int,
        floating_point: bool,
    ) -> torch.Tensor:
        if not isinstance(checkpoint_value, (list, tuple)):
            raise ValueError(f"Checkpoint {name} must be a tuple.")
        if len(checkpoint_value) != 1 or not torch.is_tensor(checkpoint_value[0]):
            raise ValueError(f"Checkpoint {name} must contain exactly one tensor.")
        restored = checkpoint_value[0].detach()
        expected_shape = (num_gaussians, 1)
        if restored.shape != expected_shape:
            raise ValueError(f"Checkpoint {name} must have shape {expected_shape}, got " f"{tuple(restored.shape)}.")
        model_device = self.model.positions.device
        if restored.device != model_device:
            raise ValueError(
                f"Checkpoint {name} must share the model device " f"{model_device}, got {restored.device}."
            )
        if restored.is_floating_point() != floating_point:
            expected_dtype = "floating point" if floating_point else "an integer dtype"
            raise ValueError(f"Checkpoint {name} must use {expected_dtype}.")
        return restored

    def _post_backward(
        self,
        step: int,
        scene_extent: float,
        train_dataset,
        batch=None,
        writer=None,
        outputs: dict[str, object] | None = None,
    ) -> bool:
        """Callback function to be executed after the `loss.backward()` call."""

        # Update densification buffer:
        if check_step_condition(step, 0, self.conf.strategy.densify.end_iteration, 1):
            with torch.cuda.nvtx.range(f"train_{step}_grad_buffer"):
                if self.scale_shape_split_enabled:
                    self.update_scale_shape_observation_count(outputs)
                if self.projected_extent_split_enabled:
                    self.update_projected_extent_buffer(outputs)
                if self.covariance_gradient_split_enabled and self._covariance_gradient_window_is_open(step):
                    self.update_covariance_gradient_buffer(
                        outputs,
                        sensor_position=batch.T_to_world[0, :3, 3],
                    )
                self.update_gradient_buffer(
                    sensor_position=batch.T_to_world[0, :3, 3],
                    sensor_forward=batch.T_to_world[0, :3, 2],
                )

        # Clamp density
        if check_step_condition(step, 0, -1, 1) and self.conf.model.density_activation == "none":
            with torch.cuda.nvtx.range(f"train_{step}_clamp_density"):
                self.model.clamp_density()

        return False

    @torch.no_grad()
    def update_scale_shape_observation_count(
        self,
        outputs: dict[str, object] | None,
    ) -> None:
        """Accumulate saturated renderer-observation counts for the window.

        The renderer's valid-projection gate can increment a Gaussian once per
        training iteration. If the sampler revisits an image, that image
        contributes another observation; this does not count distinct frames
        or require final unoccluded radiance contribution.
        """
        if outputs is None:
            raise RuntimeError("scale-shape splitting requires renderer outputs.")
        visibility = outputs.get("mog_visibility")
        if not torch.is_tensor(visibility):
            raise RuntimeError("scale-shape splitting requires " "outputs['mog_visibility'].")
        num_gaussians = self.model.num_gaussians
        if visibility.numel() != num_gaussians:
            raise ValueError(
                "mog_visibility must contain one value per Gaussian; got "
                f"{tuple(visibility.shape)} for {num_gaussians} Gaussians."
            )
        if visibility.is_complex() or (visibility.is_floating_point() and not bool(torch.isfinite(visibility).all())):
            raise ValueError("mog_visibility must contain finite real values.")
        if visibility.device != self.model.positions.device:
            raise ValueError("mog_visibility must share the Gaussian model device.")
        if self.densify_scale_shape_observation_count.shape != (
            num_gaussians,
            1,
        ):
            raise RuntimeError(
                "scale-shape observation-count buffer is not aligned with the "
                f"model: {tuple(self.densify_scale_shape_observation_count.shape)} "
                f"versus ({num_gaussians}, 1)."
            )
        visible = visibility.detach().reshape(-1).to(dtype=torch.bool)
        counts = self.densify_scale_shape_observation_count[:, 0]
        counts[visible] = torch.clamp(
            counts[visible] + 1,
            max=self.scale_shape_min_observations,
        )

    @torch.no_grad()
    def update_projected_extent_buffer(
        self,
        outputs: dict[str, object] | None,
    ) -> None:
        """Accumulate each on-screen Gaussian's largest tight half-extent."""
        if outputs is None:
            raise RuntimeError("Projected-extent splitting requires renderer outputs.")
        projected_extent = outputs.get("mog_projected_extent")
        tile_count = outputs.get("mog_tiles_count")
        if not torch.is_tensor(projected_extent):
            raise RuntimeError("Projected-extent splitting requires " "outputs['mog_projected_extent'].")
        if not torch.is_tensor(tile_count):
            raise RuntimeError("Projected-extent splitting requires " "outputs['mog_tiles_count'].")
        num_gaussians = self.model.num_gaussians
        if projected_extent.shape != (num_gaussians, 2):
            raise ValueError(
                "mog_projected_extent must have shape "
                f"({num_gaussians}, 2), got "
                f"{tuple(projected_extent.shape)}."
            )
        if tile_count.numel() != num_gaussians:
            raise ValueError(
                "mog_tiles_count must contain one value per Gaussian; got "
                f"{tuple(tile_count.shape)} for {num_gaussians} Gaussians."
            )
        if self.densify_projected_extent_max.shape != (num_gaussians,):
            raise RuntimeError(
                "Projected-extent densification buffer is not aligned with "
                f"the model: {tuple(self.densify_projected_extent_max.shape)} "
                f"versus ({num_gaussians},)."
            )
        detached_extent = projected_extent.detach()
        valid_extent = torch.isfinite(detached_extent).all(dim=1) & (detached_extent >= 0.0).all(dim=1)
        extent_max = detached_extent.amax(dim=1)
        on_screen = tile_count.detach().reshape(-1) > 0
        observed = on_screen & valid_extent & (extent_max > 0.0)
        self.densify_projected_extent_max[observed] = torch.maximum(
            self.densify_projected_extent_max[observed],
            extent_max[observed],
        )

    @staticmethod
    def _major_eigen_radius_from_inverse_covariance(
        inverse_covariance: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the major one-sigma radius with float64 conic algebra."""
        if inverse_covariance.ndim != 2 or inverse_covariance.shape[1] != 3:
            raise ValueError(
                "Inverse covariance conics must have shape (N, 3), got " f"{tuple(inverse_covariance.shape)}."
            )
        inverse_covariance_64 = inverse_covariance.detach().to(torch.float64)
        a = inverse_covariance_64[:, 0]
        b = inverse_covariance_64[:, 1]
        c = inverse_covariance_64[:, 2]
        determinant = a * c - b.square()
        trace = a + c
        discriminant = (a - c).square() + 4.0 * b.square()
        largest_denominator = trace + torch.sqrt(discriminant)
        smallest_inverse_eigenvalue = 2.0 * determinant / largest_denominator
        return torch.rsqrt(smallest_inverse_eigenvalue)

    @staticmethod
    def _classify_covariance_packets(
        packet_64: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Classify detached float64 packets with ordered, disjoint reasons."""
        if packet_64.ndim != 2 or packet_64.shape[1] != 4:
            raise ValueError(
                "Projected conic-opacity packets must have shape (N, 4), got " f"{tuple(packet_64.shape)}."
            )
        if packet_64.dtype != torch.float64:
            raise ValueError("Projected conic-opacity packet classification requires " "float64 input.")
        finite = torch.isfinite(packet_64).all(dim=1)
        nonfinite = ~finite
        a = packet_64[:, 0]
        b = packet_64[:, 1]
        c = packet_64[:, 2]
        opacity = packet_64[:, 3]
        positive_diagonal = (a > 0.0) & (c > 0.0)
        nonpositive_diagonal = finite & ~positive_diagonal
        determinant = a * c - b.square()
        determinant_domain = finite & positive_diagonal
        positive_determinant = determinant > 0.0
        nonpositive_determinant = determinant_domain & ~positive_determinant
        opacity_domain = determinant_domain & positive_determinant
        positive_opacity = opacity > 0.0
        nonpositive_opacity = opacity_domain & ~positive_opacity
        valid = opacity_domain & positive_opacity
        invalid_reasons = (
            nonfinite,
            nonpositive_diagonal,
            nonpositive_determinant,
            nonpositive_opacity,
        )
        return valid, invalid_reasons

    def _record_covariance_packet_telemetry(
        self,
        *,
        packet_64: torch.Tensor,
        gaussian_indices: torch.Tensor,
        invalid_reasons: tuple[torch.Tensor, ...],
    ) -> None:
        packet_count = packet_64.shape[0]
        self.densify_covariance_packet_observations[0] += packet_count
        invalid_union = torch.stack(invalid_reasons, dim=0).any(dim=0)
        self.densify_covariance_invalid_packet_observations[0] += invalid_union.sum()
        for reason_index, reason_mask in enumerate(invalid_reasons):
            invalid_count = int(reason_mask.sum().item())
            if invalid_count == 0:
                continue
            previous_count = int(self.densify_covariance_invalid_reason_counts[reason_index].item())
            self.densify_covariance_invalid_reason_counts[reason_index] += invalid_count
            first_empty_slot = min(
                previous_count,
                _COVARIANCE_INVALID_EXAMPLES_PER_REASON,
            )
            available_slots = _COVARIANCE_INVALID_EXAMPLES_PER_REASON - first_empty_slot
            if available_slots == 0:
                continue
            source_rows = torch.nonzero(reason_mask, as_tuple=False).reshape(-1)[:available_slots]
            destination = slice(
                first_empty_slot,
                first_empty_slot + source_rows.numel(),
            )
            self.densify_covariance_invalid_example_indices[reason_index, destination] = gaussian_indices[source_rows]
            self.densify_covariance_invalid_example_values[reason_index, destination] = packet_64[source_rows]

    @torch.no_grad()
    def update_covariance_gradient_buffer(
        self,
        outputs: dict[str, object] | None,
        *,
        sensor_position: torch.Tensor,
    ) -> None:
        """Accumulate gradient demand from the same large-radius observations."""
        if outputs is None:
            raise RuntimeError("Covariance-gradient splitting requires renderer outputs.")
        conic_opacity = outputs.get("mog_projected_conic_opacity")
        tile_count = outputs.get("mog_tiles_count")
        if not torch.is_tensor(conic_opacity):
            raise RuntimeError("Covariance-gradient splitting requires " "outputs['mog_projected_conic_opacity'].")
        if not torch.is_tensor(tile_count):
            raise RuntimeError("Covariance-gradient splitting requires " "outputs['mog_tiles_count'].")
        num_gaussians = self.model.num_gaussians
        if conic_opacity.shape != (num_gaussians, 4):
            raise ValueError(
                "mog_projected_conic_opacity must have shape "
                f"({num_gaussians}, 4), got "
                f"{tuple(conic_opacity.shape)}."
            )
        if tile_count.numel() != num_gaussians:
            raise ValueError(
                "mog_tiles_count must contain one value per Gaussian; got "
                f"{tuple(tile_count.shape)} for {num_gaussians} Gaussians."
            )
        if not conic_opacity.is_floating_point():
            raise ValueError("mog_projected_conic_opacity must be floating point.")
        if tile_count.is_floating_point():
            raise ValueError("mog_tiles_count must use an integer dtype.")
        expected_buffer_shape = (num_gaussians, 1)
        if (
            self.densify_covariance_gradient_mass.shape != expected_buffer_shape
            or self.densify_covariance_large_observations.shape != expected_buffer_shape
        ):
            raise RuntimeError(
                "Covariance-gradient buffers are not aligned with the model: "
                f"{tuple(self.densify_covariance_gradient_mass.shape)} and "
                f"{tuple(self.densify_covariance_large_observations.shape)} "
                f"versus {expected_buffer_shape}."
            )
        expected_reason_shape = (len(_COVARIANCE_INVALID_REASON_NAMES),)
        expected_example_index_shape = (
            len(_COVARIANCE_INVALID_REASON_NAMES),
            _COVARIANCE_INVALID_EXAMPLES_PER_REASON,
        )
        expected_example_value_shape = (*expected_example_index_shape, 4)
        if (
            self.densify_covariance_packet_observations.shape != (1,)
            or self.densify_covariance_invalid_packet_observations.shape != (1,)
            or self.densify_covariance_invalid_reason_counts.shape != expected_reason_shape
            or self.densify_covariance_invalid_example_indices.shape != expected_example_index_shape
            or self.densify_covariance_invalid_example_values.shape != expected_example_value_shape
        ):
            raise RuntimeError(
                "Covariance-gradient invalid-packet telemetry is not " "initialized with its fixed window shapes."
            )
        positions = self.model.positions.detach()
        position_gradient = self.model.positions.grad
        if position_gradient is None:
            raise RuntimeError("Covariance-gradient splitting requires positional gradients.")
        expected_position_shape = (num_gaussians, 3)
        if positions.shape != expected_position_shape or position_gradient.shape != expected_position_shape:
            raise ValueError("Covariance-gradient splitting requires aligned (N, 3) " "positions and gradients.")
        if sensor_position.shape != (3,):
            raise ValueError(
                "Covariance-gradient splitting requires sensor_position "
                f"shape (3,), got {tuple(sensor_position.shape)}."
            )
        if (
            not self.densify_covariance_gradient_mass.is_floating_point()
            or self.densify_covariance_large_observations.is_floating_point()
            or self.densify_covariance_packet_observations.dtype != torch.int64
            or self.densify_covariance_invalid_packet_observations.dtype != torch.int64
            or self.densify_covariance_invalid_reason_counts.dtype != torch.int64
            or self.densify_covariance_invalid_example_indices.dtype != torch.int64
            or self.densify_covariance_invalid_example_values.dtype != torch.float64
        ):
            raise RuntimeError(
                "Covariance-gradient buffers must use floating-point mass " "and their declared telemetry dtypes."
            )
        model_device = positions.device
        for name, value in (
            ("mog_projected_conic_opacity", conic_opacity),
            ("mog_tiles_count", tile_count),
            ("sensor_position", sensor_position),
            ("position_gradient", position_gradient),
            (
                "densify_covariance_gradient_mass",
                self.densify_covariance_gradient_mass,
            ),
            (
                "densify_covariance_large_observations",
                self.densify_covariance_large_observations,
            ),
            (
                "densify_covariance_packet_observations",
                self.densify_covariance_packet_observations,
            ),
            (
                "densify_covariance_invalid_packet_observations",
                self.densify_covariance_invalid_packet_observations,
            ),
            (
                "densify_covariance_invalid_reason_counts",
                self.densify_covariance_invalid_reason_counts,
            ),
            (
                "densify_covariance_invalid_example_indices",
                self.densify_covariance_invalid_example_indices,
            ),
            (
                "densify_covariance_invalid_example_values",
                self.densify_covariance_invalid_example_values,
            ),
        ):
            if value.device != model_device:
                raise ValueError(f"{name} must share the model device {model_device}, got " f"{value.device}.")
        if (
            not torch.isfinite(positions).all()
            or not torch.isfinite(position_gradient).all()
            or not torch.isfinite(sensor_position).all()
        ):
            raise RuntimeError(
                "Covariance-gradient splitting encountered non-finite " "position, gradient, or sensor state."
            )
        on_screen = tile_count.detach().reshape(-1) > 0
        if not on_screen.any():
            return
        gaussian_indices = torch.nonzero(on_screen, as_tuple=False).reshape(-1)
        observed_conic_opacity = conic_opacity.detach()[on_screen].to(torch.float64)
        valid_packet, invalid_reasons = self._classify_covariance_packets(observed_conic_opacity)
        self._record_covariance_packet_telemetry(
            packet_64=observed_conic_opacity,
            gaussian_indices=gaussian_indices,
            invalid_reasons=invalid_reasons,
        )
        if not valid_packet.any():
            return
        valid_indices = gaussian_indices[valid_packet]
        inverse_covariance = observed_conic_opacity[valid_packet, :3]
        observed_radii = self._major_eigen_radius_from_inverse_covariance(inverse_covariance)
        if not torch.isfinite(observed_radii).all():
            raise RuntimeError("Covariance-gradient splitting produced a non-finite " "on-screen eigen-radius.")
        large_radius = torch.zeros(
            num_gaussians,
            dtype=torch.bool,
            device=model_device,
        )
        large_radius[valid_indices] = observed_radii > self.covariance_gradient_split_radius_px
        nonzero_gradient = (position_gradient != 0.0).any(dim=1)
        same_observation = large_radius & nonzero_gradient
        if not same_observation.any():
            return
        distance_to_camera = (positions[same_observation] - sensor_position).norm(dim=1, keepdim=True)
        gradient_increment = (
            torch.norm(
                position_gradient[same_observation] * distance_to_camera,
                dim=-1,
                keepdim=True,
            )
            / 2.0
        )
        if not torch.isfinite(gradient_increment).all():
            raise RuntimeError("Covariance-gradient splitting produced non-finite " "positional gradient mass.")
        self.densify_covariance_gradient_mass[same_observation] += gradient_increment
        self.densify_covariance_large_observations[same_observation] += 1

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

        # Neutralize degenerate particles before the next BVH build
        if self.conf.strategy.scale_guard.enabled:
            self.sanitize_particles()

        # Densify the Gaussians

        if check_step_condition(
            step,
            self.conf.strategy.densify.start_iteration,
            self.conf.strategy.densify.end_iteration,
            self.conf.strategy.densify.frequency,
        ):
            self.densify_gaussians(
                scene_extent=scene_extent,
                writer=writer,
                step=step,
            )
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

        grad_increment = torch.norm(params_grad[mask] * distance_to_camera, dim=-1, keepdim=True) / 2
        self.densify_grad_norm_accum[mask] += grad_increment
        self.densify_grad_norm_denom[mask] += 1

        # Field-angle-aware: accumulate the gradient-weighted cos(theta) of
        # each observed Gaussian relative to this camera's optical axis. The
        # weight is the gradient increment just added, so at densify time the
        # ratio (accum / weight) is the gradient-weighted mean cos(theta) --
        # "the field angle this Gaussian's densification demand came from".
        if self.theta_aware and sensor_forward is not None:
            forward = sensor_forward / sensor_forward.norm().clamp_min(1e-8)
            cos_theta = (to_camera / distance_to_camera.clamp_min(1e-8)) @ forward
            cos_theta = cos_theta.unsqueeze(1).clamp(-1.0, 1.0)
            self.densify_cos_accum[mask] += grad_increment * cos_theta
            self.densify_cos_weight[mask] += grad_increment

        feature_grad = self.model.features_specular.grad
        if self.feature_grad_aware and feature_grad is not None:
            feature_grad_values = feature_grad.detach()[mask]
            if self.feature_grad_carrier_tail_only:
                sh_dim = sh_degree_to_specular_dim(self.model.get_max_n_features())
                feature_grad_values = feature_grad_values[:, sh_dim:]
            feature_increment = feature_grad_values.flatten(start_dim=1).norm(dim=1, keepdim=True)
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

    @torch.no_grad()
    def _validate_covariance_packet_window(
        self,
        *,
        writer: object | None,
        step: int | None,
    ) -> None:
        """Emit and enforce invalid-packet telemetry before topology changes."""
        packet_count = int(self.densify_covariance_packet_observations[0].item())
        invalid_count = int(self.densify_covariance_invalid_packet_observations[0].item())
        reason_counts = tuple(int(value) for value in self.densify_covariance_invalid_reason_counts.tolist())
        if packet_count < 0 or invalid_count < 0:
            raise RuntimeError("Covariance-gradient packet telemetry counts must be " "non-negative.")
        if invalid_count > packet_count:
            raise RuntimeError("Covariance-gradient invalid-packet count exceeds its " "on-screen packet denominator.")
        if sum(reason_counts) != invalid_count:
            raise RuntimeError(
                "Covariance-gradient ordered predicate counts do not sum to " "the invalid-packet union count."
            )
        invalid_fraction = invalid_count / packet_count if packet_count else 0.0
        boundary = f"step={step}" if step is not None else "step=unknown"
        predicate_text = ", ".join(
            f"{name}={count}"
            for name, count in zip(
                _COVARIANCE_INVALID_REASON_NAMES,
                reason_counts,
            )
        )
        logger.info(
            "Densify covariance packet window: "
            f"{boundary}, invalid={invalid_count}/{packet_count} "
            f"({invalid_fraction:.12g}), {predicate_text}, ceiling="
            f"{self.covariance_gradient_split_max_invalid_fraction:.12g}"
        )
        for reason_index, (reason_name, reason_count) in enumerate(
            zip(_COVARIANCE_INVALID_REASON_NAMES, reason_counts)
        ):
            stored_count = min(
                reason_count,
                _COVARIANCE_INVALID_EXAMPLES_PER_REASON,
            )
            if stored_count == 0:
                continue
            example_indices = self.densify_covariance_invalid_example_indices[reason_index, :stored_count].tolist()
            example_values = self.densify_covariance_invalid_example_values[reason_index, :stored_count].tolist()
            examples = tuple(zip(example_indices, example_values))
            logger.info(
                "Densify covariance packet examples: " f"{boundary}, predicate={reason_name}, values={examples}"
            )
        if writer is not None and step is not None:
            telemetry = {
                "train/densify/covgrad_conic_packets": float(packet_count),
                "train/densify/covgrad_invalid_packets": float(invalid_count),
                "train/densify/covgrad_invalid_fraction": invalid_fraction,
            }
            for reason_name, reason_count in zip(
                _COVARIANCE_INVALID_REASON_NAMES,
                reason_counts,
            ):
                telemetry[f"train/densify/covgrad_invalid_{reason_name}"] = float(reason_count)
            for metric_name, value in telemetry.items():
                writer.add_scalar(metric_name, value, step)
        if invalid_fraction > self.covariance_gradient_split_max_invalid_fraction:
            raise RuntimeError(
                "Covariance-gradient invalid packet fraction "
                f"{invalid_fraction:.12f} ({invalid_count}/{packet_count}) "
                "exceeds the preregistered per-window ceiling "
                f"{self.covariance_gradient_split_max_invalid_fraction:.12f} "
                f"at {boundary}; clone/split topology was not mutated."
            )

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(
        self,
        scene_extent: float,
        *,
        writer: object | None = None,
        step: int | None = None,
    ) -> None:
        assert (
            self.model.optimizer is not None
        ), "Optimizer need to be initialized before splitting and cloning the Gaussians"
        scene_extent = float(scene_extent)
        densify_grad_norm = self.densify_grad_norm_accum / self.densify_grad_norm_denom
        densify_grad_norm[densify_grad_norm.isnan()] = 0.0

        mean_cos = self._theta_threshold_factors()
        feature_boost = self._feature_grad_boost_factors()
        if feature_boost is not None:
            densify_grad_norm = densify_grad_norm * feature_boost.unsqueeze(1)

        densify_scores = densify_grad_norm.squeeze(-1)
        projected_extent_max = self.densify_projected_extent_max.detach().clone()
        covariance_gradient_mass = self.densify_covariance_gradient_mass.detach().clone()
        covariance_large_observations = self.densify_covariance_large_observations.detach().clone()
        covariance_total_observations = self.densify_grad_norm_denom.detach().clone()
        scale_shape_observation_count = None
        if self.scale_shape_split_enabled:
            scale_shape_observation_count = self.densify_scale_shape_observation_count.detach().clone()
        if self.covariance_gradient_split_enabled:
            self._validate_covariance_packet_window(
                writer=writer,
                step=step,
            )
        self.log_densify_stats(
            densify_scores,
            scene_extent,
            mean_cos,
            feature_boost,
            projected_extent_max,
            covariance_gradient_mass,
            covariance_large_observations,
            covariance_total_observations,
            writer=writer,
            step=step,
        )
        self.clone_gaussians(
            densify_scores,
            scene_extent,
            mean_cos,
            projected_extent_max,
            covariance_gradient_mass,
            covariance_large_observations,
            covariance_total_observations,
        )
        self.split_gaussians(
            densify_scores,
            scene_extent,
            mean_cos,
            projected_extent_max,
            covariance_gradient_mass,
            covariance_large_observations,
            covariance_total_observations,
            scale_shape_observation_count,
            writer=writer,
            step=step,
        )

        torch.cuda.empty_cache()

    @torch.no_grad()
    @jaxtyped(typechecker=beartype)
    def log_densify_stats(
        self,
        densify_grad_norm: Float[torch.Tensor, "gaussian"],
        scene_extent: float,
        mean_cos: Float[torch.Tensor, "gaussian"] | None = None,
        feature_boost: Float[torch.Tensor, "gaussian"] | None = None,
        projected_extent_max: Float[torch.Tensor, "gaussian"] | None = None,
        covariance_gradient_mass: torch.Tensor | None = None,
        covariance_large_observations: torch.Tensor | None = None,
        covariance_total_observations: torch.Tensor | None = None,
        *,
        writer: object | None = None,
        step: int | None = None,
    ) -> None:
        if not self.conf.strategy.print_stats and not self.covariance_gradient_split_enabled:
            return
        n_points = densify_grad_norm.shape[0]
        finite_mask = torch.isfinite(densify_grad_norm)
        finite_values = densify_grad_norm[finite_mask]
        if finite_values.numel() == 0:
            logger.info("Densify grad stats: no finite gradients")
            return
        positive_values = finite_values[finite_values > 0]
        if positive_values.numel() > 200000:
            stride = max(positive_values.numel() // 200000, 1)
            positive_values = positive_values[::stride][:200000]
        clone_mask, split_mask, projected_split_mask = self._densify_candidate_masks(
            densify_grad_norm,
            scene_extent,
            mean_cos,
            projected_extent_max,
            covariance_gradient_mass,
            covariance_large_observations,
            covariance_total_observations,
        )
        if self.covariance_gradient_split_enabled:
            if covariance_gradient_mass is None:
                raise RuntimeError("Covariance-gradient telemetry requires gradient mass.")
            if covariance_large_observations is None:
                raise RuntimeError("Covariance-gradient telemetry requires observation " "counts.")
            if covariance_total_observations is None:
                raise RuntimeError("Covariance-gradient telemetry requires total " "observation counts.")
            large_observation_count = covariance_large_observations.squeeze(-1)
            total_observation_count = covariance_total_observations.squeeze(-1)
            observed = large_observation_count > 0
            enough_observations = large_observation_count >= self.covariance_gradient_split_min_large_observations
            conditioned_gradient = covariance_gradient_mass.squeeze(-1) / total_observation_count.to(
                covariance_gradient_mass.dtype
            ).clamp_min(1)
            conditioned_threshold, _ = self._theta_thresholds(
                self.clone_grad_threshold,
                scene_extent,
                mean_cos,
                n_points,
            )
            conditioned_eligible = enough_observations & (conditioned_gradient >= conditioned_threshold)
            rerouted = int(projected_split_mask.sum().item())
            clone_population = rerouted + int(clone_mask.sum().item())
            reroute_fraction = rerouted / clone_population if clone_population else 0.0
            logger.info(
                "Densify covariance-gradient: "
                f"large_observed={int(observed.sum())}/{n_points}, "
                f"min_{self.covariance_gradient_split_min_large_observations}="
                f"{int(enough_observations.sum())}, "
                f"conditioned_eligible={int(conditioned_eligible.sum())}, "
                f"radius>{self.covariance_gradient_split_radius_px:g}px, "
                f"rerouted_splits={rerouted}/{clone_population} "
                f"({reroute_fraction:.6f}), "
                "ceiling="
                f"{self.covariance_gradient_split_max_reroute_fraction:.6f}"
            )
            if writer is not None and step is not None:
                telemetry = {
                    "train/densify/covgrad_large_observed": float(observed.sum().item()),
                    "train/densify/covgrad_min_observations": float(enough_observations.sum().item()),
                    "train/densify/covgrad_conditioned_eligible": float(conditioned_eligible.sum().item()),
                    "train/densify/covgrad_rerouted_splits": float(rerouted),
                    "train/densify/covgrad_clone_population": float(clone_population),
                    "train/densify/covgrad_reroute_fraction": reroute_fraction,
                }
                for metric_name, value in telemetry.items():
                    writer.add_scalar(metric_name, value, step)
        if not self.conf.strategy.print_stats:
            return
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
        if self.projected_extent_split_enabled:
            extent_values = projected_extent_max[torch.isfinite(projected_extent_max) & (projected_extent_max > 0.0)]
            logger.info(
                "Densify projected extent: "
                f"observed={extent_values.numel()}/{densify_grad_norm.numel()}, "
                f"above_{self.projected_extent_split_max_px:g}px="
                f"{int((extent_values > self.projected_extent_split_max_px).sum())}, "
                f"rerouted_splits={int(projected_split_mask.sum())}"
            )
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
    def _densify_candidate_masks(
        self,
        densify_grad_norm: torch.Tensor,
        scene_extent: float,
        mean_cos: torch.Tensor | None,
        projected_extent_max: torch.Tensor | None,
        covariance_gradient_mass: torch.Tensor | None = None,
        covariance_large_observations: torch.Tensor | None = None,
        covariance_total_observations: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Partition clone candidates from unchanged world/rerouted splits."""
        n_points = densify_grad_norm.shape[0]
        max_scale = torch.max(self.model.get_scale(), dim=1).values
        if max_scale.shape != (n_points,):
            raise ValueError(
                "Densification scores must align with model Gaussians: "
                f"{tuple(densify_grad_norm.shape)} versus "
                f"{tuple(max_scale.shape)}."
            )
        clone_thresh, size_threshold = self._theta_thresholds(
            self.clone_grad_threshold,
            scene_extent,
            mean_cos,
            n_points,
        )
        split_thresh, _ = self._theta_thresholds(
            self.split_grad_threshold,
            scene_extent,
            mean_cos,
            n_points,
        )
        world_small = max_scale <= size_threshold
        clone_candidates = (densify_grad_norm >= clone_thresh) & world_small
        projected_oversize = torch.zeros_like(
            clone_candidates,
            dtype=torch.bool,
        )
        if self.projected_extent_split_enabled:
            if projected_extent_max is None:
                raise RuntimeError("Projected-extent split decisions require the " "accumulated extent buffer.")
            if projected_extent_max.shape != (n_points,):
                raise ValueError(
                    "Projected-extent split decisions require shape "
                    f"({n_points},), got "
                    f"{tuple(projected_extent_max.shape)}."
                )
            projected_oversize = torch.isfinite(projected_extent_max) & (
                projected_extent_max > self.projected_extent_split_max_px
            )
        projected_split = clone_candidates & projected_oversize
        covariance_gradient_split = torch.zeros_like(
            clone_candidates,
            dtype=torch.bool,
        )
        if self.covariance_gradient_split_enabled:
            if covariance_gradient_mass is None:
                raise RuntimeError(
                    "Covariance-gradient split decisions require the " "accumulated gradient-mass buffer."
                )
            if covariance_large_observations is None:
                raise RuntimeError("Covariance-gradient split decisions require the " "large-observation count buffer.")
            if covariance_total_observations is None:
                raise RuntimeError(
                    "Covariance-gradient split decisions require the " "existing total-observation denominator."
                )
            expected_shape = (n_points, 1)
            if covariance_gradient_mass.shape != expected_shape:
                raise ValueError(
                    "Covariance-gradient split decisions require gradient "
                    f"mass shape {expected_shape}, got "
                    f"{tuple(covariance_gradient_mass.shape)}."
                )
            if covariance_large_observations.shape != expected_shape:
                raise ValueError(
                    "Covariance-gradient split decisions require observation "
                    f"count shape {expected_shape}, got "
                    f"{tuple(covariance_large_observations.shape)}."
                )
            if covariance_total_observations.shape != expected_shape:
                raise ValueError(
                    "Covariance-gradient split decisions require total "
                    f"observation shape {expected_shape}, got "
                    f"{tuple(covariance_total_observations.shape)}."
                )
            model_device = self.model.positions.device
            if (
                covariance_gradient_mass.device != model_device
                or covariance_large_observations.device != model_device
                or covariance_total_observations.device != model_device
            ):
                raise ValueError("Covariance-gradient split buffers must share the model " f"device {model_device}.")
            if (
                not covariance_gradient_mass.is_floating_point()
                or covariance_large_observations.is_floating_point()
                or covariance_total_observations.is_floating_point()
            ):
                raise ValueError(
                    "Covariance-gradient split buffers must use floating-point " "mass and integer observation counts."
                )
            if (
                not torch.isfinite(covariance_gradient_mass).all()
                or (covariance_gradient_mass < 0.0).any()
                or (covariance_large_observations < 0).any()
                or (covariance_total_observations < 0).any()
                or (covariance_large_observations > covariance_total_observations).any()
            ):
                raise RuntimeError(
                    "Covariance-gradient split decisions require finite, "
                    "non-negative accumulation buffers with large-observation "
                    "counts bounded by the existing total denominator."
                )
            large_observation_count = covariance_large_observations.squeeze(-1)
            total_observation_count = covariance_total_observations.squeeze(-1)
            conditioned_gradient = covariance_gradient_mass.squeeze(-1) / total_observation_count.to(
                covariance_gradient_mass.dtype
            ).clamp_min(1)
            covariance_condition = (
                large_observation_count >= self.covariance_gradient_split_min_large_observations
            ) & (conditioned_gradient >= clone_thresh)
            covariance_gradient_split = clone_candidates & covariance_condition
            clone_population = int(clone_candidates.sum().item())
            rerouted = int(covariance_gradient_split.sum().item())
            reroute_fraction = rerouted / clone_population if clone_population else 0.0
            if reroute_fraction > self.covariance_gradient_split_max_reroute_fraction:
                raise RuntimeError(
                    "Covariance-gradient reroute fraction "
                    f"{reroute_fraction:.12f} ({rerouted}/"
                    f"{clone_population}) exceeds the preregistered ceiling "
                    f"{self.covariance_gradient_split_max_reroute_fraction:.12f}."
                )
        rerouted_split = projected_split | covariance_gradient_split
        clone_mask = clone_candidates & ~rerouted_split
        world_split = (densify_grad_norm >= split_thresh) & ~world_small
        split_mask = world_split | rerouted_split
        return clone_mask, split_mask, rerouted_split

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
        projected_extent_max: Optional[torch.Tensor] = None,
        covariance_gradient_mass: torch.Tensor | None = None,
        covariance_large_observations: torch.Tensor | None = None,
        covariance_total_observations: torch.Tensor | None = None,
        scale_shape_observation_count: torch.Tensor | None = None,
        *,
        writer: object | None = None,
        step: int | None = None,
    ) -> None:
        n_init_points = self.model.num_gaussians

        if self.covariance_gradient_split_enabled and (
            covariance_gradient_mass is None
            or covariance_large_observations is None
            or covariance_total_observations is None
        ):
            raise RuntimeError("Covariance-gradient splitting requires all pre-clone " "accumulation snapshots.")
        if self.scale_shape_split_enabled and scale_shape_observation_count is None:
            raise RuntimeError("scale-shape splitting requires its pre-clone " "observation-count snapshot.")

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(
            n_init_points,
            device=self.model.positions.device,
        )

        # Here we already have the cloned points in the self.model.positions so only take the points up to size of the initial grad
        padded_grad[: densify_grad_norm.shape[0]] = densify_grad_norm.squeeze()
        padded_extent = torch.zeros_like(padded_grad)
        if projected_extent_max is not None:
            padded_extent[: projected_extent_max.shape[0]] = projected_extent_max
        padded_covariance_gradient_mass = torch.zeros(
            (n_init_points, 1),
            device=padded_grad.device,
            dtype=(covariance_gradient_mass.dtype if covariance_gradient_mass is not None else padded_grad.dtype),
        )
        if covariance_gradient_mass is not None:
            padded_covariance_gradient_mass[: covariance_gradient_mass.shape[0]] = covariance_gradient_mass
        padded_covariance_large_observations = torch.zeros(
            (n_init_points, 1),
            device=padded_grad.device,
            dtype=(covariance_large_observations.dtype if covariance_large_observations is not None else torch.int),
        )
        if covariance_large_observations is not None:
            padded_covariance_large_observations[: covariance_large_observations.shape[0]] = (
                covariance_large_observations
            )
        padded_covariance_total_observations = torch.zeros(
            (n_init_points, 1),
            device=padded_grad.device,
            dtype=(covariance_total_observations.dtype if covariance_total_observations is not None else torch.int),
        )
        if covariance_total_observations is not None:
            padded_covariance_total_observations[: covariance_total_observations.shape[0]] = (
                covariance_total_observations
            )
        padded_scale_shape_observations = torch.zeros(
            (n_init_points, 1),
            device=padded_grad.device,
            dtype=(scale_shape_observation_count.dtype if scale_shape_observation_count is not None else torch.int),
        )
        if scale_shape_observation_count is not None:
            padded_scale_shape_observations[: scale_shape_observation_count.shape[0]] = scale_shape_observation_count
        _, mask, _ = self._densify_candidate_masks(
            padded_grad,
            scene_extent,
            mean_cos,
            padded_extent,
            padded_covariance_gradient_mass,
            padded_covariance_large_observations,
            padded_covariance_total_observations,
        )

        scale_shape_mask = torch.zeros_like(mask)
        if self.scale_shape_split_enabled:
            scale_shape_mask = scale_shape_split_mask(
                split_candidates=mask,
                physical_scales=self.model.get_scale(),
                observation_count=padded_scale_shape_observations,
                thresholds=ScaleShapeThresholds(
                    anisotropy=(self.scale_shape_anisotropy_threshold),
                    min_largest_scale=(self.scale_shape_min_largest_scale),
                    min_observations=(self.scale_shape_min_observations),
                ),
            )

        stds = self.model.get_scale()[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros(
            (stds.size(0), 3),
            dtype=stds.dtype,
            device=stds.device,
        )
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.model.rotation[mask]).repeat(self.split_n_gaussians, 1, 1)
        offsets = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
        scale_shape_child_mask = scale_shape_mask[mask].repeat(self.split_n_gaussians)
        scale_shape_child_positions = torch.empty(
            (0, 3),
            dtype=self.model.positions.dtype,
            device=self.model.positions.device,
        )
        scale_shape_child_scales = torch.empty_like(scale_shape_child_positions)
        scale_shape_child_density = torch.empty(
            (0, 1),
            dtype=self.model.density.dtype,
            device=self.model.density.device,
        )
        if bool(scale_shape_mask.any()):
            (
                scale_shape_child_positions,
                scale_shape_child_scales,
            ) = deterministic_split_children(
                positions=self.model.positions[scale_shape_mask],
                physical_scales=self.model.get_scale()[scale_shape_mask],
                rotations=self.model.get_rotation()[scale_shape_mask],
            )
            scale_shape_child_density = self.model.density_activation_inv(
                transmittance_preserving_child_opacity(self.model.get_density()[scale_shape_mask])
            )
        scale_shape_count = int(scale_shape_mask.sum().item())
        split_count = int(mask.sum().item())
        if self.scale_shape_split_enabled and writer is not None and step is not None:
            writer.add_scalar(
                "train/densify/scale_shape_rerouted_splits",
                float(scale_shape_count),
                step,
            )
            writer.add_scalar(
                "train/densify/scale_shape_reroute_fraction",
                scale_shape_count / split_count if split_count else 0.0,
                step,
            )
        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Splitted {n_clone} / {n_before} ({n_clone / n_before * 100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            repeats = [self.split_n_gaussians] + [1] * (param.dim() - 1)
            if name == "positions":
                p_split = param[mask].repeat(repeats) + offsets  # [2N, 3]
                p_split[scale_shape_child_mask] = scale_shape_child_positions
            elif name == "scale":
                p_split = self.model.scale_activation_inv(
                    self.model.scale_activation(param[mask].repeat(repeats)) / (0.8 * self.split_n_gaussians)
                )
                p_split[scale_shape_child_mask] = self.model.scale_activation_inv(scale_shape_child_scales)
            elif name == "density":
                p_split = param[mask].repeat(repeats)
                p_split[scale_shape_child_mask] = scale_shape_child_density
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
                dtype=v.dtype,
            )
            return torch.cat([v[~mask], v_split])

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        if self.conf.strategy.print_stats and self.scale_shape_split_enabled:
            logger.info(
                "scale-shape deterministically rerouted "
                f"{scale_shape_count}/{split_count} existing split candidates."
            )
        self.reset_densification_buffers()

    @torch.cuda.nvtx.range("clone_gaussians")
    def clone_gaussians(
        self,
        densify_grad_norm: torch.Tensor,
        scene_extent: float,
        mean_cos: Optional[torch.Tensor] = None,
        projected_extent_max: Optional[torch.Tensor] = None,
        covariance_gradient_mass: torch.Tensor | None = None,
        covariance_large_observations: torch.Tensor | None = None,
        covariance_total_observations: torch.Tensor | None = None,
    ) -> None:
        assert densify_grad_norm is not None, "Positional gradients must be available in order to clone the Gaussians"
        densify_grad_norm = densify_grad_norm.squeeze()
        mask, _, _ = self._densify_candidate_masks(
            densify_grad_norm,
            scene_extent,
            mean_cos,
            projected_extent_max,
            covariance_gradient_mass,
            covariance_large_observations,
            covariance_total_observations,
        )

        # stats
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            logger.info(f"Cloned {n_clone} / {n_before} ({n_clone / n_before * 100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            param_new = torch.cat([param, param[mask]])
            return torch.nn.Parameter(param_new, requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    v,
                    torch.zeros(
                        (int(mask.sum()), *v.shape[1:]),
                        device=v.device,
                        dtype=v.dtype,
                    ),
                ]
            )

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.reset_densification_buffers()

    def prune_gaussians_weight(self):
        # Prune the Gaussians based on their weight
        mask = self.model.rolling_weight_contrib[:, 0] >= self.conf.strategy.prune_weight.weight_threshold
        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Weight-pruned {n_prune} / {n_before} ({n_prune / n_before * 100:.2f}%) gaussians")

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
            logger.info(f"Scale-pruned {n_prune} / {n_before} ({n_prune / n_before * 100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[mask], requires_grad=param.requires_grad)

        def update_optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
            return v[mask]

        self._update_param_with_optimizer(update_param_fn, update_optimizer_fn)
        self.prune_densification_buffers(mask)

    @torch.no_grad()
    def sanitize_particles(self) -> None:
        """Neutralize non-finite particles and clamp runaway scales.

        A non-finite position/rotation/scale/density, or a scale far
        beyond the scene, yields a degenerate proxy AABB and OptiX
        traversal against such a BVH can spin forever (100% GPU, no
        error, no progress). Runs every optimizer step when
        ``strategy.scale_guard.enabled`` so a defect born in one
        optimizer step never reaches the next render.
        """
        if self.conf.model.scale_activation != "exp":
            raise ValueError(
                "strategy.scale_guard requires the exp scale activation, " f"got {self.conf.model.scale_activation!r}."
            )
        max_pre = math.log(self.conf.strategy.scale_guard.max_world_size)
        tiny_pre = math.log(1e-6)
        scale = self.model.scale.data
        bad_rows = ~torch.isfinite(scale).all(dim=1)
        for param in (
            self.model.positions.data,
            self.model.rotation.data,
            self.model.density.data,
        ):
            bad_rows |= ~torch.isfinite(param).all(dim=1)
        oversize_rows = (scale.amax(dim=1) > max_pre) & ~bad_rows
        n_bad = int(bad_rows.sum())
        n_oversize = int(oversize_rows.sum())
        if n_bad == 0 and n_oversize == 0:
            return

        # A gaussian hovering at the cap re-clamps every step; only the
        # non-finite case is a real pathology, so oversize-only events
        # log sparsely to keep the training log readable.
        oversize_events = getattr(self, "_scale_guard_oversize_events", 0)
        self._scale_guard_oversize_events = oversize_events + 1
        if n_bad > 0 or oversize_events % 200 == 0:
            logger.warning(
                f"scale-guard: neutralized {n_bad} non-finite and clamped "
                f"{n_oversize} oversized gaussians (max_world_size="
                f"{self.conf.strategy.scale_guard.max_world_size}, "
                f"oversize event #{oversize_events + 1})"
            )
        if n_bad:
            self.model.positions.data[bad_rows] = 0.0
            self.model.rotation.data[bad_rows] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                dtype=self.model.rotation.dtype,
                device=self.model.rotation.device,
            )
            scale[bad_rows] = tiny_pre
            self.model.density.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        if n_oversize:
            scale[oversize_rows] = scale[oversize_rows].clamp(max=max_pre)

        # A poisoned Adam moment regrows the defect next step; zero the
        # touched rows (and any non-finite state entries) as well.
        touched = bad_rows | oversize_rows
        for param_group in self.model.optimizer.param_groups:
            if param_group["name"] not in (
                "positions",
                "rotation",
                "scale",
                "density",
            ):
                continue
            param = param_group["params"][0]
            for key, value in self.model.optimizer.state.get(param, {}).items():
                if key != "step" and torch.is_tensor(value) and value.shape[0] == touched.shape[0]:
                    value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    value[touched] = 0.0

    def prune_gaussians_opacity(self):
        # Prune the Gaussians based on their opacity
        mask = self.model.get_density().squeeze() >= self.prune_density_threshold

        if self.conf.strategy.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            logger.info(f"Density-pruned {n_prune} / {n_before} ({n_prune / n_before * 100:.2f}%) gaussians")

        def update_param_fn(name: str, param: torch.Tensor) -> torch.Tensor:
            return torch.nn.Parameter(param[mask], requires_grad=param.requires_grad)

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
        self.densify_cos_accum = torch.zeros((n, 1), device=self.model.device, dtype=torch.float)
        self.densify_cos_weight = torch.zeros((n, 1), device=self.model.device, dtype=torch.float)
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
        self.densify_projected_extent_max = torch.zeros(
            n,
            device=self.model.device,
            dtype=self.densify_projected_extent_max.dtype,
        )
        self.densify_scale_shape_observation_count = torch.zeros(
            (n, 1),
            device=self.model.device,
            dtype=self.densify_scale_shape_observation_count.dtype,
        )
        if not self.scale_shape_split_enabled:
            self.densify_scale_shape_observation_count = self.densify_scale_shape_observation_count[:0]
        self.densify_covariance_gradient_mass = torch.zeros(
            (n, 1),
            device=self.model.device,
            dtype=self.densify_covariance_gradient_mass.dtype,
        )
        self.densify_covariance_large_observations = torch.zeros(
            (n, 1),
            device=self.model.device,
            dtype=self.densify_covariance_large_observations.dtype,
        )
        (
            self.densify_covariance_packet_observations,
            self.densify_covariance_invalid_packet_observations,
            self.densify_covariance_invalid_reason_counts,
            self.densify_covariance_invalid_example_indices,
            self.densify_covariance_invalid_example_values,
        ) = self._empty_covariance_packet_telemetry()

    def prune_densification_buffers(self, valid_mask: torch.Tensor) -> None:
        # Update non-optimizable buffers
        self.densify_grad_norm_accum = self.densify_grad_norm_accum[valid_mask]
        self.densify_grad_norm_denom = self.densify_grad_norm_denom[valid_mask]
        self.densify_projected_extent_max = self.densify_projected_extent_max[valid_mask]
        if self.densify_scale_shape_observation_count.shape[0] == (valid_mask.shape[0]):
            self.densify_scale_shape_observation_count = self.densify_scale_shape_observation_count[valid_mask]
        self.densify_covariance_gradient_mass = self.densify_covariance_gradient_mass[valid_mask]
        self.densify_covariance_large_observations = self.densify_covariance_large_observations[valid_mask]
        if self.densify_cos_accum.shape[0] == valid_mask.shape[0]:
            self.densify_cos_accum = self.densify_cos_accum[valid_mask]
            self.densify_cos_weight = self.densify_cos_weight[valid_mask]
        if self.densify_feature_grad_accum.shape[0] == valid_mask.shape[0]:
            self.densify_feature_grad_accum = self.densify_feature_grad_accum[valid_mask]
            self.densify_feature_grad_denom = self.densify_feature_grad_denom[valid_mask]

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
