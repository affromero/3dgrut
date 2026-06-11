# SPDX-License-Identifier: Apache-2.0
"""Bounded learnable exposure time for the blur forward model.

The K exposure-sample offsets are ``(k+0.5)/K * t_exp``; making ``t_exp``
learnable lets the model track auto-exposure (measured bimodal on Hilti:
~1.2 ms bright corridor vs ~7.3 ms dark sections) instead of a fixed
scene-wide value. Bounded around the forensic initialization with a tanh
(default ±60% — wide enough to span the measured bimodality from a midpoint
init, tight enough that blur cannot absorb arbitrary appearance error),
optionally per image. Gradients reach ``t_exp`` through the differentiable
knot interpolation (`torch_knots`) and the tracer's backward ray gradients.
"""

import torch
import torch.nn as nn


class ExposureModel(nn.Module):
    """Global + optional per-image bounded exposure-time residual."""

    def __init__(
        self,
        *,
        t_exp_init_s: float,
        num_images: int = 0,
        max_rel_deviation: float = 0.6,
        lr: float = 0.005,
        reg_lambda: float = 0.1,
        warmup_steps: int = 0,
        n_iterations: int = 7000,
    ):
        super().__init__()
        if t_exp_init_s <= 0.0:
            raise ValueError(
                f"t_exp_init_s must be positive, got {t_exp_init_s}."
            )
        if not 0.0 < max_rel_deviation < 1.0:
            raise ValueError(
                "max_rel_deviation must be in (0, 1) so t_exp stays "
                f"positive, got {max_rel_deviation}."
            )
        self.t_exp_init_s = float(t_exp_init_s)
        self.max_rel_deviation = float(max_rel_deviation)
        self.lr = float(lr)
        self.reg_lambda = float(reg_lambda)
        self.warmup_steps = int(warmup_steps)
        self.n_iterations = int(n_iterations)
        self.global_raw = nn.Parameter(torch.zeros(1))
        self.per_image = num_images > 0
        if self.per_image:
            self.image_raw = nn.Parameter(torch.zeros(num_images))

    def t_exp_s(self, image_idx: int = -1, global_step: int = -1):
        """Current exposure time (seconds) as a differentiable scalar.

        Args:
            image_idx: Frame index for the per-image term (-1 = none).
            global_step: Training step; during warmup the INIT value is
                returned as a constant so the blur model trains at the
                forensic exposure before timing optimization starts.
        """
        if 0 <= global_step < self.warmup_steps:
            return torch.tensor(
                self.t_exp_init_s, device=self.global_raw.device
            )
        raw = self.global_raw
        if self.per_image and image_idx >= 0:
            raw = raw + self.image_raw[image_idx : image_idx + 1]
        deviation = self.max_rel_deviation * torch.tanh(raw)
        return (self.t_exp_init_s * (1.0 + deviation)).squeeze(0)

    def get_regularization_loss(self):
        loss = self.reg_lambda * self.global_raw.square().mean()
        if self.per_image:
            loss = loss + self.reg_lambda * self.image_raw.square().mean()
        return loss

    def max_abs_grad(self) -> float:
        grads = [
            parameter.grad.abs().max()
            for parameter in self.parameters()
            if parameter.grad is not None
        ]
        if not grads:
            return float("nan")
        return float(torch.stack(grads).max())

    def create_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.n_iterations - self.warmup_steps),
            eta_min=self.lr * 0.01,
        )
        return optimizer, scheduler
