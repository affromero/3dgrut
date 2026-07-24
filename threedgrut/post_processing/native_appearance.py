from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

NATIVE_APPEARANCE_CHANNELS = 8
NATIVE_APPEARANCE_GRID_SIZE = 33
NATIVE_APPEARANCE_REGULARIZATION = 0.1
NATIVE_APPEARANCE_STEP_SCALE = 0.1
NATIVE_APPEARANCE_POWER_GRADIENT_FLOOR = 0.1


class IndexedAppearanceAdam(torch.optim.Optimizer):
    """Adam with native-style counters for sparse appearance texels."""

    def __init__(
        self,
        weight: nn.Parameter,
        *,
        lr: float,
        beta1: float = 0.8,
        beta2: float = 0.95,
        eps: float = 1.024e-4,
    ) -> None:
        if weight.ndim != 2:
            raise ValueError("appearance weight must have shape [frames, values]")
        if weight.shape[1] % NATIVE_APPEARANCE_CHANNELS != 0:
            raise ValueError("appearance values must contain eight-channel texels")
        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": eps,
        }
        super().__init__((weight,), defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = cast(float, group["lr"])
            beta1, beta2 = cast(tuple[float, float], group["betas"])
            eps = cast(float, group["eps"])
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not gradient.is_sparse:
                    raise RuntimeError("indexed appearance Adam requires a sparse gradient")
                self._step_sparse_parameter(
                    parameter,
                    gradient.coalesce(),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                )
        return loss

    def _step_sparse_parameter(
        self,
        parameter: nn.Parameter,
        gradient: torch.Tensor,
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> None:
        state = self.state[parameter]
        texel_count = parameter.shape[1] // NATIVE_APPEARANCE_CHANNELS
        if not state:
            state["step"] = torch.zeros(
                (parameter.shape[0], texel_count),
                device=parameter.device,
                dtype=torch.int32,
            )
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)

        steps = cast(torch.Tensor, state["step"])
        exp_avg = cast(torch.Tensor, state["exp_avg"])
        exp_avg_sq = cast(torch.Tensor, state["exp_avg_sq"])
        row_indices = gradient.indices()[0]
        row_gradients = gradient.values().reshape(
            -1,
            texel_count,
            NATIVE_APPEARANCE_CHANNELS,
        )

        for gradient_index, row_tensor in enumerate(row_indices):
            row = int(row_tensor.item())
            gradient_cells = row_gradients[gradient_index]
            active = gradient_cells.ne(0).any(dim=-1)
            if not bool(active.any()):
                continue

            row_steps = steps[row]
            active_steps = torch.clamp(row_steps[active] + 1, max=65535)
            row_steps[active] = active_steps

            row_first = exp_avg[row].view(
                texel_count,
                NATIVE_APPEARANCE_CHANNELS,
            )
            row_second = exp_avg_sq[row].view(
                texel_count,
                NATIVE_APPEARANCE_CHANNELS,
            )
            first = row_first[active].to(torch.float32)
            second = row_second[active].to(torch.float32)
            active_gradient = gradient_cells[active].to(torch.float32)
            first = first.mul(beta1).add(active_gradient, alpha=1.0 - beta1)
            second = second.mul(beta2).addcmul(
                active_gradient,
                active_gradient,
                value=1.0 - beta2,
            )
            row_first[active] = first.to(row_first.dtype)
            row_second[active] = second.to(row_second.dtype)

            step_values = active_steps.to(torch.float32)
            bias_scale = torch.sqrt(1.0 - torch.pow(beta2, step_values))
            bias_scale = bias_scale / (1.0 - torch.pow(beta1, step_values))
            schedule_input = NATIVE_APPEARANCE_STEP_SCALE * step_values
            step_schedule = 2.0 / (schedule_input + schedule_input.reciprocal())
            update = first / (torch.sqrt(second) + eps)
            parameter_row = parameter[row].view(
                texel_count,
                NATIVE_APPEARANCE_CHANNELS,
            )
            update = update + (NATIVE_APPEARANCE_REGULARIZATION * parameter_row[active].to(torch.float32))
            update = update * (lr * bias_scale * step_schedule).unsqueeze(-1)
            parameter_row[active] = parameter_row[active] - update.to(parameter.dtype)


class _NativeAppearanceShoulder(torch.autograd.Function):
    """Native-style shoulder with the recovered saturated-pixel gradients."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        pred_rgb: torch.Tensor,
        sampled: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the recovered per-pixel shoulder without changing values."""
        gain = torch.exp(sampled[:, :3])
        bias = 0.1 * sampled[:, 3:6]
        quadratic = 0.125 * sampled[:, 6:7] * pred_rgb.square()
        raw_q = pred_rgb * gain + bias + quadratic
        clipped_q = raw_q.clamp(0.0, 1.0)
        gamma = torch.exp(sampled[:, 7:8])
        residual = 1.0 - clipped_q
        ctx.save_for_backward(pred_rgb, sampled)
        return 1.0 - residual.pow(gamma)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        output_gradient: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the recovered image and appearance-vector gradients."""
        if output_gradient is None:
            return None, None
        pred_rgb, sampled = ctx.saved_tensors
        gain = torch.exp(sampled[:, :3])
        bias = 0.1 * sampled[:, 3:6]
        quadratic = 0.125 * sampled[:, 6:7]
        gamma = torch.exp(sampled[:, 7:8])
        raw_q = pred_rgb * gain + bias + quadratic * pred_rgb.square()
        clipped_q = raw_q.clamp(0.0, 1.0)
        residual = 1.0 - clipped_q
        power_base = residual.clamp_min(NATIVE_APPEARANCE_POWER_GRADIENT_FLOOR)
        q_gradient = output_gradient * gamma * power_base.pow(gamma - 1.0)
        pred_gradient = q_gradient * (gain + 2.0 * quadratic * pred_rgb)
        sampled_gradient = torch.cat(
            (
                q_gradient * pred_rgb * gain,
                0.1 * q_gradient,
                0.125
                * (q_gradient * pred_rgb.square()).sum(
                    dim=-1,
                    keepdim=True,
                ),
                -gamma * (output_gradient * residual.pow(gamma) * power_base.log()).sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        return (
            pred_gradient if ctx.needs_input_grad[0] else None,
            sampled_gradient if ctx.needs_input_grad[1] else None,
        )


class NativeAppearanceGrid(nn.Module):
    """visibility-adaptive' per-image 33x33 eight-channel appearance transform."""

    def __init__(
        self,
        num_frames: int,
        *,
        source_manifest_hash: str | None = None,
        use_fp16: bool = False,
    ) -> None:
        super().__init__()
        dtype = torch.float16 if use_fp16 else torch.float32
        self.embedding = nn.Embedding(
            num_frames,
            self.value_count,
            sparse=True,
            dtype=dtype,
        )
        nn.init.zeros_(self.embedding.weight)
        manifest_bytes = bytes.fromhex(source_manifest_hash) if source_manifest_hash is not None else bytes(32)
        self.register_buffer(
            "source_manifest_hash",
            torch.tensor(tuple(manifest_bytes), dtype=torch.uint8),
        )

    @property
    def value_count(self) -> int:
        return NATIVE_APPEARANCE_CHANNELS * NATIVE_APPEARANCE_GRID_SIZE * NATIVE_APPEARANCE_GRID_SIZE

    def create_optimizer(self, lr: float) -> IndexedAppearanceAdam:
        return IndexedAppearanceAdam(self.embedding.weight, lr=lr)

    def forward(
        self,
        pred_rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        *,
        resolution: tuple[int, int],
        frame_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        frame_index = self._frame_index(frame_idx)
        if frame_index is None:
            return pred_rgb

        frame_tensor = torch.tensor(
            (frame_index,),
            device=pred_rgb.device,
            dtype=torch.long,
        )
        grid = self.embedding(frame_tensor).reshape(
            1,
            NATIVE_APPEARANCE_GRID_SIZE,
            NATIVE_APPEARANCE_GRID_SIZE,
            NATIVE_APPEARANCE_CHANNELS,
        )
        grid = grid.permute(0, 3, 1, 2)
        width, height = resolution
        norm_x = 2.0 * pixel_coords[:, 0] / max(width, 1) - 1.0
        norm_y = 2.0 * pixel_coords[:, 1] / max(height, 1) - 1.0
        sample_coords = torch.stack((norm_x, norm_y), dim=-1)
        sample_coords = sample_coords.reshape(1, -1, 1, 2).to(grid.dtype)
        sampled = F.grid_sample(
            grid,
            sample_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        sampled = sampled.to(pred_rgb.dtype)

        return _NativeAppearanceShoulder.apply(pred_rgb, sampled)

    def _frame_index(self, frame_idx: int | torch.Tensor) -> int | None:
        if torch.is_tensor(frame_idx):
            frame_index = int(frame_idx.detach().flatten()[0].item())
        else:
            frame_index = int(frame_idx)
        if not 0 <= frame_index < self.embedding.num_embeddings:
            return None
        return frame_index
