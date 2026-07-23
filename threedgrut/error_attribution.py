# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Ray-attributed Gaussian error fields for offline diagnosis.

Each score estimates the root-sum-square of the held-out metric components'
derivatives with respect to one Gaussian parameter.  Rademacher probes avoid
reducing signed pixel or patch derivatives before squaring.  The 3DGRUT
backward pass includes the renderer's front-to-back transmittance and every
Gaussian that contributed to a pixel; this is deliberately not a nearest-point
or winner-id approximation.
"""

from __future__ import annotations

import os
from enum import StrEnum

import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from plyfile import PlyData, PlyElement

from threedgrut.model.losses import ssim
from threedgrut.utils.grad_viz import (
    scale_grad_norms,
    viridis_rgb_from_scalars,
)
from threedgrut.utils.render import C0


class ErrorAttributionMetric(StrEnum):
    """Differentiable held-out losses supported by the exporter."""

    MAE = "mae"
    MSE = "mse"
    SSIM = "ssim"
    LPIPS = "lpips"
    LOW_FREQUENCY = "lowfreq_frac"


class ErrorAttributionParameter(StrEnum):
    """Gaussian parameter lenses with physically distinct units."""

    APPEARANCE = "features_albedo"
    VIEW_DEPENDENT = "features_specular"
    POSITION = "positions"
    SCALE = "scale"
    ROTATION = "rotation"
    OPACITY = "density"


@jaxtyped(typechecker=beartype)
def native_render_evidence_maps(
    *,
    accumulated_alpha: Float[torch.Tensor, "batch height width 1"],
    depth_moment: Float[torch.Tensor, "batch height width 1"],
    depth_squared_moment: Float[torch.Tensor, "batch height width 1"],
    hit_count: Float[torch.Tensor, "batch height width 1"],
    opacity_floor: float = 1e-4,
) -> dict[str, Float[torch.Tensor, "batch height width 1"]]:
    """Recover conditional depth statistics from native render moments.

    ``pred_dist`` and ``pred_dist_squared`` are front-to-back premultiplied
    moments, respectively ``sum_i w_i z_i`` and ``sum_i w_i z_i^2``.  The
    renderer's alpha is ``sum_i w_i``.  Dividing only where alpha has support
    produces conditional hit-depth mean and variance; unsupported pixels are
    marked NaN rather than assigned a fictitious depth.
    """
    if not 0.0 < opacity_floor < 1.0:
        raise ValueError("opacity_floor must be in (0, 1).")
    if (
        accumulated_alpha.shape != depth_moment.shape
        or accumulated_alpha.shape != depth_squared_moment.shape
        or accumulated_alpha.shape != hit_count.shape
    ):
        raise ValueError("Native evidence moment shapes must match alpha.")
    alpha = accumulated_alpha.clamp(0.0, 1.0)
    supported = alpha >= opacity_floor
    safe_alpha = alpha.clamp_min(opacity_floor)
    expected_depth = depth_moment / safe_alpha
    expected_depth_squared = depth_squared_moment / safe_alpha
    depth_variance = (expected_depth_squared - expected_depth.square()).clamp_min(0.0)
    nan = torch.full_like(expected_depth, torch.nan)
    return {
        "accumulated_alpha": alpha,
        "expected_depth": torch.where(supported, expected_depth, nan),
        "depth_variance": torch.where(supported, depth_variance, nan),
        "hit_count": hit_count,
    }


@jaxtyped(typechecker=beartype)
def native_contributor_ray_fields(
    *,
    accumulated_alpha: Float[torch.Tensor, "batch height width 1"],
    depth_variance: Float[torch.Tensor, "batch height width 1"],
    hit_count: Float[torch.Tensor, "batch height width 1"],
) -> dict[str, Float[torch.Tensor, "batch height width 1"]]:
    """Return ray fields with an unambiguous ``T*alpha`` 3D reduction.

    The exporter maps each returned ``q`` to Gaussians as
    ``sum_(view,pixel) T_i * alpha_i * q``. These are contributor exposures,
    not intrinsic per-Gaussian depth variance or hit count. Invalid
    conditional-depth pixels carry zero ambiguity exposure.
    """
    if (
        accumulated_alpha.shape != depth_variance.shape
        or accumulated_alpha.shape != hit_count.shape
    ):
        raise ValueError("Native contributor-ray field shapes must match.")
    return {
        "heldout_native_ownership": torch.ones_like(accumulated_alpha),
        "depth_ambiguity_exposure": torch.nan_to_num(
            depth_variance,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        "hit_congestion_exposure": torch.nan_to_num(
            hit_count,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
    }


def _mask_like(
    mask: torch.Tensor | None,
    image: torch.Tensor,
) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(image[..., :1])
    if mask.shape[1:3] != image.shape[1:3]:
        mask = F.interpolate(
            mask.permute(0, 3, 1, 2),
            size=image.shape[1:3],
            mode="nearest",
        ).permute(0, 2, 3, 1)
    if mask.shape[0] != image.shape[0]:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    return mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)


def _masked_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Make excluded pixels equal GT so patch losses have zero attribution."""
    return prediction * mask + target.detach() * (1.0 - mask)


def _gaussian_kernel(
    *,
    radius: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=dtype,
    )
    kernel = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _low_pass(residual: torch.Tensor) -> torch.Tensor:
    channels = residual.shape[1]
    kernel = _gaussian_kernel(
        radius=10,
        sigma=5.0,
        device=residual.device,
        dtype=residual.dtype,
    )
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    filtered = F.conv2d(
        residual,
        horizontal,
        padding=(0, kernel.numel() // 2),
        groups=channels,
    )
    return F.conv2d(
        filtered,
        vertical,
        padding=(kernel.numel() // 2, 0),
        groups=channels,
    )


def attribution_loss(
    metric: ErrorAttributionMetric,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    lpips_model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Return a scalar differentiable loss in the metric's native domain."""
    weights = _mask_like(mask, prediction)
    residual = prediction - target
    denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1.0)
    if metric == ErrorAttributionMetric.MAE:
        return (residual.abs() * weights).sum() / denominator
    if metric == ErrorAttributionMetric.MSE:
        return (residual.square() * weights).sum() / denominator

    masked_prediction = _masked_prediction(prediction, target, weights)
    prediction_bchw = masked_prediction.permute(0, 3, 1, 2)
    target_bchw = target.permute(0, 3, 1, 2)
    if metric == ErrorAttributionMetric.SSIM:
        return 1.0 - ssim(prediction_bchw, target_bchw)
    if metric == ErrorAttributionMetric.LPIPS:
        if lpips_model is None:
            raise ValueError("LPIPS attribution requires an LPIPS model.")
        return lpips_model(prediction_bchw.clamp(0.0, 1.0), target_bchw).mean()
    if metric == ErrorAttributionMetric.LOW_FREQUENCY:
        residual_bchw = (residual * weights).permute(0, 3, 1, 2)
        total_energy = residual_bchw.square().mean().clamp_min(1e-12)
        return _low_pass(residual_bchw).square().mean() / total_energy
    raise ValueError(f"Unsupported attribution metric: {metric!r}")


def _ssim_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return a standard local one-minus-SSIM field normalized to sum."""
    prediction_bchw = prediction.permute(0, 3, 1, 2)
    target_bchw = target.permute(0, 3, 1, 2)
    channels = prediction_bchw.shape[1]
    kernel = _gaussian_kernel(
        radius=5,
        sigma=1.5,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    window = torch.outer(kernel, kernel).reshape(1, 1, 11, 11)
    window = window.expand(channels, 1, 11, 11)

    def filter_image(image: torch.Tensor) -> torch.Tensor:
        return F.conv2d(image, window, padding=5, groups=channels)

    mu_prediction = filter_image(prediction_bchw)
    mu_target = filter_image(target_bchw)
    mu_prediction_sq = mu_prediction.square()
    mu_target_sq = mu_target.square()
    mu_product = mu_prediction * mu_target
    sigma_prediction = (
        filter_image(prediction_bchw.square()) - mu_prediction_sq
    ).clamp_min(0.0)
    sigma_target = (
        filter_image(target_bchw.square()) - mu_target_sq
    ).clamp_min(0.0)
    sigma_product = filter_image(prediction_bchw * target_bchw) - mu_product
    numerator = (2.0 * mu_product + 0.01**2) * (2.0 * sigma_product + 0.03**2)
    denominator = (mu_prediction_sq + mu_target_sq + 0.01**2) * (
        sigma_prediction + sigma_target + 0.03**2
    )
    field = 1.0 - numerator / denominator.clamp_min(1e-12)
    return field / field.numel()


def attribution_components(
    metric: ErrorAttributionMetric,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    lpips_model: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Return spatial loss components whose sum is the reported loss."""
    weights = _mask_like(mask, prediction)
    residual = prediction - target
    denominator = (weights.sum() * prediction.shape[-1]).clamp_min(1.0)
    if metric == ErrorAttributionMetric.MAE:
        return residual.abs() * weights / denominator
    if metric == ErrorAttributionMetric.MSE:
        return residual.square() * weights / denominator

    masked_prediction = _masked_prediction(prediction, target, weights)
    if metric == ErrorAttributionMetric.SSIM:
        return _ssim_components(masked_prediction, target)
    if metric == ErrorAttributionMetric.LPIPS:
        if lpips_model is None:
            raise ValueError("LPIPS attribution requires an LPIPS model.")
        prediction_bchw = masked_prediction.permute(0, 3, 1, 2)
        target_bchw = target.permute(0, 3, 1, 2)
        lpips_network = getattr(lpips_model, "net", None)
        if lpips_network is None or not hasattr(lpips_network, "spatial"):
            raise TypeError("LPIPS model does not expose a spatial network.")
        original_spatial = lpips_network.spatial
        lpips_network.spatial = True
        field = lpips_network(
            prediction_bchw.clamp(0.0, 1.0),
            target_bchw,
            normalize=True,
        )
        lpips_network.spatial = original_spatial
        return field / field.numel()
    if metric == ErrorAttributionMetric.LOW_FREQUENCY:
        residual_bchw = (residual * weights).permute(0, 3, 1, 2)
        total_energy = residual_bchw.square().mean().clamp_min(1e-12)
        field = _low_pass(residual_bchw).square() / total_energy
        return field / field.numel()
    raise ValueError(f"Unsupported attribution metric: {metric!r}")


class ErrorAttributionAccumulator:
    """Estimate non-cancelling per-view component-gradient energy."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        metrics: tuple[ErrorAttributionMetric, ...],
        parameters: tuple[ErrorAttributionParameter, ...],
        lpips_model: torch.nn.Module | None = None,
        probe_count: int = 8,
        seed: int = 0,
    ) -> None:
        """Initialize empty per-view loss and gradient accumulators."""
        if not metrics:
            raise ValueError("At least one attribution metric is required.")
        if not parameters:
            raise ValueError("At least one Gaussian parameter is required.")
        if probe_count <= 0:
            raise ValueError("probe_count must be positive.")
        self.model = model
        self.metrics = metrics
        self.parameters = parameters
        self.lpips_model = lpips_model
        self.probe_count = probe_count
        self.seed = seed
        self._squared_sums: dict[str, torch.Tensor] = {}
        self._loss_sums = {metric.value: 0.0 for metric in metrics}
        self.view_count = 0

    def _model_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        values: list[torch.nn.Parameter] = []
        for parameter in self.parameters:
            value = getattr(self.model, parameter.value)
            if not isinstance(value, torch.nn.Parameter):
                raise TypeError(
                    f"Model attribute {parameter.value!r} is not a parameter."
                )
            values.append(value)
        return tuple(values)

    def accumulate(
        self,
        *,
        outputs: dict[str, torch.Tensor | float],
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> dict[str, float]:
        """Accumulate one view using unbiased squared-Jacobian probes."""
        prediction = outputs.get("pred_rgb")
        if not isinstance(prediction, torch.Tensor):
            raise KeyError("Renderer output has no differentiable pred_rgb.")
        if not prediction.requires_grad:
            raise ValueError(
                "pred_rgb has no autograd graph. Enable native autograd so "
                "native ray attribution remains available."
            )

        model_parameters = self._model_parameters()
        view_losses: dict[str, float] = {}
        component_fields = {
            metric: attribution_components(
                metric,
                prediction,
                target,
                mask,
                self.lpips_model,
            )
            for metric in self.metrics
        }
        total_probes = len(self.metrics) * self.probe_count
        completed_probes = 0
        for metric_index, metric in enumerate(self.metrics):
            components = component_fields[metric]
            loss = components.sum()
            loss_value = float(loss.detach())
            view_losses[metric.value] = loss_value
            self._loss_sums[metric.value] += loss_value
            generator = torch.Generator(device=components.device)
            generator.manual_seed(
                self.seed + self.view_count * 104729 + metric_index * 1009
            )
            for _ in range(self.probe_count):
                signs = torch.randint(
                    0,
                    2,
                    components.shape,
                    device=components.device,
                    generator=generator,
                ).to(dtype=components.dtype)
                signs = signs.mul_(2.0).sub_(1.0)
                probe = (components * signs).sum()
                completed_probes += 1
                gradients = torch.autograd.grad(
                    probe,
                    model_parameters,
                    retain_graph=completed_probes < total_probes,
                    allow_unused=True,
                )
                for parameter, gradient in zip(
                    self.parameters,
                    gradients,
                    strict=True,
                ):
                    if gradient is None:
                        continue
                    key = f"{metric.value}:{parameter.value}"
                    squared_norm = (
                        gradient.detach()
                        .reshape(gradient.shape[0], -1)
                        .float()
                        .square()
                        .sum(dim=1)
                        .div(self.probe_count)
                        .cpu()
                    )
                    previous = self._squared_sums.get(key)
                    self._squared_sums[key] = (
                        squared_norm
                        if previous is None
                        else previous + squared_norm
                    )
        self.view_count += 1
        return view_losses

    def rms_scores(self) -> dict[str, torch.Tensor]:
        """Return estimated RMS component-gradient norms for every field."""
        if self.view_count == 0:
            raise ValueError("No views were accumulated.")
        return {
            key: (values / self.view_count).sqrt()
            for key, values in self._squared_sums.items()
        }

    def mean_losses(self) -> dict[str, float]:
        """Return the mean image-domain loss over accumulated views."""
        if self.view_count == 0:
            raise ValueError("No views were accumulated.")
        return {
            key: value / self.view_count
            for key, value in self._loss_sums.items()
        }


def recolor_gaussian_ply(
    *,
    source_path: str,
    output_path: str,
    scores: torch.Tensor,
    scale_mode: str = "p99",
    expected_density: torch.Tensor | None = None,
    visibility_threshold: float = 0.0,
) -> dict[str, float]:
    """Replace SH color and hide negligible diagnostic attribution."""
    if not 0.0 <= visibility_threshold < 1.0:
        raise ValueError("visibility_threshold must be in [0, 1).")
    source = PlyData.read(source_path, mmap=True)
    if len(source.elements) != 1 or source.elements[0].name != "vertex":
        raise ValueError("Error splat export requires a vertex-only PLY.")
    vertices = np.array(source.elements[0].data, copy=True)
    if len(vertices) != scores.numel():
        raise ValueError(
            "Attribution count does not match source PLY: "
            f"{scores.numel()} versus {len(vertices)}."
        )
    if expected_density is not None:
        if "opacity" not in (vertices.dtype.names or ()):
            raise ValueError("Source PLY is missing opacity row identity.")
        source_density = np.asarray(vertices["opacity"])
        expected = expected_density.detach().float().cpu().numpy().reshape(-1)
        if source_density.shape != expected.shape or not np.array_equal(
            source_density,
            expected,
        ):
            raise ValueError(
                "Source PLY row order does not match checkpoint density."
            )

    finite_scores = torch.nan_to_num(
        scores.detach().float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    scaled = scale_grad_norms(finite_scores, scale_mode)
    visible = scaled >= visibility_threshold
    rgb = viridis_rgb_from_scalars(scaled).cpu().numpy()
    sh_dc = (rgb - 0.5) / C0
    names = set(vertices.dtype.names or ())
    for channel in range(3):
        field = f"f_dc_{channel}"
        if field not in names:
            raise ValueError(f"Source PLY is missing {field}.")
        vertices[field] = sh_dc[:, channel]
    for field in names:
        if field.startswith("f_rest_"):
            vertices[field] = 0.0
    if visibility_threshold > 0.0:
        if "opacity" not in names:
            raise ValueError(
                "Visibility-gated error splats require an opacity field."
            )
        vertices["opacity"][~visible.numpy()] = -100.0

    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    element = PlyElement.describe(vertices, "vertex")
    PlyData(
        [element],
        text=source.text,
        byte_order=source.byte_order,
        comments=source.comments,
        obj_info=source.obj_info,
    ).write(output_path)
    positive = finite_scores[finite_scores > 0.0]
    return {
        "minimum": float(finite_scores.min()),
        "maximum": float(finite_scores.max()),
        "positive_fraction": float((finite_scores > 0.0).float().mean()),
        "visible_fraction": float(visible.float().mean()),
        "p50_positive": (
            float(torch.quantile(positive, 0.5)) if positive.numel() else 0.0
        ),
        "p95_positive": (
            float(torch.quantile(positive, 0.95)) if positive.numel() else 0.0
        ),
        "p99_positive": (
            float(torch.quantile(positive, 0.99)) if positive.numel() else 0.0
        ),
    }
