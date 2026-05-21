"""Per-particle gradient visualization helpers.

Shared by the training-time GUI, post-training playground GUI, and Viser
web GUI. Maps `model._last_grad_norms[attr]` (shape `[N_gaussians]`) to a
features tensor whose band-0 SH coefficients encode a viridis-colormapped
RGB derived from the gradient magnitudes. Used with
`MixtureOfGaussians.render_diagnostic(... sph_degree_override=0)` so the
rasterizer ignores higher-degree SH bands.
"""

from __future__ import annotations

import torch

from threedgrut.utils.quantile import bounded_quantile


GRAD_RENDER_STYLES: tuple[str, ...] = (
    "grad_positions",
    "grad_rotation",
    "grad_scale",
    "grad_density",
    "grad_features_albedo",
    "grad_features_specular",
)

GRAD_SCALE_MODES: tuple[str, ...] = ("p99", "log", "linear")


def is_grad_style(style: str) -> bool:
    return style.startswith("grad_")


def grad_attr_from_style(style: str) -> str:
    """`grad_positions` -> `positions`."""
    return style[len("grad_") :]


def scale_grad_norms(norms: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize a [N] tensor of non-negative grad magnitudes into [0, 1]."""
    norms = norms.float()
    if mode == "log":
        scaled = torch.log1p(norms)
        return (scaled / scaled.max().clamp_min(1e-9)).clamp(0.0, 1.0)
    if mode == "p99":
        p99 = bounded_quantile(norms, 0.99)
        return (norms / p99.clamp_min(1e-9)).clamp(0.0, 1.0)
    # linear
    return (norms / norms.max().clamp_min(1e-9)).clamp(0.0, 1.0)


def viridis_rgb_from_scalars(scalars_01: torch.Tensor) -> torch.Tensor:
    """Map a [N] tensor in [0, 1] to [N, 3] RGB via matplotlib viridis.

    Falls back to a viridis approximation if matplotlib is unavailable so
    headless deployments do not require pulling matplotlib.
    """
    try:
        from matplotlib import cm

        rgba = cm.viridis(scalars_01.detach().cpu().numpy())  # [N, 4]
        return torch.from_numpy(rgba[..., :3]).to(scalars_01.device, dtype=torch.float32)
    except ImportError:
        # 5-stop viridis approximation
        stops = torch.tensor(
            [
                [0.267, 0.005, 0.329],
                [0.282, 0.140, 0.458],
                [0.254, 0.265, 0.530],
                [0.207, 0.372, 0.553],
                [0.166, 0.474, 0.558],
                [0.128, 0.567, 0.551],
                [0.135, 0.659, 0.518],
                [0.267, 0.749, 0.441],
                [0.478, 0.821, 0.318],
                [0.741, 0.873, 0.150],
                [0.993, 0.906, 0.144],
            ],
            device=scalars_01.device,
            dtype=torch.float32,
        )
        x = scalars_01.clamp(0.0, 1.0) * (stops.shape[0] - 1)
        idx_lo = x.floor().long().clamp(0, stops.shape[0] - 2)
        idx_hi = idx_lo + 1
        t = (x - idx_lo.float()).unsqueeze(-1)
        return stops[idx_lo] * (1.0 - t) + stops[idx_hi] * t


def build_features_override_for_grad(
    model,
    grad_attr: str,
    scale_mode: str,
) -> torch.Tensor | None:
    """Build a features tensor whose band-0 encodes viridis(grad_norm).

    Returns None when the gradient tensor for `grad_attr` is unavailable
    (e.g., before the first backward pass, or in inference-only loading).
    The GUI should display a grey placeholder in that case.
    """
    norms_map = getattr(model, "_last_grad_norms", {}) or {}
    norms = norms_map.get(grad_attr)
    if norms is None:
        return None

    from threedgrut.utils.render import C0

    scaled = scale_grad_norms(norms, scale_mode)  # [N] in [0, 1]
    rgb = viridis_rgb_from_scalars(scaled)        # [N, 3] in [0, 1]
    sh_band0 = (rgb - 0.5) / C0                   # SH band-0 convention

    features_full = torch.zeros_like(model.get_features())
    # Band-0 lives in the first 3 elements when flattened (albedo).
    features_full.view(features_full.shape[0], -1)[:, 0:3] = sh_band0
    return features_full
