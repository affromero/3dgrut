"""Bounded quantile helpers for dense-scanner diagnostics."""

import torch


DIAGNOSTIC_QUANTILE_MAX_SAMPLES = 1_000_000


def bounded_quantile_values(
    value: torch.Tensor,
    *,
    max_samples: int = DIAGNOSTIC_QUANTILE_MAX_SAMPLES,
) -> torch.Tensor:
    """Return a deterministic strided sample small enough for quantiles."""
    if value.numel() <= max_samples:
        return value
    step = (value.numel() + max_samples - 1) // max_samples
    return value.reshape(-1)[::step][:max_samples]


def bounded_quantile(
    value: torch.Tensor,
    quantile: float,
    *,
    max_samples: int = DIAGNOSTIC_QUANTILE_MAX_SAMPLES,
) -> torch.Tensor:
    """Compute an approximate quantile without materializing huge sorts."""
    return torch.quantile(
        bounded_quantile_values(value, max_samples=max_samples),
        quantile,
    )
