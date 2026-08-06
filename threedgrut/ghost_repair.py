"""Gaussian-row operations for preregistered Ghost-directed repair."""

import copy
import hashlib

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict


class SampledGhostEvidence(BaseModel):
    """Aligned samples produced by Doctor Splat's Ghost instrument."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    us: np.ndarray
    vs: np.ndarray
    ghost: np.ndarray
    evidenced: np.ndarray


def sampled_contradiction_field(
    *,
    image_shape: tuple[int, int],
    samples: SampledGhostEvidence,
    device: torch.device,
) -> torch.Tensor:
    """Rasterize sampled Ghost indicators, preserving duplicate counts."""
    height, width = image_shape
    us, vs = samples.us, samples.vs
    ghost, evidenced = samples.ghost, samples.evidenced
    if not (us.shape == vs.shape == ghost.shape == evidenced.shape):
        message = "sampled Ghost arrays must align"
        raise ValueError(message)
    if us.ndim != 1:
        message = "sampled Ghost arrays must be one-dimensional"
        raise ValueError(message)
    if (
        np.any(us < 0)
        or np.any(us >= width)
        or np.any(vs < 0)
        or np.any(vs >= height)
    ):
        message = "sampled Ghost coordinate is outside the image"
        raise ValueError(message)
    counts = np.zeros((height, width), dtype=np.float32)
    selected = evidenced & ghost
    np.add.at(counts, (vs[selected], us[selected]), 1.0)
    return torch.from_numpy(counts).to(device=device)


def stable_largest_indices(
    values: np.ndarray,
    count: int,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Return a deterministic descending ranking with row-index tie breaks."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    if eligible is None:
        eligible = np.ones(scores.size, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    if eligible.shape != scores.shape:
        message = "eligibility mask must match score rows"
        raise ValueError(message)
    candidates = np.flatnonzero(eligible & np.isfinite(scores))
    if count <= 0 or count > candidates.size:
        message = "requested cohort does not fit eligible rows"
        raise ValueError(message)
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order[:count]]


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Map finite values to ascending empirical midranks."""
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    ranks = np.zeros(scores.size, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        return ranks
    order = np.argsort(scores[finite], kind="stable")
    sorted_rows = finite[order]
    if sorted_rows.size == 1:
        ranks[sorted_rows] = 1.0
    else:
        sorted_scores = scores[sorted_rows]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
        ends = np.r_[starts[1:], sorted_rows.size]
        denominator = sorted_rows.size - 1
        for start, end in zip(starts, ends, strict=True):
            midrank = 0.5 * (start + end - 1) / denominator
            ranks[sorted_rows[start:end]] = midrank
    return ranks


def repair_rankings(
    *,
    density: np.ndarray,
    ghost_responsibility: np.ndarray,
    opacity_sensitivity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the four deterministic M2 risk rankings."""
    if not (
        density.shape
        == ghost_responsibility.shape
        == opacity_sensitivity.shape
    ):
        message = "M2 row fields must align"
        raise ValueError(message)
    ghost_rank = percentile_ranks(ghost_responsibility)
    sensitivity_rank = percentile_ranks(opacity_sensitivity)
    return {
        "global_opacity": -np.asarray(density, dtype=np.float64),
        "ghost_responsibility": np.asarray(
            ghost_responsibility,
            dtype=np.float64,
        ),
        "opacity_sensitivity": np.asarray(
            opacity_sensitivity,
            dtype=np.float64,
        ),
        "ghost_opacity_combination": 0.5 * (ghost_rank + sensitivity_rank),
    }


def matched_pruning_cohorts(
    *,
    rankings: dict[str, np.ndarray],
    count: int,
    random_replicates: int,
    seed: int,
    eligible: np.ndarray | None = None,
) -> dict[str, list[np.ndarray]]:
    """Create deterministic arms and mutually disjoint random controls."""
    if set(rankings) != {
        "global_opacity",
        "ghost_responsibility",
        "opacity_sensitivity",
        "ghost_opacity_combination",
    }:
        message = "M2 deterministic ranking set is incomplete"
        raise ValueError(message)
    row_count = next(iter(rankings.values())).size
    if any(values.size != row_count for values in rankings.values()):
        message = "M2 ranking row counts differ"
        raise ValueError(message)
    if eligible is None:
        eligible = np.ones(row_count, dtype=bool)
    deterministic = {
        arm: stable_largest_indices(values, count, eligible=eligible)
        for arm, values in rankings.items()
    }
    pool = np.flatnonzero(np.asarray(eligible, dtype=bool))
    required = count * random_replicates
    if required > pool.size:
        message = "disjoint random controls exceed the eligible pool"
        raise ValueError(message)
    rng = np.random.default_rng(seed)
    draw = rng.choice(pool, size=required, replace=False)
    cohorts = {arm: [indices] for arm, indices in deterministic.items()}
    cohorts["random"] = [
        draw[start : start + count] for start in range(0, required, count)
    ]
    return cohorts


def ordered_indices_sha256(indices: np.ndarray) -> str:
    """Hash an ordered little-endian int64 cohort."""
    canonical = np.asarray(indices, dtype="<i8").reshape(-1)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def pruned_checkpoint(
    checkpoint: dict[str, object],
    removed_indices: np.ndarray,
) -> dict[str, object]:
    """Return an evaluation checkpoint with selected Gaussian rows removed."""
    positions = checkpoint.get("positions")
    if not torch.is_tensor(positions) or positions.ndim != 2:
        message = "checkpoint lacks row-aligned Gaussian positions"
        raise ValueError(message)
    row_count = positions.shape[0]
    removed = np.asarray(removed_indices, dtype=np.int64).reshape(-1)
    if removed.size == 0 or np.unique(removed).size != removed.size:
        message = "removed rows must be a non-empty unique cohort"
        raise ValueError(message)
    if np.any(removed < 0) or np.any(removed >= row_count):
        message = "removed row is outside the checkpoint"
        raise ValueError(message)
    for key in (
        "acquisition_appearance",
        "acquisition_visibility",
        "surface_acquisition_spline",
        "gaussian_track_acquisition",
    ):
        if checkpoint.get(key) is not None:
            message = f"M2 does not admit active {key} state"
            raise ValueError(message)
    keep = torch.ones(row_count, dtype=torch.bool, device=positions.device)
    keep[torch.as_tensor(removed, device=positions.device)] = False
    output = copy.copy(checkpoint)
    filtered_keys: list[str] = []
    for key, value in checkpoint.items():
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and value.shape[0] == row_count
        ):
            filtered = value[keep].contiguous()
            if isinstance(value, torch.nn.Parameter):
                filtered = torch.nn.Parameter(
                    filtered,
                    requires_grad=value.requires_grad,
                )
            output[key] = filtered
            filtered_keys.append(key)
    required = {
        "positions",
        "rotation",
        "scale",
        "density",
        "features_albedo",
        "features_specular",
    }
    if not required.issubset(filtered_keys):
        message = "checkpoint is missing required row-aligned fields"
        raise ValueError(message)
    output.pop("optimizer", None)
    output["post_training_prune"] = {
        "source_row_count": row_count,
        "removed_row_count": int(removed.size),
        "removed_ordered_index_sha256": ordered_indices_sha256(removed),
        "filtered_keys": sorted(filtered_keys),
    }
    return output
