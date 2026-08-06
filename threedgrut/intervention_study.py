# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Matched parameter interventions for diagnostic-field validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

import numpy as np
import torch

from threedgrut.error_attribution import (
    ErrorAttributionMetric,
    attribution_loss,
)
from threedgrut.utils.logger import logger


class RankingDirection(StrEnum):
    """Ordering used to select a field's top-risk cohort."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


class InterventionMode(StrEnum):
    """Supported deterministic parameter perturbations."""

    ADDITIVE_ABSOLUTE = "additive_absolute"
    ADDITIVE_RELATIVE_SCALE = "additive_relative_scale"


def counterfactual_cohorts(
    scores: torch.Tensor,
    cohort_size: int,
) -> dict[str, torch.Tensor]:
    """Return deterministic top, random, and low-score legacy controls."""
    if cohort_size <= 0:
        raise ValueError("counterfactual cohort size must be positive")
    flattened = _finite_scores(scores)
    if flattened.numel() == 0:
        raise ValueError("counterfactual scores are empty")
    count = min(cohort_size, flattened.numel())
    ordering = torch.argsort(flattened, stable=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    return {
        "top_density_sensitivity": ordering[-count:],
        "random_control": torch.randperm(
            flattened.numel(),
            generator=generator,
        )[:count],
        "low_density_sensitivity": ordering[:count],
    }


@contextmanager
def suppressed_density_cohort(
    model: torch.nn.Module,
    indices: torch.Tensor,
    suppression_logit: float,
) -> Iterator[None]:
    """Temporarily remove a cohort from native alpha compositing."""
    density = _model_parameter(model, "density")
    selected = indices.to(device=density.device, dtype=torch.long)
    with torch.no_grad():
        original = density.index_select(0, selected).clone()
        density.index_fill_(0, selected, suppression_logit)
    try:
        yield
    finally:
        with torch.no_grad():
            density.index_copy_(0, selected, original)


def evaluate_density_counterfactuals(
    *,
    model: torch.nn.Module,
    renderer: object,
    selected_indices: set[int],
    density_scores: torch.Tensor,
    cohort_size: int,
    suppression_logit: float,
) -> dict[str, object]:
    """Measure legacy density-suppression effects on held-out RGB MSE."""
    batches = load_certification_batches(
        renderer=renderer,
        selected_indices=selected_indices,
    )
    baseline_values = per_view_mse(model=model, batches=batches)
    baseline_mse = float(np.mean([value for _, value in baseline_values]))
    cohorts = counterfactual_cohorts(density_scores, cohort_size)
    evaluations: list[dict[str, object]] = []
    for name, indices in cohorts.items():
        with suppressed_density_cohort(
            model,
            indices,
            suppression_logit,
        ):
            values = per_view_mse(model=model, batches=batches)
        mse = float(np.mean([value for _, value in values]))
        evaluations.append(
            {
                "cohort": name,
                "gaussian_count": int(indices.numel()),
                "heldout_mse": mse,
                "delta_mse": mse - baseline_mse,
                "absolute_delta_mse": abs(mse - baseline_mse),
            }
        )
    return {
        "method": "native density-logit suppression and held-out rerender",
        "score": "mse:density RMS spatial loss-field sensitivity",
        "baseline_heldout_mse": baseline_mse,
        "suppression_logit": suppression_logit,
        "cohorts": evaluations,
        "interpretation": (
            "This is an intervention-effect comparison, not a causal label. "
            "The sensitivity ranks local effect magnitude; delta sign comes "
            "only from the actual rerender."
        ),
    }


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_names(path: str) -> set[str]:
    with open(path, encoding="utf-8") as handle:
        names = {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }
    if not names:
        raise ValueError(f"image list is empty: {path}")
    return names


def _finite_scores(scores: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        scores.detach().reshape(-1).float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def matched_cohorts_v1(
    *,
    scores: torch.Tensor,
    residual_scores: torch.Tensor,
    cohort_size: int,
    ranking: RankingDirection,
    random_seeds: tuple[int, ...],
) -> list[tuple[str, int, torch.Tensor]]:
    """Reproduce the exact overlapping controls used by M1 at b5b68ae."""
    if cohort_size <= 0:
        raise ValueError("cohort size must be positive")
    if not random_seeds or len(random_seeds) != len(set(random_seeds)):
        raise ValueError("random seeds must be present and unique")
    values = _finite_scores(scores)
    residual = _finite_scores(residual_scores)
    if values.numel() == 0 or values.shape != residual.shape:
        raise ValueError(
            "field and residual scores must have equal nonzero size"
        )
    count = min(cohort_size, values.numel())
    order = torch.argsort(values, stable=True)
    if ranking is RankingDirection.DESCENDING:
        top = order[-count:]
        bottom = order[:count]
    else:
        top = order[:count]
        bottom = order[-count:]
    residual_order = torch.argsort(residual, stable=True)
    residual_eligible = torch.ones(values.numel(), dtype=torch.bool)
    residual_eligible[top] = False
    residual_control = residual_order[residual_eligible[residual_order]]
    if residual_control.numel() < count:
        raise ValueError("not enough rows for a disjoint residual-only cohort")
    cohorts = [
        ("top_risk", 0, top),
        ("bottom_risk", 0, bottom),
        ("residual_only", 0, residual_control[-count:]),
    ]
    for replicate, seed in enumerate(random_seeds):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        cohorts.append(
            (
                "random",
                replicate,
                torch.randperm(values.numel(), generator=generator)[:count],
            )
        )
    return cohorts


def matched_cohorts(
    *,
    scores: torch.Tensor,
    residual_scores: torch.Tensor,
    cohort_size: int,
    ranking: RankingDirection,
    random_seeds: tuple[int, ...],
) -> list[tuple[str, int, torch.Tensor]]:
    """Return treatment-disjoint controls for confirmatory studies."""
    if cohort_size <= 0:
        raise ValueError("cohort size must be positive")
    if not random_seeds or len(random_seeds) != len(set(random_seeds)):
        raise ValueError("random seeds must be present and unique")
    values = _finite_scores(scores)
    residual = _finite_scores(residual_scores)
    if values.numel() == 0 or values.shape != residual.shape:
        raise ValueError(
            "field and residual scores must have equal nonzero size"
        )
    count = min(cohort_size, values.numel())
    if 4 * count > values.numel():
        raise ValueError(
            "confirmatory cohorts require four disjoint cohort-sized pools"
        )
    order = torch.argsort(values, stable=True)
    if ranking is RankingDirection.DESCENDING:
        top, bottom = order[-count:], order[:count]
    else:
        top, bottom = order[:count], order[-count:]
    eligible = torch.ones(values.numel(), dtype=torch.bool)
    eligible[top] = False
    eligible[bottom] = False
    residual_order = torch.argsort(residual, stable=True)
    residual_control = residual_order[eligible[residual_order]][-count:]
    eligible[residual_control] = False
    random_pool = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if random_pool.numel() < count:
        raise ValueError("not enough rows for treatment-disjoint controls")
    cohorts = [
        ("top_risk", 0, top),
        ("bottom_risk", 0, bottom),
        ("residual_only", 0, residual_control),
    ]
    for replicate, seed in enumerate(random_seeds):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        selection = torch.randperm(random_pool.numel(), generator=generator)[
            :count
        ]
        cohorts.append(("random", replicate, random_pool[selection]))
    return cohorts


def _index_sha256(indices: torch.Tensor) -> str:
    values = indices.detach().cpu().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def generate_intervention_provenance(
    *,
    output_dir: str,
    plan_path: str,
    result_path: str,
    manifest_path: str,
    renderer_commit: str,
    doctor_commit: str,
    runner_sha256: str,
) -> dict[str, object]:
    """Authenticate an M1 result and its deterministic cohort selection."""
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    with open(result_path, encoding="utf-8") as handle:
        result = json.load(handle)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if result["plan_sha256"] != _sha256(plan_path):
        raise ValueError("result plan hash differs from the supplied plan")
    if result["checkpoint_sha256"] != manifest["source_checkpoint_sha256"]:
        raise ValueError("result and manifest checkpoint hashes differ")
    selection_version = plan.get("cohort_selection_version", "m1_legacy_v1")
    if selection_version != "m1_legacy_v1":
        raise ValueError("M1 provenance requires m1_legacy_v1 selection")
    raw_field_ids = {
        str(field["source_field_id"]) for field in plan["fields"]
    } | {"residual_mse_exposure"}
    raw_fields: dict[str, object] = {}
    for field_id in sorted(raw_field_ids):
        path = os.path.join(output_dir, "raw", f"{field_id}.npy")
        values = np.load(path, mmap_mode="r")
        raw_fields[field_id] = {
            "path": os.path.relpath(path, output_dir),
            "sha256": _sha256(path),
            "dtype": str(values.dtype),
            "shape": list(values.shape),
        }
    residual = _load_scores(output_dir, "residual_mse_exposure")
    cohort_hashes: list[dict[str, object]] = []
    seeds = tuple(int(seed) for seed in plan["random_seeds"])
    for field in plan["fields"]:
        scores = _load_scores(output_dir, str(field["source_field_id"]))
        for cohort, replicate, indices in matched_cohorts_v1(
            scores=scores,
            residual_scores=residual,
            cohort_size=int(field["cohort_size"]),
            ranking=RankingDirection(str(field["ranking"])),
            random_seeds=seeds,
        ):
            cohort_hashes.append(
                {
                    "field": field["field"],
                    "cohort": cohort,
                    "replicate": replicate,
                    "count": int(indices.numel()),
                    "ordered_index_sha256": _index_sha256(indices),
                }
            )
    return {
        "schema_version": 1,
        "study_id": plan["study_id"],
        "scene": result["scene"],
        "renderer_commit": renderer_commit,
        "doctor_commit": doctor_commit,
        "runner_sha256": runner_sha256,
        "plan_sha256": _sha256(plan_path),
        "result_sha256": _sha256(result_path),
        "manifest_sha256": _sha256(manifest_path),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "repair_list_sha256": result["repair_list_sha256"],
        "certification_list_sha256": result["certification_list_sha256"],
        "cohort_selection_version": selection_version,
        "direction_seed": 1701,
        "raw_fields": raw_fields,
        "cohort_index_hashes": cohort_hashes,
        "runtime": {
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "interpretation": (
            "Hashes reproduce the exact M1 selections and inputs. M1 random "
            "and residual controls may overlap other deterministic cohorts; "
            "confirmatory protocols use treatment-disjoint controls."
        ),
    }


def _model_parameter(
    model: torch.nn.Module,
    parameter_name: str,
) -> torch.nn.Parameter:
    parameter = getattr(model, parameter_name, None)
    if not isinstance(parameter, torch.nn.Parameter):
        raise TypeError(f"model parameter {parameter_name!r} is unavailable")
    return parameter


def _relative_reference_scale(
    model: torch.nn.Module,
    parameter_name: str,
    parameter: torch.nn.Parameter,
) -> torch.Tensor:
    if parameter_name == "positions":
        get_scale = getattr(model, "get_scale", None)
        if not callable(get_scale):
            raise TypeError("position intervention requires model.get_scale")
        physical_scale = get_scale().detach().float().amax(dim=1)
        return physical_scale.median().clamp_min(1e-12)
    return parameter.detach().float().square().mean().sqrt().clamp_min(1e-12)


def _row_directions(
    *,
    indices: torch.Tensor,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    rows = indices.to(device=device, dtype=torch.int64).reshape(-1, 1)
    columns = torch.arange(width, device=device, dtype=torch.int64).reshape(
        1, -1
    )
    parity = (rows * 1103515245 + columns * 12345 + seed) % 2
    return parity.to(dtype=dtype).mul_(2.0).sub_(1.0)


@contextmanager
def perturbed_parameter_rows(
    *,
    model: torch.nn.Module,
    indices: torch.Tensor,
    parameter_name: str,
    mode: InterventionMode,
    magnitude: float,
    direction_seed: int,
) -> Iterator[None]:
    """Apply a bounded row intervention and restore the checkpoint exactly."""
    if magnitude == 0.0 or not math.isfinite(magnitude):
        raise ValueError("intervention magnitude must be finite and nonzero")
    parameter = _model_parameter(model, parameter_name)
    selected = indices.to(device=parameter.device, dtype=torch.long)
    with torch.no_grad():
        original = parameter.index_select(0, selected).clone()
        if mode is InterventionMode.ADDITIVE_ABSOLUTE:
            delta = torch.full_like(original, magnitude)
        else:
            reference = _relative_reference_scale(
                model,
                parameter_name,
                parameter,
            )
            directions = _row_directions(
                indices=selected,
                width=original.reshape(original.shape[0], -1).shape[1],
                device=parameter.device,
                dtype=parameter.dtype,
                seed=direction_seed,
            ).reshape_as(original)
            delta = (
                directions * reference.to(dtype=parameter.dtype) * magnitude
            )
        parameter.index_copy_(0, selected, original + delta)
    try:
        yield
    finally:
        with torch.no_grad():
            parameter.index_copy_(0, selected, original)


def load_certification_batches(
    *,
    renderer: object,
    selected_indices: set[int],
) -> list[tuple[str, object]]:
    """Transfer each fixed certification batch to the GPU exactly once."""
    dataset = getattr(renderer, "dataset", None)
    dataloader = getattr(renderer, "dataloader", None)
    if dataset is None or dataloader is None:
        raise TypeError(
            "intervention renderer must expose dataset and dataloader"
        )
    batches: list[tuple[str, object]] = []
    for index, batch in enumerate(dataloader):
        if index not in selected_indices:
            continue
        gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
        batches.append(
            (os.path.basename(str(gpu_batch.image_path)), gpu_batch)
        )
    if not batches:
        raise ValueError("no certification views were selected")
    return batches


def per_view_mse(
    *,
    model: torch.nn.Module,
    batches: list[tuple[str, object]],
) -> list[tuple[str, float]]:
    """Rerender cached certification batches and retain each RGB MSE."""
    values: list[tuple[str, float]] = []
    with torch.no_grad():
        for image_name, gpu_batch in batches:
            outputs = model(gpu_batch, train=False)
            prediction = outputs.get("pred_rgb")
            if not isinstance(prediction, torch.Tensor):
                raise KeyError("renderer output has no pred_rgb")
            mse = attribution_loss(
                ErrorAttributionMetric.MSE,
                prediction,
                gpu_batch.rgb_gt,
                gpu_batch.mask,
            )
            values.append((image_name, float(mse)))
    return values


def _load_scores(output_dir: str, field_id: str) -> torch.Tensor:
    path = os.path.join(output_dir, "raw", f"{field_id}.npy")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"preregistered field is unavailable: {field_id}"
        )
    return torch.from_numpy(np.load(path).astype(np.float32, copy=False))


def run_intervention_study(
    *,
    plan_path: str,
    scene_id: str,
    output_dir: str,
    model: torch.nn.Module,
    certification_renderer: object,
    selected_indices: set[int],
    repair_list_path: str,
    certification_list_path: str,
    checkpoint_sha256: str,
) -> dict[str, object]:
    """Execute the frozen field study on a disjoint certification split."""
    with open(plan_path, encoding="utf-8") as handle:
        plan = json.load(handle)
    if scene_id not in plan["scenes"]:
        raise ValueError(f"scene {scene_id!r} is absent from the study plan")
    overlap = _image_names(repair_list_path) & _image_names(
        certification_list_path
    )
    if overlap:
        raise ValueError(
            "repair and certification image lists overlap: "
            f"{sorted(overlap)[:5]}"
        )
    random_seeds = tuple(int(value) for value in plan["random_seeds"])
    certification_batches = load_certification_batches(
        renderer=certification_renderer,
        selected_indices=selected_indices,
    )
    baseline_views = per_view_mse(
        model=model,
        batches=certification_batches,
    )
    baseline_loss = float(np.mean([value for _, value in baseline_views]))
    residual_scores = _load_scores(output_dir, "residual_mse_exposure")
    outcomes: list[dict[str, object]] = []
    intervention_count = 0
    total_interventions = len(plan["fields"]) * (3 + len(random_seeds))
    for field in plan["fields"]:
        scores = _load_scores(output_dir, str(field["source_field_id"]))
        selection_version = plan.get(
            "cohort_selection_version", "m1_legacy_v1"
        )
        cohort_selector = {
            "m1_legacy_v1": matched_cohorts_v1,
            "disjoint_v2": matched_cohorts,
        }.get(selection_version)
        if cohort_selector is None:
            raise ValueError(
                f"unknown cohort selection version: {selection_version}"
            )
        cohorts = cohort_selector(
            scores=scores,
            residual_scores=residual_scores,
            cohort_size=int(field["cohort_size"]),
            ranking=RankingDirection(str(field["ranking"])),
            random_seeds=random_seeds,
        )
        for cohort, replicate, indices in cohorts:
            with perturbed_parameter_rows(
                model=model,
                indices=indices,
                parameter_name=str(field["parameter"]),
                mode=InterventionMode(str(field["mode"])),
                magnitude=float(field["magnitude"]),
                direction_seed=1701,
            ):
                changed_views = per_view_mse(
                    model=model,
                    batches=certification_batches,
                )
            intervention_count += 1
            logger.info(
                "Matched intervention "
                f"{intervention_count}/{total_interventions}: "
                f"{field['field']} / {cohort} / {replicate}"
            )
            changed_loss = float(
                np.mean([value for _, value in changed_views])
            )
            signed_delta = changed_loss - baseline_loss
            outcomes.append(
                {
                    "scene": scene_id,
                    "field": field["field"],
                    "cohort": cohort,
                    "replicate": replicate,
                    "gaussian_count": int(indices.numel()),
                    "baseline_loss": baseline_loss,
                    "intervened_loss": changed_loss,
                    "signed_delta_loss": signed_delta,
                    "absolute_delta_loss": abs(signed_delta),
                    "requested_views": len(selected_indices),
                    "valid_views": len(changed_views),
                    "per_view": [
                        {
                            "image": image,
                            "baseline_loss": baseline,
                            "intervened_loss": changed,
                            "signed_delta_loss": changed - baseline,
                        }
                        for (image, baseline), (_, changed) in zip(
                            baseline_views,
                            changed_views,
                            strict=True,
                        )
                    ],
                }
            )
    return {
        "schema_version": 1,
        "study_id": plan["study_id"],
        "scene": scene_id,
        "plan_sha256": _sha256(plan_path),
        "checkpoint_sha256": checkpoint_sha256,
        "repair_list_sha256": _sha256(repair_list_path),
        "certification_list_sha256": _sha256(certification_list_path),
        "outcome_metric": plan["outcome_metric"],
        "baseline_per_view": [
            {"image": image, "loss": value} for image, value in baseline_views
        ],
        "outcomes": outcomes,
        "interpretation": plan["interpretation"],
    }
