"""Fail-closed M2 certificate construction and sealed-result recovery."""

# ruff: noqa: EM101, PLR0913, TRY003

import math
from pathlib import Path

import numpy as np
from doctor_splat.post_training.ghost_repair import (
    GhostRepairPlan,
    paired_bootstrap_lower_bound,
)

from threedgrut.ghost_repair_study import (
    gate_results,
    payload_sha256,
    read_json,
    seal_output_file,
    verify_output_file,
    verify_output_tree,
    write_json_exclusive,
)


def finite_or_none(value: object) -> float | None:
    """Represent unavailable certificate scalars with strict-JSON null."""
    number = float(value)
    return number if math.isfinite(number) else None


def certificate_payload(
    *,
    plan: GhostRepairPlan,
    scene_name: str,
    selection: dict[str, object],
    selection_sha256: str,
    certification_list_sha256: str,
    baseline: dict[str, object],
    selected_result: dict[str, object],
) -> dict[str, object]:
    """Build one certificate with explicit paired-evidence availability."""
    selected_name = str(selection["selected_candidate"])
    baseline_values = baseline["values"]
    selected_values = selected_result["values"]
    gates = gate_results(baseline_values, selected_values, plan)
    before_views = [float(value) for value in baseline["ghost"]["per_view"]]
    after_views = [
        float(value) for value in selected_result["ghost"]["per_view"]
    ]
    if len(before_views) != len(after_views):
        raise ValueError("paired Ghost view arrays must align")
    before_valid = np.isfinite(before_views)
    after_valid = np.isfinite(after_views)
    paired_valid = before_valid & after_valid
    same_evidence = bool(np.array_equal(before_valid, after_valid))
    paired_count = int(paired_valid.sum())
    before_ghost = finite_or_none(baseline_values["ghost"])
    after_ghost = finite_or_none(selected_values["ghost"])
    improvement = (
        before_ghost - after_ghost
        if before_ghost is not None and after_ghost is not None
        else None
    )
    complete_evidence = bool(
        paired_count == len(before_views) == len(after_views)
    )
    lower_bound = None
    if complete_evidence and paired_count >= 2:
        lower_bound = paired_bootstrap_lower_bound(
            before=np.asarray(before_views)[paired_valid].tolist(),
            after=np.asarray(after_views)[paired_valid].tolist(),
            samples=plan.bootstrap_samples,
            seed=plan.bootstrap_seed,
            confidence_level=plan.confidence_level,
        )
    target_floor = max(
        plan.certification.minimum_target_improvement,
        float(selection["calibrated_minimum_ghost_reduction"]),
    )
    accepted = bool(
        selected_name != "baseline"
        and same_evidence
        and complete_evidence
        and paired_count >= 2
        and all(gates.values())
        and improvement is not None
        and improvement >= target_floor
        and lower_bound is not None
        and lower_bound > 0.0
    )
    return {
        "schema_version": 1,
        "scene": scene_name,
        "selection_sha256": selection_sha256,
        "certification_list_sha256": certification_list_sha256,
        "selected_candidate": selected_name,
        "baseline_values": {
            key: finite_or_none(value)
            for key, value in baseline_values.items()
        },
        "selected_values": {
            key: finite_or_none(value)
            for key, value in selected_values.items()
        },
        "metric_gates": {
            metric.value: value for metric, value in gates.items()
        },
        "ghost_improvement": improvement,
        "required_ghost_improvement": target_floor,
        "paired_bootstrap_lower_bound": lower_bound,
        "evidence_availability": {
            "baseline_ghost_views": int(before_valid.sum()),
            "selected_ghost_views": int(after_valid.sum()),
            "paired_ghost_views": paired_count,
            "same_ghost_view_mask": same_evidence,
            "complete_paired_ghost_evidence": complete_evidence,
        },
        "exploratory_random_superiority": selection["random_superiority"],
        "accepted": accepted,
    }


def sealed_certification_baseline(
    *,
    cert_dir: Path,
    selection_sha256: str,
    certification_list_sha256: str,
    certification_names: list[str],
    canonical_train_sha256: str,
    hole_support_sha256: str,
) -> dict[str, object] | None:
    """Recover a completed baseline without recomputing opened evidence."""
    baseline_dir = cert_dir / "baseline"
    result_path = baseline_dir / "result.json"
    if not result_path.exists():
        return None
    verify_output_file(result_path)
    verify_output_tree(baseline_dir / "repair_renders")
    verify_output_tree(baseline_dir / "hole_support_renders")
    marker = read_json(baseline_dir / "input_fingerprint.json")
    if set(marker) != {"schema_version", "fingerprint_sha256", "fingerprint"}:
        raise ValueError("sealed certification marker contract differs")
    fingerprint = marker["fingerprint"]
    if (
        marker["schema_version"] != 1
        or marker["fingerprint_sha256"] != payload_sha256(fingerprint)
        or fingerprint.get("role") != "certification_baseline"
    ):
        raise ValueError("sealed certification marker authentication failed")
    certification_fingerprint = fingerprint.get("scene_study", {})
    if (
        certification_fingerprint.get("role") != "certification"
        or certification_fingerprint.get("selection_sha256")
        != selection_sha256
        or certification_fingerprint.get("certification_list_sha256")
        != certification_list_sha256
    ):
        raise ValueError("sealed certification baseline identity differs")
    result = read_json(result_path)
    if result["surface"]["rendered_view_names"] != certification_names:
        raise ValueError("sealed certification view order differs")
    if (
        result["surface"]["teacher_fgd"]["teacher_train_names_sha256"]
        != canonical_train_sha256
        or result["hole"]["training_names_sha256"] != hole_support_sha256
    ):
        raise ValueError("sealed certification support identity differs")
    return result


def sealed_scene_certificate(
    *,
    path: Path,
    scene_name: str,
    selection_sha256: str,
    certification_list_sha256: str,
) -> dict[str, object] | None:
    """Return one sealed certificate only when all frozen identities match."""
    if not path.is_file():
        return None
    verify_output_file(path)
    certificate = read_json(path)
    if (
        certificate.get("scene") != scene_name
        or certificate.get("selection_sha256") != selection_sha256
        or certificate.get("certification_list_sha256")
        != certification_list_sha256
    ):
        raise ValueError("sealed certificate identity differs")
    return certificate


def seal_scene_certificate(
    *,
    path: Path,
    plan: GhostRepairPlan,
    scene_name: str,
    selection: dict[str, object],
    selection_sha256: str,
    certification_list_sha256: str,
    baseline: dict[str, object],
    selected_result: dict[str, object],
) -> dict[str, object]:
    """Write and seal one canonical certificate payload."""
    certificate = certificate_payload(
        plan=plan,
        scene_name=scene_name,
        selection=selection,
        selection_sha256=selection_sha256,
        certification_list_sha256=certification_list_sha256,
        baseline=baseline,
        selected_result=selected_result,
    )
    write_json_exclusive(path, certificate)
    seal_output_file(path)
    return certificate
