#!/usr/bin/env python3
# ruff: noqa: EM101, EM102, PLR0913, TRY003
"""Run preregistered M2 Ghost repair development or certification."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from doctor_splat.dump_val_depth import render_val
from doctor_splat.ghost_mass import arm_ghost_mass
from doctor_splat.post_training.ghost_repair import (
    DevelopmentOutcome,
    GhostRepairArm,
    GhostRepairPlan,
    calibrated_ghost_floor,
    method_level_passes,
    paired_bootstrap_lower_bound,
    random_superiority,
    select_development_candidate,
)
from doctor_splat.split_contract import load_explicit_image_list
from threedgrut.ghost_repair import (
    matched_pruning_cohorts,
    ordered_indices_sha256,
    repair_rankings,
)
from threedgrut.ghost_repair_study import (
    bind_artifact_directory,
    candidate_id,
    cohort_overlaps,
    evaluate_candidate,
    file_sha256,
    gate_results,
    ghost_responsibility,
    model_from_pruned_checkpoint,
    read_json,
    release_model,
    render_candidate,
    resolve_registered_path,
    save_cohort,
    seal_output_file,
    seal_output_tree,
    validate_runtime_contract,
    verify_output_file,
    verify_output_tree,
    write_json,
    write_json_exclusive,
)
from threedgrut.post_training.provenance import (
    artifact_fingerprint,
    scene_fingerprint,
)


def parse_args() -> argparse.Namespace:
    """Parse the phase-separated study command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("validate", "develop", "certify"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--doctor-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scene", action="append", default=[])
    return parser.parse_args()


def load_plan(path: Path) -> GhostRepairPlan:
    """Load the immutable preregistration."""
    return GhostRepairPlan.model_validate_json(
        path.read_text(encoding="utf-8")
    )


def field_from_manifest(
    manifest: Path,
    field_id: str,
    *,
    expected_checkpoint_sha256: str,
    expected_images: list[str],
) -> np.ndarray:
    """Load and authenticate one row-aligned diagnostic sidecar."""
    payload = read_json(manifest)
    expected = {
        "source_checkpoint_sha256": expected_checkpoint_sha256,
        "selected_images": expected_images,
        "split": "val",
        "post_processing_mode": "raw",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"diagnostic manifest {key} mismatch")
    matches = [field for field in payload["fields"] if field["id"] == field_id]
    if len(matches) != 1:
        raise ValueError(f"diagnostic field must occur once: {field_id}")
    field = matches[0]
    path = manifest.parent / field["raw_filename"]
    if file_sha256(path) != field["raw_sha256"]:
        raise ValueError(f"diagnostic field hash mismatch: {field_id}")
    return np.load(path, allow_pickle=False).reshape(-1)


def scene_paths(
    scene: object,
    *,
    scene_root: Path,
    doctor_root: Path,
) -> dict[str, Path]:
    """Resolve every registered development input except certification."""
    return {
        key: resolve_registered_path(
            getattr(scene, key),
            scene_root=scene_root,
            doctor_root=doctor_root,
        )
        for key in (
            "checkpoint",
            "repair_list",
            "canonical_train_list",
            "hole_support_list",
            "diagnostic_manifest",
        )
    }


def baseline_result(
    *,
    checkpoint: Path,
    bundle: Path,
    repair: object,
    hole_support: object,
    canonical_train: object,
    canonical_train_path: Path,
    output_dir: Path,
    neighbors: int,
    fingerprint: object,
) -> dict[str, object]:
    """Create or resume the development baseline."""
    bind_artifact_directory(output_dir, fingerprint)
    result_path = output_dir / "result.json"
    if result_path.is_file():
        verify_output_file(result_path)
        verify_output_tree(output_dir / "repair_renders")
        verify_output_tree(output_dir / "hole_support_renders")
        return read_json(result_path)
    repair_renders, support_renders = render_candidate(
        source_checkpoint=checkpoint,
        bundle=bundle,
        repair=repair,
        hole_support=hole_support,
        output_dir=output_dir,
        model=None,
        fingerprint=fingerprint,
    )
    result = evaluate_candidate(
        repair_renders=repair_renders,
        support_renders=support_renders,
        bundle=bundle,
        canonical_train_path=canonical_train_path,
        repair=repair,
        canonical_train=canonical_train,
        neighbors=neighbors,
    )
    write_json(result_path, result)
    seal_output_file(result_path)
    return result


def noop_ghost_results(
    *,
    checkpoint: Path,
    bundle: Path,
    repair: object,
    output_dir: Path,
    repeats: int,
    neighbors: int,
    base_fingerprint: dict[str, object],
) -> list[dict[str, object]]:
    """Measure repeated baseline rendering noise on the target contract."""
    results: list[dict[str, object]] = []
    for replicate in range(repeats):
        repeat_dir = output_dir / f"noop_{replicate:02d}" / "repair_renders"
        result_path = repeat_dir.parent / "result.json"
        noop_fingerprint = artifact_fingerprint(
            base_fingerprint, "noop", replicate=replicate
        )
        bind_artifact_directory(
            repeat_dir.parent,
            noop_fingerprint,
        )
        if result_path.is_file():
            verify_output_file(result_path)
            verify_output_tree(repeat_dir)
            results.append(read_json(result_path))
            continue
        bind_artifact_directory(
            repeat_dir,
            {"noop": noop_fingerprint, "images": list(repair.names)},
        )
        if (repeat_dir / "meta.json").is_file():
            verify_output_tree(repeat_dir)
        else:
            partial = [
                path
                for path in repeat_dir.iterdir()
                if path.name != "input_fingerprint.json"
            ]
            if partial:
                raise ValueError("partial no-op render requires recovery")
            render_val(
                str(checkpoint),
                str(bundle),
                str(repeat_dir),
                repair,
            )
            seal_output_tree(repeat_dir)
        ghost = arm_ghost_mass(
            str(repeat_dir),
            str(bundle / "sparse" / "0"),
            k=neighbors,
        )
        result = {"ghost": ghost.model_dump(mode="json")}
        write_json(result_path, result)
        seal_output_file(result_path)
        results.append(result)
    return results


def evaluate_intervention(
    *,
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    bundle: Path,
    repair: object,
    hole_support: object,
    canonical_train: object,
    canonical_train_path: Path,
    output_dir: Path,
    indices: np.ndarray,
    cohort: dict[str, object],
    neighbors: int,
    fingerprint: object,
) -> dict[str, object]:
    """Physically remove one cohort, render it, and evaluate seven axes."""
    bind_artifact_directory(output_dir, fingerprint)
    result_path = output_dir / "result.json"
    if result_path.is_file():
        verify_output_file(result_path)
        verify_output_tree(output_dir / "repair_renders")
        verify_output_tree(output_dir / "hole_support_renders")
        existing = read_json(result_path)
        if (
            existing.get("cohort", {}).get("ordered_index_sha256")
            != cohort["ordered_index_sha256"]
        ):
            raise ValueError("completed candidate does not match saved cohort")
        return existing
    model = None
    try:
        model, prune = model_from_pruned_checkpoint(checkpoint, indices)
        repair_renders, support_renders = render_candidate(
            source_checkpoint=checkpoint_path,
            bundle=bundle,
            repair=repair,
            hole_support=hole_support,
            output_dir=output_dir,
            model=model,
            fingerprint=fingerprint,
        )
    finally:
        del model
        release_model(None)
    result = evaluate_candidate(
        repair_renders=repair_renders,
        support_renders=support_renders,
        bundle=bundle,
        canonical_train_path=canonical_train_path,
        repair=repair,
        canonical_train=canonical_train,
        neighbors=neighbors,
    )
    result["cohort"] = cohort
    result["physical_prune"] = prune
    write_json(result_path, result)
    seal_output_file(result_path)
    return result


def develop_scene(
    *,
    plan: GhostRepairPlan,
    scene: object,
    scene_index: int,
    plan_path: Path,
    scene_root: Path,
    doctor_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Run all registered repair-split candidates for one scene."""
    selection_path = output_root / scene.scene / "selection.json"
    opened_path = output_root / scene.scene / "certification_opened.json"
    if opened_path.is_file():
        if not selection_path.is_file():
            raise ValueError("certification marker exists without selection")
        verify_output_file(selection_path)
        opened = read_json(opened_path)
        if opened.get("selection_sha256") != file_sha256(selection_path):
            raise ValueError("opened certification selection has changed")
        return read_json(selection_path)
    paths = scene_paths(scene, scene_root=scene_root, doctor_root=doctor_root)
    checkpoint_path = paths["checkpoint"]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    bundle = Path(str(checkpoint["config"].path)).resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"checkpoint bundle does not exist: {bundle}")
    repair = load_explicit_image_list(str(paths["repair_list"]), "M2 repair")
    canonical_train = load_explicit_image_list(
        str(paths["canonical_train_list"]), "M2 canonical training"
    )
    hole_support = load_explicit_image_list(
        str(paths["hole_support_list"]), "M2 Hole support"
    )
    base_fingerprint = scene_fingerprint(
        plan=plan,
        scene=scene,
        plan_path=plan_path,
        paths=paths,
        doctor_root=doctor_root,
        bundle=bundle,
        image_names=sorted(
            set(repair.names)
            | set(canonical_train.names)
            | set(hole_support.names)
        ),
    )
    scene_dir = output_root / scene.scene / "development"
    baseline = baseline_result(
        checkpoint=checkpoint_path,
        bundle=bundle,
        repair=repair,
        hole_support=hole_support,
        canonical_train=canonical_train,
        canonical_train_path=paths["canonical_train_list"],
        output_dir=scene_dir / "baseline",
        neighbors=plan.ghost_neighbors,
        fingerprint=artifact_fingerprint(
            base_fingerprint, "development_baseline"
        ),
    )
    noops = noop_ghost_results(
        checkpoint=checkpoint_path,
        bundle=bundle,
        repair=repair,
        output_dir=scene_dir,
        repeats=plan.noop_repeats,
        neighbors=plan.ghost_neighbors,
        base_fingerprint=base_fingerprint,
    )
    baseline_ghost = float(baseline["values"]["ghost"])
    noop_deltas = [
        baseline_ghost - float(item["ghost"]["ghost_mass"]) for item in noops
    ]
    minimum_effect = calibrated_ghost_floor(
        practical_floor=plan.practical_ghost_floor,
        noop_deltas=noop_deltas,
    )
    responsibility_path = scene_dir / "fields" / "ghost_responsibility.npy"
    bind_artifact_directory(
        responsibility_path.parent,
        artifact_fingerprint(base_fingerprint, "ghost_responsibility"),
    )
    if responsibility_path.is_file():
        verify_output_file(responsibility_path)
        responsibility = np.load(responsibility_path, allow_pickle=False)
    else:
        responsibility = ghost_responsibility(
            checkpoint_path=checkpoint_path,
            bundle=bundle,
            repair=repair,
            baseline_renders=scene_dir / "baseline" / "repair_renders",
            scratch=scene_dir / "responsibility_renderer",
            neighbors=plan.ghost_neighbors,
            seed=plan.ghost_seed,
        )
        responsibility_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(responsibility_path, responsibility, allow_pickle=False)
        seal_output_file(responsibility_path)
    sensitivity = field_from_manifest(
        paths["diagnostic_manifest"],
        "mse__density",
        expected_checkpoint_sha256=file_sha256(checkpoint_path),
        expected_images=list(repair.names),
    )
    density = checkpoint["density"].detach().cpu().numpy().reshape(-1)
    rankings = repair_rankings(
        density=density,
        ghost_responsibility=responsibility,
        opacity_sensitivity=sensitivity,
    )
    outcomes: list[DevelopmentOutcome] = []
    result_index: dict[str, str] = {}
    overlap_by_budget: dict[str, object] = {}
    for budget_index, budget in enumerate(plan.prune_budget_fractions):
        row_count = max(1, math.floor(budget * density.size))
        cohorts = matched_pruning_cohorts(
            rankings=rankings,
            count=row_count,
            random_replicates=plan.random_replicates,
            seed=plan.ghost_seed + scene_index * 1000 + budget_index,
        )
        deterministic = {
            arm: values[0]
            for arm, values in cohorts.items()
            if arm != "random"
        }
        overlap_by_budget[str(budget)] = cohort_overlaps(deterministic)
        for arm, replicates in cohorts.items():
            for replicate, indices in enumerate(replicates):
                identifier = candidate_id(arm, budget, replicate)
                candidate_dir = scene_dir / "candidates" / identifier
                candidate_fingerprint = artifact_fingerprint(
                    base_fingerprint,
                    "development_candidate",
                    candidate_id=identifier,
                    ordered_index_sha256=ordered_indices_sha256(indices),
                    row_count=int(indices.size),
                )
                bind_artifact_directory(candidate_dir, candidate_fingerprint)
                cohort = save_cohort(candidate_dir / "cohort.npy", indices)
                result = evaluate_intervention(
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    bundle=bundle,
                    repair=repair,
                    hole_support=hole_support,
                    canonical_train=canonical_train,
                    canonical_train_path=paths["canonical_train_list"],
                    output_dir=candidate_dir,
                    indices=indices,
                    cohort=cohort,
                    neighbors=plan.ghost_neighbors,
                    fingerprint=candidate_fingerprint,
                )
                result_index[identifier] = str(candidate_dir / "result.json")
                gates = gate_results(
                    baseline["values"], result["values"], plan
                )
                outcome = DevelopmentOutcome(
                    scene=scene.scene,
                    arm=GhostRepairArm(arm),
                    budget_fraction=budget,
                    replicate=replicate,
                    ghost_before=baseline_ghost,
                    ghost_after=float(result["values"]["ghost"]),
                    gate_passes=gates,
                )
                outcomes.append(outcome)
    selected = select_development_candidate(
        outcomes=outcomes,
        rule=plan.selection,
        minimum_ghost_reduction=minimum_effect,
    )
    random_is_superior = False
    selected_identifier = "baseline"
    selected_cohort = None
    if selected is not None:
        selected_identifier = candidate_id(
            selected.arm.value, selected.budget_fraction, selected.replicate
        )
        selected_result = read_json(Path(result_index[selected_identifier]))
        selected_cohort = selected_result["cohort"]
        random_reductions = [
            outcome.ghost_reduction
            for outcome in outcomes
            if outcome.arm is GhostRepairArm.RANDOM
            and outcome.budget_fraction == selected.budget_fraction
        ]
        random_is_superior = random_superiority(
            candidate_reduction=selected.ghost_reduction,
            random_reductions=random_reductions,
            quantile=plan.exploratory_random_quantile,
        )
    selection = {
        "schema_version": 1,
        "scene": scene.scene,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": file_sha256(plan_path.resolve()),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "calibrated_minimum_ghost_reduction": minimum_effect,
        "noop_ghost_deltas": noop_deltas,
        "selected_candidate": selected_identifier,
        "selected_outcome": None
        if selected is None
        else selected.model_dump(mode="json"),
        "selected_cohort": selected_cohort,
        "random_superiority": random_is_superior,
        "deterministic_overlap_by_budget": overlap_by_budget,
        "result_index": result_index,
    }
    write_json_exclusive(selection_path, selection)
    seal_output_file(selection_path)
    release_model(None)
    return selection


def certification_inputs(
    scene: object,
    *,
    scene_root: Path,
    doctor_root: Path,
) -> tuple[dict[str, Path], Path]:
    """Resolve certification only after a frozen selection exists."""
    paths = scene_paths(scene, scene_root=scene_root, doctor_root=doctor_root)
    certification = resolve_registered_path(
        scene.certification_list,
        scene_root=scene_root,
        doctor_root=doctor_root,
    )
    return paths, certification


def validate_scene_inputs(
    *,
    scene: object,
    scene_root: Path,
    doctor_root: Path,
) -> dict[str, object]:
    """Authenticate development inputs without opening certification."""
    paths = scene_paths(scene, scene_root=scene_root, doctor_root=doctor_root)
    checkpoint = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    row_count = int(checkpoint["positions"].shape[0])
    bundle = Path(str(checkpoint["config"].path)).resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"checkpoint bundle does not exist: {bundle}")
    repair = load_explicit_image_list(str(paths["repair_list"]), "M2 repair")
    canonical_train = load_explicit_image_list(
        str(paths["canonical_train_list"]), "M2 canonical training"
    )
    hole_support = load_explicit_image_list(
        str(paths["hole_support_list"]), "M2 Hole support"
    )
    if set(repair.names) & set(canonical_train.names):
        raise ValueError("repair and canonical training lists overlap")
    if not set(hole_support.names).issubset(canonical_train.names):
        raise ValueError("Hole support is not a canonical-training subset")
    sensitivity = field_from_manifest(
        paths["diagnostic_manifest"],
        "mse__density",
        expected_checkpoint_sha256=file_sha256(paths["checkpoint"]),
        expected_images=list(repair.names),
    )
    if sensitivity.size != row_count:
        raise ValueError("opacity sensitivity does not align with checkpoint")
    return {
        "scene": scene.scene,
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "source_rows": row_count,
        "bundle": str(bundle),
        "repair_views": len(repair.names),
        "canonical_train_views": len(canonical_train.names),
        "hole_support_views": len(hole_support.names),
        "diagnostic_manifest_sha256": file_sha256(
            paths["diagnostic_manifest"]
        ),
        "certification_opened": False,
    }


def certify_scene(
    *,
    plan: GhostRepairPlan,
    scene: object,
    plan_path: Path,
    scene_root: Path,
    doctor_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Open the sealed split after authenticating the frozen selection."""
    selection_path = output_root / scene.scene / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"selection is not frozen: {scene.scene}")
    verify_output_file(selection_path)
    selection = read_json(selection_path)
    if selection["plan_sha256"] != file_sha256(plan_path.resolve()):
        raise ValueError("selection plan hash differs from certification plan")
    paths = scene_paths(scene, scene_root=scene_root, doctor_root=doctor_root)
    checkpoint_path = paths["checkpoint"]
    if selection["source_checkpoint_sha256"] != file_sha256(checkpoint_path):
        raise ValueError("selection checkpoint hash differs at certification")
    selection_sha256 = file_sha256(selection_path)
    write_json_exclusive(
        output_root / scene.scene / "certification_opened.json",
        {
            "schema_version": 1,
            "plan_sha256": file_sha256(plan_path.resolve()),
            "selection_sha256": selection_sha256,
        },
    )
    _, certification_path = certification_inputs(
        scene, scene_root=scene_root, doctor_root=doctor_root
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    bundle = Path(str(checkpoint["config"].path)).resolve()
    certification = load_explicit_image_list(
        str(certification_path), "M2 certification"
    )
    repair = load_explicit_image_list(str(paths["repair_list"]), "M2 repair")
    canonical_train = load_explicit_image_list(
        str(paths["canonical_train_list"]), "M2 canonical training"
    )
    hole_support = load_explicit_image_list(
        str(paths["hole_support_list"]), "M2 Hole support"
    )
    certification_names = set(certification.names)
    if certification_names & set(repair.names):
        raise ValueError("certification and repair views overlap")
    if certification_names & set(canonical_train.names):
        raise ValueError("certification and canonical training views overlap")
    if certification_names & set(hole_support.names):
        raise ValueError("certification and Hole-support views overlap")
    base_fingerprint = scene_fingerprint(
        plan=plan,
        scene=scene,
        plan_path=plan_path,
        paths=paths,
        doctor_root=doctor_root,
        bundle=bundle,
        image_names=sorted(
            certification_names
            | set(repair.names)
            | set(canonical_train.names)
            | set(hole_support.names)
        ),
    )
    certification_fingerprint = artifact_fingerprint(
        base_fingerprint,
        "certification",
        certification_list_sha256=file_sha256(certification_path),
        selection_sha256=selection_sha256,
    )
    cert_dir = output_root / scene.scene / "certification"
    baseline = baseline_result(
        checkpoint=checkpoint_path,
        bundle=bundle,
        repair=certification,
        hole_support=hole_support,
        canonical_train=canonical_train,
        canonical_train_path=paths["canonical_train_list"],
        output_dir=cert_dir / "baseline",
        neighbors=plan.ghost_neighbors,
        fingerprint=artifact_fingerprint(
            certification_fingerprint, "certification_baseline"
        ),
    )
    selected_name = selection["selected_candidate"]
    if selected_name == "baseline":
        selected_result = baseline
    else:
        cohort_path = Path(selection["selected_cohort"]["path"])
        if (
            file_sha256(cohort_path)
            != selection["selected_cohort"]["file_sha256"]
        ):
            raise ValueError(
                "selected cohort hash changed before certification"
            )
        indices = np.load(cohort_path, allow_pickle=False)
        selected_result = evaluate_intervention(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            bundle=bundle,
            repair=certification,
            hole_support=hole_support,
            canonical_train=canonical_train,
            canonical_train_path=paths["canonical_train_list"],
            output_dir=cert_dir / "selected" / selected_name,
            indices=indices,
            cohort=selection["selected_cohort"],
            neighbors=plan.ghost_neighbors,
            fingerprint=artifact_fingerprint(
                certification_fingerprint,
                "certification_selected",
                selected_candidate=selected_name,
                cohort=selection["selected_cohort"],
            ),
        )
    gates = gate_results(baseline["values"], selected_result["values"], plan)
    before_views = baseline["ghost"]["per_view"]
    after_views = selected_result["ghost"]["per_view"]
    lower_bound = paired_bootstrap_lower_bound(
        before=before_views,
        after=after_views,
        samples=plan.bootstrap_samples,
        seed=plan.bootstrap_seed,
        confidence_level=plan.confidence_level,
    )
    improvement = float(baseline["values"]["ghost"]) - float(
        selected_result["values"]["ghost"]
    )
    target_floor = max(
        plan.certification.minimum_target_improvement,
        float(selection["calibrated_minimum_ghost_reduction"]),
    )
    accepted = (
        selected_name != "baseline"
        and all(gates.values())
        and improvement >= target_floor
        and lower_bound > 0.0
    )
    certificate = {
        "schema_version": 1,
        "scene": scene.scene,
        "selection_sha256": selection_sha256,
        "certification_list_sha256": file_sha256(certification_path),
        "selected_candidate": selected_name,
        "baseline_values": baseline["values"],
        "selected_values": selected_result["values"],
        "metric_gates": {
            metric.value: value for metric, value in gates.items()
        },
        "ghost_improvement": improvement,
        "required_ghost_improvement": target_floor,
        "paired_bootstrap_lower_bound": lower_bound,
        "exploratory_random_superiority": selection["random_superiority"],
        "accepted": accepted,
    }
    write_json_exclusive(
        output_root / scene.scene / "certificate.json", certificate
    )
    seal_output_file(output_root / scene.scene / "certificate.json")
    return certificate


def aggregate_certificates(
    *,
    plan: GhostRepairPlan,
    plan_path: Path,
    output_root: Path,
) -> None:
    """Emit the method certificate once every scene certificate is sealed."""
    certificates: list[dict[str, object]] = []
    plan_sha256 = file_sha256(plan_path)
    for scene in plan.scenes:
        scene_root = output_root / scene.scene
        certificate_path = scene_root / "certificate.json"
        selection_path = scene_root / "selection.json"
        opened_path = scene_root / "certification_opened.json"
        if not (
            certificate_path.is_file()
            and selection_path.is_file()
            and opened_path.is_file()
        ):
            return
        opened = read_json(opened_path)
        selection_sha256 = file_sha256(selection_path)
        if (
            opened.get("plan_sha256") != plan_sha256
            or opened.get("selection_sha256") != selection_sha256
        ):
            raise ValueError(f"invalid certification marker: {scene.scene}")
        certificate = read_json(certificate_path)
        verify_output_file(certificate_path)
        if (
            certificate.get("scene") != scene.scene
            or certificate.get("selection_sha256") != selection_sha256
        ):
            raise ValueError(f"invalid scene certificate: {scene.scene}")
        certificates.append(certificate)
    method_pass = method_level_passes(
        accepted=[bool(item["accepted"]) for item in certificates],
        ghost_improvements=[
            float(item["ghost_improvement"]) for item in certificates
        ],
        rule=plan.method_level,
    )
    write_json_exclusive(
        output_root / "method_certificate.json",
        {
            "schema_version": 1,
            "study_id": plan.study_id,
            "plan_sha256": plan_sha256,
            "scene_certificates": certificates,
            "method_level_pass": method_pass,
        },
    )
    seal_output_file(output_root / "method_certificate.json")


def main() -> None:
    """Run the requested phase for selected or all preregistered scenes."""
    args = parse_args()
    plan_path = args.plan.resolve()
    scene_root = args.scene_root.resolve()
    doctor_root = args.doctor_root.resolve()
    output_root = args.output_root.resolve()
    plan = load_plan(plan_path)
    validate_runtime_contract(plan)
    selected_scenes = [
        scene
        for scene in plan.scenes
        if not args.scene or scene.scene in args.scene
    ]
    if len(selected_scenes) != (
        len(args.scene) if args.scene else len(plan.scenes)
    ):
        raise ValueError("requested scene is not registered")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.phase == "validate":
        audits = [
            validate_scene_inputs(
                scene=scene,
                scene_root=scene_root,
                doctor_root=doctor_root,
            )
            for scene in selected_scenes
        ]
        audit_path = output_root / "input_audit.json"
        write_json_exclusive(
            audit_path,
            {
                "schema_version": 1,
                "plan_sha256": file_sha256(plan_path),
                "scenes": audits,
            },
        )
        seal_output_file(audit_path)
        return
    if args.phase == "develop":
        for scene_index, scene in enumerate(plan.scenes):
            if scene not in selected_scenes:
                continue
            develop_scene(
                plan=plan,
                scene=scene,
                scene_index=scene_index,
                plan_path=plan_path,
                scene_root=scene_root,
                doctor_root=doctor_root,
                output_root=output_root,
            )
        return
    for scene in selected_scenes:
        certify_scene(
            plan=plan,
            scene=scene,
            plan_path=plan_path,
            scene_root=scene_root,
            doctor_root=doctor_root,
            output_root=output_root,
        )
    aggregate_certificates(
        plan=plan,
        plan_path=plan_path,
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
