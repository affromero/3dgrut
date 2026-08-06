"""Executable M2 Ghost-directed pruning study.

Development reads only the registered repair split. Certification is a
separate entry point that requires a frozen selection artifact.
"""

# ruff: noqa: EM101, EM102, FBT003, PLR0913, TRY003

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from doctor_splat.dump_val_depth import explicit_renderer, render_val
from doctor_splat.ghost_mass import (
    MARGIN,
    N_SAMPLES,
    arm_ghost_mass,
    ghost_cameras,
    sampled_ghost_evidence,
)
from doctor_splat.instruments import cams_and_centers
from doctor_splat.metrics.coverage import evaluate as evaluate_hole
from doctor_splat.metrics.depth_io import (
    load_render_distance,
    load_render_meta,
)
from doctor_splat.post_training.contract import (
    CertificateMetric,
    MetricDirection,
)
from doctor_splat.post_training.ghost_repair import (
    GhostRepairPlan,
)
from doctor_splat.split_contract import (
    ExplicitImageList,
)
from doctor_splat.surface_metrics import evaluate as evaluate_surface
from doctor_splat.trainpair_reproj import arm_residual

from threedgrut.ghost_repair import (
    SampledGhostEvidence,
    ordered_indices_sha256,
    pruned_checkpoint,
    sampled_contradiction_field,
)
from threedgrut.model.factory import create_gaussian_model


def file_sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    """Hash canonical JSON data used to bind resumable artifacts."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: object) -> None:
    """Atomically write canonical, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json_exclusive(path: Path, payload: object) -> None:
    """Create immutable JSON once, or verify an identical existing value."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable artifact differs: {path}") from None


def bind_artifact_directory(path: Path, fingerprint: object) -> None:
    """Bind a resumable directory to the complete registered inputs."""
    marker = path / "input_fingerprint.json"
    expected = {
        "schema_version": 1,
        "fingerprint_sha256": payload_sha256(fingerprint),
        "fingerprint": fingerprint,
    }
    if marker.is_file():
        if read_json(marker) != expected:
            raise ValueError(f"artifact input fingerprint differs: {path}")
        return
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"unbound artifact directory is not reusable: {path}")
    write_json_exclusive(marker, expected)


def seal_output_file(path: Path) -> None:
    """Seal one completed output file with its content hash and size."""
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_name(f"{path.name}.integrity.json")
    write_json_exclusive(
        sidecar,
        {
            "schema_version": 1,
            "file": path.name,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        },
    )


def verify_output_file(path: Path) -> None:
    """Reject a missing or changed output file before resuming it."""
    sidecar = path.with_name(f"{path.name}.integrity.json")
    if not path.is_file() or not sidecar.is_file():
        message = f"sealed output is incomplete: {path}"
        raise ValueError(message)
    expected = read_json(sidecar)
    observed = {
        "schema_version": 1,
        "file": path.name,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if expected != observed:
        message = f"sealed output changed: {path}"
        raise ValueError(message)


def _output_tree_files(path: Path) -> dict[str, dict[str, int | str]]:
    """Hash every payload file in a rendered output tree."""
    excluded = {"input_fingerprint.json", "output_manifest.json"}
    return {
        str(file.relative_to(path)): {
            "sha256": file_sha256(file),
            "size_bytes": file.stat().st_size,
        }
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name not in excluded
    }


def seal_output_tree(path: Path) -> None:
    """Seal a completed render directory and every payload within it."""
    files = _output_tree_files(path)
    if not files:
        message = f"cannot seal an empty output tree: {path}"
        raise ValueError(message)
    write_json_exclusive(
        path / "output_manifest.json",
        {"schema_version": 1, "files": files},
    )


def verify_output_tree(path: Path) -> None:
    """Reject altered, added, removed, or unsealed rendered payloads."""
    manifest = path / "output_manifest.json"
    if not manifest.is_file():
        message = f"render output is not sealed: {path}"
        raise ValueError(message)
    expected = read_json(manifest)
    observed = {"schema_version": 1, "files": _output_tree_files(path)}
    if expected != observed:
        message = f"render output changed: {path}"
        raise ValueError(message)


def read_json(path: Path) -> object:
    """Read one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_registered_path(
    registered: object,
    *,
    scene_root: Path,
    doctor_root: Path,
) -> Path:
    """Resolve and authenticate one preregistered input."""
    relative = Path(registered.path)
    candidates = [scene_root / relative, doctor_root / relative]
    if relative.parts and relative.parts[0] == "doctor-splat":
        candidates.insert(0, doctor_root.joinpath(*relative.parts[1:]))
    existing = [
        candidate.resolve() for candidate in candidates if candidate.is_file()
    ]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"registered path must resolve uniquely: {registered.path}"
        )
    resolved = existing[0]
    if file_sha256(resolved) != registered.sha256:
        raise ValueError(f"registered input hash mismatch: {registered.path}")
    return resolved


def validate_runtime_contract(plan: GhostRepairPlan) -> None:
    """Bind the registered Ghost constants to the installed instrument."""
    if plan.ghost_samples_per_view != N_SAMPLES:
        raise ValueError("registered Ghost sample count differs from runtime")
    if plan.ghost_margin != MARGIN:
        raise ValueError("registered Ghost margin differs from runtime")


def model_from_pruned_checkpoint(
    checkpoint: dict[str, object],
    removed: np.ndarray,
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Materialize the physical-row-removal intervention on the GPU."""
    pruned = pruned_checkpoint(checkpoint, removed)
    pruned["config"].render.splat.ut_require_all_sigma_points_valid = True
    model = create_gaussian_model(pruned["config"], checkpoint=pruned)
    model.init_from_checkpoint(pruned, setup_optimizer=False)
    model = model.cuda()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, pruned["post_training_prune"]


def release_model(model: torch.nn.Module | None) -> None:
    """Release a candidate before the next large model is constructed."""
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def ghost_responsibility(
    *,
    checkpoint_path: Path,
    bundle: Path,
    repair: ExplicitImageList,
    baseline_renders: Path,
    scratch: Path,
    neighbors: int,
    seed: int,
) -> np.ndarray:
    """Reduce registered contradiction samples to Gaussian T-alpha ownership."""
    renderer, renderer_names = explicit_renderer(
        str(checkpoint_path),
        str(bundle),
        str(scratch),
        repair,
    )
    metas = sorted(
        load_render_meta(str(baseline_renders)),
        key=lambda item: item.iteration,
    )
    if [meta.name for meta in metas] != renderer_names:
        raise ValueError(
            "baseline render order differs from repair renderer order"
        )
    cameras = ghost_cameras(str(bundle / "sparse" / "0"))
    distances = {
        meta.name: load_render_distance(str(baseline_renders), meta)
        for meta in metas
    }
    centers = {
        name: -cameras[name]["R"].T @ cameras[name]["t"]
        for name in renderer_names
    }
    responsibility = (
        torch.zeros_like(renderer.model.density.detach()).cpu().float()
    )
    for batch, name in zip(renderer.dataloader, renderer_names, strict=True):
        near = sorted(
            (other for other in renderer_names if other != name),
            key=lambda other: float(
                np.linalg.norm(centers[other] - centers[name])
            ),
        )[:neighbors]
        rng = np.random.default_rng(seed)
        us, vs, ghost, evidenced = sampled_ghost_evidence(
            distances[name],
            cameras[name],
            [(cameras[other], distances[other]) for other in near],
            rng,
        )
        gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)
        field = sampled_contradiction_field(
            image_shape=distances[name].shape,
            samples=SampledGhostEvidence(
                us=us,
                vs=vs,
                ghost=ghost,
                evidenced=evidenced,
            ),
            device=gpu_batch.rays_ori.device,
        )
        with torch.no_grad():
            weighted = renderer.model.render_responsibility(gpu_batch, field)[
                "diagnostic_weighted_sum"
            ]
        responsibility += (
            weighted.detach().cpu().float().reshape_as(responsibility)
        )
    output = responsibility.reshape(-1).numpy()
    model = renderer.model
    del renderer
    del model
    release_model(None)
    return output


def render_candidate(
    *,
    source_checkpoint: Path,
    bundle: Path,
    repair: ExplicitImageList,
    hole_support: ExplicitImageList,
    output_dir: Path,
    model: torch.nn.Module | None,
    fingerprint: object,
) -> tuple[Path, Path]:
    """Render the exact development and Hole-support lists."""
    bind_artifact_directory(output_dir, fingerprint)
    repair_dir = output_dir / "repair_renders"
    support_dir = output_dir / "hole_support_renders"
    for render_dir, image_list, role in (
        (repair_dir, repair, "repair_renders"),
        (support_dir, hole_support, "hole_support_renders"),
    ):
        render_fingerprint = {
            "candidate": fingerprint,
            "render_role": role,
            "images": list(image_list.names),
        }
        bind_artifact_directory(render_dir, render_fingerprint)
        if (render_dir / "meta.json").is_file():
            verify_output_tree(render_dir)
            continue
        partial = [
            path
            for path in render_dir.iterdir()
            if path.name != "input_fingerprint.json"
        ]
        if partial:
            message = (
                f"partial render requires explicit recovery: {render_dir}"
            )
            raise ValueError(message)
        render_val(
            str(source_checkpoint),
            str(bundle),
            str(render_dir),
            image_list,
            model=model,
        )
        seal_output_tree(render_dir)
    return repair_dir, support_dir


def evaluate_candidate(
    *,
    repair_renders: Path,
    support_renders: Path,
    bundle: Path,
    canonical_train_path: Path,
    repair: ExplicitImageList,
    canonical_train: ExplicitImageList,
    neighbors: int,
) -> dict[str, object]:
    """Evaluate all seven registered development axes."""
    surface = evaluate_surface(
        renders=str(repair_renders),
        bundle_sparse=str(bundle / "sparse" / "0"),
        canonical_train_list=str(canonical_train_path),
    )
    ghost = arm_ghost_mass(
        str(repair_renders),
        str(bundle / "sparse" / "0"),
        k=neighbors,
    )
    permitted = [*repair.names, *canonical_train.names]
    cameras, centers = cams_and_centers(str(bundle), permitted)
    crr = arm_residual(
        str(repair_renders),
        repair.names,
        cameras,
        centers,
        canonical_train.names,
        str(bundle / "images"),
        3,
    )
    hole = evaluate_hole(
        heldout_renders=str(repair_renders),
        training_renders=str(support_renders),
        bundle_sparse=str(bundle / "sparse" / "0"),
    )
    values = {
        "psnr": surface.photometric.psnr_db,
        "ssim": surface.photometric.ssim,
        "lpips": surface.photometric.lpips,
        "fgd": surface.teacher_fgd.fgd,
        "crr": crr.residual,
        "ghost": ghost.ghost_mass,
        "hole": hole.hole_mass,
    }
    return {
        "values": values,
        "surface": surface.model_dump(mode="json"),
        "crr": crr.model_dump(mode="json"),
        "ghost": ghost.model_dump(mode="json"),
        "hole": hole.model_dump(mode="json"),
    }


def gate_results(
    baseline: dict[str, float],
    candidate: dict[str, float],
    plan: GhostRepairPlan,
) -> dict[CertificateMetric, bool]:
    """Apply every registered noninferiority gate in its declared direction."""
    results: dict[CertificateMetric, bool] = {}
    for gate in plan.certification.metric_gates:
        before = float(baseline[gate.metric.value])
        after = float(candidate[gate.metric.value])
        if gate.direction is MetricDirection.MAXIMIZE:
            results[gate.metric] = after >= before - gate.max_regression
        else:
            results[gate.metric] = after <= before + gate.max_regression
    return results


def cohort_overlaps(
    deterministic: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    """Report overlap without changing any preregistered cohort."""
    names = sorted(deterministic)
    output: dict[str, dict[str, float | int]] = {}
    for left_index, left_name in enumerate(names):
        left = set(deterministic[left_name].tolist())
        for right_name in names[left_index + 1 :]:
            right = set(deterministic[right_name].tolist())
            intersection = len(left & right)
            union = len(left | right)
            output[f"{left_name}__{right_name}"] = {
                "intersection": intersection,
                "jaccard": intersection / union if union else 0.0,
            }
    return output


def save_cohort(path: Path, indices: np.ndarray) -> dict[str, object]:
    """Persist the exact ordered row intervention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = np.asarray(indices, dtype=np.int64)
    if path.is_file():
        verify_output_file(path)
        existing = np.load(path, allow_pickle=False)
        if not np.array_equal(existing, canonical):
            message = f"sealed cohort differs: {path}"
            raise ValueError(message)
    else:
        np.save(path, canonical, allow_pickle=False)
        seal_output_file(path)
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "ordered_index_sha256": ordered_indices_sha256(indices),
        "row_count": int(indices.size),
    }


def candidate_id(arm: str, budget: float, replicate: int) -> str:
    """Stable directory identifier for one registered candidate."""
    budget_ppm = round(budget * 1_000_000)
    return f"{arm}__b{budget_ppm:05d}ppm__r{replicate:02d}"
