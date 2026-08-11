# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Export held-out ray-attributed Gaussian error fields as colored PLYs."""

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from klogr.path import path_basename, path_join, path_mkdir, path_relative_to
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from render_common_eval import _build_renderer
from threedgrut import intervention_study
from threedgrut.datasets.protocols import Batch
from threedgrut.error_attribution import (
    ErrorAttributionAccumulator,
    ErrorAttributionMetric,
    ErrorAttributionParameter,
    attribution_loss,
    camera_support_indicators,
    heldout_ownership_dominance,
    native_contributor_ray_fields,
    native_render_evidence_maps,
    native_structural_gaussian_fields,
    ownership_weighted_mean,
    recolor_gaussian_ply,
)
from threedgrut.export_utils import (
    sample_indices as _sample_indices,
    scene_relative_path as _scene_relative_path,
)
from threedgrut.model.factory import create_gaussian_model
from threedgrut.render import POST_PROCESSING_EVAL_MODE_RAW
from threedgrut.split_membership import (
    dataset_membership,
    training_membership_provenance,
    use_dataset_membership,
)
from threedgrut.utils.logger import logger

DEFAULT_VISIBILITY_THRESHOLD = 0.0
DEFAULT_OWNERSHIP_SUPPORT_THRESHOLD = 1e-6
DEFAULT_NATIVE_OPACITY_FLOOR = 1e-4
MANIFEST_SCHEMA_VERSION = 5
M8_RECONSTRUCTION_MANIFEST_SCHEMA_VERSION = 6

_counterfactual_cohorts = intervention_study.counterfactual_cohorts
_evaluate_density_counterfactuals = intervention_study.evaluate_density_counterfactuals
_suppressed_density_cohort = intervention_study.suppressed_density_cohort
run_intervention_study = intervention_study.run_intervention_study


def _enum_values(enum_type: type[ErrorAttributionMetric]) -> tuple[str, ...]:
    return tuple(member.value for member in enum_type)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--source-ply",
        help=("Row-aligned source PLY. When omitted, export one from the " "checkpoint into the output directory."),
    )
    parser.add_argument("--eval-bundle", required=True)
    parser.add_argument("--holdout-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--scene-root",
        required=True,
        help="Scene root used to persist portable manifest-relative paths.",
    )
    parser.add_argument("--camera-type", required=True)
    parser.add_argument("--processing", required=True)
    parser.add_argument("--source-quality-id", required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument(
        "--attribution-probes",
        type=int,
        default=8,
        help="Deterministic Rademacher probes per metric and held-out view.",
    )
    parser.add_argument(
        "--attribution-seed",
        type=int,
        default=0,
        help="Deterministic seed for Rademacher attribution probes.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=_enum_values(ErrorAttributionMetric),
        default=[member.value for member in ErrorAttributionMetric],
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=tuple(member.value for member in ErrorAttributionParameter),
        default=[
            ErrorAttributionParameter.APPEARANCE.value,
            ErrorAttributionParameter.POSITION.value,
            ErrorAttributionParameter.OPACITY.value,
        ],
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=0,
        help="0 evaluates every held-out view; positive values sample evenly.",
    )
    parser.add_argument(
        "--training-support-max-views",
        type=int,
        default=0,
        help=(
            "0 aggregates every training view; positive values sample evenly "
            "for the native per-Gaussian support exports."
        ),
    )
    parser.add_argument(
        "--ownership-support-threshold",
        type=float,
        default=DEFAULT_OWNERSHIP_SUPPORT_THRESHOLD,
        help=(
            "Minimum mean native T*alpha ownership per rendered training ray "
            "for a camera to count as meaningful support."
        ),
    )
    parser.add_argument(
        "--export-fields",
        nargs="+",
        default=None,
        metavar="METRIC:PARAMETER",
        help=("Optional subset of the computed metric/parameter cross-product " "to materialize as PLY files."),
    )
    parser.add_argument(
        "--normalization",
        choices=("p95", "p99", "log", "linear"),
        default="p99",
    )
    parser.add_argument(
        "--visibility-threshold",
        type=float,
        default=DEFAULT_VISIBILITY_THRESHOLD,
        help=(
            "Optionally hide Gaussians below this normalized attribution. "
            "The default preserves every source Gaussian and its opacity."
        ),
    )
    parser.add_argument(
        "--counterfactual-cohort-size",
        type=int,
        default=0,
        help=(
            "When positive, evaluate top, random, and low density-sensitivity "
            "Gaussian cohorts by suppressing them and rerendering held-out "
            "views. Zero disables this expensive intervention evaluation."
        ),
    )
    parser.add_argument(
        "--m8-reconstruction-fields",
        action="store_true",
        help=(
            "Evaluate policy fields on authenticated reconstruction images. "
            "Selected and training-support image sets must be identical."
        ),
    )
    parser.add_argument(
        "--counterfactual-suppression-logit",
        type=float,
        default=-20.0,
        help="Pre-sigmoid density value assigned to a suppressed cohort.",
    )
    parser.add_argument(
        "--intervention-plan",
        help="Frozen matched-intervention study JSON.",
    )
    parser.add_argument(
        "--certification-list",
        help="Disjoint image list used only for intervention outcomes.",
    )
    parser.add_argument(
        "--intervention-scene-id",
        help="Scene identifier declared by the intervention plan.",
    )
    parser.add_argument(
        "--intervention-max-views",
        type=int,
        default=0,
        help="0 uses every certification view; positive values sample evenly.",
    )
    return parser.parse_args()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_mode_contract(
    *,
    m8_reconstruction_fields: bool,
    selected_images: list[str],
    training_support_images: list[str],
) -> tuple[int, str]:
    """Validate the image-domain contract for one diagnostic manifest."""
    if not m8_reconstruction_fields:
        if set(selected_images).intersection(training_support_images):
            raise ValueError(
                "held-out and training-support diagnostic images overlap"
            )
        return MANIFEST_SCHEMA_VERSION, "heldout_diagnostic_fields"
    if selected_images != training_support_images:
        raise ValueError(
            "M8 reconstruction fields require identical ordered "
            "selected and training-support image sets"
        )
    return (
        M8_RECONSTRUCTION_MANIFEST_SCHEMA_VERSION,
        "m8_reconstruction_policy_fields",
    )


def _write_raw_field(
    *,
    output_dir: str,
    field_id: str,
    scores: torch.Tensor,
) -> dict[str, object]:
    """Atomically persist row-aligned float32 scores with provenance."""
    raw_dir = path_join(output_dir, "raw")
    path_mkdir(raw_dir, parents=True, exist_ok=True)
    output_path = path_join(raw_dir, f"{field_id}.npy")
    temporary_path = f"{output_path}.tmp"
    values = scores.detach().reshape(-1).float().cpu().numpy()
    with open(temporary_path, "wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary_path, output_path)
    return {
        "raw_filename": path_relative_to(output_path, output_dir),
        "raw_sha256": _sha256(output_path),
        "raw_dtype": "float32",
        "raw_shape": [int(values.size)],
    }


def _field_label(metric: str, parameter: str) -> str:
    metric_labels = {
        "mae": "MAE",
        "mse": "MSE / PSNR",
        "ssim": "SSIM loss",
        "lpips": "LPIPS",
        "lowfreq_frac": "Doctor low-frequency residual",
        "registered_loss": "Registered 0.8 L1 + 0.2 one-minus-SSIM loss",
    }
    parameter_labels = {
        "sh_dc_rgb": "SH DC RGB coefficients",
        "features_specular": "higher-order SH RGB coefficients",
        "positions": "position",
        "scale": "scale",
        "rotation": "rotation",
        "density": "opacity",
    }
    return f"{metric_labels[metric]} · {parameter_labels[parameter]}"


def _write_native_evidence_map(
    *,
    output_dir: str,
    image_name: str,
    outputs: dict[str, torch.Tensor | float],
) -> dict[str, torch.Tensor]:
    """Persist native alpha/depth moments for one held-out view."""
    with torch.no_grad():
        evidence = native_render_evidence_maps(
            accumulated_alpha=outputs["pred_opacity"],
            depth_moment=outputs["pred_dist"],
            depth_squared_moment=outputs["pred_dist_squared"],
            hit_count=outputs["hits_count"],
            opacity_floor=DEFAULT_NATIVE_OPACITY_FLOOR,
        )
    native_dir = path_join(output_dir, "maps", "native")
    path_mkdir(native_dir, parents=True, exist_ok=True)
    image_hash = hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:16]
    fields = {name: value.detach().squeeze(0).squeeze(-1).float().cpu().numpy() for name, value in evidence.items()}
    np.savez_compressed(
        path_join(native_dir, f"map_{image_hash}.npz"),
        image_name=np.array(image_name),
        opacity_validity_floor=np.array(DEFAULT_NATIVE_OPACITY_FLOOR),
        **fields,
    )
    return evidence


def _accumulate_native_contributor_fields(
    *,
    model: torch.nn.Module,
    gpu_batch: Batch,
    evidence: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
) -> None:
    """Backproject selected native ray fields with exact ``T*alpha`` weights."""
    ray_fields = native_contributor_ray_fields(
        accumulated_alpha=evidence["accumulated_alpha"],
        depth_variance=evidence["depth_variance"],
        hit_count=evidence["hit_count"],
    )
    for field_id, ray_field in ray_fields.items():
        weighted_sum = model.render_responsibility(
            gpu_batch,
            ray_field.squeeze(0).squeeze(-1).float(),
        )["diagnostic_weighted_sum"]
        scores[field_id] += weighted_sum.detach().float().cpu().reshape_as(scores[field_id])


def _export_training_support_fields(
    *,
    checkpoint: dict[str, object],
    model: torch.nn.Module,
    output_dir: str,
    eval_bundle: str,
    checkpoint_path: str,
    original_training_bundle: str,
    source_ply: str,
    scale_mode: str,
    visibility_threshold: float,
    maximum_views: int,
    ownership_support_threshold: float,
) -> tuple[list[dict[str, object]], int, list[str], torch.Tensor]:
    """Export exact native T*alpha training support for every Gaussian."""
    renderer = _build_renderer(
        checkpoint,
        model,
        out_dir=output_dir,
        eval_bundle=eval_bundle,
        post_processing_mode=POST_PROCESSING_EVAL_MODE_RAW,
        split="train",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(checkpoint_path),
        original_training_bundle=original_training_bundle,
    )
    selected_indices = _sample_indices(len(renderer.dataloader), maximum_views)
    support = torch.zeros_like(
        model.density.detach(),
        dtype=torch.float32,
        device="cpu",
    )
    visible_view_count = torch.zeros_like(support)
    supporting_camera_count = torch.zeros_like(support)
    narrow_view_coverage = torch.zeros(
        (model.density.shape[0], 3),
        dtype=torch.float32,
        device="cpu",
    )
    selected_names: list[str] = []
    logger.info("Computing native per-Gaussian training support for " f"{len(selected_indices)} views.")
    for index, batch in enumerate(renderer.dataloader):
        if index not in selected_indices:
            continue
        gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)
        ray_diagnostic = torch.ones(
            gpu_batch.rays_ori.shape[1:3],
            device=gpu_batch.rays_ori.device,
            dtype=torch.float32,
        )
        responsibility = (
            model.render_responsibility(
                gpu_batch,
                ray_diagnostic,
            )["responsibility"]
            .detach()
            .float()
            .cpu()
            .reshape_as(support)
        )
        support += responsibility
        nonzero, supporting = camera_support_indicators(
            responsibility=responsibility,
            ray_count=ray_diagnostic.numel(),
            ownership_support_threshold=ownership_support_threshold,
        )
        visible_view_count += nonzero
        supporting_camera_count += supporting
        camera_center = gpu_batch.T_to_world[0, :3, 3].detach()
        directions = camera_center.unsqueeze(0) - model.positions.detach()
        directions = directions / torch.linalg.vector_norm(
            directions,
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(directions.dtype).eps)
        narrow_view_coverage += (responsibility.to(device=directions.device) * directions).detach().float().cpu()
        selected_names.append(path_basename(str(gpu_batch.image_path)))

    field_specs = (
        (
            "training_support",
            "Training-ray support",
            support,
            "Exact sum over training rays of front-to-back T*alpha ownership.",
            "coverage",
        ),
        (
            "training_view_count",
            "Training-view visibility",
            visible_view_count,
            "Number of training cameras with nonzero native T*alpha support.",
            "coverage",
        ),
        (
            "supporting_camera_count",
            "Supporting-camera count",
            supporting_camera_count,
            "Number of training cameras whose mean native T*alpha ownership "
            "per rendered ray meets the recorded support threshold.",
            "coverage",
        ),
        (
            "narrow_training_view_coverage",
            "Narrow training-view coverage",
            torch.linalg.vector_norm(narrow_view_coverage, dim=-1, keepdim=True)
            / support.clamp_min(torch.finfo(support.dtype).eps),
            "Native ownership-weighted resultant camera-direction magnitude "
            "over training views. High means the Gaussian was seen from a "
            "narrow angular range; it is an insufficient-view candidate, "
            "not a visibility or geometry verdict.",
            "structural_candidate",
        ),
    )
    ply_dir = path_join(output_dir, "ply")
    fields: list[dict[str, object]] = []
    for field_id, label, scores, attribution, metric_id in field_specs:
        output_path = path_join(ply_dir, f"{field_id}.ply")
        statistics = recolor_gaussian_ply(
            source_path=source_ply,
            output_path=output_path,
            scores=scores.reshape(-1),
            scale_mode=scale_mode,
            expected_density=model.density,
            visibility_threshold=visibility_threshold,
        )
        raw_metadata = _write_raw_field(
            output_dir=output_dir,
            field_id=field_id,
            scores=scores,
        )
        fields.append(
            {
                "id": field_id,
                "label": label,
                "metric_id": metric_id,
                "parameter": field_id,
                "artifact_kind": "ply",
                "filename": path_relative_to(output_path, output_dir),
                "artifact_sha256": _sha256(output_path),
                **raw_metadata,
                "normalization": scale_mode,
                "visibility_threshold": visibility_threshold,
                "statistics": statistics,
                "exact_localization": True,
                "attribution": attribution,
            }
        )
    return fields, len(selected_names), selected_names, support


def main() -> None:
    """Export the requested held-out metric/parameter Gaussian fields."""
    args = _parse_args()
    if args.max_views < 0:
        raise ValueError("--max-views must be non-negative.")
    if args.training_support_max_views < 0:
        raise ValueError("--training-support-max-views must be non-negative.")
    if args.attribution_probes <= 0:
        raise ValueError("--attribution-probes must be positive.")
    if args.ownership_support_threshold < 0.0:
        raise ValueError("--ownership-support-threshold must be non-negative.")
    if args.counterfactual_cohort_size < 0:
        raise ValueError("--counterfactual-cohort-size must be non-negative.")
    study_arguments = (
        args.intervention_plan,
        args.certification_list,
        args.intervention_scene_id,
    )
    if any(study_arguments) and not all(study_arguments):
        raise ValueError("intervention plan, certification list, and scene id are " "required together")
    if args.intervention_max_views < 0:
        raise ValueError("--intervention-max-views must be non-negative.")
    if not 0.0 <= args.visibility_threshold < 1.0:
        raise ValueError("--visibility-threshold must be in [0, 1).")
    checkpoint_path = os.path.abspath(args.checkpoint)
    eval_bundle = os.path.abspath(args.eval_bundle)
    output_dir = os.path.abspath(args.output_dir)
    scene_root = os.path.abspath(args.scene_root)
    os.makedirs(output_dir, exist_ok=True)
    source_ply = (
        os.path.abspath(args.source_ply)
        if args.source_ply is not None
        else path_join(output_dir, "source_checkpoint.ply")
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
    checkpoint_training_membership = dataset_membership(conf)
    original_training_bundle = str(conf.path)
    conf.path = eval_bundle
    conf.dataset.holdout_image_list_path = os.path.abspath(args.holdout_list)
    conf.dataset.train_exclude_image_list_path = None
    conf.dataset.shutter_type = "GLOBAL"
    conf.dataset.load_exif = False
    conf.dataset.sky_mask_folder = None
    conf.loss.use_sky_opacity = False
    # Current native 3DGUT kernels require the complete unscented-transform
    # sigma-point set. Old checkpoints may carry the retired false value.
    conf.render.splat.ut_require_all_sigma_points_valid = True

    model = create_gaussian_model(conf, checkpoint=checkpoint)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    if args.source_ply is None:
        model.export_ply(source_ply)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.build_acc()
    renderer = _build_renderer(
        checkpoint,
        model,
        out_dir=output_dir,
        eval_bundle=eval_bundle,
        post_processing_mode=POST_PROCESSING_EVAL_MODE_RAW,
        split="val",
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_sha256(checkpoint_path),
        original_training_bundle=original_training_bundle,
    )

    metrics = tuple(ErrorAttributionMetric(value) for value in args.metrics)
    parameters = tuple(ErrorAttributionParameter(value) for value in args.parameters)
    available_field_keys = {f"{metric.value}:{parameter.value}" for metric in metrics for parameter in parameters}
    export_field_keys = set(args.export_fields) if args.export_fields is not None else available_field_keys
    unknown_field_keys = export_field_keys - available_field_keys
    if unknown_field_keys:
        raise ValueError(
            "--export-fields contains fields outside the selected metrics and "
            f"parameters: {sorted(unknown_field_keys)}"
        )
    lpips_model = None
    if ErrorAttributionMetric.LPIPS in metrics:
        lpips_model = (
            LearnedPerceptualImagePatchSimilarity(
                net_type="vgg",
                normalize=True,
            )
            .cuda()
            .eval()
        )
        for parameter in lpips_model.parameters():
            parameter.requires_grad_(False)
    accumulator = ErrorAttributionAccumulator(
        model=model,
        metrics=metrics,
        parameters=parameters,
        lpips_model=lpips_model,
        probe_count=args.attribution_probes,
        seed=args.attribution_seed,
    )
    selected_indices = _sample_indices(
        len(renderer.dataloader),
        args.max_views,
    )
    native_contributor_scores = {
        "heldout_native_ownership": torch.zeros_like(
            model.density.detach(),
            dtype=torch.float32,
            device="cpu",
        ),
        "depth_ambiguity_exposure": torch.zeros_like(
            model.density.detach(),
            dtype=torch.float32,
            device="cpu",
        ),
        "hit_congestion_exposure": torch.zeros_like(
            model.density.detach(),
            dtype=torch.float32,
            device="cpu",
        ),
        "residual_mse_exposure": torch.zeros_like(
            model.density.detach(),
            dtype=torch.float32,
            device="cpu",
        ),
    }
    selected_names: list[str] = []
    logger.info("Computing ray-attributed error fields for " f"{len(selected_indices)} held-out views.")
    for index, batch in enumerate(renderer.dataloader):
        if index not in selected_indices:
            continue
        gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)
        outputs = model(gpu_batch, train=False)
        losses = accumulator.accumulate(
            outputs=outputs,
            target=gpu_batch.rgb_gt,
            mask=gpu_batch.mask,
        )
        image_name = path_basename(str(gpu_batch.image_path))
        selected_names.append(image_name)
        native_evidence = _write_native_evidence_map(
            output_dir=output_dir,
            image_name=image_name,
            outputs=outputs,
        )
        _accumulate_native_contributor_fields(
            model=model,
            gpu_batch=gpu_batch,
            evidence=native_evidence,
            scores=native_contributor_scores,
        )
        residual_mse = (outputs["pred_rgb"] - gpu_batch.rgb_gt).square().mean(dim=-1)
        if gpu_batch.mask is not None:
            residual_mse = residual_mse * gpu_batch.mask.squeeze(-1)
        residual_weighted_sum = model.render_responsibility(
            gpu_batch,
            residual_mse.squeeze(0).float(),
        )["diagnostic_weighted_sum"]
        native_contributor_scores["residual_mse_exposure"] += (
            residual_weighted_sum.detach().float().cpu().reshape_as(native_contributor_scores["residual_mse_exposure"])
        )
        rendered_losses = ", ".join(f"{name}={value:.6f}" for name, value in losses.items())
        logger.info(
            f"Attribution view {len(selected_names)}/{len(selected_indices)}: "
            f"{selected_names[-1]} ({rendered_losses})"
        )

    rms_scores = accumulator.rms_scores()
    rms_standard_errors = accumulator.rms_standard_errors()
    counterfactual: dict[str, object] | None = None
    if args.counterfactual_cohort_size > 0:
        density_key = "mse:density"
        density_scores = rms_scores.get(density_key)
        if density_scores is None:
            raise ValueError("Density counterfactuals require --metrics mse and " "--parameters density.")
        logger.info(
            "Evaluating held-out density suppression controls for "
            f"{args.counterfactual_cohort_size} Gaussians per cohort."
        )
        counterfactual = _evaluate_density_counterfactuals(
            model=model,
            renderer=renderer,
            selected_indices=selected_indices,
            density_scores=density_scores,
            cohort_size=args.counterfactual_cohort_size,
            suppression_logit=args.counterfactual_suppression_logit,
        )
        counterfactual_path = path_join(output_dir, "interventions.json")
        with open(counterfactual_path, "w", encoding="utf-8") as handle:
            json.dump(counterfactual, handle, indent=2, sort_keys=True)

    fields: list[dict[str, object]] = []
    ply_dir = os.path.join(output_dir, "ply")
    for key, scores in rms_scores.items():
        if key not in export_field_keys:
            continue
        metric, parameter = key.split(":", maxsplit=1)
        filename = f"{metric}__{parameter}.ply"
        output_path = os.path.join(ply_dir, filename)
        statistics = recolor_gaussian_ply(
            source_path=source_ply,
            output_path=output_path,
            scores=scores,
            scale_mode=args.normalization,
            expected_density=model.density,
            visibility_threshold=args.visibility_threshold,
        )
        artifact_sha256 = _sha256(output_path)
        field_id = f"{metric}__{parameter}"
        raw_metadata = _write_raw_field(
            output_dir=output_dir,
            field_id=field_id,
            scores=scores,
        )
        uncertainty = rms_standard_errors.get(key)
        uncertainty_metadata = (
            {}
            if uncertainty is None
            else {
                "uncertainty_method": ("delta-method standard error from within-view " "Rademacher probe variance"),
                **{
                    f"uncertainty_{name}": value
                    for name, value in _write_raw_field(
                        output_dir=output_dir,
                        field_id=f"{field_id}__standard_error",
                        scores=uncertainty,
                    ).items()
                },
            }
        )
        fields.append(
            {
                "id": field_id,
                "label": _field_label(metric, parameter),
                "metric_id": metric,
                "parameter": parameter,
                "artifact_kind": "ply",
                "filename": os.path.relpath(output_path, output_dir),
                "artifact_sha256": artifact_sha256,
                **raw_metadata,
                **uncertainty_metadata,
                "normalization": args.normalization,
                "visibility_threshold": args.visibility_threshold,
                "statistics": statistics,
                "exact_localization": False,
                "attribution": (
                    "Deterministic Hutchinson estimate of RMS spatial-"
                    "component gradient norm through native 3DGRUT "
                    "front-to-back alpha compositing"
                ),
            }
        )

    mean_hit_congestion, congestion_validity = ownership_weighted_mean(
        weighted_exposure=native_contributor_scores["hit_congestion_exposure"],
        ownership=native_contributor_scores["heldout_native_ownership"],
    )
    native_contributor_scores["mean_hit_congestion"] = mean_hit_congestion
    native_contributor_scores["mean_hit_congestion_validity"] = congestion_validity
    native_contributor_specs = (
        (
            "heldout_native_ownership",
            "Held-out native ownership",
            "Exact sum over held-out rays of front-to-back T*alpha ownership.",
        ),
        (
            "depth_ambiguity_exposure",
            "Depth-ambiguity exposure",
            "Exact T*alpha-weighted exposure to native conditional depth "
            "variance; not an intrinsic per-Gaussian variance.",
        ),
        (
            "hit_congestion_exposure",
            "Hit-congestion exposure",
            "Exact T*alpha-weighted exposure to native accepted-hit count; " "not an intrinsic per-Gaussian hit count.",
        ),
        (
            "mean_hit_congestion",
            "Mean hit congestion",
            "Ownership-weighted mean native accepted-hit count over held-out " "rays with positive ownership.",
        ),
        (
            "mean_hit_congestion_validity",
            "Mean hit congestion validity",
            "One where held-out native ownership is positive and the "
            "ownership-weighted mean is defined; zero otherwise.",
        ),
        (
            "residual_mse_exposure",
            "Residual-only control exposure",
            "Exact T*alpha-weighted exposure to held-out per-pixel RGB MSE; "
            "retained as a matched intervention control, not a diagnosis.",
        ),
    )
    for field_id, label, attribution in native_contributor_specs:
        field_scores = native_contributor_scores[field_id]
        output_path = os.path.join(ply_dir, f"{field_id}.ply")
        statistics = recolor_gaussian_ply(
            source_path=source_ply,
            output_path=output_path,
            scores=field_scores.reshape(-1),
            scale_mode=args.normalization,
            expected_density=model.density,
            visibility_threshold=args.visibility_threshold,
        )
        raw_metadata = _write_raw_field(
            output_dir=output_dir,
            field_id=field_id,
            scores=field_scores,
        )
        fields.append(
            {
                "id": field_id,
                "label": label,
                "metric_id": (
                    "native_ray_mean" if field_id.startswith("mean_hit_congestion") else "native_ray_exposure"
                ),
                "parameter": field_id,
                "artifact_kind": "ply",
                "filename": os.path.relpath(output_path, output_dir),
                "artifact_sha256": _sha256(output_path),
                **raw_metadata,
                "normalization": args.normalization,
                "visibility_threshold": args.visibility_threshold,
                "statistics": statistics,
                "exact_localization": True,
                "attribution": attribution,
            }
        )

    with use_dataset_membership(conf, checkpoint_training_membership):
        (
            training_fields,
            training_support_view_count,
            training_support_images,
            training_support_scores,
        ) = _export_training_support_fields(
            checkpoint=checkpoint,
            model=model,
            output_dir=output_dir,
            eval_bundle=eval_bundle,
            checkpoint_path=checkpoint_path,
            original_training_bundle=original_training_bundle,
            source_ply=source_ply,
            scale_mode=args.normalization,
            visibility_threshold=args.visibility_threshold,
            maximum_views=args.training_support_max_views,
            ownership_support_threshold=args.ownership_support_threshold,
        )
    fields.extend(training_fields)
    heldout_ownership = native_contributor_scores["heldout_native_ownership"]
    mean_training_support = training_support_scores / max(
        training_support_view_count,
        1,
    )
    heldout_dominance = heldout_ownership_dominance(
        heldout_ownership=heldout_ownership / max(len(selected_names), 1),
        training_ownership=mean_training_support,
    )
    structural_fields = native_structural_gaussian_fields(
        positions=model.positions.detach(),
        covariances=model.get_covariance().detach(),
        physical_scales=model.get_scale().detach(),
    )
    structural_specs = (
        (
            "scale_to_neighbor_spacing",
            "Scale relative to neighbour spacing",
            structural_fields["scale_to_neighbor_spacing"],
            "Largest physical Gaussian standard deviation divided by nearest "
            "centre spacing. High values flag oversized-footprint candidates "
            "that should be tested by scale reduction or splitting.",
        ),
        (
            "nearest_covariance_overlap",
            "Nearest covariance overlap",
            structural_fields["nearest_covariance_overlap"],
            "Nearest-centre covariance-support overlap. High values are local "
            "duplicate-layer candidates, not a duplicate-geometry conclusion.",
        ),
        (
            "heldout_ownership_dominance",
            "Held-out ownership dominance",
            heldout_dominance,
            "Bounded held-out ownership share H/(H+S), where H and S are "
            "mean held-out and training T*alpha ownership. High values "
            "identify view-specific contributions that merit checking for "
            "insufficient observations or floaters.",
        ),
    )
    for field_id, label, scores, attribution in structural_specs:
        output_path = os.path.join(ply_dir, f"{field_id}.ply")
        statistics = recolor_gaussian_ply(
            source_path=source_ply,
            output_path=output_path,
            scores=scores.reshape(-1),
            scale_mode=args.normalization,
            expected_density=model.density,
            visibility_threshold=args.visibility_threshold,
        )
        raw_metadata = _write_raw_field(
            output_dir=output_dir,
            field_id=field_id,
            scores=scores,
        )
        fields.append(
            {
                "id": field_id,
                "label": label,
                "metric_id": "structural_candidate",
                "parameter": field_id,
                "artifact_kind": "ply",
                "filename": os.path.relpath(output_path, output_dir),
                "artifact_sha256": _sha256(output_path),
                **raw_metadata,
                "normalization": args.normalization,
                "visibility_threshold": args.visibility_threshold,
                "statistics": statistics,
                "exact_localization": True,
                "attribution": attribution,
            }
        )
    intervention_study = None
    if args.intervention_plan is not None:
        conf.dataset.holdout_image_list_path = os.path.abspath(args.certification_list)
        certification_renderer = _build_renderer(
            checkpoint,
            model,
            out_dir=path_join(output_dir, "certification"),
            eval_bundle=eval_bundle,
            post_processing_mode=POST_PROCESSING_EVAL_MODE_RAW,
            split="val",
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_sha256(checkpoint_path),
            original_training_bundle=original_training_bundle,
        )
        certification_indices = _sample_indices(
            len(certification_renderer.dataloader),
            args.intervention_max_views,
        )
        intervention_study = run_intervention_study(
            plan_path=os.path.abspath(args.intervention_plan),
            scene_id=args.intervention_scene_id,
            output_dir=output_dir,
            model=model,
            certification_renderer=certification_renderer,
            selected_indices=certification_indices,
            repair_list_path=os.path.abspath(args.holdout_list),
            certification_list_path=os.path.abspath(args.certification_list),
            checkpoint_sha256=_sha256(checkpoint_path),
        )
        study_path = path_join(output_dir, "intervention_study.json")
        with open(study_path, "w", encoding="utf-8") as handle:
            json.dump(intervention_study, handle, indent=2, sort_keys=True)
    schema_version, evaluation_role = _manifest_mode_contract(
        m8_reconstruction_fields=args.m8_reconstruction_fields,
        selected_images=selected_names,
        training_support_images=training_support_images,
    )
    manifest = {
        "schema_version": schema_version,
        "evaluation_role": evaluation_role,
        "source_checkpoint": _scene_relative_path(
            checkpoint_path,
            scene_root,
        ),
        "source_checkpoint_sha256": _sha256(checkpoint_path),
        "source_ply": _scene_relative_path(source_ply, scene_root),
        "source_ply_sha256": _sha256(source_ply),
        "eval_bundle": _scene_relative_path(eval_bundle, scene_root),
        "split": "val",
        "post_processing_mode": "raw",
        "camera_type": args.camera_type,
        "processing": args.processing,
        "source_quality_id": args.source_quality_id,
        "frame": args.frame,
        "view_count": accumulator.view_count,
        "attribution_probes": args.attribution_probes,
        "attribution_seed": args.attribution_seed,
        "attribution_estimator": ("sqrt(mean_views(mean_local_components(" "squared_parameter_block_gradient_norm)))"),
        "attribution_components": accumulator.component_metadata(),
        "ownership_support_threshold": args.ownership_support_threshold,
        "ownership_support_threshold_units": "mean T*alpha per rendered ray",
        "native_depth_opacity_floor": DEFAULT_NATIVE_OPACITY_FLOOR,
        "visibility_threshold": args.visibility_threshold,
        "selected_images": selected_names,
        "training_support_view_count": training_support_view_count,
        "training_support_images": training_support_images,
        "training_membership": training_membership_provenance(
            checkpoint_training_membership
        ),
        "mean_losses": accumulator.mean_losses(),
        "counterfactual_intervention": (
            None
            if counterfactual is None
            else {
                "filename": "interventions.json",
                "artifact_sha256": _sha256(path_join(output_dir, "interventions.json")),
                "contents": counterfactual,
            }
        ),
        "matched_intervention_study": (
            None
            if intervention_study is None
            else {
                "filename": "intervention_study.json",
                "artifact_sha256": _sha256(path_join(output_dir, "intervention_study.json")),
                "contents": intervention_study,
            }
        ),
        "native_evidence_maps": {
            "accumulated_alpha": "native front-to-back accumulated alpha",
            "expected_depth": "native conditional hit-distance mean",
            "depth_variance": "native conditional hit-distance variance",
            "hit_count": "native accepted Gaussian-hit count",
        },
        "native_contributor_fields": {
            "heldout_native_ownership": ("exact held-out sum of front-to-back T*alpha ownership"),
            "depth_ambiguity_exposure": ("exact T*alpha-weighted native depth-variance exposure"),
            "hit_congestion_exposure": ("exact T*alpha-weighted native hit-count exposure"),
            "mean_hit_congestion": ("ownership-weighted mean native accepted-hit count"),
            "mean_hit_congestion_validity": ("positive held-out ownership validity mask"),
            "residual_mse_exposure": ("exact T*alpha-weighted held-out per-pixel RGB MSE control"),
        },
        "fields": fields,
        "doctor_metric_coverage": {
            "lowfreq_frac": "ray-attributed splat field",
            "training_support": ("exact native sum of training-ray T*alpha ownership"),
            "training_view_count": ("exact count of training cameras with nonzero T*alpha support"),
            "supporting_camera_count": (
                "count of training cameras meeting the recorded mean " "ownership-per-ray threshold"
            ),
            "psnr": "same local ordering as mse attribution",
            "masked_psnr": "same masked local ordering as mse attribution",
            "mae": "ray-attributed splat field",
            "mse": "ray-attributed splat field",
            "ssim": "patch-loss attribution, not literal localization",
            "lpips": "feature-loss attribution, not literal localization",
            "train_val_gap": "camera/split visualization",
            "cam_score": "camera trajectory visualization",
            "cc_gap_db": "camera and 2D residual visualization",
            "ghost_mass": "requires native depth/free-space evidence",
            "crr": "requires measured-to-rendered ray segments",
            "reprojection_residual": "camera and 2D residual visualization",
            "geom_med": "requires native Leica depth or SfM anchors",
            "cov_gap": "requires graph-supported SfM anchors",
            "leak_bias": "requires native Leica depth or SfM anchors",
            "bundle_graph_health": "camera/anchor graph visualization",
            "fgd": "global scalar; no exact per-splat field exists",
        },
    }
    manifest_path = os.path.join(output_dir, "error_splats.json")
    temporary_manifest_path = f"{manifest_path}.tmp"
    with open(
        temporary_manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_manifest_path, manifest_path)
    logger.info(f"Wrote {len(fields)} Gaussian error fields to {output_dir}.")


if __name__ == "__main__":
    main()
