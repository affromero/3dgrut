# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Export held-out ray-attributed Gaussian error fields as colored PLYs."""

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from klogr.path import (
    path_basename,
    path_join,
    path_mkdir,
    path_relative_to,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from render_common_eval import _build_renderer
from threedgrut.datasets.protocols import Batch
from threedgrut.error_attribution import (
    ErrorAttributionAccumulator,
    ErrorAttributionMetric,
    ErrorAttributionParameter,
    attribution_loss,
    heldout_ownership_dominance,
    native_contributor_ray_fields,
    native_render_evidence_maps,
    native_structural_gaussian_fields,
    recolor_gaussian_ply,
)
from threedgrut.model.factory import create_gaussian_model
from threedgrut.render import POST_PROCESSING_EVAL_MODE_RAW
from threedgrut.utils.logger import logger

DEFAULT_VISIBILITY_THRESHOLD = 0.0


def _enum_values(enum_type: type[ErrorAttributionMetric]) -> tuple[str, ...]:
    return tuple(member.value for member in enum_type)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-ply", required=True)
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
        "--export-fields",
        nargs="+",
        default=None,
        metavar="METRIC:PARAMETER",
        help=(
            "Optional subset of the computed metric/parameter cross-product "
            "to materialize as PLY files."
        ),
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
        "--counterfactual-suppression-logit",
        type=float,
        default=-20.0,
        help="Pre-sigmoid density value assigned to a suppressed cohort.",
    )
    return parser.parse_args()


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_relative_path(path: str, scene_root: str) -> str:
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(scene_root)
    if os.path.commonpath((resolved_path, resolved_root)) != resolved_root:
        raise ValueError(f"Path escapes --scene-root: {path}")
    return os.path.relpath(resolved_path, resolved_root)


def _sample_indices(count: int, maximum: int) -> set[int]:
    if maximum <= 0 or maximum >= count:
        return set(range(count))
    if maximum == 1:
        return {count // 2}
    return {
        round(index * (count - 1) / (maximum - 1)) for index in range(maximum)
    }


def _field_label(metric: str, parameter: str) -> str:
    metric_labels = {
        "mae": "MAE",
        "mse": "MSE / PSNR",
        "ssim": "SSIM loss",
        "lpips": "LPIPS",
        "lowfreq_frac": "Doctor low-frequency residual",
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
        )
    native_dir = path_join(output_dir, "maps", "native")
    path_mkdir(native_dir, parents=True, exist_ok=True)
    image_hash = hashlib.sha256(image_name.encode("utf-8")).hexdigest()[:16]
    fields = {
        name: value.detach().squeeze(0).squeeze(-1).float().cpu().numpy()
        for name, value in evidence.items()
    }
    np.savez_compressed(
        path_join(native_dir, f"map_{image_hash}.npz"),
        image_name=np.array(image_name),
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
        scores[field_id] += weighted_sum.detach().float().cpu().reshape_as(
            scores[field_id]
        )


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
    narrow_view_coverage = torch.zeros(
        (model.density.shape[0], 3),
        dtype=torch.float32,
        device="cpu",
    )
    selected_names: list[str] = []
    logger.info(
        "Computing native per-Gaussian training support for "
        f"{len(selected_indices)} views."
    )
    for index, batch in enumerate(renderer.dataloader):
        if index not in selected_indices:
            continue
        gpu_batch = renderer.dataset.get_gpu_batch_with_intrinsics(batch)
        ray_diagnostic = torch.ones(
            gpu_batch.rays_ori.shape[1:3],
            device=gpu_batch.rays_ori.device,
            dtype=torch.float32,
        )
        responsibility = model.render_responsibility(
            gpu_batch,
            ray_diagnostic,
        )["responsibility"].detach().float().cpu().reshape_as(support)
        support += responsibility
        visible_view_count += (responsibility > 0.0).float()
        camera_center = gpu_batch.T_to_world[0, :3, 3].detach()
        directions = camera_center.unsqueeze(0) - model.positions.detach()
        directions = directions / torch.linalg.vector_norm(
            directions,
            dim=-1,
            keepdim=True,
        ).clamp_min(torch.finfo(directions.dtype).eps)
        narrow_view_coverage += (
            responsibility.to(device=directions.device)
            * directions
        ).detach().float().cpu()
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
        fields.append(
            {
                "id": field_id,
                "label": label,
                "metric_id": metric_id,
                "parameter": field_id,
                "artifact_kind": "ply",
                "filename": path_relative_to(output_path, output_dir),
                "artifact_sha256": _sha256(output_path),
                "normalization": scale_mode,
                "visibility_threshold": visibility_threshold,
                "statistics": statistics,
                "exact_localization": True,
                "attribution": attribution,
            }
        )
    return fields, len(selected_names), selected_names, support


def _counterfactual_cohorts(
    scores: torch.Tensor,
    cohort_size: int,
) -> dict[str, torch.Tensor]:
    """Return deterministic top, random, and low-score index controls."""
    if cohort_size <= 0:
        raise ValueError("counterfactual cohort size must be positive.")
    flattened = torch.nan_to_num(
        scores.detach().reshape(-1).float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if flattened.numel() == 0:
        raise ValueError("Counterfactual scores are empty.")
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
def _suppressed_density_cohort(
    model: torch.nn.Module,
    indices: torch.Tensor,
    suppression_logit: float,
) -> Iterator[None]:
    """Temporarily remove a Gaussian cohort from native alpha compositing."""
    density = getattr(model, "density", None)
    if not isinstance(density, torch.nn.Parameter):
        raise TypeError("Counterfactual export requires model.density parameter.")
    selected = indices.to(device=density.device, dtype=torch.long)
    with torch.no_grad():
        original = density.index_select(0, selected).clone()
        density.index_fill_(0, selected, suppression_logit)
    try:
        yield
    finally:
        with torch.no_grad():
            density.index_copy_(0, selected, original)


def _mean_heldout_mse(
    *,
    model: torch.nn.Module,
    renderer: object,
    selected_indices: set[int],
) -> float:
    """Rerender selected validation views and report their mean RGB MSE."""
    dataset = getattr(renderer, "dataset", None)
    dataloader = getattr(renderer, "dataloader", None)
    if dataset is None or dataloader is None:
        raise TypeError("Counterfactual renderer must expose dataset and dataloader.")
    values: list[float] = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if index not in selected_indices:
                continue
            gpu_batch = dataset.get_gpu_batch_with_intrinsics(batch)
            outputs = model(gpu_batch, train=False)
            prediction = outputs.get("pred_rgb")
            if not isinstance(prediction, torch.Tensor):
                raise KeyError("Renderer output has no pred_rgb.")
            mse = attribution_loss(
                ErrorAttributionMetric.MSE,
                prediction,
                gpu_batch.rgb_gt,
                gpu_batch.mask,
            )
            values.append(float(mse.detach()))
    if not values:
        raise ValueError("No held-out views were selected for intervention.")
    return float(np.mean(values))


def _evaluate_density_counterfactuals(
    *,
    model: torch.nn.Module,
    renderer: object,
    selected_indices: set[int],
    density_scores: torch.Tensor,
    cohort_size: int,
    suppression_logit: float,
) -> dict[str, object]:
    """Measure actual held-out MSE changes under matched density controls."""
    baseline_mse = _mean_heldout_mse(
        model=model,
        renderer=renderer,
        selected_indices=selected_indices,
    )
    cohorts = _counterfactual_cohorts(density_scores, cohort_size)
    evaluations: list[dict[str, object]] = []
    for name, indices in cohorts.items():
        with _suppressed_density_cohort(model, indices, suppression_logit):
            mse = _mean_heldout_mse(
                model=model,
                renderer=renderer,
                selected_indices=selected_indices,
            )
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


def main() -> None:
    """Export the requested held-out metric/parameter Gaussian fields."""
    args = _parse_args()
    if args.max_views < 0:
        raise ValueError("--max-views must be non-negative.")
    if args.training_support_max_views < 0:
        raise ValueError("--training-support-max-views must be non-negative.")
    if args.attribution_probes <= 0:
        raise ValueError("--attribution-probes must be positive.")
    if args.counterfactual_cohort_size < 0:
        raise ValueError("--counterfactual-cohort-size must be non-negative.")
    if not 0.0 <= args.visibility_threshold < 1.0:
        raise ValueError("--visibility-threshold must be in [0, 1).")
    checkpoint_path = os.path.abspath(args.checkpoint)
    source_ply = os.path.abspath(args.source_ply)
    eval_bundle = os.path.abspath(args.eval_bundle)
    output_dir = os.path.abspath(args.output_dir)
    scene_root = os.path.abspath(args.scene_root)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
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
    parameters = tuple(
        ErrorAttributionParameter(value) for value in args.parameters
    )
    available_field_keys = {
        f"{metric.value}:{parameter.value}"
        for metric in metrics
        for parameter in parameters
    }
    export_field_keys = (
        set(args.export_fields)
        if args.export_fields is not None
        else available_field_keys
    )
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
    }
    selected_names: list[str] = []
    logger.info(
        "Computing ray-attributed error fields for "
        f"{len(selected_indices)} held-out views."
    )
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
        rendered_losses = ", ".join(
            f"{name}={value:.6f}" for name, value in losses.items()
        )
        logger.info(
            f"Attribution view {len(selected_names)}/{len(selected_indices)}: "
            f"{selected_names[-1]} ({rendered_losses})"
        )

    rms_scores = accumulator.rms_scores()
    counterfactual: dict[str, object] | None = None
    if args.counterfactual_cohort_size > 0:
        density_key = "mse:density"
        density_scores = rms_scores.get(density_key)
        if density_scores is None:
            raise ValueError(
                "Density counterfactuals require --metrics mse and "
                "--parameters density."
            )
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
        fields.append(
            {
                "id": f"{metric}__{parameter}",
                "label": _field_label(metric, parameter),
                "metric_id": metric,
                "parameter": parameter,
                "artifact_kind": "ply",
                "filename": os.path.relpath(output_path, output_dir),
                "artifact_sha256": artifact_sha256,
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
            "Exact T*alpha-weighted exposure to native accepted-hit count; "
            "not an intrinsic per-Gaussian hit count.",
        ),
    )
    for field_id, label, attribution in native_contributor_specs:
        output_path = os.path.join(ply_dir, f"{field_id}.ply")
        statistics = recolor_gaussian_ply(
            source_path=source_ply,
            output_path=output_path,
            scores=native_contributor_scores[field_id].reshape(-1),
            scale_mode=args.normalization,
            expected_density=model.density,
            visibility_threshold=args.visibility_threshold,
        )
        fields.append(
            {
                "id": field_id,
                "label": label,
                "metric_id": "native_ray_exposure",
                "parameter": field_id,
                "artifact_kind": "ply",
                "filename": os.path.relpath(output_path, output_dir),
                "artifact_sha256": _sha256(output_path),
                "normalization": args.normalization,
                "visibility_threshold": args.visibility_threshold,
                "statistics": statistics,
                "exact_localization": True,
                "attribution": attribution,
            }
        )

    (
        training_fields,
        training_support_view_count,
        training_support_images,
        training_support_scores,
    ) = (
        _export_training_support_fields(
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
        )
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
        fields.append(
            {
                "id": field_id,
                "label": label,
                "metric_id": "structural_candidate",
                "parameter": field_id,
                "artifact_kind": "ply",
                "filename": os.path.relpath(output_path, output_dir),
                "artifact_sha256": _sha256(output_path),
                "normalization": args.normalization,
                "visibility_threshold": args.visibility_threshold,
                "statistics": statistics,
                "exact_localization": True,
                "attribution": attribution,
            }
        )
    manifest = {
        "schema_version": 3,
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
        "visibility_threshold": args.visibility_threshold,
        "selected_images": selected_names,
        "training_support_view_count": training_support_view_count,
        "training_support_images": training_support_images,
        "mean_losses": accumulator.mean_losses(),
        "counterfactual_intervention": (
            None
            if counterfactual is None
            else {
                "filename": "interventions.json",
                "artifact_sha256": _sha256(
                    path_join(output_dir, "interventions.json")
                ),
                "contents": counterfactual,
            }
        ),
        "native_evidence_maps": {
            "accumulated_alpha": "native front-to-back accumulated alpha",
            "expected_depth": "native conditional hit-distance mean",
            "depth_variance": "native conditional hit-distance variance",
            "hit_count": "native accepted Gaussian-hit count",
        },
        "native_contributor_fields": {
            "heldout_native_ownership": (
                "exact held-out sum of front-to-back T*alpha ownership"
            ),
            "depth_ambiguity_exposure": (
                "exact T*alpha-weighted native depth-variance exposure"
            ),
            "hit_congestion_exposure": (
                "exact T*alpha-weighted native hit-count exposure"
            ),
        },
        "fields": fields,
        "doctor_metric_coverage": {
            "lowfreq_frac": "ray-attributed splat field",
            "training_support": (
                "exact native sum of training-ray T*alpha ownership"
            ),
            "training_view_count": (
                "exact count of training cameras with nonzero T*alpha support"
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
    with open(
        os.path.join(output_dir, "error_splats.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    logger.info(f"Wrote {len(fields)} Gaussian error fields to {output_dir}.")


if __name__ == "__main__":
    main()
