# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Export held-out ray-attributed Gaussian error fields as colored PLYs."""

import argparse
import hashlib
import json
import os

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from render_common_eval import _build_renderer
from threedgrut.error_attribution import (
    ErrorAttributionAccumulator,
    ErrorAttributionMetric,
    ErrorAttributionParameter,
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
        "features_albedo": "appearance",
        "features_specular": "view-dependent color",
        "positions": "position",
        "scale": "scale",
        "rotation": "rotation",
        "density": "opacity",
    }
    return f"{metric_labels[metric]} · {parameter_labels[parameter]}"


def main() -> None:
    """Export the requested held-out metric/parameter Gaussian fields."""
    args = _parse_args()
    if args.max_views < 0:
        raise ValueError("--max-views must be non-negative.")
    if args.attribution_probes <= 0:
        raise ValueError("--attribution-probes must be positive.")
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
        selected_names.append(os.path.basename(str(gpu_batch.image_path)))
        rendered_losses = ", ".join(
            f"{name}={value:.6f}" for name, value in losses.items()
        )
        logger.info(
            f"Attribution view {len(selected_names)}/{len(selected_indices)}: "
            f"{selected_names[-1]} ({rendered_losses})"
        )

    fields: list[dict[str, object]] = []
    ply_dir = os.path.join(output_dir, "ply")
    for key, scores in accumulator.rms_scores().items():
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
        "mean_losses": accumulator.mean_losses(),
        "fields": fields,
        "doctor_metric_coverage": {
            "lowfreq_frac": "ray-attributed splat field",
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
