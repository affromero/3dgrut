# SPDX-License-Identifier: Apache-2.0
"""Render a trained checkpoint through a COMMON held-out eval bundle.

The same-domain re-render for the X5 equirect-vs-fisheye A/B: each arm's
trained Gaussian field is scored against the IDENTICAL held-out cameras
and GT by overriding the checkpoint's dataset ``path`` + name-based
holdout list, so PSNR/SSIM/LPIPS are comparable ACROSS camera
representations (an ERP-trained field rendered through the held-out
fisheye cameras, and vice versa -- the field is camera-agnostic
geometry+SH, so this is well defined in 3DGUT).

Raw field rendering remains the default because learned camera corrections
need not be valid across different eval bundles. ``--post-processing-mode
inference`` restores exact checkpoint state only when camera identities match
and PPISP controller training is proven. It applies a sealed novel-view
contract: ``frame_idx=-1``, ``sequence_idx=-1``, and no EXIF exposure.
``inference_sequence_metadata`` is a separate temporal LuminanceAffine arm
conditioned only on the dataset sequence index parsed from the image name; it
still uses ``frame_idx=-1`` and no EXIF. The
``ppisp_camera_only_diagnostic`` mode is explicitly noncanonical and applies
only checkpoint CRF/vignette while forcing novel-view exposure/color identity.
Eval always uses ``shutter_type=GLOBAL`` and disables EXIF loading.

Mirrors ``render_rs_eval.py`` but takes an arbitrary eval bundle +
holdout list and enables the extra (SSIM/LPIPS + cc_*) metrics.
"""

import argparse
import os

import torch

from threedgrut.datasets.utils import (
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
)
from threedgrut.model.factory import create_gaussian_model
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.view_conditioned_anchor_field import (
    ViewConditionedAnchorField,
)
from threedgrut.render import (
    POST_PROCESSING_EVAL_MODE_INFERENCE,
    POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
    POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC,
    POST_PROCESSING_EVAL_MODE_RAW,
    POST_PROCESSING_EVAL_MODES,
    POST_PROCESSING_SOURCE_CHECKPOINT,
    POST_PROCESSING_SOURCE_NONE,
    Renderer,
    _sha256_of_file,
    configure_luminance_affine_sequence_metadata,
    configure_ppisp_camera_only_diagnostic,
    load_checkpoint_post_processing,
    upgrade_legacy_checkpoint_config,
)


def _post_processing_for_mode(
    checkpoint: dict,
    mode: str,
) -> torch.nn.Module | None:
    if mode == POST_PROCESSING_EVAL_MODE_RAW:
        return None
    if mode not in (
        POST_PROCESSING_EVAL_MODE_INFERENCE,
        POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
        POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC,
    ):
        raise ValueError(
            f"Unsupported post-processing mode {mode!r}. Expected one of " f"{POST_PROCESSING_EVAL_MODES}."
        )
    post_processing = load_checkpoint_post_processing(
        checkpoint,
        require_controller_ready=(mode == POST_PROCESSING_EVAL_MODE_INFERENCE),
    )
    if post_processing is None:
        raise ValueError(
            "Inference post-processing was requested, but the checkpoint " "does not contain post-processing state."
        )
    if mode == POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA:
        return configure_luminance_affine_sequence_metadata(post_processing)
    if mode == POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC:
        return configure_ppisp_camera_only_diagnostic(post_processing)
    return post_processing


def _build_renderer(
    checkpoint: dict,
    model: MixtureOfGaussians | ViewConditionedAnchorField,
    *,
    eval_bundle: str,
    out_dir: str,
    post_processing_mode: str,
    split: str = "val",
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
    original_training_bundle: str | None = None,
) -> Renderer:
    post_processing = _post_processing_for_mode(
        checkpoint,
        post_processing_mode,
    )
    return Renderer.from_preloaded_model(
        model,
        out_dir=out_dir,
        path=eval_bundle,
        global_step=checkpoint["global_step"],
        split=split,
        compute_extra_metrics=True,
        post_processing=post_processing,
        post_processing_mode=post_processing_mode,
        post_processing_source=(
            POST_PROCESSING_SOURCE_CHECKPOINT if post_processing is not None else POST_PROCESSING_SOURCE_NONE
        ),
        strict_post_processing_contract=(post_processing is not None),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        original_training_bundle=original_training_bundle,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--eval-bundle", required=True, type=str)
    parser.add_argument("--holdout-list", required=True, type=str)
    parser.add_argument(
        "--train-exclude-list",
        type=str,
        help=(
            "Required with --split train. Names embargoed from the "
            "holdout complement before train metrics are computed."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help=(
            "Render the held-out validation split or the complementary "
            "training split under the same sealed inference contract."
        ),
    )
    parser.add_argument(
        "--post-processing-mode",
        choices=POST_PROCESSING_EVAL_MODES,
        default=POST_PROCESSING_EVAL_MODE_RAW,
        help=(
            "raw scores the Gaussian field only; inference is sealed and "
            "requires a proven controller; inference_sequence_metadata "
            "uses only temporal LuminanceAffine filename sequence metadata; "
            "ppisp_camera_only_diagnostic keeps only CRF/vignette and is "
            "noncanonical"
        ),
    )
    args = parser.parse_args()
    if args.split == "train" and args.train_exclude_list is None:
        parser.error(
            "--split train requires --train-exclude-list so embargoed "
            "frames cannot enter train metrics."
        )
    if args.split == "val" and args.train_exclude_list is not None:
        parser.error(
            "--train-exclude-list is only valid with --split train."
        )
    return args


def _read_unique_image_names(path: str, *, label: str) -> set[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    names: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    if not names:
        raise ValueError(f"{label} must contain at least one image name.")
    unique_names = set(names)
    if len(unique_names) != len(names):
        raise ValueError(f"{label} contains duplicate image names.")
    return unique_names


def _registered_image_names(eval_bundle: str) -> set[str]:
    sparse_path = os.path.join(eval_bundle, "sparse", "0")
    binary_path = os.path.join(sparse_path, "images.bin")
    text_path = os.path.join(sparse_path, "images.txt")
    if os.path.isfile(binary_path):
        extrinsics = read_colmap_extrinsics_binary(binary_path)
    elif os.path.isfile(text_path):
        extrinsics = read_colmap_extrinsics_text(text_path)
    else:
        raise FileNotFoundError(
            "Eval bundle has no COLMAP images.bin or images.txt: "
            f"{sparse_path}"
        )
    names = [os.path.basename(extrinsic.name) for extrinsic in extrinsics]
    unique_names = set(names)
    if len(unique_names) != len(names):
        raise ValueError(
            "Eval bundle image basenames are not unique; train exclusions "
            "would be ambiguous."
        )
    return unique_names


def _validate_train_split_contract(
    *,
    eval_bundle: str,
    holdout_list: str,
    train_exclude_list: str,
) -> None:
    registered = _registered_image_names(eval_bundle)
    holdout = _read_unique_image_names(
        holdout_list,
        label="Holdout image list",
    )
    excluded = _read_unique_image_names(
        train_exclude_list,
        label="Train exclude image list",
    )
    unknown_holdout = holdout - registered
    if unknown_holdout:
        raise ValueError(
            "Holdout image list contains unregistered names: "
            f"{sorted(unknown_holdout)[:5]}"
        )
    overlap = excluded & holdout
    if overlap:
        raise ValueError(
            "Train exclude image list overlaps the holdout: "
            f"{sorted(overlap)[:5]}"
        )
    unknown_excluded = excluded - registered
    if unknown_excluded:
        raise ValueError(
            "Train exclude image list contains unregistered names: "
            f"{sorted(unknown_excluded)[:5]}"
        )
    remaining = registered - holdout - excluded
    if not remaining:
        raise ValueError(
            "Train exclusion contract leaves no images for train metrics."
        )


def main() -> None:
    args = _parse_args()
    if args.split == "train":
        _validate_train_split_contract(
            eval_bundle=args.eval_bundle,
            holdout_list=args.holdout_list,
            train_exclude_list=args.train_exclude_list,
        )

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_sha256 = _sha256_of_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = upgrade_legacy_checkpoint_config(checkpoint["config"])
    original_training_bundle = str(conf.path)
    # Score every arm on the SAME held-out cameras + GT, GLOBAL shutter.
    conf.path = args.eval_bundle
    conf.dataset.holdout_image_list_path = args.holdout_list
    conf.dataset.train_exclude_image_list_path = (
        args.train_exclude_list if args.split == "train" else None
    )
    conf.dataset.shutter_type = "GLOBAL"
    conf.dataset.load_exif = False
    # Eval-only: drop training sidecars tied to the TRAINING bundle. A
    # fisheye-trained checkpoint carries dataset.sky_mask_folder=sky_masks;
    # the ERP eval bundle (and any cross-domain bundle) need not have that
    # dir, and 3DGRUT raises when it is set but absent. cc_psnr is the fair
    # metric, so sky-opacity supervision is irrelevant at eval time.
    conf.dataset.sky_mask_folder = None
    conf.loss.use_sky_opacity = False

    model = create_gaussian_model(conf, checkpoint=checkpoint)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    renderer = _build_renderer(
        checkpoint,
        model,
        out_dir=args.out_dir,
        eval_bundle=args.eval_bundle,
        post_processing_mode=args.post_processing_mode,
        split=args.split,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        original_training_bundle=original_training_bundle,
    )
    renderer.render_all()


if __name__ == "__main__":
    main()
