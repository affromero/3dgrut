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

from threedgrut.model.model import MixtureOfGaussians
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
    model: MixtureOfGaussians,
    *,
    eval_bundle: str,
    out_dir: str,
    post_processing_mode: str,
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
        split="val",
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
    parser.add_argument("--out-dir", required=True, type=str)
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_sha256 = _sha256_of_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    conf = checkpoint["config"]
    original_training_bundle = str(conf.path)
    # Score every arm on the SAME held-out cameras + GT, GLOBAL shutter.
    conf.path = args.eval_bundle
    conf.dataset.holdout_image_list_path = args.holdout_list
    conf.dataset.shutter_type = "GLOBAL"
    conf.dataset.load_exif = False
    # Eval-only: drop training sidecars tied to the TRAINING bundle. A
    # fisheye-trained checkpoint carries dataset.sky_mask_folder=sky_masks;
    # the ERP eval bundle (and any cross-domain bundle) need not have that
    # dir, and 3DGRUT raises when it is set but absent. cc_psnr is the fair
    # metric, so sky-opacity supervision is irrelevant at eval time.
    conf.dataset.sky_mask_folder = None
    conf.loss.use_sky_opacity = False

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    renderer = _build_renderer(
        checkpoint,
        model,
        out_dir=args.out_dir,
        eval_bundle=args.eval_bundle,
        post_processing_mode=args.post_processing_mode,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        original_training_bundle=original_training_bundle,
    )
    renderer.render_all()


if __name__ == "__main__":
    main()
