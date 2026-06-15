# SPDX-License-Identifier: Apache-2.0
"""Render a trained checkpoint through a COMMON held-out eval bundle.

The same-domain re-render for the X5 equirect-vs-fisheye A/B: each arm's
trained Gaussian field is scored against the IDENTICAL held-out cameras
and GT by overriding the checkpoint's dataset ``path`` + name-based
holdout list, so PSNR/SSIM/LPIPS are comparable ACROSS camera
representations (an ERP-trained field rendered through the held-out
fisheye cameras, and vice versa -- the field is camera-agnostic
geometry+SH, so this is well defined in 3DGUT).

The trained per-camera ``LuminanceAffine`` is deliberately NOT applied:
its gain table is indexed by the TRAINING set's cameras/frames, which do
not correspond to the held-out frames of a *different* bundle. The fair,
arm-symmetric metric is therefore the test-time color-corrected
``cc_psnr`` / ``cc_ssim`` / ``cc_lpips`` that ``render_all`` computes via
``color_correct_affine`` (a per-image affine fit between render and GT).
Eval always uses ``shutter_type=GLOBAL`` so the rolling-shutter
asymmetry lives only in training, never in the comparison.

Mirrors ``render_rs_eval.py`` but takes an arbitrary eval bundle +
holdout list and enables the extra (SSIM/LPIPS + cc_*) metrics.
"""

import argparse

import torch

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.render import Renderer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--eval-bundle", required=True, type=str)
    parser.add_argument("--holdout-list", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    # Score every arm on the SAME held-out cameras + GT, GLOBAL shutter.
    conf.path = args.eval_bundle
    conf.dataset.holdout_image_list_path = args.holdout_list
    conf.dataset.shutter_type = "GLOBAL"

    model = MixtureOfGaussians(conf)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    renderer = Renderer.from_preloaded_model(
        model,
        out_dir=args.out_dir,
        path=args.eval_bundle,
        global_step=checkpoint["global_step"],
        split="val",
        compute_extra_metrics=True,
    )
    renderer.render_all()
