# SPDX-License-Identifier: Apache-2.0
"""Evaluate a trained RS splat against the clean held-out scene (GLOBAL).

Both RS0 (global-naive) and RS1 (shutter-aware) checkpoints are scored
the SAME way: against ``scene_clean`` (clean-center frames, no
``images_end.txt``) with ``shutter_type=GLOBAL`` and the dataset holdout
set to EVERY clean frame, so the val split is the full held-out set and
both arms render at the center pose T(0). The rolling-shutter asymmetry
therefore lives only in the training data, never in the eval -- the
apples-to-apples guard for the RS0-vs-RS1 comparison.
"""

import argparse

import torch

from threedgrut.model.factory import create_gaussian_model
from threedgrut.render import Renderer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--scene-clean", required=True, type=str)
    parser.add_argument("--holdout-list", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, weights_only=False)
    conf = checkpoint["config"]
    # Force GLOBAL-at-center eval on the FULL clean held-out set.
    conf.path = args.scene_clean
    conf.dataset.holdout_image_list_path = args.holdout_list
    conf.dataset.shutter_type = "GLOBAL"

    model = create_gaussian_model(conf, checkpoint=checkpoint)
    model.init_from_checkpoint(checkpoint, setup_optimizer=False)
    model.build_acc()

    renderer = Renderer.from_preloaded_model(
        model,
        out_dir=args.out_dir,
        path=args.scene_clean,
        global_step=checkpoint["global_step"],
        split="val",
    )
    renderer.render_all()
