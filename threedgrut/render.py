# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from typing import Optional

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.post_processing import LuminanceAffine
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer
from threedgrut.utils.render import (
    apply_post_processing,
    post_processing_camera_idx,
    post_processing_camera_index_mode,
)


def _load_luminance_affine_state_compat(
    module: LuminanceAffine,
    saved_state: dict,
) -> list[str]:
    current_state = module.state_dict()
    filtered: dict = {}
    dropped: list[str] = []
    for key, value in saved_state.items():
        target = current_state.get(key)
        if target is None:
            continue
        if (
            hasattr(value, "shape")
            and hasattr(target, "shape")
            and tuple(value.shape) != tuple(target.shape)
        ):
            if (
                key == "residual_grid"
                and value.ndim == 4
                and target.ndim == 4
                and value.shape[:2] == target.shape[:2]
            ):
                filtered[key] = F.interpolate(
                    value.to(device=target.device, dtype=target.dtype),
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                continue
            dropped.append(key)
            continue
        filtered[key] = value
    module.load_state_dict(filtered, strict=False)
    return dropped


def _git_sha_and_dirty(repo_dir: str) -> tuple[str, bool]:
    """Return (HEAD sha, dirty) for the git repo containing ``repo_dir``.

    Returns ("Unknown", False) if ``repo_dir`` is not inside a git work tree
    or git is unavailable, so provenance never crashes a render.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "-C", repo_dir, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        status = subprocess.check_output(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        return sha, bool(status.strip())
    except Exception:
        return "Unknown", False


def _sha256_of_file(path: Optional[str]) -> Optional[str]:
    """Return the hex sha256 of ``path``, or None if missing/unreadable."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Renderer:
    def __init__(
        self,
        model,
        conf,
        global_step,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        compute_extra_metrics=True,
        post_processing=None,
        split="val",
    ) -> None:
        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.split = split
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics
        self.post_processing = post_processing
        self._post_processing_camera_index_mode = (
            post_processing_camera_index_mode(conf)
        )

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros(
                (3,), dtype=torch.float32, device="cuda"
            )
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones(
                (3,), dtype=torch.float32, device="cuda"
            )
        else:
            assert False, (
                f"{conf.model.background.color} is not a supported background color."
            )

    def create_test_dataloader(self, conf):
        """Create the requested render dataloader for the configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        if self.split == "train":
            dataset, _ = datasets.make(
                name=conf.dataset.type,
                config=conf,
                ray_jitter=None,
            )
        elif self.split == "val":
            dataset = datasets.make_test(name=conf.dataset.type, config=conf)
        else:
            raise ValueError(
                f"Unsupported render split {self.split!r}. Expected train or val."
            )

        # Configure DataLoader arguments for the current platform
        dataloader_kwargs = configure_dataloader_for_platform(
            {
                "num_workers": 8,
                "batch_size": 1,
                "shuffle": False,
                "collate_fn": None,
            }
        )

        dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        model=None,
        computes_extra_metrics=True,
        split="val",
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(
            conf, object_name, out_dir, experiment_name, use_wandb=False
        )

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        model.build_acc()

        # Load post-processing if present in checkpoint
        post_processing = None
        method = conf.post_processing.method
        if "post_processing" in checkpoint and method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            # Derive config from training settings to match trainer.py
            use_controller = conf.post_processing.get("use_controller", True)
            n_distillation_steps = conf.post_processing.get(
                "n_distillation_steps", 5000
            )
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = (
                    main_training_steps / conf.n_iterations
                )
                controller_distillation = True
            elif use_controller:
                controller_activation_ratio = 0.8
                controller_distillation = False
            else:
                controller_activation_ratio = 0.0
                controller_distillation = False

            ppisp_config = PPISPConfig(
                use_controller=use_controller,
                controller_distillation=controller_distillation,
                controller_activation_ratio=controller_activation_ratio,
            )

            post_processing = PPISP.from_state_dict(
                checkpoint["post_processing"]["module"], config=ppisp_config
            )
            post_processing = post_processing.to("cuda")
            num_cameras = post_processing.crf_params.shape[0]
            num_frames = post_processing.exposure_params.shape[0]
            logger.info(
                f"📷 {method.upper()} loaded from checkpoint: {num_cameras} cameras, {num_frames} frames"
            )
        elif "post_processing" in checkpoint and method == "luminance_affine":
            state = checkpoint["post_processing"]["module"]
            num_cameras = state["camera_log_gain"].shape[0]
            num_frames = state["frame_log_gain"].shape[0]
            post_processing = LuminanceAffine(
                num_cameras=num_cameras,
                num_frames=num_frames,
                lr=conf.post_processing.get("lr", 1e-3),
                reg_lambda=conf.post_processing.get("reg_lambda", 1e-2),
                use_frame_residual=conf.post_processing.get(
                    "use_frame_residual",
                    False,
                ),
                max_log_gain=conf.post_processing.get("max_log_gain", 0.25),
                max_bias=conf.post_processing.get("max_bias", 0.10),
                use_color_matrix=conf.post_processing.get(
                    "use_color_matrix",
                    False,
                ),
                max_matrix_delta=conf.post_processing.get(
                    "max_matrix_delta",
                    0.10,
                ),
                color_matrix_reg_lambda=conf.post_processing.get(
                    "color_matrix_reg_lambda",
                    0.25,
                ),
                use_radial_affine=conf.post_processing.get(
                    "use_radial_affine",
                    False,
                ),
                radial_band_count=conf.post_processing.get(
                    "radial_band_count",
                    4,
                ),
                radial_max_log_gain=conf.post_processing.get(
                    "radial_max_log_gain",
                    0.08,
                ),
                radial_max_bias=conf.post_processing.get(
                    "radial_max_bias",
                    0.03,
                ),
                radial_reg_lambda=conf.post_processing.get(
                    "radial_reg_lambda",
                    0.50,
                ),
                use_residual_grid=conf.post_processing.get(
                    "use_residual_grid",
                    False,
                ),
                residual_grid_size=conf.post_processing.get(
                    "residual_grid_size",
                    32,
                ),
                residual_grid_max=conf.post_processing.get(
                    "residual_grid_max",
                    0.05,
                ),
                residual_grid_reg_lambda=conf.post_processing.get(
                    "residual_grid_reg_lambda",
                    0.01,
                ),
                use_residual_grid_edge_gate=conf.post_processing.get(
                    "use_residual_grid_edge_gate",
                    False,
                ),
                residual_grid_gate_floor=conf.post_processing.get(
                    "residual_grid_gate_floor",
                    0.20,
                ),
                use_temporal_affine=conf.post_processing.get(
                    "use_temporal_affine",
                    False,
                ),
                temporal_num_knots=conf.post_processing.get(
                    "temporal_num_knots",
                    32,
                ),
                temporal_max_sequence_idx=conf.post_processing.get(
                    "temporal_max_sequence_idx",
                    400,
                ),
                temporal_max_log_gain=conf.post_processing.get(
                    "temporal_max_log_gain",
                    0.08,
                ),
                temporal_max_bias=conf.post_processing.get(
                    "temporal_max_bias",
                    0.03,
                ),
                temporal_reg_lambda=conf.post_processing.get(
                    "temporal_reg_lambda",
                    0.50,
                ),
            ).to("cuda")
            dropped = _load_luminance_affine_state_compat(
                post_processing,
                state,
            )
            if dropped:
                logger.warning(
                    "Dropping shape-mismatched luminance-affine buffers "
                    f"during render restore: {sorted(dropped)}."
                )
            logger.info(
                f"📷 {method.upper()} loaded from checkpoint: "
                f"{num_cameras} cameras, {num_frames} frames"
            )

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
            post_processing=post_processing,
            split=split,
        )

    @classmethod
    def from_preloaded_model(
        cls,
        model,
        out_dir,
        path="",
        save_gt=True,
        writer=None,
        global_step=None,
        compute_extra_metrics=False,
        post_processing=None,
        split=None,
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
            post_processing=post_processing,
            split=split or conf.render.get("split", "val"),
        )

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to(
                    "cuda"
                ),
                "lpips": LearnedPerceptualImagePatchSimilarity(
                    net_type="vgg", normalize=True
                ).to("cuda"),
            }

        render_leaf = "renders" if self.split == "val" else "train_renders"
        gt_leaf = "gt" if self.split == "val" else "train_gt"
        output_path_renders = os.path.join(
            self.out_dir, f"ours_{int(self.global_step)}", render_leaf
        )
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(
                self.out_dir, f"ours_{int(self.global_step)}", gt_leaf
            )
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        masked_psnr = []
        mask_coverage = []
        ssim = []
        lpips = []
        cc_psnr = []
        cc_ssim = []
        cc_lpips = []
        inference_time = []
        per_frame_metrics = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(
            task_name="Rendering",
            total_steps=len(self.dataloader),
            color="orange1",
        )

        for iteration, batch in enumerate(self.dataloader):
            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)

            # Apply post-processing
            if self.post_processing is not None:
                outputs = apply_post_processing(
                    self.post_processing,
                    outputs,
                    gpu_batch,
                    training=False,
                    camera_idx_override=post_processing_camera_idx(
                        int(
                            getattr(
                                gpu_batch,
                                "post_processing_camera_idx",
                                gpu_batch.camera_idx,
                            )
                        ),
                        self._post_processing_camera_index_mode,
                    ),
                )

            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(
                    output_path_renders, "{0:05d}".format(iteration) + ".png"
                ),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(
                        output_path_gt, "{0:05d}".format(iteration) + ".png"
                    ),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](
                outputs["pred_rgb"], gpu_batch.rgb_gt
            ).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            progress_psnr = psnr[-1]
            if gpu_batch.mask is not None:
                mask = gpu_batch.mask
                masked_error = (
                    torch.square(outputs["pred_rgb"] - gpu_batch.rgb_gt) * mask
                )
                masked_denominator = torch.clamp_min(
                    mask.sum() * gpu_batch.rgb_gt.shape[-1],
                    1.0,
                )
                masked_mse = masked_error.sum() / masked_denominator
                masked_psnr_single_img = (
                    -10.0 * torch.log10(torch.clamp_min(masked_mse, 1e-12))
                ).item()
                masked_psnr.append(masked_psnr_single_img)
                mask_coverage.append(mask.mean().item())
                progress_psnr = masked_psnr[-1]
                logger.info(
                    f"Frame {iteration}, image: {os.path.basename(gpu_batch.image_path)}, "
                    f"PSNR: {psnr[-1]}, masked PSNR: {masked_psnr[-1]}"
                )
            else:
                logger.info(
                    f"Frame {iteration}, image: {os.path.basename(gpu_batch.image_path)}, "
                    f"PSNR: {psnr[-1]}"
                )

            frame_metrics = {
                "eval_index": int(iteration),
                "split_frame_idx": int(gpu_batch.frame_idx),
                "split": self.split,
                "camera_idx": int(gpu_batch.camera_idx),
                "image_name": os.path.basename(gpu_batch.image_path),
                "image_path": gpu_batch.image_path,
                "psnr": float(psnr_single_img),
            }
            if masked_psnr:
                frame_metrics["masked_psnr"] = float(masked_psnr[-1])
            if mask_coverage:
                frame_metrics["mask_coverage"] = float(mask_coverage[-1])
            per_frame_metrics.append(frame_metrics)

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            if self.compute_extra_metrics:
                # evaluate on full image
                ssim.append(
                    criterions["ssim"](
                        pred_rgb_full.permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )
                lpips.append(
                    criterions["lpips"](
                        pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )

            # Color-corrected metrics
            pred_rgb_cc = color_correct_affine(pred_rgb_full, rgb_gt_full)
            cc_psnr.append(
                criterions["psnr"](pred_rgb_cc, rgb_gt_full).item()
            )
            if self.compute_extra_metrics:
                cc_ssim.append(
                    criterions["ssim"](
                        pred_rgb_cc.permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )
                cc_lpips.append(
                    criterions["lpips"](
                        pred_rgb_cc.clip(0, 1).permute(0, 3, 1, 2),
                        rgb_gt_full.permute(0, 3, 1, 2),
                    ).item()
                )

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(
                task_name="Rendering",
                advance=1,
                iteration=f"{str(iteration)}",
                psnr=progress_psnr,
            )

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_masked_psnr = np.mean(masked_psnr) if masked_psnr else None
        mean_mask_coverage = np.mean(mask_coverage) if mask_coverage else None
        mean_ssim = np.mean(ssim) if ssim else None
        mean_lpips = np.mean(lpips) if lpips else None
        mean_cc_psnr = np.mean(cc_psnr) if cc_psnr else None
        mean_cc_ssim = np.mean(cc_ssim) if cc_ssim else None
        mean_cc_lpips = np.mean(cc_lpips) if cc_lpips else None
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(mean_psnr=mean_psnr, std_psnr=std_psnr)
        if mean_masked_psnr is not None:
            table["mean_masked_psnr"] = mean_masked_psnr
        if mean_mask_coverage is not None:
            table["mean_mask_coverage"] = mean_mask_coverage
        if mean_ssim is not None:
            table["mean_ssim"] = mean_ssim
        if mean_lpips is not None:
            table["mean_lpips"] = mean_lpips
        if mean_cc_psnr is not None:
            table["mean_cc_psnr"] = mean_cc_psnr
        if mean_cc_ssim is not None:
            table["mean_cc_ssim"] = mean_cc_ssim
        if mean_cc_lpips is not None:
            table["mean_cc_lpips"] = mean_cc_lpips

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = (
                f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"
            )

        # Save metrics to JSON file
        metrics_json = dict(mean_psnr=float(mean_psnr))
        if mean_masked_psnr is not None:
            metrics_json["mean_masked_psnr"] = float(mean_masked_psnr)
        if mean_mask_coverage is not None:
            metrics_json["mean_mask_coverage"] = float(mean_mask_coverage)
        if mean_ssim is not None:
            metrics_json["mean_ssim"] = float(mean_ssim)
        if mean_lpips is not None:
            metrics_json["mean_lpips"] = float(mean_lpips)
        if mean_cc_psnr is not None:
            metrics_json["mean_cc_psnr"] = float(mean_cc_psnr)
        if mean_cc_ssim is not None:
            metrics_json["mean_cc_ssim"] = float(mean_cc_ssim)
        if mean_cc_lpips is not None:
            metrics_json["mean_cc_lpips"] = float(mean_cc_lpips)
        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"📄 Metrics saved to: {metrics_path}")
        per_frame_metrics_name = (
            "per_frame_metrics.json"
            if self.split == "val"
            else "per_frame_train_metrics.json"
        )
        per_frame_metrics_path = os.path.join(self.out_dir, per_frame_metrics_name)
        with open(per_frame_metrics_path, "w") as f:
            json.dump(per_frame_metrics, f, indent=2)
        logger.info(f"📄 Per-frame metrics saved to: {per_frame_metrics_path}")

        # Provenance sidecar: pin exactly which code + seed + holdout produced
        # these metrics so a run is reproducible and auditable.
        submodule_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(
            os.path.join(submodule_dir, os.pardir, os.pardir, os.pardir)
        )
        submodule_sha, submodule_dirty = _git_sha_and_dirty(submodule_dir)
        parent_sha, parent_dirty = _git_sha_and_dirty(parent_dir)
        holdout_path = getattr(
            self.dataset, "holdout_image_list_path", None
        )
        evaluated_frames = [
            fm["image_name"] for fm in per_frame_metrics if "image_name" in fm
        ]
        provenance = {
            "submodule_git_sha": submodule_sha,
            "submodule_git_dirty": submodule_dirty,
            "parent_git_sha": parent_sha,
            "parent_git_dirty": parent_dirty,
            "experiment_name": self.conf.get("experiment_name", ""),
            "config_path": self.conf.get("path", ""),
            "n_iterations": int(self.conf.get("n_iterations", 0)),
            "global_step": int(self.global_step),
            "seed_initialization": int(
                self.conf.get("seed_initialization", -1)
            ),
            "split": self.split,
            "holdout_image_list_path": holdout_path,
            "holdout_image_list_sha256": _sha256_of_file(holdout_path),
            "evaluated_frame_count": len(evaluated_frames),
            "evaluated_frame_names": evaluated_frames,
        }
        provenance_path = os.path.join(self.out_dir, "provenance.json")
        with open(provenance_path, "w") as f:
            json.dump(provenance, f, indent=2)
        logger.info(f"📄 Provenance saved to: {provenance_path}")

        logger.log_table(
            f"⭐ Test Metrics - Step {self.global_step}", record=table
        )

        if self.writer is not None:
            self.writer.add_scalar("test/psnr", mean_psnr, self.global_step)
            if mean_masked_psnr is not None:
                self.writer.add_scalar(
                    "test/masked_psnr", mean_masked_psnr, self.global_step
                )
            if mean_mask_coverage is not None:
                self.writer.add_scalar(
                    "test/mask_coverage", mean_mask_coverage, self.global_step
                )
            if mean_ssim is not None:
                self.writer.add_scalar(
                    "test/ssim", mean_ssim, self.global_step
                )
            if mean_lpips is not None:
                self.writer.add_scalar(
                    "test/lpips", mean_lpips, self.global_step
                )
            if mean_cc_psnr is not None:
                self.writer.add_scalar(
                    "test/color_corrected_psnr", mean_cc_psnr, self.global_step
                )
            if mean_cc_ssim is not None:
                self.writer.add_scalar(
                    "test/color_corrected_ssim", mean_cc_ssim, self.global_step
                )
            if mean_cc_lpips is not None:
                self.writer.add_scalar(
                    "test/color_corrected_lpips",
                    mean_cc_lpips,
                    self.global_step,
                )
            self.writer.add_scalar(
                "time/test/inference", mean_inference_time, self.global_step
            )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "test/image/best_psnr",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "test/image/worst_psnr",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
