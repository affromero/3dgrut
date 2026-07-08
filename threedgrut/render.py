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

import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer
from threedgrut.post_processing import LuminanceAffine
from threedgrut.utils.render import post_processing_camera_index_mode
from threedgrut.utils.render import (
    apply_background,
    apply_feature_decoder,
    apply_post_processing,
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
        feature_decoder=None,
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics
        self.post_processing = post_processing
        if self.post_processing is not None:
            self.post_processing.camera_index_mode = post_processing_camera_index_mode(conf)
        self.feature_decoder = feature_decoder

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
        from threedgrut.datasets.utils import configure_dataloader_for_platform

        dataset = datasets.make_test(name=conf.dataset.type, config=conf)

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
        cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # Pre-2.0 checkpoints embed configs without the keys 2.0 requires
        # (feature_type, particle-feature knobs, ...). Merge the checkpoint
        # config over today's defaults so new keys get legacy-equivalent
        # defaults while every trained value wins where present.
        from omegaconf import OmegaConf

        _configs_dir = Path(__file__).resolve().parents[1] / "configs"
        if OmegaConf.select(conf, "model.feature_type") is None:
            defaults = OmegaConf.load(_configs_dir / "base_gs.yaml")
            method = OmegaConf.select(conf, "render.method") or "3dgut"
            # mirror the hydra group chain: 3dgut.yaml inherits 3dgrt.yaml
            render_conf = OmegaConf.load(_configs_dir / "render" / "3dgrt.yaml")
            if method != "3dgrt":
                render_conf = OmegaConf.merge(
                    render_conf,
                    OmegaConf.load(_configs_dir / "render" / f"{method}.yaml"),
                )
            render_conf.pop("defaults", None)
            render_defaults = OmegaConf.create(
                {"render": OmegaConf.to_container(render_conf)}
            )
            conf = OmegaConf.merge(defaults, render_defaults, conf)
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        model.build_acc()

        # Load post-processing if present in checkpoint
        post_processing = None
        method = conf.post_processing.method
        if "post_processing" in checkpoint and method == "linear-to-srgb":
            from threedgrut.utils.post_processing_linear_to_srgb import (
                LinearToSrgbPostProcessing,
            )

            post_processing = LinearToSrgbPostProcessing()
            post_processing.load_state_dict(checkpoint["post_processing"]["module"])
            post_processing = post_processing.to("cuda")
            logger.info("Linear-to-sRGB post-processing loaded from checkpoint")
        elif "post_processing" in checkpoint and method == "ppisp":
            from ppisp import PPISP, PPISPConfig

            # Derive config from training settings to match trainer.py
            use_controller = conf.post_processing.get("use_controller", True)
            n_distillation_steps = conf.post_processing.get("n_distillation_steps", 5000)
            if use_controller and n_distillation_steps > 0:
                main_training_steps = conf.n_iterations - n_distillation_steps
                controller_activation_ratio = main_training_steps / conf.n_iterations
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

            post_processing = PPISP.from_state_dict(checkpoint["post_processing"]["module"], config=ppisp_config)
            post_processing = post_processing.to("cuda")
            num_cameras = post_processing.crf_params.shape[0]
            num_frames = post_processing.exposure_params.shape[0]
            logger.info(f"📷 {method.upper()} loaded from checkpoint: {num_cameras} cameras, {num_frames} frames")

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

        # Load feature decoder for nht models
        feature_decoder = None
        if "feature_decoder" in checkpoint:
            from threedgrut.model.feature_decoder import FeatureDecoder
            from threedgrut.model.features import Features

            if model.feature_type == Features.Type.NHT:
                conf_model = conf.model
                dec = conf_model.nht_decoder
                hidden_dim = dec.hidden_dim
                num_layers = getattr(dec, "num_layers", 4)
                dir_encoding = getattr(dec, "dir_encoding", "SphericalHarmonics")
                dir_encoding_degree = getattr(dec, "dir_encoding_degree", 3)
                sh_scale = getattr(dec, "sh_scale", 1.0)
                output_activation = getattr(dec, "output_activation", "Sigmoid")
                unpremultiply_alpha = getattr(dec, "unpremultiply_alpha", False)
                ema_decay = getattr(dec, "ema_decay", 0.0)
                ema_start_step = getattr(dec, "ema_start_step", 0)
                feature_decoder = FeatureDecoder(
                    ray_feature_dim=model.ray_feature_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dir_encoding=dir_encoding,
                    dir_encoding_degree=dir_encoding_degree,
                    sh_scale=sh_scale,
                    output_activation=output_activation,
                    ema_decay=ema_decay,
                    ema_start_step=ema_start_step,
                    unpremultiply_alpha=unpremultiply_alpha,
                ).to("cuda")
                feature_decoder.load_state_dict(checkpoint["feature_decoder"]["module"])
                ema_state = checkpoint["feature_decoder"].get("ema")
                if ema_state is not None:
                    feature_decoder.load_ema_state_dict(ema_state)
                    feature_decoder.apply_ema_shadow()
                feature_decoder.eval()
                logger.info("🎨 Feature decoder loaded from checkpoint")

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
            feature_decoder=feature_decoder,
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
        feature_decoder=None,
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
            feature_decoder=feature_decoder,
        )

    @torch.no_grad()
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        cc_psnr = []
        cc_ssim = []
        cc_lpips = []
        inference_time = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)
            if self.feature_decoder is not None:
                outputs = apply_feature_decoder(
                    self.feature_decoder,
                    outputs,
                    gpu_batch,
                    training=False,
                    center_ray_encoding=bool(getattr(self.conf.model.nht_decoder, "center_ray_encoding", False)),
                )
            outputs = apply_background(self.model.background, outputs, gpu_batch, training=False)

            # Apply post-processing
            if self.post_processing is not None:
                outputs = apply_post_processing(self.post_processing, outputs, gpu_batch, training=False)

            pred_features_full = outputs["pred_features"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_features_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            pred_img_to_write = pred_features_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_features"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_features_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_features_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Color-corrected metrics
            pred_features_cc = color_correct_affine(pred_features_full, rgb_gt_full)
            cc_psnr.append(criterions["psnr"](pred_features_cc, rgb_gt_full).item())
            cc_ssim.append(
                criterions["ssim"](
                    pred_features_cc.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            cc_lpips.append(
                criterions["lpips"](
                    pred_features_cc.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        mean_cc_psnr = np.mean(cc_psnr)
        mean_cc_ssim = np.mean(cc_ssim)
        mean_cc_lpips = np.mean(cc_lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            mean_cc_psnr=mean_cc_psnr,
            mean_cc_ssim=mean_cc_ssim,
            mean_cc_lpips=mean_cc_lpips,
            std_psnr=std_psnr,
        )

        if self.conf.render.enable_kernel_timings:
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        # Save metrics to JSON file
        metrics_json = dict(
            mean_psnr=float(mean_psnr),
            mean_ssim=float(mean_ssim),
            mean_lpips=float(mean_lpips),
            mean_cc_psnr=float(mean_cc_psnr),
            mean_cc_ssim=float(mean_cc_ssim),
            mean_cc_lpips=float(mean_cc_lpips),
            mean_inference_time_ms=float(mean_inference_time),
        )
        metrics_path = os.path.join(self.out_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"📄 Metrics saved to: {metrics_path}")

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("cc_psnr/test", mean_cc_psnr, self.global_step)
            self.writer.add_scalar("cc_ssim/test", mean_cc_ssim, self.global_step)
            self.writer.add_scalar("cc_lpips/test", mean_cc_lpips, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time
