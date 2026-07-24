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

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
from omegaconf import DictConfig
from torchmetrics import PeakSignalNoiseRatio
from omegaconf import OmegaConf
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.datasets.protocols import Batch
from threedgrut.model.factory import create_gaussian_model
from threedgrut.model.local_projection_field import LocalProjectionField
from threedgrut.model.native_camera_extrinsics import (
    NativeCameraExtrinsics,
)
from threedgrut.post_processing import (
    LuminanceAffine,
    MultiscalePPISPConfig,
    PredictiveMultiscalePPISP,
    view_context_inference_contract,
)
from threedgrut.post_processing.checkpoint_contract import (
    module_state_sha256,
)
from threedgrut.utils.color_correct import color_correct_affine
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer
from threedgrut.utils.render import (
    apply_feature_decoder,
    apply_post_processing,
    post_processing_camera_idx,
    post_processing_camera_index_mode,
)

POST_PROCESSING_EVAL_MODE_RAW = "raw"
POST_PROCESSING_EVAL_MODE_INFERENCE = "inference"
POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA = "inference_sequence_metadata"
POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC = "ppisp_camera_only_diagnostic"
POST_PROCESSING_EVAL_MODES = (
    POST_PROCESSING_EVAL_MODE_RAW,
    POST_PROCESSING_EVAL_MODE_INFERENCE,
    POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
    POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC,
)
POST_PROCESSING_SOURCE_NONE = "none"
POST_PROCESSING_SOURCE_RUNTIME = "runtime"
POST_PROCESSING_SOURCE_CHECKPOINT = "checkpoint"
_CHECKPOINT_CAMERA_KEYS_ATTR = "_hax_checkpoint_camera_keys"
_CHECKPOINT_CAMERA_FRAME_COUNTS_ATTR = "_hax_checkpoint_camera_frame_counts"
_CHECKPOINT_CAMERA_INDEX_MODE_ATTR = "_hax_checkpoint_camera_index_mode"
_RESTORATION_MANIFEST_ATTR = "_hax_restoration_manifest"
_RUNTIME_POLICY_ATTR = "_hax_runtime_policy"


def upgrade_legacy_checkpoint_config(conf):
    """Return a checkpoint config merged over today's defaults.

    Checkpoints trained before newer config keys existed (feature_type,
    particle-feature knobs, NHT controls, ...) must stay loadable
    everywhere a checkpoint config is consumed. When the marker key
    `model.feature_type` is absent, merge the checkpoint config over the
    current base + render-group defaults so missing keys resolve to
    legacy-equivalent values while every trained value wins.
    """
    if OmegaConf.select(conf, "model.feature_type") is not None:
        return conf
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    defaults = OmegaConf.load(configs_dir / "base_gs.yaml")
    method = OmegaConf.select(conf, "render.method") or "3dgut"
    # mirror the hydra group chain: 3dgut.yaml inherits 3dgrt.yaml
    render_conf = OmegaConf.load(configs_dir / "render" / "3dgrt.yaml")
    if method != "3dgrt":
        render_conf = OmegaConf.merge(
            render_conf,
            OmegaConf.load(configs_dir / "render" / f"{method}.yaml"),
        )
    render_conf.pop("defaults", None)
    render_defaults = OmegaConf.create(
        {"render": OmegaConf.to_container(render_conf)}
    )
    return OmegaConf.merge(defaults, render_defaults, conf)


def _validated_camera_keys(
    value,
    *,
    label: str,
) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an ordered list of camera keys.")
    keys = [str(key) for key in value]
    if not keys or any(not key for key in keys):
        raise ValueError(f"{label} must contain non-empty camera keys.")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} contains duplicate camera keys: {keys}.")
    return keys


def _validated_camera_frame_counts(
    value,
    *,
    label: str,
    allow_zero: bool = False,
) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an ordered list of frame counts.")
    counts = [int(count) for count in value]
    invalid = any(count < 0 for count in counts) if allow_zero else any(count <= 0 for count in counts)
    if not counts or invalid:
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must contain {requirement} frame counts: {counts}.")
    return counts


def _checkpoint_scheduler_last_epoch(checkpoint: dict) -> int | None:
    post_processing_state = checkpoint["post_processing"]
    schedulers = post_processing_state.get("schedulers")
    if not isinstance(schedulers, (list, tuple)) or not schedulers:
        return None
    scheduler_state = schedulers[0]
    if not isinstance(scheduler_state, dict):
        return None
    last_epoch = scheduler_state.get("last_epoch")
    if last_epoch is None:
        return None
    return int(last_epoch)


def _ppisp_controller_restore_manifest(
    checkpoint: dict,
    *,
    use_controller: bool,
    configured_activation_step: int,
    require_controller_ready: bool,
    use_multiscale_controller: bool = False,
    use_view_context: bool = False,
) -> dict:
    post_processing_state = checkpoint["post_processing"]
    explicit = post_processing_state.get("inference_contract")
    explicit = explicit if isinstance(explicit, dict) else None
    checkpoint_global_step = int(checkpoint["global_step"])
    scheduler_last_epoch = _checkpoint_scheduler_last_epoch(checkpoint)
    activation_step = configured_activation_step
    proof_source = "legacy_scheduler_state"
    explicit_controller_trained = None
    explicit_multiscale_trained = None
    explicit_view_context_enabled = None
    explicit_view_context_trained = None
    explicit_view_context_contract = None
    inherited_controller_proves_training = False

    if explicit is not None:
        proof_source = "checkpoint_inference_contract"
        contract_global_step = int(explicit.get("checkpoint_global_step", checkpoint_global_step))
        if contract_global_step != checkpoint_global_step:
            raise ValueError("PPISP inference-contract global step does not match the " "checkpoint global step.")
        activation_step = int(explicit.get("controller_activation_step", activation_step))
        contract_scheduler_epoch = explicit.get("scheduler_last_epoch")
        if contract_scheduler_epoch is not None:
            contract_scheduler_epoch = int(contract_scheduler_epoch)
            if scheduler_last_epoch is None:
                raise ValueError(
                    "PPISP inference contract records scheduler state, but "
                    "the checkpoint does not contain that scheduler."
                )
            if contract_scheduler_epoch != scheduler_last_epoch:
                raise ValueError(
                    "PPISP inference-contract scheduler epoch does not match " "the serialized scheduler state."
                )
        explicit_controller_trained = explicit.get("controller_trained")
        explicit_multiscale_trained = explicit.get(
            "multiscale_controller_trained"
        )
        explicit_view_context_enabled = explicit.get(
            "multiscale_view_context_enabled"
        )
        explicit_view_context_trained = explicit.get(
            "multiscale_view_context_trained"
        )
        explicit_view_context_contract = explicit.get(
            "multiscale_view_context_contract"
        )
        inherited = explicit.get("frozen_parent_controller")
        if inherited is not None:
            if not isinstance(inherited, dict):
                raise ValueError(
                    "Frozen parent controller proof must be a mapping."
                )
            if inherited.get("schema_version") != 1:
                raise ValueError(
                    "Unsupported frozen parent controller proof version."
                )
            config = checkpoint.get("config")
            post_processing_config = getattr(
                config,
                "post_processing",
                None,
            )
            frozen_inference = bool(
                post_processing_config is not None
                and post_processing_config.get(
                    "frozen_inference_during_training",
                    False,
                )
            )
            if not frozen_inference:
                raise ValueError(
                    "Frozen parent controller proof requires a checkpoint "
                    "configured for frozen inference during training."
                )
            if scheduler_last_epoch is not None:
                raise ValueError(
                    "Frozen parent controller checkpoint must not contain a "
                    "local post-processing scheduler."
                )
            module_state = post_processing_state.get("module")
            if not isinstance(module_state, dict):
                raise ValueError(
                    "Frozen parent controller checkpoint has no module state."
                )
            expected_module_sha256 = inherited.get(
                "module_state_sha256"
            )
            actual_module_sha256 = module_state_sha256(module_state)
            if expected_module_sha256 != actual_module_sha256:
                raise ValueError(
                    "Frozen parent controller module-state hash mismatch."
                )
            parent_global_step = int(
                inherited.get("parent_checkpoint_global_step", -1)
            )
            parent_activation_step = int(
                inherited.get("parent_controller_activation_step", -1)
            )
            parent_scheduler_epoch = int(
                inherited.get("parent_scheduler_last_epoch", -1)
            )
            inherited_controller_proves_training = bool(
                inherited.get("parent_controller_trained") is True
                and parent_activation_step >= 0
                and parent_scheduler_epoch > parent_activation_step
                and parent_global_step >= parent_activation_step
            )
            if not inherited_controller_proves_training:
                raise ValueError(
                    "Frozen parent controller metadata does not prove a "
                    "controller update."
                )
            proof_source = "checkpoint_frozen_parent_contract"

    if use_view_context and (
        explicit_view_context_enabled is not True
        or explicit_view_context_trained is not True
        or explicit_view_context_contract
        != view_context_inference_contract()
    ):
        raise ValueError(
            "View-conditioned PPISP requires an exact versioned inference "
            "contract proving its context trained with serialized train-fold "
            "normalization."
        )

    scheduler_proves_training = (
        scheduler_last_epoch is not None
        and activation_step >= 0
        and scheduler_last_epoch > activation_step
        and checkpoint_global_step >= activation_step
    )
    controller_trained = bool(
        use_controller
        and (
            scheduler_proves_training
            or inherited_controller_proves_training
        )
        and explicit_controller_trained is not False
    )
    controller_ready = not use_controller or controller_trained
    multiscale_trained = bool(
        use_multiscale_controller
        and controller_trained
        and explicit_multiscale_trained is True
    )
    multiscale_ready = (
        not use_multiscale_controller or multiscale_trained
    )
    view_context_trained = bool(
        use_view_context
        and multiscale_trained
        and explicit_view_context_trained is True
    )
    view_context_ready = not use_view_context or view_context_trained
    if require_controller_ready and (
        not controller_ready
        or not multiscale_ready
        or not view_context_ready
    ):
        raise ValueError(
            "PPISP controller-backed common evaluation requires proof that "
            "the controller was trained. The checkpoint scheduler/activation "
            "metadata does not prove a controller update; use raw or "
            "ppisp_camera_only_diagnostic evaluation instead."
        )

    return {
        "enabled": bool(use_controller),
        "trained": controller_trained,
        "ready_for_controller_inference": controller_ready,
        "activation_step": int(activation_step),
        "scheduler_last_epoch": scheduler_last_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "proof_source": proof_source,
        "multiscale_enabled": bool(use_multiscale_controller),
        "multiscale_trained": multiscale_trained,
        "multiscale_ready_for_inference": multiscale_ready,
        "view_context_enabled": bool(use_view_context),
        "view_context_trained": view_context_trained,
        "view_context_ready_for_inference": view_context_ready,
        "view_context_contract": (
            explicit_view_context_contract if use_view_context else None
        ),
    }


def load_checkpoint_post_processing(
    checkpoint: dict,
    device: str = "cuda",
    *,
    require_controller_ready: bool = False,
) -> torch.nn.Module | None:
    """Restore checkpoint post-processing without changing learned state.

    ``require_controller_ready`` makes PPISP restoration fail closed unless
    scheduler and activation metadata prove that its controller trained.
    """
    if "post_processing" not in checkpoint:
        return None

    post_processing_state = checkpoint["post_processing"]
    checkpoint_camera_keys = _validated_camera_keys(
        post_processing_state.get("camera_keys"),
        label="Checkpoint post-processing camera_keys",
    )
    checkpoint_camera_frame_counts = _validated_camera_frame_counts(
        post_processing_state.get("camera_frame_counts"),
        label="Checkpoint post-processing camera_frame_counts",
    )
    if (checkpoint_camera_keys is None) != (checkpoint_camera_frame_counts is None):
        raise ValueError(
            "Checkpoint post-processing camera keys and frame counts must " "either both be present or both be absent."
        )
    if (
        checkpoint_camera_keys is not None
        and checkpoint_camera_frame_counts is not None
        and len(checkpoint_camera_keys) != len(checkpoint_camera_frame_counts)
    ):
        raise ValueError("Checkpoint post-processing camera keys/counts have different " "lengths.")
    checkpoint_camera_index_mode = post_processing_state.get("camera_index_mode")
    if checkpoint_camera_keys is not None and checkpoint_camera_index_mode is None:
        raise ValueError("Checkpoint has durable camera keys/counts but no camera index " "mode.")

    conf = upgrade_legacy_checkpoint_config(checkpoint["config"])
    method = conf.post_processing.method
    controller_manifest = None
    if method == "ppisp":
        from ppisp import PPISP, PPISPConfig

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

        configured_activation_step = int(controller_activation_ratio * int(conf.n_iterations))
        use_multiscale_controller = bool(
            conf.post_processing.get("use_multiscale_controller", False)
        )
        use_view_context = bool(
            conf.post_processing.get(
                "multiscale_use_view_context",
                False,
            )
        )
        if use_view_context and not use_multiscale_controller:
            raise ValueError(
                "Checkpoint enables multiscale view context without the "
                "multiscale controller."
            )
        controller_manifest = _ppisp_controller_restore_manifest(
            checkpoint,
            use_controller=bool(use_controller),
            configured_activation_step=configured_activation_step,
            require_controller_ready=require_controller_ready,
            use_multiscale_controller=use_multiscale_controller,
            use_view_context=use_view_context,
        )

        ppisp_config = PPISPConfig(
            use_controller=use_controller,
            controller_distillation=controller_distillation,
            controller_activation_ratio=controller_activation_ratio,
        )
        try:
            if use_multiscale_controller:
                post_processing = PredictiveMultiscalePPISP.from_state_dict(
                    post_processing_state["module"],
                    config=ppisp_config,
                    multiscale_config=MultiscalePPISPConfig(
                        coarse_grid_size=conf.post_processing.get(
                            "multiscale_coarse_grid_size",
                            4,
                        ),
                        fine_grid_size=conf.post_processing.get(
                            "multiscale_fine_grid_size",
                            16,
                        ),
                        coarse_max_log_gain=conf.post_processing.get(
                            "multiscale_coarse_max_log_gain",
                            0.04,
                        ),
                        coarse_max_bias=conf.post_processing.get(
                            "multiscale_coarse_max_bias",
                            0.02,
                        ),
                        fine_max_log_gain=conf.post_processing.get(
                            "multiscale_fine_max_log_gain",
                            0.02,
                        ),
                        fine_max_bias=conf.post_processing.get(
                            "multiscale_fine_max_bias",
                            0.01,
                        ),
                        magnitude_regularization=(
                            conf.post_processing.get(
                                "multiscale_magnitude_regularization",
                                0.01,
                            )
                        ),
                        total_variation_regularization=(
                            conf.post_processing.get(
                                "multiscale_total_variation_regularization",
                                0.01,
                            )
                        ),
                        init_seed=conf.post_processing.get(
                            "multiscale_init_seed",
                            20_260_717,
                        ),
                        use_view_context=use_view_context,
                    ),
                ).to(device)
            else:
                post_processing = PPISP.from_state_dict(
                    post_processing_state["module"],
                    config=ppisp_config,
                ).to(device)
        except RuntimeError as exc:
            raise ValueError(
                "Exact PPISP checkpoint restoration failed; evaluation will " "not drop or transform learned state."
            ) from exc
        num_cameras = post_processing.crf_params.shape[0]
        num_frames = post_processing.exposure_params.shape[0]
    elif method == "luminance_affine":
        state = post_processing_state["module"]
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
        ).to(device)
        try:
            post_processing.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "Exact LuminanceAffine checkpoint restoration failed; "
                "evaluation will not interpolate, drop, or synthesize state."
            ) from exc
    else:
        raise ValueError("Checkpoint contains post-processing state for unsupported " f"method {method!r}.")

    if checkpoint_camera_keys is not None:
        if len(checkpoint_camera_keys) != int(num_cameras):
            raise ValueError(
                "Checkpoint post-processing camera key count does not match "
                f"the restored module: {len(checkpoint_camera_keys)} keys for "
                f"{int(num_cameras)} camera slots."
            )
        if checkpoint_camera_frame_counts is None:
            raise ValueError("Checkpoint has durable camera keys but no positive per-camera " "frame counts.")

    restoration_manifest = {
        "method": str(method),
        "restore_policy": "strict_exact",
        "exact": True,
        "state_key_count": len(post_processing_state["module"]),
        "transformed_keys": [],
        "dropped_keys": [],
        "controller": controller_manifest,
    }
    setattr(
        post_processing,
        _CHECKPOINT_CAMERA_KEYS_ATTR,
        checkpoint_camera_keys,
    )
    setattr(
        post_processing,
        _CHECKPOINT_CAMERA_FRAME_COUNTS_ATTR,
        checkpoint_camera_frame_counts,
    )
    setattr(
        post_processing,
        _CHECKPOINT_CAMERA_INDEX_MODE_ATTR,
        (str(checkpoint_camera_index_mode) if checkpoint_camera_index_mode is not None else None),
    )
    setattr(post_processing, _RESTORATION_MANIFEST_ATTR, restoration_manifest)
    setattr(post_processing, _RUNTIME_POLICY_ATTR, "checkpoint_default")
    post_processing.eval()
    logger.info(f"📷 {method.upper()} loaded from checkpoint: " f"{num_cameras} cameras, {num_frames} frames")
    return post_processing


def configure_ppisp_camera_only_diagnostic(
    post_processing: torch.nn.Module,
) -> torch.nn.Module:
    """Disable PPISP exposure/color control while preserving camera effects."""
    restoration_manifest = getattr(
        post_processing,
        _RESTORATION_MANIFEST_ATTR,
        None,
    )
    if (
        not isinstance(restoration_manifest, dict)
        or restoration_manifest.get("method") != "ppisp"
        or not hasattr(post_processing, "config")
    ):
        raise ValueError("ppisp_camera_only_diagnostic requires an exactly restored PPISP " "checkpoint module.")
    post_processing.config.use_controller = False
    setattr(
        post_processing,
        _RUNTIME_POLICY_ATTR,
        "ppisp_camera_only_identity_exposure_color",
    )
    return post_processing


def configure_luminance_affine_sequence_metadata(
    post_processing: torch.nn.Module,
) -> LuminanceAffine:
    """Validate the restored temporal-affine sequence conditioning contract."""
    if not isinstance(post_processing, LuminanceAffine):
        raise ValueError(
            "inference_sequence_metadata requires a restored "
            "LuminanceAffine module; PPISP and other post-processors are "
            "not valid for this mode."
        )
    if post_processing.use_temporal_affine is not True:
        raise ValueError("inference_sequence_metadata requires LuminanceAffine with " "use_temporal_affine=true.")

    restoration_manifest = getattr(
        post_processing,
        _RESTORATION_MANIFEST_ATTR,
        None,
    )
    if (
        not isinstance(restoration_manifest, dict)
        or restoration_manifest.get("method") != "luminance_affine"
        or restoration_manifest.get("restore_policy") != "strict_exact"
        or restoration_manifest.get("exact") is not True
        or restoration_manifest.get("transformed_keys", [])
        or restoration_manifest.get("dropped_keys", [])
    ):
        raise ValueError(
            "inference_sequence_metadata requires exact LuminanceAffine " "checkpoint restoration metadata."
        )

    checkpoint_camera_keys = _validated_camera_keys(
        getattr(post_processing, _CHECKPOINT_CAMERA_KEYS_ATTR, None),
        label="Sequence-metadata checkpoint camera keys",
    )
    checkpoint_camera_frame_counts = _validated_camera_frame_counts(
        getattr(
            post_processing,
            _CHECKPOINT_CAMERA_FRAME_COUNTS_ATTR,
            None,
        ),
        label="Sequence-metadata checkpoint camera frame counts",
    )
    checkpoint_camera_index_mode = getattr(
        post_processing,
        _CHECKPOINT_CAMERA_INDEX_MODE_ATTR,
        None,
    )
    if checkpoint_camera_keys is None or checkpoint_camera_frame_counts is None or not checkpoint_camera_index_mode:
        raise ValueError(
            "inference_sequence_metadata requires durable checkpoint camera "
            "keys, positive frame counts, and camera index mode; legacy "
            "numeric camera state is ambiguous."
        )
    if len(checkpoint_camera_keys) != len(checkpoint_camera_frame_counts):
        raise ValueError(
            "inference_sequence_metadata checkpoint camera keys and frame " "counts must have identical lengths."
        )
    if len(checkpoint_camera_keys) != _post_processing_num_cameras(post_processing):
        raise ValueError(
            "inference_sequence_metadata checkpoint camera keys do not "
            "exactly cover the restored LuminanceAffine camera slots."
        )

    setattr(
        post_processing,
        _RUNTIME_POLICY_ATTR,
        "luminance_affine_non_rgb_sequence_metadata",
    )
    return post_processing


def _resolve_post_processing_eval_mode(
    post_processing: torch.nn.Module | None,
    mode: str | None,
) -> str:
    if mode is None:
        if post_processing is None:
            return POST_PROCESSING_EVAL_MODE_RAW
        return POST_PROCESSING_EVAL_MODE_INFERENCE
    if mode not in POST_PROCESSING_EVAL_MODES:
        raise ValueError(
            f"Unsupported post-processing eval mode {mode!r}. Expected one " f"of {POST_PROCESSING_EVAL_MODES}."
        )
    if mode == POST_PROCESSING_EVAL_MODE_RAW and post_processing is not None:
        raise ValueError("Raw eval mode cannot receive a post-processing module.")
    if (
        mode
        in (
            POST_PROCESSING_EVAL_MODE_INFERENCE,
            POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA,
            POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC,
        )
        and post_processing is None
    ):
        raise ValueError(f"{mode} eval mode requires a restored post-processing module.")
    return mode


def _resolve_post_processing_source(
    post_processing: torch.nn.Module | None,
    source: str | None,
) -> str:
    if post_processing is None:
        if source not in (None, POST_PROCESSING_SOURCE_NONE):
            raise ValueError("A post-processing source cannot be set without a module.")
        return POST_PROCESSING_SOURCE_NONE
    if source is None:
        return POST_PROCESSING_SOURCE_RUNTIME
    if source not in (
        POST_PROCESSING_SOURCE_RUNTIME,
        POST_PROCESSING_SOURCE_CHECKPOINT,
    ):
        raise ValueError(f"Unsupported post-processing source {source!r}.")
    return source


def _restore_local_projection_field(
    checkpoint: dict,
    *,
    renderer: object,
    device: torch.device,
) -> tuple[LocalProjectionField, str] | None:
    """Restore and bind a saved source-frame projection field for rendering."""
    field_checkpoint = checkpoint.get("local_projection_field")
    if field_checkpoint is None:
        return None
    if not isinstance(field_checkpoint, dict):
        raise ValueError("Local projection field checkpoint state is invalid.")
    if field_checkpoint.get("format_version") != (
        LocalProjectionField.checkpoint_format_version
    ):
        raise ValueError(
            "Local projection field checkpoint version is invalid."
        )
    if field_checkpoint.get("algorithm") != (
        LocalProjectionField.checkpoint_algorithm
    ):
        raise ValueError(
            "Local projection field checkpoint algorithm is invalid."
        )
    source_frame_manifest_hash = field_checkpoint.get(
        "source_frame_manifest_hash"
    )
    if not isinstance(source_frame_manifest_hash, str) or not (
        source_frame_manifest_hash
    ):
        raise ValueError(
            "Local projection field checkpoint manifest is invalid."
        )
    module_state = field_checkpoint.get("module")
    if not isinstance(module_state, dict):
        raise ValueError("Local projection field module state is invalid.")
    values = module_state.get("values")
    if (
        not torch.is_tensor(values)
        or values.ndim != 4
        or values.shape[-1] != 2
        or values.dtype is not torch.float32
    ):
        raise ValueError("Local projection field values are invalid.")
    field = LocalProjectionField(
        num_source_frames=int(values.shape[0]),
        grid_height=int(values.shape[1]),
        grid_width=int(values.shape[2]),
    )
    try:
        field.load_state_dict(module_state)
    except RuntimeError as exc:
        raise ValueError(
            "Local projection field module state is invalid."
        ) from exc
    field.validate_state()
    setter = getattr(renderer, "set_local_projection_field", None)
    if not callable(setter):
        raise RuntimeError(
            "Checkpoint uses a local projection field, but the renderer does "
            "not support it."
        )
    field = field.to(device)
    setter(field)
    return field, source_frame_manifest_hash


def _validate_local_projection_field_dataset(
    *,
    field: LocalProjectionField,
    source_frame_manifest_hash: str,
    dataset: object,
) -> None:
    """Reject render datasets that cannot preserve field source-frame mapping."""
    source_frame_count = getattr(dataset, "get_source_frame_count", None)
    source_manifest = getattr(dataset, "get_source_frame_manifest_hash", None)
    if not callable(source_frame_count) or not callable(source_manifest):
        raise ValueError(
            "Local projection field rendering requires stable source-frame "
            "metadata."
        )
    if int(source_frame_count()) != field.num_source_frames:
        raise ValueError("Local projection field source-frame count mismatch.")
    if source_manifest() != source_frame_manifest_hash:
        raise ValueError(
            "Local projection field source-frame manifest mismatch."
        )


def _native_absolute_camera_rendering_enabled(conf: DictConfig) -> bool:
    """Return whether checkpoint rendering owns native absolute cameras."""
    camera_conf = conf.get("camera_residual")
    if camera_conf is None:
        return False
    return bool(camera_conf.get("enabled", False)) and bool(
        camera_conf.get("native_absolute_colmap", False)
    )


def _restore_native_camera_extrinsics(
    checkpoint: dict[str, object],
    *,
    conf: DictConfig,
    dataset: object,
    device: torch.device,
) -> NativeCameraExtrinsics | None:
    """Restore native absolute camera poses for checkpoint rendering."""
    if not _native_absolute_camera_rendering_enabled(conf):
        return None
    camera_checkpoint = checkpoint.get("camera_residual")
    if camera_checkpoint is None:
        msg = (
            "Native absolute camera rendering requires camera_residual "
            "checkpoint state."
        )
        raise ValueError(msg)
    if not isinstance(camera_checkpoint, dict):
        msg = "Native camera checkpoint state must be a dictionary."
        raise TypeError(msg)
    if camera_checkpoint.get("format_version") != (
        NativeCameraExtrinsics.checkpoint_format_version
    ):
        msg = "Native camera checkpoint format version is invalid."
        raise ValueError(msg)
    if camera_checkpoint.get("algorithm") != (
        NativeCameraExtrinsics.checkpoint_algorithm
    ):
        msg = "Native camera checkpoint algorithm is invalid."
        raise ValueError(msg)
    source_frame_manifest_hash = camera_checkpoint.get(
        "source_frame_manifest_hash"
    )
    if not isinstance(source_frame_manifest_hash, str):
        msg = "Native camera checkpoint manifest must be a string."
        raise TypeError(msg)
    if not source_frame_manifest_hash:
        msg = "Native camera checkpoint manifest is invalid."
        raise ValueError(msg)
    source_extrinsics = getattr(
        dataset,
        "get_source_frame_colmap_extrinsics",
        None,
    )
    source_manifest = getattr(
        dataset,
        "get_source_frame_manifest_hash",
        None,
    )
    if not callable(source_extrinsics) or not callable(source_manifest):
        msg = "Native camera rendering requires stable source-frame metadata."
        raise TypeError(msg)
    dataset_manifest_hash = source_manifest()
    if dataset_manifest_hash != source_frame_manifest_hash:
        msg = "Native camera checkpoint manifest mismatch."
        raise ValueError(msg)
    initial_qvecs, initial_tvecs = source_extrinsics()
    camera_residual = NativeCameraExtrinsics(
        initial_qvecs=torch.as_tensor(initial_qvecs, dtype=torch.float32),
        initial_tvecs=torch.as_tensor(initial_tvecs, dtype=torch.float32),
    )
    if camera_checkpoint.get("optimizer_group_manifest") != (
        camera_residual.optimizer_group_manifest()
    ):
        msg = "Native camera checkpoint optimizer-group manifest is invalid."
        raise ValueError(msg)
    module_state = camera_checkpoint.get("module")
    if module_state is None:
        msg = "Native camera checkpoint module state is missing."
        raise ValueError(msg)
    if not isinstance(module_state, dict):
        msg = "Native camera checkpoint module state must be a dictionary."
        raise TypeError(msg)
    saved_initial_qvecs = module_state.get("initial_qvecs")
    saved_initial_tvecs = module_state.get("initial_tvecs")
    if not torch.is_tensor(saved_initial_qvecs) or not torch.is_tensor(
        saved_initial_tvecs
    ):
        msg = "Native camera checkpoint initial poses must be tensors."
        raise TypeError(msg)
    expected_initial_qvecs = camera_residual.initial_qvecs
    expected_initial_tvecs = camera_residual.initial_tvecs
    if (
        saved_initial_qvecs.shape != expected_initial_qvecs.shape
        or saved_initial_tvecs.shape != expected_initial_tvecs.shape
    ):
        msg = (
            "Native camera checkpoint initial-pose shape does not match the "
            "render dataset."
        )
        raise ValueError(msg)
    if not torch.allclose(
        saved_initial_qvecs.detach().to(
            device=expected_initial_qvecs.device,
            dtype=expected_initial_qvecs.dtype,
        ),
        expected_initial_qvecs,
        rtol=0.0,
        atol=1.0e-6,
    ) or not torch.allclose(
        saved_initial_tvecs.detach().to(
            device=expected_initial_tvecs.device,
            dtype=expected_initial_tvecs.dtype,
        ),
        expected_initial_tvecs,
        rtol=0.0,
        atol=1.0e-6,
    ):
        msg = (
            "Native camera checkpoint initial poses do not match the render "
            "dataset."
        )
        raise ValueError(msg)
    try:
        camera_residual.load_state_dict(module_state, strict=True)
    except RuntimeError as exc:
        msg = (
            "Native camera checkpoint state does not match the render dataset."
        )
        raise ValueError(msg) from exc
    camera_residual.validate_state()
    camera_residual = camera_residual.to(device)
    camera_residual.eval()
    return camera_residual


def _apply_native_camera_extrinsics(
    camera_residual: NativeCameraExtrinsics,
    gpu_batch: Batch,
    *,
    global_step: int,
) -> Batch:
    """Apply restored native poses without accepting unindexed batches."""
    if getattr(gpu_batch, "source_frame_idx", None) is None:
        msg = (
            "Native camera rendering requires source_frame_idx on every batch."
        )
        raise ValueError(msg)
    return camera_residual(gpu_batch, global_step=global_step)


def _use_known_frame_post_processing(
    *,
    conf: DictConfig,
    post_processing: torch.nn.Module | None,
    camera_residual: NativeCameraExtrinsics | None,
) -> bool:
    """Match trainer reconstruction scoring for eligible native checkpoints."""
    configured = bool(
        conf.post_processing.get("apply_known_frame_in_eval", False)
    )
    if camera_residual is None or post_processing is None:
        return configured
    if not bool(getattr(post_processing, "use_native_appearance_grid", False)):
        return configured
    if int(conf.dataset.test_split_interval) > 0:
        return configured
    holdout_path = conf.dataset.get("holdout_image_list_path")
    if bool(str(holdout_path or "").strip()):
        return configured
    return True


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


def _sha256_of_tree(path: str) -> str | None:
    """Hash relative file names and bytes under a directory."""
    if not path or not os.path.isdir(path):
        return None
    hasher = hashlib.sha256()
    file_count = 0
    for root, directories, files in os.walk(path):
        directories.sort()
        for file_name in sorted(files):
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, path).replace(
                os.sep,
                "/",
            )
            hasher.update(relative_path.encode("utf-8"))
            hasher.update(b"\0")
            with open(file_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    hasher.update(chunk)
            hasher.update(b"\0")
            file_count += 1
    if file_count == 0:
        return None
    return hasher.hexdigest()


def _sha256_of_json(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _post_processing_num_cameras(module: torch.nn.Module) -> int:
    for attribute_name in ("crf_params", "camera_log_gain"):
        value = getattr(module, attribute_name, None)
        if value is not None and hasattr(value, "shape") and value.shape:
            return int(value.shape[0])
    raise ValueError("Restored post-processing module does not expose a camera-slot table.")


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
        post_processing_mode=None,
        post_processing_source=None,
        strict_post_processing_contract=False,
        checkpoint_path=None,
        checkpoint_sha256=None,
        original_training_bundle=None,
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
        if self.post_processing is not None:
            self.post_processing.camera_index_mode = post_processing_camera_index_mode(conf)
        self.feature_decoder = feature_decoder
        self.post_processing_mode = _resolve_post_processing_eval_mode(
            post_processing,
            post_processing_mode,
        )
        self.post_processing_source = _resolve_post_processing_source(
            post_processing,
            post_processing_source,
        )
        if self.post_processing_mode == (POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA):
            if self.post_processing_source != (POST_PROCESSING_SOURCE_CHECKPOINT):
                raise ValueError(
                    "inference_sequence_metadata requires a " "checkpoint-restored post-processing module."
                )
            configure_luminance_affine_sequence_metadata(self.post_processing)
        self.strict_post_processing_contract = bool(strict_post_processing_contract)
        self.checkpoint_path = os.path.abspath(checkpoint_path) if checkpoint_path else None
        self.checkpoint_sha256 = (
            str(checkpoint_sha256) if checkpoint_sha256 is not None else _sha256_of_file(self.checkpoint_path)
        )
        self.original_training_bundle = str(original_training_bundle) if original_training_bundle is not None else None
        self._post_processing_camera_index_mode = post_processing_camera_index_mode(conf)
        self._configure_post_processing_camera_contract()
        if self.post_processing_mode == (
            POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA
        ) and self._post_processing_camera_contract_status != ("validated_by_key"):
            raise ValueError(
                "inference_sequence_metadata requires an exact durable " "checkpoint-to-eval camera mapping."
            )
        self.eval_sparse_path = os.path.join(
            str(self.conf.get("path", "")),
            "sparse",
            "0",
        )
        self.eval_sparse_sha256 = _sha256_of_tree(self.eval_sparse_path)
        self.holdout_image_list_path = getattr(
            self.dataset,
            "holdout_image_list_path",
            None,
        )
        self.holdout_image_list_sha256 = _sha256_of_file(self.holdout_image_list_path)
        self.train_exclude_image_list_path = (
            getattr(
                self.dataset,
                "train_exclude_image_list_path",
                None,
            )
            if self.split == "train"
            else None
        )
        self.train_exclude_image_list_sha256 = _sha256_of_file(
            self.train_exclude_image_list_path
        )
        self.camera_residual: NativeCameraExtrinsics | None = None
        self._use_known_frame_post_processing = bool(
            conf.post_processing.get("apply_known_frame_in_eval", False)
        )

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def _eval_post_processing_camera_contract(
        self,
    ) -> tuple[list[str], list[int]]:
        if self._post_processing_camera_index_mode == ("single_physical_camera"):
            frames_fn = getattr(
                self.dataset,
                "get_post_processing_frames_per_camera",
                None,
            )
            if frames_fn is None:
                frames_fn = self.dataset.get_frames_per_camera
            counts = [sum(int(count) for count in frames_fn())]
            return ["physical_camera"], counts

        names_fn = getattr(
            self.dataset,
            "get_post_processing_camera_names",
            None,
        )
        frames_fn = getattr(
            self.dataset,
            "get_post_processing_frames_per_camera",
            None,
        )
        if names_fn is None or frames_fn is None:
            camera_count = _post_processing_num_cameras(self.post_processing)
            if camera_count == 1:
                return ["legacy_single_camera"], [len(self.dataset)]
            if not self.strict_post_processing_contract:
                return (
                    [f"camera_index_{index}" for index in range(camera_count)],
                    [1] * camera_count,
                )
            raise ValueError(
                "Strict common evaluation requires the eval dataset to expose "
                "ordered physical camera keys and frame counts."
            )
        keys = _validated_camera_keys(
            names_fn(),
            label="Eval post-processing camera keys",
        )
        counts = _validated_camera_frame_counts(
            frames_fn(),
            label="Eval post-processing camera frame counts",
            allow_zero=True,
        )
        if keys is None or counts is None or len(keys) != len(counts):
            raise ValueError("Eval post-processing camera keys/counts are incomplete or " "have different lengths.")
        return keys, counts

    def _configure_post_processing_camera_contract(self) -> None:
        self._post_processing_checkpoint_camera_keys = None
        self._post_processing_checkpoint_camera_frame_counts = None
        self._post_processing_eval_camera_keys = []
        self._post_processing_eval_camera_frame_counts = []
        self._post_processing_camera_mapping = None
        self._post_processing_camera_contract_status = "not_applicable"
        if self.post_processing is None:
            return
        if all(
            getattr(self.post_processing, attribute_name, None) is None
            for attribute_name in ("crf_params", "camera_log_gain")
        ):
            # Camera-agnostic post-processing (e.g. linear-to-srgb) has no
            # camera-slot table; the per-camera contract does not apply.
            return

        module_camera_count = _post_processing_num_cameras(self.post_processing)
        checkpoint_keys = getattr(
            self.post_processing,
            _CHECKPOINT_CAMERA_KEYS_ATTR,
            None,
        )
        checkpoint_counts = getattr(
            self.post_processing,
            _CHECKPOINT_CAMERA_FRAME_COUNTS_ATTR,
            None,
        )
        checkpoint_index_mode = getattr(
            self.post_processing,
            _CHECKPOINT_CAMERA_INDEX_MODE_ATTR,
            None,
        )
        eval_keys, eval_counts = self._eval_post_processing_camera_contract()
        self._post_processing_eval_camera_keys = eval_keys
        self._post_processing_eval_camera_frame_counts = eval_counts

        if checkpoint_index_mode is not None and checkpoint_index_mode != self._post_processing_camera_index_mode:
            raise ValueError(
                "Checkpoint and evaluator use different post-processing "
                "camera index modes: "
                f"{checkpoint_index_mode!r} versus "
                f"{self._post_processing_camera_index_mode!r}."
            )

        if checkpoint_keys is None:
            if getattr(self, "post_processing_mode", None) == (POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA):
                raise ValueError(
                    "inference_sequence_metadata requires durable checkpoint "
                    "camera mapping; legacy camera state is ambiguous."
                )
            if self.strict_post_processing_contract and not (module_camera_count == 1 and len(eval_keys) == 1):
                raise ValueError(
                    "Multi-camera common evaluation requires durable ordered "
                    "camera keys and positive frame counts in the checkpoint."
                )
            if module_camera_count == 1 and len(eval_keys) == 1:
                self._post_processing_camera_mapping = [0]
                self._post_processing_camera_contract_status = "legacy_single_camera_proven"
                return
            self._post_processing_camera_contract_status = "legacy_numeric_unvalidated"
            return

        checkpoint_keys = _validated_camera_keys(
            checkpoint_keys,
            label="Restored checkpoint camera keys",
        )
        checkpoint_counts = _validated_camera_frame_counts(
            checkpoint_counts,
            label="Restored checkpoint camera frame counts",
        )
        if checkpoint_keys is None or checkpoint_counts is None:
            raise ValueError("Checkpoint camera metadata is incomplete.")
        if len(checkpoint_keys) != module_camera_count:
            raise ValueError("Checkpoint camera keys do not match restored module slots.")
        if len(checkpoint_keys) != len(checkpoint_counts):
            raise ValueError("Checkpoint camera keys and positive frame counts differ in " "length.")
        if set(checkpoint_keys) != set(eval_keys):
            raise ValueError(
                "Checkpoint and eval physical camera keys differ: " f"checkpoint={checkpoint_keys}, eval={eval_keys}."
            )

        checkpoint_index_by_key = {key: index for index, key in enumerate(checkpoint_keys)}
        self._post_processing_camera_mapping = [checkpoint_index_by_key[key] for key in eval_keys]
        self._post_processing_checkpoint_camera_keys = checkpoint_keys
        self._post_processing_checkpoint_camera_frame_counts = checkpoint_counts
        self._post_processing_camera_contract_status = "validated_by_key"

    def _post_processing_camera_mapping_manifest(self) -> list[dict]:
        mapping = self._post_processing_camera_mapping
        if mapping is None:
            return []
        checkpoint_keys = self._post_processing_checkpoint_camera_keys
        manifest = []
        for eval_index, checkpoint_index in enumerate(mapping):
            checkpoint_key = None
            if checkpoint_keys is not None:
                checkpoint_key = checkpoint_keys[checkpoint_index]
            manifest.append(
                {
                    "eval_index": eval_index,
                    "eval_key": self._post_processing_eval_camera_keys[eval_index],
                    "eval_frame_count": (self._post_processing_eval_camera_frame_counts[eval_index]),
                    "checkpoint_index": checkpoint_index,
                    "checkpoint_key": checkpoint_key,
                    "checkpoint_frame_count": (
                        self._post_processing_checkpoint_camera_frame_counts[checkpoint_index]
                        if self._post_processing_checkpoint_camera_frame_counts is not None
                        else None
                    ),
                }
            )
        return manifest

    def _eval_image_relative_name(
        self,
        iteration: int,
        image_path: str,
    ) -> str:
        extrinsics = getattr(self.dataset, "cam_extrinsics", None)
        if extrinsics is not None and iteration < len(extrinsics):
            relative_name = getattr(extrinsics[iteration], "name", None)
            if relative_name:
                return str(relative_name).replace("\\", "/")
        return os.path.basename(image_path)

    def _provenance_eval_camera_contract(
        self,
    ) -> tuple[list[str], list[int]]:
        if self._post_processing_eval_camera_keys:
            return (
                self._post_processing_eval_camera_keys,
                self._post_processing_eval_camera_frame_counts,
            )
        names_fn = getattr(
            self.dataset,
            "get_post_processing_camera_names",
            None,
        )
        counts_fn = getattr(
            self.dataset,
            "get_post_processing_frames_per_camera",
            None,
        )
        if names_fn is None or counts_fn is None:
            return [], []
        keys = _validated_camera_keys(
            names_fn(),
            label="Provenance eval camera keys",
        )
        counts = _validated_camera_frame_counts(
            counts_fn(),
            label="Provenance eval camera frame counts",
            allow_zero=True,
        )
        if keys is None or counts is None or len(keys) != len(counts):
            raise ValueError("Cannot write provenance for incomplete eval camera metadata.")
        return keys, counts

    def _common_eval_contract_provenance(
        self,
        per_frame_metrics: list[dict],
    ) -> dict:
        evaluated_frames = [
            frame["image_relative_name"] for frame in per_frame_metrics if "image_relative_name" in frame
        ]
        uses_sequence_metadata = self.post_processing_mode == (POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA)
        uses_view_context = bool(
            self.post_processing is not None
            and getattr(
                self.post_processing,
                "use_view_context",
                False,
            )
        )
        sequence_metadata = [
            {
                "image_relative_name": frame["image_relative_name"],
                "sequence_idx": frame.get("sequence_idx"),
            }
            for frame in per_frame_metrics
            if "image_relative_name" in frame
        ]
        if uses_sequence_metadata:
            if len(sequence_metadata) != len(evaluated_frames) or any(
                record["sequence_idx"] is None or int(record["sequence_idx"]) < 0 for record in sequence_metadata
            ):
                raise ValueError(
                    "inference_sequence_metadata provenance requires a "
                    "non-negative dataset sequence index for every evaluated "
                    "frame."
                )
        eval_camera_keys, eval_camera_frame_counts = self._provenance_eval_camera_contract()
        checkpoint_camera_keys = self._post_processing_checkpoint_camera_keys
        checkpoint_camera_frame_counts = self._post_processing_checkpoint_camera_frame_counts
        restoration_manifest = (
            getattr(
                self.post_processing,
                _RESTORATION_MANIFEST_ATTR,
                None,
            )
            if self.post_processing is not None
            else {
                "method": "none",
                "restore_policy": "none",
                "exact": True,
                "state_key_count": 0,
                "transformed_keys": [],
                "dropped_keys": [],
                "controller": None,
            }
        )
        runtime_policy = (
            getattr(
                self.post_processing,
                _RUNTIME_POLICY_ATTR,
                "runtime_module",
            )
            if self.post_processing is not None
            else "none"
        )
        if uses_sequence_metadata:
            render_signal_contract = "field_plus_temporal_luminance_affine_non_rgb_" "sequence_metadata"
        elif uses_view_context:
            render_signal_contract = (
                "field_plus_sealed_predictive_view_context"
            )
        elif self.post_processing_mode == (POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC):
            render_signal_contract = "field_plus_ppisp_camera_only_diagnostic_noncanonical"
        elif self.post_processing is None:
            render_signal_contract = "field_only_raw"
        else:
            render_signal_contract = "field_plus_sealed_novel_view_post_processing"
        eval_bundle_path = str(self.conf.get("path", ""))
        eval_sparse_path = getattr(
            self,
            "eval_sparse_path",
            os.path.join(eval_bundle_path, "sparse", "0"),
        )
        eval_sparse_sha256 = getattr(
            self,
            "eval_sparse_sha256",
            None,
        )
        if eval_sparse_sha256 is None:
            eval_sparse_sha256 = _sha256_of_tree(eval_sparse_path)
        holdout_path = getattr(
            self,
            "holdout_image_list_path",
            getattr(self.dataset, "holdout_image_list_path", None),
        )
        holdout_sha256 = getattr(
            self,
            "holdout_image_list_sha256",
            None,
        )
        if holdout_sha256 is None:
            holdout_sha256 = _sha256_of_file(holdout_path)
        checkpoint_sha256 = getattr(self, "checkpoint_sha256", None)
        if checkpoint_sha256 is None:
            checkpoint_sha256 = _sha256_of_file(self.checkpoint_path)
        if uses_sequence_metadata:
            inference_contract = "frame_idx_minus_one_sequence_idx_from_dataset_image_name_" "exposure_prior_none"
            conditioning_source = "non_rgb_sequence_metadata"
            conditioning_contract = "dataset_filename_sequence_idx_only_no_rgb_no_exif_no_" "per_frame_latent"
        elif uses_view_context:
            inference_contract = (
                "sealed_frame_minus_one_exif_none_plus_render_context"
            )
            conditioning_source = (
                "render_rgb_ray_distance_opacity_and_camera_pose"
            )
            conditioning_contract = view_context_inference_contract()
        elif self.post_processing is not None:
            inference_contract = "sealed_frame_minus_one_sequence_minus_one_exif_none"
            conditioning_source = "none"
            conditioning_contract = "none"
        else:
            inference_contract = "none"
            conditioning_source = "none"
            conditioning_contract = "none"

        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "original_training_bundle": self.original_training_bundle,
            "eval_bundle_path": eval_bundle_path,
            "eval_sparse_path": eval_sparse_path,
            "eval_sparse_sha256": eval_sparse_sha256,
            "eval_camera_keys": eval_camera_keys,
            "eval_camera_frame_counts": eval_camera_frame_counts,
            "eval_camera_keys_sha256": _sha256_of_json(eval_camera_keys),
            "checkpoint_camera_keys": checkpoint_camera_keys,
            "checkpoint_camera_frame_counts": (checkpoint_camera_frame_counts),
            "checkpoint_camera_keys_sha256": (
                _sha256_of_json(checkpoint_camera_keys) if checkpoint_camera_keys is not None else None
            ),
            "post_processing_camera_mapping": (self._post_processing_camera_mapping_manifest()),
            "post_processing_camera_contract_status": (self._post_processing_camera_contract_status),
            "post_processing_mode": self.post_processing_mode,
            "post_processing_source": self.post_processing_source,
            "render_signal_contract": render_signal_contract,
            "post_processing_method": (
                self.conf.post_processing.get("method", "none") if self.post_processing is not None else "none"
            ),
            "post_processing_inference_contract": inference_contract,
            "post_processing_conditioning_source": conditioning_source,
            "post_processing_conditioning_contract": conditioning_contract,
            "post_processing_uses_per_frame_latent": False,
            "post_processing_uses_sequence_idx": uses_sequence_metadata,
            "post_processing_uses_exif_exposure": False,
            "post_processing_sequence_metadata": (sequence_metadata if uses_sequence_metadata else []),
            "post_processing_sequence_metadata_sha256": (
                _sha256_of_json(sequence_metadata) if uses_sequence_metadata else None
            ),
            "post_processing_runtime_policy": runtime_policy,
            "post_processing_restoration_manifest": restoration_manifest,
            "post_processing_is_noncanonical_diagnostic": (
                self.post_processing_mode == POST_PROCESSING_EVAL_MODE_PPISP_CAMERA_ONLY_DIAGNOSTIC
            ),
            "dataset_load_exif": bool(self.conf.dataset.get("load_exif", True)),
            "holdout_image_list_path": holdout_path,
            "holdout_image_list_sha256": holdout_sha256,
            "train_exclude_image_list_path": (
                getattr(
                    self,
                    "train_exclude_image_list_path",
                    None,
                )
            ),
            "train_exclude_image_list_sha256": (
                getattr(
                    self,
                    "train_exclude_image_list_sha256",
                    None,
                )
            ),
            "evaluated_frame_count": len(evaluated_frames),
            "evaluated_frame_names": evaluated_frames,
            "evaluated_frame_names_sha256": _sha256_of_json(evaluated_frames),
        }

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""
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
            raise ValueError(f"Unsupported render split {self.split!r}. Expected train or val.")

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
        holdout_image_list_path=None,
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint_sha256 = _sha256_of_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        global_step = checkpoint["global_step"]

        conf = upgrade_legacy_checkpoint_config(checkpoint["config"])
        original_training_bundle = str(conf.path)
        if holdout_image_list_path is not None:
            conf.dataset.holdout_image_list_path = holdout_image_list_path
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
            model = create_gaussian_model(
                conf,
                checkpoint=checkpoint,
            )
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
        model.build_acc()
        restored_local_projection_field = _restore_local_projection_field(
            checkpoint,
            renderer=getattr(model, "renderer", None),
            device=torch.device("cuda"),
        )

        # Load post-processing if present in checkpoint. Linear-to-sRGB is a
        # global (non-per-camera) 2.0 transform that the checkpoint-contract
        # loader does not model, so it is restored inline; every other method
        # goes through the exact-restore path that also stamps the durable
        # camera-key metadata the eval camera contract relies on.
        method = conf.post_processing.method
        if "post_processing" in checkpoint and method == "linear-to-srgb":
            from threedgrut.utils.post_processing_linear_to_srgb import (
                LinearToSrgbPostProcessing,
            )

            post_processing = LinearToSrgbPostProcessing()
            post_processing.load_state_dict(checkpoint["post_processing"]["module"])
            post_processing = post_processing.to("cuda")
            logger.info("Linear-to-sRGB post-processing loaded from checkpoint")
        else:
            post_processing = load_checkpoint_post_processing(checkpoint)

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

        renderer = Renderer(
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
            post_processing_mode=(
                POST_PROCESSING_EVAL_MODE_INFERENCE if post_processing is not None else POST_PROCESSING_EVAL_MODE_RAW
            ),
            post_processing_source=(
                POST_PROCESSING_SOURCE_CHECKPOINT if post_processing is not None else POST_PROCESSING_SOURCE_NONE
            ),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            original_training_bundle=original_training_bundle,
            split=split,
        )
        if restored_local_projection_field is not None:
            field, source_frame_manifest_hash = restored_local_projection_field
            _validate_local_projection_field_dataset(
                field=field,
                source_frame_manifest_hash=source_frame_manifest_hash,
                dataset=renderer.dataset,
            )
        if _native_absolute_camera_rendering_enabled(conf):
            renderer.camera_residual = _restore_native_camera_extrinsics(
                checkpoint,
                conf=conf,
                dataset=renderer.dataset,
                device=torch.device("cuda"),
            )
        else:
            renderer.camera_residual = None
        use_known_frame = _use_known_frame_post_processing(
            conf=conf,
            post_processing=post_processing,
            camera_residual=renderer.camera_residual,
        )
        if use_known_frame != getattr(
            renderer,
            "_use_known_frame_post_processing",
            False,
        ):
            logger.info(
                "Standalone native reconstruction rendering uses known "
                "source-frame appearance to match trainer validation."
            )
        renderer._use_known_frame_post_processing = use_known_frame
        return renderer

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
        post_processing_mode=None,
        post_processing_source=None,
        strict_post_processing_contract=False,
        checkpoint_path=None,
        checkpoint_sha256=None,
        original_training_bundle=None,
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
            feature_decoder=feature_decoder,
            post_processing_mode=post_processing_mode,
            post_processing_source=post_processing_source,
            strict_post_processing_contract=strict_post_processing_contract,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            original_training_bundle=original_training_bundle,
            split=split or conf.render.get("split", "val"),
        )

    def _sequence_idx_from_eval_batch(self, gpu_batch: Batch) -> int:
        parser = getattr(self.dataset, "_sequence_idx_from_path", None)
        image_path = getattr(gpu_batch, "image_path", None)
        if not callable(parser) or not image_path:
            raise ValueError(
                "inference_sequence_metadata requires the eval dataset's "
                "deterministic image-name sequence parser and source path."
            )
        parsed_sequence_idx = int(parser(str(image_path)))
        batch_sequence_idx = int(getattr(gpu_batch, "sequence_idx", -1))
        if parsed_sequence_idx < 0:
            raise ValueError(
                "inference_sequence_metadata could not parse a non-negative "
                f"sequence index from image name {image_path!r}."
            )
        if batch_sequence_idx != parsed_sequence_idx:
            raise ValueError(
                "Eval batch sequence_idx does not match the dataset's "
                "deterministic image-name parser: "
                f"{batch_sequence_idx} versus {parsed_sequence_idx}."
            )
        return parsed_sequence_idx

    def _apply_inference_post_processing(
        self,
        outputs: dict[str, torch.Tensor | float],
        gpu_batch: Batch,
    ) -> dict[str, torch.Tensor | float]:
        """Apply sealed novel-view post-processing without held-out metadata."""
        if self.post_processing is None:
            return outputs
        eval_camera_index = post_processing_camera_idx(
            int(
                getattr(
                    gpu_batch,
                    "post_processing_camera_idx",
                    gpu_batch.camera_idx,
                )
            ),
            self._post_processing_camera_index_mode,
        )
        mapping = getattr(
            self,
            "_post_processing_camera_mapping",
            None,
        )
        checkpoint_camera_index = eval_camera_index
        if mapping is not None:
            if not 0 <= eval_camera_index < len(mapping):
                raise ValueError(
                    "Eval post-processing camera index is outside the "
                    "validated camera mapping: "
                    f"{eval_camera_index} versus {len(mapping)} slots."
                )
            checkpoint_camera_index = mapping[eval_camera_index]

        sequence_idx = -1
        if getattr(self, "post_processing_mode", None) == (POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA):
            sequence_idx = self._sequence_idx_from_eval_batch(gpu_batch)

        sealed_batch = copy.copy(gpu_batch)
        sealed_batch.sequence_idx = sequence_idx
        sealed_batch.exposure = None
        return apply_post_processing(
            self.post_processing,
            outputs,
            sealed_batch,
            training=False,
            camera_idx_override=checkpoint_camera_index,
            use_known_frame=getattr(
                self,
                "_use_known_frame_post_processing",
                False,
            ),
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

        render_leaf = "renders" if self.split == "val" else "train_renders"
        gt_leaf = "gt" if self.split == "val" else "train_gt"
        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", render_leaf)
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", gt_leaf)
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

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)
            if self.camera_residual is not None:
                gpu_batch = _apply_native_camera_extrinsics(
                    self.camera_residual,
                    gpu_batch,
                    global_step=int(self.global_step),
                )

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

            # Apply sealed novel-view post-processing: camera-mapped to the
            # checkpoint slots, with exposure/sequence sealing per eval mode.
            # This is the single post-processing pass (do not also call
            # apply_post_processing directly, or effects apply twice).
            outputs = self._apply_inference_post_processing(outputs, gpu_batch)

            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            progress_psnr = psnr[-1]
            if gpu_batch.mask is not None:
                mask = gpu_batch.mask
                masked_error = torch.square(outputs["pred_rgb"] - gpu_batch.rgb_gt) * mask
                masked_denominator = torch.clamp_min(
                    mask.sum() * gpu_batch.rgb_gt.shape[-1],
                    1.0,
                )
                masked_mse = masked_error.sum() / masked_denominator
                masked_psnr_single_img = (-10.0 * torch.log10(torch.clamp_min(masked_mse, 1e-12))).item()
                masked_psnr.append(masked_psnr_single_img)
                mask_coverage.append(mask.mean().item())
                progress_psnr = masked_psnr[-1]
                logger.info(
                    f"Frame {iteration}, image: {os.path.basename(gpu_batch.image_path)}, "
                    f"PSNR: {psnr[-1]}, masked PSNR: {masked_psnr[-1]}"
                )
            else:
                logger.info(f"Frame {iteration}, image: {os.path.basename(gpu_batch.image_path)}, " f"PSNR: {psnr[-1]}")

            frame_metrics = {
                "eval_index": int(iteration),
                "split_frame_idx": int(gpu_batch.frame_idx),
                "split": self.split,
                "camera_idx": int(gpu_batch.camera_idx),
                "image_name": os.path.basename(gpu_batch.image_path),
                "image_relative_name": self._eval_image_relative_name(
                    iteration,
                    gpu_batch.image_path,
                ),
                "image_path": gpu_batch.image_path,
                "psnr": float(psnr_single_img),
            }
            if self.post_processing_mode == (POST_PROCESSING_EVAL_MODE_INFERENCE_SEQUENCE_METADATA):
                frame_metrics["sequence_idx"] = int(gpu_batch.sequence_idx)
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
                frame_metrics["ssim"] = float(ssim[-1])
                frame_metrics["lpips"] = float(lpips[-1])

            if masked_psnr:
                frame_metrics["masked_psnr"] = float(masked_psnr[-1])
            if mask_coverage:
                frame_metrics["mask_coverage"] = float(mask_coverage[-1])
            per_frame_metrics.append(frame_metrics)

            # Color-corrected metrics
            pred_rgb_cc = color_correct_affine(pred_rgb_full, rgb_gt_full)
            cc_psnr.append(criterions["psnr"](pred_rgb_cc, rgb_gt_full).item())
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

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=progress_psnr)

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
            table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"

        # Save metrics to JSON file
        metrics_json = dict(mean_psnr=float(mean_psnr))
        metrics_json["mean_inference_time_ms"] = float(mean_inference_time)
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
        per_frame_metrics_name = "per_frame_metrics.json" if self.split == "val" else "per_frame_train_metrics.json"
        per_frame_metrics_path = os.path.join(self.out_dir, per_frame_metrics_name)
        with open(per_frame_metrics_path, "w") as f:
            json.dump(per_frame_metrics, f, indent=2)
        logger.info(f"📄 Per-frame metrics saved to: {per_frame_metrics_path}")

        # Provenance sidecar: pin exactly which code + seed + holdout produced
        # these metrics so a run is reproducible and auditable.
        submodule_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(submodule_dir, os.pardir, os.pardir, os.pardir))
        submodule_sha, submodule_dirty = _git_sha_and_dirty(submodule_dir)
        parent_sha, parent_dirty = _git_sha_and_dirty(parent_dir)
        provenance = {
            "submodule_git_sha": submodule_sha,
            "submodule_git_dirty": submodule_dirty,
            "parent_git_sha": parent_sha,
            "parent_git_dirty": parent_dirty,
            "experiment_name": self.conf.get("experiment_name", ""),
            "config_path": self.conf.get("path", ""),
            "n_iterations": int(self.conf.get("n_iterations", 0)),
            "global_step": int(self.global_step),
            "seed_initialization": int(self.conf.get("seed_initialization", -1)),
            "split": self.split,
        }
        provenance |= self._common_eval_contract_provenance(per_frame_metrics)
        provenance_path = os.path.join(self.out_dir, "provenance.json")
        with open(provenance_path, "w") as f:
            json.dump(provenance, f, indent=2)
        logger.info(f"📄 Provenance saved to: {provenance_path}")

        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("test/psnr", mean_psnr, self.global_step)
            if mean_masked_psnr is not None:
                self.writer.add_scalar("test/masked_psnr", mean_masked_psnr, self.global_step)
            if mean_mask_coverage is not None:
                self.writer.add_scalar("test/mask_coverage", mean_mask_coverage, self.global_step)
            if mean_ssim is not None:
                self.writer.add_scalar("test/ssim", mean_ssim, self.global_step)
            if mean_lpips is not None:
                self.writer.add_scalar("test/lpips", mean_lpips, self.global_step)
            if mean_cc_psnr is not None:
                self.writer.add_scalar("test/color_corrected_psnr", mean_cc_psnr, self.global_step)
            if mean_cc_ssim is not None:
                self.writer.add_scalar("test/color_corrected_ssim", mean_cc_ssim, self.global_step)
            if mean_cc_lpips is not None:
                self.writer.add_scalar(
                    "test/color_corrected_lpips",
                    mean_cc_lpips,
                    self.global_step,
                )
            self.writer.add_scalar("time/test/inference", mean_inference_time, self.global_step)

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
