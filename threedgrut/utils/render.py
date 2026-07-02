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


import torch
from omegaconf import DictConfig

## NOTE: SPH code from gaussian-splatting, from plenoctree, from ???
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

POST_PROCESSING_CAMERA_INDEX_DATASET = "dataset"
POST_PROCESSING_CAMERA_INDEX_SINGLE_PHYSICAL = "single_physical_camera"
POST_PROCESSING_CAMERA_INDEX_MODES = frozenset(
    (
        POST_PROCESSING_CAMERA_INDEX_DATASET,
        POST_PROCESSING_CAMERA_INDEX_SINGLE_PHYSICAL,
    )
)


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def post_processing_camera_index_mode(conf: DictConfig) -> str:
    mode = conf.post_processing.get(
        "camera_index_mode",
        POST_PROCESSING_CAMERA_INDEX_DATASET,
    )
    if mode not in POST_PROCESSING_CAMERA_INDEX_MODES:
        raise ValueError(
            "Unsupported post_processing.camera_index_mode="
            f"{mode!r}. Expected one of "
            f"{sorted(POST_PROCESSING_CAMERA_INDEX_MODES)}."
        )
    return mode


def post_processing_frames_per_camera(
    frames_per_camera: list[int],
    camera_index_mode: str,
) -> list[int]:
    if camera_index_mode == POST_PROCESSING_CAMERA_INDEX_DATASET:
        return frames_per_camera
    if camera_index_mode == POST_PROCESSING_CAMERA_INDEX_SINGLE_PHYSICAL:
        return [sum(frames_per_camera)]
    raise ValueError(
        "Unsupported post-processing camera index mode: "
        f"{camera_index_mode!r}."
    )


def post_processing_camera_idx(
    camera_idx: int,
    camera_index_mode: str,
) -> int:
    if camera_index_mode == POST_PROCESSING_CAMERA_INDEX_DATASET:
        return camera_idx
    if camera_index_mode == POST_PROCESSING_CAMERA_INDEX_SINGLE_PHYSICAL:
        return 0
    raise ValueError(
        "Unsupported post-processing camera index mode: "
        f"{camera_index_mode!r}."
    )


def _edge_gate(image_bhwc: torch.Tensor) -> torch.Tensor:
    """Return a robust 0-1 edge gate for a `[1, H, W, C]` image tensor."""
    image = torch.nan_to_num(image_bhwc.detach())
    if image.shape[-1] > 1:
        image = image.mean(dim=-1, keepdim=True)
    magnitude = torch.zeros_like(image)
    magnitude[:, :, 1:, :] = magnitude[:, :, 1:, :] + torch.abs(
        image[:, :, 1:, :] - image[:, :, :-1, :]
    )
    magnitude[:, 1:, :, :] = magnitude[:, 1:, :, :] + torch.abs(
        image[:, 1:, :, :] - image[:, :-1, :, :]
    )
    finite = magnitude[torch.isfinite(magnitude)]
    if finite.numel() == 0:
        return torch.zeros_like(magnitude)
    scale = torch.quantile(finite, 0.95).clamp_min(1e-6)
    return (magnitude / scale).clamp(0.0, 1.0)


def _residual_grid_edge_gate(outputs: dict) -> torch.Tensor | None:
    """Build a render/depth edge gate for image-conditioned residuals."""
    pred_rgb = outputs.get("pred_rgb")
    if pred_rgb is None:
        return None
    gate = _edge_gate(pred_rgb)
    pred_dist = outputs.get("pred_dist")
    if pred_dist is not None:
        gate = torch.maximum(gate, _edge_gate(pred_dist))
    pred_opacity = outputs.get("pred_opacity")
    if pred_opacity is not None:
        opacity = torch.nan_to_num(pred_opacity.detach()).clamp(0.0, 1.0)
        gate = gate * opacity
    return gate


def apply_post_processing(
    post_processing,
    outputs: dict,
    gpu_batch,
    training: bool = False,
    camera_idx_override: int | None = None,
) -> dict:
    """Apply post-processing to rendered output.

    Args:
        post_processing: Post-processing module
        outputs: Model outputs including pred_rgb
        gpu_batch: Batch containing camera_idx, frame_idx, sequence_idx,
            pixel_coords, exposure
        training: If True, use actual frame_idx; if False, use -1 for novel view mode
        camera_idx_override: Optional post-processing-only camera index.

    Returns:
        Updated outputs dict with post-processed pred_rgb
    """
    assert outputs["pred_rgb"].shape[0] == 1, (
        "Post-processing requires batch_size=1"
    )

    pred_rgb = outputs["pred_rgb"]
    camera_idx = (
        gpu_batch.camera_idx
        if camera_idx_override is None
        else camera_idx_override
    )
    frame_idx = gpu_batch.frame_idx if training else -1
    sequence_idx = getattr(gpu_batch, "sequence_idx", -1)
    H, W = pred_rgb.shape[1], pred_rgb.shape[2]

    # Flatten: [1, H, W, 3] -> [H*W, 3]
    # Ensure contiguous memory for CUDA kernels
    pred_rgb_flat = pred_rgb.contiguous().view(-1, 3)
    pixel_coords_flat = gpu_batch.pixel_coords.contiguous().view(-1, 2)

    # Apply post-processing
    residual_grid_gate = None
    if getattr(post_processing, "use_residual_grid_edge_gate", False):
        residual_grid_gate = _residual_grid_edge_gate(outputs)
        if residual_grid_gate is not None:
            residual_grid_gate = residual_grid_gate.contiguous().view(-1)
    kwargs = {
        "resolution": (W, H),
        "camera_idx": camera_idx,
        "frame_idx": frame_idx,
        "exposure_prior": gpu_batch.exposure,
    }
    if hasattr(post_processing, "use_temporal_affine"):
        kwargs["sequence_idx"] = sequence_idx
    if hasattr(post_processing, "use_residual_grid"):
        kwargs["residual_grid_gate"] = residual_grid_gate
    pred_rgb_pp = post_processing(
        pred_rgb_flat,
        pixel_coords_flat,
        **kwargs,
    )

    # Reshape back: [H*W, 3] -> [1, H, W, 3]
    outputs["pred_rgb"] = pred_rgb_pp.view(pred_rgb.shape)
    return outputs
