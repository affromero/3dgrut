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

## NOTE: SPH code from gaussian-splatting, from plenoctree, from ???
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
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


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


POST_PROCESSING_CAMERA_INDEX_DATASET = "dataset"
POST_PROCESSING_CAMERA_INDEX_SINGLE_PHYSICAL = "single_physical_camera"
POST_PROCESSING_CAMERA_INDEX_MODES = frozenset(
    (
        POST_PROCESSING_CAMERA_INDEX_DATASET,
        POST_PROCESSING_CAMERA_INDEX_SINGLE_PHYSICAL,
    )
)


def post_processing_camera_index_mode(conf) -> str:
    """Selected camera bucketing mode for per-camera post-processing.

    "dataset" keeps the dataset's camera indices (one post-processing bucket
    per dataset camera); "single_physical_camera" folds every frame into one
    bucket, for captures where the dataset splits one physical camera into
    many logical ones.
    """
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
        f"Unsupported post-processing camera index mode: {camera_index_mode!r}."
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
        f"Unsupported post-processing camera index mode: {camera_index_mode!r}."
    )


def apply_feature_decoder(
    feature_decoder,
    outputs: dict,
    gpu_batch,
    training: bool = False,
    center_ray_encoding: bool = False,
) -> dict:
    """Apply feature decoder to N-dimensional feature map."""
    if feature_decoder is None:
        return outputs

    feature_map = outputs["pred_rgb"]  # [B, H, W, N] alpha-blended features
    alpha = outputs["pred_opacity"]  # [B, H, W] or [B, H, W, 1]
    B, H, W, N = feature_map.shape

    rays_dir_cam = gpu_batch.rays_dir  # [B, H, W, 3]
    # GUI camera batches keep poses on the CPU while rays/features are on the
    # GPU. Match the pose device and ray dtype before computing world rays.
    R = gpu_batch.T_to_world[:, :3, :3].to(device=rays_dir_cam.device, dtype=rays_dir_cam.dtype)  # [B, 3, 3]
    if center_ray_encoding:
        # center-ray mode uses the camera optical axis, i.e. row 2 of the
        # world-to-camera view matrix, which is equivalent to column 2 of
        # camera-to-world for OpenCV convention.
        center_ray_world = torch.nn.functional.normalize(R[:, :, 2], dim=-1)
        rays_dir_world = center_ray_world.view(B, 1, 1, 3).expand(B, H, W, 3)
    else:
        rays_dir_world = torch.einsum("bij,bhwj->bhwi", R, rays_dir_cam)
        rays_dir_world = torch.nn.functional.normalize(rays_dir_world, dim=-1)

    features_flat = feature_map.contiguous().view(-1, N)
    ray_dir_flat = rays_dir_world.contiguous().view(-1, 3)
    if alpha.dim() == 3:
        alpha = alpha.unsqueeze(-1)  # [B, H, W, 1]
    alpha_flat = alpha.contiguous().view(-1, 1)

    rgb_flat = feature_decoder(features_flat, ray_dir_flat, alpha=alpha_flat)
    outputs["pred_rgb"] = rgb_flat.view(B, H, W, 3)

    return outputs


def _edge_gate(image_bhwc: torch.Tensor) -> torch.Tensor:
    """Return a robust 0-1 edge gate for a `[1, H, W, C]` image tensor."""
    image = torch.nan_to_num(image_bhwc.detach())
    if image.shape[-1] > 1:
        image = image.mean(dim=-1, keepdim=True)
    magnitude = torch.zeros_like(image)
    magnitude[:, :, 1:, :] = magnitude[:, :, 1:, :] + torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    magnitude[:, 1:, :, :] = magnitude[:, 1:, :, :] + torch.abs(image[:, 1:, :, :] - image[:, :-1, :, :])
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
    use_known_frame: bool | None = None,
) -> dict:
    """Apply post-processing to rendered output.

    Args:
        post_processing: Post-processing module
        outputs: Model outputs including pred_rgb
        gpu_batch: Batch containing camera_idx, frame_idx, sequence_idx,
            pixel_coords, exposure
        training: If True, use actual frame_idx; if False, use -1 for novel view mode
        camera_idx_override: Optional post-processing-only camera index (used by
            common-eval to map the eval camera onto the checkpoint camera slot).

    Returns:
        Updated outputs dict with post-processed pred_rgb
    """
    assert outputs["pred_rgb"].shape[0] == 1, "Post-processing requires batch_size=1"

    pred_rgb = outputs["pred_rgb"]
    camera_idx = (
        camera_idx_override
        if camera_idx_override is not None
        else post_processing_camera_idx(
            getattr(gpu_batch, "post_processing_camera_idx", gpu_batch.camera_idx),
            getattr(
                post_processing,
                "camera_index_mode",
                POST_PROCESSING_CAMERA_INDEX_DATASET,
            ),
        )
    )
    apply_known_frame = (
        training if use_known_frame is None else use_known_frame
    )
    known_frame_idx = (
        getattr(gpu_batch, "source_frame_idx", -1)
        if getattr(post_processing, "use_native_appearance_grid", False)
        else gpu_batch.frame_idx
    )
    frame_idx = known_frame_idx if apply_known_frame else -1
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
    post_processing_kwargs = {
        "resolution": (W, H),
        "camera_idx": camera_idx,
        "frame_idx": frame_idx,
        "exposure_prior": gpu_batch.exposure,
    }
    if hasattr(post_processing, "use_temporal_affine"):
        post_processing_kwargs["sequence_idx"] = sequence_idx
    if hasattr(post_processing, "use_residual_grid"):
        post_processing_kwargs["residual_grid_gate"] = residual_grid_gate
    if getattr(post_processing, "use_view_context", False):
        if bool(getattr(gpu_batch, "rays_in_world_space", False)):
            raise ValueError(
                "View-conditioned post-processing requires camera-space "
                "rays and a non-identity camera-to-world pose."
            )
        end_pose = getattr(gpu_batch, "T_to_world_end", None)
        if end_pose is not None and not torch.allclose(
            end_pose,
            gpu_batch.T_to_world,
            atol=1e-6,
        ):
            raise ValueError(
                "View-conditioned post-processing supports global shutter "
                "only; distinct start and end poses are ambiguous."
            )
        post_processing_kwargs["render_ray_distance"] = outputs.get("pred_dist")
        post_processing_kwargs["render_opacity"] = outputs.get("pred_opacity")
        post_processing_kwargs["camera_to_world"] = gpu_batch.T_to_world
    pred_rgb_pp = post_processing(
        pred_rgb_flat,
        pixel_coords_flat,
        **post_processing_kwargs,
    )

    # Reshape back: [H*W, 3] -> [1, H, W, 3]
    outputs["pred_rgb"] = pred_rgb_pp.view(pred_rgb.shape)
    return outputs
