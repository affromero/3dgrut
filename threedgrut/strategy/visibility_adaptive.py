"""Recovered deterministic visibility-adaptive Gaussian densification primitives."""

import math

import torch

from threedgrut.model.losses import (
    indexed_camera_loss_weight,
    fixed_image_loss_denominator,
)
from threedgrut.optimizers.visibility_selective_adam import VisibilitySelectiveAdam
from threedgrut.strategy.base import BaseStrategy
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import quaternion_to_so3

VISIBILITY_ADAPTIVE_CHILD_OFFSET = 0.6744800209999084
VISIBILITY_ADAPTIVE_CHILD_SCALE = 0.8
VISIBILITY_ADAPTIVE_FLOAT_EPSILON = 1.1920928955078125e-7
VISIBILITY_ADAPTIVE_MAX_ITERATIONS = 100_000
VISIBILITY_ADAPTIVE_MAX_CLOUD_COUNT = 16_777_216
VISIBILITY_ADAPTIVE_STATUS_WINDOW_SIZE = 32.0
VISIBILITY_ADAPTIVE_SPARSE_INITIAL_OPACITY = 0.1
VISIBILITY_ADAPTIVE_DENSE_INITIAL_OPACITY = 0.1
VISIBILITY_ADAPTIVE_IMAGE_PREFETCH = 16
VISIBILITY_ADAPTIVE_IMAGE_HUBER_WEIGHT = 0.8
VISIBILITY_ADAPTIVE_IMAGE_SSIM_WEIGHT = 0.2
VISIBILITY_ADAPTIVE_LIDAR_LOG_WEIGHT = 0.05
VISIBILITY_ADAPTIVE_POSITION_LEARNING_RATE = 0.001
VISIBILITY_ADAPTIVE_SH_LEARNING_RATE = 0.1
VISIBILITY_ADAPTIVE_OPACITY_LEARNING_RATE = 0.05
VISIBILITY_ADAPTIVE_SCALE_LEARNING_RATE = 0.005
VISIBILITY_ADAPTIVE_ROTATION_LEARNING_RATE = 0.001
VISIBILITY_ADAPTIVE_APPEARANCE_LEARNING_RATE = 0.05
VISIBILITY_ADAPTIVE_ENCODER_LEARNING_RATE = 0.001
VISIBILITY_ADAPTIVE_CAMERA_QUATERNION_LEARNING_RATE = 0.00001
VISIBILITY_ADAPTIVE_CAMERA_TRANSLATION_LEARNING_RATE = 0.001
VISIBILITY_ADAPTIVE_COHERENCE_SCALE = 8.0
VISIBILITY_ADAPTIVE_COHERENCE_INTERVAL = 100
VISIBILITY_ADAPTIVE_COHERENCE_BLOCK_SIZE = 1024
VISIBILITY_ADAPTIVE_COHERENCE_PARTIAL_PREFIX = 512
VISIBILITY_ADAPTIVE_TOPOLOGY_INTERVAL = 100
VISIBILITY_ADAPTIVE_MIN_BATCH_SIZE = 600
VISIBILITY_ADAPTIVE_ITERATIVE_PRUNE_FRACTION = 0.1
VISIBILITY_ADAPTIVE_ITERATIVE_PRUNE_WEIGHT_THRESHOLD = 50.0
VISIBILITY_ADAPTIVE_REGULAR_PRUNE_RADIUS_THRESHOLD = 0.01
VISIBILITY_ADAPTIVE_DENSIFY_GRAD_THRESHOLD = 0.0002
VISIBILITY_ADAPTIVE_DENSIFY_SIZE_THRESHOLD = 8.0
VISIBILITY_ADAPTIVE_DENSIFY_WEIGHT_THRESHOLD = 150.0
VISIBILITY_ADAPTIVE_COMPACT_WEIGHT_MINIMUM = 1.0e-6
VISIBILITY_ADAPTIVE_ALPHA_LEVELS = 255.0
VISIBILITY_ADAPTIVE_SIZE_DECAY = 0.9
VISIBILITY_ADAPTIVE_RADIUS_DECAY = 0.98
VISIBILITY_ADAPTIVE_SCALE_STATUS_MINIMUM = 0.1
VISIBILITY_ADAPTIVE_SCALE_SHRINK_STATUS = 1.5
VISIBILITY_ADAPTIVE_SCALE_SHRINK_FACTOR = 0.95
VISIBILITY_ADAPTIVE_ENVIRONMENT_ROTATION_GRADIENT = 1.0e-7
VISIBILITY_ADAPTIVE_ENVIRONMENT_RADIAL_GRADIENT = 1.0e-8
VISIBILITY_ADAPTIVE_ENVIRONMENT_AXIAL_GRADIENT = 1.0e-9
VISIBILITY_ADAPTIVE_ENVIRONMENT_AXIAL_RATIO = 1.0 / 30.0
VISIBILITY_ADAPTIVE_ENVIRONMENT_SCALE_RATIO_LIMIT = 0.5
VISIBILITY_ADAPTIVE_WEIGHT_ACCUMULATION_MULTIPLIER = 1.0
VISIBILITY_ADAPTIVE_RENDERER_VISIBILITY_SOURCE = "renderer"
VISIBILITY_ADAPTIVE_COLOR_GRADIENT_WEIGHT_VISIBILITY_SOURCE = (
    "color_gradient_weight"
)
VISIBILITY_ADAPTIVE_FORWARD_VISIBILITY_SOURCE = "forward_visibility"
VISIBILITY_ADAPTIVE_MASK_CENTER_WEIGHT_VISIBILITY_SOURCE = "mask_center_weight"
VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_CONIC_RADIUS = "conic_radius"
VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_EXTENT_MAX = "extent_max"
VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT = "rendered_weight"
VISIBILITY_ADAPTIVE_PROJECTED_SIZE_MULTIPLIER = 1.0
VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_RAW = "raw_position"
VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_TANGENT = "tangent_distance"
VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN = "jacobian"
VISIBILITY_ADAPTIVE_TANGENT_PROJECTED_GRADIENT_SCALE = 0.8
VISIBILITY_ADAPTIVE_STATUS_GRADIENT_SCALE = 1.0
VISIBILITY_ADAPTIVE_OPTIMIZE_ENCODING = True
VISIBILITY_ADAPTIVE_OPTIMIZE_APPEARANCE = True
VISIBILITY_ADAPTIVE_OPTIMIZE_EXTRINSICS = True
VISIBILITY_ADAPTIVE_OPTIMIZE_DISTORTION = True
VISIBILITY_ADAPTIVE_COARSE_TO_FINE = True


def _visibility_adaptive_expand_morton_bits(values: torch.Tensor) -> torch.Tensor:
    expanded = values & 0x3FF
    expanded = (expanded | (expanded << 16)) & 0x030000FF
    expanded = (expanded | (expanded << 8)) & 0x0300F00F
    expanded = (expanded | (expanded << 4)) & 0x030C30C3
    expanded = (expanded | (expanded << 2)) & 0x09249249
    return expanded


@torch.no_grad()
def visibility_adaptive_coherence_permutation(
    *,
    positions: torch.Tensor,
    scale: float,
    include_prefix: bool = True,
) -> torch.Tensor:
    """Return the native block-local Z/Y/X Morton permutation."""
    point_count = positions.shape[0]
    if positions.shape != (point_count, 3):
        raise ValueError("Native coherence positions must have shape (N, 3).")
    if point_count == 0:
        raise ValueError("Native coherence requires at least one Gaussian.")
    if scale <= 0.0:
        raise ValueError("Native coherence scale must be positive.")

    start_index = (
        0
        if include_prefix
        else min(VISIBILITY_ADAPTIVE_COHERENCE_PARTIAL_PREFIX, point_count)
    )
    if start_index == point_count:
        return torch.arange(
            point_count,
            dtype=torch.long,
            device=positions.device,
        )

    coordinates = torch.trunc(positions[start_index:] * scale).to(
        dtype=torch.int64
    )
    x = _visibility_adaptive_expand_morton_bits(coordinates[:, 0])
    y = _visibility_adaptive_expand_morton_bits(coordinates[:, 1])
    z = _visibility_adaptive_expand_morton_bits(coordinates[:, 2])
    keys = z | (y << 1) | (x << 2)

    block_size = VISIBILITY_ADAPTIVE_COHERENCE_BLOCK_SIZE
    processed_count = point_count - start_index
    full_count = processed_count - processed_count % block_size
    permutations: list[torch.Tensor] = []
    if start_index > 0:
        permutations.append(
            torch.arange(
                start_index,
                dtype=torch.long,
                device=positions.device,
            )
        )
    if full_count > 0:
        local = torch.argsort(
            keys[:full_count].reshape(-1, block_size),
            dim=1,
            stable=True,
        )
        offsets = torch.arange(
            0,
            full_count,
            block_size,
            dtype=local.dtype,
            device=local.device,
        )[:, None]
        permutations.append((local + offsets + start_index).reshape(-1))
    if full_count < processed_count:
        tail = (
            torch.argsort(keys[full_count:], stable=True)
            + full_count
            + start_index
        )
        permutations.append(tail)
    return torch.cat(permutations)


def visibility_adaptive_topology_schedule(
    *,
    step: int,
    total_iterations: int,
    iteration_per_batch: int,
    prune_interval: int = VISIBILITY_ADAPTIVE_TOPOLOGY_INTERVAL,
    densify_interval: int = VISIBILITY_ADAPTIVE_TOPOLOGY_INTERVAL,
) -> tuple[bool, bool, bool]:
    """Return native Step events followed by the host batch-prune event."""
    if step < 0:
        raise ValueError("Native topology step must be nonnegative.")
    if total_iterations <= 0:
        raise ValueError("Native topology iterations must be positive.")
    if iteration_per_batch <= 0:
        raise ValueError(
            "Native topology requires at least one effective batch image."
        )
    if prune_interval < 0 or densify_interval < 0:
        raise ValueError("Native topology intervals must be nonnegative.")

    batch_size = max(
        VISIBILITY_ADAPTIVE_MIN_BATCH_SIZE,
        iteration_per_batch,
    )
    # The native host enables both Step intervals when one effective image
    # batch completes. Its boundary update executes the regular actions in the
    # same topology pass as the batch prune, then repeats at relative offsets.
    completed_iteration = step + 1
    regular_interval_enabled = completed_iteration >= batch_size
    regular_phase = completed_iteration - batch_size
    regular_prune = (
        regular_interval_enabled
        and prune_interval > 0
        and regular_phase % prune_interval == 0
    )
    batch_start = (completed_iteration // batch_size) * batch_size
    densify_enabled = (
        total_iterations >= completed_iteration + batch_size
        and total_iterations >= batch_start + 2 * batch_size
    )
    regular_densify = (
        regular_interval_enabled
        and densify_interval > 0
        and regular_phase % densify_interval == 0
        and densify_enabled
    )
    iterative_prune = (
        completed_iteration >= batch_size
        and completed_iteration % batch_size == 0
        and total_iterations >= completed_iteration + batch_size
    )
    return regular_prune, regular_densify, iterative_prune


@torch.no_grad()
def visibility_adaptive_regular_prune_keep_mask(
    *,
    physical_opacity: torch.Tensor,
    status_radius: torch.Tensor,
    opacity_threshold: float,
    radius_threshold: float,
) -> torch.Tensor:
    """Keep scheduled-prune rows that are opaque or recently large."""
    point_count = physical_opacity.shape[0]
    if physical_opacity.shape != (point_count,):
        raise ValueError("Native regular-prune opacity must have shape (N,).")
    if status_radius.shape != (point_count,):
        raise ValueError("Native regular-prune radius must have shape (N,).")
    if status_radius.device != physical_opacity.device:
        raise ValueError("Native regular-prune inputs must share a device.")
    if not math.isfinite(opacity_threshold) or opacity_threshold <= 0.0:
        raise ValueError("Native regular-prune opacity threshold must be positive.")
    if not math.isfinite(radius_threshold) or radius_threshold <= 0.0:
        raise ValueError("Native regular-prune radius threshold must be positive.")
    return (physical_opacity >= opacity_threshold) | (
        status_radius >= radius_threshold
    )


@torch.no_grad()
def visibility_adaptive_update_point_status(
    *,
    status: torch.Tensor,
    point_indices: torch.Tensor,
    projected_size_pixels: torch.Tensor,
    visible_measurements: torch.Tensor,
    window_size: float,
) -> torch.Tensor:
    """Apply the recovered native per-visible-point status recurrence."""
    point_count = status.shape[0]
    if status.shape != (point_count, 4):
        msg = "Native point status must have shape (N, 4)."
        raise ValueError(msg)
    visible_count = point_indices.shape[0]
    if point_indices.shape != (visible_count,):
        msg = "Native visible point indices must have shape (V,)."
        raise ValueError(msg)
    if projected_size_pixels.shape != (point_count,):
        msg = "Native projected point sizes must have shape (N,)."
        raise ValueError(msg)
    if visible_measurements.shape != (visible_count, 2):
        msg = "Native visible measurements must have shape (V, 2)."
        raise ValueError(msg)
    if window_size <= 0.0:
        msg = "Native point-status window size must be positive."
        raise ValueError(msg)

    updated = status.clone()
    previous = status[point_indices]
    observation_count = previous[:, 0] + 1.0
    normalized_radii = visible_measurements[:, 0]
    projected_gradient_pixels = visible_measurements[:, 1]
    rolling_gradient = (
        previous[:, 3] + projected_gradient_pixels / window_size
    )
    over_window = observation_count > window_size
    rolling_gradient = torch.where(
        over_window,
        rolling_gradient * window_size / observation_count,
        rolling_gradient,
    )

    updated[point_indices, 0] = torch.minimum(
        observation_count,
        torch.full_like(observation_count, window_size),
    )
    updated[point_indices, 1] = torch.maximum(
        previous[:, 1] * VISIBILITY_ADAPTIVE_SIZE_DECAY,
        projected_size_pixels[point_indices],
    )
    updated[point_indices, 2] = torch.maximum(
        previous[:, 2] * VISIBILITY_ADAPTIVE_RADIUS_DECAY,
        normalized_radii,
    )
    updated[point_indices, 3] = rolling_gradient
    return updated


@torch.no_grad()
def visibility_adaptive_projected_size_from_weight(
    *,
    rendered_weight: torch.Tensor,
) -> torch.Tensor:
    """Return the native projected-size signal from rendered weight."""
    if rendered_weight.ndim == 2 and rendered_weight.shape[1] == 1:
        scalar_weight = rendered_weight.reshape(-1)
    elif rendered_weight.ndim == 1:
        scalar_weight = rendered_weight
    else:
        raise ValueError("Native rendered weight must be N or Nx1.")

    finite_weight = torch.nan_to_num(
        scalar_weight,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return torch.clamp(finite_weight, min=0.0)


@torch.no_grad()
def visibility_adaptive_projected_size_pixels(
    *,
    rendered_weight: torch.Tensor,
    projected_extent: torch.Tensor,
    projected_conic_opacity: torch.Tensor,
    image_width: int,
    image_height: int,
    source: str = VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT,
    multiplier: float | torch.Tensor = VISIBILITY_ADAPTIVE_PROJECTED_SIZE_MULTIPLIER,
) -> torch.Tensor:
    """Return the projected-size signal used by native topology state."""
    multiplier_tensor = torch.as_tensor(
        multiplier,
        dtype=projected_conic_opacity.dtype,
        device=projected_conic_opacity.device,
    )
    if multiplier_tensor.shape != ():
        raise ValueError("Native projected-size multiplier must be scalar.")
    if (
        not torch.isfinite(multiplier_tensor).item()
        or multiplier_tensor.item() <= 0.0
    ):
        raise ValueError(
            "Native projected-size multiplier must be finite and positive."
        )
    if source == VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT:
        return multiplier_tensor * visibility_adaptive_projected_size_from_weight(
            rendered_weight=rendered_weight,
        )
    if source == VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_EXTENT_MAX:
        point_count = projected_extent.shape[0]
        if projected_extent.shape != (point_count, 2):
            raise ValueError("Native projected extent must have shape Nx2.")
        finite_extent = torch.nan_to_num(
            projected_extent,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return multiplier_tensor * torch.clamp(finite_extent, min=0.0).amax(
            dim=1
        )
    if source != VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_CONIC_RADIUS:
        raise ValueError(f"Unsupported native projected-size source: {source}.")
    return multiplier_tensor * visibility_adaptive_projected_radius_pixels(
        projected_conic_opacity=projected_conic_opacity,
        image_width=image_width,
        image_height=image_height,
    )


@torch.no_grad()
def visibility_adaptive_projected_gradient_pixels(
    *,
    projected_position_gradient: torch.Tensor,
    image_width: int,
    image_height: int,
    position_gradient: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    camera_to_world: torch.Tensor | None = None,
    focal_length: torch.Tensor | None = None,
    proxy_mode: str = VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN,
    proxy_scale: float = VISIBILITY_ADAPTIVE_TANGENT_PROJECTED_GRADIENT_SCALE,
) -> torch.Tensor:
    """Return native screen-space gradient or a 3DGRUT ray-gradient proxy."""
    point_count = projected_position_gradient.shape[0]
    if projected_position_gradient.shape != (point_count, 2):
        raise ValueError("Native projected gradients must be Nx2.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Native projected gradient image size is invalid.")

    finite_gradient = torch.nan_to_num(
        projected_position_gradient,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    projected_gradient_pixels = torch.sqrt(
        torch.square(0.5 * image_width * finite_gradient[:, 0])
        + torch.square(0.5 * image_height * finite_gradient[:, 1])
    )
    if torch.count_nonzero(projected_gradient_pixels) > 0:
        return projected_gradient_pixels

    if position_gradient is None:
        raise ValueError(
            "3DGRUT projected-gradient proxy requires position gradients."
        )
    if position_gradient.shape != (point_count, 3):
        raise ValueError("3DGRUT position gradients must be Nx3.")
    finite_position_gradient = torch.nan_to_num(
        position_gradient,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if proxy_mode == VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_RAW:
        return torch.linalg.vector_norm(finite_position_gradient, dim=1)
    if positions is not None and camera_to_world is not None:
        if proxy_mode == VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_TANGENT:
            return visibility_adaptive_tangent_projected_gradient_pixels(
                positions=positions,
                position_gradient=finite_position_gradient,
                camera_to_world=camera_to_world,
                scale=proxy_scale,
            )
        if (
            proxy_mode == VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN
            and focal_length is not None
        ):
            return visibility_adaptive_jacobian_projected_gradient_pixels(
                positions=positions,
                position_gradient=finite_position_gradient,
                camera_to_world=camera_to_world,
                focal_length=focal_length,
                image_width=image_width,
                image_height=image_height,
            )
        if proxy_mode != VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN:
            raise ValueError(
                f"Unknown native projected-gradient proxy {proxy_mode!r}."
            )
    return torch.linalg.vector_norm(finite_position_gradient, dim=1)


@torch.no_grad()
def visibility_adaptive_tangent_projected_gradient_pixels(
    *,
    positions: torch.Tensor,
    position_gradient: torch.Tensor,
    camera_to_world: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Approximate native image-plane gradients from tangent world gradients."""
    point_count = positions.shape[0]
    if positions.shape != (point_count, 3):
        raise ValueError("Native tangent-gradient positions must be Nx3.")
    if position_gradient.shape != (point_count, 3):
        raise ValueError("Native tangent-gradient position grads must be Nx3.")
    if camera_to_world.shape != (4, 4):
        raise ValueError("Native tangent-gradient camera pose must be 4x4.")
    if scale <= 0.0:
        raise ValueError("Native tangent-gradient scale must be positive.")

    dtype = positions.dtype
    device = positions.device
    camera_position = camera_to_world[:3, 3].to(device=device, dtype=dtype)
    offsets = positions - camera_position[None, :]
    distances = torch.linalg.vector_norm(offsets, dim=1)
    safe_distances = torch.where(
        distances > VISIBILITY_ADAPTIVE_FLOAT_EPSILON,
        distances,
        torch.ones_like(distances),
    )
    directions = offsets / safe_distances[:, None]
    finite_gradient = torch.nan_to_num(
        position_gradient.to(device=device, dtype=dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    radial = torch.sum(finite_gradient * directions, dim=1, keepdim=True)
    tangent = finite_gradient - radial * directions
    return (
        float(scale)
        * distances
        * torch.linalg.vector_norm(tangent, dim=1)
    )


@torch.no_grad()
def visibility_adaptive_jacobian_projected_gradient_pixels(
    *,
    positions: torch.Tensor,
    position_gradient: torch.Tensor,
    camera_to_world: torch.Tensor,
    focal_length: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Approximate native image-plane gradient from world-position gradient."""
    point_count = positions.shape[0]
    if positions.shape != (point_count, 3):
        raise ValueError("Native projected-gradient positions must be Nx3.")
    if position_gradient.shape != (point_count, 3):
        raise ValueError("Native projected-gradient position grads must be Nx3.")
    if camera_to_world.shape != (4, 4):
        raise ValueError("Native projected-gradient camera pose must be 4x4.")
    if focal_length.shape != (2,):
        raise ValueError("Native projected-gradient focal length must be 2D.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Native projected-gradient image size is invalid.")

    dtype = positions.dtype
    device = positions.device
    focal = focal_length.to(device=device, dtype=dtype)
    world_to_camera = torch.linalg.inv(
        camera_to_world.to(device=device, dtype=dtype)
    )
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    camera_positions = positions @ rotation.T + translation
    x = camera_positions[:, 0]
    y = camera_positions[:, 1]
    z = camera_positions[:, 2]
    safe_z = torch.where(
        torch.abs(z) > VISIBILITY_ADAPTIVE_FLOAT_EPSILON,
        z,
        torch.ones_like(z),
    )
    inv_z = torch.reciprocal(safe_z)
    inv_z2 = inv_z * inv_z
    zeros = torch.zeros_like(z)
    j0_camera = torch.stack(
        (
            focal[0] * inv_z,
            zeros,
            -focal[0] * x * inv_z2,
        ),
        dim=1,
    )
    j1_camera = torch.stack(
        (
            zeros,
            focal[1] * inv_z,
            -focal[1] * y * inv_z2,
        ),
        dim=1,
    )
    j0 = j0_camera @ rotation
    j1 = j1_camera @ rotation
    finite_gradient = torch.nan_to_num(
        position_gradient.to(device=device, dtype=dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    jj00 = torch.sum(j0 * j0, dim=1)
    jj01 = torch.sum(j0 * j1, dim=1)
    jj11 = torch.sum(j1 * j1, dim=1)
    jg0 = torch.sum(j0 * finite_gradient, dim=1)
    jg1 = torch.sum(j1 * finite_gradient, dim=1)
    determinant = jj00 * jj11 - jj01 * jj01
    valid = (
        torch.isfinite(determinant)
        & torch.isfinite(camera_positions).all(dim=1)
        & (torch.abs(z) > VISIBILITY_ADAPTIVE_FLOAT_EPSILON)
        & (torch.abs(determinant) > VISIBILITY_ADAPTIVE_FLOAT_EPSILON)
    )
    projected_u = torch.zeros_like(z)
    projected_v = torch.zeros_like(z)
    projected_u[valid] = (
        jj11[valid] * jg0[valid] - jj01[valid] * jg1[valid]
    ) / determinant[valid]
    projected_v[valid] = (
        -jj01[valid] * jg0[valid] + jj00[valid] * jg1[valid]
    ) / determinant[valid]

    return torch.sqrt(
        torch.square(0.5 * image_width * projected_u)
        + torch.square(0.5 * image_height * projected_v)
    )


def _visibility_adaptive_batch_focal_length(batch: object) -> torch.Tensor | None:
    camera_params = getattr(
        batch,
        "intrinsics_OpenCVPinholeCameraModelParameters",
        None,
    )
    if not isinstance(camera_params, dict):
        return None
    focal_length = camera_params.get("focal_length")
    if focal_length is None:
        return None
    focal_tensor = torch.as_tensor(focal_length)
    if focal_tensor.shape != (2,):
        return None
    return focal_tensor


def _validate_native_gradient_preprocess_tensors(
    *,
    raw_scales: torch.Tensor,
    raw_rotations: torch.Tensor,
    positions: torch.Tensor,
    scale_gradient: torch.Tensor,
    rotation_gradient: torch.Tensor,
    point_status: torch.Tensor,
    point_indices: torch.Tensor,
    environment_mask: torch.Tensor,
    camera_position: torch.Tensor,
    accumulation_weight_source: torch.Tensor,
    accumulated_weight: torch.Tensor,
) -> None:
    point_count = raw_scales.shape[0]
    expected_shapes = (
        (raw_scales, (point_count, 3), "raw scales"),
        (raw_rotations, (point_count, 4), "raw rotations"),
        (positions, (point_count, 3), "positions"),
        (scale_gradient, (point_count, 3), "scale gradients"),
        (rotation_gradient, (point_count, 4), "rotation gradients"),
        (point_status, (point_count, 4), "point status"),
        (environment_mask, (point_count,), "environment mask"),
        (camera_position, (3,), "camera position"),
        (
            accumulation_weight_source,
            (point_count, 1),
            "accumulation weight source",
        ),
        (accumulated_weight, (point_count, 1), "accumulated weight"),
    )
    for tensor, expected_shape, name in expected_shapes:
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Native {name} must have shape {expected_shape}."
            )
        if tensor.device != raw_scales.device:
            raise ValueError(f"Native {name} must share the Gaussian device.")
    if point_indices.ndim != 1 or point_indices.dtype != torch.int64:
        raise ValueError(
            "Native visible point indices must be a one-dimensional int64 "
            "tensor."
        )
    if point_indices.device != raw_scales.device:
        raise ValueError(
            "Native visible point indices must share the Gaussian device."
        )
    if environment_mask.dtype != torch.bool:
        raise ValueError("Native environment mask must use bool dtype.")
    float_tensors = (
        raw_scales,
        raw_rotations,
        positions,
        scale_gradient,
        rotation_gradient,
        point_status,
        camera_position,
        accumulation_weight_source,
        accumulated_weight,
    )
    if any(tensor.dtype != torch.float32 for tensor in float_tensors):
        raise ValueError(
            "Native gradient preprocessing requires float32 tensors."
        )
    if point_indices.numel() == 0:
        return
    if point_indices.min() < 0 or point_indices.max() >= point_count:
        raise ValueError("Native visible point indices are out of range.")


@torch.no_grad()
def visibility_adaptive_preprocess_gradients(
    *,
    raw_scales: torch.Tensor,
    raw_rotations: torch.Tensor,
    positions: torch.Tensor,
    scale_gradient: torch.Tensor,
    rotation_gradient: torch.Tensor,
    point_status: torch.Tensor,
    point_indices: torch.Tensor,
    environment_mask: torch.Tensor,
    camera_position: torch.Tensor,
    image_gradient_scale: float | torch.Tensor,
    accumulation_weight_source: torch.Tensor,
    accumulated_weight: torch.Tensor,
    weight_accumulation_multiplier: float,
) -> None:
    """Apply the recovered native gradient kernel and activation backward."""
    _validate_native_gradient_preprocess_tensors(
        raw_scales=raw_scales,
        raw_rotations=raw_rotations,
        positions=positions,
        scale_gradient=scale_gradient,
        rotation_gradient=rotation_gradient,
        point_status=point_status,
        point_indices=point_indices,
        environment_mask=environment_mask,
        camera_position=camera_position,
        accumulation_weight_source=accumulation_weight_source,
        accumulated_weight=accumulated_weight,
    )
    if isinstance(image_gradient_scale, torch.Tensor):
        if image_gradient_scale.shape != ():
            raise ValueError("Native image gradient scale must be scalar.")
        if image_gradient_scale.device != raw_scales.device:
            raise ValueError(
                "Native image gradient scale must share the Gaussian device."
            )
    if weight_accumulation_multiplier <= 0.0:
        raise ValueError(
            "Native weight accumulation multiplier must be positive."
        )
    if point_indices.numel() == 0:
        return

    accumulated_weight.index_add_(
        0,
        point_indices,
        weight_accumulation_multiplier
        * accumulation_weight_source.index_select(0, point_indices),
    )

    selected_raw_scales = raw_scales.index_select(0, point_indices)
    physical_scales = torch.exp(selected_raw_scales)
    selected_scale_gradient = scale_gradient.index_select(0, point_indices)
    selected_environment = environment_mask.index_select(0, point_indices)

    ordinary_local = torch.nonzero(
        ~selected_environment, as_tuple=False
    ).squeeze(1)
    if ordinary_local.numel() != 0:
        ordinary_points = point_indices.index_select(0, ordinary_local)
        ordinary_scales = physical_scales.index_select(0, ordinary_local)
        largest_axis = torch.argmax(ordinary_scales, dim=1)
        status_radius = point_status.index_select(0, ordinary_points)[:, 2]
        regularized = status_radius > VISIBILITY_ADAPTIVE_SCALE_STATUS_MINIMUM
        regularized_local = ordinary_local[regularized]
        regularized_axis = largest_axis[regularized]
        selected_scale_gradient[regularized_local, regularized_axis] += (
            image_gradient_scale
            * (status_radius[regularized] - VISIBILITY_ADAPTIVE_SCALE_STATUS_MINIMUM)
        )

        shrink = status_radius > VISIBILITY_ADAPTIVE_SCALE_SHRINK_STATUS
        shrink_local = ordinary_local[shrink]
        shrink_axis = largest_axis[shrink]
        selected_raw_scales[shrink_local, shrink_axis] = torch.log(
            physical_scales[shrink_local, shrink_axis]
            * VISIBILITY_ADAPTIVE_SCALE_SHRINK_FACTOR
        )

    environment_local = torch.nonzero(
        selected_environment, as_tuple=False
    ).squeeze(1)
    if environment_local.numel() != 0:
        environment_points = point_indices.index_select(0, environment_local)
        environment_scales = physical_scales.index_select(0, environment_local)
        offsets = (
            positions.index_select(0, environment_points) - camera_position
        )
        distances = torch.linalg.vector_norm(offsets, dim=1)
        directions = offsets / distances[:, None]
        raw_environment_rotations = raw_rotations.index_select(
            0, environment_points
        )
        w, x, y, z = raw_environment_rotations.unbind(dim=1)
        rotation_axis = torch.stack(
            (
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * x.square() - 2.0 * y.square(),
            ),
            dim=1,
        )
        axis_error = rotation_axis - directions
        error_x, error_y, error_z = axis_error.unbind(dim=1)
        rotation_delta = torch.stack(
            (
                2.0 * y * error_x - 2.0 * x * error_y,
                2.0 * z * error_x - 2.0 * w * error_y - 4.0 * x * error_z,
                2.0 * w * error_x + 2.0 * z * error_y - 4.0 * y * error_z,
                2.0 * x * error_x + 2.0 * y * error_y,
            ),
            dim=1,
        )
        rotation_gradient.index_add_(
            0,
            environment_points,
            VISIBILITY_ADAPTIVE_ENVIRONMENT_ROTATION_GRADIENT * rotation_delta,
        )

        environment_scale_delta = torch.stack(
            (
                VISIBILITY_ADAPTIVE_ENVIRONMENT_RADIAL_GRADIENT
                * environment_scales[:, 0]
                / distances,
                VISIBILITY_ADAPTIVE_ENVIRONMENT_RADIAL_GRADIENT
                * environment_scales[:, 1]
                / distances,
                VISIBILITY_ADAPTIVE_ENVIRONMENT_AXIAL_GRADIENT
                * (
                    2.0 * environment_scales[:, 2]
                    - VISIBILITY_ADAPTIVE_ENVIRONMENT_AXIAL_RATIO
                    * (environment_scales[:, 0] + environment_scales[:, 1])
                ),
            ),
            dim=1,
        )
        selected_scale_gradient[environment_local] += environment_scale_delta

        scale_limit = VISIBILITY_ADAPTIVE_ENVIRONMENT_SCALE_RATIO_LIMIT * distances
        radial_scales = environment_scales[:, :2]
        clamp_radial = (
            radial_scales / distances[:, None]
            > VISIBILITY_ADAPTIVE_ENVIRONMENT_SCALE_RATIO_LIMIT
        )
        environment_raw_scales = selected_raw_scales.index_select(
            0, environment_local
        )
        environment_raw_scales[:, :2] = torch.where(
            clamp_radial,
            torch.log(scale_limit[:, None]),
            environment_raw_scales[:, :2],
        )
        selected_raw_scales.index_copy_(
            0, environment_local, environment_raw_scales
        )

    raw_scales.index_copy_(0, point_indices, selected_raw_scales)
    scale_gradient.index_copy_(
        0,
        point_indices,
        selected_scale_gradient,
    )


@torch.no_grad()
def visibility_adaptive_projected_radius_pixels(
    *,
    projected_conic_opacity: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Recover the longest projected conic radius in pixels."""
    point_count = projected_conic_opacity.shape[0]
    if projected_conic_opacity.shape != (point_count, 4):
        raise ValueError(
            "Native projected conic and opacity must have shape (N, 4)."
        )
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Native point status requires a positive viewport.")

    conic_a = projected_conic_opacity[:, 0]
    conic_b = projected_conic_opacity[:, 1]
    conic_c = projected_conic_opacity[:, 2]
    opacity = projected_conic_opacity[:, 3]
    discriminant = torch.sqrt(
        (conic_a - conic_c).square() + 4.0 * conic_b.square()
    )
    smallest_eigenvalue = 0.5 * (conic_a + conic_c - discriminant)
    alpha_ratio = VISIBILITY_ADAPTIVE_ALPHA_LEVELS * opacity
    valid = (smallest_eigenvalue > 0.0) & (alpha_ratio > 1.0)
    safe_eigenvalue = torch.where(
        valid,
        smallest_eigenvalue,
        torch.ones_like(smallest_eigenvalue),
    )
    safe_log_ratio = torch.where(
        valid,
        torch.log(alpha_ratio),
        torch.zeros_like(alpha_ratio),
    )
    radius_pixels = torch.sqrt(2.0 * safe_log_ratio / safe_eigenvalue)
    return torch.where(
        valid,
        radius_pixels,
        torch.zeros_like(radius_pixels),
    )


@torch.no_grad()
def visibility_adaptive_normalized_radius(
    *,
    projected_conic_opacity: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Recover the native longest projected radius in viewport units."""
    radius_pixels = visibility_adaptive_projected_radius_pixels(
        projected_conic_opacity=projected_conic_opacity,
        image_width=image_width,
        image_height=image_height,
    )
    return radius_pixels / max(image_width, image_height)


@torch.no_grad()
def visibility_adaptive_densify_mask(
    *,
    status_grad: torch.Tensor,
    status_size: torch.Tensor,
    accumulated_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact native CUB densification selection predicate."""
    if not (
        status_grad.shape == status_size.shape == accumulated_weight.shape
    ):
        msg = "Native densification status arrays must match."
        raise ValueError(msg)
    return (
        (status_grad > VISIBILITY_ADAPTIVE_DENSIFY_GRAD_THRESHOLD)
        & (status_size > VISIBILITY_ADAPTIVE_DENSIFY_SIZE_THRESHOLD)
        & (accumulated_weight < VISIBILITY_ADAPTIVE_DENSIFY_WEIGHT_THRESHOLD)
    )


@torch.no_grad()
def visibility_adaptive_compact_point_weights(
    *,
    numerator: torch.Tensor,
    transmittance: torch.Tensor,
    reference: torch.Tensor,
    multiplier: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Return native projected-pixel consistency weights."""
    if not (
        numerator.shape
        == transmittance.shape
        == reference.shape
        == multiplier.shape
    ):
        raise ValueError("Native compact-weight inputs must match.")
    if sigma <= 0.0:
        raise ValueError("Native compact-weight sigma must be positive.")

    derived = numerator / (1.0 - transmittance + VISIBILITY_ADAPTIVE_FLOAT_EPSILON)
    valid = (derived > VISIBILITY_ADAPTIVE_COMPACT_WEIGHT_MINIMUM) & (
        reference > VISIBILITY_ADAPTIVE_COMPACT_WEIGHT_MINIMUM
    )
    relative_difference = (derived - reference) / derived
    exponent = -0.5 * (relative_difference / sigma).square()
    weights = multiplier * torch.exp(exponent)
    return torch.where(valid, weights, torch.zeros_like(weights))


@torch.no_grad()
def visibility_adaptive_split_opacity(
    *, physical_opacity: torch.Tensor
) -> torch.Tensor:
    """Return both child opacities from the native volume split."""
    point_count = physical_opacity.shape[0]
    if physical_opacity.shape != (point_count, 1):
        msg = "Native split opacity must have shape (N, 1)."
        raise ValueError(msg)
    if torch.any((physical_opacity < 0.0) | (physical_opacity > 1.0)):
        msg = "Native split physical opacity must be in [0, 1]."
        raise ValueError(msg)

    epsilon = VISIBILITY_ADAPTIVE_FLOAT_EPSILON
    retained_transmittance = torch.clamp(
        1.0 - 0.5 * physical_opacity - epsilon,
        min=0.0,
    )
    child_opacity = (
        physical_opacity
        * (1.0 - torch.pow(retained_transmittance, 1.0 / 3.0))
        / (0.5 * physical_opacity + epsilon)
    )
    return child_opacity[:, None, :].expand(-1, 2, -1).reshape(-1, 1)


@torch.no_grad()
def visibility_adaptive_split_children(
    *,
    positions: torch.Tensor,
    physical_scales: torch.Tensor,
    rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create the two deterministic children appended for each parent."""
    point_count = positions.shape[0]
    if positions.shape != (point_count, 3):
        msg = "Native split positions must have shape (N, 3)."
        raise ValueError(msg)
    if physical_scales.shape != (point_count, 3):
        msg = "Native split scales must have shape (N, 3)."
        raise ValueError(msg)
    if rotations.shape != (point_count, 4):
        msg = "Native split rotations must have shape (N, 4)."
        raise ValueError(msg)
    if torch.any(physical_scales <= 0.0):
        msg = "Native split physical scales must be positive."
        raise ValueError(msg)

    largest_axis = torch.argmax(physical_scales, dim=1)
    local_axes = torch.nn.functional.one_hot(largest_axis, num_classes=3).to(
        dtype=positions.dtype
    )
    rotation_matrices = quaternion_to_so3(rotations)
    world_axes = torch.bmm(
        rotation_matrices, local_axes.unsqueeze(-1)
    ).squeeze(-1)
    max_scale = physical_scales.gather(1, largest_axis[:, None])
    offsets = world_axes * max_scale * VISIBILITY_ADAPTIVE_CHILD_OFFSET
    child_positions = torch.stack(
        (positions + offsets, positions - offsets), dim=1
    ).reshape(-1, 3)
    child_scales = (
        physical_scales[:, None, :].expand(-1, 2, -1).reshape(-1, 3)
        * VISIBILITY_ADAPTIVE_CHILD_SCALE
    )
    child_rotations = rotations[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
    return child_positions, child_scales, child_rotations


class VisibilityAdaptiveStrategy(BaseStrategy):
    """Recovered visibility-adaptive status, deterministic split, and opacity prune flow."""

    def __init__(self, config, model) -> None:
        super().__init__(config=config, model=model)
        if self.conf.model.density_activation != "sigmoid":
            raise ValueError(
                "Visibility-adaptive strategy requires sigmoid density activation."
            )
        if self.conf.model.scale_activation != "exp":
            raise ValueError(
                "Visibility-adaptive strategy requires exp scale activation."
            )
        if not bool(
            self.conf.loss.get(
                "use_fixed_image_loss_denominator",
                False,
            )
        ):
            raise ValueError(
                "Visibility-adaptive strategy requires its RGB loss denominator."
            )
        if not bool(self.conf.loss.use_camera_loss_weights):
            raise ValueError(
                "Visibility-adaptive strategy requires source-camera loss weights."
            )
        if not bool(
            self.conf.loss.get(
                "camera_loss_weights_use_physical_camera",
                False,
            )
        ):
            raise ValueError(
                "Visibility-adaptive strategy requires physical-camera loss weights."
            )
        self.point_status = torch.empty((0, 4))
        self.accumulated_weight = torch.empty((0, 1))
        self.coherence_pass_count = 0

    def _optimizer_visibility(
        self,
        *,
        batch,
        renderer_visibility: torch.Tensor,
        forward_visibility: torch.Tensor | None,
        projected_position: torch.Tensor,
        rendered_weight: torch.Tensor,
    ) -> torch.Tensor:
        source = str(
            self.conf.strategy.get(
                "optimizer_visibility_source",
                VISIBILITY_ADAPTIVE_RENDERER_VISIBILITY_SOURCE,
            )
        )
        if source == VISIBILITY_ADAPTIVE_RENDERER_VISIBILITY_SOURCE:
            return renderer_visibility
        if source == VISIBILITY_ADAPTIVE_FORWARD_VISIBILITY_SOURCE:
            if forward_visibility is None:
                raise RuntimeError(
                    "Forward visibility requires a renderer forward-visibility "
                    "output."
                )
            if forward_visibility.shape != renderer_visibility.shape:
                raise ValueError("Forward visibility has the wrong shape.")
            return renderer_visibility & forward_visibility
        if source == VISIBILITY_ADAPTIVE_MASK_CENTER_WEIGHT_VISIBILITY_SOURCE:
            return self._mask_center_weight_visibility(
                batch=batch,
                renderer_visibility=renderer_visibility,
                projected_position=projected_position,
                rendered_weight=rendered_weight,
            )
        if source != VISIBILITY_ADAPTIVE_COLOR_GRADIENT_WEIGHT_VISIBILITY_SOURCE:
            raise ValueError(
                "Unsupported native optimizer visibility source: "
                f"{source!r}."
            )
        color_gradient = self.model.features_albedo.grad
        if color_gradient is None:
            raise RuntimeError(
                "Color-gradient optimizer visibility requires albedo "
                "gradients."
            )
        min_rendered_weight = float(
            self.conf.strategy.get(
                "optimizer_visibility_min_rendered_weight",
                0.0,
            )
        )
        min_color_gradient_norm = float(
            self.conf.strategy.get(
                "optimizer_visibility_min_color_gradient_norm",
                0.0,
            )
        )
        color_gradient_norm = color_gradient.detach().norm(dim=1)
        contribution_visibility = (
            rendered_weight.reshape(-1) > min_rendered_weight
        ) & (color_gradient_norm > min_color_gradient_norm)
        return renderer_visibility & contribution_visibility

    def _mask_center_weight_visibility(
        self,
        *,
        batch,
        renderer_visibility: torch.Tensor,
        projected_position: torch.Tensor,
        rendered_weight: torch.Tensor,
    ) -> torch.Tensor:
        mask = getattr(batch, "mask", None)
        if mask is None:
            raise RuntimeError(
                "mask_center_weight visibility requires a training mask."
            )
        if mask.ndim == 4:
            image_mask = mask[0]
        elif mask.ndim in (2, 3):
            image_mask = mask
        else:
            raise ValueError("Training mask must be HxW, HxWx1, or 1xHxWx1.")
        if image_mask.ndim == 3:
            if image_mask.shape[-1] == 1:
                image_mask = image_mask[..., 0]
            elif image_mask.shape[0] == 1:
                image_mask = image_mask[0]
            else:
                raise ValueError(
                    "Training mask must be HxW, HxWx1, or 1xHxW."
                )
        image_mask = image_mask.to(
            device=projected_position.device,
            dtype=torch.bool,
        )
        height = int(image_mask.shape[0])
        width = int(image_mask.shape[1])
        xy = torch.round(projected_position).to(dtype=torch.long)
        x = xy[:, 0]
        y = xy[:, 1]
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        center_visible = torch.zeros_like(renderer_visibility)
        center_visible[inside] = image_mask[y[inside], x[inside]]
        min_rendered_weight = float(
            self.conf.strategy.get(
                "optimizer_visibility_min_rendered_weight",
                0.0,
            )
        )
        contribution_visible = rendered_weight.reshape(-1) > min_rendered_weight
        return renderer_visibility & center_visible & contribution_visible

    def _validate_environment_mask(self, *, initialize_if_empty: bool) -> None:
        mask = getattr(self.model, "environment_mask", None)
        if not isinstance(mask, torch.Tensor):
            raise ValueError(
                "Visibility-adaptive model is missing its environment mask."
            )
        point_count = self.model.num_gaussians
        if mask.numel() == 0 and initialize_if_empty:
            mask = torch.zeros(
                (point_count,),
                dtype=torch.bool,
                device=self.model.device,
            )
            self.model.environment_mask = mask
        if mask.dtype != torch.bool:
            raise ValueError("Native environment mask must use bool dtype.")
        if mask.shape != (point_count,):
            raise ValueError(
                "Native environment mask must have one value per Gaussian."
            )
        if mask.device != self.model.positions.device:
            raise ValueError(
                "Native environment mask must share the Gaussian device."
            )

    def init_densification_buffer(self, checkpoint=None) -> None:
        point_count = self.model.num_gaussians
        self._validate_environment_mask(
            initialize_if_empty=checkpoint is None,
        )
        if checkpoint is None:
            self.point_status = torch.zeros(
                (point_count, 4),
                dtype=torch.float32,
                device=self.model.device,
            )
            self.accumulated_weight = torch.zeros(
                (point_count, 1),
                dtype=torch.float32,
                device=self.model.device,
            )
            self.coherence_pass_count = 0
            return

        self.point_status = checkpoint["visibility_adaptive_point_status"][
            0
        ].detach()
        self.accumulated_weight = checkpoint[
            "visibility_adaptive_accumulated_weight"
        ][0].detach()
        coherence_pass_count = checkpoint[
            "visibility_adaptive_coherence_pass_count"
        ][0].detach()
        if self.point_status.shape != (point_count, 4):
            raise ValueError(
                "Visibility-adaptive checkpoint point status has the wrong shape."
            )
        if self.accumulated_weight.shape != (point_count, 1):
            raise ValueError(
                "Visibility-adaptive checkpoint accumulated weight has the wrong "
                "shape."
            )
        if (
            coherence_pass_count.shape != (1,)
            or coherence_pass_count.dtype != torch.int64
        ):
            raise ValueError(
                "Visibility-adaptive checkpoint coherence pass count is invalid."
            )
        self.coherence_pass_count = int(coherence_pass_count.item())
        if self.coherence_pass_count < 0:
            raise ValueError(
                "Visibility-adaptive checkpoint coherence pass count is negative."
            )

    def get_strategy_parameters(self) -> dict:
        return {
            "visibility_adaptive_point_status": (self.point_status,),
            "visibility_adaptive_accumulated_weight": (self.accumulated_weight,),
            "visibility_adaptive_coherence_pass_count": (
                torch.tensor(
                    (self.coherence_pass_count,),
                    dtype=torch.int64,
                    device=self.point_status.device,
                ),
            ),
        }

    @torch.no_grad()
    def _post_backward(
        self,
        step: int,
        scene_extent: float,
        train_dataset,
        batch=None,
        writer=None,
        outputs=None,
    ) -> bool:
        if outputs is None:
            raise RuntimeError(
                "Visibility-adaptive strategy requires renderer outputs."
            )
        if batch is None:
            raise RuntimeError("Visibility-adaptive strategy requires a batch.")
        required = (
            "mog_visibility",
            "mog_projected_conic_opacity",
            "mog_projected_extent",
            "mog_projected_position",
            "mog_projected_position_gradient",
            "mog_accumulated_weight",
            "pred_rgb",
        )
        missing = [name for name in required if name not in outputs]
        if missing:
            raise RuntimeError(
                "Visibility-adaptive strategy requires renderer outputs: "
                + ", ".join(missing)
            )

        point_count = self.model.num_gaussians
        projected_conic_opacity = outputs["mog_projected_conic_opacity"]
        projected_extent = outputs["mog_projected_extent"]
        projected_position = outputs["mog_projected_position"]
        projected_position_gradient = outputs[
            "mog_projected_position_gradient"
        ]
        rendered_weight = outputs["mog_accumulated_weight"].reshape(-1, 1)
        forward_visibility = outputs.get("mog_forward_visibility")
        if forward_visibility is not None:
            if not torch.is_tensor(forward_visibility):
                raise TypeError("Native forward visibility must be a tensor.")
            forward_visibility = forward_visibility.reshape(-1).to(torch.bool)
        visibility = self._optimizer_visibility(
            batch=batch,
            renderer_visibility=outputs["mog_visibility"]
            .reshape(-1)
            .to(torch.bool),
            forward_visibility=forward_visibility,
            projected_position=projected_position,
            rendered_weight=rendered_weight,
        )
        outputs["mog_visibility"] = visibility.reshape_as(
            outputs["mog_visibility"]
        )
        if visibility.shape != (point_count,):
            raise ValueError("Native visibility has the wrong shape.")
        if projected_conic_opacity.shape != (point_count, 4):
            raise ValueError(
                "Native projected conic and opacity have the wrong shape."
            )
        if projected_extent.shape != (point_count, 2):
            raise ValueError("Native projected extent has the wrong shape.")
        if projected_position.shape != (point_count, 2):
            raise ValueError("Native projected positions have the wrong shape.")
        if projected_position_gradient.shape != (point_count, 2):
            raise ValueError(
                "Native projected position gradients have the wrong shape."
            )
        if rendered_weight.shape != (point_count, 1):
            raise ValueError("Native accumulated weight has the wrong shape.")
        scale_gradient = self.model.scale.grad
        if scale_gradient is None:
            raise RuntimeError(
                "Visibility-adaptive strategy requires scale gradients."
            )
        rotation_gradient = self.model.rotation.grad
        if rotation_gradient is None:
            raise RuntimeError(
                "Visibility-adaptive strategy requires rotation gradients."
            )
        position_gradient = self.model.positions.grad
        if position_gradient is None:
            raise RuntimeError(
                "Visibility-adaptive strategy requires position gradients."
            )
        sensor_position = batch.T_to_world[0, :3, 3]
        image_height = outputs["pred_rgb"].shape[1]
        image_width = outputs["pred_rgb"].shape[2]
        projected_size = visibility_adaptive_projected_size_pixels(
            rendered_weight=rendered_weight,
            projected_extent=projected_extent,
        projected_conic_opacity=projected_conic_opacity,
        image_width=image_width,
        image_height=image_height,
        source=str(
            self.conf.strategy.get(
                "projected_size_source",
                VISIBILITY_ADAPTIVE_PROJECTED_SIZE_SOURCE_RENDERED_WEIGHT,
            )
        ),
        multiplier=float(
            self.conf.strategy.get(
                "projected_size_multiplier",
                VISIBILITY_ADAPTIVE_PROJECTED_SIZE_MULTIPLIER,
            )
        ),
    )
        renderer_projected_gradient_pixels = outputs.get(
            "mog_projected_gradient_pixels"
        )
        if renderer_projected_gradient_pixels is None:
            focal_length = _visibility_adaptive_batch_focal_length(batch)
            projected_gradient_pixels = visibility_adaptive_projected_gradient_pixels(
                projected_position_gradient=projected_position_gradient,
                image_width=image_width,
                image_height=image_height,
                position_gradient=position_gradient,
                positions=self.model.positions,
                camera_to_world=batch.T_to_world[0],
                focal_length=focal_length,
                proxy_mode=str(
                    self.conf.strategy.get(
                        "projected_gradient_proxy_mode",
                        VISIBILITY_ADAPTIVE_PROJECTED_GRADIENT_PROXY_JACOBIAN,
                    )
                ),
                proxy_scale=float(
                    self.conf.strategy.get(
                        "projected_gradient_proxy_scale",
                        VISIBILITY_ADAPTIVE_TANGENT_PROJECTED_GRADIENT_SCALE,
                    )
                ),
            )
        else:
            if not torch.is_tensor(renderer_projected_gradient_pixels):
                raise TypeError(
                    "Native renderer projected pixel gradients must be a "
                    "tensor."
                )
            if renderer_projected_gradient_pixels.shape != (point_count,):
                raise ValueError(
                    "Native renderer projected pixel gradients must have "
                    "shape (N,)."
                )
            projected_gradient_pixels = renderer_projected_gradient_pixels
        normalized_radius = visibility_adaptive_normalized_radius(
            projected_conic_opacity=projected_conic_opacity,
            image_width=image_width,
            image_height=image_height,
        )
        status_gradient_scale = float(
            self.conf.strategy.get(
                "status_gradient_scale",
                VISIBILITY_ADAPTIVE_STATUS_GRADIENT_SCALE,
            )
        )
        if (
            not math.isfinite(status_gradient_scale)
            or status_gradient_scale <= 0.0
        ):
            raise ValueError(
                "Visibility-adaptive status gradient scale must be finite and "
                "positive."
            )
        point_indices = torch.nonzero(visibility, as_tuple=False).squeeze(1)
        measurements = torch.stack(
            (
                normalized_radius.index_select(0, point_indices),
                status_gradient_scale
                * projected_gradient_pixels.index_select(0, point_indices),
            ),
            dim=1,
        )
        self.point_status = visibility_adaptive_update_point_status(
            status=self.point_status,
            point_indices=point_indices,
            projected_size_pixels=projected_size,
            visible_measurements=measurements,
            window_size=VISIBILITY_ADAPTIVE_STATUS_WINDOW_SIZE,
        )
        camera_loss_multiplier = indexed_camera_loss_weight(
            camera_index=int(batch.post_processing_camera_idx),
            configured_weights=list(self.conf.loss.camera_loss_weights),
        )
        image_gradient_scale = camera_loss_multiplier / (
            fixed_image_loss_denominator(
                rgb=outputs["pred_rgb"],
                mask=batch.mask,
                image_scale=getattr(batch, "native_image_scale", 1.0),
                min_valid_fraction=float(
                    self.conf.loss.get(
                        "fixed_image_loss_min_valid_fraction",
                        0.8,
                    )
                ),
            )
        )
        visibility_adaptive_preprocess_gradients(
            raw_scales=self.model.scale,
            raw_rotations=self.model.rotation,
            positions=self.model.positions,
            scale_gradient=scale_gradient,
            rotation_gradient=rotation_gradient,
            point_status=self.point_status,
            point_indices=point_indices,
            environment_mask=self.model.environment_mask,
            camera_position=sensor_position,
            image_gradient_scale=image_gradient_scale,
            accumulation_weight_source=projected_size[:, None],
            accumulated_weight=self.accumulated_weight,
            weight_accumulation_multiplier=float(
                self.conf.strategy.get(
                    "weight_accumulation_multiplier",
                    VISIBILITY_ADAPTIVE_WEIGHT_ACCUMULATION_MULTIPLIER,
                )
            ),
        )
        return False

    @torch.no_grad()
    def _post_optimizer_step(
        self,
        step: int,
        scene_extent: float,
        train_dataset,
        batch=None,
        writer=None,
    ) -> bool:
        topology_total_iterations = int(
            self.conf.strategy.get("topology_total_iterations", 0)
        )
        if topology_total_iterations <= 0:
            topology_total_iterations = int(self.conf.n_iterations)
        configured_batch_size = int(
            self.conf.strategy.get("topology_batch_size", 0)
        )
        if configured_batch_size < 0:
            raise ValueError(
                "Native topology batch size must be nonnegative."
            )
        topology_batch_size = (
            configured_batch_size
            if configured_batch_size > 0
            else len(train_dataset)
        )
        regular_prune, densify, iterative_prune = (
            visibility_adaptive_topology_schedule(
                step=step,
                total_iterations=topology_total_iterations,
                iteration_per_batch=topology_batch_size,
                prune_interval=int(self.conf.strategy.prune.frequency),
                densify_interval=int(self.conf.strategy.densify.frequency),
            )
        )
        changed = False
        if iterative_prune:
            changed = self._iterative_prune() or changed
        if regular_prune:
            changed = self._scheduled_prune() or changed
        if densify:
            changed = self._densify() or changed
        if step % VISIBILITY_ADAPTIVE_COHERENCE_INTERVAL == 0:
            changed = self._coherence() or changed
        return changed

    def _require_native_optimizer(self) -> None:
        if not isinstance(self.model.optimizer, VisibilitySelectiveAdam):
            raise RuntimeError(
                "Visibility-adaptive topology requires VisibilitySelectiveAdam; ordinary "
                "Adam state cannot preserve the recovered semantics."
            )

    @staticmethod
    def _repeat_selected(
        tensor: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        selected = tensor[mask]
        return (
            selected[:, None]
            .expand(
                -1,
                2,
                *tensor.shape[1:],
            )
            .reshape(
                selected.shape[0] * 2,
                *tensor.shape[1:],
            )
        )

    @torch.no_grad()
    def _densify(self) -> bool:
        self._require_native_optimizer()
        mask = visibility_adaptive_densify_mask(
            status_grad=self.point_status[:, 3],
            status_size=self.point_status[:, 1],
            accumulated_weight=self.accumulated_weight[:, 0],
        )
        selected = torch.nonzero(mask, as_tuple=False).squeeze(1)
        selected_count = int(selected.numel())
        if selected_count == 0:
            return False
        projected_count = mask.shape[0] + 2 * selected_count
        if projected_count > int(self.conf.strategy.max_gaussians):
            if self.conf.strategy.print_stats:
                logger.info(
                    "Visibility-adaptive skipped densification because "
                    f"count={projected_count} exceeds capacity="
                    f"{int(self.conf.strategy.max_gaussians)}."
                )
            return False

        child_positions, child_scales, _ = visibility_adaptive_split_children(
            positions=self.model.positions[mask],
            physical_scales=self.model.get_scale()[mask],
            rotations=self.model.get_rotation()[mask],
        )
        child_rotations = self._repeat_selected(self.model.rotation, mask)
        child_density = self.model.density_activation_inv(
            visibility_adaptive_split_opacity(
                physical_opacity=self.model.get_density()[mask]
            )
        )
        child_raw_scales = self.model.scale_activation_inv(child_scales)
        parent_density = child_density[::2]
        parent_raw_scales = child_raw_scales[::2]

        def update_parameter(
            name: str, parameter: torch.Tensor
        ) -> torch.Tensor:
            parent_values = parameter
            if name == "positions":
                children = child_positions
            elif name == "density":
                parent_values = parameter.clone()
                parent_values[mask] = parent_density
                children = child_density
            elif name == "scale":
                parent_values = parameter.clone()
                parent_values[mask] = parent_raw_scales
                children = child_raw_scales
            elif name == "rotation":
                children = child_rotations
            else:
                children = self._repeat_selected(parameter, mask)
            return torch.nn.Parameter(
                torch.cat((parent_values, children)),
                requires_grad=parameter.requires_grad,
            )

        def update_optimizer(key: str, value: torch.Tensor) -> torch.Tensor:
            if key == "gaussian_steps":
                children = self._repeat_selected(value, mask)
            else:
                child_count = int(mask.sum()) * 2
                children = torch.zeros(
                    (child_count, *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
            return torch.cat((value, children))

        self._update_param_with_optimizer(
            update_parameter,
            update_optimizer,
        )
        child_environment = self._repeat_selected(
            self.model.environment_mask,
            mask,
        )
        self.model.environment_mask = torch.cat(
            (self.model.environment_mask, child_environment)
        )
        child_count = selected_count * 2
        parent_status = self.point_status.clone()
        parent_status[mask] = 0.0
        child_status = torch.zeros(
            (child_count, 4),
            dtype=self.point_status.dtype,
            device=self.point_status.device,
        )
        self.point_status = torch.cat((parent_status, child_status))
        self.accumulated_weight = torch.cat(
            (
                self.accumulated_weight,
                self._repeat_selected(self.accumulated_weight, mask),
            )
        )
        if self.conf.strategy.print_stats:
            logger.info(
                f"Visibility-adaptive split {selected_count} Gaussians; "
                f"count={self.model.num_gaussians}."
            )
        return True

    @torch.no_grad()
    def _prune(self) -> bool:
        self._require_native_optimizer()
        threshold = float(self.conf.strategy.prune.density_threshold)
        keep = self.model.get_density().reshape(-1) >= threshold
        return self._compact(keep)

    @torch.no_grad()
    def _scheduled_prune(self) -> bool:
        self._require_native_optimizer()
        opacity_threshold = float(self.conf.strategy.prune.density_threshold)
        keep = visibility_adaptive_regular_prune_keep_mask(
            physical_opacity=self.model.get_density().reshape(-1),
            status_radius=self.point_status[:, 2],
            opacity_threshold=opacity_threshold,
            radius_threshold=VISIBILITY_ADAPTIVE_REGULAR_PRUNE_RADIUS_THRESHOLD,
        )
        return self._compact(keep)

    @torch.no_grad()
    def _iterative_prune(self) -> bool:
        self._require_native_optimizer()
        weights = self.accumulated_weight[:, 0]
        pivot_index = int(
            weights.shape[0] * VISIBILITY_ADAPTIVE_ITERATIVE_PRUNE_FRACTION
        )
        sorted_weights = torch.sort(weights).values
        pivot = torch.fmin(
            sorted_weights[pivot_index],
            torch.tensor(
                VISIBILITY_ADAPTIVE_ITERATIVE_PRUNE_WEIGHT_THRESHOLD,
                dtype=weights.dtype,
                device=weights.device,
            ),
        )
        keep = weights > pivot
        changed = self._compact(keep)
        self.accumulated_weight.zero_()
        return changed

    @torch.no_grad()
    def _coherence(self) -> bool:
        self._require_native_optimizer()
        permutation = visibility_adaptive_coherence_permutation(
            positions=self.model.positions,
            scale=VISIBILITY_ADAPTIVE_COHERENCE_SCALE,
            include_prefix=bool(self.coherence_pass_count & 1),
        )
        self.coherence_pass_count += 1
        expected = torch.arange(
            permutation.shape[0],
            dtype=permutation.dtype,
            device=permutation.device,
        )
        if torch.equal(permutation, expected):
            return False

        def update_parameter(
            name: str, parameter: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.Parameter(
                parameter[permutation],
                requires_grad=parameter.requires_grad,
            )

        def update_optimizer(key: str, value: torch.Tensor) -> torch.Tensor:
            return value[permutation]

        self._update_param_with_optimizer(
            update_parameter,
            update_optimizer,
        )
        self.model.environment_mask = (
            self.model.environment_mask[permutation]
        )
        self.point_status = self.point_status[permutation]
        self.accumulated_weight = self.accumulated_weight[permutation]
        return True

    @torch.no_grad()
    def _finalize_training(self) -> bool:
        return self._prune()

    @torch.no_grad()
    def _compact(self, keep: torch.Tensor) -> bool:
        if torch.all(keep):
            return False
        if not torch.any(keep):
            raise RuntimeError(
                "Visibility-adaptive prune would remove every Gaussian."
            )

        def update_parameter(
            name: str, parameter: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.Parameter(
                parameter[keep],
                requires_grad=parameter.requires_grad,
            )

        def update_optimizer(key: str, value: torch.Tensor) -> torch.Tensor:
            return value[keep]

        self._update_param_with_optimizer(
            update_parameter,
            update_optimizer,
        )
        pruned = int((~keep).sum())
        self.model.environment_mask = (
            self.model.environment_mask[keep]
        )
        self.point_status = self.point_status[keep]
        self.accumulated_weight = self.accumulated_weight[keep]
        if self.conf.strategy.print_stats:
            logger.info(
                f"Visibility-adaptive pruned {pruned} Gaussians; "
                f"count={self.model.num_gaussians}."
            )
        return True
