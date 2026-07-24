"""Offline replay of native per-image projection distortion."""

import torch
import torch.nn.functional as functional


def sample_native_distortion(
    distortion_map: torch.Tensor,
    pixel_coords: torch.Tensor,
    *,
    resolution: tuple[int, int],
) -> torch.Tensor:
    """Bilinearly sample a native 12x12 normalized-offset field."""
    if distortion_map.ndim != 3 or distortion_map.shape[-1] != 2:
        raise ValueError("distortion_map must have shape [H, W, 2].")
    if pixel_coords.ndim != 4 or pixel_coords.shape[-1] != 2:
        raise ValueError("pixel_coords must have shape [B, H, W, 2].")
    width, height = resolution
    if width <= 0 or height <= 0:
        raise ValueError("resolution must be positive.")

    normalized = torch.empty_like(pixel_coords)
    normalized[..., 0] = 2.0 * pixel_coords[..., 0] / width - 1.0
    normalized[..., 1] = 2.0 * pixel_coords[..., 1] / height - 1.0
    field = distortion_map.permute(2, 0, 1).unsqueeze(0)
    if pixel_coords.shape[0] != 1:
        field = field.expand(pixel_coords.shape[0], -1, -1, -1)
    sampled = functional.grid_sample(
        field,
        normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.permute(0, 2, 3, 1)


def solve_native_base_pixels(
    distortion_map: torch.Tensor,
    target_pixel_coords: torch.Tensor,
    *,
    focal_length: torch.Tensor,
    resolution: tuple[int, int],
    sign: float,
    iterations: int,
) -> torch.Tensor:
    """Invert ``target = base + sign * focal * D(base)``."""
    if sign not in (-1.0, 1.0):
        raise ValueError("native distortion sign must be -1 or 1.")
    if iterations < 1:
        raise ValueError("native distortion iterations must be positive.")
    focal = focal_length.to(
        device=target_pixel_coords.device,
        dtype=target_pixel_coords.dtype,
    ).reshape(1, 1, 1, 2)
    base = target_pixel_coords
    for _ in range(iterations):
        offsets = sample_native_distortion(
            distortion_map,
            base,
            resolution=resolution,
        )
        base = target_pixel_coords - sign * focal * offsets
    if not torch.isfinite(base).all():
        raise ValueError("native distortion inverse produced non-finite pixels.")
    return base


def warp_native_render(
    pred_rgb: torch.Tensor,
    pixel_coords: torch.Tensor,
    distortion_map: torch.Tensor,
    *,
    focal_length: torch.Tensor,
    sign: float,
    iterations: int,
) -> torch.Tensor:
    """Approximate native projection replay by inverse-warping a render."""
    if pred_rgb.ndim != 4 or pred_rgb.shape[-1] != 3:
        raise ValueError("pred_rgb must have shape [B, H, W, 3].")
    height, width = pred_rgb.shape[1:3]
    if pixel_coords.shape[:3] != pred_rgb.shape[:3]:
        raise ValueError("pixel_coords and pred_rgb resolutions must match.")
    base = solve_native_base_pixels(
        distortion_map,
        pixel_coords,
        focal_length=focal_length,
        resolution=(width, height),
        sign=sign,
        iterations=iterations,
    )
    sample_grid = torch.empty_like(base)
    sample_grid[..., 0] = 2.0 * base[..., 0] / width - 1.0
    sample_grid[..., 1] = 2.0 * base[..., 1] / height - 1.0
    warped = functional.grid_sample(
        pred_rgb.permute(0, 3, 1, 2),
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped.permute(0, 2, 3, 1)
