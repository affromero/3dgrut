"""MV-DINO feature losses for native fisheye rim supervision."""

import importlib
import math
import os
from functools import lru_cache
from typing import Protocol, cast

import torch
import torch.nn.functional as F
import yaml
from beartype import beartype
from jaxtyping import Float, jaxtyped
from klogr.path import path_exists, path_join, path_open

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_DINO_INPUT_SIZE = 518


class _SafeConfig:
    """Attribute config object for MVFM model construction."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    def __getattr__(self, name: str) -> object:
        """MVFM configs probe optional flags through namespace attributes."""
        return False


class _ModelFactory(Protocol):
    def __call__(self, args: _SafeConfig) -> torch.nn.Module:
        """Build the MV-DINO model."""


class _CheckpointLoader(Protocol):
    def load_model(
        self,
        model: torch.nn.Module,
        *,
        latest: bool = False,
    ) -> torch.nn.Module:
        """Load weights into the MV-DINO model."""


class _CheckpointFactory(Protocol):
    def __call__(self, checkpoint_dir: str) -> _CheckpointLoader:
        """Build the MV-DINO checkpoint loader."""


def _resolve_exp_dir(exp_dir: str | None) -> str:
    """Resolve the directory containing ``<model_name>/args.yaml``."""
    if exp_dir:
        return exp_dir
    env_exp_dir = os.environ.get("HAX_MVDINO_EXP_DIR")
    if env_exp_dir:
        return env_exp_dir
    mvfm_root = os.environ.get("HAX_MVFM_ROOT")
    if mvfm_root:
        return path_join(mvfm_root, "experiments")
    msg = (
        "MV-DINO rim loss needs loss.mvdino_exp_dir, HAX_MVDINO_EXP_DIR, "
        "or HAX_MVFM_ROOT. Expected a directory containing "
        "<model_name>/args.yaml and <model_name>/checkpoints/best.pth."
    )
    raise RuntimeError(msg)


@lru_cache(maxsize=4)
def _load_mvdino(
    exp_dir: str | None,
    name: str,
    device: str,
) -> torch.nn.Module:
    """Load a frozen MV-DINO model from a local pretrained snapshot."""
    importlib.import_module("kornia_dedode_shim")
    checkpointing = importlib.import_module("dino3d.checkpointing")
    model_utils = importlib.import_module("dino3d.models.utils.utils")

    resolved_exp_dir = _resolve_exp_dir(exp_dir)
    model_root = path_join(resolved_exp_dir, name)
    args_path = path_join(model_root, "args.yaml")
    if not path_exists(args_path):
        msg = (
            f"MV-DINO args file not found at {args_path}. "
            "Download the pretrained snapshot with "
            "`huggingface_hub.snapshot_download(repo_id='Leoseg/dinov2_reg', "
            "local_dir=<exp_dir>)`."
        )
        raise FileNotFoundError(msg)

    with path_open(args_path, "r") as handle:
        raw_args = yaml.safe_load(handle)
    if not isinstance(raw_args, dict):
        msg = f"MV-DINO args file at {args_path} must contain a mapping."
        raise TypeError(msg)
    config_kwargs: dict[str, object] = {}
    for key, value in raw_args.items():
        if not isinstance(key, str):
            msg = (
                "MV-DINO args file keys must be strings; "
                f"found {type(key).__name__} at {args_path}."
            )
            raise TypeError(msg)
        config_kwargs[key] = value

    model_factory = cast(
        _ModelFactory,
        model_utils.get_dino3d_model,
    )
    checkpoint_factory = cast(
        _CheckpointFactory,
        checkpointing.CheckPoint,
    )
    model = model_factory(_SafeConfig(**config_kwargs)).to(device)
    checkpoint_dir = path_join(model_root, "checkpoints")
    model = checkpoint_factory(checkpoint_dir).load_model(model, latest=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(requires_grad=False)
    return model


@lru_cache(maxsize=4)
def _load_feature_upsampler(device: str) -> torch.nn.Module:
    """Load the optional frozen feature upsampler."""
    upsampler = torch.hub.load("wimmerth/anyup", "anyup", trust_repo=True)
    upsampler = upsampler.to(device).eval()
    for param in upsampler.parameters():
        param.requires_grad_(requires_grad=False)
    return upsampler


@jaxtyped(typechecker=beartype)
def _plucker_from_rays(
    rays_dir: Float[torch.Tensor, "height width 3"],
    t_to_world: Float[torch.Tensor, "4 4"],
) -> Float[torch.Tensor, "6 height width"]:
    """Build fixed world-frame Plucker rays from renderer ray directions."""
    with torch.no_grad():
        rotation = t_to_world[:3, :3]
        center = t_to_world[:3, 3]
        direction = torch.einsum("ij,hwj->hwi", rotation, rays_dir)
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(
            1e-6
        )
        direction = direction.permute(2, 0, 1)
        moment = torch.cross(
            center.view(3, 1, 1).expand_as(direction),
            direction,
            dim=0,
        )
        plucker = torch.cat([direction, moment], dim=0)
        return torch.nan_to_num(plucker, nan=0.0, posinf=0.0, neginf=0.0)


def _theta_cos_bounds(
    theta_min_deg: float, theta_max_deg: float
) -> tuple[float, float]:
    """Return inclusive cos(theta) bounds for theta_min <= theta <= theta_max."""
    cos_min = math.cos(math.radians(theta_max_deg))
    cos_max = math.cos(math.radians(theta_min_deg))
    return cos_min, cos_max


def _imagenet_stats(
    device: torch.device,
    *,
    batched: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ImageNet normalization tensors on the target device."""
    shape = (1, 3, 1, 1) if batched else (3, 1, 1)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(*shape)
    std = torch.tensor(_IMAGENET_STD, device=device).view(*shape)
    return mean, std


@jaxtyped(typechecker=beartype)
def _rim_crop_origins(
    cos_t: Float[torch.Tensor, "height width"],
    cos_min: float,
    cos_max: float,
    n_crops: int,
    crop: int,
) -> list[tuple[int, int]]:
    """Pick crop origins distributed around the theta rim annulus."""
    height, width = cos_t.shape
    if crop > height or crop > width:
        msg = (
            f"MV-DINO rim crop {crop}px exceeds render size {height}x{width}."
        )
        raise RuntimeError(msg)

    rim = (cos_t >= cos_min) & (cos_t <= cos_max)
    ys, xs = torch.where(rim)
    if ys.numel() == 0:
        return []

    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    angles = torch.atan2(ys.float() - center_y, xs.float() - center_x)
    origins: list[tuple[int, int]] = []
    for crop_index in range(n_crops):
        target = -math.pi + 2.0 * math.pi * (crop_index + 0.5) / n_crops
        selected = int(torch.argmin(torch.abs(angles - target)))
        origin_y = int(ys[selected].item()) - crop // 2
        origin_x = int(xs[selected].item()) - crop // 2
        origin_y = max(0, min(height - crop, origin_y))
        origin_x = max(0, min(width - crop, origin_x))
        origins.append((origin_y, origin_x))
    return origins


@jaxtyped(typechecker=beartype)
def mvdino_feature_rim_loss(
    rgb_pred: Float[torch.Tensor, "batch height width 3"],
    rgb_gt: Float[torch.Tensor, "batch height width 3"],
    rays_dir: Float[torch.Tensor, "batch height width 3"],
    t_to_world: Float[torch.Tensor, "batch 4 4"],
    depth_ray_z: Float[torch.Tensor, "batch height width 1"]
    | Float[torch.Tensor, "batch height width"]
    | None,
    mask: Float[torch.Tensor, "batch height width 1"] | None,
    *,
    theta_min_deg: float = 60.0,
    theta_max_deg: float = 80.0,
    exp_dir: str | None = None,
    name: str = "dinov2_reg",
    use_feature_upsampler: bool = False,
    upsampled_feature_size: int = 512,
) -> Float[torch.Tensor, ""]:
    """Compute MV-DINO feature L1 over the fisheye rim band."""
    device = rgb_pred.device
    model = _load_mvdino(exp_dir, name, str(device))
    mean, std = _imagenet_stats(device, batched=True)
    batch_size, height, width, _ = rgb_pred.shape
    cos_min, cos_max = _theta_cos_bounds(theta_min_deg, theta_max_deg)

    total = rgb_pred.new_zeros(())
    for batch_index in range(batch_size):
        pred_chw = rgb_pred[batch_index].permute(2, 0, 1).unsqueeze(0)
        gt_chw = rgb_gt[batch_index].permute(2, 0, 1).unsqueeze(0)
        pred_in = (
            F.interpolate(
                pred_chw,
                (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            - mean
        ) / std
        gt_in = (
            F.interpolate(
                gt_chw,
                (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            - mean
        ) / std
        image_stack = torch.stack([pred_in[0], gt_in[0]], dim=0).unsqueeze(0)

        plucker = _plucker_from_rays(
            rays_dir[batch_index],
            t_to_world[batch_index],
        ).unsqueeze(0)
        plucker_small = F.interpolate(
            plucker,
            (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        plucker_stack = plucker_small.unsqueeze(1).expand(
            1,
            2,
            6,
            _DINO_INPUT_SIZE,
            _DINO_INPUT_SIZE,
        )

        features = model(
            image_stack,
            plucker_stack.contiguous(),
            intrinsics=None,
            extrinsics=None,
        )
        pred_features = features[0, 0]
        gt_features = features[0, 1]
        if use_feature_upsampler:
            upsampler = _load_feature_upsampler(str(device))
            guide_pred = F.interpolate(
                pred_in,
                (upsampled_feature_size, upsampled_feature_size),
                mode="bilinear",
                align_corners=False,
            )
            guide_gt = F.interpolate(
                gt_in,
                (upsampled_feature_size, upsampled_feature_size),
                mode="bilinear",
                align_corners=False,
            )
            output_size = (upsampled_feature_size, upsampled_feature_size)
            pred_features = upsampler(
                guide_pred,
                pred_features.unsqueeze(0),
                output_size=output_size,
            )[0]
            gt_features = upsampler(
                guide_gt,
                gt_features.unsqueeze(0),
                output_size=output_size,
            )[0]

        feature_height, feature_width = pred_features.shape[1:]
        diff = (pred_features - gt_features).abs().mean(dim=0)
        if depth_ray_z is not None:
            cos_t = depth_ray_z[batch_index]
            if cos_t.ndim == 3 and cos_t.shape[-1] == 1:
                cos_t = cos_t[..., 0]
            cos_small = F.interpolate(
                cos_t.view(1, 1, height, width),
                size=(feature_height, feature_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            rim = (cos_small >= cos_min) & (cos_small <= cos_max)
        else:
            yy, xx = torch.meshgrid(
                torch.arange(feature_height, device=device),
                torch.arange(feature_width, device=device),
                indexing="ij",
            )
            radius = torch.sqrt(
                (yy - (feature_height - 1) / 2) ** 2
                + (xx - (feature_width - 1) / 2) ** 2
            ) / (min(feature_height, feature_width) / 2)
            rim = (radius > 0.65) & (radius < 0.93)

        if mask is not None:
            batch_mask = mask[batch_index]
            if batch_mask.ndim == 3:
                batch_mask = batch_mask[..., 0]
            mask_small = F.interpolate(
                batch_mask.view(1, 1, height, width).float(),
                size=(feature_height, feature_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            rim = rim & (mask_small > 0.5)
        if bool(rim.sum() == 0):
            continue
        total = total + diff[rim].mean()

    out = total / max(batch_size, 1)
    if not torch.isfinite(out):
        return rgb_pred.sum() * 0.0
    return out


@jaxtyped(typechecker=beartype)
def mvdino_rim_crop_loss(
    rgb_pred: Float[torch.Tensor, "batch height width 3"],
    rgb_gt: Float[torch.Tensor, "batch height width 3"],
    rays_dir: Float[torch.Tensor, "batch height width 3"],
    t_to_world: Float[torch.Tensor, "batch 4 4"],
    depth_ray_z: Float[torch.Tensor, "batch height width 1"]
    | Float[torch.Tensor, "batch height width"]
    | None,
    mask: Float[torch.Tensor, "batch height width 1"] | None,
    *,
    theta_min_deg: float = 60.0,
    theta_max_deg: float = 80.0,
    exp_dir: str | None = None,
    name: str = "dinov2_reg",
    n_crops: int = 8,
    crop: int = 518,
) -> Float[torch.Tensor, ""]:
    """Compute MV-DINO feature L1 on native-resolution rim crops."""
    device = rgb_pred.device
    model = _load_mvdino(exp_dir, name, str(device))
    mean, std = _imagenet_stats(device, batched=False)
    batch_size, height, width, _ = rgb_pred.shape
    cos_min, cos_max = _theta_cos_bounds(theta_min_deg, theta_max_deg)

    total = rgb_pred.new_zeros(())
    n_used = 0
    for batch_index in range(batch_size):
        if depth_ray_z is not None:
            cos_t = depth_ray_z[batch_index]
            if cos_t.ndim == 3 and cos_t.shape[-1] == 1:
                cos_t = cos_t[..., 0]
        else:
            yy, xx = torch.meshgrid(
                torch.arange(height, device=device).float(),
                torch.arange(width, device=device).float(),
                indexing="ij",
            )
            radius = torch.sqrt(
                (yy - (height - 1) / 2) ** 2 + (xx - (width - 1) / 2) ** 2
            ) / (min(height, width) / 2)
            cos_t = torch.cos(torch.asin(radius.clamp(0, 1)))

        origins = _rim_crop_origins(cos_t, cos_min, cos_max, n_crops, crop)
        plucker_full = _plucker_from_rays(
            rays_dir[batch_index],
            t_to_world[batch_index],
        )
        pred_chw = rgb_pred[batch_index].permute(2, 0, 1)
        gt_chw = rgb_gt[batch_index].permute(2, 0, 1)
        for origin_y, origin_x in origins:
            y_slice = slice(origin_y, origin_y + crop)
            x_slice = slice(origin_x, origin_x + crop)
            pred_crop = (pred_chw[:, y_slice, x_slice] - mean) / std
            gt_crop = (gt_chw[:, y_slice, x_slice] - mean) / std
            crop_cos = cos_t[y_slice, x_slice]
            rim_crop = (crop_cos >= cos_min) & (crop_cos <= cos_max)
            if mask is not None:
                batch_mask = mask[batch_index]
                if batch_mask.ndim == 3:
                    batch_mask = batch_mask[..., 0]
                rim_crop = rim_crop & (batch_mask[y_slice, x_slice] > 0.5)
            if bool(rim_crop.sum() == 0):
                continue

            pred_input = F.interpolate(
                pred_crop.unsqueeze(0),
                (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
            gt_input = F.interpolate(
                gt_crop.unsqueeze(0),
                (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
            image_stack = torch.stack([pred_input, gt_input], dim=0).unsqueeze(
                0
            )
            plucker_crop = plucker_full[:, y_slice, x_slice].unsqueeze(0)
            plucker_small = F.interpolate(
                plucker_crop,
                (_DINO_INPUT_SIZE, _DINO_INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            plucker_stack = plucker_small.unsqueeze(1).expand(
                1,
                2,
                6,
                _DINO_INPUT_SIZE,
                _DINO_INPUT_SIZE,
            )
            features = model(
                image_stack,
                plucker_stack.contiguous(),
                intrinsics=None,
                extrinsics=None,
            )
            pred_features = features[0, 0]
            gt_features = features[0, 1]
            feature_height, feature_width = pred_features.shape[1:]
            rim_small = (
                F.interpolate(
                    rim_crop.float().view(1, 1, crop, crop),
                    (feature_height, feature_width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                > 0.5
            )
            if bool(rim_small.sum() == 0):
                continue
            diff = (pred_features - gt_features).abs().mean(dim=0)
            total = total + diff[rim_small].mean()
            n_used += 1

    out = total / max(n_used, 1)
    if not torch.isfinite(out):
        return rgb_pred.sum() * 0.0
    return out
