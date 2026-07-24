"""Visibility-adaptive absolute COLMAP extrinsic optimization."""

from dataclasses import replace

import torch
from torch import nn

from threedgrut.datasets.protocols import Batch
from threedgrut.optimizers.native_camera_adam import NativeCameraAdam

NATIVE_CAMERA_CHECKPOINT_VERSION = 3
NATIVE_CAMERA_ALGORITHM = "native_absolute_colmap"


def colmap_qvec_to_w2c_rotation(qvec: torch.Tensor) -> torch.Tensor:
    """Convert a unit scalar-first COLMAP quaternion to a W2C rotation."""
    if qvec.shape != (4,):
        msg = f"COLMAP qvec must have shape (4,), got {qvec.shape}."
        raise ValueError(msg)
    w, x, y, z = qvec.unbind()
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        )
    ).reshape(3, 3)


def colmap_w2c_to_world_rays(
    *,
    rays_ori: torch.Tensor,
    rays_dir: torch.Tensor,
    qvec: torch.Tensor,
    tvec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bake one absolute COLMAP W2C pose into camera-space rays."""
    if rays_ori.shape != rays_dir.shape or rays_ori.shape[-1] != 3:
        msg = "Ray origins and directions must have matching (..., 3) shapes."
        raise ValueError(msg)
    if tvec.shape != (3,):
        msg = f"COLMAP tvec must have shape (3,), got {tvec.shape}."
        raise ValueError(msg)

    rotation = colmap_qvec_to_w2c_rotation(qvec)
    center = -(tvec @ rotation)

    finite_origins = torch.isfinite(rays_ori).all(dim=-1, keepdim=True)
    finite_directions = torch.isfinite(rays_dir).all(dim=-1, keepdim=True)
    safe_origins = torch.where(
        finite_origins,
        rays_ori,
        torch.zeros_like(rays_ori),
    )
    safe_directions = torch.where(
        finite_directions,
        rays_dir,
        torch.zeros_like(rays_dir),
    )
    world_origins = safe_origins @ rotation + center
    world_directions = safe_directions @ rotation
    world_origins = torch.where(finite_origins, world_origins, rays_ori)
    world_directions = torch.where(
        finite_directions,
        world_directions,
        rays_dir,
    )
    return world_origins, world_directions


def colmap_w2c_to_c2w(
    *,
    qvec: torch.Tensor,
    tvec: torch.Tensor,
) -> torch.Tensor:
    """Return a differentiable camera-to-world matrix from COLMAP W2C."""
    if tvec.shape != (3,):
        msg = f"COLMAP tvec must have shape (3,), got {tvec.shape}."
        raise ValueError(msg)
    rotation = colmap_qvec_to_w2c_rotation(qvec)
    center = -(tvec @ rotation)
    c2w = torch.eye(4, dtype=qvec.dtype, device=qvec.device)
    c2w = c2w.clone()
    c2w[:3, :3] = rotation.transpose(0, 1)
    c2w[:3, 3] = center
    return c2w


class NativeCameraExtrinsics(nn.Module):
    """Absolute per-image COLMAP qvec/tvec state used by visibility-adaptive."""

    checkpoint_format_version = NATIVE_CAMERA_CHECKPOINT_VERSION
    checkpoint_algorithm = NATIVE_CAMERA_ALGORITHM
    optimize_per_image = True

    def __init__(
        self,
        *,
        initial_qvecs: torch.Tensor,
        initial_tvecs: torch.Tensor,
    ) -> None:
        """Initialize absolute poses from the pre-split COLMAP model."""
        super().__init__()
        if initial_qvecs.ndim != 2 or initial_qvecs.shape[1] != 4:
            msg = "Native camera qvecs must have shape (N, 4)."
            raise ValueError(msg)
        if initial_tvecs.shape != (initial_qvecs.shape[0], 3):
            msg = "Native camera tvecs must have shape (N, 3)."
            raise ValueError(msg)
        qvecs = initial_qvecs.to(dtype=torch.float32)
        tvecs = initial_tvecs.to(dtype=torch.float32)
        if not bool(torch.isfinite(qvecs).all()) or not bool(
            torch.isfinite(tvecs).all()
        ):
            msg = "Native camera initialization must be finite."
            raise ValueError(msg)
        qvec_norms = torch.linalg.vector_norm(qvecs, dim=1)
        if not bool(
            torch.allclose(
                qvec_norms, torch.ones_like(qvec_norms), atol=1e-4, rtol=0.0
            )
        ):
            msg = "Native camera initialization requires unit qvecs."
            raise ValueError(msg)
        self.qvecs = nn.Parameter(qvecs.clone())
        self.tvecs = nn.Parameter(tvecs.clone())
        self.register_buffer("initial_qvecs", qvecs.clone())
        self.register_buffer("initial_tvecs", tvecs.clone())

    def is_active(self, global_step: int) -> bool:
        """Native absolute extrinsics are active from the first step."""
        return global_step >= -1

    def optimizer_group_manifest(self) -> tuple[tuple[str, ...], ...]:
        """Return the checkpoint-stable native parameter grouping."""
        return (("qvecs",), ("tvecs",))

    def create_optimizer(
        self,
    ) -> tuple[
        NativeCameraAdam,
        torch.optim.lr_scheduler.LRScheduler,
    ]:
        """Create the recovered per-image Adam and constant scheduler."""
        optimizer = NativeCameraAdam(
            (
                {"name": "qvecs", "params": [self.qvecs]},
                {"name": "tvecs", "params": [self.tvecs]},
            ),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda _: 1.0,
        )
        return optimizer, scheduler

    def forward(self, batch: Batch, global_step: int = -1) -> Batch:
        """Apply absolute pose values and their local ray Jacobian."""
        if not self.is_active(global_step):
            return batch
        source_frame_idx = int(batch.source_frame_idx)
        if not 0 <= source_frame_idx < self.qvecs.shape[0]:
            msg = (
                "Invalid source_frame_idx for native camera extrinsics: "
                f"{source_frame_idx}."
            )
            raise ValueError(msg)
        qvec = self.qvecs[source_frame_idx]
        tvec = self.tvecs[source_frame_idx]
        world_origins, world_directions = colmap_w2c_to_world_rays(
            rays_ori=batch.rays_ori,
            rays_dir=batch.rays_dir,
            qvec=qvec,
            tvec=tvec,
        )
        rotation = colmap_qvec_to_w2c_rotation(qvec)
        center = -(tvec @ rotation)
        detached_rotation = rotation.detach()
        detached_center = center.detach()
        proxy_origins = (
            world_origins - detached_center
        ) @ detached_rotation.transpose(0, 1)
        proxy_directions = world_directions @ detached_rotation.transpose(0, 1)
        proxy_origins = batch.rays_ori + (
            proxy_origins - proxy_origins.detach()
        )
        proxy_directions = batch.rays_dir + (
            proxy_directions - proxy_directions.detach()
        )
        c2w = colmap_w2c_to_c2w(qvec=qvec, tvec=tvec)
        c2w = c2w.to(dtype=batch.T_to_world.dtype).unsqueeze(0)
        c2w = c2w.expand(batch.T_to_world.shape[0], -1, -1)
        return replace(
            batch,
            rays_ori=proxy_origins,
            rays_dir=proxy_directions,
            T_to_world=c2w,
            rays_in_world_space=False,
        )

    def get_regularization_loss(self) -> torch.Tensor:
        """Native camera optimization has no pose regularizer."""
        return self.qvecs.sum() * 0.0

    def max_abs_grad(self) -> float:
        """Return the largest absolute qvec/tvec gradient."""
        gradients = [
            parameter.grad.detach().abs().max()
            for parameter in (self.qvecs, self.tvecs)
            if parameter.grad is not None
        ]
        if not gradients:
            return 0.0
        return torch.stack(gradients).max().item()

    def validate_state(self) -> None:
        """Fail if a loaded or optimized absolute pose is invalid."""
        if not bool(torch.isfinite(self.qvecs).all()) or not bool(
            torch.isfinite(self.tvecs).all()
        ):
            msg = "Native camera state contains non-finite values."
            raise ValueError(msg)
        norms = torch.linalg.vector_norm(self.qvecs, dim=1)
        if not bool(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0.0)
        ):
            msg = "Native camera state contains non-unit qvecs."
            raise ValueError(msg)

    @torch.no_grad()
    def stats(self) -> dict[str, float]:
        """Return maximum pose drift from the source COLMAP model."""
        current_qvecs = self.qvecs / torch.linalg.vector_norm(
            self.qvecs,
            dim=1,
            keepdim=True,
        )
        initial_qvecs = self.initial_qvecs / torch.linalg.vector_norm(
            self.initial_qvecs,
            dim=1,
            keepdim=True,
        )
        cosine = torch.sum(current_qvecs * initial_qvecs, dim=1).abs()
        rotation = 2.0 * torch.acos(cosine.clamp(0.0, 1.0))
        current_centers = self._camera_centers(self.qvecs, self.tvecs)
        initial_centers = self._camera_centers(
            self.initial_qvecs,
            self.initial_tvecs,
        )
        center_delta = torch.linalg.vector_norm(
            current_centers - initial_centers,
            dim=1,
        )
        return {
            "rotation_norm_rad": rotation.max().item(),
            "translation_norm_m": center_delta.max().item(),
            "rolling_rotation_norm_rad": 0.0,
            "rolling_translation_norm_m": 0.0,
            "max_abs_grad": self.max_abs_grad(),
        }

    @staticmethod
    def _camera_centers(
        qvecs: torch.Tensor,
        tvecs: torch.Tensor,
    ) -> torch.Tensor:
        w, x, y, z = qvecs.unbind(dim=1)
        rotations = torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=1,
        ).reshape(-1, 3, 3)
        return -torch.bmm(tvecs.unsqueeze(1), rotations).squeeze(1)
