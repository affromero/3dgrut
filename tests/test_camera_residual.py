import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch
from threedgrut.model.camera_residual import CameraResidual
from threedgrut.trainer import Trainer3DGRUT


def make_residual(
    *,
    num_images: int = 0,
    optimize_per_camera: bool = False,
    optimize_per_image: bool = False,
    rotation_lr: float = 1.0e-4,
    translation_lr: float = 1.0e-4,
    lr_end_fraction: float = 1.0,
    warmup_steps: int = 0,
) -> CameraResidual:
    return CameraResidual(
        num_cameras=2,
        num_images=num_images,
        lr=1.0e-4,
        rotation_lr=rotation_lr,
        translation_lr=translation_lr,
        betas=(0.9, 0.99),
        eps=1.0e-10,
        lr_end_fraction=lr_end_fraction,
        warmup_steps=warmup_steps,
        reg_lambda=0.0,
        max_rotation_rad=0.1,
        max_translation_m=0.2,
        max_rolling_rotation_rad=0.01,
        max_rolling_translation_m=0.02,
        optimize_global=False,
        optimize_per_camera=optimize_per_camera,
        optimize_per_image=optimize_per_image,
        optimize_rolling_per_camera=False,
        n_iterations=4,
    )


def make_batch(*, frame_idx: int, source_frame_idx: int) -> Batch:
    return Batch(
        rays_ori=torch.zeros(1, 1, 1, 3),
        rays_dir=torch.tensor([[[[0.0, 0.0, 1.0]]]]),
        T_to_world=torch.eye(4).unsqueeze(0),
        camera_idx=0,
        frame_idx=frame_idx,
        source_frame_idx=source_frame_idx,
    )


def test_per_image_residual_uses_stable_source_frame_index() -> None:
    residual = make_residual(num_images=3, optimize_per_image=True)
    with torch.no_grad():
        residual.image_translation_raw[2, 0] = torch.atanh(
            torch.tensor(0.5)
        )

    corrected = residual(make_batch(frame_idx=0, source_frame_idx=2))

    assert corrected.rays_ori[0, 0, 0, 0].item() == pytest.approx(0.1)


def test_per_image_residual_rejects_unknown_source_frame_index() -> None:
    residual = make_residual(num_images=2, optimize_per_image=True)

    with pytest.raises(ValueError, match="source_frame_idx"):
        residual(make_batch(frame_idx=0, source_frame_idx=2))


def test_per_image_residual_rejects_missing_source_frame_index() -> None:
    residual = make_residual(num_images=2, optimize_per_image=True)

    with pytest.raises(ValueError, match="source_frame_idx"):
        residual(make_batch(frame_idx=0, source_frame_idx=-1))


def test_optimizer_separates_rotation_and_translation_parameters() -> None:
    residual = make_residual(
        num_images=3,
        optimize_per_image=True,
        rotation_lr=1.0e-5,
        translation_lr=1.0e-3,
    )
    optimizer, _scheduler = residual.create_optimizer()
    names_by_id = {
        id(parameter): name for name, parameter in residual.named_parameters()
    }

    assert [group["lr"] for group in optimizer.param_groups] == [
        1.0e-5,
        1.0e-3,
    ]
    assert {
        names_by_id[id(parameter)]
        for parameter in optimizer.param_groups[0]["params"]
    } == {"image_rotation_raw"}
    assert {
        names_by_id[id(parameter)]
        for parameter in optimizer.param_groups[1]["params"]
    } == {"image_translation_raw"}
    assert optimizer.defaults["betas"] == (0.9, 0.99)
    assert optimizer.defaults["eps"] == 1.0e-10
    assert residual.optimizer_group_manifest() == (
        ("image_rotation_raw",),
        ("image_translation_raw",),
    )


def test_optimizer_manifest_distinguishes_optimization_modes() -> None:
    per_image = make_residual(num_images=3, optimize_per_image=True)
    per_camera = make_residual(optimize_per_camera=True)

    assert (
        per_image.optimizer_group_manifest()
        != per_camera.optimizer_group_manifest()
    )


def test_warmup_keeps_residual_inactive_until_boundary() -> None:
    residual = make_residual(
        num_images=1,
        optimize_per_image=True,
        warmup_steps=3,
    )
    batch = make_batch(frame_idx=0, source_frame_idx=0)

    assert not residual.is_active(2)
    assert residual(batch, global_step=2) is batch
    assert residual.is_active(3)


def test_first_adam_step_has_expected_bounded_physical_displacement() -> None:
    residual = make_residual(
        num_images=1,
        optimize_per_image=True,
        rotation_lr=0.01,
        translation_lr=0.02,
    )
    optimizer, _scheduler = residual.create_optimizer()
    rotation, translation = residual._bounded(0, 0)
    (-rotation.sum() - translation.sum()).backward()

    optimizer.step()
    rotation, translation = residual._bounded(0, 0)

    assert rotation[0, 0].item() == pytest.approx(
        0.1 * torch.tanh(torch.tensor(0.01)).item()
    )
    assert translation[0, 0].item() == pytest.approx(
        0.2 * torch.tanh(torch.tensor(0.02)).item()
    )


def test_stats_report_active_per_image_residuals() -> None:
    residual = make_residual(num_images=3, optimize_per_image=True)
    with torch.no_grad():
        residual.image_rotation_raw[2, 0] = torch.atanh(torch.tensor(0.5))
        residual.image_translation_raw[1, 1] = torch.atanh(
            torch.tensor(0.25)
        )

    stats = residual.stats()

    assert stats["rotation_norm_rad"] == pytest.approx(0.05)
    assert stats["translation_norm_m"] == pytest.approx(0.05)


def test_gradient_guard_accepts_later_empty_batches_after_connectivity() -> None:
    residual = make_residual(num_images=1, optimize_per_image=True)
    trainer = object.__new__(Trainer3DGRUT)
    trainer.camera_residual = residual
    trainer.conf = OmegaConf.create(
        {
            "camera_residual": {
                "fail_on_zero_grad": True,
                "fail_after_steps": 2,
                "min_abs_grad": 1.0e-12,
            }
        }
    )
    trainer._camera_residual_gradient_observed = False
    residual.image_rotation_raw.grad = torch.ones_like(
        residual.image_rotation_raw
    )

    trainer._validate_camera_residual_gradient(global_step=0)
    residual.image_rotation_raw.grad = None
    trainer._validate_camera_residual_gradient(global_step=10)

    assert trainer._camera_residual_gradient_observed


def test_gradient_guard_rejects_never_connected_residual() -> None:
    trainer = object.__new__(Trainer3DGRUT)
    trainer.camera_residual = make_residual(
        num_images=1,
        optimize_per_image=True,
    )
    trainer.conf = OmegaConf.create(
        {
            "camera_residual": {
                "fail_on_zero_grad": True,
                "fail_after_steps": 2,
                "min_abs_grad": 1.0e-12,
            }
        }
    )
    trainer._camera_residual_gradient_observed = False

    with pytest.raises(RuntimeError, match="no gradient reached"):
        trainer._validate_camera_residual_gradient(global_step=2)


def test_scheduler_preserves_group_ratio_and_reaches_end_fraction() -> None:
    residual = make_residual(
        num_images=1,
        optimize_per_image=True,
        rotation_lr=1.0e-5,
        translation_lr=1.0e-3,
        lr_end_fraction=0.25,
    )
    optimizer, scheduler = residual.create_optimizer()

    for _step in range(4):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-6)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(2.5e-4)
    assert (
        optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"]
    ) == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"rotation_lr": 0.0}, "rotation_lr"),
        ({"translation_lr": float("inf")}, "translation_lr"),
    ),
)
def test_optimizer_config_rejects_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_residual(**kwargs)
