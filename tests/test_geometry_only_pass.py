"""Tests for the independent sparse geometry supervision pass."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers.sparse_geometry_adam import SparseGeometryAdam
from threedgrut.trainer import Trainer3DGRUT


class _GeometryModel:
    """Minimal callable model exposing the geometry optimizer contract."""

    def __init__(self) -> None:
        self.positions = torch.nn.Parameter(torch.zeros((2, 3)))
        self.density = torch.nn.Parameter(torch.zeros((2, 1)))
        self.rotation = torch.nn.Parameter(torch.zeros((2, 4)))
        self.scale = torch.nn.Parameter(torch.zeros((2, 3)))
        self.build_count = 0
        self.build_modes: list[bool] = []
        self.optimizer = torch.optim.Adam(
            [
                {"name": name, "params": [getattr(self, name)], "lr": 0.01}
                for name in ("positions", "density", "rotation", "scale")
            ]
        )

    def __call__(
        self,
        gpu_batch: object,
        *,
        train: bool,
        frame_id: int,
    ) -> dict[str, torch.Tensor]:
        del train, frame_id
        return {"pred_rgb": torch.zeros_like(gpu_batch.rgb_gt)}

    def build_acc(self, *, rebuild: bool) -> None:
        self.build_count += 1
        self.build_modes.append(rebuild)


def _loss_config(*, geometry_enabled: bool) -> object:
    return OmegaConf.create(
        {
            "use_l1": True,
            "lambda_l1": 1.0,
            "l1_loss_type": "absolute",
            "erp_latitude_weighting": False,
            "use_l2": False,
            "use_ssim": False,
            "use_opacity": False,
            "use_scale": False,
            "use_sky_opacity": False,
            "use_depth": True,
            "lambda_depth": 0.5,
            "depth_loss_type": "log_l1",
            "depth_apply_rgb_mask": True,
            "depth_min_m": 0.05,
            "use_rim_depth": False,
            "use_rim_hf": False,
            "use_mvdino_rim": False,
            "use_camera_loss_weights": False,
            "use_image_loss_weights": False,
            "geometry_only_pass": {
                "enabled": geometry_enabled,
                "start_iteration": 101,
                "frequency": 2,
                "learning_rate_scale": 0.25,
            },
        }
    )


def _loss_trainer(*, geometry_enabled: bool) -> Trainer3DGRUT:
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.device = torch.device("cpu")
    trainer.conf = OmegaConf.create({"loss": _loss_config(geometry_enabled=geometry_enabled)})
    return trainer


def _depth_batch() -> SimpleNamespace:
    return SimpleNamespace(
        rgb_gt=torch.zeros((1, 2, 2, 3)),
        mask=torch.ones((1, 2, 2, 1)),
        sky_mask=None,
        depth_gt=torch.full((1, 2, 2, 1), 2.0),
        rays_dir=torch.tensor([[[[0.0, 0.0, 1.0]]]]).expand(1, 2, 2, 3),
        rays_in_world_space=False,
        depth_ray_z=None,
        intrinsics_EquirectCameraModelParameters=None,
        camera_idx=0,
        image_path="camera_0001.png",
    )


def _depth_outputs() -> dict[str, torch.Tensor]:
    return {
        "pred_rgb": torch.zeros((1, 2, 2, 3)),
        "pred_dist": torch.ones((1, 2, 2, 1), requires_grad=True),
    }


def test_geometry_only_pass_isolates_depth_from_image_loss() -> None:
    """Measured depth remains reportable but does not enter image Adam."""
    isolated = _loss_trainer(geometry_enabled=True).get_losses(
        _depth_batch(),
        _depth_outputs(),
    )
    joint = _loss_trainer(geometry_enabled=False).get_losses(
        _depth_batch(),
        _depth_outputs(),
    )

    assert isolated["depth_valid_count"].item() == 4
    assert isolated["depth_loss"].item() > 0.0
    assert isolated["total_loss"].item() == pytest.approx(0.0)
    assert joint["total_loss"].item() == pytest.approx(joint["depth_loss"].item())


def _validation_trainer() -> Trainer3DGRUT:
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "optimizer": {"type": "adam"},
            "loss": _loss_config(geometry_enabled=True),
            "strategy": {
                "method": "GSStrategy",
                "densify": {
                    "start_iteration": 1,
                    "end_iteration": 100,
                },
                "prune": {
                    "start_iteration": 1,
                    "end_iteration": 100,
                },
                "prune_scale": {
                    "start_iteration": -1,
                    "end_iteration": -1,
                },
                "density_decay": {
                    "start_iteration": -1,
                    "end_iteration": -1,
                },
                "reset_density": {
                    "start_iteration": 1,
                    "end_iteration": 100,
                },
                "scale_guard": {"enabled": False},
            },
            "model": {"density_activation": "sigmoid"},
        }
    )
    trainer.camera_residual = None
    trainer._distillation_start_step = -1
    return trainer


def test_geometry_only_pass_schedule_starts_after_topology_mutation() -> None:
    """The pass starts on its preregistered step and follows its cadence."""
    trainer = _validation_trainer()

    trainer._validate_geometry_only_pass_config()

    assert not trainer._geometry_only_pass_is_scheduled(100)
    assert trainer._geometry_only_pass_is_scheduled(101)
    assert not trainer._geometry_only_pass_is_scheduled(102)
    assert trainer._geometry_only_pass_is_scheduled(103)


def test_geometry_only_pass_rejects_topology_overlap() -> None:
    """Independent optimizer state cannot span a Gaussian row mutation."""
    trainer = _validation_trainer()
    trainer.conf.loss.geometry_only_pass.start_iteration = 100

    with pytest.raises(ValueError, match="after all topology"):
        trainer._validate_geometry_only_pass_config()


def test_geometry_only_pass_rejects_unbounded_topology_mutation() -> None:
    """An end iteration of -1 means active forever, not already ended."""
    trainer = _validation_trainer()
    trainer.conf.strategy.prune_scale.start_iteration = 1

    with pytest.raises(ValueError, match="unbounded.*prune_scale"):
        trainer._validate_geometry_only_pass_config()


def test_geometry_only_pass_rejects_later_density_mutation() -> None:
    """External density decay cannot invalidate independent Adam moments."""
    trainer = _validation_trainer()
    trainer.conf.strategy.density_decay.start_iteration = 101
    trainer.conf.strategy.density_decay.end_iteration = 200

    with pytest.raises(ValueError, match="topology and density"):
        trainer._validate_geometry_only_pass_config()


def test_geometry_only_pass_rejects_nonstandard_image_optimizer() -> None:
    """The isolated arm cannot silently alter the image optimizer contract."""
    trainer = _validation_trainer()
    trainer.conf.optimizer.type = "selective_adam"

    with pytest.raises(ValueError, match="ordinary Adam"):
        trainer._validate_geometry_only_pass_config()


def test_geometry_only_pass_rejects_ppisp_distillation_freeze() -> None:
    """A scheduled Gaussian freeze cannot invalidate the sparse optimizer."""
    trainer = _validation_trainer()
    trainer._distillation_start_step = 101

    with pytest.raises(ValueError, match="distillation freezes Gaussian"):
        trainer._validate_geometry_only_pass_config()


def test_geometry_only_pass_rejects_parameter_replacement() -> None:
    """Runtime row replacement cannot silently retain stale sparse state."""
    parameters = {
        "positions": torch.nn.Parameter(torch.zeros((2, 3))),
        "density": torch.nn.Parameter(torch.zeros((2, 1))),
        "rotation": torch.nn.Parameter(torch.zeros((2, 4))),
        "scale": torch.nn.Parameter(torch.zeros((2, 3))),
    }
    optimizer = SparseGeometryAdam(
        [{"name": name, "params": [parameter], "lr": 0.01} for name, parameter in parameters.items()]
    )
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.model = SimpleNamespace(**parameters)
    trainer.model.positions = torch.nn.Parameter(torch.zeros((3, 3)))

    with pytest.raises(RuntimeError, match="topology mutation"):
        trainer._validate_geometry_only_parameter_identity(optimizer)


def test_geometry_only_pass_refits_before_render_and_steps_geometry() -> None:
    """The depth render sees the preceding image update's current geometry."""
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.model = _GeometryModel()
    trainer.conf = OmegaConf.create(
        {
            "enable_writer": False,
            "loss": _loss_config(geometry_enabled=True),
        }
    )
    trainer.geometry_only_optimizer = None
    trainer._pending_geometry_optimizer_state = None
    trainer.get_losses = lambda gpu_batch, outputs: {
        "depth_valid_count": torch.ones(1),
        "depth_loss": trainer.model.positions[0].sum(),
    }
    gpu_batch = SimpleNamespace(rgb_gt=torch.zeros((1, 2, 2, 3)))

    geometry_updated = trainer._run_geometry_only_pass(
        gpu_batch=gpu_batch,
        global_step=101,
    )

    assert geometry_updated
    assert torch.all(trainer.model.positions[0] < 0.0)
    assert trainer.model.build_count == 1
    assert trainer.model.build_modes == [False]


def test_checkpoint_configuration_reenables_frozen_gaussians() -> None:
    """A phase-frozen checkpoint resumes according to the new run contract."""
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    model.conf = OmegaConf.create(
        {
            "model": {
                "optimize_position": True,
                "optimize_density": True,
                "optimize_features_albedo": False,
                "optimize_features_specular": True,
                "optimize_rotation": True,
                "optimize_scale": True,
            }
        }
    )
    for name, shape in (
        ("positions", (2, 3)),
        ("density", (2, 1)),
        ("features_albedo", (2, 3)),
        ("features_specular", (2, 4)),
        ("rotation", (2, 4)),
        ("scale", (2, 3)),
    ):
        parameter = torch.nn.Parameter(torch.zeros(shape))
        parameter.requires_grad = False
        setattr(model, name, parameter)

    model.set_optimizable_parameters()

    assert model.positions.requires_grad
    assert model.density.requires_grad
    assert not model.features_albedo.requires_grad
    assert model.features_specular.requires_grad
    assert model.rotation.requires_grad
    assert model.scale.requires_grad


def test_checkpoint_optimizer_group_mismatch_fails_loudly() -> None:
    """Resume cannot silently replace checkpoint moments with fresh Adam."""
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    parameters = {
        "positions": torch.nn.Parameter(torch.zeros((2, 3))),
        "density": torch.nn.Parameter(torch.zeros((2, 1))),
        "features_albedo": torch.nn.Parameter(torch.zeros((2, 3))),
        "features_specular": torch.nn.Parameter(torch.zeros((2, 4))),
        "rotation": torch.nn.Parameter(torch.zeros((2, 4))),
        "scale": torch.nn.Parameter(torch.zeros((2, 3))),
    }
    for name, parameter in parameters.items():
        setattr(model, name, parameter)
    model.scene_extent = 1.0
    model.conf = OmegaConf.create(
        {
            "optimizer": {
                "type": "adam",
                "lr": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "resume_lr_scale": 1.0,
                "scale_position_lr_by_scene_extent": False,
                "params": {name: {"lr": 0.01} for name in parameters},
            },
            "scheduler": {},
        }
    )
    incompatible = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]).state_dict()

    with pytest.raises(ValueError, match="resume contract"):
        model.setup_optimizer(state_dict=incompatible)
