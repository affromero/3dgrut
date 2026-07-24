"""Behavior tests for configured Gaussian optimizer hyperparameters."""

import torch
from omegaconf import OmegaConf

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.optimizers.visibility_selective_adam import (
    VisibilitySelectiveAdam,
    FP16GlobalAdam,
)


def test_adam_uses_configured_betas_and_epsilon() -> None:
    """The optimizer must preserve recovered native Adam scalars."""
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    model.positions = torch.nn.Parameter(torch.zeros((1, 3)))
    model.scene_extent = 1.0
    model.conf = OmegaConf.create(
        {
            "optimizer": {
                "type": "adam",
                "lr": 0.0,
                "betas": [0.9, 0.99],
                "eps": 1.0e-10,
                "params": {"positions": {"lr": 0.001}},
            },
            "scheduler": {},
        }
    )

    model.setup_optimizer()

    parameter_group = model.optimizer.param_groups[0]
    assert parameter_group["betas"] == (0.9, 0.99)
    assert parameter_group["eps"] == 1.0e-10


def test_position_skip_scheduler_preserves_native_learning_rate() -> None:
    """Native per-Gaussian decay must not be compounded globally."""
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    model.positions = torch.nn.Parameter(torch.zeros((1, 3)))
    model.scene_extent = 1.0
    model.conf = OmegaConf.create(
        {
            "optimizer": {
                "type": "adam",
                "lr": 0.0,
                "betas": [0.9, 0.99],
                "eps": 1.0e-10,
                "params": {"positions": {"lr": 0.001}},
            },
            "scheduler": {
                "positions": {
                    "type": "skip",
                    "lr_init": 0.001,
                    "lr_final": 1.0e-6,
                    "max_steps": 30_000,
                }
            },
        }
    )

    model.setup_optimizer()
    model.scheduler_step(10_000)

    assert model.optimizer.param_groups[0]["lr"] == 0.001


def test_native_optimizer_splits_shared_color_encoder_state() -> None:
    """Global decoder weights must not enter per-Gaussian topology state."""
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    for name, width in (
        ("positions", 3),
        ("density", 1),
        ("features_albedo", 3),
        ("features_specular", 8),
        ("rotation", 4),
        ("scale", 3),
    ):
        setattr(
            model,
            name,
            torch.nn.Parameter(torch.zeros((2, width))),
        )
    model.color_encoder = torch.nn.Parameter(torch.zeros((45, 8)))
    model.scene_extent = 1.0
    model.conf = OmegaConf.create(
        {
            "optimizer": {
                "type": "visibility_selective_adam",
                "lr": 0.0,
                "betas": [0.9, 0.99],
                "eps": 1.0e-10,
                "scale_position_lr_by_scene_extent": False,
                "params": {
                    "positions": {"lr": 0.001},
                    "density": {"lr": 0.05},
                    "features_albedo": {"lr": 0.1},
                    "features_specular": {"lr": 0.1},
                    "color_encoder": {
                        "lr": 0.001,
                        "betas": [0.9, 0.99],
                        "eps": 1.0e-10,
                    },
                    "rotation": {"lr": 0.001},
                    "scale": {"lr": 0.005},
                },
            },
            "scheduler": {},
        }
    )

    model.setup_optimizer()

    assert isinstance(model.optimizer, VisibilitySelectiveAdam)
    assert {
        str(group["name"]) for group in model.optimizer.param_groups
    } == {
        "positions",
        "density",
        "features_albedo",
        "features_specular",
        "rotation",
        "scale",
    }
    assert model.color_encoder_optimizer is not None
    assert isinstance(
        model.color_encoder_optimizer,
        FP16GlobalAdam,
    )
    assert len(model.color_encoder_optimizer.param_groups) == 1
    assert (
        model.color_encoder_optimizer.param_groups[0]["params"][0]
        is model.color_encoder
    )
