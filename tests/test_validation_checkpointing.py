"""Tests for validation checkpointing and geometry-prior losses."""

import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.strategy.gs import GSStrategy
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.misc import sh_degree_to_specular_dim


class _DummyBackground:
    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.state_dict = state_dict


def _checkpointing_trainer(saved_paths: list[str]) -> Trainer3DGRUT:
    """Create a Trainer3DGRUT shell without initializing CUDA state."""
    trainer = object.__new__(Trainer3DGRUT)
    trainer.conf = OmegaConf.create(
        {
            "checkpoint": {
                "save_last_on_validation": False,
                "save_best_on_validation": True,
            },
            "early_stopping": {
                "enabled": True,
                "metric": "masked_psnr",
                "patience": 3,
                "min_delta": 0.03,
                "restore_best_on_end": True,
            },
        }
    )
    trainer.global_step = 22000
    trainer._should_stop_training = False
    trainer._best_validation_score = 26.408542533419027
    trainer._best_validation_step = 20000
    trainer._early_stopping_reference_score = 26.408542533419027
    trainer._early_stopping_reference_step = 20000
    trainer._stale_validation_count = 0
    trainer._best_checkpoint_path = "/tmp/previous_best.pt"

    def save_checkpoint(
        *,
        last_checkpoint: bool = False,
        best_checkpoint: bool = False,
    ) -> str:
        assert not last_checkpoint
        assert best_checkpoint
        saved_paths.append("/tmp/current_best.pt")
        return "/tmp/current_best.pt"

    setattr(trainer, "save_checkpoint", save_checkpoint)
    return trainer


def test_best_checkpoint_tracks_small_positive_validation_gain() -> None:
    """Small PSNR gains should save ckpt_best without resetting patience."""
    saved_paths: list[str] = []
    trainer = _checkpointing_trainer(saved_paths)
    metrics = {"masked_psnr": [26.435249677344935]}

    trainer._handle_validation_checkpointing(metrics)

    assert saved_paths == ["/tmp/current_best.pt"]
    assert trainer._best_checkpoint_path == "/tmp/current_best.pt"
    assert trainer._best_validation_score == pytest.approx(26.435249677344935)
    assert trainer._best_validation_step == 22000
    assert trainer._early_stopping_reference_score == pytest.approx(
        26.408542533419027
    )
    assert trainer._early_stopping_reference_step == 20000
    assert trainer._stale_validation_count == 1


def test_position_anchor_loss_penalizes_center_drift() -> None:
    """Position-anchor loss should be a robust mean center drift penalty."""
    trainer = object.__new__(Trainer3DGRUT)
    trainer.device = "cpu"
    trainer.conf = OmegaConf.create(
        {
            "loss": {
                "use_l1": False,
                "lambda_l1": 1.0,
                "use_l2": False,
                "lambda_l2": 1.0,
                "use_ssim": False,
                "lambda_ssim": 0.2,
                "use_foundation_feature": False,
                "lambda_foundation_feature": 0.0,
                "use_opacity": False,
                "lambda_opacity": 0.0,
                "use_scale": False,
                "lambda_scale": 0.0,
                "use_sky_opacity": False,
                "lambda_sky_opacity": 0.0,
                "use_position_anchor": True,
                "lambda_position_anchor": 0.003,
                "position_anchor_epsilon": 0.02,
                "use_equirect_consistency": False,
                "lambda_equirect_consistency": 0.0,
                "use_camera_loss_weights": False,
                "camera_loss_weights": [],
            }
        }
    )
    trainer.model = SimpleNamespace(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]]),
        position_anchor=torch.zeros((2, 3)),
    )
    gpu_batch = SimpleNamespace(
        rgb_gt=torch.zeros((1, 1, 1, 3)),
        mask=None,
    )
    outputs = {"pred_rgb": torch.zeros((1, 1, 1, 3))}

    losses = trainer.get_losses(gpu_batch, outputs)

    expected_raw = (
        torch.sqrt(torch.tensor(0.03 * 0.03 + 0.02 * 0.02)) - 0.02
    ) / 2.0
    expected_weighted = expected_raw * 0.003
    assert losses["position_anchor_loss_raw"].item() == pytest.approx(
        expected_raw.item()
    )
    assert losses["position_anchor_loss"].item() == pytest.approx(
        expected_weighted.item()
    )
    assert losses["total_loss"].item() == pytest.approx(
        expected_weighted.item()
    )


def test_checkpoint_load_repairs_mismatched_position_anchor() -> None:
    """Old checkpoints with stale anchors should still reload safely."""
    num_gaussians = 5
    max_n_features = 3
    positions = torch.arange(
        num_gaussians * 3,
        dtype=torch.float32,
    ).reshape(num_gaussians, 3)
    model = object.__new__(MixtureOfGaussians)
    model.progressive_training = False
    model.feature_type = "sh"
    model.max_n_features = max_n_features
    model.device = "cpu"
    model.background = _DummyBackground()

    checkpoint = {
        "positions": positions,
        "rotation": torch.zeros((num_gaussians, 4)),
        "scale": torch.ones((num_gaussians, 3)),
        "density": torch.ones((num_gaussians, 1)),
        "features_albedo": torch.zeros((num_gaussians, 3)),
        "features_specular": torch.zeros(
            (num_gaussians, sh_degree_to_specular_dim(max_n_features))
        ),
        "position_anchor": torch.zeros((2, 3)),
        "n_active_features": max_n_features,
        "max_n_features": max_n_features,
        "scene_extent": 1.0,
        "background": {},
    }

    model.init_from_checkpoint(checkpoint, setup_optimizer=False)

    assert model.position_anchor.shape == positions.shape
    assert torch.equal(model.position_anchor, positions)


def test_clone_gaussians_extends_position_anchor() -> None:
    """Densification must keep checkpoint-only anchor buffers aligned."""
    positions = torch.nn.Parameter(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
    )
    model = SimpleNamespace(
        positions=positions,
        rotation=torch.nn.Parameter(torch.zeros((3, 4))),
        scale=torch.nn.Parameter(torch.full((3, 3), 0.1)),
        density=torch.nn.Parameter(torch.ones((3, 1))),
        features_albedo=torch.nn.Parameter(torch.zeros((3, 3))),
        features_specular=torch.nn.Parameter(torch.zeros((3, 45))),
            position_anchor=torch.tensor(
                [
                    [10.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0],
                ]
            ),
            tangent_plane_normal_anchor=torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            device="cpu",
        )
    model.optimizer = torch.optim.Adam(
        [
            {"params": [model.positions], "name": "positions"},
            {"params": [model.rotation], "name": "rotation"},
            {"params": [model.scale], "name": "scale"},
            {"params": [model.density], "name": "density"},
            {"params": [model.features_albedo], "name": "features_albedo"},
            {
                "params": [model.features_specular],
                "name": "features_specular",
            },
        ],
        lr=0.01,
    )
    model.get_scale = lambda: model.scale
    model.get_positions = lambda: model.positions
    model.scale_activation = lambda value: value
    model.scale_activation_inv = lambda value: value
    model.density_activation = lambda value: value
    model.density_activation_inv = lambda value: value
    def reset_position_anchor() -> None:
        model.position_anchor = model.positions.detach().clone()
        reset_tangent_plane_normal_anchor()

    def reset_tangent_plane_normal_anchor() -> None:
        model.tangent_plane_normal_anchor = torch.zeros_like(
            model.positions.detach()
        )
        model.tangent_plane_normal_anchor[:, 2] = 1.0

    model.reset_position_anchor = reset_position_anchor
    model.reset_tangent_plane_normal_anchor = reset_tangent_plane_normal_anchor

    strategy = object.__new__(GSStrategy)
    strategy.model = model
    strategy.conf = SimpleNamespace(
        strategy=SimpleNamespace(print_stats=False),
    )
    strategy.clone_grad_threshold = 0.5
    strategy.relative_size_threshold = 1.0
    strategy.residual_density_control = {}
    strategy.densify_grad_norm_accum = torch.empty((0, 1))
    strategy.densify_grad_norm_denom = torch.empty((0, 1))
    strategy.densify_abs_grad_norm_accum = torch.empty((0, 1))
    strategy.densify_signed_grad_accum = torch.empty((0, 3))
    strategy.structure_axis_x_accum = torch.empty((0, 1))
    strategy.structure_axis_y_accum = torch.empty((0, 1))
    strategy.structure_local_axis_accum = torch.empty((0, 3))
    strategy.structure_axis_denom = torch.empty((0, 1), dtype=torch.int)
    strategy.sadgs_accum_eta = torch.empty((0, 1))

    strategy.clone_gaussians(
        torch.tensor([0.6, 0.1, 0.7]),
        scene_extent=1.0,
    )

    expected_anchor = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(model.position_anchor, expected_anchor)
    assert model.position_anchor.shape == model.positions.shape


def test_prior_anchor_field_injection_appends_anchored_gaussians() -> None:
    """Detached anchor fields should become optimizer-backed splats."""
    positions = torch.nn.Parameter(
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    model = SimpleNamespace(
        positions=positions,
        rotation=torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            )
        ),
        scale=torch.nn.Parameter(torch.full((2, 3), 0.1)),
        density=torch.nn.Parameter(torch.ones((2, 1))),
        features_albedo=torch.nn.Parameter(torch.zeros((2, 3))),
        features_specular=torch.nn.Parameter(torch.zeros((2, 6))),
        view_albedo_delta_logits=torch.nn.Parameter(torch.zeros((3, 2, 3))),
        view_density_delta_logits=torch.nn.Parameter(torch.zeros((3, 2, 1))),
        position_anchor=positions.detach().clone(),
        tangent_plane_normal_anchor=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        device="cpu",
    )
    model.optimizer = torch.optim.Adam(
        [
            {"params": [model.positions], "name": "positions"},
            {"params": [model.rotation], "name": "rotation"},
            {"params": [model.scale], "name": "scale"},
            {"params": [model.density], "name": "density"},
            {"params": [model.features_albedo], "name": "features_albedo"},
            {
                "params": [model.features_specular],
                "name": "features_specular",
            },
            {
                "params": [model.view_albedo_delta_logits],
                "name": "view_albedo_delta_logits",
            },
            {
                "params": [model.view_density_delta_logits],
                "name": "view_density_delta_logits",
            },
        ],
        lr=0.01,
    )
    model.get_positions = lambda: model.positions
    model.scale_activation_inv = lambda value: value
    model.density_activation_inv = lambda value: value

    def validate_fields() -> None:
        assert model.position_anchor.shape == model.positions.shape
        assert model.tangent_plane_normal_anchor.shape == model.positions.shape
        assert model.view_albedo_delta_logits.shape == (3, 4, 3)
        assert model.view_density_delta_logits.shape == (3, 4, 1)

    model.validate_fields = validate_fields

    strategy = object.__new__(GSStrategy)
    strategy.model = model
    strategy.conf = OmegaConf.create({"model": {"default_density": 2.0}})
    strategy.residual_density_control = {
        "use_prior_anchor_field": True,
        "prior_anchor_max_points": 2,
        "prior_anchor_min_confidence": 0.2,
        "prior_anchor_scale_m": 0.01,
        "prior_anchor_density_multiplier": 0.5,
    }
    strategy._prior_anchor_injected = False
    strategy.residual_density_stats = {}
    strategy.densify_grad_norm_accum = torch.empty((0, 1))
    strategy.densify_grad_norm_denom = torch.empty((0, 1), dtype=torch.int)
    strategy.densify_abs_grad_norm_accum = torch.empty((0, 1))
    strategy.densify_signed_grad_accum = torch.empty((0, 3))
    strategy.structure_axis_x_accum = torch.empty((0, 1))
    strategy.structure_axis_y_accum = torch.empty((0, 1))
    strategy.structure_local_axis_accum = torch.empty((0, 3))
    strategy.structure_axis_denom = torch.empty((0, 1), dtype=torch.int)
    strategy.sadgs_accum_eta = torch.empty((0, 1))

    with tempfile.TemporaryDirectory() as directory:
        positions_path = os.path.join(directory, "positions.npy")
        colors_path = os.path.join(directory, "colors.npy")
        confidence_path = os.path.join(directory, "confidence.npy")
        manifest_path = os.path.join(directory, "anchors.json")
        np.save(
            positions_path,
            np.asarray(
                [
                    [10.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0],
                    [40.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        np.save(
            colors_path,
            np.asarray(
                [
                    [0, 0, 0],
                    [255, 0, 0],
                    [0, 0, 255],
                    [0, 255, 0],
                ],
                dtype=np.uint8,
            ),
        )
        np.save(
            confidence_path,
            np.asarray([0.1, 0.9, 0.7, 0.5], dtype=np.float32),
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "coordinate_frame": "world",
                    "detached": True,
                    "positions_path": positions_path,
                    "colors_path": colors_path,
                    "confidence_path": confidence_path,
                },
                handle,
            )
        strategy.residual_density_control[
            "prior_anchor_field_manifest_path"
        ] = manifest_path

        assert strategy.inject_prior_anchor_field_once(0, writer=None)
        assert not strategy.inject_prior_anchor_field_once(1, writer=None)

    expected_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
        ]
    )
    assert torch.equal(model.positions.detach(), expected_positions)
    assert torch.equal(model.position_anchor, expected_positions)
    assert torch.equal(
        model.tangent_plane_normal_anchor[-2:],
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    )
    assert torch.equal(
        model.scale.detach()[-2:],
        torch.tensor([[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]),
    )
    assert torch.equal(model.density.detach()[-2:], torch.ones((2, 1)))
    assert model.features_albedo.detach()[-2:].abs().sum() > 0
    assert model.view_albedo_delta_logits.shape == (3, 4, 3)
    assert model.view_density_delta_logits.shape == (3, 4, 1)
    for group in model.optimizer.param_groups:
        assert group["params"][0] is getattr(model, group["name"])
    assert strategy.residual_density_stats["prior_anchor_injected_count"] == 2.0
    assert strategy.residual_density_stats["prior_anchor_valid_count"] == 3.0
