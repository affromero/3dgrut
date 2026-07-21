from __future__ import annotations

import copy
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from threedgrut.datasets.protocols import Batch
from threedgrut.model.factory import (
    GaussianRepresentation,
    checkpoint_representation,
)
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.view_conditioned_anchor_field import (
    VIEW_LOCAL_RESIDUAL_LAYER,
    ViewConditionedAnchorField,
    deterministic_voxel_anchors,
    load_fold_safe_point_source,
)
from threedgrut.strategy.fixed_anchor import FixedAnchorStrategy
from threedgrut.trainer import Trainer3DGRUT
from threedgrut.utils.misc import quaternion_to_so3


class _TestBackground(torch.nn.Module):
    def forward(
        self,
        ray_to_world: torch.Tensor,
        rays_d: torch.Tensor,
        rgb: torch.Tensor,
        opacity: torch.Tensor,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del ray_to_world, rays_d, train
        return rgb, opacity


class _RecordingTracer:
    def __init__(self, conf: DictConfig) -> None:
        del conf

    def render(
        self,
        view: object,
        batch: Batch,
        train: bool = False,
        frame_id: int = 0,
    ) -> dict[str, torch.Tensor]:
        del batch, train, frame_id
        return {
            "positions": getattr(view, "positions"),
            "rotations": getattr(view, "get_rotation")(),
            "scales": getattr(view, "get_scale")(),
            "densities": getattr(view, "get_density")(),
            "features": getattr(view, "get_features")(),
        }

    def build_acc(self, model, rebuild: bool = True) -> None:
        del model, rebuild


def _make_background(
    name: str,
    conf: DictConfig,
) -> _TestBackground:
    del name, conf
    return _TestBackground()


def _compose_anchor_config(manifest_path: str) -> DictConfig:
    config_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "configs",
        )
    )
    with initialize_config_dir(
        config_dir=config_dir,
        version_base=None,
    ):
        return compose(
            config_name="apps/colmap_3dgut",
            overrides=[
                "model.representation=view_conditioned_anchor",
                f"model.anchor_field.source_manifest_path={manifest_path}",
                "optimizer.type=adam",
                "camera_residual.enabled=false",
                "export_ply.enabled=false",
                "export_usd.enabled=false",
                "dataset.blur_samples=1",
                "dataset.rs_ray_injection=false",
                "with_gui=false",
                "with_viser_gui=false",
            ],
        )


def _write_anchor_source(tmp_path: os.PathLike[str]) -> str:
    directory = os.fspath(tmp_path)
    positions = np.array(
        (
            (0.01, 0.01, 0.01),
            (0.05, 0.04, 0.03),
            (0.30, 0.00, 0.00),
            (0.34, 0.02, 0.01),
        ),
        dtype=np.float32,
    )
    colors = np.array(
        (
            (255.0, 0.0, 0.0),
            (127.0, 0.0, 0.0),
            (0.0, 255.0, 0.0),
            (0.0, 127.0, 0.0),
        ),
        dtype=np.float32,
    )
    normals = np.array(
        (
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    layers = np.array((0.0, 0.0, 1.0, 1.0), dtype=np.float32)
    components = np.array((0.0, 0.0, 1.0, 1.0), dtype=np.float32)
    points_path = os.path.join(directory, "anchor_points.npy")
    np.save(
        points_path,
        np.concatenate(
            (
                positions,
                colors,
                normals,
                layers[:, None],
                components[:, None],
            ),
            axis=1,
        ),
        allow_pickle=False,
    )
    with open(points_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    manifest = {
        "schema_version": 4,
        "sealed_test_used": False,
        "geometry_input_contract": (
            "train_piecewise_surface_plane_vertices_plus_"
            "residual_surfel_centers"
        ),
        "fold_contract_sha256": "1" * 64,
        "training_image_names_sha256": "2" * 64,
        "source_image_names_sha256": "3" * 64,
        "images_txt_sha256": "6" * 64,
        "anchor_materializer_sha256": "d" * 64,
        "surface_mesh_sha256": "7" * 64,
        "surface_result_sha256": "8" * 64,
        "surface_result_schema_version": 1,
        "surface_result_status": "supported",
        "surface_development_rgb_used": False,
        "surface_development_depth_used_for_scoring_only": True,
        "surface_gates_sha256": (
            "2473b66c7abe63411b9f4983e39e71f547aec126ba4e93983933c57931f272bf"
        ),
        "surface_driver_sha256": "9" * 64,
        "visibility_result_sha256": "a" * 64,
        "visibility_preregistration_sha256": "b" * 64,
        "visibility_driver_sha256": "c" * 64,
        "view_local_clearance_radius_m": 1.0,
        "component_separated_voxelization": True,
        "points_file": os.path.basename(points_path),
        "points_format": (
            "npy_float32_xyz_train_rgb_normal_layer_component_v1"
        ),
        "points_sha256": digest,
        "point_count": int(positions.shape[0]),
        "plane_anchor_count": 2,
        "residual_anchor_count": 2,
        "voxel_size_m": 0.15,
        "color_fusion": {
            "sealed_test_used": False,
            "training_image_names_sha256": "2" * 64,
            "source_depth_provenance_sha256": "4" * 64,
            "training_payload_sha256": "5" * 64,
        },
    }
    manifest_path = os.path.join(directory, "anchor_source.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    return manifest_path


def _batch(
    camera_center: tuple[float, float, float],
    *,
    camera_idx: int,
    frame_idx: int,
    rays_in_world_space: bool = False,
) -> Batch:
    pose = torch.eye(4, device="cuda").unsqueeze(0)
    pose[0, :3, 3] = torch.tensor(camera_center, device="cuda")
    rays_ori = torch.zeros((1, 2, 2, 3), device="cuda")
    if rays_in_world_space:
        rays_ori[:] = torch.tensor(camera_center, device="cuda")
        pose = torch.eye(4, device="cuda").unsqueeze(0)
    return Batch(
        rays_ori=rays_ori,
        rays_dir=torch.ones((1, 2, 2, 3), device="cuda"),
        T_to_world=pose,
        T_to_world_end=pose.clone(),
        camera_idx=camera_idx,
        frame_idx=frame_idx,
        rays_in_world_space=rays_in_world_space,
    )


def test_voxel_anchor_reduction_is_input_order_invariant() -> None:
    points = torch.tensor(
        (
            (0.01, 0.01, 0.01),
            (0.04, 0.02, 0.03),
            (0.31, 0.01, 0.02),
            (0.34, 0.03, 0.04),
        )
    )
    colors = torch.tensor(
        (
            (255.0, 0.0, 0.0),
            (127.0, 0.0, 0.0),
            (0.0, 255.0, 0.0),
            (0.0, 127.0, 0.0),
        )
    )
    permutation = torch.tensor((2, 0, 3, 1))
    expected = deterministic_voxel_anchors(
        points,
        colors,
        voxel_size_m=0.15,
    )
    actual = deterministic_voxel_anchors(
        points[permutation],
        colors[permutation],
        voxel_size_m=0.15,
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_untagged_checkpoint_is_an_ordinary_mixture() -> None:
    assert (
        checkpoint_representation({"positions": torch.zeros(1, 3)})
        == GaussianRepresentation.MIXTURE
    )


def test_anchor_source_rejects_actual_training_name_mismatch(
    tmp_path: os.PathLike[str],
) -> None:
    manifest_path = _write_anchor_source(tmp_path)

    with pytest.raises(
        ValueError,
        match="actual filtered training dataset",
    ):
        load_fold_safe_point_source(
            manifest_path,
            expected_training_image_names=["front_0001.png"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", 3, "Unsupported anchor source"),
        (
            "surface_result_schema_version",
            2,
            "supported surface-result schema",
        ),
        (
            "surface_result_status",
            "rejected",
            "supported surface result",
        ),
        (
            "surface_development_rgb_used",
            True,
            "must not use development RGB",
        ),
        (
            "surface_development_depth_used_for_scoring_only",
            False,
            "scoring-only",
        ),
        (
            "surface_gates_sha256",
            "e" * 64,
            "surface-gate digest",
        ),
        (
            "surface_gates_sha256",
            None,
            "surface-gate digest",
        ),
    ),
)
def test_anchor_source_rejects_invalid_surface_acceptance_manifest(
    tmp_path: os.PathLike[str],
    field: str,
    value: object,
    message: str,
) -> None:
    """Portable source loading enforces the exact v4 surface acceptance."""
    manifest_path = _write_anchor_source(tmp_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest[field] = value
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match=message):
        load_fold_safe_point_source(manifest_path)


def test_anchor_source_rejects_non_unit_normals(
    tmp_path: os.PathLike[str],
) -> None:
    manifest_path = _write_anchor_source(tmp_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    points_path = os.path.join(
        os.path.dirname(manifest_path),
        manifest["points_file"],
    )
    payload = np.load(points_path, allow_pickle=False)
    payload[0, 6:9] = 0.0
    np.save(points_path, payload, allow_pickle=False)
    with open(points_path, "rb") as handle:
        manifest["points_sha256"] = hashlib.sha256(
            handle.read()
        ).hexdigest()
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="unit length"):
        load_fold_safe_point_source(manifest_path)


def test_anchor_source_rejects_unknown_layer(
    tmp_path: os.PathLike[str],
) -> None:
    manifest_path = _write_anchor_source(tmp_path)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    points_path = os.path.join(
        os.path.dirname(manifest_path),
        manifest["points_file"],
    )
    payload = np.load(points_path, allow_pickle=False)
    payload[0, 9] = 2.0
    np.save(points_path, payload, allow_pickle=False)
    with open(points_path, "rb") as handle:
        manifest["points_sha256"] = hashlib.sha256(
            handle.read()
        ).hexdigest()
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="unsupported layer"):
        load_fold_safe_point_source(manifest_path)


def test_anchor_representation_rejects_non_global_shutter(
    tmp_path: os.PathLike[str],
) -> None:
    conf = _compose_anchor_config(_write_anchor_source(tmp_path))
    conf.dataset.shutter_type = "ROLLING"

    with pytest.raises(ValueError, match="GLOBAL shutter"):
        ViewConditionedAnchorField(conf, scene_extent=1.0)


def test_anchor_representation_rejects_depth_loss(
    tmp_path: os.PathLike[str],
) -> None:
    conf = _compose_anchor_config(_write_anchor_source(tmp_path))
    conf.loss.use_depth = True

    with pytest.raises(ValueError, match="image-only"):
        ViewConditionedAnchorField(conf, scene_extent=1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_view_conditioning_ignores_discrete_camera_and_frame_ids(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    manifest_path = _write_anchor_source(tmp_path)
    model = ViewConditionedAnchorField(
        _compose_anchor_config(manifest_path),
        scene_extent=1.0,
    )
    model.initialize_from_source_manifest()

    first = model(
        _batch((1.0, 0.0, 0.0), camera_idx=0, frame_idx=1),
        train=True,
    )
    same_pose_different_ids = model(
        _batch((1.0, 0.0, 0.0), camera_idx=19, frame_idx=991),
        train=True,
    )
    changed_pose = model(
        _batch((0.0, 1.0, 0.0), camera_idx=0, frame_idx=1),
        train=True,
    )

    torch.testing.assert_close(
        first["features"],
        same_pose_different_ids["features"],
    )
    torch.testing.assert_close(
        first["positions"],
        changed_pose["positions"],
    )
    assert first["features"].shape == (16, 48)
    assert torch.count_nonzero(first["features"][:, 3:]) == 0
    assert not torch.equal(first["features"], changed_pose["features"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_world_ray_origin_supplies_continuous_camera_center(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-baked global rays retain the physical camera center."""
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    model = ViewConditionedAnchorField(
        _compose_anchor_config(_write_anchor_source(tmp_path)),
        scene_extent=1.0,
    )
    model.initialize_from_source_manifest()

    camera_space = model(
        _batch((1.0, 0.0, 0.0), camera_idx=0, frame_idx=0),
        train=False,
    )
    world_space = model(
        _batch(
            (1.0, 0.0, 0.0),
            camera_idx=7,
            frame_idx=19,
            rays_in_world_space=True,
        ),
        train=False,
    )

    torch.testing.assert_close(
        world_space["features"],
        camera_space["features"],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_anchor_decoder_gradients_and_checkpoint_round_trip(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    manifest_path = _write_anchor_source(tmp_path)
    conf = _compose_anchor_config(manifest_path)
    conf.model.anchor_field.use_view_local_residual_gate = True
    conf.model.anchor_field.dc_mode = "bounded_delta"
    conf.model.anchor_field.covariance_mode = "all_surface"
    model = ViewConditionedAnchorField(conf, scene_extent=1.0)
    model.initialize_from_source_manifest()
    model.setup_optimizer()
    batch = _batch((1.0, 0.0, 0.0), camera_idx=0, frame_idx=0)
    output = model(batch, train=True)
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()

    assert model.anchor_features.grad is not None
    assert float(model.anchor_features.grad.abs().max()) > 0.0
    assert model.geometry_decoder[2].weight.grad is not None
    assert float(model.geometry_decoder[2].weight.grad.abs().max()) > 0.0
    assert model.appearance_decoder[2].weight.grad is not None
    assert float(model.appearance_decoder[2].weight.grad.abs().max()) > 0.0
    model.optimizer.step()

    checkpoint = model.get_model_parameters()
    assert checkpoint["optimizer"]["state"]
    restored = ViewConditionedAnchorField(conf, scene_extent=9.0)
    restored.init_from_checkpoint(checkpoint)
    expected = model(batch, train=False)
    actual = restored(batch, train=False)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    with pytest.raises(RuntimeError, match="explicit bake"):
        restored.export_ply("forbidden.ply")
    invalid_checkpoint = copy.deepcopy(checkpoint)
    invalid_checkpoint["anchor_source_manifest"][
        "surface_result_status"
    ] = "rejected"
    rejected = ViewConditionedAnchorField(conf, scene_extent=9.0)
    with pytest.raises(ValueError, match="supported surface result"):
        rejected.init_from_checkpoint(invalid_checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_residual_gate_uses_immutable_anchor_position(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    conf = _compose_anchor_config(_write_anchor_source(tmp_path))
    conf.model.anchor_field.use_view_local_residual_gate = True
    model = ViewConditionedAnchorField(conf, scene_extent=1.0)
    model.initialize_from_source_manifest()

    near = model(
        _batch((0.0, 0.0, 0.0), camera_idx=2, frame_idx=7),
        train=False,
    )
    far = model(
        _batch((2.0, 0.0, 0.0), camera_idx=9, frame_idx=99),
        train=False,
    )
    exact_center = model.anchor_positions[2].detach().clone()
    exact_center[0] += 1.0
    exact = model._materialize(exact_center)

    near_density = near["densities"].reshape(model.num_anchors, 4)
    far_density = far["densities"].reshape(model.num_anchors, 4)
    exact_density = exact.get_density().reshape(model.num_anchors, 4)
    assert torch.all(near_density[:2] > 0.0)
    assert torch.count_nonzero(near_density[2:]) == 0
    assert torch.all(far_density[2:] > 0.0)
    assert torch.count_nonzero(exact_density[2]) == 0
    assert model.num_gaussians == 16
    assert torch.all(
        model.anchor_layers[2:] == VIEW_LOCAL_RESIDUAL_LAYER
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_surface_covariance_uses_surface_normal(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    conf = _compose_anchor_config(_write_anchor_source(tmp_path))
    conf.model.anchor_field.covariance_mode = "all_surface"
    model = ViewConditionedAnchorField(conf, scene_extent=1.0)
    model.initialize_from_source_manifest()

    rotations = model.get_rotation().reshape(model.num_anchors, 4, 4)
    rotation_matrices = quaternion_to_so3(rotations[:, 0])
    normal_axis = rotation_matrices[:, :, 2]
    scales = model.get_scale().reshape(model.num_anchors, 4, 3)

    torch.testing.assert_close(
        normal_axis,
        model.anchor_normals,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        scales[..., 2],
        torch.full_like(scales[..., 2], 0.003),
    )
    assert torch.all(scales[..., :2] > scales[..., 2:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bounded_dc_delta_changes_only_radiance(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    conf = _compose_anchor_config(_write_anchor_source(tmp_path))
    conf.model.anchor_field.dc_mode = "bounded_delta"
    conf.model.anchor_field.optimization_mode = "dc_only"
    model = ViewConditionedAnchorField(conf, scene_extent=1.0)
    model.initialize_from_source_manifest()
    model.setup_optimizer()
    camera_center = torch.tensor((1.0, 0.0, 0.0), device="cuda")
    before = model._materialize(camera_center)
    geometry_before = tuple(
        value.detach().clone()
        for value in (
            before.positions,
            before.get_rotation(),
            before.get_scale(),
            before.get_density(),
        )
    )
    features_before = before.get_features().detach().clone()

    loss = -before.get_features()[:, 0].mean()
    loss.backward()
    model.optimizer.step()
    after = model._materialize(camera_center)

    assert model.anchor_dc_delta_raw.grad is not None
    assert float(model.anchor_dc_delta_raw.grad.abs().max()) > 0.0
    for parameter in model.geometry_decoder.parameters():
        assert parameter.grad is None
    for parameter in model.appearance_decoder.parameters():
        assert parameter.grad is None
    assert model.anchor_features.grad is None
    for expected, actual in zip(
        geometry_before,
        (
            after.positions,
            after.get_rotation(),
            after.get_scale(),
            after.get_density(),
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert not torch.equal(after.get_features(), features_before)
    assert bool(torch.isfinite(after.get_features()).all())
    assert float(
        torch.tanh(model.anchor_dc_delta_raw).detach().abs().max()
    ) <= 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hybrid_covariance_preserves_isotropic_coverage(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    conf = _compose_anchor_config(_write_anchor_source(tmp_path))
    conf.model.anchor_field.covariance_mode = "hybrid_surface"
    model = ViewConditionedAnchorField(conf, scene_extent=1.0)
    model.initialize_from_source_manifest()
    with torch.no_grad():
        model.geometry_decoder[2].bias[3:6] = torch.tensor(
            (-4.0, -3.0, -2.0),
            device="cuda",
        )

    rotations = model.get_rotation().reshape(model.num_anchors, 4, 4)
    rotation_matrices = quaternion_to_so3(
        rotations[:, 1:].reshape(-1, 4)
    ).reshape(model.num_anchors, 3, 3, 3)
    detail_normal_axis = rotation_matrices[..., 2]
    scales = model.get_scale().reshape(model.num_anchors, 4, 3)

    torch.testing.assert_close(
        scales[:, 0, 0],
        scales[:, 0, 1],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        scales[:, 0, 1],
        scales[:, 0, 2],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        detail_normal_axis,
        model.anchor_normals[:, None, :].expand(-1, 3, -1),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        scales[:, 1:, 2],
        torch.full_like(scales[:, 1:, 2], 0.003),
    )


def test_fixed_anchor_strategy_never_mutates_topology() -> None:
    model = SimpleNamespace(num_gaussians=3)
    strategy = FixedAnchorStrategy(
        {"method": "FixedAnchorStrategy"},
        model,
    )
    strategy.init_densification_buffer()
    assert strategy.pre_backward(1, 1.0, object()) is False
    assert strategy.post_backward(1, 1.0, object()) is False
    assert strategy.post_optimizer_step(1, 1.0, object()) is False
    assert strategy.get_strategy_parameters() == {
        "strategy_state": {
            "name": "FixedAnchorStrategy",
            "version": 2,
            "expected_gaussian_count": 3,
        }
    }
    strategy.init_densification_buffer(
        strategy.get_strategy_parameters()
    )
    with pytest.raises(
        ValueError,
        match="incompatible strategy state",
    ):
        strategy.init_densification_buffer(
            {
                "strategy_state": {
                    "name": "FixedAnchorStrategy",
                    "version": 1,
                    "expected_gaussian_count": 3,
                }
            }
        )
    model.num_gaussians = 2
    with pytest.raises(
        RuntimeError,
        match="Gaussian count changed",
    ):
        strategy.post_optimizer_step(2, 1.0, object())


def test_fixed_anchor_strategy_captures_count_after_initialization() -> None:
    model = SimpleNamespace(num_gaussians=0)
    strategy = FixedAnchorStrategy(
        {"method": "FixedAnchorStrategy"},
        model,
    )
    model.num_gaussians = 3
    strategy.init_densification_buffer()

    assert strategy.get_strategy_parameters()["strategy_state"][
        "expected_gaussian_count"
    ] == 3


def test_fixed_anchor_strategy_accepts_ordinary_gaussian_mixture() -> None:
    """A mixture can isolate optimizer behavior without topology mutation."""
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(trainer.model)
    trainer.model.positions = torch.nn.Parameter(torch.zeros((3, 3)))
    trainer.conf = OmegaConf.create(
        {"strategy": {"method": "FixedAnchorStrategy"}}
    )

    trainer.init_densification_and_pruning_strategy(trainer.conf)
    trainer.strategy.init_densification_buffer()

    assert isinstance(trainer.strategy, FixedAnchorStrategy)
    assert trainer.strategy.model is trainer.model


def test_anchor_representation_rejects_mutating_strategy() -> None:
    """The anchor field cannot silently enter Gaussian topology mutation."""
    trainer = Trainer3DGRUT.__new__(Trainer3DGRUT)
    trainer.model = ViewConditionedAnchorField.__new__(
        ViewConditionedAnchorField
    )
    trainer.conf = OmegaConf.create(
        {"strategy": {"method": "GSStrategy"}}
    )

    with pytest.raises(
        ValueError,
        match="requires FixedAnchorStrategy",
    ):
        trainer.init_densification_and_pruning_strategy(trainer.conf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_anchor_scales_obey_frequency_preserving_bounds(
    tmp_path: os.PathLike[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field."
        "threedgut_tracer.Tracer",
        _RecordingTracer,
    )
    monkeypatch.setattr(
        "threedgrut.model.view_conditioned_anchor_field.background.make",
        _make_background,
    )
    conf = _compose_anchor_config(_write_anchor_source(tmp_path))
    model = ViewConditionedAnchorField(conf, scene_extent=1.0)
    model.initialize_from_source_manifest()

    scales = model.get_scale()

    detached_scales = scales.detach().reshape(-1, 4, 3)
    assert float(detached_scales.min()) >= 0.003 - 1.0e-7
    assert float(detached_scales[:, 0, :].max()) <= 0.1125 + 1.0e-7
    assert float(detached_scales[:, 1:, :].max()) <= 0.0525 + 1.0e-7
    densities = model.get_density().detach().reshape(-1, 4)
    assert float(densities[:, 0].max()) <= 0.12 + 1.0e-7
    assert float(densities[:, 1:].max()) < 1.0
