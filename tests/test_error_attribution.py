"""Tests for held-out Gaussian error attribution and PLY recoloring."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from render_error_splats import (
    DEFAULT_OWNERSHIP_SUPPORT_THRESHOLD,
    DEFAULT_VISIBILITY_THRESHOLD,
    MANIFEST_SCHEMA_VERSION,
    M8_RECONSTRUCTION_MANIFEST_SCHEMA_VERSION,
    _counterfactual_cohorts,
    _manifest_mode_contract,
    _suppressed_density_cohort,
    _view_index_contract,
    _write_raw_field,
)
from threedgrut.error_attribution import (
    ErrorAttributionAccumulator,
    ErrorAttributionMetric,
    ErrorAttributionParameter,
    attribution_components,
    attribution_loss,
    camera_support_indicators,
    heldout_ownership_dominance,
    native_contributor_ray_fields,
    native_render_evidence_maps,
    native_structural_gaussian_fields,
    ownership_weighted_mean,
    recolor_gaussian_ply,
)
from threedgrut.export_utils import scene_relative_path
from threedgrut.model.losses import ssim
from threedgrut.split_membership import (
    dataset_membership,
    training_membership_provenance,
    use_dataset_membership,
)


def test_native_direct_views_can_cover_all_attribution_views() -> None:
    """M8 can estimate direct fields on all views while probing eight."""
    attribution, direct = _view_index_contract(
        count=17,
        attribution_maximum=8,
        native_maximum=0,
    )

    assert attribution == {0, 2, 5, 7, 9, 11, 14, 16}
    assert direct == set(range(17))


def test_native_direct_views_follow_legacy_sampling_when_unspecified() -> None:
    """Existing callers retain the prior single-subset behavior."""
    attribution, direct = _view_index_contract(
        count=17,
        attribution_maximum=8,
        native_maximum=None,
    )

    assert direct == attribution


def test_error_splat_manifest_paths_are_scene_relative(
    tmp_path: os.PathLike[str],
) -> None:
    """Portable manifests reject checkpoint paths outside the scene."""
    scene_root = os.path.join(tmp_path, "scene")
    checkpoint = os.path.join(scene_root, "generated_files", "ckpt.pt")

    assert scene_relative_path(checkpoint, scene_root) == os.path.join(
        "generated_files",
        "ckpt.pt",
    )
    with pytest.raises(ValueError, match="escapes --scene-root"):
        scene_relative_path(os.path.join(tmp_path, "outside.pt"), scene_root)


class _TwoPixelModel(torch.nn.Module):
    """Minimal row-aligned parameter surface for attribution tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features_albedo = torch.nn.Parameter(torch.zeros((1, 1)))


class _DensityModel(torch.nn.Module):
    """Minimal density parameter surface for intervention tests."""

    def __init__(self) -> None:
        super().__init__()
        self.density = torch.nn.Parameter(torch.tensor([[1.0], [2.0], [3.0]]))


class _IndependentPixelModel(torch.nn.Module):
    """Two row-aligned Gaussian parameters with independent image pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.features_albedo = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))


def test_manifest_contract_records_threshold_and_raw_field_version() -> None:
    """New exports use the explicit support and raw-field manifest contract."""
    assert MANIFEST_SCHEMA_VERSION == 5
    assert M8_RECONSTRUCTION_MANIFEST_SCHEMA_VERSION == 6
    assert DEFAULT_OWNERSHIP_SUPPORT_THRESHOLD > 0.0


def test_m8_reconstruction_mode_requires_ordered_training_subset() -> None:
    assert _manifest_mode_contract(
        m8_reconstruction_fields=True,
        selected_images=["a.png", "c.png"],
        training_support_images=["a.png", "b.png", "c.png"],
    ) == (6, "m8_reconstruction_policy_fields")

    with pytest.raises(ValueError, match="ordered subset"):
        _manifest_mode_contract(
            m8_reconstruction_fields=True,
            selected_images=["c.png", "a.png"],
            training_support_images=["a.png", "b.png", "c.png"],
        )


def test_heldout_mode_rejects_training_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _manifest_mode_contract(
            m8_reconstruction_fields=False,
            selected_images=["a.png"],
            training_support_images=["a.png"],
        )


def test_training_support_temporarily_restores_checkpoint_membership() -> None:
    """Repair-view overrides cannot leak into training-support exports."""
    dataset = SimpleNamespace(
        holdout_image_list_path="repair.txt",
        train_exclude_image_list_path=None,
        test_split_interval=0,
    )
    dataset.get = lambda key: getattr(dataset, key, None)
    conf = SimpleNamespace(dataset=dataset)
    checkpoint_membership = {
        "holdout_image_list_path": None,
        "train_exclude_image_list_path": None,
        "test_split_interval": 8,
    }

    with use_dataset_membership(conf, checkpoint_membership):
        assert dataset_membership(conf) == checkpoint_membership

    assert dataset.holdout_image_list_path == "repair.txt"
    assert dataset.test_split_interval == 0
    assert training_membership_provenance(checkpoint_membership) == {
        "source": "checkpoint_test_split_interval",
        **checkpoint_membership,
    }


def test_raw_field_round_trips_row_aligned_float32(
    tmp_path: os.PathLike[str],
) -> None:
    """Raw sidecars preserve scientific values independently of PLY color."""
    metadata = _write_raw_field(
        output_dir=str(tmp_path),
        field_id="mse__density",
        scores=torch.tensor([[1.25], [2.5]], dtype=torch.float64),
    )

    values = np.load(os.path.join(tmp_path, metadata["raw_filename"]))
    assert values.dtype == np.float32
    assert values.tolist() == pytest.approx([1.25, 2.5])
    assert metadata["raw_shape"] == [2]
    assert len(metadata["raw_sha256"]) == 64


def test_error_splat_export_preserves_full_scene_by_default() -> None:
    """Default error colors retain every source Gaussian's opacity."""
    assert DEFAULT_VISIBILITY_THRESHOLD == 0.0


def test_counterfactual_cohorts_have_ranked_and_deterministic_controls() -> (
    None
):
    """Counterfactual controls retain the requested cohort size."""
    scores = torch.tensor([0.2, 0.9, 0.1, 0.5])

    cohorts = _counterfactual_cohorts(scores, cohort_size=2)

    assert cohorts["top_density_sensitivity"].tolist() == [3, 1]
    assert cohorts["low_density_sensitivity"].tolist() == [2, 0]
    assert cohorts["random_control"].numel() == 2
    assert torch.equal(
        cohorts["random_control"],
        _counterfactual_cohorts(
            scores,
            cohort_size=2,
        )["random_control"],
    )


def test_density_suppression_is_restored_after_intervention() -> None:
    """A counterfactual export does not leave the trained model mutated."""
    model = _DensityModel()
    original = model.density.detach().clone()

    with _suppressed_density_cohort(
        model,
        torch.tensor([0, 2]),
        suppression_logit=-20.0,
    ):
        assert torch.equal(
            model.density.detach().reshape(-1),
            torch.tensor([-20.0, 2.0, -20.0]),
        )

    assert torch.equal(model.density.detach(), original)


def test_masked_mae_excludes_invalid_pixels() -> None:
    """MAE ignores pixels excluded by the scanner validity mask."""
    prediction = torch.tensor([[[[1.0, 0.5, 0.0], [1.0, 1.0, 1.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0], [0.0]]]])

    loss = attribution_loss(
        ErrorAttributionMetric.MAE,
        prediction,
        target,
        mask,
    )

    assert float(loss) == pytest.approx(0.5)


def test_mse_components_reduce_rgb_to_one_scalar_per_valid_pixel() -> None:
    """The MSE sensitivity domain contains pixels, not RGB channel entries."""
    prediction = torch.tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[[1.0], [0.0]]]])

    components, component_count = attribution_components(
        ErrorAttributionMetric.MSE,
        prediction,
        target,
        mask,
    )

    assert components.shape == (1, 1, 2, 1)
    assert components.reshape(-1).tolist() == pytest.approx([14.0 / 3.0, 0.0])
    assert component_count == pytest.approx(1.0)
    assert attribution_loss(
        ErrorAttributionMetric.MSE,
        prediction,
        target,
        mask,
    ).item() == pytest.approx(14.0 / 3.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused SSIM requires CUDA")
def test_registered_loss_matches_frozen_training_image_objective() -> None:
    """Registered sensitivity uses the exact scalar training image loss."""
    prediction = torch.full((1, 12, 13, 3), 0.5, device="cuda")
    target = torch.zeros_like(prediction)

    components, component_count = attribution_components(
        ErrorAttributionMetric.REGISTERED_LOSS,
        prediction,
        target,
        None,
    )

    expected = 0.8 * (prediction - target).abs().mean() + 0.2 * (
        1.0
        - ssim(
            prediction.permute(0, 3, 1, 2),
            target.permute(0, 3, 1, 2),
        )
    )
    assert components.shape == (1,)
    assert component_count == pytest.approx(1.0)
    assert torch.isfinite(components).all()
    assert components.item() == pytest.approx(expected.item())
    assert attribution_loss(
        ErrorAttributionMetric.REGISTERED_LOSS,
        prediction,
        target,
        None,
    ).item() == pytest.approx(expected.item())


def test_registered_loss_rejects_empty_mask() -> None:
    """Registered-loss sensitivity abstains without reconstruction pixels."""
    image = torch.zeros((1, 12, 12, 3))

    with pytest.raises(ValueError, match="no valid pixels"):
        attribution_components(
            ErrorAttributionMetric.REGISTERED_LOSS,
            image,
            image,
            torch.zeros((1, 12, 12, 1)),
        )


def test_registered_loss_component_metadata_is_serializable() -> None:
    """The registered objective carries its sealed component domain."""
    accumulator = ErrorAttributionAccumulator(
        model=_TwoPixelModel(),
        metrics=(ErrorAttributionMetric.REGISTERED_LOSS,),
        parameters=(ErrorAttributionParameter.APPEARANCE,),
    )

    assert accumulator.component_metadata() == {
        "registered_loss": {
            "domain": "scalar_0.8_l1_plus_0.2_valid_padding_dssim",
            "per_view_effective_counts": [],
            "estimator": (
                "sqrt(mean_views(squared_scalar_loss_gradient_norm))"
            ),
            "uncertainty": "none_exact_gradient",
        }
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused SSIM requires CUDA")
def test_registered_loss_sensitivity_is_exact_scalar_gradient() -> None:
    """Registered-loss sensitivity has no probe approximation or uncertainty."""
    model = _TwoPixelModel().cuda()
    prediction = model.features_albedo.reshape(1, 1, 1, 1).expand(
        1, 12, 12, 3
    )
    target = torch.full_like(prediction, 0.25)
    accumulator = ErrorAttributionAccumulator(
        model=model,
        metrics=(ErrorAttributionMetric.REGISTERED_LOSS,),
        parameters=(ErrorAttributionParameter.APPEARANCE,),
        probe_count=8,
    )

    loss = attribution_loss(
        ErrorAttributionMetric.REGISTERED_LOSS,
        prediction,
        target,
        None,
    )
    expected_gradient = torch.autograd.grad(
        loss,
        model.features_albedo,
        retain_graph=True,
    )[0].abs()
    accumulator.accumulate(
        outputs={"pred_rgb": prediction},
        target=target,
        mask=None,
    )

    assert torch.allclose(
        accumulator.rms_scores()["registered_loss:sh_dc_rgb"],
        expected_gradient.reshape(1).cpu(),
    )
    assert accumulator.rms_standard_errors() == {}


def test_empty_pixel_mask_is_rejected() -> None:
    """A sensitivity field cannot assign a score with no valid pixels."""
    image = torch.zeros((1, 1, 1, 3))

    with pytest.raises(ValueError, match="no valid pixels"):
        attribution_components(
            ErrorAttributionMetric.MSE,
            image,
            image,
            torch.zeros((1, 1, 1, 1)),
        )


def test_low_frequency_fraction_is_larger_for_smooth_residual() -> None:
    """The low-frequency Doctor ratio responds more to smooth residual."""
    target = torch.zeros((1, 32, 32, 3))
    smooth = torch.ones_like(target) * 0.2
    checker = torch.zeros_like(target)
    checker[:, ::2, ::2] = 0.2
    checker[:, 1::2, 1::2] = 0.2

    smooth_loss = attribution_loss(
        ErrorAttributionMetric.LOW_FREQUENCY,
        smooth,
        target,
        None,
    )
    checker_loss = attribution_loss(
        ErrorAttributionMetric.LOW_FREQUENCY,
        checker,
        target,
        None,
    )

    assert float(smooth_loss) > float(checker_loss)


def test_component_probes_do_not_cancel_opposing_pixel_gradients() -> None:
    """Opposing pixel derivatives retain positive attribution energy."""
    model = _TwoPixelModel()
    prediction = model.features_albedo.reshape(1, 1, 1, 1).expand(1, 1, 2, 1)
    target = torch.tensor([[[[-1.0], [1.0]]]])
    accumulator = ErrorAttributionAccumulator(
        model=model,
        metrics=(ErrorAttributionMetric.MSE,),
        parameters=(ErrorAttributionParameter.APPEARANCE,),
        probe_count=16,
    )

    losses = accumulator.accumulate(
        outputs={"pred_rgb": prediction},
        target=target,
        mask=None,
    )

    assert losses["mse"] == pytest.approx(1.0)
    assert accumulator.rms_scores()["mse:sh_dc_rgb"].item() > 0.0


def test_hutchinson_score_matches_exact_local_jacobian_energy() -> None:
    """RMS scores match exhaustive local gradients for independent pixels."""
    model = _IndependentPixelModel()
    prediction = model.features_albedo.reshape(1, 1, 2, 1).expand(
        -1, -1, -1, 3
    )
    target = torch.zeros_like(prediction)
    accumulator = ErrorAttributionAccumulator(
        model=model,
        metrics=(ErrorAttributionMetric.MSE,),
        parameters=(ErrorAttributionParameter.APPEARANCE,),
        probe_count=2,
    )

    accumulator.accumulate(
        outputs={"pred_rgb": prediction},
        target=target,
        mask=None,
    )

    expected = torch.tensor([2.0**0.5, 8.0**0.5])
    assert torch.allclose(
        accumulator.rms_scores()["mse:sh_dc_rgb"],
        expected,
    )
    metadata = accumulator.component_metadata()["mse"]
    assert metadata["domain"] == "valid_rgb_mean_pixel"
    assert metadata["per_view_effective_counts"] == [2.0]


def test_probe_standard_error_reports_stochastic_uncertainty() -> None:
    """Probe uncertainty is available when at least two probes were used."""
    model = _TwoPixelModel()
    prediction = model.features_albedo.reshape(1, 1, 1, 1).expand(1, 1, 2, 3)
    target = torch.tensor([[[[-1.0] * 3, [1.0] * 3]]])
    accumulator = ErrorAttributionAccumulator(
        model=model,
        metrics=(ErrorAttributionMetric.MSE,),
        parameters=(ErrorAttributionParameter.APPEARANCE,),
        probe_count=16,
    )

    accumulator.accumulate(
        outputs={"pred_rgb": prediction},
        target=target,
        mask=None,
    )

    standard_error = accumulator.rms_standard_errors()["mse:sh_dc_rgb"]
    assert standard_error.shape == (1,)
    assert standard_error.item() > 0.0


def test_native_evidence_recovers_conditional_depth_variance() -> None:
    """Second moments yield variance only where native alpha has support."""
    evidence = native_render_evidence_maps(
        accumulated_alpha=torch.tensor([[[[0.5], [0.0]]]]),
        depth_moment=torch.tensor([[[[1.0], [0.0]]]]),
        depth_squared_moment=torch.tensor([[[[5.0], [0.0]]]]),
        hit_count=torch.tensor([[[[2.0], [0.0]]]]),
    )

    assert evidence["accumulated_alpha"][0, 0, 0, 0].item() == pytest.approx(
        0.5
    )
    assert evidence["expected_depth"][0, 0, 0, 0].item() == pytest.approx(2.0)
    assert evidence["depth_variance"][0, 0, 0, 0].item() == pytest.approx(6.0)
    assert torch.isnan(evidence["expected_depth"][0, 0, 1, 0])


def test_native_contributor_fields_zero_unsupported_depth_variance() -> None:
    """A 3D ray exposure omits undefined conditional-depth pixels."""
    fields = native_contributor_ray_fields(
        accumulated_alpha=torch.tensor([[[[0.5], [0.0]]]]),
        depth_variance=torch.tensor([[[[6.0], [torch.nan]]]]),
        hit_count=torch.tensor([[[[2.0], [0.0]]]]),
    )

    assert torch.equal(
        fields["heldout_native_ownership"],
        torch.ones((1, 1, 2, 1)),
    )
    assert torch.equal(
        fields["depth_ambiguity_exposure"],
        torch.tensor([[[[6.0], [0.0]]]]),
    )
    assert torch.equal(
        fields["hit_congestion_exposure"],
        torch.tensor([[[[2.0], [0.0]]]]),
    )


def test_structural_fields_separate_spacing_and_covariance_overlap() -> None:
    """Nearest-centre fields expose footprint and support overlap separately."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    covariance = torch.eye(3).repeat(2, 1, 1) * 0.01
    scales = torch.tensor([[0.2, 0.1, 0.1], [0.2, 0.1, 0.1]])

    fields = native_structural_gaussian_fields(
        positions=positions,
        covariances=covariance,
        physical_scales=scales,
    )

    assert torch.allclose(
        fields["scale_to_neighbor_spacing"],
        torch.tensor([[2.0], [2.0]]),
    )
    assert torch.all(fields["nearest_covariance_overlap"] > 0.7)


def test_camera_support_threshold_is_resolution_normalized() -> None:
    """Supporting-camera decisions use mean ownership per rendered ray."""
    responsibility = torch.tensor([[0.0], [0.2], [2.0]])

    nonzero, supporting = camera_support_indicators(
        responsibility=responsibility,
        ray_count=100,
        ownership_support_threshold=0.01,
    )

    assert torch.equal(nonzero, torch.tensor([[0.0], [1.0], [1.0]]))
    assert torch.equal(supporting, torch.tensor([[0.0], [0.0], [1.0]]))


def test_ownership_weighted_mean_has_explicit_validity() -> None:
    """Mean congestion distinguishes zero ownership from a measured zero."""
    mean, valid = ownership_weighted_mean(
        weighted_exposure=torch.tensor([[6.0], [0.0]]),
        ownership=torch.tensor([[2.0], [0.0]]),
    )

    assert torch.equal(mean, torch.tensor([[3.0], [0.0]]))
    assert torch.equal(valid, torch.tensor([[1.0], [0.0]]))


def test_heldout_ownership_dominance_is_bounded_and_zero_without_support() -> (
    None
):
    """Held-out dominance cannot be inflated by a zero training denominator."""
    score = heldout_ownership_dominance(
        heldout_ownership=torch.tensor([[2.0], [0.0], [0.0]]),
        training_ownership=torch.tensor([[2.0], [3.0], [0.0]]),
    )

    assert torch.allclose(score, torch.tensor([[0.5], [0.0], [0.0]]))


def test_recolor_preserves_geometry_and_zeros_view_dependent_color(
    tmp_path: os.PathLike[str],
) -> None:
    """PLY recoloring changes SH only and retains Gaussian geometry."""
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("f_rest_0", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    vertices = np.zeros(2, dtype=dtype)
    geometry_fields = (
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    for index, field in enumerate(geometry_fields, start=1):
        vertices[field] = [float(index), float(index) + 0.5]
    vertices["f_rest_0"] = 4.0
    vertices["opacity"] = [1.0, 2.0]
    source = os.path.join(tmp_path, "source.ply")
    output = os.path.join(tmp_path, "error.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(source)

    statistics = recolor_gaussian_ply(
        source_path=str(source),
        output_path=str(output),
        scores=torch.tensor([0.0, 2.0]),
    )

    recolored = PlyData.read(output)["vertex"]
    for field in geometry_fields:
        assert np.array_equal(recolored[field], vertices[field])
    assert np.asarray(recolored["opacity"]).tolist() == [1.0, 2.0]
    assert np.asarray(recolored["f_rest_0"]).tolist() == [0.0, 0.0]
    assert recolored["f_dc_0"][0] != recolored["f_dc_0"][1]
    assert statistics["positive_fraction"] == pytest.approx(0.5)


def test_recolor_rejects_source_density_row_order_mismatch(
    tmp_path: os.PathLike[str],
) -> None:
    """Checkpoint positions authenticate the source PLY row ordering."""
    vertices = np.zeros(
        2,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
        ],
    )
    vertices["x"] = [1.0, 2.0]
    vertices["opacity"] = [1.0, 2.0]
    source = os.path.join(tmp_path, "source.ply")
    output = os.path.join(tmp_path, "error.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(source)

    with pytest.raises(ValueError, match="row order"):
        recolor_gaussian_ply(
            source_path=source,
            output_path=output,
            scores=torch.ones(2),
            expected_density=torch.tensor([[2.0], [1.0]]),
        )


def test_recolor_hides_negligible_attribution(
    tmp_path: os.PathLike[str],
) -> None:
    """Diagnostic PLYs suppress low-error seed structure by opacity."""
    vertices = np.zeros(
        3,
        dtype=[
            ("x", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
        ],
    )
    vertices["opacity"] = 2.0
    source = os.path.join(tmp_path, "source.ply")
    output = os.path.join(tmp_path, "error.ply")
    PlyData([PlyElement.describe(vertices, "vertex")]).write(source)

    statistics = recolor_gaussian_ply(
        source_path=source,
        output_path=output,
        scores=torch.tensor([0.0, 0.05, 1.0]),
        scale_mode="linear",
        visibility_threshold=0.1,
    )

    recolored = PlyData.read(output)["vertex"]
    assert np.asarray(recolored["opacity"]).tolist() == [-100.0, -100.0, 2.0]
    assert statistics["visible_fraction"] == pytest.approx(1.0 / 3.0)
