"""Tests for held-out Gaussian error attribution and PLY recoloring."""

import os

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement

from render_error_splats import (
    DEFAULT_VISIBILITY_THRESHOLD,
    _scene_relative_path,
)
from threedgrut.error_attribution import (
    ErrorAttributionAccumulator,
    ErrorAttributionMetric,
    ErrorAttributionParameter,
    attribution_loss,
    recolor_gaussian_ply,
)


def test_error_splat_manifest_paths_are_scene_relative(
    tmp_path: os.PathLike[str],
) -> None:
    """Portable manifests reject checkpoint paths outside the scene."""
    scene_root = os.path.join(tmp_path, "scene")
    checkpoint = os.path.join(scene_root, "generated_files", "ckpt.pt")

    assert _scene_relative_path(checkpoint, scene_root) == os.path.join(
        "generated_files",
        "ckpt.pt",
    )
    with pytest.raises(ValueError, match="escapes --scene-root"):
        _scene_relative_path(os.path.join(tmp_path, "outside.pt"), scene_root)


class _TwoPixelModel(torch.nn.Module):
    """Minimal row-aligned parameter surface for attribution tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features_albedo = torch.nn.Parameter(torch.zeros((1, 1)))


def test_error_splat_export_preserves_full_scene_by_default() -> None:
    """Default error colors retain every source Gaussian's opacity."""
    assert DEFAULT_VISIBILITY_THRESHOLD == 0.0


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
    assert accumulator.rms_scores()["mse:features_albedo"].item() > 0.0


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
