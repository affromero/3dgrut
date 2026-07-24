import pytest
import torch
from threedgrut.post_processing import LuminanceAffine


def test_color_matrix_identity_initialization() -> None:
    module = LuminanceAffine(
        num_cameras=2,
        num_frames=4,
        use_color_matrix=True,
    )
    pred_rgb = torch.tensor([[0.2, 0.5, 0.7]], dtype=torch.float32)
    pixel_coords = torch.tensor([[8.0, 4.0]], dtype=torch.float32)

    corrected = module(
        pred_rgb,
        pixel_coords,
        resolution=(16, 8),
        camera_idx=0,
        frame_idx=0,
    )

    assert torch.allclose(corrected, pred_rgb)
    assert torch.equal(module.get_regularization_loss(), torch.tensor(0.0))


def test_color_matrix_applies_bounded_camera_channel_mixing() -> None:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=1,
        use_color_matrix=True,
        max_matrix_delta=0.2,
    )
    with torch.no_grad():
        module.color_matrix_raw[0, 0, 1] = 1.0

    pred_rgb = torch.tensor([[0.2, 0.5, 0.1]], dtype=torch.float32)
    pixel_coords = torch.tensor([[8.0, 4.0]], dtype=torch.float32)
    corrected = module(
        pred_rgb,
        pixel_coords,
        resolution=(16, 8),
        camera_idx=0,
        frame_idx=0,
    )

    expected_red = pred_rgb[0, 0] + pred_rgb[0, 1] * torch.tanh(
        torch.tensor(1.0)
    ) * 0.2
    assert torch.allclose(corrected[0, 0], expected_red)
    assert torch.allclose(corrected[0, 1:], pred_rgb[0, 1:])
    assert module.get_regularization_loss().item() > 0.0


def test_radial_affine_identity_initialization() -> None:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=1,
        use_radial_affine=True,
        radial_band_count=3,
    )
    pred_rgb = torch.tensor([[0.2, 0.5, 0.7]], dtype=torch.float32)
    pixel_coords = torch.tensor([[8.0, 4.0]], dtype=torch.float32)

    corrected = module(
        pred_rgb,
        pixel_coords,
        resolution=(16, 8),
        camera_idx=0,
        frame_idx=0,
    )

    assert torch.allclose(corrected, pred_rgb)
    assert torch.equal(module.get_regularization_loss(), torch.tensor(0.0))


def test_radial_affine_applies_bounded_rim_gain() -> None:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=1,
        use_radial_affine=True,
        radial_band_count=2,
        radial_max_log_gain=0.1,
        radial_max_bias=0.0,
    )
    with torch.no_grad():
        module.radial_log_gain_raw[0, 1, 0] = 1.0

    pred_rgb = torch.tensor(
        [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]],
        dtype=torch.float32,
    )
    pixel_coords = torch.tensor(
        [[7.5, 3.5], [15.0, 7.0]],
        dtype=torch.float32,
    )
    corrected = module(
        pred_rgb,
        pixel_coords,
        resolution=(16, 8),
        camera_idx=0,
        frame_idx=0,
    )

    expected_rim_red = pred_rgb[1, 0] * torch.exp(
        torch.tanh(torch.tensor(1.0)) * 0.1
    )
    assert torch.allclose(corrected[0], pred_rgb[0])
    assert torch.allclose(corrected[1, 0], expected_rim_red)
    assert torch.allclose(corrected[1, 1:], pred_rgb[1, 1:])
    assert module.get_regularization_loss().item() > 0.0


def test_radial_affine_caches_full_raster_coordinates() -> None:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=1,
        use_radial_affine=True,
        radial_band_count=2,
    )
    x_coords = torch.arange(8, dtype=torch.float32)
    y_coords = torch.arange(4, dtype=torch.float32)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    pixel_coords = torch.stack((x_grid, y_grid), dim=-1).reshape(-1, 2)
    pred_rgb = torch.full((pixel_coords.shape[0], 3), 0.5)

    module(
        pred_rgb,
        pixel_coords,
        resolution=(8, 4),
        camera_idx=0,
        frame_idx=0,
    )

    assert len(module._radial_sample_grids) == 1


def test_frame_residual_grid_applies_only_to_known_frame() -> None:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=2,
        use_residual_grid=True,
        use_frame_residual_grid=True,
        residual_grid_size=3,
        residual_grid_max=1.0,
    )
    with torch.no_grad():
        module.frame_residual_grid.weight[1].fill_(0.2)

    pred_rgb = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)
    pixel_coords = torch.tensor([[4.0, 4.0]], dtype=torch.float32)
    corrected = module(
        pred_rgb,
        pixel_coords,
        resolution=(8, 8),
        camera_idx=0,
        frame_idx=1,
    )
    novel_view = module(
        pred_rgb,
        pixel_coords,
        resolution=(8, 8),
        camera_idx=0,
        frame_idx=-1,
    )

    assert torch.allclose(corrected, pred_rgb + 0.2)
    assert torch.allclose(novel_view, pred_rgb)

    corrected.sum().backward()
    assert module.frame_residual_grid.weight.grad.is_sparse
    optimizers = module.create_optimizers()
    assert len(optimizers) == 2
    assert isinstance(optimizers[1], torch.optim.SparseAdam)


def test_native_appearance_is_exclusive_with_affine_corrections() -> None:
    with pytest.raises(ValueError, match="exclusive"):
        LuminanceAffine(
            num_cameras=1,
            num_frames=1,
            use_native_appearance_grid=True,
            use_radial_affine=True,
        )


def test_native_appearance_bypasses_camera_affine_parameters() -> None:
    module = LuminanceAffine(
        num_cameras=1,
        num_frames=1,
        use_native_appearance_grid=True,
    )
    with torch.no_grad():
        module.camera_log_gain.fill_(1.0)
        module.camera_bias.fill_(1.0)
    pred_rgb = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float32)

    corrected = module(
        pred_rgb,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        resolution=(1, 1),
        camera_idx=0,
        frame_idx=0,
    )

    assert torch.allclose(corrected, pred_rgb)
    assert len(module.create_optimizers()) == 1
