"""Behavior tests for held-out-predictive multiscale PPISP."""

from types import SimpleNamespace

import pytest
import torch
from ppisp import PPISP, PPISPConfig
from threedgrut.post_processing.predictive_multiscale_ppisp import (
    MultiscaleFieldController,
    MultiscalePPISPConfig,
    PredictiveMultiscalePPISP,
)
from threedgrut.utils.render import apply_post_processing


class _RecordingViewContextPostProcessor(torch.nn.Module):
    """Record the context passed through the shared render path."""

    use_view_context = True

    def __init__(self) -> None:
        super().__init__()
        self.render_ray_distance: torch.Tensor | None = None
        self.render_opacity: torch.Tensor | None = None
        self.camera_to_world: torch.Tensor | None = None

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        *,
        resolution: tuple[int, int],
        camera_idx: int,
        frame_idx: int,
        exposure_prior: torch.Tensor | None,
        render_ray_distance: torch.Tensor | None,
        render_opacity: torch.Tensor | None,
        camera_to_world: torch.Tensor | None,
    ) -> torch.Tensor:
        """Record context while leaving RGB unchanged."""
        assert pixel_coords.shape == (4, 2)
        assert resolution == (2, 2)
        assert camera_idx == 0
        assert frame_idx == -1
        assert exposure_prior is None
        self.render_ray_distance = render_ray_distance
        self.render_opacity = render_opacity
        self.camera_to_world = camera_to_world
        return rgb


def test_multiscale_heads_start_as_exact_identity_fields() -> None:
    """Both learned heads must be zero before optimization."""
    controller = MultiscaleFieldController(
        coarse_grid_size=4,
        fine_grid_size=16,
    )

    coarse, fine = controller(torch.rand((1, 3, 32, 48)))

    torch.testing.assert_close(coarse, torch.zeros_like(coarse))
    torch.testing.assert_close(fine, torch.zeros_like(fine))


def test_identical_rgb_can_predict_sensor_position_dependent_fields() -> None:
    """Sensor coordinates must break translation ambiguity on flat renders."""
    controller = MultiscaleFieldController(
        coarse_grid_size=4,
        fine_grid_size=16,
    )
    with torch.no_grad():
        for layer in (
            controller.encoder[0],
            controller.encoder[2],
            controller.encoder[4],
            controller.fine_head,
        ):
            layer.weight.zero_()
            layer.bias.zero_()
        controller.encoder[0].weight[0, 3, 1, 1] = 1.0
        controller.encoder[2].weight[0, 0, 1, 1] = 1.0
        controller.encoder[4].weight[0, 0, 1, 1] = 1.0
        controller.fine_head.weight[0, 0, 0, 0] = 1.0

    _, fine = controller(torch.zeros((1, 3, 32, 48)))

    assert fine[0, 0, 8, 0] < fine[0, 0, 8, -1]


def test_identical_rgb_can_predict_view_context_dependent_fields() -> None:
    """Novel-view geometry must disambiguate identical rendered pixels."""
    controller = MultiscaleFieldController(
        coarse_grid_size=4,
        fine_grid_size=16,
        input_channels=19,
    )
    with torch.no_grad():
        for layer in (
            controller.encoder[0],
            controller.encoder[2],
            controller.encoder[4],
            controller.fine_head,
        ):
            layer.weight.zero_()
            layer.bias.zero_()
        controller.encoder[0].weight[0, 7, 1, 1] = 1.0
        controller.encoder[2].weight[0, 0, 1, 1] = 1.0
        controller.encoder[4].weight[0, 0, 1, 1] = 1.0
        controller.fine_head.weight[0, 0, 0, 0] = 1.0
    image = torch.zeros((1, 3, 32, 48))
    first_context = torch.zeros((1, 14, 32, 48))
    second_context = first_context.clone()
    second_context[:, 2] = 0.5

    _, first = controller(image, first_context)
    _, second = controller(image, second_context)

    assert torch.count_nonzero(first) == 0
    assert torch.count_nonzero(second) > 0


def test_shared_render_path_passes_exact_view_context() -> None:
    """Training and evaluation share the same renderer-context wiring."""
    module = _RecordingViewContextPostProcessor()
    pose = torch.eye(4).unsqueeze(0)
    distance = torch.ones((1, 2, 2))
    opacity = torch.full((1, 2, 2, 1), 0.75)
    outputs = {
        "pred_rgb": torch.zeros((1, 2, 2, 3)),
        "pred_dist": distance,
        "pred_opacity": opacity,
    }
    batch = SimpleNamespace(
        camera_idx=0,
        frame_idx=0,
        sequence_idx=-1,
        pixel_coords=torch.zeros((1, 2, 2, 2)),
        exposure=None,
        T_to_world=pose,
        T_to_world_end=None,
        rays_in_world_space=False,
    )

    apply_post_processing(module, outputs, batch)

    assert module.render_ray_distance is distance
    assert module.render_opacity is opacity
    assert module.camera_to_world is pose


def test_shared_render_path_rejects_world_space_ray_pose() -> None:
    """An identity pose cannot silently stand in for injected world rays."""
    module = _RecordingViewContextPostProcessor()
    outputs = {
        "pred_rgb": torch.zeros((1, 2, 2, 3)),
        "pred_dist": torch.ones((1, 2, 2)),
        "pred_opacity": torch.ones((1, 2, 2, 1)),
    }
    batch = SimpleNamespace(
        camera_idx=0,
        frame_idx=0,
        sequence_idx=-1,
        pixel_coords=torch.zeros((1, 2, 2, 2)),
        exposure=None,
        T_to_world=torch.eye(4).unsqueeze(0),
        T_to_world_end=None,
        rays_in_world_space=True,
    )

    with pytest.raises(ValueError, match="camera-space rays"):
        apply_post_processing(module, outputs, batch)


def test_grid_and_bound_contracts_fail_before_training() -> None:
    """Invalid architecture and correction bounds fail before allocation."""
    config = PPISPConfig(use_controller=True)
    if not torch.cuda.is_available():
        pytest.skip("PPISP allocates its state on CUDA.")

    with pytest.raises(ValueError, match="grid sizes"):
        PredictiveMultiscalePPISP(
            num_cameras=1,
            num_frames=1,
            config=config,
            multiscale_config=MultiscalePPISPConfig(
                coarse_grid_size=8,
                fine_grid_size=4,
            ),
        )
    with pytest.raises(ValueError, match="bounds"):
        PredictiveMultiscalePPISP(
            num_cameras=1,
            num_frames=1,
            config=config,
            multiscale_config=MultiscalePPISPConfig(
                coarse_max_bias=0.0,
            ),
        )
    with pytest.raises(ValueError, match="frozen-geometry"):
        PredictiveMultiscalePPISP(
            num_cameras=1,
            num_frames=1,
            config=PPISPConfig(
                use_controller=True,
                controller_distillation=False,
            ),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PPISP allocates and evaluates on CUDA.",
)
def test_zero_initialized_multiscale_path_preserves_ppisp_output() -> None:
    """Enabling a fresh local controller does not perturb PPISP output."""
    config = PPISPConfig(
        use_controller=True,
        controller_distillation=True,
        controller_activation_ratio=0.5,
    )
    module = PredictiveMultiscalePPISP(
        num_cameras=1,
        num_frames=2,
        config=config,
    )
    module._controller_activation_step = 5
    module._ppisp_scheduler = SimpleNamespace(last_epoch=6)
    height, width = 8, 8
    rgb = torch.rand((height * width, 3), device="cuda")
    y, x = torch.meshgrid(
        torch.arange(height, device="cuda"),
        torch.arange(width, device="cuda"),
        indexing="ij",
    )
    pixel_coords = torch.stack((x, y), dim=-1).reshape(-1, 2).float()

    expected = PPISP.forward(
        module,
        rgb,
        pixel_coords,
        (width, height),
        camera_idx=0,
        frame_idx=-1,
    )
    actual = module(
        rgb,
        pixel_coords,
        (width, height),
        camera_idx=0,
        frame_idx=-1,
    )

    torch.testing.assert_close(actual, expected)
    assert module.get_regularization_loss().isfinite()
    state_names = tuple(module.state_dict())
    assert not any(
        "frame" in name for name in state_names if "multiscale" in name
    )
    optimizers = module.create_optimizers()
    assert len(optimizers) == 2
    assert len(optimizers[1].param_groups) == 2


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PPISP allocates and evaluates on CUDA.",
)
def test_exact_state_restoration_preserves_multiscale_parameters() -> None:
    """Exact restore retains every global and local learned tensor."""
    config = PPISPConfig(
        use_controller=True,
        controller_distillation=True,
        controller_activation_ratio=0.5,
    )
    source = PredictiveMultiscalePPISP(
        num_cameras=2,
        num_frames=3,
        config=config,
    )
    with torch.no_grad():
        source.multiscale_controllers[1].fine_head.bias.fill_(0.25)

    restored = PredictiveMultiscalePPISP.from_state_dict(
        source.state_dict(),
        config=config,
    )

    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PPISP allocates and evaluates on CUDA.",
)
def test_view_context_restore_preserves_train_fold_normalization() -> None:
    """Strict restore retains train-only normalization and architecture."""
    config = PPISPConfig(
        use_controller=True,
        controller_distillation=True,
        controller_activation_ratio=0.5,
    )
    multiscale_config = MultiscalePPISPConfig(use_view_context=True)
    source = PredictiveMultiscalePPISP(
        num_cameras=1,
        num_frames=2,
        config=config,
        multiscale_config=multiscale_config,
        view_center=torch.tensor([1.0, 2.0, 3.0]),
        view_scale=4.0,
    )

    restored = PredictiveMultiscalePPISP.from_state_dict(
        source.state_dict(),
        config=config,
        multiscale_config=multiscale_config,
    )

    assert restored.use_view_context is True
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PPISP allocates and evaluates on CUDA.",
)
def test_view_context_fails_when_ready_context_is_missing() -> None:
    """A trained context model must never fall back to RGB-only inference."""
    config = PPISPConfig(
        use_controller=True,
        controller_distillation=True,
        controller_activation_ratio=0.5,
    )
    module = PredictiveMultiscalePPISP(
        num_cameras=1,
        num_frames=2,
        config=config,
        multiscale_config=MultiscalePPISPConfig(use_view_context=True),
        view_center=torch.zeros(3),
        view_scale=1.0,
    )
    module._controller_activation_step = 5
    module._ppisp_scheduler = SimpleNamespace(last_epoch=6)
    height, width = 8, 8
    rgb = torch.rand((height * width, 3), device="cuda")
    y, x = torch.meshgrid(
        torch.arange(height, device="cuda"),
        torch.arange(width, device="cuda"),
        indexing="ij",
    )
    pixel_coords = torch.stack((x, y), dim=-1).reshape(-1, 2).float()

    with pytest.raises(ValueError, match="requires renderer ray"):
        module(
            rgb,
            pixel_coords,
            (width, height),
            camera_idx=0,
            frame_idx=-1,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PPISP allocates and evaluates on CUDA.",
)
def test_local_controller_initialization_preserves_global_rng() -> None:
    """Adding local fields cannot change the matched Gaussian RNG stream."""
    config = PPISPConfig(
        use_controller=True,
        controller_distillation=True,
        controller_activation_ratio=0.5,
    )
    torch.manual_seed(1234)
    PPISP(num_cameras=1, num_frames=2, config=config)
    expected_next = torch.rand(8)
    expected_cuda_next = torch.rand(8, device="cuda")

    torch.manual_seed(1234)
    PredictiveMultiscalePPISP(
        num_cameras=1,
        num_frames=2,
        config=config,
    )
    actual_next = torch.rand(8)
    actual_cuda_next = torch.rand(8, device="cuda")

    torch.testing.assert_close(actual_next, expected_next)
    torch.testing.assert_close(actual_cuda_next, expected_cuda_next)
