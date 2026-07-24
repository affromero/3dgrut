"""Behavioral coverage for the native classic-EWA renderer contract."""

import torch
from omegaconf import OmegaConf

from threedgrut.datasets.protocols import Batch
from threedgrut import native_ewa_renderer


class _Background:
    def __init__(self, color: float = 0.0) -> None:
        self.color = color
        self.inputs: tuple[torch.Tensor, torch.Tensor, bool] | None = None

    def __call__(
        self,
        rays_ori: torch.Tensor,
        rays_dir: torch.Tensor,
        rgb: torch.Tensor,
        opacity: torch.Tensor,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inputs = (rays_ori, rays_dir, train)
        return rgb + self.color * (1.0 - opacity), opacity


class _Gaussians:
    def __init__(self) -> None:
        self.max_sh_degree = 0
        self.n_active_features = 0
        self.positions = torch.nn.Parameter(
            torch.tensor(((0.0, 0.0, 2.0), (1.0, 0.0, 2.0)))
        )
        self.rotation = torch.nn.Parameter(
            torch.tensor(((1.0, 0.0, 0.0, 0.0),) * 2)
        )
        self.scale = torch.nn.Parameter(torch.zeros((2, 3)))
        self.density = torch.nn.Parameter(torch.zeros((2, 1)))
        self.features_albedo = torch.nn.Parameter(torch.ones((2, 3)))
        self.environment_mask = torch.zeros(2, dtype=torch.bool)
        self.background = _Background()

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[0]

    def get_positions(self) -> torch.Tensor:
        return self.positions

    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.rotation, dim=1)

    def get_scale(self) -> torch.Tensor:
        return torch.exp(self.scale)

    def get_density(self) -> torch.Tensor:
        return torch.sigmoid(self.density)

    def get_features_albedo(self) -> torch.Tensor:
        return self.features_albedo


def _batch() -> Batch:
    rgb = torch.zeros((1, 1, 1, 3))
    rays = torch.tensor(([[[0.0, 0.0, 1.0]]],))
    return Batch(
        rays_ori=torch.zeros_like(rays),
        rays_dir=rays,
        T_to_world=torch.eye(4).unsqueeze(0),
        rgb_gt=rgb,
        depth_ray_z=torch.ones((1, 1, 1, 1)),
        intrinsics_OpenCVPinholeCameraModelParameters={
            "focal_length": (1.0, 1.0),
            "principal_point": (0.0, 0.0),
        },
    )


def _tracer() -> native_ewa_renderer.Tracer:
    return native_ewa_renderer.Tracer(
        OmegaConf.create(
            {
                "render": {
                    "native_ewa": {
                        "eps2d": 0.1,
                        "near_plane": 0.01,
                        "tile_size": 16,
                        "global_z_order": True,
                    }
                }
            }
        )
    )


def test_native_ewa_uses_recovered_default_near_plane() -> None:
    """Default projection culls the native minimum camera-depth region."""
    tracer = native_ewa_renderer.Tracer(
        OmegaConf.create({"render": {"native_ewa": {}}})
    )

    assert tracer.near_plane == 0.1


def _projection_backward_inputs(
    *,
    requires_grad: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build one FOV-clipped packed projection surrogate."""
    positions = torch.tensor(
        ((4.0, 0.5, 2.0),),
        requires_grad=requires_grad,
    )
    scales = torch.tensor(((0.75, 0.6, 0.5),))
    rotations = torch.tensor(((1.0, 0.0, 0.0, 0.0),))
    densities = torch.tensor((0.5,))
    gaussian_ids = torch.tensor((0,), dtype=torch.int64)
    camera_to_world = torch.eye(4).unsqueeze(0)
    intrinsics = torch.eye(3).unsqueeze(0)
    intrinsics[0, 0, 0] = 1.0
    intrinsics[0, 1, 1] = 1.0
    intrinsics[0, 0, 2] = 1.0
    intrinsics[0, 1, 2] = 1.0
    means2d = positions[:, :2]
    conics = torch.stack(
        (positions[:, 0], positions[:, 1], positions[:, 2]),
        dim=1,
    )
    opacities = positions[:, 0]
    depths = positions[:, 2]
    return (
        positions,
        scales,
        rotations,
        densities,
        gaussian_ids,
        camera_to_world,
        intrinsics,
        means2d,
        conics,
        opacities,
        depths,
    )


def test_native_ewa_accepts_extension_style_intrinsic_pairs() -> None:
    """Camera bindings may expose iterable pair objects instead of tuples."""
    assert native_ewa_renderer._pair(range(11, 13), "focal_length") == (
        11.0,
        12.0,
    )


def test_frozen_fov_clamp_backward_preserves_forward_values() -> None:
    """Frozen FOV clipping changes only the selected position VJP."""
    (
        positions,
        scales,
        rotations,
        densities,
        gaussian_ids,
        camera_to_world,
        intrinsics,
        means2d,
        conics,
        opacities,
        depths,
    ) = _projection_backward_inputs(requires_grad=True)
    weights = (
        torch.tensor(((0.25, -0.5),)),
        torch.tensor(((0.75, -0.25, 0.5),)),
        torch.tensor((0.125,)),
        torch.tensor((-0.375,)),
    )
    corrected = native_ewa_renderer._apply_frozen_fov_clamp_backward(
        mode=native_ewa_renderer.FOVClampBackward.FROZEN,
        positions=positions,
        scales=scales,
        rotations=rotations,
        densities=densities,
        gaussian_ids=gaussian_ids,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        depths=depths,
        width=2,
        height=2,
        eps2d=0.1,
    )
    for source, result in zip(
        (means2d, conics, opacities, depths),
        corrected,
        strict=True,
    ):
        torch.testing.assert_close(result, source)
    loss = sum(
        (value * weight).sum()
        for value, weight in zip(corrected, weights, strict=True)
    )
    loss.backward()
    assert positions.grad is not None

    (
        baseline_positions,
        baseline_scales,
        baseline_rotations,
        baseline_densities,
        _,
        baseline_camera_to_world,
        baseline_intrinsics,
        baseline_means2d,
        baseline_conics,
        baseline_opacities,
        baseline_depths,
    ) = _projection_backward_inputs(requires_grad=True)
    baseline_loss = sum(
        (value * weight).sum()
        for value, weight in zip(
            (
                baseline_means2d,
                baseline_conics,
                baseline_opacities,
                baseline_depths,
            ),
            weights,
            strict=True,
        )
    )
    baseline_loss.backward()
    assert baseline_positions.grad is not None
    autodiff_vjp = native_ewa_renderer._fov_clamp_projection_position_vjp(
        positions=baseline_positions.detach(),
        scales=baseline_scales,
        rotations=baseline_rotations,
        densities=baseline_densities,
        camera_to_world=baseline_camera_to_world[0],
        intrinsics=baseline_intrinsics[0],
        means2d_gradient=weights[0],
        conics_gradient=weights[1],
        depths_gradient=weights[3],
        opacities_gradient=weights[2],
        width=2,
        height=2,
        eps2d=0.1,
        freeze_clamp_backward=False,
    )
    frozen_vjp = native_ewa_renderer._fov_clamp_projection_position_vjp(
        positions=baseline_positions.detach(),
        scales=baseline_scales,
        rotations=baseline_rotations,
        densities=baseline_densities,
        camera_to_world=baseline_camera_to_world[0],
        intrinsics=baseline_intrinsics[0],
        means2d_gradient=weights[0],
        conics_gradient=weights[1],
        depths_gradient=weights[3],
        opacities_gradient=weights[2],
        width=2,
        height=2,
        eps2d=0.1,
        freeze_clamp_backward=True,
    )
    torch.testing.assert_close(
        positions.grad,
        baseline_positions.grad + frozen_vjp - autodiff_vjp,
    )


def test_layered_depth_tile_lists_support_rectangular_tiles() -> None:
    """Selected-layer lists use independently configured tile dimensions."""
    offsets, flatten_ids = native_ewa_renderer._layered_depth_tile_lists(
        means2d=torch.tensor(((8.0, 4.0), (24.0, 12.0))),
        conics=torch.tensor(((1.0, 0.0, 1.0),) * 2),
        opacities=torch.tensor((0.5, 0.5)),
        depths=torch.tensor((2.0, 1.0)),
        width=32,
        height=16,
        tile_width=16,
        tile_height=8,
        alpha_cutoff=1.0 / 255.0,
    )

    torch.testing.assert_close(flatten_ids, torch.tensor((0, 1), dtype=torch.int32))
    torch.testing.assert_close(
        offsets,
        torch.tensor((((0, 1), (1, 1)),), dtype=torch.int32),
    )


def test_layered_depth_tile_lists_include_fixed_coverage_margin() -> None:
    """The EWA tile list includes the fixed support-boundary margin."""
    offsets, flatten_ids = native_ewa_renderer._layered_depth_tile_lists(
        means2d=torch.tensor(((15.65, 4.0),)),
        conics=torch.tensor(((100.0, 0.0, 100.0),)),
        opacities=torch.tensor((0.5,)),
        depths=torch.tensor((1.0,)),
        width=32,
        height=8,
        tile_width=16,
        tile_height=8,
        alpha_cutoff=1.0 / 255.0,
    )

    torch.testing.assert_close(
        offsets,
        torch.tensor((((0, 1),),), dtype=torch.int32),
    )
    torch.testing.assert_close(
        flatten_ids,
        torch.tensor((0, 0), dtype=torch.int32),
    )


def test_layered_depth_tile_row_batches_preserve_depth_buffers(
    monkeypatch,
) -> None:
    """Tile-row batching preserves ordered compositing and its gradients."""

    def fake_rasterize_to_indices_in_range(
        range_start: int,
        range_end: int,
        transmittance: torch.Tensor,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del range_start, range_end, transmittance, conics, opacities, tile_size
        gaussian_ids = flatten_ids.to(dtype=torch.int64)
        if height == 32:
            pixel_y = gaussian_ids * 16
        else:
            pixel_y = torch.zeros_like(gaussian_ids)
        pixel_ids = pixel_y * width
        image_ids = torch.zeros_like(gaussian_ids)
        assert means2d.shape == (1, 2, 2)
        assert offsets.shape[2] == 1
        return gaussian_ids, pixel_ids, image_ids

    monkeypatch.setattr(
        native_ewa_renderer,
        "_load_gsplat_rasterize_to_indices_in_range",
        lambda: fake_rasterize_to_indices_in_range,
    )
    offsets = torch.tensor([[[0], [1]]], dtype=torch.int32)
    flatten_ids = torch.tensor((0, 1), dtype=torch.int32)

    def render(
        tile_rows_per_batch: int,
        activation_checkpoint: bool,
    ) -> tuple[torch.Tensor, ...]:
        means2d = torch.tensor(
            ((0.5, 0.5), (0.5, 16.5)),
            requires_grad=True,
        )
        opacities = torch.tensor((0.5, 0.5), requires_grad=True)
        outputs = native_ewa_renderer._layered_depth_accumulation(
            means2d=means2d,
            conics=torch.tensor(((1.0, 0.0, 1.0),) * 2),
            opacities=opacities,
            depths=torch.tensor((1.0, 2.0)),
            offsets=offsets,
            flatten_ids=flatten_ids,
            width=16,
            height=32,
            tile_width=16,
            vertical_scale=1,
            alpha_cutoff=1.0 / 255.0,
            tile_rows_per_batch=tile_rows_per_batch,
            activation_checkpoint=activation_checkpoint,
        )
        sum(output.sum() for output in outputs).backward()
        assert means2d.grad is not None
        assert opacities.grad is not None
        return (*outputs, means2d.grad, opacities.grad)

    full = render(tile_rows_per_batch=2, activation_checkpoint=False)
    batched = render(tile_rows_per_batch=1, activation_checkpoint=True)
    for full_value, batched_value in zip(full, batched, strict=True):
        torch.testing.assert_close(full_value, batched_value)


def test_layered_depth_checkpoint_accepts_an_empty_tile_band() -> None:
    """A sparse frame may have no selected-layer contribution in one band."""
    means2d = torch.tensor(((0.5, 0.5),), requires_grad=True)
    outputs = native_ewa_renderer._layered_depth_accumulation(
        means2d=means2d,
        conics=torch.tensor(((1.0, 0.0, 1.0),)),
        opacities=torch.tensor((0.5,)),
        depths=torch.tensor((1.0,)),
        offsets=torch.zeros((1, 1, 1), dtype=torch.int32),
        flatten_ids=torch.empty((0,), dtype=torch.int32),
        width=16,
        height=16,
        tile_width=16,
        vertical_scale=1,
        alpha_cutoff=1.0 / 255.0,
        tile_rows_per_batch=1,
        activation_checkpoint=True,
    )

    raw_depth, transmittance, median_depth = outputs
    torch.testing.assert_close(raw_depth, torch.zeros_like(raw_depth))
    torch.testing.assert_close(
        transmittance,
        torch.ones_like(transmittance),
    )
    torch.testing.assert_close(median_depth, torch.zeros_like(median_depth))
    means2d.sum().backward()
    assert means2d.grad is not None


def test_layered_depth_uses_the_selected_compositor(monkeypatch) -> None:
    """The fused option receives the same stretched packed layer inputs."""
    captured: dict[str, object] = {}

    def fake_composite_layered_depth(
        *,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
        alpha_cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured["means2d"] = means2d
        captured["conics"] = conics
        captured["opacities"] = opacities
        captured["depths"] = depths
        captured["offsets"] = offsets
        captured["flatten_ids"] = flatten_ids
        captured["width"] = width
        captured["height"] = height
        captured["tile_size"] = tile_size
        captured["alpha_cutoff"] = alpha_cutoff
        shape = (1, height, width, 1)
        return (
            torch.zeros(shape),
            torch.ones(shape),
            torch.zeros(shape),
        )

    monkeypatch.setattr(
        native_ewa_renderer,
        "composite_layered_depth",
        fake_composite_layered_depth,
    )
    outputs = native_ewa_renderer._render_layered_depth_buffers(
        gaussian_ids=torch.tensor((0,)),
        means2d=torch.tensor(((8.0, 8.0),)),
        conics=torch.tensor(((1.0, 0.0, 1.0),)),
        opacities=torch.tensor((0.5,)),
        depths=torch.tensor((2.0,)),
        selected_layer_mask=torch.tensor((True,)),
        semantic_mask=torch.ones((1, 16, 16, 1), dtype=torch.uint8),
        width=16,
        height=16,
        near_plane=0.1,
        tile_width=16,
        tile_height=16,
        alpha_cutoff=1.0 / 255.0,
        transparency_threshold=0.0,
        tile_rows_per_batch=1,
        activation_checkpoint=True,
        compositor=native_ewa_renderer.LayeredDepthCompositor.FUSED,
    )

    assert captured["width"] == 16
    assert captured["height"] == 16
    assert captured["tile_size"] == 16
    torch.testing.assert_close(
        outputs["layered_transparency"],
        torch.ones((1, 16, 16, 1)),
    )


def test_layered_depth_excludes_near_plane_rows(monkeypatch) -> None:
    """The rebuilt depth layer preserves the native projection eligibility."""
    captured: dict[str, torch.Tensor] = {}

    def fake_composite_layered_depth(
        *,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        opacities: torch.Tensor,
        depths: torch.Tensor,
        offsets: torch.Tensor,
        flatten_ids: torch.Tensor,
        width: int,
        height: int,
        tile_size: int,
        alpha_cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del conics, opacities, offsets, flatten_ids, tile_size, alpha_cutoff
        captured["means2d"] = means2d
        captured["depths"] = depths
        shape = (1, height, width, 1)
        return (
            torch.zeros(shape),
            torch.ones(shape),
            torch.zeros(shape),
        )

    monkeypatch.setattr(
        native_ewa_renderer,
        "composite_layered_depth",
        fake_composite_layered_depth,
    )
    native_ewa_renderer._render_layered_depth_buffers(
        gaussian_ids=torch.tensor((0, 1, 2)),
        means2d=torch.tensor(((8.0, 8.0),) * 3),
        conics=torch.tensor(((1.0, 0.0, 1.0),) * 3),
        opacities=torch.tensor((0.5, 0.5, 0.5)),
        depths=torch.tensor((0.099, 0.1, 0.101)),
        selected_layer_mask=torch.tensor((True, True, True)),
        semantic_mask=torch.ones((1, 16, 16, 1), dtype=torch.uint8),
        width=16,
        height=16,
        near_plane=0.1,
        tile_width=16,
        tile_height=16,
        alpha_cutoff=1.0 / 255.0,
        transparency_threshold=0.0,
        tile_rows_per_batch=1,
        activation_checkpoint=True,
        compositor=native_ewa_renderer.LayeredDepthCompositor.FUSED,
    )

    torch.testing.assert_close(
        captured["depths"],
        torch.tensor((0.1, 0.101)),
    )
    assert captured["means2d"].shape == (2, 2)


def test_native_ewa_scatter_and_gradient_hooks_follow_packed_metadata(
    monkeypatch,
) -> None:
    """Packed EWA gradients retain native pixel-coordinate scaling."""
    captured: dict[str, object] = {}

    def fake_rasterization(**kwargs: object):
        means = kwargs["means"]
        assert isinstance(means, torch.Tensor)
        packed_means2d = means[:, :2].flip(0)
        rgb = packed_means2d.sum().reshape(1, 1, 1, 1).expand(
            -1, -1, -1, 3
        )
        depth = torch.full((1, 1, 1, 1), 2.0)
        rendered = torch.cat((rgb, depth), dim=-1)
        alpha = torch.full((1, 1, 1, 1), 0.75)
        return (
            rendered,
            alpha,
            {
                "gaussian_ids": torch.tensor((1, 0)),
                "means2d": packed_means2d,
                "conics": torch.tensor(((1.0, 0.0, 1.0),) * 2),
                "opacities": torch.tensor((0.5, 0.25)),
                "depths": torch.tensor((2.0, 3.0)),
                "radii": torch.tensor(((3, 2), (1, 1))),
                "tiles_per_gauss": torch.tensor((4, 2)),
                "isect_offsets": torch.tensor(([[0]],)),
                "flatten_ids": torch.tensor((0, 1)),
                "tile_size": 16,
            },
        )

    def fake_rectangular_tile_lists(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        return torch.zeros((1, 1, 1), dtype=torch.int32), torch.tensor(
            (0, 1),
            dtype=torch.int32,
        )

    def fake_composite_rectangular_ewa(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured["render_kwargs"] = kwargs
        means2d = kwargs["means2d"]
        assert isinstance(means2d, torch.Tensor)
        rgb = means2d.sum().reshape(1, 1, 1, 1).expand(-1, -1, -1, 3)
        return (
            rgb,
            torch.full((1, 1, 1, 1), 0.75),
            torch.full((1, 1, 1, 1), 2.0),
        )

    def fake_masked_rectangular_ewa_forward_contribution(
        **kwargs: object,
    ) -> torch.Tensor:
        captured["contribution_kwargs"] = kwargs
        means2d = kwargs["means2d"]
        assert isinstance(means2d, torch.Tensor)
        return torch.ones((means2d.shape[0],), dtype=means2d.dtype)

    def fake_masked_rectangular_ewa_forward_visibility(
        **kwargs: object,
    ) -> torch.Tensor:
        captured["forward_visibility_kwargs"] = kwargs
        return torch.tensor((True, False), dtype=torch.bool)

    monkeypatch.setattr(
        native_ewa_renderer,
        "_load_gsplat_rasterization",
        lambda: fake_rasterization,
    )
    monkeypatch.setattr(
        native_ewa_renderer,
        "_layered_depth_tile_lists",
        fake_rectangular_tile_lists,
    )
    monkeypatch.setattr(
        native_ewa_renderer,
        "composite_rectangular_ewa",
        fake_composite_rectangular_ewa,
    )
    monkeypatch.setattr(
        native_ewa_renderer,
        "masked_rectangular_ewa_forward_contribution",
        fake_masked_rectangular_ewa_forward_contribution,
    )
    monkeypatch.setattr(
        native_ewa_renderer,
        "masked_rectangular_ewa_forward_visibility",
        fake_masked_rectangular_ewa_forward_visibility,
    )
    gaussians = _Gaussians()
    outputs = _tracer().render(gaussians, _batch(), train=True)

    outputs["pred_rgb"].sum().backward()

    assert gaussians.positions.grad is not None
    torch.testing.assert_close(
        outputs["mog_visibility"], torch.tensor(((True,), (True,)))
    )
    torch.testing.assert_close(
        outputs["mog_projected_extent"], torch.tensor(((1.0, 1.0), (3.0, 2.0)))
    )
    torch.testing.assert_close(
        outputs["mog_accumulated_weight"], torch.ones((2, 1)))
    torch.testing.assert_close(
        outputs["mog_forward_visibility"],
        torch.tensor(((False,), (True,))),
    )
    torch.testing.assert_close(
        outputs["mog_projected_gradient_pixels"],
        torch.full((2,), float(4.5**0.5)),
    )
    contribution_kwargs = captured["contribution_kwargs"]
    assert isinstance(contribution_kwargs, dict)
    assert contribution_kwargs["environment_mask"].tolist() == [False, False]
    assert contribution_kwargs["pixel_mask"] is None
    assert contribution_kwargs["transmittance_cutoff"] == 0.01
    forward_visibility_kwargs = captured["forward_visibility_kwargs"]
    assert isinstance(forward_visibility_kwargs, dict)
    assert forward_visibility_kwargs["pixel_mask"] is None
    assert forward_visibility_kwargs["transmittance_cutoff"] == 0.01
    render_kwargs = captured["render_kwargs"]
    assert isinstance(render_kwargs, dict)
    assert render_kwargs["transmittance_cutoff"] == 0.01


def test_native_ewa_screen_gradient_norm_uses_matching_viewport_axes() -> None:
    """Native topology scales each screen-gradient component by its own axis."""
    actual = native_ewa_renderer._screen_gradient_pixel_norm(
        torch.tensor(((1.0, 0.0), (0.0, 1.0))),
        width=4,
        height=2,
    )

    torch.testing.assert_close(actual, torch.tensor((2.0, 1.0)))


def test_native_ewa_composes_the_model_background(monkeypatch) -> None:
    """Configured background composition must happen in model dispatch."""

    def fake_rasterization(**kwargs: object):
        means = kwargs["means"]
        assert isinstance(means, torch.Tensor)
        rendered = torch.zeros((1, 1, 1, 4))
        alpha = torch.full((1, 1, 1, 1), 0.75)
        return (
            rendered,
            alpha,
            {
                "gaussian_ids": torch.tensor((0, 1)),
                "means2d": means[:, :2],
                "conics": torch.tensor(((1.0, 0.0, 1.0),) * 2),
                "opacities": torch.tensor((0.5, 0.25)),
                "depths": torch.tensor((2.0, 3.0)),
                "radii": torch.tensor(((3, 2), (1, 1))),
                "tiles_per_gauss": torch.tensor((4, 2)),
                "isect_offsets": torch.tensor(([[0]],)),
                "flatten_ids": torch.tensor((0, 1)),
                "tile_size": 16,
            },
        )

    def fake_rectangular_tile_lists(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        return torch.zeros((1, 1, 1), dtype=torch.int32), torch.tensor(
            (0, 1),
            dtype=torch.int32,
        )

    def fake_composite_rectangular_ewa(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs
        return (
            torch.zeros((1, 1, 1, 3)),
            torch.full((1, 1, 1, 1), 0.75),
            torch.full((1, 1, 1, 1), 2.0),
        )

    monkeypatch.setattr(
        native_ewa_renderer,
        "_load_gsplat_rasterization",
        lambda: fake_rasterization,
    )
    monkeypatch.setattr(
        native_ewa_renderer,
        "_layered_depth_tile_lists",
        fake_rectangular_tile_lists,
    )
    monkeypatch.setattr(
        native_ewa_renderer,
        "composite_rectangular_ewa",
        fake_composite_rectangular_ewa,
    )
    gaussians = _Gaussians()
    gaussians.background = _Background(color=1.0)

    outputs = _tracer().render(gaussians, _batch())

    torch.testing.assert_close(
        outputs["pred_rgb"], torch.full((1, 1, 1, 3), 0.25)
    )
    assert gaussians.background.inputs is not None
    assert gaussians.background.inputs[2] is False


def test_native_ewa_rejects_non_sh0_radiance() -> None:
    """The native backend refuses a color representation it does not render."""
    gaussians = _Gaussians()
    gaussians.max_sh_degree = 1

    try:
        _tracer().render(gaussians, _batch())
    except ValueError as exc:
        assert "SH0" in str(exc)
    else:
        raise AssertionError("native_ewa accepted non-SH0 radiance")
