"""Behavioral coverage for the native classic LiDAR depth renderer."""

import pytest
import torch
from omegaconf import OmegaConf
from threedgrut import native_lidar_renderer
from threedgrut.native_lidar_renderer import (
    NativeLidarGaussianState,
    NativeLidarRenderResult,
    NativeLidarRenderer,
    camera_to_world_to_viewmat,
    native_projected_covariance_is_valid,
    parse_native_lidar_renderer,
    render_classic_expected_depth,
)
import threedgrut.trainer as trainer_module
from threedgrut.trainer import Trainer3DGRUT


def test_camera_to_world_to_viewmat_inverts_native_pose() -> None:
    """The native camera pose maps its world-space center to the origin."""
    camera_to_world = torch.eye(4).unsqueeze(0)
    camera_to_world[0, :3, 3] = torch.tensor((2.0, -3.0, 4.0))

    viewmat = camera_to_world_to_viewmat(camera_to_world)
    world_center = torch.tensor((2.0, -3.0, 4.0, 1.0))

    assert torch.allclose(
        viewmat[0] @ world_center, torch.tensor((0.0, 0.0, 0.0, 1.0))
    )


def test_parse_native_lidar_renderer_rejects_unknown_value() -> None:
    """Unsupported renderer selections fail before a training step starts."""
    assert (
        parse_native_lidar_renderer("classic_expected_depth")
        is NativeLidarRenderer.CLASSIC_EXPECTED_DEPTH
    )

    with pytest.raises(ValueError, match="not-a-renderer") as exc_info:
        parse_native_lidar_renderer("not-a-renderer")
    assert "classic_expected_depth" in str(exc_info.value)


def test_classic_lidar_step_bypasses_the_rgb_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Depth-only native updates must not invoke an RGB-only backend."""

    class Sampler:
        intrinsics = (10.0, 10.0, 0.5, 0.5)

        def sample(
            self,
            *,
            rays_per_step: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            float,
        ]:
            del rays_per_step
            rays_ori = torch.zeros((1, 1, 1, 3), device=device, dtype=dtype)
            rays_dir = torch.tensor(
                [[[[0.0, 0.0, 1.0]]]], device=device, dtype=dtype
            )
            camera_to_world = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
            sample_grid = torch.zeros((1, 1, 1, 2), device=device, dtype=dtype)
            depth_gt = torch.full((1, 1, 1, 1), 2.0, device=device, dtype=dtype)
            ray_z = torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
            return (
                rays_ori,
                rays_dir,
                camera_to_world,
                sample_grid,
                depth_gt,
                ray_z,
                1.0,
            )

    class Model:
        def __init__(self) -> None:
            self.positions = torch.nn.Parameter(
                torch.tensor(((0.0, 0.0, 3.0),))
            )
            self.density = torch.nn.Parameter(torch.tensor(((0.0,),)))
            self.rotation = torch.nn.Parameter(
                torch.tensor(((1.0, 0.0, 0.0, 0.0),))
            )
            self.scale = torch.nn.Parameter(torch.zeros((1, 3)))
            self.environment_mask = torch.zeros(1, dtype=torch.bool)
            self.optimizer = torch.optim.SGD([self.positions], lr=0.1)
            self.called = False

        def get_positions(self) -> torch.Tensor:
            return self.positions

        def get_rotation(self) -> torch.Tensor:
            return self.rotation

        def get_scale(self) -> torch.Tensor:
            return self.scale

        def get_density(self) -> torch.Tensor:
            return self.density

        def build_acc(self, rebuild: bool = True) -> None:
            del rebuild

        def __call__(
            self, *args: object, **kwargs: object
        ) -> dict[str, torch.Tensor]:
            del args, kwargs
            self.called = True
            raise AssertionError("Classic LiDAR must not invoke model RGB render")

    def fake_classic_depth(
        *,
        gaussian_state: NativeLidarGaussianState,
        camera_to_world: torch.Tensor,
        intrinsics: object,
        sample_grid: torch.Tensor,
        image_size: int,
    ) -> NativeLidarRenderResult:
        del camera_to_world, intrinsics, sample_grid, image_size
        captured["skip_mask"] = gaussian_state.skip_mask
        depth = gaussian_state.positions[:, 2].reshape(1, 1, 1, 1)
        return NativeLidarRenderResult(
            depth=depth,
            alpha=torch.ones_like(depth),
            visibility=torch.ones(
                (gaussian_state.positions.shape[0],),
                device=depth.device,
                dtype=torch.bool,
            ),
        )

    captured: dict[str, object] = {}

    def dump(**kwargs: object) -> None:
        captured.update(kwargs)

    model = Model()
    initial_position = model.positions.detach().clone()
    trainer = object.__new__(Trainer3DGRUT)
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.lidar_ray_sampler = Sampler()
    trainer.conf = OmegaConf.create({"enable_writer": False})
    trainer._dump_native_lidar_gradient_probe = dump
    conf = OmegaConf.create(
        {
            "lidar_supervision": {
                "frequency": 1,
                "rays_per_step": 0,
                "renderer": "classic_expected_depth",
                "min_depth_m": 0.1,
                "lambda_range": 1.0,
            }
        }
    )
    monkeypatch.setattr(
        trainer_module,
        "render_classic_expected_depth",
        fake_classic_depth,
    )

    loss = trainer.run_lidar_step(1, conf)

    assert loss is not None
    assert not model.called
    assert captured["outputs"] == {}
    assert torch.equal(captured["skip_mask"], model.environment_mask)
    assert not torch.equal(model.positions.detach(), initial_position)


def test_native_projected_covariance_gate_uses_the_unregularized_det() -> None:
    """The native 0.005 gate is evaluated before its 0.1 diagonal offset."""
    regularized_covariances = torch.tensor(
        (
            (0.2, 0.0, 0.2),
            (0.15, 0.0, 0.15),
        )
    )
    conics = torch.stack(
        (
            1.0 / regularized_covariances[:, 0],
            regularized_covariances[:, 1],
            1.0 / regularized_covariances[:, 2],
        ),
        dim=-1,
    )

    valid = native_projected_covariance_is_valid(conics)

    assert torch.equal(valid, torch.tensor((True, False)))


def test_sparse_expected_depth_evaluates_continuous_query_coordinates() -> None:
    """The sparse renderer uses the LiDAR float2 directly, not a raster sample."""
    means2d = torch.tensor(
        (((1.5, 1.5), (1.5, 1.5)),),
        requires_grad=True,
    )
    depths = torch.tensor(((2.0, 4.0),), requires_grad=True)
    conics = torch.tensor(
        (((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)),),
        requires_grad=True,
    )
    opacities = torch.tensor((0.5, 0.5), requires_grad=True)
    sample_grid = torch.tensor([[[[-0.25, -0.25]]]])

    result = native_lidar_renderer._render_native_sparse_expected_depth(
        means2d=means2d,
        depths=depths,
        conics=conics,
        opacities=opacities,
        sample_grid=sample_grid,
        image_size=4,
        tile_width=1,
        tile_height=1,
        isect_offsets=torch.tensor(([[0]],)),
        flatten_ids=torch.tensor((0, 1)),
    )

    assert isinstance(result, NativeLidarRenderResult)
    depth = result.depth
    alpha = result.alpha
    point_alpha = 0.5 * torch.exp(torch.tensor(-0.25))
    expected_alpha = 1.0 - (1.0 - point_alpha).square()
    expected_depth = (
        point_alpha * 2.0 + (1.0 - point_alpha) * point_alpha * 4.0
    ) / expected_alpha
    assert torch.allclose(alpha, expected_alpha.reshape(1, 1, 1, 1))
    assert torch.allclose(depth, expected_depth.reshape(1, 1, 1, 1))

    (depth + alpha).sum().backward()
    assert means2d.grad is not None
    assert torch.isfinite(means2d.grad).all()
    assert depths.grad is None
    assert conics.grad is not None
    assert torch.isfinite(conics.grad).all()
    assert opacities.grad is not None
    assert torch.isfinite(opacities.grad).all()
    assert torch.equal(result.visibility, torch.tensor((True, True)))


def test_sparse_expected_depth_stops_after_low_transmittance_batch() -> None:
    """Later warp-sized batches do not affect an effectively opaque ray."""
    splat_count = 33
    point_alpha = 0.95
    depths = torch.cat(
        (
            torch.tensor((2.0, 4.0, 100.0)),
            torch.full((29,), 1_000.0),
            torch.tensor((1_000_000.0,)),
        )
    )
    result = native_lidar_renderer._render_native_sparse_expected_depth(
        means2d=torch.full((1, splat_count, 2), 1.5),
        depths=depths.unsqueeze(0),
        conics=torch.tensor((1.0, 0.0, 1.0)).repeat(
            1,
            splat_count,
            1,
        ),
        opacities=torch.full((splat_count,), point_alpha),
        sample_grid=torch.tensor([[[[0.0, 0.0]]]]),
        image_size=4,
        tile_width=1,
        tile_height=1,
        isect_offsets=torch.tensor(([[0]],)),
        flatten_ids=torch.arange(splat_count),
    )

    depth = result.depth
    alpha = result.alpha
    active_splat_count = 32
    weights = point_alpha * (1.0 - point_alpha) ** torch.arange(
        active_splat_count
    )
    expected_alpha = weights.sum()
    expected_depth = (weights * depths[:active_splat_count]).sum() / expected_alpha
    assert torch.allclose(alpha, torch.full((1, 1, 1, 1), expected_alpha))
    assert torch.allclose(depth, torch.full((1, 1, 1, 1), expected_depth))
    assert torch.equal(
        result.visibility,
        torch.cat(
            (
                torch.ones(active_splat_count, dtype=torch.bool),
                torch.zeros(1, dtype=torch.bool),
            )
        ),
    )


def test_sparse_expected_depth_uses_native_straight_through_alpha_cap() -> None:
    """The DLL keeps the unclamped opacity derivative at its 0.99 cap."""
    opacities = torch.tensor((2.0,), requires_grad=True)
    result = native_lidar_renderer._render_native_sparse_expected_depth(
        means2d=torch.tensor((((1.5, 1.5),),)),
        depths=torch.tensor(((2.0,),)),
        conics=torch.tensor((((1.0, 0.0, 1.0),),)),
        opacities=opacities,
        sample_grid=torch.tensor([[[[0.0, 0.0]]]]),
        image_size=4,
        tile_width=1,
        tile_height=1,
        isect_offsets=torch.tensor(([[0]],)),
        flatten_ids=torch.tensor((0,)),
    )

    assert torch.allclose(result.alpha, torch.full((1, 1, 1, 1), 0.99))
    result.alpha.sum().backward()

    assert torch.allclose(opacities.grad, torch.ones_like(opacities))


def test_native_lidar_float_tile_bounds_use_compensation_and_truncation() -> None:
    """Float native bounds preserve compensation before tile conversion."""
    lower_x, lower_y, upper_x, upper_y = (
        native_lidar_renderer._native_lidar_float_tile_bounds(
            means2d=torch.tensor((((6.9, 1.5),),)),
            conics=torch.tensor((((5.0, 0.0, 5.0),),)),
            opacities=torch.tensor(((0.5,),)),
            compensations=torch.tensor(((0.05,),)),
            tile_width=3,
            tile_height=2,
        )
    )

    assert torch.equal(lower_x, torch.tensor((1,)))
    assert torch.equal(lower_y, torch.tensor((0,)))
    assert torch.equal(upper_x, torch.tensor((2,)))
    assert torch.equal(upper_y, torch.tensor((1,)))


def test_native_lidar_tile_lists_keep_active_pairs_in_depth_order() -> None:
    """Only active tile intersections remain, ordered by tile then depth."""
    offsets, flatten_ids = native_lidar_renderer._native_lidar_sorted_tile_lists(
        lower_x=torch.tensor((0, 1, 0)),
        lower_y=torch.tensor((0, 0, 1)),
        upper_x=torch.tensor((2, 3, 2)),
        upper_y=torch.tensor((1, 1, 2)),
        depths=torch.tensor((2.0, 1.0, 3.0)),
        culling_mask=torch.tensor((True, True, True)),
        active_tiles=torch.tensor(
            (
                (True, True, False),
                (True, False, True),
            )
        ),
    )

    assert torch.equal(
        offsets,
        torch.tensor(
            (
                (
                    (0, 1, 3),
                    (3, 4, 4),
                ),
            )
        ),
    )
    assert torch.equal(flatten_ids, torch.tensor((0, 1, 0, 2)))


def test_classic_expected_depth_uses_native_sparse_effective_opacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public renderer uses native bins and projection compensation."""
    captured: dict[str, object] = {}

    def fake_projection(
        *args: object,
        **kwargs: object,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del args
        captured["projection_kwargs"] = kwargs
        radii = torch.tensor(([[1, 1], [1, 1]],))
        means2d = torch.full((1, 2, 2), 1.5)
        depths = torch.tensor(((2.5, 3.5),))
        conics = torch.tensor(
            (
                (
                    (5.0, 0.0, 5.0),
                    (5.0, 0.0, 5.0),
                ),
            )
        )
        return radii, means2d, depths, conics, torch.full((1, 2), 0.5)

    monkeypatch.setattr(
        native_lidar_renderer,
        "_load_gsplat_fully_fused_projection",
        lambda: fake_projection,
    )
    positions = torch.tensor(((0.0, 0.0, 2.0), (0.0, 0.0, 3.0)))
    rotations = torch.tensor(
        ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    )
    scales = torch.tensor(((0.1, 0.1, 0.1), (0.1, 0.1, 0.1)))
    opacities = torch.tensor(((0.5,), (0.25,)))
    camera_to_world = torch.eye(4).unsqueeze(0)
    sample_grid = torch.tensor([[[[0.0, 0.0]]]])

    result = render_classic_expected_depth(
        gaussian_state=NativeLidarGaussianState(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
        ),
        camera_to_world=camera_to_world,
        intrinsics=[10.0, 11.0, 2.0, 3.0],
        sample_grid=sample_grid,
        image_size=4,
    )

    assert torch.allclose(
        result.depth,
        torch.full((1, 1, 1, 1), 2.7727273),
    )
    assert torch.allclose(result.alpha, torch.full((1, 1, 1, 1), 0.34375))
    assert torch.equal(result.visibility, torch.tensor((True, True)))
    projection_kwargs = captured["projection_kwargs"]
    assert isinstance(projection_kwargs, dict)
    assert projection_kwargs["calc_compensations"] is True


def test_sparse_depth_renderer_omits_skip_masked_gaussians() -> None:
    """The native skip mask removes a splat after list construction."""
    result = native_lidar_renderer._render_native_sparse_expected_depth(
        means2d=torch.tensor(((((1.5, 1.5), (1.5, 1.5)),))),
        depths=torch.tensor(((2.5, 3.5),)),
        conics=torch.tensor(((((5.0, 0.0, 5.0), (5.0, 0.0, 5.0)),))),
        opacities=torch.tensor((0.5, 0.25)),
        sample_grid=torch.tensor([[[[0.0, 0.0]]]]),
        image_size=4,
        tile_width=1,
        tile_height=1,
        isect_offsets=torch.zeros((1, 1, 1), dtype=torch.long),
        flatten_ids=torch.tensor((0, 1), dtype=torch.long),
        skip_mask=torch.tensor((False, True)),
    )

    assert torch.allclose(result.depth, torch.full((1, 1, 1, 1), 2.5))
    assert torch.allclose(result.alpha, torch.full((1, 1, 1, 1), 0.5))
    assert torch.equal(result.visibility, torch.tensor((True, False)))
