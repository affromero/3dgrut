"""CPU behavior tests for surface-aligned COLMAP initialization."""

import os

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

import threedgrut.model.geometry as geometry_module
from threedgrut.model.geometry import (
    SurfaceAlignedPCAConfig,
    SurfaceAlignedPCAResult,
    surface_aligned_pca_initialize,
)
from threedgrut.model.model import (
    MixtureOfGaussians,
    _validated_surface_aligned_pca_config,
)
from threedgrut.utils.misc import (
    quaternion_to_so3,
    so3_to_quaternion_wxyz,
    to_np,
    to_torch,
)
from threedgrut.utils.render import RGB2SH


def _surface_config(
    *,
    query_chunk_size: int = 65_536,
    max_neighbor_radius_m: float = 0.25,
) -> SurfaceAlignedPCAConfig:
    return SurfaceAlignedPCAConfig(
        num_support_points=32,
        max_neighbor_radius_m=max_neighbor_radius_m,
        max_normal_to_mid_ratio=0.25,
        min_mid_to_max_ratio=0.10,
        min_mid_eigenvalue_m2=1.0e-12,
        min_thickness_ratio=0.125,
        query_chunk_size=query_chunk_size,
    )


def _incumbent_parameters(
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.full((count, 3), 0.1, dtype=torch.float32)
    generator = torch.Generator().manual_seed(91)
    rotations = torch.rand(
        (count, 4),
        dtype=torch.float32,
        generator=generator,
    )
    return scales, rotations


def _plane_points(
    rotation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xy = torch.cartesian_prod(
        torch.linspace(-0.035, 0.035, 8, dtype=torch.float64),
        torch.linspace(-0.035, 0.035, 8, dtype=torch.float64),
    )
    points = torch.column_stack((xy, torch.zeros(xy.shape[0], dtype=torch.float64)))
    normal = torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64)
    if rotation is not None:
        points = points @ rotation.transpose(0, 1)
        normal = rotation[:, 2]
    return points, normal


def _initialize_plane(
    rotation: torch.Tensor | None = None,
    *,
    query_chunk_size: int = 65_536,
) -> tuple[SurfaceAlignedPCAResult, torch.Tensor]:
    points, normal = _plane_points(rotation)
    scales, rotations = _incumbent_parameters(points.shape[0])
    result = surface_aligned_pca_initialize(
        points,
        scales.to(dtype=torch.float64),
        rotations.to(dtype=torch.float64),
        _surface_config(query_chunk_size=query_chunk_size),
    )
    return result, normal


def _assert_surface_covariance(
    result: SurfaceAlignedPCAResult,
    normal: torch.Tensor,
) -> None:
    assert bool(result.reliable_mask.all())
    rotations = quaternion_to_so3(result.raw_rotations_wxyz)
    determinants = torch.linalg.det(rotations)
    torch.testing.assert_close(
        determinants,
        torch.ones_like(determinants),
        atol=1.0e-12,
        rtol=0.0,
    )
    recovered_normals = rotations[:, :, 2]
    alignment = torch.abs(recovered_normals @ normal)
    torch.testing.assert_close(
        alignment,
        torch.ones_like(alignment),
        atol=1.0e-12,
        rtol=0.0,
    )
    incumbent_scale = torch.tensor(0.1, dtype=torch.float32).double()
    expected_scales = torch.stack(
        (
            incumbent_scale,
            incumbent_scale,
            0.125 * incumbent_scale,
        )
    ).repeat(result.physical_scales.shape[0], 1)
    torch.testing.assert_close(
        result.physical_scales,
        expected_scales,
        atol=1.0e-12,
        rtol=0.0,
    )

    scale_matrices = torch.diag_embed(result.physical_scales.square())
    covariance = rotations @ scale_matrices @ rotations.transpose(1, 2)
    normal_variance = torch.einsum(
        "i,nij,j->n",
        normal,
        covariance,
        normal,
    )
    torch.testing.assert_close(
        normal_variance,
        expected_scales[:, 2].square(),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_so3_wxyz_roundtrip_is_canonical_and_proper() -> None:
    generator = torch.Generator().manual_seed(7)
    quaternions = torch.randn(
        (128, 4),
        dtype=torch.float64,
        generator=generator,
    )
    half_turns = torch.tensor(
        (
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=torch.float64,
    )
    quaternions = torch.cat((quaternions, half_turns), dim=0)
    quaternions = torch.nn.functional.normalize(quaternions, dim=1)
    rotations = quaternion_to_so3(quaternions)

    recovered = so3_to_quaternion_wxyz(rotations)
    roundtrip = quaternion_to_so3(recovered)

    torch.testing.assert_close(
        roundtrip,
        rotations,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        torch.linalg.det(roundtrip),
        torch.ones(roundtrip.shape[0], dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    for quaternion in recovered:
        first_nonzero = quaternion[quaternion != 0.0][0]
        assert float(first_nonzero) > 0.0


def test_axis_aligned_plane_uses_local_z_for_the_thin_axis() -> None:
    result, normal = _initialize_plane()

    _assert_surface_covariance(result, normal)
    assert result.audit.reliable_point_count == 64
    assert result.audit.flattened_point_count == 64
    assert result.audit.anisotropy_quantiles == (8.0, 8.0, 8.0)


def test_rotated_plane_uses_local_z_for_the_thin_axis() -> None:
    axis = torch.tensor((1.0, -2.0, 0.5), dtype=torch.float64)
    axis = axis / torch.linalg.norm(axis)
    angle = torch.tensor(0.73, dtype=torch.float64)
    quaternion = torch.cat(
        (
            torch.cos(angle / 2.0).reshape(1),
            axis * torch.sin(angle / 2.0),
        )
    ).reshape(1, 4)
    rotation = quaternion_to_so3(quaternion)[0]

    result, normal = _initialize_plane(rotation)

    _assert_surface_covariance(result, normal)


def test_unreliable_geometries_preserve_incumbent_values_exactly() -> None:
    cube_corners = torch.cartesian_prod(
        torch.tensor((-0.05, 0.05)),
        torch.tensor((-0.05, 0.05)),
        torch.tensor((-0.05, 0.05)),
    ).repeat_interleave(4, dim=0)
    line = torch.column_stack(
        (
            torch.linspace(-0.1, 0.1, 40),
            torch.zeros(40),
            torch.zeros(40),
        )
    )
    duplicates = torch.zeros((40, 3))

    for points in (cube_corners, line, duplicates):
        scales, rotations = _incumbent_parameters(points.shape[0])
        result = surface_aligned_pca_initialize(
            points,
            scales,
            rotations,
            _surface_config(),
        )

        assert not bool(result.reliable_mask.any())
        assert torch.equal(result.physical_scales, scales)
        assert torch.equal(result.raw_rotations_wxyz, rotations)


def test_sparse_and_nonfinite_rows_remain_unmodified() -> None:
    sparse_points = torch.zeros((31, 3))
    sparse_scales, sparse_rotations = _incumbent_parameters(31)
    sparse_result = surface_aligned_pca_initialize(
        sparse_points,
        sparse_scales,
        sparse_rotations,
        _surface_config(),
    )
    assert sparse_result.audit.insufficient_support_count == 31
    assert torch.equal(sparse_result.physical_scales, sparse_scales)
    assert torch.equal(
        sparse_result.raw_rotations_wxyz,
        sparse_rotations,
    )

    points, _ = _plane_points()
    points[0, 0] = torch.nan
    scales, rotations = _incumbent_parameters(points.shape[0])
    result = surface_aligned_pca_initialize(
        points,
        scales.to(dtype=torch.float64),
        rotations.to(dtype=torch.float64),
        _surface_config(),
    )
    assert not bool(result.reliable_mask[0])
    assert torch.equal(result.physical_scales[0], scales[0].double())
    assert torch.equal(
        result.raw_rotations_wxyz[0],
        rotations[0].double(),
    )
    assert result.audit.nonfinite_input_count == 1


def test_surface_pca_requires_at_least_three_support_points() -> None:
    points, _ = _plane_points()
    scales, rotations = _incumbent_parameters(points.shape[0])
    config = _surface_config()._replace(num_support_points=2)

    with pytest.raises(ValueError, match="at least 3"):
        surface_aligned_pca_initialize(
            points,
            scales.to(dtype=torch.float64),
            rotations.to(dtype=torch.float64),
            config,
        )


def test_invalid_incumbent_scale_rows_are_preserved_exactly() -> None:
    points, _ = _plane_points()
    scales, rotations = _incumbent_parameters(points.shape[0])
    scales = scales.to(dtype=torch.float64)
    rotations = rotations.to(dtype=torch.float64)
    scales[0, 1] = -0.1
    scales[1, 1] = 0.2
    scales[2, 2] = torch.inf

    result = surface_aligned_pca_initialize(
        points,
        scales,
        rotations,
        _surface_config(),
    )

    assert not bool(result.reliable_mask[:3].any())
    assert torch.equal(result.physical_scales[:3], scales[:3])
    assert torch.equal(
        result.raw_rotations_wxyz[:3],
        rotations[:3],
    )
    assert result.audit.invalid_incumbent_scale_count == 2
    assert result.audit.nonfinite_input_count == 1
    assert bool(result.reliable_mask[3:].all())


def test_k32_counts_anchor_and_accepts_the_exact_radius_boundary() -> None:
    angles = torch.arange(30, dtype=torch.float64) * (2.0 * np.pi / 30.0)
    near_neighbors = torch.column_stack(
        (
            0.1 * torch.cos(angles),
            0.1 * torch.sin(angles),
            torch.zeros(30, dtype=torch.float64),
        )
    )
    points = torch.cat(
        (
            torch.zeros((1, 3), dtype=torch.float64),
            near_neighbors,
            torch.tensor(((0.25, 0.0, 0.0),), dtype=torch.float64),
            torch.tensor(((0.30, 0.0, 0.0),), dtype=torch.float64),
        ),
        dim=0,
    )
    scales, rotations = _incumbent_parameters(points.shape[0])

    result = surface_aligned_pca_initialize(
        points,
        scales.to(dtype=torch.float64),
        rotations.to(dtype=torch.float64),
        _surface_config(max_neighbor_radius_m=0.25),
    )

    assert bool(result.reliable_mask[0])
    assert result.physical_scales[0, 2] == pytest.approx(0.0125)

    below_boundary = surface_aligned_pca_initialize(
        points,
        scales.to(dtype=torch.float64),
        rotations.to(dtype=torch.float64),
        _surface_config(max_neighbor_radius_m=float(np.nextafter(0.25, 0.0))),
    )
    assert not bool(below_boundary.reliable_mask[0])


def test_query_chunk_size_does_not_change_results() -> None:
    small_chunks, _ = _initialize_plane(query_chunk_size=7)
    one_chunk, _ = _initialize_plane(query_chunk_size=65_536)

    assert torch.equal(
        small_chunks.physical_scales,
        one_chunk.physical_scales,
    )
    assert torch.equal(
        small_chunks.raw_rotations_wxyz,
        one_chunk.raw_rotations_wxyz,
    )
    assert torch.equal(
        small_chunks.reliable_mask,
        one_chunk.reliable_mask,
    )
    assert small_chunks.audit == one_chunk.audit


@pytest.mark.parametrize(
    ("negative_eigenvalue", "expected_reliable", "expected_negative_count"),
    (
        (-1.0e-20, True, 0),
        (-1.0e-8, False, 64),
    ),
)
def test_negative_eigenvalues_only_clamp_float64_roundoff(
    monkeypatch: pytest.MonkeyPatch,
    negative_eigenvalue: float,
    expected_reliable: bool,
    expected_negative_count: int,
) -> None:
    original_eigh = geometry_module.np.linalg.eigh

    def injected_eigh(
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = original_eigh(covariance)
        eigenvalues = eigenvalues.copy()
        eigenvalues[:, 0] = negative_eigenvalue
        return eigenvalues, eigenvectors

    monkeypatch.setattr(geometry_module.np.linalg, "eigh", injected_eigh)

    result, _ = _initialize_plane()

    assert bool(result.reliable_mask.all()) is expected_reliable
    assert result.audit.materially_negative_eigenvalue_count == expected_negative_count


def _initialization_config() -> DictConfig:
    return OmegaConf.create(
        {
            "method": "colmap",
            "use_observation_points": True,
            "surface_aligned_pca": {
                "enabled": True,
                "num_support_points": 32,
                "max_neighbor_radius_m": 0.25,
                "max_normal_to_mid_ratio": 0.25,
                "min_mid_to_max_ratio": 0.10,
                "min_mid_eigenvalue_m2": 1.0e-12,
                "min_thickness_ratio": 0.125,
                "query_chunk_size": 65_536,
                "expected_point_count": 32,
                "expected_observer_count": 2,
            },
        }
    )


def test_enabled_validation_accepts_a_valid_colmap_contract() -> None:
    points = torch.zeros((32, 3))
    observers = torch.zeros((2, 3))
    config = _initialization_config()

    validated = _validated_surface_aligned_pca_config(
        config,
        points,
        observers,
    )

    assert validated == _surface_config()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (("method", "random"), "method=colmap"),
        (("use_observation_points", False), "use_observation_points=true"),
        (
            ("surface_aligned_pca.expected_point_count", 33),
            "expected_point_count mismatch",
        ),
        (
            ("surface_aligned_pca.expected_observer_count", 3),
            "expected_observer_count mismatch",
        ),
        (
            ("surface_aligned_pca.max_neighbor_radius_m", -0.25),
            "max_neighbor_radius_m must be finite and positive",
        ),
        (
            ("surface_aligned_pca.min_thickness_ratio", 1.1),
            "min_thickness_ratio must be finite and in",
        ),
        (
            ("surface_aligned_pca.num_support_points", 31.5),
            "num_support_points must be a finite integer",
        ),
        (
            ("surface_aligned_pca.num_support_points", 2),
            "num_support_points must be at least 3",
        ),
        (
            ("surface_aligned_pca.expected_point_count", 32.5),
            "expected_point_count must be null or a non-negative integer",
        ),
    ),
)
def test_enabled_validation_rejects_contract_mismatches(
    mutation: tuple[str, object],
    message: str,
) -> None:
    config = _initialization_config()
    OmegaConf.update(config, mutation[0], mutation[1])

    with pytest.raises(ValueError, match=message):
        _validated_surface_aligned_pca_config(
            config,
            torch.zeros((32, 3)),
            torch.zeros((2, 3)),
        )


def test_enabled_validation_rejects_bad_observers_and_sparse_points() -> None:
    config = _initialization_config()
    with pytest.raises(ValueError, match="nonempty finite observer"):
        _validated_surface_aligned_pca_config(
            config,
            torch.zeros((32, 3)),
            torch.tensor(((torch.nan, 0.0, 0.0), (0.0, 0.0, 0.0))),
        )

    config.surface_aligned_pca.expected_point_count = None
    with pytest.raises(ValueError, match="at least 32 COLMAP points"):
        _validated_surface_aligned_pca_config(
            config,
            torch.zeros((31, 3)),
            torch.zeros((2, 3)),
        )


def test_disabled_validation_is_a_noop() -> None:
    config = _initialization_config()
    config.surface_aligned_pca.enabled = False
    config.method = "random"
    config.use_observation_points = False

    validated = _validated_surface_aligned_pca_config(
        config,
        torch.empty((0, 3)),
        torch.empty((0, 3)),
    )

    assert validated is None


def test_composed_colmap_config_has_exact_default_off_contract() -> None:
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="apps/colmap_3dgut")

    surface_config = OmegaConf.to_container(
        config.initialization.surface_aligned_pca,
        resolve=True,
    )
    assert surface_config == {
        "enabled": False,
        "num_support_points": 32,
        "max_neighbor_radius_m": 0.25,
        "max_normal_to_mid_ratio": 0.25,
        "min_mid_to_max_ratio": 0.10,
        "min_mid_eigenvalue_m2": 1.0e-12,
        "min_thickness_ratio": 0.125,
        "query_chunk_size": 65_536,
        "expected_point_count": None,
        "expected_observer_count": None,
    }


class _InitializerHarness:
    def __init__(self) -> None:
        self.positions: torch.nn.Parameter
        self.rotation: torch.nn.Parameter
        self.scale: torch.nn.Parameter
        self.density: torch.nn.Parameter
        self.features_albedo: torch.nn.Parameter
        self.features_specular: torch.nn.Parameter
        self.scene_extent: float
        self.device = torch.device("cpu")
        self.max_n_features = 0
        self.conf = OmegaConf.create(
            {
                "seed_initialization": 42,
                "initialization": {"observation_scale_factor": 0.01},
                "model": {
                    "default_scale_factor": 1.0,
                    "default_density": 0.1,
                    "use_gabor_carrier": False,
                    "use_siren_carrier": False,
                },
            }
        )
        self.scale_activation_inv = torch.log
        self.density_activation_inv = lambda value: torch.log(value / (1.0 - value))

    def ensure_scene_extent_from_points(
        self,
        positions: torch.Tensor,
    ) -> None:
        self.scene_extent = float(torch.linalg.norm(positions, dim=1).max())

    def _specular_feature_dim(self) -> int:
        return 0

    def set_optimizable_parameters(self) -> None:
        return None

    def setup_optimizer(self) -> None:
        return None

    def validate_fields(self) -> None:
        return None


def _run_initializer(
    points: torch.Tensor,
    observers: torch.Tensor,
    surface_config: SurfaceAlignedPCAConfig | None,
) -> _InitializerHarness:
    harness = _InitializerHarness()
    if surface_config is None:
        MixtureOfGaussians.default_initialize_from_points(
            harness,
            points,
            observers,
            colors=None,
            use_observer_pts=True,
        )
    else:
        MixtureOfGaussians.default_initialize_from_points(
            harness,
            points,
            observers,
            colors=None,
            use_observer_pts=True,
            surface_aligned_pca_config=surface_config,
        )
    return harness


def _assert_initialized_fields_equal(
    left: _InitializerHarness,
    right: _InitializerHarness,
) -> None:
    field_names = (
        "positions",
        "rotation",
        "scale",
        "density",
        "features_albedo",
        "features_specular",
    )
    for field_name in field_names:
        assert torch.equal(
            getattr(left, field_name),
            getattr(right, field_name),
        )


def test_disabled_path_is_bitwise_and_does_not_advance_global_rng() -> None:
    points, _ = _plane_points()
    observers = torch.tensor(
        ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        dtype=torch.float64,
    )
    torch.manual_seed(808)
    global_rng_before = torch.get_rng_state().clone()

    first = _run_initializer(points, observers, None)
    global_rng_after_first = torch.get_rng_state().clone()
    second = _run_initializer(points, observers, None)
    global_rng_after_second = torch.get_rng_state().clone()

    _assert_initialized_fields_equal(first, second)
    assert torch.equal(global_rng_before, global_rng_after_first)
    assert torch.equal(global_rng_before, global_rng_after_second)

    expected_generator = torch.Generator().manual_seed(42)
    expected_rotations = torch.rand(
        (points.shape[0], 4),
        dtype=torch.float32,
        generator=expected_generator,
    )
    expected_colors = torch.randint(
        0,
        256,
        (points.shape[0], 3),
        dtype=torch.uint8,
        generator=expected_generator,
    )
    observer_distances = torch.linalg.norm(
        points - observers[0],
        dim=1,
    )
    expected_scales = torch.log(torch.clamp_min(observer_distances, 1.0e-7) * 0.01)[:, None].repeat(1, 3).float()
    expected_density_values = torch.full(
        (points.shape[0], 1),
        0.1,
        dtype=torch.float32,
    )
    expected_density = torch.log(expected_density_values / (1.0 - expected_density_values))
    expected_albedo = to_torch(
        RGB2SH(to_np(expected_colors.float() / 255.0)),
        device="cpu",
    ).float()

    assert torch.equal(first.positions, points.float())
    assert torch.equal(first.rotation, expected_rotations)
    assert torch.equal(first.scale, expected_scales)
    assert torch.equal(first.density, expected_density)
    assert torch.equal(first.features_albedo, expected_albedo)


def test_enabled_unreliable_rows_preserve_rng_and_all_incumbent_values() -> None:
    points = torch.column_stack(
        (
            torch.linspace(-0.1, 0.1, 40),
            torch.zeros(40),
            torch.zeros(40),
        )
    )
    observers = torch.tensor(((0.0, 1.0, 0.0),))

    incumbent = _run_initializer(points, observers, None)
    candidate = _run_initializer(points, observers, _surface_config())

    _assert_initialized_fields_equal(incumbent, candidate)


def test_enabled_initializer_changes_only_reliable_shape_parameters() -> None:
    points, _ = _plane_points()
    observers = torch.tensor(
        ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        dtype=torch.float64,
    )

    incumbent = _run_initializer(points, observers, None)
    candidate = _run_initializer(points, observers, _surface_config())

    invariant_fields = (
        "positions",
        "density",
        "features_albedo",
        "features_specular",
    )
    for field_name in invariant_fields:
        assert torch.equal(
            getattr(incumbent, field_name),
            getattr(candidate, field_name),
        )
    assert not torch.equal(incumbent.rotation, candidate.rotation)
    assert not torch.equal(incumbent.scale, candidate.scale)
    physical_scales = torch.exp(candidate.scale)
    torch.testing.assert_close(
        physical_scales[:, :2],
        torch.exp(incumbent.scale[:, :2]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        physical_scales[:, 2],
        0.125 * physical_scales[:, 0],
        atol=1.0e-8,
        rtol=1.0e-7,
    )
