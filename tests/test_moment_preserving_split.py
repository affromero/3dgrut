"""Behavior tests for the opt-in Gauss-Hermite split operator."""

import math
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf
from threedgrut.optimizers import SelectiveAdam
from threedgrut.optimizers.visibility_decayed_adam import (
    VisibilityDecayedAdam,
)
from threedgrut.strategy.gs import GSStrategy
from threedgrut.strategy.moment_preserving_split import (
    GAUSS_HERMITE_CHILD_COUNT,
    GAUSS_HERMITE_WEIGHTS,
    gauss_hermite_split_children,
)
from threedgrut.utils.misc import quaternion_to_so3

_BETA = 0.390625


class _StrategyModel:
    """Small CPU model exposing the optimizer-backed strategy contract."""

    def __init__(
        self,
        *,
        optimizer_kind: str,
        protected_gaussian_count: int = 0,
    ) -> None:
        self.device = torch.device("cpu")
        self.protected_gaussian_count = protected_gaussian_count
        self.positions = torch.nn.Parameter(
            torch.tensor(
                (
                    (0.0, 0.0, 0.0),
                    (5.0, 0.0, 0.0),
                )
            )
        )
        self.density = torch.nn.Parameter(
            torch.logit(torch.tensor(((0.4,), (0.2,))))
        )
        self.features_albedo = torch.nn.Parameter(
            torch.tensor(
                (
                    (1.0, 2.0, 3.0),
                    (4.0, 5.0, 6.0),
                )
            )
        )
        self.features_specular = torch.nn.Parameter(
            torch.tensor(
                (
                    (7.0, 8.0),
                    (9.0, 10.0),
                )
            )
        )
        half_sqrt = math.sqrt(0.5)
        self.rotation = torch.nn.Parameter(
            torch.tensor(
                (
                    (half_sqrt, 0.0, 0.0, half_sqrt),
                    (1.0, 0.0, 0.0, 0.0),
                )
            )
        )
        self.scale = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    (
                        (0.5, 2.0, 0.25),
                        (0.1, 0.1, 0.1),
                    )
                )
            )
        )
        groups = [
            {
                "params": [getattr(self, name)],
                "name": name,
                "lr": 0.001,
            }
            for name in self.parameter_names
        ]
        if optimizer_kind == "visibility":
            self.optimizer = VisibilityDecayedAdam(groups)
        elif optimizer_kind == "selective":
            with patch("threedgrut.optimizers.load_optimizer_plugin"):
                self.optimizer = SelectiveAdam(groups)
        elif optimizer_kind == "adam":
            self.optimizer = torch.optim.Adam(groups)
        else:
            message = f"Unknown optimizer kind: {optimizer_kind}."
            raise ValueError(message)
        self._populate_optimizer_state(optimizer_kind)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return all per-Gaussian optimizer group names."""
        return (
            "positions",
            "density",
            "features_albedo",
            "features_specular",
            "rotation",
            "scale",
        )

    @property
    def num_gaussians(self) -> int:
        """Return the current Gaussian population."""
        return self.positions.shape[0]

    def _populate_optimizer_state(self, optimizer_kind: str) -> None:
        """Seed nonzero row state so topology updates can be audited."""
        for name in self.parameter_names:
            parameter = getattr(self, name)
            state = self.optimizer.state[parameter]
            if optimizer_kind != "visibility":
                state["step"] = torch.tensor(3.0)
            state_dtype = (
                torch.float16
                if optimizer_kind == "visibility"
                and name in {"features_albedo", "features_specular"}
                else parameter.dtype
            )
            row_value = torch.tensor(
                (1.0, 2.0),
                dtype=state_dtype,
            ).reshape((2,) + (1,) * (parameter.ndim - 1))
            state["exp_avg"] = row_value.expand_as(parameter).clone()
            state["exp_avg_sq"] = (
                (3.0 * row_value).expand_as(parameter).clone()
            )
            if optimizer_kind == "visibility" and name == "positions":
                state["gaussian_steps"] = torch.tensor(
                    (11, 22),
                    dtype=torch.int32,
                )

    def get_scale(self) -> torch.Tensor:
        """Return physical standard deviations."""
        return self.scale_activation(self.scale)

    def get_positions(self) -> torch.Tensor:
        """Return physical Gaussian centers."""
        return self.positions

    def get_rotation(self) -> torch.Tensor:
        """Return normalized quaternions."""
        return torch.nn.functional.normalize(self.rotation, dim=1)

    def get_density(self) -> torch.Tensor:
        """Return physical alpha values."""
        return torch.sigmoid(self.density)

    @staticmethod
    def scale_activation(value: torch.Tensor) -> torch.Tensor:
        """Map log-scale parameters to physical standard deviations."""
        return torch.exp(value)

    @staticmethod
    def scale_activation_inv(value: torch.Tensor) -> torch.Tensor:
        """Map physical standard deviations to log-scale parameters."""
        return torch.log(value)

    @staticmethod
    def density_activation_inv(value: torch.Tensor) -> torch.Tensor:
        """Map physical alpha values to logits."""
        return torch.logit(value)

    @staticmethod
    def refresh_protected_gradient_hooks() -> None:
        """Keep the no-op protected-prefix hook contract."""


def _strategy(
    *,
    enabled: bool,
    optimizer_kind: str = "adam",
    beta: float = _BETA,
    projected_extent_enabled: bool = False,
    cancellation_conditioned_enabled: bool = False,
    protected_gaussian_count: int = 0,
) -> tuple[GSStrategy, _StrategyModel]:
    """Build a matched ordinary or moment-preserving strategy."""
    model = _StrategyModel(
        optimizer_kind=optimizer_kind,
        protected_gaussian_count=protected_gaussian_count,
    )
    config = OmegaConf.create(
        {
            "render": {"method": "3dgut"},
            "model": {
                "density_activation": "sigmoid",
                "scale_activation": "exp",
            },
            "strategy": {
                "print_stats": False,
                "densify": {
                    "frequency": 300,
                    "start_iteration": 500,
                    "end_iteration": 15_000,
                    "clone_grad_threshold": 0.2,
                    "split_grad_threshold": 0.4,
                    "relative_size_threshold": 1.0,
                    "split": {"n_gaussians": 2},
                    "moment_preserving_split": {
                        "enabled": enabled,
                        "beta": beta,
                    },
                    "theta_aware": {"enabled": False},
                    "feature_grad": {"enabled": False},
                    "tile_coverage_weighted_gradient": {"enabled": False},
                    "absolute_ray_gradient_diagnostics": {"enabled": False},
                    "absolute_ray_gradient_densification": {"enabled": False},
                    "cancellation_conditioned_split": {
                        "enabled": cancellation_conditioned_enabled,
                        "cancellation_threshold": 0.5,
                        "extent_px": 8.0,
                        "min_joint_observations": 2,
                        "min_joint_fraction": 0.5,
                        "max_reroute_fraction": 0.75,
                    },
                    "projected_extent_split": {
                        "enabled": projected_extent_enabled,
                        "max_px": 8.0,
                    },
                    "covariance_gradient_split": {"enabled": False},
                    "scale_shape_split": {"enabled": False},
                },
                "prune": {"density_threshold": 0.005},
                "reset_density": {"new_max_density": 0.01},
                "scale_guard": {"enabled": False},
            },
        }
    )
    strategy = GSStrategy(config, model)
    strategy.init_densification_buffer()
    return strategy, model


def _physical_covariance(
    scales: torch.Tensor,
    rotations: torch.Tensor,
) -> torch.Tensor:
    """Construct covariance from standard deviations and quaternions."""
    rotation_matrix = quaternion_to_so3(rotations)
    diagonal = torch.diag_embed(scales.square())
    return rotation_matrix @ diagonal @ rotation_matrix.transpose(1, 2)


def test_children_preserve_weighted_mean_covariance_and_fourth_moment() -> (
    None
):
    """Rotated and unsorted parents retain moments through degree five."""
    dtype = torch.float64
    positions = torch.tensor(
        ((1.0, 2.0, 3.0), (-2.0, 0.5, 4.0)),
        dtype=dtype,
    )
    scales = torch.tensor(
        ((0.5, 2.0, 0.25), (1.001, 1.0, 0.999)),
        dtype=dtype,
    )
    half_sqrt = math.sqrt(0.5)
    rotations = torch.tensor(
        (
            (half_sqrt, 0.0, 0.0, half_sqrt),
            (1.0, 0.0, 0.0, 0.0),
        ),
        dtype=dtype,
    )
    opacities = torch.tensor(((0.4,), (0.8,)), dtype=dtype)

    children = gauss_hermite_split_children(
        positions=positions,
        physical_scales=scales,
        rotations=rotations,
        physical_opacities=opacities,
        beta=_BETA,
    )

    parent_count = positions.shape[0]
    weights = torch.tensor(
        GAUSS_HERMITE_WEIGHTS,
        dtype=dtype,
    )[:, None, None]
    child_positions = children.positions.reshape(
        GAUSS_HERMITE_CHILD_COUNT,
        parent_count,
        3,
    )
    child_scales = children.scales.reshape(
        GAUSS_HERMITE_CHILD_COUNT,
        parent_count,
        3,
    )
    weighted_mean = (weights * child_positions).sum(dim=0)
    torch.testing.assert_close(weighted_mean, positions)

    repeated_rotations = rotations.repeat(
        GAUSS_HERMITE_CHILD_COUNT,
        1,
    )
    child_covariance = _physical_covariance(
        children.scales,
        repeated_rotations,
    ).reshape(GAUSS_HERMITE_CHILD_COUNT, parent_count, 3, 3)
    displacement = child_positions - positions[None]
    between_covariance = (
        displacement[..., :, None] * displacement[..., None, :]
    )
    mixture_covariance = (
        weights[..., None] * (child_covariance + between_covariance)
    ).sum(dim=0)
    parent_covariance = _physical_covariance(scales, rotations)
    torch.testing.assert_close(
        mixture_covariance,
        parent_covariance,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    largest_axis = scales.argmax(dim=1)
    local_axis = torch.nn.functional.one_hot(
        largest_axis,
        num_classes=3,
    ).to(dtype=dtype)
    world_axis = torch.bmm(
        quaternion_to_so3(rotations),
        local_axis.unsqueeze(-1),
    ).squeeze(-1)
    axial_displacement = (displacement * world_axis[None]).sum(dim=2)
    child_axial_variance = (
        child_scales.gather(
            2,
            largest_axis[None, :, None].expand(
                GAUSS_HERMITE_CHILD_COUNT,
                -1,
                -1,
            ),
        ).squeeze(2)
        ** 2
    )
    fourth_moment = (
        weights.squeeze(-1)
        * (
            axial_displacement**4
            + 6.0 * axial_displacement.square() * child_axial_variance
            + 3.0 * child_axial_variance.square()
        )
    ).sum(dim=0)
    parent_axial_variance = (
        scales.gather(
            1,
            largest_axis[:, None],
        ).squeeze(1)
        ** 2
    )
    torch.testing.assert_close(
        fourth_moment,
        3.0 * parent_axial_variance.square(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_child_scale_uses_sqrt_beta_and_preserves_transverse_scales() -> None:
    """Beta is a variance fraction, not a standard-deviation multiplier."""
    positions = torch.zeros((1, 3), dtype=torch.float64)
    scales = torch.tensor(((0.5, 2.0, 0.25),), dtype=torch.float64)
    rotations = torch.tensor(
        ((1.0, 0.0, 0.0, 0.0),),
        dtype=torch.float64,
    )
    opacities = torch.tensor(((0.5,),), dtype=torch.float64)

    children = gauss_hermite_split_children(
        positions=positions,
        physical_scales=scales,
        rotations=rotations,
        physical_opacities=opacities,
        beta=_BETA,
    )

    expected = torch.tensor(
        ((0.5, 1.25, 0.25),),
        dtype=torch.float64,
    ).repeat(GAUSS_HERMITE_CHILD_COUNT, 1)
    torch.testing.assert_close(children.scales, expected)


def test_child_alpha_preserves_colocated_transmittance() -> None:
    """Optical-depth weights preserve parent alpha at the same location."""
    dtype = torch.float64
    opacities = torch.tensor(
        ((0.0,), (1.0e-8,), (0.2,), (0.95,), (0.999999,)),
        dtype=dtype,
    )
    parent_count = opacities.shape[0]
    children = gauss_hermite_split_children(
        positions=torch.zeros((parent_count, 3), dtype=dtype),
        physical_scales=torch.ones((parent_count, 3), dtype=dtype),
        rotations=torch.tensor(
            ((1.0, 0.0, 0.0, 0.0),),
            dtype=dtype,
        ).repeat(parent_count, 1),
        physical_opacities=opacities,
        beta=_BETA,
    )

    child_opacity = children.opacities.reshape(
        GAUSS_HERMITE_CHILD_COUNT,
        parent_count,
        1,
    )
    child_transmittance = (1.0 - child_opacity).prod(dim=0)
    torch.testing.assert_close(
        child_transmittance,
        1.0 - opacities,
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    positive_parent = opacities.squeeze(1) > 0.0
    child_optical_depth = -torch.log1p(-child_opacity[:, positive_parent])
    parent_optical_depth = -torch.log1p(-opacities[positive_parent])
    optical_depth_fraction = (
        child_optical_depth / parent_optical_depth[None]
    ).squeeze(2)
    expected_fraction = torch.tensor(
        GAUSS_HERMITE_WEIGHTS,
        dtype=dtype,
    )[:, None].expand_as(optical_depth_fraction)
    torch.testing.assert_close(
        optical_depth_fraction,
        expected_fraction,
        rtol=1.0e-12,
        atol=1.0e-15,
    )


@pytest.mark.parametrize(
    "optimizer_kind",
    ["adam", "selective", "visibility"],
)
def test_strategy_split_preserves_rows_and_zero_initializes_children(
    optimizer_kind: str,
) -> None:
    """Every supported optimizer keeps retained state and clean child rows."""
    strategy, model = _strategy(
        enabled=True,
        optimizer_kind=optimizer_kind,
    )
    old_parameter_state = {
        name: {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in model.optimizer.state[
                getattr(model, name)
            ].items()
        }
        for name in model.parameter_names
    }
    rng_before = torch.random.get_rng_state()

    strategy.split_gaussians(
        densify_grad_norm=torch.tensor((1.0, 0.0)),
        scene_extent=1.0,
    )

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert model.num_gaussians == 4
    for name in model.parameter_names:
        parameter = getattr(model, name)
        assert parameter.shape[0] == 4
        state = model.optimizer.state[parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            assert state[key].shape[0] == 4
            torch.testing.assert_close(
                state[key][0],
                old_parameter_state[name][key][1],
            )
            assert not bool(state[key][1:].any())
    position_state = model.optimizer.state[model.positions]
    if optimizer_kind == "visibility":
        torch.testing.assert_close(
            position_state["gaussian_steps"],
            torch.tensor((22, 0, 0, 0), dtype=torch.int32),
        )
        model.optimizer.step(torch.ones(4, dtype=torch.bool))
    else:
        for name in model.parameter_names:
            getattr(model, name).grad = torch.zeros_like(getattr(model, name))
        if optimizer_kind == "selective":
            torch.optim.Adam.step(model.optimizer)
        else:
            model.optimizer.step()

    torch.testing.assert_close(
        model.features_albedo[1:],
        torch.tensor(((1.0, 2.0, 3.0),)).repeat(3, 1),
    )
    torch.testing.assert_close(
        model.get_scale()[1:],
        torch.tensor(((0.5, 1.25, 0.25),)).repeat(3, 1),
    )
    child_transmittance = (1.0 - model.get_density()[1:]).prod()
    torch.testing.assert_close(
        child_transmittance,
        torch.tensor(0.6),
    )


def test_enabled_split_keeps_protected_parent_and_candidate_thresholds() -> (
    None
):
    """The operator cannot bypass candidate or protected-prefix selection."""
    strategy, model = _strategy(
        enabled=True,
        protected_gaussian_count=1,
    )
    strategy.split_gaussians(
        densify_grad_norm=torch.tensor((1.0, 0.0)),
        scene_extent=1.0,
    )

    assert model.num_gaussians == 2
    torch.testing.assert_close(
        model.positions,
        torch.tensor(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
    )


def test_cancellation_rerouting_uses_moment_preserving_children() -> None:
    """Conflicting small Gaussians receive deterministic optical-depth splits."""
    strategy, model = _strategy(
        enabled=True,
        cancellation_conditioned_enabled=True,
    )
    strategy.densify_cancellation_joint_observations[:] = torch.tensor(
        ((2,), (0,)),
        dtype=torch.int32,
    )
    strategy.densify_cancellation_valid_observations[:] = torch.tensor(
        ((2,), (2,)),
        dtype=torch.int32,
    )
    parent_opacity = model.get_density()[0].detach().clone()

    strategy.split_gaussians(
        densify_grad_norm=torch.tensor((1.0, 1.0)),
        scene_extent=100.0,
        cancellation_joint_observations=(
            strategy.densify_cancellation_joint_observations.clone()
        ),
        cancellation_valid_observations=(
            strategy.densify_cancellation_valid_observations.clone()
        ),
    )

    assert model.num_gaussians == 4
    child_opacity = model.get_density()[1:]
    child_transmittance = (1.0 - child_opacity).prod()
    torch.testing.assert_close(
        child_transmittance,
        1.0 - parent_opacity.squeeze(),
    )
    torch.testing.assert_close(
        model.features_albedo[1:],
        torch.tensor(((1.0, 2.0, 3.0),)).repeat(3, 1),
    )


@pytest.mark.parametrize("beta", [0.0, 1.0, float("inf"), float("nan")])
def test_enabled_strategy_rejects_invalid_beta(beta: float) -> None:
    """Invalid preregistered variance fractions fail before training."""
    with pytest.raises(ValueError, match="beta"):
        _strategy(enabled=True, beta=beta)


def test_enabled_strategy_rejects_nonordinary_candidate_routing() -> None:
    """The structural arm cannot silently change its split operator."""
    with pytest.raises(ValueError, match="alternate structural split"):
        _strategy(
            enabled=True,
            projected_extent_enabled=True,
        )


def test_default_off_retains_two_child_random_split_and_rng_use() -> None:
    """The old split remains stochastic and emits two children when disabled."""
    strategy, model = _strategy(enabled=False)
    torch.manual_seed(7)
    rng_before = torch.random.get_rng_state()

    strategy.split_gaussians(
        densify_grad_norm=torch.tensor((1.0, 0.0)),
        scene_extent=1.0,
    )

    assert model.num_gaussians == 3
    assert not torch.equal(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(
        model.get_scale()[1:],
        torch.tensor(((0.3125, 1.25, 0.15625),)).repeat(2, 1),
    )
