import pytest
import torch

from threedgrut.model.model import (
    PROTECTED_GAUSSIAN_PREFIX_VERSION,
    MixtureOfGaussians,
    tensor_prefix_sha256,
)
from threedgrut.strategy.base import BaseStrategy
from threedgrut.strategy.gs import (
    exclude_protected_prefix,
    retain_protected_prefix,
)


def _model_with_protected_prefix() -> MixtureOfGaussians:
    model = MixtureOfGaussians.__new__(MixtureOfGaussians)
    torch.nn.Module.__init__(model)
    model.positions = torch.nn.Parameter(
        torch.arange(15, dtype=torch.float32).reshape(5, 3)
    )
    model.rotation = torch.nn.Parameter(
        torch.arange(20, dtype=torch.float32).reshape(5, 4)
    )
    model.scale = torch.nn.Parameter(
        torch.arange(15, dtype=torch.float32).reshape(5, 3)
    )
    model.density = torch.nn.Parameter(torch.zeros((5, 1)))
    model.features_albedo = torch.nn.Parameter(torch.zeros((5, 3)))
    model.features_specular = torch.nn.Parameter(torch.zeros((5, 3)))
    model.protected_gaussian_count = 0
    model._protected_prefix_metadata = {}
    model._protected_gradient_handles = []
    geometry_sha256 = {
        name: tensor_prefix_sha256(getattr(model, name), 2)
        for name in ("positions", "rotation", "scale")
    }
    checkpoint = {
        "protected_gaussian_prefix": {
            "version": PROTECTED_GAUSSIAN_PREFIX_VERSION,
            "count": 2,
            "geometry_sha256": geometry_sha256,
            "selection_sha256": "selection",
        }
    }
    model._load_protected_prefix_metadata(checkpoint)
    return model


def test_protected_prefix_geometry_gradients_are_zero() -> None:
    model = _model_with_protected_prefix()

    loss = model.positions.sum() + model.rotation.sum() + model.scale.sum()
    loss.backward()

    for parameter in (model.positions, model.rotation, model.scale):
        assert torch.count_nonzero(parameter.grad[:2]) == 0
        assert torch.all(parameter.grad[2:] == 1)


def test_parameter_replacement_refreshes_protected_hooks() -> None:
    model = _model_with_protected_prefix()
    model.optimizer = torch.optim.Adam(
        [{"params": [model.positions], "name": "positions"}],
        lr=1e-3,
    )
    strategy = BaseStrategy(config=None, model=model)

    strategy._update_param_with_optimizer(
        lambda _name, parameter: torch.nn.Parameter(parameter.detach().clone()),
        None,
    )
    model.positions.sum().backward()

    assert torch.count_nonzero(model.positions.grad[:2]) == 0
    assert torch.all(model.positions.grad[2:] == 1)


def test_protected_prefix_rejects_geometry_hash_mismatch() -> None:
    model = _model_with_protected_prefix()
    checkpoint = {
        "protected_gaussian_prefix": {
            "version": PROTECTED_GAUSSIAN_PREFIX_VERSION,
            "count": 2,
            "geometry_sha256": {
                "positions": "wrong",
                "rotation": tensor_prefix_sha256(model.rotation, 2),
                "scale": tensor_prefix_sha256(model.scale, 2),
            },
        }
    }

    with pytest.raises(ValueError, match="positions"):
        model._load_protected_prefix_metadata(checkpoint)


def test_protected_optimizer_moments_must_remain_zero() -> None:
    model = _model_with_protected_prefix()
    model.optimizer = torch.optim.Adam(
        [
            {"params": [model.positions], "name": "positions"},
            {"params": [model.rotation], "name": "rotation"},
            {"params": [model.scale], "name": "scale"},
        ],
        lr=1e-3,
    )
    (model.positions.sum() + model.rotation.sum() + model.scale.sum()).backward()
    model.optimizer.step()
    model.validate_protected_optimizer_state()

    state = model.optimizer.state[model.positions]
    state["exp_avg"][0, 0] = 1
    with pytest.raises(ValueError, match="positions.exp_avg"):
        model.validate_protected_optimizer_state()


def test_topology_masks_preserve_protected_prefix() -> None:
    candidates = torch.tensor([True, True, True, False])
    retained = torch.tensor([False, False, True, False])

    assert exclude_protected_prefix(candidates, 2).tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert retain_protected_prefix(retained, 2).tolist() == [
        True,
        True,
        True,
        False,
    ]
