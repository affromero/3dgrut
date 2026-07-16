"""Tests for the independent sparse geometry optimizer."""

from copy import deepcopy

import pytest
import torch

from threedgrut.optimizers.sparse_geometry_adam import SparseGeometryAdam


def _groups(parameters: dict[str, torch.nn.Parameter]) -> list[dict[str, object]]:
    return [{"name": name, "params": [parameter], "lr": 0.01} for name, parameter in parameters.items()]


def _parameters(
    *,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.nn.Parameter]:
    return {
        "positions": torch.nn.Parameter(torch.zeros((2, 3), dtype=dtype)),
        "density": torch.nn.Parameter(torch.zeros((2, 1), dtype=dtype)),
        "rotation": torch.nn.Parameter(torch.zeros((2, 4), dtype=dtype)),
        "scale": torch.nn.Parameter(torch.zeros((2, 3), dtype=dtype)),
    }


def test_sparse_geometry_adam_updates_only_reached_rows() -> None:
    """Inactive geometry rows and their moment state remain byte-stable."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    loss = sum(parameter[0].sum() for parameter in parameters.values())

    loss.backward()
    optimizer.step()

    for parameter in parameters.values():
        assert torch.all(parameter[0] < 0.0)
        assert torch.equal(parameter[1], torch.zeros_like(parameter[1]))
        state = optimizer.state[parameter]
        assert state["row_steps"].tolist() == [1, 0]
        assert torch.count_nonzero(state["exp_avg"][1]) == 0
        assert torch.count_nonzero(state["exp_avg_sq"][1]) == 0


def test_sparse_geometry_adam_does_not_drift_on_empty_sparse_loss() -> None:
    """An empty sparse batch performs no update from historical moments."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    sum(parameter[0].sum() for parameter in parameters.values()).backward()
    optimizer.step()
    optimizer.zero_grad()
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    zero_loss = sum((parameter * 0.0).sum() for parameter in parameters.values())

    zero_loss.backward()
    optimizer.step()

    for name, parameter in parameters.items():
        assert torch.equal(parameter, before[name])
        assert optimizer.state[parameter]["row_steps"].tolist() == [1, 0]


def test_sparse_geometry_adam_rejects_appearance_groups() -> None:
    """The optimizer cannot accidentally own an appearance parameter."""
    parameters = _parameters()
    groups = _groups(parameters)
    groups.append(
        {
            "name": "features_albedo",
            "params": [torch.nn.Parameter(torch.zeros((2, 3)))],
        }
    )

    with pytest.raises(ValueError, match="requires exactly"):
        SparseGeometryAdam(groups)


def test_sparse_geometry_adam_rejects_duplicate_geometry_groups() -> None:
    """Every geometry role must be owned exactly once."""
    groups = _groups(_parameters())
    groups.append(
        {
            "name": "positions",
            "params": [torch.nn.Parameter(torch.zeros((2, 3)))],
        }
    )

    with pytest.raises(ValueError, match="once each"):
        SparseGeometryAdam(groups)


def test_sparse_geometry_adam_copies_only_primary_geometry_groups() -> None:
    """Independent state inherits scaled geometry rates and excludes color."""
    parameters = _parameters()
    color = torch.nn.Parameter(torch.zeros((2, 3)))
    primary_groups = _groups(parameters)
    primary_groups.append({"name": "features_albedo", "params": [color], "lr": 0.002})
    primary = torch.optim.Adam(primary_groups, betas=(0.8, 0.95), eps=1.0e-7)

    geometry = SparseGeometryAdam.from_primary_optimizer(
        primary,
        learning_rate_scale=0.25,
    )

    assert {group["name"] for group in geometry.param_groups} == set(parameters)
    assert all(group["lr"] == pytest.approx(0.0025) for group in geometry.param_groups)
    assert all(group["betas"] == (0.8, 0.95) for group in geometry.param_groups)
    assert color not in geometry.state


def test_sparse_geometry_adam_matches_adam_for_one_always_active_row() -> None:
    """The row-wise update is standard Adam when support is continuous."""
    sparse_parameters = _parameters()
    reference_parameters = {
        name: torch.nn.Parameter(parameter.detach().clone()) for name, parameter in sparse_parameters.items()
    }
    sparse = SparseGeometryAdam(
        _groups(sparse_parameters),
        betas=(0.8, 0.95),
        eps=1.0e-7,
    )
    reference = torch.optim.Adam(
        _groups(reference_parameters),
        betas=(0.8, 0.95),
        eps=1.0e-7,
    )

    for gradient_scale in (1.0, -0.25, 0.5):
        for parameter in sparse_parameters.values():
            parameter.grad = torch.zeros_like(parameter)
            parameter.grad[0] = gradient_scale
        for parameter in reference_parameters.values():
            parameter.grad = torch.zeros_like(parameter)
            parameter.grad[0] = gradient_scale
        sparse.step()
        reference.step()
        sparse.zero_grad()
        reference.zero_grad()

    for name, parameter in sparse_parameters.items():
        assert torch.allclose(parameter[0], reference_parameters[name][0])
        assert torch.equal(parameter[1], reference_parameters[name][1])


def test_sparse_geometry_adam_resume_preserves_rowwise_state() -> None:
    """Checkpoint restoration produces the same next sparse update."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    sum(parameter[0].sum() for parameter in parameters.values()).backward()
    optimizer.step()

    restored_parameters = {
        name: torch.nn.Parameter(parameter.detach().clone()) for name, parameter in parameters.items()
    }
    restored = SparseGeometryAdam(_groups(restored_parameters))
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    optimizer.zero_grad()

    sum(parameter[1].sum() for parameter in parameters.values()).backward()
    sum(parameter[1].sum() for parameter in restored_parameters.values()).backward()
    optimizer.step()
    restored.step()

    for name, parameter in parameters.items():
        assert torch.equal(parameter, restored_parameters[name])
        assert torch.equal(
            optimizer.state[parameter]["row_steps"],
            restored.state[restored_parameters[name]]["row_steps"],
        )
        assert restored.state[restored_parameters[name]]["row_steps"].dtype is torch.int64


def test_sparse_geometry_adam_resume_rejects_reordered_groups() -> None:
    """Optimizer moments cannot attach to a different geometry role."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    sum(parameter[0].sum() for parameter in parameters.values()).backward()
    optimizer.step()
    restored_parameters = _parameters()
    reversed_groups = list(reversed(_groups(restored_parameters)))
    restored = SparseGeometryAdam(reversed_groups)

    with pytest.raises(ValueError, match="group order differs"):
        restored.load_state_dict(deepcopy(optimizer.state_dict()))


def test_sparse_geometry_adam_resume_rejects_changed_row_count() -> None:
    """A topology-mutated checkpoint cannot reuse stale row moments."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    sum(parameter[0].sum() for parameter in parameters.values()).backward()
    optimizer.step()
    state_dict = deepcopy(optimizer.state_dict())
    first_state = next(iter(state_dict["state"].values()))
    first_state["row_steps"] = torch.zeros(3, dtype=torch.int64)

    with pytest.raises(ValueError, match="row-step shape differs"):
        SparseGeometryAdam(_groups(_parameters())).load_state_dict(state_dict)


def test_sparse_geometry_adam_resume_restores_integer_steps_for_fp16() -> None:
    """PyTorch's parameter-dtype cast cannot corrupt discrete row ages."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    sum(parameter[0].sum() for parameter in parameters.values()).backward()
    optimizer.step()
    restored_parameters = _parameters(dtype=torch.float16)
    restored = SparseGeometryAdam(_groups(restored_parameters))

    restored.load_state_dict(deepcopy(optimizer.state_dict()))

    for parameter in restored_parameters.values():
        assert restored.state[parameter]["row_steps"].dtype is torch.int64


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("lr", -0.01, "invalid lr"),
        ("eps", 0.0, "invalid eps"),
        ("betas", (-0.1, 0.999), "invalid betas"),
        ("betas", (0.9, 1.0), "invalid betas"),
    ],
)
def test_sparse_geometry_adam_rejects_invalid_hyperparameters(
    key: str,
    value: object,
    message: str,
) -> None:
    """Invalid Adam coefficients fail before any parameter can move."""
    groups = _groups(_parameters())
    groups[0][key] = value

    with pytest.raises(ValueError, match=message):
        SparseGeometryAdam(groups)


def test_sparse_geometry_adam_rejects_non_finite_gradients() -> None:
    """Invalid sparse gradients fail instead of masquerading as no support."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    parameters["positions"].grad = torch.full_like(
        parameters["positions"],
        torch.nan,
    )

    with pytest.raises(FloatingPointError, match="positions"):
        optimizer.step()


def test_sparse_geometry_adam_non_finite_failure_is_atomic() -> None:
    """A later invalid group cannot leave an earlier group half-updated."""
    parameters = _parameters()
    optimizer = SparseGeometryAdam(_groups(parameters))
    parameters["positions"].grad = torch.ones_like(parameters["positions"])
    parameters["scale"].grad = torch.full_like(parameters["scale"], torch.inf)
    before = {name: parameter.detach().clone() for name, parameter in parameters.items()}

    with pytest.raises(FloatingPointError, match="scale"):
        optimizer.step()

    for name, parameter in parameters.items():
        assert torch.equal(parameter, before[name])
