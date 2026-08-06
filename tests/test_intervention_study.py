"""Tests for matched diagnostic-field interventions."""

import torch

from threedgrut.intervention_study import (
    InterventionMode,
    RankingDirection,
    matched_cohorts,
    perturbed_parameter_rows,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.density = torch.nn.Parameter(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        self.positions = torch.nn.Parameter(torch.ones((4, 3)))

    def get_scale(self) -> torch.Tensor:
        return torch.full((4, 3), 2.0)


def test_matched_cohorts_preserve_size_and_determinism() -> None:
    scores = torch.tensor([0.2, 0.9, 0.1, 0.5])
    residual = torch.tensor([0.8, 0.3, 0.2, 0.1])

    cohorts = matched_cohorts(
        scores=scores,
        residual_scores=residual,
        cohort_size=2,
        ranking=RankingDirection.DESCENDING,
        random_seeds=(4, 9),
    )

    assert [(name, replicate) for name, replicate, _ in cohorts] == [
        ("top_risk", 0),
        ("bottom_risk", 0),
        ("residual_only", 0),
        ("random", 0),
        ("random", 1),
    ]
    assert cohorts[0][2].tolist() == [3, 1]
    assert cohorts[1][2].tolist() == [2, 0]
    assert cohorts[2][2].tolist() == [2, 0]
    assert not set(cohorts[0][2].tolist()) & set(cohorts[2][2].tolist())
    assert all(indices.numel() == 2 for _, _, indices in cohorts)


def test_absolute_intervention_restores_checkpoint_rows() -> None:
    model = _Model()
    original = model.density.detach().clone()

    with perturbed_parameter_rows(
        model=model,
        indices=torch.tensor([1, 3]),
        parameter_name="density",
        mode=InterventionMode.ADDITIVE_ABSOLUTE,
        magnitude=-0.25,
        direction_seed=0,
    ):
        torch.testing.assert_close(
            model.density.detach().reshape(-1),
            torch.tensor([1.0, 1.75, 3.0, 3.75]),
        )

    torch.testing.assert_close(model.density.detach(), original)


def test_relative_position_intervention_uses_physical_scale() -> None:
    model = _Model()
    original = model.positions.detach().clone()

    with perturbed_parameter_rows(
        model=model,
        indices=torch.tensor([0]),
        parameter_name="positions",
        mode=InterventionMode.ADDITIVE_RELATIVE_SCALE,
        magnitude=0.1,
        direction_seed=0,
    ):
        delta = (model.positions.detach()[0] - original[0]).abs()
        torch.testing.assert_close(delta, torch.full((3,), 0.2))

    torch.testing.assert_close(model.positions.detach(), original)
