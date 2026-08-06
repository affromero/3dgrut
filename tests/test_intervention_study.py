"""Tests for matched diagnostic-field interventions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from threedgrut.intervention_study import (
    InterventionMode,
    RankingDirection,
    generate_intervention_provenance,
    matched_cohorts,
    matched_cohorts_v1,
    perturbed_parameter_rows,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.density = torch.nn.Parameter(
            torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        )
        self.positions = torch.nn.Parameter(torch.ones((4, 3)))

    def get_scale(self) -> torch.Tensor:
        return torch.full((4, 3), 2.0)


def test_matched_cohorts_preserve_size_and_determinism() -> None:
    """Confirmatory controls stay disjoint from treatment and each other."""
    scores = torch.arange(12, dtype=torch.float32)
    residual = torch.arange(12, 0, -1, dtype=torch.float32)

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
    treatment = set(cohorts[0][2].tolist())
    controls = [set(indices.tolist()) for _, _, indices in cohorts[1:]]
    assert all(not treatment & control for control in controls)
    assert not controls[0] & controls[1]
    assert all(not controls[0] & control for control in controls[2:])
    assert all(not controls[1] & control for control in controls[2:])
    assert all(indices.numel() == 2 for _, _, indices in cohorts)


def test_m1_cohorts_reproduce_legacy_overlap() -> None:
    """The provenance path retains the exact historical M1 selection."""
    scores = torch.tensor([0.2, 0.9, 0.1, 0.5])
    residual = torch.tensor([0.8, 0.3, 0.2, 0.1])

    cohorts = matched_cohorts_v1(
        scores=scores,
        residual_scores=residual,
        cohort_size=2,
        ranking=RankingDirection.DESCENDING,
        random_seeds=(4, 9),
    )

    assert cohorts[0][2].tolist() == [3, 1]
    assert cohorts[1][2].tolist() == [2, 0]
    assert cohorts[2][2].tolist() == [2, 0]


def test_provenance_hashes_raw_fields_and_exact_cohort_indices(
    tmp_path: Path,
) -> None:
    """Provenance authenticates source arrays and ordered cohort rows."""
    raw = tmp_path / "raw"
    raw.mkdir()
    scores = torch.arange(8, dtype=torch.float32).numpy()
    residual = scores[::-1].copy()
    np.save(raw / "field.npy", scores)
    np.save(raw / "residual_mse_exposure.npy", residual)
    plan = {
        "study_id": "m1",
        "scenes": ["scene"],
        "random_seeds": [4, 9],
        "fields": [
            {
                "field": "test",
                "source_field_id": "field",
                "cohort_size": 2,
                "ranking": "descending",
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    checkpoint_hash = "a" * 64
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "scene": "scene",
                "plan_sha256": plan_hash,
                "checkpoint_sha256": checkpoint_hash,
                "repair_list_sha256": "b" * 64,
                "certification_list_sha256": "c" * 64,
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"source_checkpoint_sha256": checkpoint_hash})
    )

    provenance = generate_intervention_provenance(
        output_dir=str(tmp_path),
        plan_path=str(plan_path),
        result_path=str(result_path),
        manifest_path=str(manifest_path),
        renderer_commit="d" * 40,
        doctor_commit="e" * 40,
        runner_sha256="f" * 64,
    )

    assert (
        provenance["result_sha256"]
        == hashlib.sha256(result_path.read_bytes()).hexdigest()
    )
    assert len(provenance["cohort_index_hashes"]) == 5
    assert provenance["raw_fields"]["field"]["shape"] == [8]


def test_absolute_intervention_restores_checkpoint_rows() -> None:
    """Absolute perturbations restore checkpoint rows on context exit."""
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
    """Relative position perturbations scale with physical Gaussians."""
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
