"""Behavioral tests for M2 Ghost-directed Gaussian pruning."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from threedgrut.ghost_repair import (
    SampledGhostEvidence,
    matched_pruning_cohorts,
    ordered_indices_sha256,
    percentile_ranks,
    pruned_checkpoint,
    repair_rankings,
    sampled_contradiction_field,
    stable_largest_indices,
)
from threedgrut.ghost_repair_study import (
    bind_artifact_directory,
    candidate_id,
    cohort_overlaps,
    file_sha256,
    resolve_registered_path,
    seal_output_file,
    seal_output_tree,
    verify_output_file,
    verify_output_tree,
    write_json_exclusive,
)


def test_sampled_field_preserves_duplicate_contradiction_counts() -> None:
    """Repeated contradicted samples retain their statistical weight."""
    field = sampled_contradiction_field(
        image_shape=(3, 4),
        samples=SampledGhostEvidence(
            us=np.array([1, 1, 2, 3]),
            vs=np.array([1, 1, 1, 2]),
            ghost=np.array([True, True, False, True]),
            evidenced=np.array([True, True, True, False]),
        ),
        device=torch.device("cpu"),
    )
    assert field[1, 1].item() == 2.0
    assert field.sum().item() == 2.0


def test_stable_ranking_breaks_ties_by_row_index() -> None:
    """Equal scores have a deterministic row-index order."""
    result = stable_largest_indices(np.array([1.0, 3.0, 3.0, 2.0]), 3)
    np.testing.assert_array_equal(result, [1, 2, 3])


def test_percentile_ranks_are_stable_and_bounded() -> None:
    """Rank fusion produces comparable unit-interval fields."""
    result = percentile_ranks(np.array([4.0, 1.0, 3.0, 2.0]))
    np.testing.assert_allclose(result, [1.0, 0.0, 2 / 3, 1 / 3])


def test_percentile_ranks_assign_equal_values_equal_midranks() -> None:
    """Tied zero fields do not acquire row-index signal before selection."""
    result = percentile_ranks(np.array([0.0, 0.0, 0.0, 1.0]))
    np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3, 1.0])


def test_matched_controls_sample_full_population_and_each_other_disjointly() -> (
    None
):
    """Random controls are unbiased samples and remain mutually disjoint."""
    rows = 200
    values = np.arange(rows, dtype=np.float64)
    rankings = repair_rankings(
        density=values,
        ghost_responsibility=values[::-1].copy(),
        opacity_sensitivity=np.roll(values, 20),
    )
    cohorts = matched_pruning_cohorts(
        rankings=rankings,
        count=10,
        random_replicates=4,
        seed=17,
    )
    controls = cohorts["random"]
    assert any(row >= 190 for row in np.concatenate(controls))
    for index, left in enumerate(controls):
        for right in controls[index + 1 :]:
            assert set(left.tolist()).isdisjoint(right.tolist())


def test_pruning_filters_every_gaussian_tensor_and_records_hash() -> None:
    """Physical pruning preserves alignment and emits immutable provenance."""
    rows = 6
    checkpoint = {
        "positions": torch.nn.Parameter(
            torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)
        ),
        "rotation": torch.zeros(rows, 4),
        "scale": torch.zeros(rows, 3),
        "density": torch.zeros(rows, 1),
        "features_albedo": torch.zeros(rows, 3),
        "features_specular": torch.zeros(rows, 6),
        "environment_mask": torch.zeros(rows, dtype=torch.bool),
        "background": {"fixed": True},
        "optimizer": {"stale": True},
    }
    removed = np.array([1, 4])
    result = pruned_checkpoint(checkpoint, removed)
    assert result["positions"].shape == (4, 3)
    assert isinstance(result["positions"], torch.nn.Parameter)
    assert result["environment_mask"].shape == (4,)
    assert "optimizer" not in result
    assert result["post_training_prune"]["removed_ordered_index_sha256"] == (
        ordered_indices_sha256(removed)
    )
    assert checkpoint["positions"].shape == (6, 3)


def test_pruning_rejects_active_representation_state() -> None:
    """M2 cannot silently prune an unsupported representation state."""
    checkpoint = {
        "positions": torch.zeros(2, 3),
        "acquisition_visibility": {"state": torch.ones(1)},
    }
    with pytest.raises(ValueError, match="acquisition_visibility"):
        pruned_checkpoint(checkpoint, np.array([0]))


def test_candidate_identity_and_overlap_report_are_deterministic() -> None:
    """Study artifacts retain stable arm identities and declared overlap."""
    assert candidate_id("ghost_responsibility", 0.005, 0) == (
        "ghost_responsibility__b05000ppm__r00"
    )
    overlap = cohort_overlaps(
        {
            "left": np.array([1, 2, 3]),
            "right": np.array([3, 4, 5]),
        }
    )
    assert overlap["left__right"] == {
        "intersection": 1,
        "jaccard": 0.2,
    }


def test_registered_path_must_match_hash_and_resolve_once(
    tmp_path: Path,
) -> None:
    """A preregistered artifact cannot drift or resolve ambiguously."""
    scene_root = tmp_path / "scene"
    doctor_root = tmp_path / "doctor"
    scene_root.mkdir()
    doctor_root.mkdir()
    artifact = scene_root / "artifact.txt"
    artifact.write_text("fixed\n", encoding="utf-8")
    registered = SimpleNamespace(
        path="artifact.txt",
        sha256=file_sha256(artifact),
    )
    assert (
        resolve_registered_path(
            registered,
            scene_root=scene_root,
            doctor_root=doctor_root,
        )
        == artifact.resolve()
    )
    registered.sha256 = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_registered_path(
            registered,
            scene_root=scene_root,
            doctor_root=doctor_root,
        )


def test_resumable_artifacts_reject_changed_or_unbound_inputs(
    tmp_path: Path,
) -> None:
    """A partial run cannot be reused under a different study identity."""
    bound = tmp_path / "bound"
    bind_artifact_directory(bound, {"plan": "first"})
    bind_artifact_directory(bound, {"plan": "first"})
    with pytest.raises(ValueError, match="fingerprint differs"):
        bind_artifact_directory(bound, {"plan": "second"})
    unbound = tmp_path / "unbound"
    unbound.mkdir()
    (unbound / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not reusable"):
        bind_artifact_directory(unbound, {"plan": "first"})


def test_exclusive_json_is_immutable(tmp_path: Path) -> None:
    """Selections and opening markers can only be created once."""
    path = tmp_path / "selection.json"
    write_json_exclusive(path, {"selection": "first"})
    write_json_exclusive(path, {"selection": "first"})
    with pytest.raises(ValueError, match="immutable artifact differs"):
        write_json_exclusive(path, {"selection": "second"})


def test_sealed_file_rejects_content_changes(tmp_path: Path) -> None:
    """A modified result cannot be resumed under a valid input marker."""
    path = tmp_path / "result.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")
    seal_output_file(path)
    verify_output_file(path)
    path.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sealed output changed"):
        verify_output_file(path)


def test_sealed_render_tree_rejects_added_payload(tmp_path: Path) -> None:
    """Render manifests cover every array and metadata file in the tree."""
    render = tmp_path / "render"
    render.mkdir()
    (render / "meta.json").write_text("{}\n", encoding="utf-8")
    (render / "distance.npy").write_bytes(b"fixed")
    seal_output_tree(render)
    verify_output_tree(render)
    (render / "unexpected.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="render output changed"):
        verify_output_tree(render)
