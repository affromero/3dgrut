"""Behavioral tests for validation-image support visualization."""

import torch

from threedgrut.trainer import _make_validation_image_tiles


def _tiles(mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Build a minimal validation tile group."""
    rgb_gt = torch.full((2, 2, 3), 0.8)
    rgb_pred = torch.full((2, 2, 3), 0.6)
    scalar = torch.zeros((2, 2, 1))
    return _make_validation_image_tiles(
        rgb_gt=rgb_gt,
        rgb_pred=rgb_pred,
        pred_dist=scalar,
        pred_opacity=scalar,
        hit_counts=scalar,
        mask=mask,
        sky_mask=None,
        semantic_label_masks=None,
        max_hit_count=1,
    )


def test_validation_tiles_distinguish_raw_and_scored_support() -> None:
    """Excluded pixels are checkerboarded only in scored RGB panels."""
    mask = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]]])

    tiles = _tiles(mask)

    assert torch.equal(
        tiles["raw_input_rgb"],
        torch.full((2, 2, 3), 0.8),
    )
    assert torch.equal(
        tiles["scored_keep_mask"],
        mask.expand(2, 2, 3),
    )
    assert torch.equal(
        tiles["scored_input_rgb"][:, 0],
        torch.full((2, 3), 0.8),
    )
    assert torch.equal(
        tiles["scored_prediction_rgb"][:, 0],
        torch.full((2, 3), 0.6),
    )
    assert torch.allclose(
        tiles["scored_input_rgb"][:, 1],
        torch.full((2, 3), 0.18),
    )
    assert torch.allclose(
        tiles["scored_prediction_rgb"][:, 1],
        torch.full((2, 3), 0.18),
    )


def test_validation_tiles_leave_unmasked_rgb_unchanged() -> None:
    """Datasets without support masks keep identical raw and scored panels."""
    tiles = _tiles(None)

    assert torch.equal(tiles["scored_input_rgb"], tiles["raw_input_rgb"])
    assert torch.equal(
        tiles["scored_prediction_rgb"],
        torch.full((2, 2, 3), 0.6),
    )
    assert torch.equal(
        tiles["scored_keep_mask"],
        torch.ones((2, 2, 3)),
    )
