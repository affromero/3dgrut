"""Behavior tests for configurable layered-depth adjoint injection."""

import torch

from threedgrut.model.losses import (
    layered_depth_adjoint_gradients,
    layered_depth_adjoint_injection,
)


def test_layered_depth_adjoint_remaps_labels_and_masks_invalid_pixels() -> None:
    """Label remapping and class biases remain data-profile configuration."""
    raw_depth = torch.zeros((1, 1, 3, 1))
    layer_transparency = torch.full_like(raw_depth, 0.5)
    median_depth = torch.zeros_like(raw_depth)
    semantic_mask = torch.tensor([[[[42], [9], [0]]]], dtype=torch.uint8)

    raw_gradient, transparency_gradient = layered_depth_adjoint_gradients(
        raw_depth=raw_depth,
        layer_transparency=layer_transparency,
        median_depth=median_depth,
        semantic_mask=semantic_mask,
        normalized_weight=1.0,
        semantic_label_remap_from=42,
        semantic_label_remap_to=7,
        primary_semantic_label=7,
        primary_transparency_bias=-0.1,
        secondary_semantic_label=9,
        secondary_transparency_bias=-0.01,
    )

    torch.testing.assert_close(raw_gradient, torch.zeros_like(raw_gradient))
    torch.testing.assert_close(
        transparency_gradient,
        torch.tensor([[[[0.0], [0.06333333], [0.0]]]]),
    )


def test_layered_depth_adjoint_injection_uses_only_supplied_gradients() -> None:
    """The injected scalar backpropagates the configured first-order adjoint."""
    raw_depth = torch.tensor([[[[0.07]]]], requires_grad=True)
    layer_transparency = torch.tensor([[[[0.05]]]], requires_grad=True)
    median_depth = torch.tensor([[[[0.1]]]], requires_grad=True)
    semantic_mask = torch.ones((1, 1, 1, 1), dtype=torch.uint8)

    expected_raw, expected_transparency = layered_depth_adjoint_gradients(
        raw_depth=raw_depth,
        layer_transparency=layer_transparency,
        median_depth=median_depth,
        semantic_mask=semantic_mask,
        normalized_weight=1.0,
    )
    loss = layered_depth_adjoint_injection(
        raw_depth=raw_depth,
        layer_transparency=layer_transparency,
        median_depth=median_depth,
        semantic_mask=semantic_mask,
        normalized_weight=1.0,
    )

    loss.backward()

    assert raw_depth.grad is not None
    assert layer_transparency.grad is not None
    torch.testing.assert_close(raw_depth.grad, expected_raw)
    torch.testing.assert_close(layer_transparency.grad, expected_transparency)
    assert median_depth.grad is None
