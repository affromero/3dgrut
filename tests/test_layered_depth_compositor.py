"""Numerical coverage for the fused layered-depth compositor."""

import pytest
import torch

from threedgrut.layered_depth_compositor import (
    composite_layered_depth,
    masked_ewa_contribution,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The layered-depth compositor requires CUDA.",
)
def test_fused_layered_depth_matches_single_splat_autograd() -> None:
    """One tile agrees with direct alpha compositing and its derivatives."""
    device = torch.device("cuda")
    means2d = torch.tensor(
        ((7.5, 7.5),),
        device=device,
        requires_grad=True,
    )
    conics = torch.tensor(
        ((1.0, 0.0, 1.0),),
        device=device,
        requires_grad=True,
    )
    opacities = torch.tensor((0.5,), device=device, requires_grad=True)
    depths = torch.tensor((2.0,), device=device, requires_grad=True)
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.tensor((0,), device=device, dtype=torch.int32)

    raw_depth, transparency, _ = composite_layered_depth(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        depths=depths,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=16,
        tile_size=16,
        alpha_cutoff=1.0 / 255.0,
    )
    coordinates = torch.arange(16, device=device, dtype=means2d.dtype) + 0.5
    pixel_y, pixel_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    delta_x = pixel_x - means2d[0, 0]
    delta_y = pixel_y - means2d[0, 1]
    sigma = (
        0.5
        * (conics[0, 0] * delta_x.square() + conics[0, 2] * delta_y.square())
        + conics[0, 1] * delta_x * delta_y
    )
    alpha = torch.clamp_max(opacities[0] * torch.exp(-sigma), 0.99)
    visible = alpha >= 1.0 / 255.0
    expected_raw_depth = torch.where(
        visible,
        alpha * depths[0],
        torch.zeros_like(alpha),
    ).reshape(1, 16, 16, 1)
    expected_transparency = torch.where(
        visible,
        1.0 - alpha,
        torch.ones_like(alpha),
    ).reshape(1, 16, 16, 1)

    torch.testing.assert_close(raw_depth, expected_raw_depth)
    torch.testing.assert_close(transparency, expected_transparency)
    fused_loss = (raw_depth + 0.3 * transparency).sum()
    expected_loss = (expected_raw_depth + 0.3 * expected_transparency).sum()
    fused_gradients = torch.autograd.grad(
        fused_loss,
        (means2d, conics, opacities, depths),
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected_loss,
        (means2d, conics, opacities, depths),
    )
    for fused_gradient, expected_gradient in zip(
        fused_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            fused_gradient,
            expected_gradient,
            rtol=3.0e-4,
            atol=3.0e-5,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The layered-depth compositor requires CUDA.",
)
def test_masked_ewa_contribution_respects_mask_and_cutoff() -> None:
    """Valid pixels accumulate alpha before the recovered stop condition."""
    device = torch.device("cuda")
    means2d = torch.full((4, 2), 7.5, device=device)
    conics = torch.tensor(
        ((1.0, 0.0, 1.0),) * 4,
        device=device,
    )
    opacities = torch.tensor((0.8, 0.8, 0.96, 0.5), device=device)
    pixel_mask = torch.zeros((1, 16, 16, 1), device=device)
    pixel_mask[0, 7, 7, 0] = True
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.arange(4, device=device, dtype=torch.int32)

    contribution = masked_ewa_contribution(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        pixel_mask=pixel_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=16,
        tile_size=16,
        alpha_cutoff=1.0 / 255.0,
        transmittance_cutoff=0.01,
    )

    torch.testing.assert_close(
        contribution,
        torch.tensor((0.8, 0.16, 0.0384, 0.0), device=device),
    )
    fully_valid = masked_ewa_contribution(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        pixel_mask=torch.ones_like(pixel_mask),
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=16,
        tile_size=16,
        alpha_cutoff=1.0 / 255.0,
        transmittance_cutoff=0.01,
    )
    unmasked = masked_ewa_contribution(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        pixel_mask=None,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=16,
        tile_size=16,
        alpha_cutoff=1.0 / 255.0,
        transmittance_cutoff=0.01,
    )
    torch.testing.assert_close(unmasked, fully_valid)
