"""Numerical coverage for the rectangular classic-EWA compositor."""

import pytest
import torch

from threedgrut.rectangular_ewa_compositor import (
    composite_rectangular_ewa,
    masked_rectangular_ewa_contribution,
    masked_rectangular_ewa_forward_contribution,
    masked_rectangular_ewa_forward_visibility,
)


_ALPHA_CUTOFF = 1.0 / 255.0
_TRANSMITTANCE_CUTOFF = 0.01


def _reference_composite(
    *,
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    environment_mask: torch.Tensor,
    pixel_mask: torch.Tensor,
    transmittance_cutoff: float = _TRANSMITTANCE_CUTOFF,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render one rectangular EWA scene with native terminal state."""
    height, width = pixel_mask.shape[1:3]
    coordinate_x = torch.arange(
        width,
        device=means2d.device,
        dtype=means2d.dtype,
    ) + 0.5
    coordinate_y = torch.arange(
        height,
        device=means2d.device,
        dtype=means2d.dtype,
    ) + 0.5
    pixel_y, pixel_x = torch.meshgrid(coordinate_y, coordinate_x, indexing="ij")
    valid_pixels = pixel_mask[0, :, :, 0] > 0
    rgb = torch.zeros((height, width, 3), device=means2d.device)
    all_transmittance = torch.ones((height, width), device=means2d.device)
    raw_depth = torch.zeros((height, width), device=means2d.device)
    base_transmittance = torch.ones((height, width), device=means2d.device)
    base_open = valid_pixels
    for index in range(means2d.shape[0]):
        delta_x = means2d[index, 0] - pixel_x
        delta_y = means2d[index, 1] - pixel_y
        sigma = 0.5 * (
            conics[index, 0] * delta_x.square()
            + conics[index, 2] * delta_y.square()
        ) + conics[index, 1] * delta_x * delta_y
        alpha = torch.clamp_max(opacities[index] * torch.exp(-sigma), 0.99)
        active = (
            valid_pixels
            & base_open
            & (sigma >= 0.0)
            & (alpha >= _ALPHA_CUTOFF)
        )
        alpha = torch.where(active, alpha, torch.zeros_like(alpha))
        rgb = rgb + all_transmittance[..., None] * alpha[..., None] * colors[index]
        all_transmittance = all_transmittance * (1.0 - alpha)
        if not bool(environment_mask[index].item()):
            raw_depth = raw_depth + (
                base_transmittance * alpha * depths[index]
            )
            base_transmittance = base_transmittance * (1.0 - alpha)
            base_open = base_open & (
                base_transmittance >= transmittance_cutoff
            )
    base_closed = valid_pixels & ~base_open
    all_alpha = 1.0 - all_transmittance
    base_alpha = 1.0 - base_transmittance
    epsilon = torch.finfo(base_alpha.dtype).eps
    terminal_rgb = torch.where(
        base_closed[..., None],
        rgb / all_alpha[..., None].clamp_min(epsilon),
        rgb,
    )
    expected_depth = torch.where(
        base_closed,
        raw_depth / base_alpha.clamp_min(epsilon),
        raw_depth,
    )
    reported_all_transmittance = torch.where(
        base_open,
        all_transmittance,
        torch.zeros_like(all_transmittance),
    )
    return (
        terminal_rgb.unsqueeze(0),
        (1.0 - reported_all_transmittance).unsqueeze(0).unsqueeze(-1),
        expected_depth.unsqueeze(0).unsqueeze(-1),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The rectangular EWA compositor requires CUDA.",
)
def test_rectangular_ewa_matches_reference_and_gradients() -> None:
    """RGB, alpha, depth, and packed gradients match direct composition."""
    device = torch.device("cuda")

    def make_inputs() -> tuple[torch.Tensor, ...]:
        means2d = torch.tensor(
            ((7.1, 3.7), (8.4, 4.2), (6.3, 3.4)),
            device=device,
            requires_grad=True,
        )
        conics = torch.tensor(
            ((0.65, 0.08, 0.75), (0.8, -0.06, 0.55), (0.4, 0.1, 0.7)),
            device=device,
            requires_grad=True,
        )
        opacities = torch.tensor(
            (0.12, 0.17, 0.08),
            device=device,
            requires_grad=True,
        )
        colors = torch.tensor(
            ((0.2, 0.4, 0.8), (0.9, 0.1, 0.3), (0.6, 0.7, 0.2)),
            device=device,
            requires_grad=True,
        )
        depths = torch.tensor((2.0, 3.0, 8.0), device=device, requires_grad=True)
        return means2d, conics, opacities, colors, depths

    environment_mask = torch.tensor((False, False, True), device=device)
    pixel_mask = torch.ones((1, 8, 16, 1), device=device)
    pixel_mask[0, 0, 0, 0] = 0
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.tensor((0, 1, 2), device=device, dtype=torch.int32)
    output_weights = (
        torch.linspace(0.1, 0.9, 8 * 16 * 3, device=device).reshape(1, 8, 16, 3),
        torch.linspace(0.2, 0.7, 8 * 16, device=device).reshape(1, 8, 16, 1),
        torch.linspace(0.3, 0.8, 8 * 16, device=device).reshape(1, 8, 16, 1),
    )

    fused_inputs = make_inputs()
    fused_outputs = composite_rectangular_ewa(
        means2d=fused_inputs[0],
        conics=fused_inputs[1],
        opacities=fused_inputs[2],
        colors=fused_inputs[3],
        depths=fused_inputs[4],
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=8,
    )
    fused_loss = sum(
        (output * weight).sum()
        for output, weight in zip(fused_outputs, output_weights, strict=True)
    )
    fused_gradients = torch.autograd.grad(fused_loss, fused_inputs)

    reference_inputs = make_inputs()
    reference_outputs = _reference_composite(
        means2d=reference_inputs[0],
        conics=reference_inputs[1],
        opacities=reference_inputs[2],
        colors=reference_inputs[3],
        depths=reference_inputs[4],
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
    )
    reference_loss = sum(
        (output * weight).sum()
        for output, weight in zip(reference_outputs, output_weights, strict=True)
    )
    reference_gradients = torch.autograd.grad(reference_loss, reference_inputs)

    for fused, reference in zip(fused_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(fused, reference, rtol=2.0e-5, atol=2.0e-6)
    for fused, reference in zip(
        fused_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(fused, reference, rtol=3.0e-4, atol=3.0e-5)
    torch.testing.assert_close(fused_outputs[0][0, 0, 0], torch.zeros(3, device=device))
    torch.testing.assert_close(
        fused_outputs[1][0, 0, 0],
        torch.ones(1, device=device),
    )
    torch.testing.assert_close(
        fused_outputs[2][0, 0, 0],
        torch.zeros(1, device=device),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The rectangular EWA compositor requires CUDA.",
)
def test_rectangular_ewa_terminal_finalization_matches_gradients() -> None:
    """Closed base layers normalize output and stop later environment rows."""
    device = torch.device("cuda")

    def make_inputs() -> tuple[torch.Tensor, ...]:
        means2d = torch.tensor(
            ((7.5, 3.5),) * 5,
            device=device,
            requires_grad=True,
        )
        conics = torch.tensor(
            ((100.0, 0.0, 100.0),) * 5,
            device=device,
            requires_grad=True,
        )
        opacities = torch.full((5,), 0.75, device=device, requires_grad=True)
        colors = torch.tensor(
            (
                (0.2, 0.3, 0.4),
                (0.4, 0.1, 0.6),
                (0.8, 0.7, 0.2),
                (0.1, 0.9, 0.5),
                (0.9, 0.2, 0.8),
            ),
            device=device,
            requires_grad=True,
        )
        depths = torch.tensor(
            (2.0, 3.0, 4.0, 5.0, 6.0),
            device=device,
            requires_grad=True,
        )
        return means2d, conics, opacities, colors, depths

    environment_mask = torch.tensor(
        (False, False, True, False, True),
        device=device,
    )
    pixel_mask = torch.zeros((1, 8, 16, 1), device=device)
    pixel_mask[0, 3, 7, 0] = True
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.arange(5, device=device, dtype=torch.int32)
    output_weights = (
        torch.zeros((1, 8, 16, 3), device=device),
        torch.zeros((1, 8, 16, 1), device=device),
        torch.zeros((1, 8, 16, 1), device=device),
    )
    output_weights[0][0, 3, 7] = torch.tensor(
        (0.2, -0.3, 0.5),
        device=device,
    )
    output_weights[1][0, 3, 7, 0] = 0.7
    output_weights[2][0, 3, 7, 0] = -0.6

    fused_inputs = make_inputs()
    fused_outputs = composite_rectangular_ewa(
        means2d=fused_inputs[0],
        conics=fused_inputs[1],
        opacities=fused_inputs[2],
        colors=fused_inputs[3],
        depths=fused_inputs[4],
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=8,
        transmittance_cutoff=0.0625,
    )
    fused_loss = sum(
        (output * weight).sum()
        for output, weight in zip(fused_outputs, output_weights, strict=True)
    )
    fused_gradients = torch.autograd.grad(fused_loss, fused_inputs)

    reference_inputs = make_inputs()
    reference_outputs = _reference_composite(
        means2d=reference_inputs[0],
        conics=reference_inputs[1],
        opacities=reference_inputs[2],
        colors=reference_inputs[3],
        depths=reference_inputs[4],
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        transmittance_cutoff=0.0625,
    )
    reference_loss = sum(
        (output * weight).sum()
        for output, weight in zip(reference_outputs, output_weights, strict=True)
    )
    reference_gradients = torch.autograd.grad(reference_loss, reference_inputs)

    for fused, reference in zip(fused_outputs, reference_outputs, strict=True):
        torch.testing.assert_close(fused, reference, rtol=4.0e-4, atol=4.0e-5)
    for fused, reference in zip(
        fused_gradients,
        reference_gradients,
        strict=True,
    ):
        torch.testing.assert_close(fused, reference, rtol=8.0e-4, atol=8.0e-5)
    torch.testing.assert_close(
        fused_outputs[1][0, 3, 7],
        torch.ones(1, device=device),
    )
    for gradient in fused_gradients:
        torch.testing.assert_close(gradient[-1], torch.zeros_like(gradient[-1]))


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The rectangular EWA compositor requires CUDA.",
)
def test_rectangular_ewa_preserves_separate_contribution_rounding() -> None:
    """RGB accumulation retains the product rounding before addition."""
    device = torch.device("cuda")
    means2d = torch.tensor(((7.5, 3.5), (7.5, 3.5)), device=device)
    conics = torch.tensor(((1.0, 0.0, 1.0),) * 2, device=device)
    opacities = torch.tensor(
        (0.8216638565063477, 0.774450421333313),
        device=device,
    )
    colors = torch.tensor(
        (
            (0.3152284622192383, 0.0, 0.0),
            (0.6214599609375, 0.0, 0.0),
        ),
        device=device,
    )
    depths = torch.tensor((2.0, 3.0), device=device)
    environment_mask = torch.tensor((False, False), device=device)
    pixel_mask = torch.zeros((1, 8, 16, 1), device=device)
    pixel_mask[0, 3, 7, 0] = True
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.tensor((0, 1), device=device, dtype=torch.int32)

    rgb, _, _ = composite_rectangular_ewa(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        colors=colors,
        depths=depths,
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=8,
        transmittance_cutoff=0.0,
    )

    torch.testing.assert_close(
        rgb[0, 3, 7, 0],
        torch.tensor(0.34484320878982544, device=device),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The rectangular EWA compositor requires CUDA.",
)
def test_masked_rectangular_ewa_contribution_respects_stop_condition() -> None:
    """Contribution uses the same 16x8 mask, cutoff, and recurrence."""
    device = torch.device("cuda")
    means2d = torch.tensor(
        ((7.5, 3.5),) * 4,
        device=device,
    )
    conics = torch.tensor(
        ((1.0, 0.0, 1.0),) * 4,
        device=device,
    )
    opacities = torch.tensor((0.8, 0.8, 0.96, 0.5), device=device)
    pixel_mask = torch.zeros((1, 8, 16, 1), device=device)
    pixel_mask[0, 3, 7, 0] = True
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.arange(4, device=device, dtype=torch.int32)

    contribution = masked_rectangular_ewa_contribution(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        pixel_mask=pixel_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=8,
    )

    torch.testing.assert_close(
        contribution,
        torch.tensor((0.8, 0.16, 0.0384, 0.0), device=device),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The rectangular EWA compositor requires CUDA.",
)
def test_masked_rectangular_ewa_forward_contribution_tracks_base_termination() -> None:
    """Environment samples preserve the base-layer termination budget."""
    device = torch.device("cuda")
    means2d = torch.tensor(((7.5, 3.5),) * 4, device=device)
    conics = torch.tensor(((100.0, 0.0, 100.0),) * 4, device=device)
    opacities = torch.full((4,), 0.9, device=device)
    environment_mask = torch.tensor((False, True, False, True), device=device)
    pixel_mask = torch.zeros((1, 8, 16, 1), device=device)
    pixel_mask[0, 3, 7, 0] = True
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.arange(4, device=device, dtype=torch.int32)

    contribution = masked_rectangular_ewa_forward_contribution(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=8,
        transmittance_cutoff=0.05,
    )

    torch.testing.assert_close(
        contribution,
        torch.tensor((0.9, 0.09, 0.009, 0.0), device=device),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="The rectangular EWA compositor requires CUDA.",
)
def test_masked_rectangular_ewa_visibility_tracks_base_termination() -> None:
    """Environment samples remain visible without consuming base visibility."""
    device = torch.device("cuda")
    means2d = torch.tensor(((7.5, 3.5),) * 4, device=device)
    conics = torch.tensor(((100.0, 0.0, 100.0),) * 4, device=device)
    opacities = torch.full((4,), 0.9, device=device)
    environment_mask = torch.tensor((False, True, False, True), device=device)
    pixel_mask = torch.zeros((1, 8, 16, 1), device=device)
    pixel_mask[0, 3, 7, 0] = True
    offsets = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)
    flatten_ids = torch.arange(4, device=device, dtype=torch.int32)

    visibility = masked_rectangular_ewa_forward_visibility(
        means2d=means2d,
        conics=conics,
        opacities=opacities,
        environment_mask=environment_mask,
        pixel_mask=pixel_mask,
        offsets=offsets,
        flatten_ids=flatten_ids,
        width=16,
        height=8,
        transmittance_cutoff=0.05,
    )

    torch.testing.assert_close(
        visibility,
        torch.tensor((True, True, True, False), device=device),
    )
