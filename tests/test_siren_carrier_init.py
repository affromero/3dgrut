"""Tests for SIREN per-Gaussian carrier initialization."""

import torch
from omegaconf import OmegaConf
from threedgrut.model.model import (
    initial_siren_carrier_tail,
    siren_carrier_bias_coeffs,
    siren_carrier_coeffs,
)


def _siren_conf(*, output_init_scale: float) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "use_gabor_carrier": False,
                "use_siren_carrier": True,
                "siren_hidden_dim": 6,
                "siren_init_seed": 123,
                "siren_init_scale": 0.2,
                "siren_output_init_scale": output_init_scale,
            }
        }
    )


def _siren_output_slots(conf: OmegaConf) -> tuple[slice, slice]:
    hidden_dim = int(conf.model.siren_hidden_dim)
    b1_offset = hidden_dim * 2
    w2_offset = b1_offset + siren_carrier_bias_coeffs(hidden_dim)
    b2_offset = w2_offset + hidden_dim
    return (
        slice(w2_offset * 3, b2_offset * 3),
        slice(b2_offset * 3, (b2_offset + 1) * 3),
    )


def test_siren_output_init_defaults_to_noop() -> None:
    """Default SIREN output initialization preserves the no-op residual."""
    conf = _siren_conf(output_init_scale=0.0)
    tail = initial_siren_carrier_tail(
        num_gaussians=4,
        device="cpu",
        dtype=torch.float32,
        conf=conf,
    )

    assert tail.shape == (4, 3 * siren_carrier_coeffs(conf))
    w2_slot, b2_slot = _siren_output_slots(conf)
    assert torch.count_nonzero(tail[:, : w2_slot.start]) > 0
    assert torch.count_nonzero(tail[:, w2_slot]) == 0
    assert torch.count_nonzero(tail[:, b2_slot]) == 0


def test_siren_output_init_can_seed_final_weights() -> None:
    """SIREN output initialization can seed trainable final-layer weights."""
    conf = _siren_conf(output_init_scale=0.02)
    tail = initial_siren_carrier_tail(
        num_gaussians=4,
        device="cpu",
        dtype=torch.float32,
        conf=conf,
    )

    w2_slot, b2_slot = _siren_output_slots(conf)
    assert torch.count_nonzero(tail[:, w2_slot]) > 0
    assert torch.count_nonzero(tail[:, b2_slot]) == 0
    assert torch.max(torch.abs(tail[:, w2_slot])) <= 0.02
