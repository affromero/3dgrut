# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analytic positional carrier residuals (Gabor / per-particle SIREN).

Carrier parameters are packed after the SH coefficients in
``features_specular``; the tracer evaluates them per-ray on the canonical
hit position when ``FEATURE_TRANSFORM_TYPE == 2``.
"""

import numpy as np
import torch
from omegaconf import DictConfig

GABOR_CARRIER_NUM_TERMS = 3
GABOR_CARRIER_EXTRA_COEFFS = GABOR_CARRIER_NUM_TERMS + 3
SIREN_CARRIER_INPUT_DIM = 5
SIREN_CARRIER_DEFAULT_HIDDEN_DIM = 6


def gabor_carrier_enabled(conf) -> bool:
    return bool(conf.model.get("use_gabor_carrier", False))


def siren_carrier_enabled(conf) -> bool:
    return bool(conf.model.get("use_siren_carrier", False))


def validate_carrier_config(conf) -> None:
    if gabor_carrier_enabled(conf) and siren_carrier_enabled(conf):
        raise ValueError(
            "model.use_gabor_carrier and model.use_siren_carrier are "
            "mutually exclusive."
        )


def gabor_carrier_coeffs(conf) -> int:
    if not gabor_carrier_enabled(conf):
        return 0
    num_terms = int(conf.model.get("gabor_num_terms", GABOR_CARRIER_NUM_TERMS))
    if num_terms != GABOR_CARRIER_NUM_TERMS:
        raise ValueError(
            "model.gabor_num_terms currently supports exactly 3 terms; "
            f"got {num_terms}."
        )
    return GABOR_CARRIER_EXTRA_COEFFS


def siren_carrier_hidden_dim(conf) -> int:
    hidden_dim = int(
        conf.model.get(
            "siren_hidden_dim",
            SIREN_CARRIER_DEFAULT_HIDDEN_DIM,
        )
    )
    if hidden_dim <= 0:
        raise ValueError(
            "model.siren_hidden_dim must be positive; got "
            f"{hidden_dim}."
        )
    return hidden_dim


def siren_carrier_bias_coeffs(hidden_dim: int) -> int:
    """Return the packed first-layer bias coefficient count."""
    return (hidden_dim + 2) // 3


def siren_carrier_coeffs(conf: DictConfig) -> int:
    """Return per-channel coefficient count for the SIREN carrier."""
    if not siren_carrier_enabled(conf):
        return 0
    hidden_dim = siren_carrier_hidden_dim(conf)
    w1_coeffs = hidden_dim * 2
    b1_coeffs = siren_carrier_bias_coeffs(hidden_dim)
    w2_coeffs = hidden_dim
    b2_coeffs = 1
    return w1_coeffs + b1_coeffs + w2_coeffs + b2_coeffs


def carrier_specular_dim(conf) -> int:
    validate_carrier_config(conf)
    return 3 * (gabor_carrier_coeffs(conf) + siren_carrier_coeffs(conf))


def initial_gabor_carrier_tail(
    *,
    num_gaussians: int,
    device: str | torch.device,
    dtype: torch.dtype,
    conf,
) -> torch.Tensor:
    tail = torch.zeros(
        (num_gaussians, 3 * gabor_carrier_coeffs(conf)),
        dtype=dtype,
        device=device,
    )
    if tail.shape[1] == 0:
        return tail
    frequency = 0.5 * float(conf.model.get("gabor_max_frequency", 4.0))
    angles = torch.tensor(
        [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0],
        dtype=dtype,
        device=device,
    )
    tail[:, 9:12] = frequency * torch.cos(angles)[None, :]
    tail[:, 12:15] = frequency * torch.sin(angles)[None, :]
    return tail


def initial_siren_carrier_tail(
    *,
    num_gaussians: int,
    device: str | torch.device,
    dtype: torch.dtype,
    conf: DictConfig,
) -> torch.Tensor:
    """Initialize packed SIREN carrier coefficients."""
    hidden_dim = siren_carrier_hidden_dim(conf)
    coeffs = siren_carrier_coeffs(conf)
    tail = torch.zeros(
        (num_gaussians, 3 * coeffs),
        dtype=dtype,
        device=device,
    )
    if tail.shape[1] == 0:
        return tail

    seed = int(conf.model.get("siren_init_seed", 20260626))
    init_scale = float(
        conf.model.get("siren_init_scale", 1.0 / SIREN_CARRIER_INPUT_DIM)
    )
    if init_scale < 0.0:
        raise ValueError(
            f"model.siren_init_scale must be non-negative; got {init_scale}."
        )
    output_init_scale = float(conf.model.get("siren_output_init_scale", 0.0))
    if output_init_scale < 0.0:
        raise ValueError(
            "model.siren_output_init_scale must be non-negative; got "
            f"{output_init_scale}."
        )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    w1 = (
        2.0
        * torch.rand(
            (num_gaussians, hidden_dim, SIREN_CARRIER_INPUT_DIM),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        - 1.0
    ) * init_scale
    b1 = (
        2.0
        * torch.rand(
            (num_gaussians, hidden_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        - 1.0
    ) * init_scale

    for hidden_idx in range(hidden_dim):
        w1_slot = 2 * hidden_idx
        tail[:, (w1_slot * 3) : (w1_slot * 3 + 3)] = w1[
            :, hidden_idx, :3
        ]
        tail[:, ((w1_slot + 1) * 3)] = w1[:, hidden_idx, 3]
        tail[:, ((w1_slot + 1) * 3 + 1)] = w1[:, hidden_idx, 4]

    b1_offset = hidden_dim * 2
    for hidden_idx in range(hidden_dim):
        flat_idx = (b1_offset + hidden_idx // 3) * 3 + hidden_idx % 3
        tail[:, flat_idx] = b1[:, hidden_idx]

    if output_init_scale > 0.0:
        w2 = (
            2.0
            * torch.rand(
                (num_gaussians, hidden_dim, 3),
                dtype=dtype,
                device=device,
                generator=generator,
            )
            - 1.0
        ) * output_init_scale
        w2_offset = b1_offset + siren_carrier_bias_coeffs(hidden_dim)
        for hidden_idx in range(hidden_dim):
            coeff_idx = (w2_offset + hidden_idx) * 3
            tail[:, coeff_idx : coeff_idx + 3] = w2[:, hidden_idx, :]

    return tail


def initial_carrier_tail(
    *,
    num_gaussians: int,
    device: str | torch.device,
    dtype: torch.dtype,
    conf,
) -> torch.Tensor:
    validate_carrier_config(conf)
    if gabor_carrier_enabled(conf):
        return initial_gabor_carrier_tail(
            num_gaussians=num_gaussians,
            device=device,
            dtype=dtype,
            conf=conf,
        )
    if siren_carrier_enabled(conf):
        return initial_siren_carrier_tail(
            num_gaussians=num_gaussians,
            device=device,
            dtype=dtype,
            conf=conf,
        )
    return torch.zeros((num_gaussians, 0), dtype=dtype, device=device)

