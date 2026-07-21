# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION
# & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from collections.abc import Mapping

import torch


def module_state_sha256(
    state: Mapping[str, torch.Tensor],
) -> str:
    """Hash an ordered tensor state without relying on pickle bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise ValueError(
                "Post-processing module state must contain tensors only."
            )
        contiguous = tensor.detach().cpu().contiguous()
        descriptors = (
            name,
            str(contiguous.dtype),
            ",".join(str(size) for size in contiguous.shape),
        )
        for descriptor in descriptors:
            encoded = descriptor.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def inherited_controller_inference_contract(
    parent_contract: Mapping[str, object],
    *,
    module_sha256: str,
    checkpoint_global_step: int,
) -> dict[str, object]:
    """Bind an unchanged trained controller to a continuation checkpoint."""
    if len(module_sha256) != 64:
        raise ValueError(
            "Frozen post-processing module hash must be a SHA-256."
        )
    if checkpoint_global_step < 0:
        raise ValueError("Checkpoint global step must be non-negative.")
    if not bool(parent_contract.get("controller_trained", False)):
        raise ValueError(
            "Parent post-processing contract does not prove training."
        )
    parent_global_step = int(parent_contract.get("checkpoint_global_step", -1))
    parent_activation_step = int(
        parent_contract.get("controller_activation_step", -1)
    )
    parent_scheduler_value = parent_contract.get(
        "scheduler_last_epoch",
        -1,
    )
    parent_trained = bool(parent_contract.get("controller_trained", False))
    if parent_scheduler_value is None:
        prior_inherited = parent_contract.get("frozen_parent_controller")
        if not isinstance(prior_inherited, Mapping):
            raise ValueError(
                "Frozen continuation parent has no inherited controller proof."
            )
        if prior_inherited.get("schema_version") != 1:
            raise ValueError("Unsupported inherited controller proof version.")
        if prior_inherited.get("module_state_sha256") != module_sha256:
            raise ValueError("Frozen continuation controller hash changed.")
        parent_global_step = int(
            prior_inherited.get("parent_checkpoint_global_step", -1)
        )
        parent_activation_step = int(
            prior_inherited.get("parent_controller_activation_step", -1)
        )
        parent_scheduler_epoch = int(
            prior_inherited.get("parent_scheduler_last_epoch", -1)
        )
        parent_trained = bool(
            prior_inherited.get("parent_controller_trained", False)
        )
    else:
        parent_scheduler_epoch = int(parent_scheduler_value)
    if (
        not parent_trained
        or parent_activation_step < 0
        or parent_scheduler_epoch <= parent_activation_step
        or parent_global_step < parent_activation_step
    ):
        raise ValueError(
            "Parent post-processing scheduler metadata does not prove "
            "training."
        )
    inherited = {
        "schema_version": 1,
        "module_state_sha256": module_sha256,
        "parent_checkpoint_global_step": parent_global_step,
        "parent_controller_activation_step": parent_activation_step,
        "parent_scheduler_last_epoch": parent_scheduler_epoch,
        "parent_controller_trained": True,
    }
    current = dict(parent_contract)
    current.update(
        {
            "schema_version": 4,
            "checkpoint_global_step": checkpoint_global_step,
            "scheduler_last_epoch": None,
            "frozen_parent_controller": inherited,
        }
    )
    return current
