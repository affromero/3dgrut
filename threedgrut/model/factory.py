# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.representation import (
    GaussianRepresentation,
    REPRESENTATION_VERSION,
)
from threedgrut.model.view_conditioned_anchor_field import (
    ViewConditionedAnchorField,
)


def checkpoint_representation(
    checkpoint: dict[str, object],
) -> GaussianRepresentation:
    """Return checkpoint representation, treating untagged checkpoints as ordinary."""
    representation = checkpoint.get("representation")
    if representation is None:
        return GaussianRepresentation.MIXTURE
    if not isinstance(representation, dict):
        raise ValueError("Checkpoint representation metadata must be a mapping.")
    name = representation.get("name")
    version = representation.get("version")
    if version != REPRESENTATION_VERSION:
        raise ValueError(
            "Unsupported checkpoint representation version: "
            f"{version!r}; expected {REPRESENTATION_VERSION}."
        )
    try:
        return GaussianRepresentation(str(name))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported checkpoint representation: {name!r}."
        ) from exc


def configured_representation(conf: DictConfig) -> GaussianRepresentation:
    """Return the explicitly configured representation."""
    value = str(
        conf.model.get(
            "representation",
            GaussianRepresentation.MIXTURE.value,
        )
    )
    try:
        return GaussianRepresentation(value)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported model.representation: {value!r}."
        ) from exc


def create_gaussian_model(
    conf: DictConfig,
    *,
    scene_extent: float | None = None,
    checkpoint: dict[str, object] | None = None,
) -> MixtureOfGaussians | ViewConditionedAnchorField:
    """Create the configured or checkpoint-tagged Gaussian model."""
    representation = (
        checkpoint_representation(checkpoint)
        if checkpoint is not None
        else configured_representation(conf)
    )
    configured = configured_representation(conf)
    if checkpoint is not None and configured != representation:
        raise ValueError(
            "Configured and checkpoint Gaussian representations differ: "
            f"configured={configured.value!r}, "
            f"checkpoint={representation.value!r}."
        )
    if representation == GaussianRepresentation.MIXTURE:
        return MixtureOfGaussians(conf, scene_extent=scene_extent)

    return ViewConditionedAnchorField(
        conf,
        scene_extent=scene_extent,
    )
