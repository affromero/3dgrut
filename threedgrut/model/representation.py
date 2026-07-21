# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class GaussianRepresentation(str, Enum):
    """Trainable Gaussian representation stored in a checkpoint."""

    MIXTURE = "gaussian_mixture"
    VIEW_CONDITIONED_ANCHOR = "view_conditioned_anchor"


REPRESENTATION_VERSION = 1
