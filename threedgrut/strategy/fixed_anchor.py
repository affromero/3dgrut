# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from threedgrut.strategy.base import BaseStrategy


class FixedAnchorStrategy(BaseStrategy):
    """Preserve a model's Gaussian topology throughout training."""

    def __init__(self, config: object, model: object) -> None:
        super().__init__(config, model)
        self.expected_gaussian_count: int | None = None

    def _capture_topology(self) -> None:
        gaussian_count = int(self.model.num_gaussians)
        if gaussian_count <= 0:
            raise ValueError(
                "Fixed topology requires at least one Gaussian."
            )
        if self.expected_gaussian_count is None:
            self.expected_gaussian_count = gaussian_count

    def _validate_topology(self) -> None:
        if self.expected_gaussian_count is None:
            raise RuntimeError(
                "Fixed topology was not captured after initialization."
            )
        gaussian_count = int(self.model.num_gaussians)
        if gaussian_count != self.expected_gaussian_count:
            raise RuntimeError(
                "Fixed-topology Gaussian count changed: "
                f"expected {self.expected_gaussian_count}, got "
                f"{gaussian_count}."
            )

    def init_densification_buffer(
        self,
        checkpoint: dict[str, object] | None = None,
    ) -> None:
        """Validate checkpoint topology without allocating mutation buffers."""
        self._capture_topology()
        self._validate_topology()
        if checkpoint is None:
            return
        state = checkpoint.get("strategy_state")
        if not isinstance(state, dict):
            raise ValueError(
                "Fixed-topology checkpoint is missing strategy state."
            )
        if (
            state.get("name") != "FixedAnchorStrategy"
            or state.get("version") != 2
        ):
            raise ValueError(
                "Fixed-topology checkpoint has incompatible strategy state."
            )
        checkpoint_count = state.get("expected_gaussian_count")
        if checkpoint_count != self.expected_gaussian_count:
            raise ValueError(
                "Fixed-topology checkpoint Gaussian count differs from the "
                "restored model."
            )

    def _post_optimizer_step(
        self,
        step: int,
        scene_extent: float,
        train_dataset: object,
        batch: object | None = None,
        writer: object | None = None,
    ) -> bool:
        del step, scene_extent, train_dataset, batch, writer
        self._validate_topology()
        return False

    def get_strategy_parameters(self) -> dict[str, object]:
        self._validate_topology()
        return {
            "strategy_state": {
                "name": "FixedAnchorStrategy",
                "version": 2,
                "expected_gaussian_count": (
                    self.expected_gaussian_count
                ),
            }
        }
