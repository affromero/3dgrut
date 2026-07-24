"""Immutable replay of visibility-adaptive image selections."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator, Mapping
from typing import NamedTuple

from torch.utils.data import Sampler


class NativeImageSelection(NamedTuple):
    """One native image choice with its measured render resolution."""

    ordinal: int
    priority: int
    subindex: int
    width: int
    height: int
    sequence: int


class NativeImageReplay:
    """A validated, immutable sequence decoded from a CUPTI sampler trace."""

    def __init__(self, trace_path: str) -> None:
        if not trace_path:
            raise ValueError("Native image replay requires a trace path.")
        self.trace_path = os.path.abspath(trace_path)
        if not os.path.isfile(self.trace_path):
            raise FileNotFoundError(
                "Native image replay trace does not exist: "
                f"{self.trace_path}"
            )
        self.trace_sha256 = self._sha256(self.trace_path)
        self.selections = self._read_trace(self.trace_path)

    @staticmethod
    def _sha256(trace_path: str) -> str:
        digest = hashlib.sha256()
        with open(trace_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _read_trace(trace_path: str) -> tuple[NativeImageSelection, ...]:
        image_rows: dict[int, tuple[int, int, int]] = {}
        update_rows: dict[int, tuple[int, int]] = {}
        with open(trace_path, "r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                fields = raw_line.rstrip("\n").split("\t")
                if not fields or fields[0] not in {"image", "update"}:
                    continue
                if len(fields) != 5:
                    raise ValueError(
                        "Malformed native replay row at "
                        f"{trace_path}:{line_number}."
                    )
                try:
                    sequence = int(fields[1])
                    values = tuple(int(value) for value in fields[2:])
                except ValueError as error:
                    raise ValueError(
                        "Native replay rows must use integer fields at "
                        f"{trace_path}:{line_number}."
                    ) from error
                if sequence <= 0:
                    raise ValueError(
                        "Native replay sequence must start at one: "
                        f"{trace_path}:{line_number}."
                    )
                if fields[0] == "image":
                    if sequence in image_rows:
                        raise ValueError(
                            "Duplicate native image replay sequence "
                            f"{sequence}."
                        )
                    ordinal, priority, subindex = values
                    if ordinal < 0 or priority < 0 or subindex < 0:
                        raise ValueError(
                            "Native image replay values must be non-negative "
                            f"at {trace_path}:{line_number}."
                        )
                    image_rows[sequence] = (ordinal, priority, subindex)
                else:
                    if sequence in update_rows:
                        raise ValueError(
                            "Duplicate native update replay sequence "
                            f"{sequence}."
                        )
                    _visible_count, width, height = values
                    if width <= 0 or height <= 0:
                        raise ValueError(
                            "Native replay dimensions must be positive at "
                            f"{trace_path}:{line_number}."
                        )
                    update_rows[sequence] = (width, height)

        image_sequences = set(image_rows)
        update_sequences = set(update_rows)
        if not image_sequences:
            raise ValueError("Native image replay trace has no image rows.")
        if image_sequences != update_sequences:
            raise ValueError(
                "Native image replay requires one image and update row per "
                "sequence."
            )
        ordered_sequences = tuple(sorted(image_sequences))
        expected_sequences = tuple(range(1, len(ordered_sequences) + 1))
        if ordered_sequences != expected_sequences:
            raise ValueError(
                "Native image replay sequences must be contiguous from one."
            )
        selections = []
        for sequence in ordered_sequences:
            ordinal, priority, subindex = image_rows[sequence]
            width, height = update_rows[sequence]
            selections.append(
                NativeImageSelection(
                    ordinal=ordinal,
                    priority=priority,
                    subindex=subindex,
                    width=width,
                    height=height,
                    sequence=sequence,
                )
            )
        return tuple(selections)

    def __len__(self) -> int:
        return len(self.selections)

    def selection_at(self, step: int) -> NativeImageSelection:
        if step < 0 or step >= len(self):
            raise IndexError(
                "Native image replay has no selection for training step "
                f"{step}; trace length is {len(self)}."
            )
        return self.selections[step]

    def manifest(self) -> dict[str, str | int]:
        """Return immutable source evidence for checkpoint validation."""
        return {
            "trace_sha256": self.trace_sha256,
            "selection_count": len(self),
        }

    def validate_manifest(self, manifest: Mapping[str, object]) -> None:
        """Reject resume when the source trace differs from its checkpoint."""
        expected_hash = manifest.get("trace_sha256")
        expected_count = manifest.get("selection_count")
        if expected_hash != self.trace_sha256:
            raise ValueError("Native image replay trace hash differs on resume.")
        if expected_count != len(self):
            raise ValueError(
                "Native image replay selection count differs on resume."
            )


class NativeImageReplaySampler(Sampler[NativeImageSelection]):
    """Yield an immutable native selection range for one trainer pass."""

    def __init__(
        self,
        replay: NativeImageReplay,
        *,
        start_step: int,
        end_step: int,
    ) -> None:
        if start_step < 0 or end_step < start_step:
            raise ValueError("Native image replay step range is invalid.")
        if end_step > len(replay):
            raise ValueError(
                "Native image replay trace is shorter than the requested "
                f"training range: {end_step} > {len(replay)}."
            )
        self.replay = replay
        self.start_step = start_step
        self.end_step = end_step

    def __iter__(self) -> Iterator[NativeImageSelection]:
        for step in range(self.start_step, self.end_step):
            yield self.replay.selection_at(step)

    def __len__(self) -> int:
        return self.end_step - self.start_step
