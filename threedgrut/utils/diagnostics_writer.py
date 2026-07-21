"""Per-step training diagnostics → Parquet sidecar (AI-readable substrate).

Writes a tabular log of per-step training metrics to a Parquet file in
the run's output directory. Mirrors wandb scalars so the same data is
available offline without a wandb account or live process:

    >>> import pandas as pd
    >>> df = pd.read_parquet("runs/<exp>/<run>/diagnostics.parquet")
    >>> df.query("grad_positions_p95 > 0.1").head()

Uses `pyarrow.parquet.ParquetWriter` in row-group append mode — multiple
`write_table()` calls into one open writer, finalized by `close()`. A
hard kill mid-write may leave a truncated file (acceptable risk; reruns
regenerate).

Schema columns are versioned. Readers should use `pd.read_parquet`,
which handles missing legacy columns as NaN. Bump `SCHEMA_VERSION` when
the structure changes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from threedgrut.utils.logger import logger
from threedgrut.utils.quantile import bounded_quantile


SCHEMA_VERSION = 3

# Per-attribute grad-norm summaries (one mean/p95/max each for 6 params).
_GRAD_PARAMS = (
    "positions",
    "rotation",
    "scale",
    "density",
    "features_albedo",
    "features_specular",
)


def _build_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("step", pa.int64()),
        pa.field("schema_version", pa.int32()),
        # Loss components
        pa.field("total_loss", pa.float32()),
        pa.field("loss_l1", pa.float32()),
        pa.field("loss_ssim", pa.float32()),
        # Per-attribute grad summaries
        *[pa.field(f"grad_{a}_mean", pa.float32()) for a in _GRAD_PARAMS],
        *[pa.field(f"grad_{a}_p95", pa.float32()) for a in _GRAD_PARAMS],
        *[pa.field(f"grad_{a}_max", pa.float32()) for a in _GRAD_PARAMS],
        # Particle state
        pa.field("n_gaussians", pa.int64()),
        # Timing
        pa.field("step_total_ms", pa.float32()),
        # BVH stats from OptiX tracer (zeros when backend has no BVH, e.g. 3DGUT)
        pa.field("bvh_last_build_time_ms", pa.float32()),
        pa.field("bvh_primitive_count", pa.int64()),
        pa.field("bvh_gas_buffer_bytes", pa.int64()),
        pa.field("bvh_gas_buffer_tmp_bytes", pa.int64()),
        pa.field("bvh_g_prim_aabb_bytes", pa.int64()),
        pa.field("bvh_last_build_was_full_rebuild", pa.bool_()),
    ]
    return pa.schema(fields)


def _scalar_summaries(t: torch.Tensor) -> tuple[float, float, float]:
    """Return (mean, p95, max) for a 1-D non-negative tensor."""
    if t is None or t.numel() == 0:
        return (0.0, 0.0, 0.0)
    t = t.detach().float()
    return (
        float(t.mean().item()),
        float(bounded_quantile(t, 0.95).item()),
        float(t.max().item()),
    )


class DiagnosticsWriter:
    """Buffered per-step Parquet writer.

    `log(step, **metrics)` appends a row to an in-memory buffer.
    `flush()` writes the buffer as a single Parquet row group via
    `ParquetWriter.write_table(...)`. Auto-flushes when buffer reaches
    `flush_steps` rows. `close()` performs a final flush and closes the
    file. Missing metrics for a row are recorded as `None` (NaN in
    pandas).
    """

    def __init__(self, output_path: str, flush_steps: int = 100) -> None:
        self.output_path = output_path
        self.flush_steps = max(1, int(flush_steps))
        self._schema = _build_schema()
        self._field_names = [f.name for f in self._schema]
        self._buffer: dict[str, list[Any]] = defaultdict(list)
        self._writer: pq.ParquetWriter | None = None
        self._rows = 0
        logger.info(f"DiagnosticsWriter -> {output_path} (flush every {self.flush_steps} steps)")

    def _ensure_writer(self) -> None:
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.output_path, self._schema, compression="snappy")

    def log(self, step: int, **metrics: float | int | None) -> None:
        """Append a row. Missing schema columns become null."""
        metrics = {**metrics, "step": int(step), "schema_version": SCHEMA_VERSION}
        for name in self._field_names:
            self._buffer[name].append(metrics.get(name))
        self._rows += 1
        if self._rows >= self.flush_steps:
            self.flush()

    def flush(self) -> None:
        if self._rows == 0:
            return
        try:
            self._ensure_writer()
            table = pa.table(
                {name: self._buffer.get(name, []) for name in self._field_names},
                schema=self._schema,
            )
            self._writer.write_table(table)
        except Exception as exc:
            logger.warning(f"DiagnosticsWriter flush failed: {exc}")
        finally:
            self._buffer.clear()
            self._rows = 0

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                logger.warning(f"DiagnosticsWriter close failed: {exc}")
            self._writer = None
        logger.info(f"DiagnosticsWriter closed: {self.output_path}")

    def compute_per_step_metrics(
        self,
        batch_losses: dict,
        grad_norms: dict,
        n_gaussians: int,
        step_total_ms: float,
        bvh_stats: dict | None = None,
    ) -> dict:
        """Convenience: collect the schema columns from training state."""
        out: dict[str, float | int] = {
            "total_loss": float(batch_losses.get("total_loss", 0.0)) if batch_losses else 0.0,
            "loss_l1": float(batch_losses.get("l1_loss", 0.0)) if batch_losses else 0.0,
            "loss_ssim": float(batch_losses.get("ssim_loss", 0.0)) if batch_losses else 0.0,
            "n_gaussians": int(n_gaussians),
            "step_total_ms": float(step_total_ms),
        }
        for attr in _GRAD_PARAMS:
            mean, p95, max_v = _scalar_summaries(grad_norms.get(attr))
            out[f"grad_{attr}_mean"] = mean
            out[f"grad_{attr}_p95"] = p95
            out[f"grad_{attr}_max"] = max_v
        bvh_stats = bvh_stats or {}
        out["bvh_last_build_time_ms"] = float(bvh_stats.get("last_build_time_ms", 0.0))
        out["bvh_primitive_count"] = int(bvh_stats.get("primitive_count", 0))
        out["bvh_gas_buffer_bytes"] = int(bvh_stats.get("gas_buffer_bytes", 0))
        out["bvh_gas_buffer_tmp_bytes"] = int(bvh_stats.get("gas_buffer_tmp_bytes", 0))
        out["bvh_g_prim_aabb_bytes"] = int(bvh_stats.get("g_prim_aabb_bytes", 0))
        out["bvh_last_build_was_full_rebuild"] = bool(
            bvh_stats.get("last_build_was_full_rebuild", False)
        )
        return out
