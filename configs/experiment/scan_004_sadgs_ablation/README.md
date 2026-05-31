# Scan_004 SAD-GS ablation matrix — Hydra experiment configs

This directory holds version-controlled Hydra YAML configs for the
Scan_004 SAD-GS ablation matrix. It replaces the ephemeral
`/tmp/sadgs_ablation_driver.sh` from the original ablation, which had
two known failure modes:

1. Hand-assembled Hydra overrides as bash arrays let the depth-loss
   boolean-gate bug pass silently (`lambda_dense_depth=0.05` without
   `use_dense_depth=true`). The bug invalidated cells C and D of the
   original matrix.
2. Drivers in `/tmp/` are not reproducible from the repo SHA.

## Structure

| File | Purpose |
|---|---|
| `base.yaml` | Shared overrides — single source of truth for the matrix. |
| `cell_A_control.yaml` | Vanilla baseline. |
| `cell_B_sadgs_recal.yaml` | Max Planck SAD-GS, recalibrated thresholds (TAU=2.5, cap=500k). |
| `cell_C_grad.yaml` | DA3 gradient depth loss only (bug-fixed). |
| `sweep_lambda.yaml` | Hydra-Optuna sweep skeleton for UMich port (DEFERRED). |

## Running a single cell

```bash
cd dependencies/3dgrut
.venv/bin/python train.py \
    --config-name=experiment/scan_004_sadgs_ablation/cell_C_grad
```

The Pydantic cross-field validator in
`blk_windows/process_b2g/splat/experiments/config_validators.py` runs
at config-load and rejects inconsistent combinations (e.g. setting
`lambda_X` without the corresponding `use_X` boolean gate).

## Running the full matrix

```bash
.venv/bin/python -m blk_windows.process_b2g.splat.experiments.run_ablation_matrix \
    --cells A B C
```

The runner collects `metrics.json` from each cell and prints a markdown
comparison table.

## Adding a new cell

1. Copy an existing cell file as `cell_<X>_<description>.yaml`.
2. Update `defaults` list (usually just `base` + `_self_`).
3. Specify cell-specific overrides under nested keys.
4. Set `experiment_name` to match the file name minus `.yaml`.
5. If introducing a new cross-field invariant (e.g. a new `use_X` /
   `lambda_X` pair), add a Pydantic model_validator to
   `config_validators.py`.

## Historical results (sadgs_ablation_v1)

Pre-rebase, pre-Phase -1 infra. Recorded for comparison.

| Cell | masked PSNR | Δ vs A | Note |
|---|---|---|---|
| A control | 24.380 | — | vanilla baseline |
| B Max Planck default | 24.341 | −0.039 | TAU=1.0 too permissive |
| B2 Max Planck recal | 24.482 | +0.102 | TAU=2.5, below signal |
| C (buggy) | 24.345 | −0.035 | bool gate not set, loss never fired |
| D (buggy) | 24.357 | −0.023 | same bug as C |
| C-grad (bug-fixed) | 24.365 | −0.015 | depth-grad loss confirmed null on this regime |

See `blk_windows/process_b2g/splat/EXPERIMENTS.md` for the full
negative-result discussion.
