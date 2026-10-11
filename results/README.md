# Results

Only small summary CSVs are tracked here. Raw predictions, per-sample outputs,
caches and checkpoints are written elsewhere (see `paths` in `configs/base.yaml`)
and are git-ignored.

## Results CSV schema (long format)

Every metric of every run is one row:

| column       | type   | description                                                                 |
|--------------|--------|-----------------------------------------------------------------------------|
| `experiment` | string | Experiment ID from the registry, optionally with a variant: `E7`, `E3:w4`, `E5:taylor`, `A4:n64` |
| `budget`     | string | `B1`, `B2`, `B3`, or empty for runs that are not budget-matched             |
| `seed`       | int    | `0`, `1`, `2` for training arms; `0` otherwise                              |
| `metric`     | string | Metric name, e.g. `small_finding_recall`, `lesion_mask_effect`, `ece`, `peak_memory_bytes` |
| `value`      | float  | Metric value                                                                |
| `ci_low`     | float  | Lower bound of the 95% paired bootstrap CI (empty if not computed)          |
| `ci_high`    | float  | Upper bound of the 95% paired bootstrap CI (empty if not computed)          |

The column order is defined once in `medcompress.utils.io.RESULTS_COLUMNS`; use
`medcompress.utils.io.append_results` to write rows.

## Layout

```
results/
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/metrics.csv      # per run (git-ignored if large)
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/efficiency.csv
  summary/                                                       # from scripts/08_aggregate_results.py
```

Primary endpoints (pre-registered): `small_finding_recall` and `lesion_mask_effect`
at `B2`, comparing `E7` against the best of `E5` and `E6`.
