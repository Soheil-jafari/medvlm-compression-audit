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
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/metrics.csv      # per run
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/efficiency.csv
  comparisons/<A>_vs_<B>/<budget or nobudget>/comparison.csv     # paired differences with CIs
  selection/<round>/<candidate>/metrics.csv                      # student selection (D05), e.g. R1:qwen3_vl_2b
  data/summary.csv                                               # aggregate data statistics (RC-D1), no per-image rows
  smoke/<card>.csv                                               # smoke tests, e.g. RC-M1
  summary/                                                       # written by scripts/08_aggregate_results.py
```

Rows in a run file must match their folder (experiment, budget, seed). Comparison
rows use `experiment = "<A>_vs_<B>"` and metrics such as `diff:small_finding_recall`
or `p:small_finding_recall`; they are computed where the per-image predictions live
(the GPU session), because per-image outputs are never committed.

Before committing results, run `pytest tests/test_results.py`: it validates every CSV
in this folder (schema, known experiment IDs and variants, budgets, seeds, CI order,
folder consistency).

Primary endpoints (pre-registered): `small_finding_recall` and `lesion_mask_effect`
at `B2`, comparing `E7` against the best of `E5` and `E6`.
