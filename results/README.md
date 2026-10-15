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
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/metrics.csv      # per run, validation split
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/metrics_test.csv # per run, test split (after the freeze)
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/efficiency.csv       # development GPU (Kaggle T4)
  <ID>[/<variant>]/<budget or nobudget>/seed<k>/efficiency_final.csv # rented timing GPU (RC-T1), metrics "final:<name>"
  comparisons/<A>_vs_<B>/<budget or nobudget>/comparison.csv     # paired differences with CIs (step 09); ":" in a name is "-" in the folder
  comparisons/choices.json                                       # comparator, matched budget, arms left out (step 09)
  selection/<round>/<candidate>/metrics.csv                      # student selection (D05), e.g. R1:qwen3_vl_2b
  data/summary.csv                                               # aggregate data statistics (RC-D1), no per-image rows
  data/<name>_overlap.csv                                        # external images that are near-copies of ours (RC-X1)
  smoke/<card>.csv                                               # smoke tests, e.g. RC-M1
  summary/                                                       # written by scripts/08_aggregate_results.py
```

Rows in a run file must match their folder (experiment, budget, seed).

The split is part of the file name, not a column: `metrics.csv` holds validation scores (and
the training, removal and recovery bookkeeping of the run), `metrics_test.csv` the test scores.
Scoring the test split never overwrites the validation rows. The aggregation adds a `split`
column (`val`, `test`, or empty for `efficiency.csv`, comparisons and data rows), summarises the
splits apart in `summary/by_arm.csv`, and builds the paper tables, figures, `numbers.tex` and
the verdicts from the test split and the split-free rows only. Validation rows serve Gate 2, the
choice of the best of E5/E6 and the student checks. Comparison rows are computed on the test
split unless their run card says otherwise. Comparison
rows use `experiment = "<A>_vs_<B>"` and metrics such as `diff:small_finding_recall`
or `p:small_finding_recall`; `scripts/09_compare_arms.py` computes them where the per-probe
outputs (`outcomes_test.npz`) are gathered (run card RC-C1), because per-probe outputs are never
committed.

Before committing results, run `pytest tests/test_results.py`: it validates every CSV
in this folder (schema, known experiment IDs and variants, budgets, seeds, CI order,
folder consistency).

Primary endpoints (pre-registered): `small_finding_recall` and `lesion_mask_effect`
at `B2`, comparing `E7` against the best of `E5` and `E6`.
