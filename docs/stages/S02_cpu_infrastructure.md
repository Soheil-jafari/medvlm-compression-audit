# S02: CPU infrastructure (no GPU, no data)

**Goal.** Every piece of the pipeline that is plain arithmetic on predictions, labels
and CSVs is implemented and unit-tested, so that GPU runs only produce raw outputs
and everything after them is automatic.

**Depends on.** S00. Independent of the open decisions, so this is the agent's main
work while S01 waits on the team.

## Tasks

All with unit tests on small synthetic inputs. Replace entries in
`test_stubs_raise_not_implemented_with_todo` as stubs are implemented.

- [x] [agent] `evaluation/statistics.py`: paired bootstrap CI, McNemar test, Holm
  correction, mean and SD over seeds, Kendall's tau with bootstrap CI (for H4).
- [x] [agent] `evaluation/clinical.py`: per-finding sensitivity and specificity,
  macro-F1, overall accuracy.
- [x] [agent] `evaluation/small_findings.py` and `rare_findings.py`: recall by
  lesion-size stratum and by prevalence group.
- [x] [agent] `evaluation/calibration.py`: ECE, Brier score, risk-coverage curve.
- [x] [agent] `evaluation/image_reliance.py`: `answer_flip_rate`,
  `lesion_mask_effect` (the pure parts).
- [x] [agent] `evaluation/language.py` and `hallucination.py`: yes/no answer parsing
  and hallucination / omission / negation rates from parsed answers.
- [x] [agent] `data/probes.build_probes`, `data/splits` (patient splits, nested
  calibration subsets, split fingerprint), `data/masks` pure functions on arrays,
  `data/dedup` distance logic (imaging libraries imported lazily).
- [x] [agent] `compression/pruning/criteria/clinical.py` pure parts:
  `load_finding_weights`, `compute_weights`, `delta`, `aggregate_importance`.
- [x] [agent] `budgets.py`: budget search against a measured-footprint function
  (tested with a mock).
- [ ] [agent] `evaluation/statistics.aggregate_results` and
  `scripts/08_aggregate_results.py`: read `results/`, write `results/summary/`,
  every table and figure listed in `docs/FINAL_GOAL.md` section 3,
  `paper/tables/numbers.tex` and `docs/results/VERDICTS.md`. End-to-end test on a
  synthetic results tree that lives only under `tests/fixtures/` and is labelled
  synthetic in every file.
- [x] [agent] Tests over the real `results/` tree: every CSV matches the schema;
  budget-matched runs are within tolerance (active once budgets are frozen).
- [ ] [agent] Test that `paper/sections/05_results.tex` and the abstract contain no
  hand-typed result numbers (only `\input` tables and `numbers.tex` macros).

## Outputs

Implemented modules with tests; a working `08_aggregate_results.py` on synthetic data.

## Exit gate

All tasks ticked; the aggregation step produces every table and figure file from the
synthetic fixture; CI green.
