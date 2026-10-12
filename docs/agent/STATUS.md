# Status

**Last updated:** 2026-10-12, iteration 29 (interactive run mode)
**Overall:** IN PROGRESS (not `DONE`)
**Current stage:** S05 (teacher cache, Gate 1 tooling); S03 waiting on RC-D1, S04 waiting on RC-M1
**Article shape:** not yet known (decided by Gate 2 and H3; see `docs/FINAL_GOAL.md`)

## Next agent action

1. S09: recovery (`run_prune_and_recover`), matched-budget search, efficiency measurement.
2. D06 memo (training recipe, calibration bins, bootstrap resamples, image size, token
   budget) and D07 memo (pruning scoring settings) for the team to confirm.
3. S06: run cards for E1/E2 and the budget reference; S12 work-ahead (related work).

## Waiting on the team

| # | What | Who | Since |
|---|---|---|---|
| 1 | Review `research-loop` and open the pull request into `main` | Soheil | 2026-10-12 |
| 2 | Optional: full names for the copyright line in `LICENSE` and `CITATION.cff` | team | 2026-10-12 |
| 2b | CI on `main` is red (import error fixed on `research-loop` in a7ec704); merging `research-loop` into `main` turns it green | Soheil | 2026-10-12 |
| 4 | Each member confirms own access to Kvasir-VQA-x1 and Kvasir-SEG (inbox) | all | 2026-10-12 |
| 5 | Run RC-D1 (`docs/runs/RC-D1_data_preparation.md`) and commit `results/data/summary.csv` | Farzan | 2026-10-12 |
| 6 | Run RC-M1 (`docs/runs/RC-M1_model_smoke_test.md`, Kaggle 2 x T4) and commit `results/smoke/RC-M1.csv` | Soheil | 2026-10-12 |

## Stages

| Stage | Title | Status | Blocker |
|---|---|---|---|
| S00 | Repository health and loop setup | done, pending PR merge | team review |
| S01 | Decisions | done (dataset-access confirmations pending) | - |
| S02 | CPU infrastructure | done | - |
| S03 | Data pipeline | waiting on RC-D1 | team runs RC-D1 |
| S04 | Model wrappers and unit surgery | code done; waiting on RC-M1 | team runs RC-M1 |
| S05 | Teacher: Gate 1, cache | code and run cards done (RC-E0a, RC-E0b blocked) | RC-D1, RC-M1, open settings |
| S05b | Student selection 10 -> 5 -> 3 -> 1 | code, ranking and run cards RC-S0 to RC-S3 done (blocked) | RC-M1, then RC-S0; freeze after RC-S0 |
| S06 | Students E1, E2, budget reference | trainer done; recipe memo and run cards next | S05, S05b |
| S07 | Pre-registration freeze | not started | S03, S06 |
| S08 | Quantization E3, E4, E9 | not started | S07 |
| S09 | Pruning E5-E8, Gate 2 | criteria and scoring stage done; recovery, budget search next | S07 |
| S10 | Ablations A1-A6 | not started | S09 |
| S11 | External test, timing, aggregation | not started | S08-S10 |
| S12 | Paper and release | not started | S11 (method/related work can start early) |

## Gates

| Gate | Status |
|---|---|
| Gate 1: teacher uses the image, damage visible | not run |
| Gate 2: E7 vs best of E5/E6 at B2 (validation) | not run |
| Pre-registration freeze | not frozen |
