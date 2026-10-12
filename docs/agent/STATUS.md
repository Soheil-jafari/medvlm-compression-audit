# Status

**Last updated:** 2026-10-12, iteration 42 (interactive run mode)
**Overall:** IN PROGRESS (not `DONE`)
**Current stage:** S05 (teacher cache, Gate 1 tooling); S03 waiting on RC-D1, S04 waiting on RC-M1
**Article shape:** not yet known (decided by Gate 2 and H3; see `docs/FINAL_GOAL.md`)

## Next agent action

0. Bug hunt requested by Soheil (iteration 35 onwards): evaluation and statistics fixed (split
   files, image-level bootstrap, bootstrap memory); training pool and teacher cache fixed
   (iteration 36). Pruning, budgets and efficiency next. Then quantization (S08).
1. Student: Qwen2.5-VL-3B chosen with checks (D05 amendment, 2026-10-12; fallback Qwen3-VL-2B).
   After RC-M1 and RC-S0, propose the image side, the G2 threshold and the training length from
   the measured speed, and pin the revision.
2. Quantization stage (S08): code and run cards done (iterations 37-39). Pre-registration
   draft and freeze guard done (iteration 40). Pruning grid cards RC-E5 to RC-E8 done
   (iteration 41); ablation cards RC-A1 to RC-A6 done, A3 ranking and score reuse fixed
   (iteration 42). Next: aggregation and figures, RC-X1, RC-T1 and the paper (S11, S12).
3. D07 section 1 (how a unit is switched off while scored), the timing hardware and B1-B3 are
   asked later; none is needed before E2 exists.

Nothing heavy runs in the agent session: training, scoring and measurements are written as run
cards for the team (decision of 2026-10-12).

## Waiting on the team

| # | What | Who | Since |
|---|---|---|---|
| 1 | Review `research-loop` and open the pull request into `main` | Soheil | 2026-10-12 |
| 2 | Optional: full names for the copyright line in `LICENSE` and `CITATION.cff` | team | 2026-10-12 |
| 2b | CI on `main` is red (import error fixed on `research-loop` in a7ec704); merging `research-loop` into `main` turns it green | Soheil | 2026-10-12 |
| 4 | Each member confirms own access to Kvasir-VQA-x1 and Kvasir-SEG (inbox) | all | 2026-10-12 |
| 5 | Run RC-D1 (`docs/runs/RC-D1_data_preparation.md`) and commit `results/data/summary.csv` | Farzan | 2026-10-12 |
| 7 | Tell Amir and Farzan that the staged selection was replaced by one student with checks (D05 amendment) | Soheil | 2026-10-12 |
| 8 | Confirm the fixed training pool (one pool from `seeds.data_split` for E1, E2, E4 and the recoveries; the teacher cache holds only that pool) | Soheil | 2026-10-12 |
| 10 | Answer the [team] items of the pre-registration draft (comparator rule, headline accuracy for H1/H4, finding thresholds, prevalence cut points, cap, tiers, training pool, A3 ranking, reuse of E7 for `A4:n128` and `A5:logprob_drop`) | all | 2026-10-12 |
| 9 | Run RC-Q0 (`docs/runs/RC-Q0_quantization_smoke_test.md`, one T4, no data) and commit `results/smoke/RC-Q0.csv` | Soheil | 2026-10-12 |
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
| S07 | Pre-registration freeze | draft and freeze guard done; values after RC-D1, RC-S0, RC-B0; team items open | RC-D1, team answers |
| S08 | Quantization E3, E4, E9 | code and run cards done (RC-Q0 ready; RC-E3, RC-E4, RC-E9 blocked) | RC-Q0, S07 freeze, E2 |
| S09 | Pruning E5-E8, Gate 2 | code and run cards done (RC-G2, RC-E5 to RC-E8, all blocked) | D07 (scoring method, budgets), S07 freeze, E2 |
| S10 | Ablations A1-A6 | run cards done (RC-A1 to RC-A6, blocked); team: A3 ranking, E7 reuse | S09, S07 freeze |
| S11 | External test, timing, aggregation | not started | S08-S10 |
| S12 | Paper and release | not started | S11 (method/related work can start early) |

## Gates

| Gate | Status |
|---|---|
| Gate 1: teacher uses the image, damage visible | not run |
| Gate 2: E7 vs best of E5/E6 at B2 (validation) | not run |
| Pre-registration freeze | not frozen |
