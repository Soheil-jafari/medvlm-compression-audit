# Status

**Last updated:** 2026-10-16, iteration 48 (interactive run mode)
**Overall:** IN PROGRESS (not `DONE`)
**Current stage:** S05b: image side measured by RC-S0b (ready, next card); S03 waiting on RC-D1
**Article shape:** not yet known (decided by Gate 2 and H3; see `docs/FINAL_GOAL.md`)

## Next agent action

0. Bug hunt requested by Soheil (iteration 35 onwards): evaluation and statistics fixed (split
   files, image-level bootstrap, bootstrap memory); training pool and teacher cache fixed
   (iteration 36). Pruning, budgets and efficiency next. Then quantization (S08).
1. Student: Qwen2.5-VL-3B chosen with checks (D05 amendment, 2026-10-12; fallback Qwen3-VL-2B).
   RC-M1 and RC-Q0 passed (2026-10-15; kernel class and versions pinned, iteration 46). RC-S0
   passed for both models (2026-10-16). Soheil's answers (2026-10-16, iteration 48): 10,000
   examples and 625 steps decided; the image side is chosen by RC-S0b (448 / 560 / 672) and the
   D05 rule; a training-length check (RC-L1, 30,000 examples) is pre-registered; G2 still a
   proposal. Next: read RC-S0b, project the GPU hours per side, propose the side; then pin the
   student revision. RC-L1 needs code first (run identities, parent from config, cache
   extension, comparison rows).
2. Quantization stage (S08): code and run cards done (iterations 37-39). Pre-registration
   draft and freeze guard done (iteration 40). Pruning grid cards RC-E5 to RC-E8 done
   (iteration 41); ablation cards RC-A1 to RC-A6 done, A3 ranking and score reuse fixed
   (iteration 42); comparison step, RC-C1 and RC-R1 done (iteration 43); RC-T1, RC-X1 part A
   and the A4 curve done (iteration 44); appendix outputs done (iteration 45). Next: the paper
   text (S12) while the run cards wait; the EndoBench loader after RC-X1 part A.
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
| 10 | Answer the [team] items of the pre-registration draft (comparator rule, headline accuracy for H1/H4, finding thresholds, prevalence cut points, cap, tiers, training pool, A3 ranking, reuse of E7 for `A4:n128` and `A5:logprob_drop`, H2 matching, McNemar on the majority answer) | all | 2026-10-12 |
| 11 | Choose the rented timing GPU and the precision (`hardware.final_timing`, `hardware.precision`) | team | 2026-10-12 |
| 12 | Run RC-X1 part A: EndoBench terms, format and overlap (no GPU) | Farzan | 2026-10-12 |
| 13 | Run RC-S0b (`docs/runs/RC-S0b_image_size_measurement.md`, Kaggle 2 x T4) and commit `results/smoke/RC-S0b/` | Soheil | 2026-10-16 |
| 14 | State the GPU hours the team has for the study (`TODO: team GPU hours available`, D05 image-side rule) | team | 2026-10-16 |
| 15 | Decide the G2 threshold (proposal 0.10) and the recovery rounding (proposal 157 steps) | team | 2026-10-16 |

## Stages

| Stage | Title | Status | Blocker |
|---|---|---|---|
| S00 | Repository health and loop setup | done, pending PR merge | team review |
| S01 | Decisions | done (dataset-access confirmations pending) | - |
| S02 | CPU infrastructure | done | - |
| S03 | Data pipeline | waiting on RC-D1 | team runs RC-D1 |
| S04 | Model wrappers and unit surgery | done (RC-M1 passed 2026-10-15) | - |
| S05 | Teacher: Gate 1, cache | code and run cards done (RC-E0a, RC-E0b blocked) | RC-D1, open settings |
| S05b | Student checks (D05 amendment: one student) | RC-S0 done 2026-10-16: both pass G1; training size decided; RC-S0b ready (image side) | RC-S0b, team GPU hours, G2; then the selection freeze |
| S06 | Students E1, E2, budget reference | trainer done; recipe memo and run cards next | S05, S05b |
| S07 | Pre-registration freeze | draft and freeze guard done; values after RC-D1, RC-S0, RC-B0; team items open | RC-D1, team answers |
| S08 | Quantization E3, E4, E9 | RC-Q0 done 2026-10-15 (TorchLinear, versions pinned); RC-E3, RC-E4, RC-E9 blocked | S07 freeze, E2, RC-D1 |
| S09 | Pruning E5-E8, Gate 2 | code and run cards done (RC-G2, RC-E5 to RC-E8, all blocked) | D07 (scoring method, budgets), S07 freeze, E2 |
| S10 | Ablations A1-A6 | run cards done (RC-A1 to RC-A6, blocked); team: A3 ranking, E7 reuse | S09, S07 freeze |
| S11 | External test, timing, aggregation | code and cards done (comparison step, appendix; RC-R1, RC-C1, RC-T1, RC-X1 part A); waiting for results | S08-S10; RC-X1 part A for the EndoBench loader |
| S12 | Paper and release | not started | S11 (method/related work can start early) |

## Gates

| Gate | Status |
|---|---|
| Gate 1: teacher uses the image, damage visible | not run |
| Gate 2: E7 vs best of E5/E6 at B2 (validation) | not run |
| Pre-registration freeze | not frozen |
