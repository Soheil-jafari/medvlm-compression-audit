# Status

**Last updated:** 2026-10-12, iteration 12 (interactive run mode)
**Overall:** IN PROGRESS (not `DONE`)
**Current stage:** S02 (CPU infrastructure); S01 done except dataset-access confirmations
**Article shape:** not yet known (decided by Gate 2 and H3; see `docs/FINAL_GOAL.md`)

## Next agent action

S02 (CPU infrastructure), in order: `evaluation/statistics.py` (paired bootstrap,
McNemar, Holm, seed aggregation, Kendall's tau), then the metric modules, then the
data pure functions, then the clinical-criterion pure parts, then aggregation.

## Waiting on the team

| # | What | Who | Since |
|---|---|---|---|
| 1 | Review `research-loop` and open the pull request into `main` | Soheil | 2026-10-12 |
| 2 | Optional: full names for the copyright line in `LICENSE` and `CITATION.cff` | team | 2026-10-12 |
| 2b | CI on `main` is red (import error fixed on `research-loop` in a7ec704); merging `research-loop` into `main` turns it green | Soheil | 2026-10-12 |
| 4 | Each member confirms own access to Kvasir-VQA-x1 and Kvasir-SEG (inbox) | all | 2026-10-12 |

## Stages

| Stage | Title | Status | Blocker |
|---|---|---|---|
| S00 | Repository health and loop setup | done, pending PR merge | team review |
| S01 | Decisions | done (dataset-access confirmations pending) | - |
| S02 | CPU infrastructure | in progress (statistics, metrics done) | - |
| S03 | Data pipeline | not started | - (datasets decided) |
| S04 | Model wrappers and unit surgery | not started | - (teacher decided) |
| S05 | Teacher: Gate 1, cache | not started | S03, S04 |
| S05b | Student selection 10 -> 5 -> 3 -> 1 | pool fixed: 10 Qwen-family models | S03, S04; freeze after RC-S0 |
| S06 | Students E1, E2, budget reference | not started | S05, S05b |
| S07 | Pre-registration freeze | not started | S03, S06 |
| S08 | Quantization E3, E4, E9 | not started | S07 |
| S09 | Pruning E5-E8, Gate 2 | not started | S07 |
| S10 | Ablations A1-A6 | not started | S09 |
| S11 | External test, timing, aggregation | not started | S08-S10 |
| S12 | Paper and release | not started | S11 (method/related work can start early) |

## Gates

| Gate | Status |
|---|---|
| Gate 1: teacher uses the image, damage visible | not run |
| Gate 2: E7 vs best of E5/E6 at B2 (validation) | not run |
| Pre-registration freeze | not frozen |
