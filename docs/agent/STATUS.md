# Status

**Last updated:** 2026-10-12, iteration 7 (interactive run mode)
**Overall:** IN PROGRESS (not `DONE`)
**Current stage:** S01 (decisions), with S02 worked in parallel
**Article shape:** not yet known (decided by Gate 2 and H3; see `docs/FINAL_GOAL.md`)

## Next agent action

1. S01: D02 datasets memo (`docs/decisions/D02_datasets.md`), including a re-check of
   the new retinal VQA sets as a possible second domain.
2. Then S02 (CPU infrastructure), starting with `evaluation/statistics.py`.

## Waiting on the team

| # | What | Who | Since |
|---|---|---|---|
| 1 | Review `research-loop` and open the pull request into `main` | Soheil | 2026-10-12 |
| 2 | Answer D03 (licence): memo `docs/decisions/D03_licence.md`; also confirm full names for the copyright line and any institutional licence or IP policy | team | 2026-10-12 |
| 2b | CI on `main` is red (import error fixed on `research-loop` in a7ec704); merging `research-loop` into `main` turns it green | Soheil | 2026-10-12 |
| 3 | Answer D02 (datasets): memo `docs/decisions/D02_datasets.md` | team | 2026-10-12 |
| 4 | Each member confirms own access to the chosen datasets | all | - |

## Stages

| Stage | Title | Status | Blocker |
|---|---|---|---|
| S00 | Repository health and loop setup | done, pending PR merge | team review |
| S01 | Decisions | in progress | D01 and D05 decided; D02 and D03 waiting on the team |
| S02 | CPU infrastructure | not started | - |
| S03 | Data pipeline | not started | S01 datasets |
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
