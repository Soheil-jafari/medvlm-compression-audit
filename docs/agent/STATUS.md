# Status

**Last updated:** 2026-10-12, iteration 4 (interactive run mode)
**Overall:** IN PROGRESS (not `DONE`)
**Current stage:** S01 (decisions), with S02 worked in parallel
**Article shape:** not yet known (decided by Gate 2 and H3; see `docs/FINAL_GOAL.md`)

## Next agent action

1. S01: write the D01 memo, now **teacher only**, in `docs/decisions/D01_teacher.md`,
   then D02 (datasets). The student is chosen by the staged selection in S05b (memo
   D05, written in iteration 3); student candidate facts are already in
   `configs/models/*_student.yaml`. D01 and D02 need Hugging Face pages: in
   iteration 2 `huggingface.co` was not reachable from the unattended session. If
   that persists, prepare a small CPU script plus run card that collects the facts
   (revisions, licences, tokenizer-file hashes, parameter counts) for a team member
   to run.
2. While the team answers: S02, starting with `evaluation/statistics.py` (paired
   bootstrap, McNemar, Holm, seed aggregation, Kendall's tau).

## Waiting on the team

| # | What | Who | Since |
|---|---|---|---|
| 1 | Review `research-loop` and open the pull request into `main` | Soheil | 2026-10-12 |
| 2 | Answer D03 (licence): memo `docs/decisions/D03_licence.md`; also confirm full names for the copyright line and any institutional licence or IP policy | team | 2026-10-12 |
| 2b | CI on `main` is red (import error fixed on `research-loop` in a7ec704); merging `research-loop` into `main` turns it green | Soheil | 2026-10-12 |
| 3 | Answer D01 (teacher) and D02 (datasets) once the memos are up | team | - |
| 4 | Each member confirms own access to the chosen datasets | all | - |

## Stages

| Stage | Title | Status | Blocker |
|---|---|---|---|
| S00 | Repository health and loop setup | done, pending PR merge | team review |
| S01 | Decisions | in progress | D03, D05 memos up; D01 (teacher), D02 not written yet (hub pages unreachable in iteration 2) |
| S02 | CPU infrastructure | not started | - |
| S03 | Data pipeline | not started | S01 datasets |
| S04 | Model wrappers and unit surgery | not started | S01 model pair |
| S05 | Teacher: Gate 1, cache | not started | S03, S04 |
| S05b | Student selection 10 -> 5 -> 3 -> 1 | D05 answered (teacher family only, keep size, weights accepted) | D01 teacher fixes the pool; S03, S04 |
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
