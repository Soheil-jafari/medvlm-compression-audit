# Stages

The project is split into 14 stages (S00-S12 plus S05b). Each stage file lists its
goal, what it depends on, the tasks (tagged `[agent]` or `[team:<name>]`), its outputs and the
**exit gate** that must hold before the stage counts as done. Live status is in
`docs/agent/STATUS.md`; the end target is `docs/FINAL_GOAL.md`.

| Stage | Title | Depends on | Mostly done by |
|---|---|---|---|
| [S00](S00_repo_health.md) | Repository health and loop setup | - | agent |
| [S01](S01_decisions.md) | Decisions: domain, teacher, datasets, licence, selection protocol | S00 | agent memos, team decides |
| [S02](S02_cpu_infrastructure.md) | CPU infrastructure: metrics, statistics, aggregation | S00 | agent |
| [S03](S03_data_pipeline.md) | Data pipeline: readers, splits, masks, dedup, data cards | S01, S02 | agent code, team runs on real data |
| [S04](S04_model_wrappers.md) | Model wrappers and unit surgery | S01 | agent code, team GPU smoke test |
| [S05](S05_teacher_gate1.md) | Teacher (E0): Gate 1 and teacher cache | S03, S04 | team GPU runs, agent analysis |
| [S05b](S05b_student_selection.md) | Student selection, 10 -> 5 -> 3 -> 1 (memo D05) | S01, S03, S04 (R2-R3 also S05) | agent code, team GPU runs |
| [S06](S06_student_training.md) | Students E1, E2 and the budget reference | S05, S05b | team GPU runs, agent code |
| [S07](S07_preregistration.md) | Pre-registration freeze | S03, S06 | agent draft, team approves |
| [S08](S08_quantization.md) | Quantization arms E3, E4, E9 | S07 | agent code, team GPU runs |
| [S09](S09_pruning_gate2.md) | Pruning arms E5-E8 and Gate 2 | S07 | agent code, team GPU runs |
| [S10](S10_ablations.md) | Ablations A1-A6 at B2 | S09 | team GPU runs, agent analysis |
| [S11](S11_final_evaluation.md) | External test, timing GPU, aggregation, verdicts | S08-S10 | agent, team timing runs |
| [S12](S12_paper.md) | Paper and release | S11 (drafting starts early) | agent drafts, team reviews |

S02 does not depend on any open decision and is where the agent works while S01
waits on the team. Paper sections that do not need results (method, related work,
setup) can be drafted at any time.

## Gates

- **Gate 1 (S05):** the teacher uses the image (blank-image and lesion-mask checks
  pass the pre-agreed thresholds) and compression damage is measurable. If it fails,
  the loop stops and the team chooses a new pair, task or domain.
- **Gate 2 (S09):** on the validation split at B2, does E7 beat the best of E5/E6?
  This decides the framing (shape A or B in `docs/FINAL_GOAL.md`). It never stops
  the pre-registered test-split runs.
