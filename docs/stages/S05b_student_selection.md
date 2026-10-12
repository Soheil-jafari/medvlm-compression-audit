# S05b: Student selection (10 -> 5 -> 3 -> 1)

> **Amended 2026-10-12 (Soheil):** one student, Qwen2.5-VL-3B, with Round 0 and the image-use
> checks for the two models Qwen2.5-VL-3B and Qwen3-VL-2B (RC-S0). Rounds R1 to R3 and the run
> cards RC-S1 to RC-S3 are kept but not run. The tasks below describe the full protocol.

**Goal.** Choose the student every compression arm starts from, by a fixed staged
comparison of 10 architectures on the validation split: a simplified common screen
of 10, the top 5 rebuilt as new models with the full recipe, the top 3 trained with
the exact E2 recipe, then one winner.

**Depends on.** S01 (teacher decided; D05 answered and frozen), S03 (validation
split, masks), S04 (generic loader and yes/no scoring). Round 1 does not need the
teacher, so it can run while S05 runs; Rounds 2 and 3 need the S05 teacher cache.
Protocol: `docs/decisions/D05_student_selection.md`; settings:
`configs/student_selection.yaml`.

## Tasks

- [ ] [team] Answer D05 (Q1 tokenizer rule, Q2 size rule, Q3 score weights) and
  record the freeze date in `configs/student_selection.yaml` before Round 1.
- [x] [agent] Generic loader for the 10 candidate families (native classes where
  they exist, remote code otherwise), with each model's chat template, fp16, yes/no
  log-probability scoring and LoRA on decoder attention and MLP. Heavy tests on tiny
  random configs where the architecture class is available offline. (S04; the families
  that load only with their own code are listed in `models/loading.py` and need an
  explicit opt-in after the code has been read.)
- [x] [agent] Selection runner `scripts/select_student.py --round R1 --candidate NAME`
  (`selection/rounds.py`): reads `configs/student_selection.yaml`, runs one round for one
  candidate and writes `results/selection/<round>/<candidate>/metrics.csv` in the results
  schema with `experiment` = `<round>:<candidate>` (for example `R1:qwen3_vl_2b`). The
  per-probe answers go to `$MEDCOMPRESS_OUTPUT_ROOT/selection/` and are never committed.
  Tested end to end on the tiny offline model (R0 to R3).
- [x] [agent] Ranking step (`selection/ranking.py`, `selection/driver.py`;
  `select_student.py --rank R1`, `--report`): score S, gates, paired bootstrap CIs between
  neighbouring ranks, and the advance list, written to
  `results/selection/<round>/ranking.csv` and `docs/results/SELECTION.md`. Needs the
  per-probe answers of all candidates in one place, so the person who ranks collects them
  (run cards RC-S1 to RC-S3); the report needs only committed CSVs.
- [x] [agent] Run card **RC-S0** (feasibility of all 10 on one T4). [ ] Then set
  `compute_unit_b`, the validation subset size, the G2 threshold and the image side from
  its numbers (proposal in the inbox; the team confirms the freeze).
- [ ] [team] Execute RC-S0; replace failures with reserves in order.
- [x] [agent] Run cards **RC-S1** (10 candidates), **RC-S2** (top 5, new models) and
  **RC-S3** (top 3, 2 seeds), written blocked; each is set to `ready` after the previous
  ranking is committed.
- [ ] [team:Soheil] [team:Amir] [team:Farzan] Execute RC-S1 to RC-S3 (split the
  candidates across the three Kaggle accounts).
- [ ] [team] Confirm the final student in `docs/decisions_log.md`; the agent then
  sets its config to `DECIDED` with a pinned revision and points
  `configs/base.yaml: components.student` at it.

## Outputs

`results/selection/R0-R3/`, `docs/results/SELECTION.md`, the decided student config,
and an appendix table of all four rounds for the paper.

## Exit gate

One student chosen by the frozen protocol, on validation data only, and logged;
teacher tokenizer compatibility confirmed at runtime if logit distillation is used.
