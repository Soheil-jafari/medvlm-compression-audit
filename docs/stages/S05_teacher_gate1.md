# S05: Teacher (E0): Gate 1 and teacher cache

**Goal.** Show that the teacher actually uses the image on this data (Gate 1), then
cache its outputs once for all distillation and recovery runs.

**Depends on.** S03, S04.

## Tasks

- [x] [agent] Implement `evaluation/image_reliance.py` stage pieces
  (`build_stress_inputs`, `mask_audit`, `check_teacher_image_reliance`) and
  `models/teacher_cache.py` (top-k probabilities and the generated answer text,
  sharded files, manifest). The answer text allows sequence-level distillation if
  D05 chooses option B.
- [x] [agent] Gate 1 threshold proposal in `docs/decisions/D04_gate1.md` (for
  example: accuracy drop with a blank image, lesion-mask effect above zero with the
  CI excluding zero). Validation split only.
- [x] [team] Approve the Gate 1 thresholds in the inbox **before** RC-E0a runs (D04 answered 2026-10-12).
- [x] [agent] Evaluation runner (`evaluation/runner.py`, `probe_suites.py`, `free_text.py`; used by
  `scripts/06_evaluate.py` for every experiment) with tests on fake and tiny models.
- [x] [agent] Run cards **RC-E0a** (Gate 1 checks on the validation split) and
  **RC-E0b** (teacher cache on the training split, E0 evaluation on validation).
- [ ] [team:Soheil] Execute RC-E0a; execute RC-E0b only if Gate 1 passes.
- [ ] [agent] `docs/results/E0.md` from the committed CSVs, with a Gate 1 verdict
  proposal.
- [ ] [team] Confirm the Gate 1 verdict in `docs/decisions_log.md`.

## Outputs

Gate 1 result, teacher cache on team storage (manifest hash recorded in the run
card), E0 validation metrics.

## Exit gate

Gate 1 passed and logged; cache manifest hash recorded. If Gate 1 fails, the loop
stops and the team chooses a new pair, task or domain (`docs/FINAL_GOAL.md` section 7).
