# S09: Pruning arms E5-E8 and Gate 2

**Goal.** Generic pruning (E5), output-similarity pruning (E6), the clinical
criterion (E7) and E7 + 4-bit (E8) at B1-B3 with identical recovery, plus the Gate 2
reading.

**Depends on.** S07 (frozen), S06, S04 (surgery).

## Tasks

- [x] [agent] Criteria: `magnitude`, `activation`, `taylor`, `output_similarity`,
  `random`, and the model-facing parts of `clinical` (`score`, `ablate`), with the scoring
  stage `scripts/04_score_pruning_units.py` (open settings: group size, calibration size,
  and for E7 the clinical keys; decided by the D07 memo).
- [x] [agent] Budget-matched removal (`budgets.search_matched_budget`,
  `pruning/apply.py`) and the shared recovery (`compression/recovery.py`: LoRA +
  offline KD, same subset and steps for every arm), and the efficiency measurement
  (`evaluation/efficiency.py`, stage 07). Open settings are named by the stages (exit 4).
- [ ] [agent] Run card **RC-G2** first: E5 (3 criteria), E6 and E7 at B2, seed 0,
  scored on the **validation** split only.
- [ ] [team:Amir] Execute RC-G2.
- [ ] [agent] Gate 2 note in `docs/results/GATE2.md`: E7 vs best of E5/E6 on
  validation, and the implied paper shape (A or B). Gate 2 does not stop any run.
- [ ] [team] Record the Gate 2 reading in `docs/decisions_log.md`.
- [ ] [agent] Run cards for the full grid: E5 (3 criteria x 3 budgets x 3 seeds),
  E6 (3 x 3), E7 (3 x 3), E8 (3 x 3), each evaluated on validation and test.
- [ ] [team:Amir] Execute the grid.
- [ ] [agent] `docs/results/E5.md` ... `E8.md`; check every footprint is within the
  pre-registered tolerance.

## Outputs

45 pruning runs and 9 E8 runs with metrics, footprints and mask audits; Gate 2 note.

## Exit gate

All runs committed and within budget tolerance; Gate 2 logged.
