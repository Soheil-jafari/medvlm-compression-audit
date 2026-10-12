# S11: External test, timing GPU, aggregation and verdicts

**Goal.** Produce every number the article needs, with the pre-registered
statistics, and decide H1-H4.

**Depends on.** S08, S09, S10.

## Tasks

- [x] [agent] Comparison step `scripts/09_compare_arms.py` (H1-H4 evidence from the per-probe
  test outcomes, seeds averaged, image-level bootstrap) and run card **RC-C1**; run card
  **RC-R1** for the test-split evaluation of E0, E1 and E2 after the freeze.
- [ ] [team:Soheil] Execute RC-R1 and, when every arm is evaluated, RC-C1.
- [ ] [agent] Run card **RC-X1**: main arms on the deduplicated external test set
  (table T6).
- [ ] [team:Farzan] Execute RC-X1.
- [ ] [agent] Run card **RC-T1**: final timing of all arms on the rented timing GPU,
  with the pre-registered protocol (table T5), including the low-precision kernel
  check.
- [ ] [team:Amir] Rent the timing GPU and execute RC-T1.
- [ ] [agent] Run `scripts/08_aggregate_results.py` over all committed results;
  commit `results/summary/`, `paper/tables/`, `paper/figures/`,
  `paper/tables/numbers.tex` and `docs/results/VERDICTS.md`.
- [ ] [agent] Tick sections 2, 3 and 4 of `docs/FINAL_GOAL.md`, each with a pointer
  to the evidence file.

## Outputs

All tables and figures, verdicts for H1-H4.

## Exit gate

`docs/FINAL_GOAL.md` sections 2-4 fully ticked.
