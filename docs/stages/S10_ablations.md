# S10: Ablations A1-A6 at B2

**Goal.** Show which parts of the clinical criterion matter: weights (A1, A2),
random control (A3), calibration size (A4), Delta definition (A5), unit granularity
(A6).

**Depends on.** S09 (criterion and recovery code, E7 at B2).

## Tasks

- [ ] [team] Decide whether ablation variants identical to the E7 default (`A4:n128`,
  `A5:logprob_drop`) reuse the E7 B2 runs (proposal: yes; `docs/preregistration.md` section 7).
  Must be settled in the pre-registration.
- [ ] [agent] After D07 section 1: if part of the units are scored with the first-order estimate,
  restrict A5 to the exactly scored unit types (both variants) and drop `A5:logprob_drop` from
  `SAME_AS_E7`; compute A1/A2 by re-weighting E7's cached per-probe effects (D07 section 1).
- [x] [agent] Run cards for A1-A6 (RC-A1 to RC-A6): 27 runs if `A4:n128` and `A5:logprob_drop`
  reuse E7 at B2 (proposed, `registry.SAME_AS_E7`; a test checks the settings are identical),
  33 otherwise. A3 ranks its random scores without the per-parameter division (iteration 42).
- [ ] [team:Amir] Execute the run cards.
- [ ] [agent] `docs/results/A1.md` ... `A6.md` and the A4 calibration-size curve.

## Outputs

Ablation metrics at B2, 3 seeds each.

## Exit gate

All ablation runs committed; table T4 and the A4 appendix curve can be generated.
