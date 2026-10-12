# S10: Ablations A1-A6 at B2

**Goal.** Show which parts of the clinical criterion matter: weights (A1, A2),
random control (A3), calibration size (A4), Delta definition (A5), unit granularity
(A6).

**Depends on.** S09 (criterion and recovery code, E7 at B2).

## Tasks

- [ ] [team] Decide whether ablation variants identical to the E7 default (for
  example `A5:logprob_drop`) reuse the E7 B2 runs (open item in
  `docs/experiment_plan.md`). Must be settled in the pre-registration.
- [ ] [agent] Run cards for A1-A6 (33 runs, or fewer if E7 runs are reused).
- [ ] [team:Amir] Execute the run cards.
- [ ] [agent] `docs/results/A1.md` ... `A6.md` and the A4 calibration-size curve.

## Outputs

Ablation metrics at B2, 3 seeds each.

## Exit gate

All ablation runs committed; table T4 and the A4 appendix curve can be generated.
