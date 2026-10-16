# Run card RC-L1: training-length check (E2 and E7 at 30,000 examples, B2, seed 0)

## 1. What and why

- Stage: S06 / S09 (secondary analysis, pre-registration section 6; decision 3 of 2026-10-16)
- Runs: 2, both seed 0:
  - **E2@30k:** E2 (D06 recipe) trained on the first 30,000 examples of the seeded training
    pool instead of 10,000: 1,875 steps.
  - **E7@30k at B2:** the clinical criterion applied to E2@30k: scored on E2@30k, pruned to the
    same absolute B2 target as the main runs, recovered for 25 % of the E2@30k steps on the first
    25 % of its pool (7,500 examples), with the rounding of the main recovery.
  Each is evaluated on the validation and the test split.
- Purpose: show whether the order of E2 and E7 on the primary endpoints holds with three times
  the training data. Not confirmatory: reported in the appendix whatever it shows, and it
  changes nothing in the main analysis.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- Inputs from the main study: RC-E2 seed 0 and RC-E7 B2 seed 0 done (the 10,000-example side of
  the comparison), RC-B0 done (the B2 target), the pre-registration frozen.
- Code the agent adds first (none of it exists yet):
  1. run identities for the two runs, so their results and checkpoints never overwrite the main
     E2 and E7 seed 0;
  2. the parent of a pruning arm read from its config instead of the fixed "E2", in the four
     places that load it (scoring, removal and recovery, the evaluation reload, the score
     fingerprint), so that E7@30k starts from E2@30k;
  3. an extension of the teacher cache from 10,000 to 30,000 examples: the 30,000 begin with
     the 10,000, so the first 10 shards are reused and only 20,000 answers are generated (the
     cache now refuses any change of its pool); E7@30k's recovery examples (the first 7,500) are
     already inside the main cache;
  4. the comparison rows of pre-registration section 6 in the comparison step.
- Settings: the image side (RC-S0b), the rounding of the recovery steps (D06), and everything
  RC-E2 and RC-E7 need.

Platform: Kaggle, 2 x T4 for the cache extension (the teacher), one T4 for the training,
scoring, recovery and evaluation. Dataset access as in RC-E2 and RC-E7.

Runtime (all **estimates** at the image side 448; they scale with the RC-S0b numbers if another
side is chosen):

| Step | Estimate |
|---|---|
| Teacher cache, 20,000 more examples | 20,000 x (0.314 s + T / 4.48 tokens per s), T from RC-D1 (RC-M1 teacher speeds) |
| E2@30k training | 30,000 examples x 0.74 s, about 6.2 h (RC-S0's one-example LoRA step) |
| E7@30k scoring | about 3.8 h (D07, proposed mix, 0.199 s per probe) |
| E7@30k recovery | 7,500 x 0.74 s, about 1.5 h |
| Footprint search, 4 evaluation passes, benchmark | not measured yet (RC-D1 gives the split sizes) |

About 11.5 h plus the cache extension and the evaluations.

## 3. Commands

Written when the card is set to `ready`: the run identities of section 2 do not exist yet.

## 4. What to commit back

The `results/` files of the two runs (metrics, efficiency, removal and budget search for
E7@30k, the unit scores), and this card with section 6 filled. Checkpoints and `outcomes*.npz`
stay on team storage as private datasets; the comparison step needs the test-split outcomes.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Only the data grew | E2@30k `metrics.csv` | `train_examples_seen` = 30,000 and `train_steps` = 1,875 |
| Recovery as pre-registered | E7@30k `metrics.csv` | `recovery_steps` = 25 % of 1,875 with the main rounding |
| Budget met | E7@30k `budget_within_tolerance` | 1 (the same B2 target as the main E7 seed 0) |

The recovery fingerprint of E7@30k differs from the main pruning arms only in its steps and
examples; that is the point of the check.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- Per step: wall time, peak GPU memory:
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
- Anything unusual:
