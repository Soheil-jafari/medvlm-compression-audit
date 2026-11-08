# Schedule: who runs what, week by week (written 2026-10-17)

Every number of hours below is an **estimate**. Each one comes from the RC-S0b run 2
measurements at the decided image side 672 (synthetic probes) and the decided recipe
(micro-batch 1 x 16), unless the row names another source. Where a number cannot be known yet,
it is written as a symbol and stays **unknown** until the named run measures it.

Budget: 90 Kaggle GPU hours per week, 30 per person (Soheil, Amir, Farzan; decision of
2026-10-17). A T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a
17-minute T4 x2 session used 16 minutes), so the hours of the cards on T4 x2 (RC-TT1, RC-E0a,
RC-E0b, RC-E9, RC-R1's teacher pass, RC-L1's cache extension) count as written. Not known here:
the session time limit (TODO: the team checks on Kaggle). Training saves its state and continues
from it (`docs/team_setup.md`).

## 1. Measured inputs (RC-S0b run 2, side 672, `results/smoke/RC-S0b/`)

| Quantity | Value |
|---|---|
| Student scoring, per probe | 0.414 s |
| Student LoRA step, one example (warm) | 0.849 s, so one update of 16 examples at 1 x 16 is about 13.6 s |
| Student generation, one answer at a time | 4.99 tokens per second |
| Teacher scoring, per probe (2 x T4) | 0.584 s |
| Teacher generation, one answer at a time | 2.66 tokens per second |

Measured on real data since then (they replace the RC-S0b step where they differ):

| Quantity | Value |
|---|---|
| E1 update (answers only), 3 seeds, 2026-11-06 | 13.5-14.7 s (training 2.34-2.55 h); validation pass 0.47-0.53 h |
| E2 update (distillation, split backward), seed 0, 2026-11-08 | **27.2 s** (training 4.72 h); validation pass 0.53 h; peak 12.66 GB |

Every distillation step below (E2, E4, the recoveries, RC-L1) uses E2's 27.2 s, not 13.6 s
(recomputed 2026-11-08; the design is unchanged). The recoveries distil from the same cache,
so 27.2 s is the estimate; a pruned student may be somewhat faster, which is not measured.

## 2. Hours per run (estimates)

| Run | Known part | Unknown part |
|---|---|---|
| RC-TT1, our teacher (RC-TT0, 2026-10-19) | 38,000 examples x 1.39 s = **14.7 h** on T4 x2, counted once: 14.7 GPU hours (2,375 updates) | model load, saves, the validation pass |
| E1 training, one seed | **2.34-2.55 h** measured (625 updates) | |
| E2 training, one seed | 625 updates x 27.2 s = **4.72 h** (measured on seed 0); with the validation pass **about 5.2 h** | model load, saves |
| Recovery of one pruned run | 157 updates x 27.2 s = **1.2 h** (was 0.6 h at 13.6 s; rounding of 156.25 still open) | |
| Clinical scoring, proposed mix (E7, and as the code stands A1, A2, A5) | 108 units x 640 probes x 0.414 s = **7.9 h** | the method (D07 section 1) and the probe count are not frozen |
| A4 at n32 / n64 / n256 | **2.0 / 4.0 / 15.9 h** (probes assumed to scale with the calibration images) | |
| A6 layers only / KV and channel groups | **2.6 / 5.3 h** | |
| E5, E6, A3 scoring | minutes (not measured) | |
| GPTQ, one variant (E3 w4, w8; E8 w4) | **under 1 h** (card estimate from RC-Q0) | |
| E4 training, one seed | 1.5 to 3 x E2's 4.72 h = **7.1 to 14.1 h** (card estimate; was 3.5 to 7.1 h) | E4's memory at 1 x 16 |
| E9 (the 7B at 4 bit) | **1 to 2 h** (card estimate) | |
| RC-L1: E2@30k / E7@30k recovery / E7@30k scoring | 1,875 and 469 updates x 27.2 s: **14.2 / 3.5 / 7.9 h** (was 7.1 / 1.8 / 7.9 h) | the cache for 20,000 more answers: 2 C |
| Teacher cache, 10,000 answers (RC-E0b) | | **C**: unknown until RC-E0b times its first 100 answers (they are written to `results/E0/nobudget/seed0/metrics.csv` at once) |
| Gate 1 and E0 validation (RC-E0a) | Gate 1: at most about 2,240 scored probes x 0.584 s = **0.36 h**; the validation: the same probes (0.36 h) plus 1,000 free-text answers, **about 1.3 h** one at a time (about 12 tokens per answer, an assumption; batched, not measured); about **2 h** in all plus setup (2026-10-31, from the frozen counts) | the setup; the batched generation speed |
| One evaluation pass of the student, validation or test | | **V**, **T**: probe rows x conditions x 0.414 s, plus 1,000 free-text answers. One answer at a time runs at about 5 tokens per second; the speed of the batched answers of stage 06 is not measured. RC-E1's first validation step gives the first real V |
| One budget search (each pruned run) | | **S**: several footprint measurements, not measured |

Known total: see section 7 (recomputed 2026-11-08 for the seed plan of pre-registration
section 11 item 12: 3 seeds for the main tier only, seed 0 for every other arm): about
**234-241 GPU hours** for the remaining plan (recomputed again on 2026-11-08 for the pre-freeze
additions: A1 and A2 re-weight E7's effects, A1 at B2 gets seeds 1 and 2, E10 is added),
estimates, plus C, 2 C, G and the budget searches S. (Before the seed plan, with 3 seeds at B2 for every arm: about 329-351 h; with 3 seeds at
every budget and 13.6 s per update: 285-297 h.) A1 and A2 no longer score again what E7 scored: they
re-weight E7's per-finding effects of the same seed (implemented 2026-11-08), which saves about
15.8 h of scoring at seed 0.

## 3. Dependency chain

```
RC-D1 pass 1 (shared data) ──> label values (agent, team) ──> RC-D1 pass 2 (counts)
   │                                                                 │
   ├──> RC-E1 training, 3 seeds (needs only the shared data)         ├──> finding list, max_new_tokens
   │        └──> RC-E1 validation (needs the finding list) <──────────┤
   │                                                                 └──> RC-E0a Gate 1
   └──> RC-X1 part A (CPU)                                                   │
                                                                     RC-E0b teacher cache (E0 val in RC-E0a)
                                                                             │ (hand-off: teacher cache)
                          RC-E2 seed 0 (done 2026-11-08) <───────────────────┤
                               │ (hand-off: e2-seed0-adapter)                └──> RC-E4 seed 0 (cache only)
                          RC-B0 (E2 seed 0) ──> B1-B3 ──> FREEZE (all three approve)
                                                               │
         ┌───────────────┬───────────────────┬────────────────┼─────────────────┐
       RC-E3 seed 0    RC-E9            RC-G2 (B2, seed 0)   RC-A1..A6 seed 0 (after E7 B2)
                                             │
                     choose_comparator.py (CPU): comparator + futility rule
                                             │
               seed 0 of the rest: E5/E6/E7 at B1, B3; E8 seed 0 ──> exploratory
                                             │ (no futility stop)
               main tier seeds 1, 2: RC-E2, then E7 and the comparator at B2
                                             │
                         RC-R1 (E0, E1, E2 test) ──> RC-C1 (CPU) ──> paper
                         RC-L1 (code first; after E7 B2 seed 0, RC-B0, freeze)
```

Each run of seed k reads the E2 adapter of seed k (checked in the code: scoring, removal,
recovery, evaluation, E3 and E8). Seeds 1 and 2 exist only for the main tier (E2, E7 and the
comparator at B2) and only after seed 0's futility check: the run cards refuse them before
`results/seed_plan/comparator_B2.json` is committed, or when it says stop. E4 starts from the pretrained student and reads only the
teacher cache. The budget reference is always E2 seed 0 (RC-B0), for every seed, through git.

## 4. Week by week

Weeks are counted from the week RC-D1 runs. Since 2026-10-18 (D01 and D02 amendments) the
teacher has its own LoRA (RC-TT0, RC-TT1, at most 15 GPU hours by rule 3) before Gate 1. The
image-disjoint split also makes validation and test about 15 % of the images each, so V and T are
larger than the old split's 39 / 30 images would have given; they stay unknown until RC-D1 run 4. The **known** GPU hours of each person per week
are given against the 30 h each person has. The unknown parts are named next to them.

| Week | Soheil | Farzan | Amir | Known GPU h (S / F / A) |
|---|---|---|---|---|
| 1 | RC-D1-embed done (2026-10-19, T4, about 2.5 minutes of embedding); labelling 298 pairs by two raters (done 2026-10-22; no third member, D02 amendment 4), no top-up was needed; RC-D1-label's scoring (done); RC-D1 run 4 (done, not accepted, D02 amendment 5); the rule sweep (done, stopped, D02 amendment 6), the held-out filter sweep, then run 4b = pass 1 (CPU, D02 amendments 5 and 6); share the dataset; RC-D1 pass 2 (CPU) once the label values are confirmed. RC-TT0 is done | account setup; RC-X1 part A (CPU) once the shared data is shared | account setup; attach and verify the shared data | 0 / 0 / 0 |
| 2 | RC-TT1, our teacher's training: **done 2026-10-31** (two sessions, 15.17 h of training measured); then RC-E0a (Gate 1 and the E0 validation, about 2 h plus setup, an estimate) and RC-E0b (the cache, C unknown; `--stop-after-hours 9` this week, the rest next) with our adapter (T4 x2); RC-E1 seed 0 | RC-E1 seed 1 | RC-E1 seed 2 | 14.7 + 2.4 + G + C + 2 V / 2.4 + V / 2.4 + V |
| 3 | RC-E2 seed 0 (done 2026-11-08); RC-B0 (minutes, E2 seed 0 hand-off). Then B1-B3 and the **freeze** | — | — | 4.72 + 0.53 (measured) / 0 / 0 |

From week 4 on, by phase (seed plan of 2026-11-08; Farzan's assignments removed; **[team]** =
no owner yet). The hours are in section 7.

| Phase | Soheil | Amir | **[team]** (no owner yet) |
|---|---|---|---|
| A. after the freeze | RC-G2 (the five B2 runs of seed 0: E5 x 3, E6, E7); then `scripts/choose_comparator.py` (CPU): the comparator and the futility decision | | |
| B. seed 0 of the rest (exploratory) | RC-E3 seed 0 (`w8`, `w4`, `w8a8`); RC-E4 seed 0; RC-E8 seed 0 at B2; the E6 trend (B1, B3); RC-A1..A6 seed 0 (proposal of the cards) | the E5 trend (B1, B3) | RC-E9; the E7 and E8 trend (B1, B3); RC-E10 (E10, the native 2B) |
| C. main tier seeds 1 and 2 (only without a futility stop) | | seed 2: RC-E2, then E7 and the comparator at B2, and A1 at B2 (secondary) | seed 1: RC-E2, then E7 and the comparator at B2, and A1 at B2 (secondary) |
| D. tests and the rest | RC-R1; E0's validation again (RC-E0a section 3c); RC-L1 E2@30k and the cache extension; RC-C1 (CPU) | RC-L1 E7@30k; RC-T1 on the rented GPU (not Kaggle) | |

Week 2 held RC-TT1 (15.17 h measured) and E1 seed 0 (2.34 h measured). E2 seeds 1 and 2 are no
longer in week 3: under the futility rule they run in phase C, after seed 0's check.

## 5. What runs in parallel

- Week 2: the three E1 seeds run while Soheil runs Gate 1 and the cache. E1 needs only the
  shared data.
- Week 3: E2 seed 0 (done). Seeds 1 and 2 wait for the futility check (phase C).
- After the freeze: RC-G2 first (it decides the comparator and the futility rule); then the
  seed-0 runs of phase B on every account at once; then, without a futility stop, the main
  tier's seeds 1 and 2 on two accounts at once.

## 6. Critical path

RC-D1-embed -> the pair labels (two raters) -> RC-D1-label -> RC-D1 pass 1 (run 4) -> label values ->
RC-D1 pass 2 -> finding list ->
RC-E0a (Gate 1, E0 validation) -> RC-E0b (the teacher cache, **C unknown**) -> RC-E2 seed 0 -> RC-B0 -> B1-B3
and the freeze -> RC-G2 -> `choose_comparator.py` (the futility rule) -> E2, E7 and the
comparator at seeds 1 and 2 -> RC-R1 -> RC-C1. The seed-0 runs of phase B, the ablations and
RC-L1 are needed for the paper but not for the confirmatory comparisons (H1-H3).

Things that change this schedule when they become known:
- C (RC-E0b's first 100 answers);
- V (RC-E1's first validation step);
- the session time limit (Kaggle; the T4 x2 quota rule is settled: counts once);
- the D07 scoring method;
- the A1/A2 reuse decision.

The agent updates this file when each of them arrives.

## 7. Compression runs and the main tier across the accounts (recomputed 2026-11-08)

**Seeds** (pre-registration section 11 item 12, amendment of 2026-11-08, Soheil and Amir;
replaces item 11): 3 seeds (0, 1, 2) for the **main-claim tier** only, E2, E7 at B2 and the
comparator at B2; **seed 0 only** for every other arm (exploratory). Seeds 1 and 2 of the main
tier run only if E7 at B2 does not rank below the comparator on validation, seed 0 (the
futility rule, read by the run cards from `results/seed_plan/comparator_B2.json`). Never 2
seeds.

**Split** (Farzan's assignments removed, 2026-11-08): Soheil keeps seed 0 at B2 and without a
budget, Amir seed 2 of the main tier and the E5 trend; seed 1 of the main tier, RC-E9 and the
E7 / E8 trend have **no owner yet [team]**. The seed-0 trend runs read Soheil's
`e2-seed0-adapter` (a cross-account hand-off) and seed 0's committed pruning scores
(`unit_scores.csv` and its settings file; commit them before the trend runs, or the scoring
runs again).

**Before any of them can run:** the freeze (`preregistration.frozen_on`); RC-B0 and B1-B3;
`budgets.measurement.hardware` (Amir); `recovery.steps` and `recovery.data_subset`; D07 section
1 (`pruning.clinical.ablation_method`); E2 seed 0's adapter (done). Seeds 1 and 2 also need the
committed comparator choice and their own E2 adapters (phase C).

| Account | Runs | Card commands (`scripts/run_card.py`, inputs found by their manifests) | Attach | Hands back |
|---|---|---|---|---|
| Soheil | RC-B0; RC-G2; seed 0 at B2 and without a budget: E3 (3 variants), E4, E8; the E6 trend (B1, B3); RC-A1..A6 seed 0; E0's validation again (RC-E0a section 3c); RC-R1; RC-C1; RC-L1 E2@30k | `--card RC-G2`; then `python scripts/choose_comparator.py` (CPU); `--card RC-E3 --seed 0`; `--card RC-E4 --seed 0`; `--card RC-E8 --seed 0 --budget B2`; `--card RC-E6 --seed 0 --budget B1 --budget B3`; `--card RC-A<n> --seed 0` | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache`, `g2-b2-seed0-pruned`, `e7-seed0-pruned` (E8); for RC-R1 `e0-teacher-adapter` and the E1 / E2 adapters; for RC-C1 every `*_outcomes` | results (git), including seed 0's `unit_scores.csv` and `results/seed_plan/comparator_B2.json`; `e2-seed0-adapter` to the other accounts |
| Amir | the E5 trend of seed 0 (B1, B3); main tier **seed 2**: E2, then E7 and the comparator at B2; RC-L1 E7@30k | `--card RC-E5 --seed 0 --budget B1 --budget B3`; `--card RC-E2 --seed 2`; `--card RC-E7 --seed 2`; `--card RC-E5 --seed 2` or `--card RC-E6 --seed 2` (whichever holds the comparator) | `medcompress-data`, `e2-seed0-adapter` (trend), `e2-seed2-adapter`, `e0-teacher-cache` | results (git); the `*_outcomes` of seed 2 and of the trend runs, and `e2-seed2-adapter`, to Soheil |
| **[team]** (no owner) | main tier **seed 1** (as seed 2); RC-E9; the E7 and E8 trend of seed 0 (B1, B3) | `--card RC-E2 --seed 1`; `--card RC-E7 --seed 1`; the comparator's card `--seed 1`; `--card RC-E9`; `--card RC-E7 --seed 0 --budget B1 --budget B3`; `--card RC-E8 --seed 0 --budget B1 --budget B3` | as seed 2, with `e2-seed1-adapter`; `e0-teacher-adapter` (E9); `e7-seed0-pruned` (E8 trend) | results (git); the outcomes and `e2-seed1-adapter` to Soheil |

Make every folder under `/kaggle/working/handoff/` ONE private dataset named like the folder
(`e7-seed1-pruned`), shared with the team.

**Hours** (all estimates). One evaluation of a run on validation, the test and the unfiltered
test is about **1.7 h**: the validation pass of E1 took 0.47-0.53 h and E2's 0.53 h (measured),
the test is the same size, the unfiltered test about 1.4 times the probes (an assumption). A
recovery is 157 updates x 27.2 s = **1.2 h** (the distillation loop's measured step on E2,
2026-11-08; a pruned student may be faster, not measured). E2 per seed: **5.2 h** (4.72 h
training + 0.53 h validation, measured on seed 0).

| Block | Runs | Known part | Evaluation | In all |
|---|---|---|---|---|
| Seed 0 at B2 and without a budget: E3 (3), E4 (1), E5 (3), E6 (1), E7 (1), E8 (1); RC-G2's five runs are these B2 runs | 10 | E3 3 x under 1 h; E4 7.1-14.1 h; recoveries 5 x 1.2 h; E7 scoring 7.9 h; E8 under 1 h | 10 x 1.7 h | **41.9-49.0 h** + 5 S |
| Seed 0's trend at B1 and B3: E5 (6), E6 (2), E7 (2), E8 (2) | 12 | 10 x 1.2 h recovery; E8 2 x under 1 h | 12 x 1.7 h | **34.2 h** + 10 S (E5 17.3, E6 5.8, E7 5.8, E8 5.4) |
| Main tier, seeds 1 and 2 (not run after a futility stop) | 2 x 3 | per seed: E2 5.2 h; E7 7.9 h scoring + 1.2 h recovery; the comparator 1.2 h recovery (E5 / E6 scoring: minutes) | per seed 2 x 1.7 h | **37.8 h** (18.9 per seed: E2 5.2, E7 10.8, comparator 2.9) + 4 S |
| A1 at B2, seeds 1 and 2 (secondary; under the futility rule) | 2 | per seed 1.2 h recovery (E7's effects re-weighted, no scoring) | 2 x 1.7 h | **5.8 h** + 2 S |
| Ablations, seed 0 (A1, A2, A3, A4 x 3, A5, A6 x 2; `A4:n128` and `A5:logprob_drop` are E7) | 9 | 48.3 h (37.7 h scoring: A1 and A2 re-weight E7's effects; + 9 x 1.2 h recovery) | 9 x 1.7 h | **63.7 h** + 9 S (was 79.5 h with A1 and A2 scoring) |
| RC-E10 (seed 0) | 1 (+ its 4-bit run) | about 4 h (the card: training on the 2B, not measured, and its validation); GPTQ `w4` under 1 h | 4-bit validation 1.7 h; the tests about 1.2 h | about **7.9 h** |
| RC-L1 (seed 0) | 2 | E2@30k 14.2 h; E7@30k recovery 3.5 h and scoring 7.9 h | 2 x 1.7 h | **29.0 h** + 2 C + S |
| RC-R1 | 7 test pairs (E2 seeds 1, 2 only without a futility stop) | | E0 about 2 h (T4 x2), six students about 1.2 h each | about **9 h** |
| E0's validation again | 1 | | 0.8 h (measured first pass) | **0.8 h** |
| RC-E9 | 1 | 1-2 h | the teacher's two tests, about 2 h | about **3.5 h** |

| Account | Load (known part, estimates) |
|---|---|
| Soheil | 41.9-49.0 (seed 0 at B2 and no budget) + 5.8 (E6 trend) + 63.7 (ablations) + 9 (RC-R1) + 0.8 (E0) + 15.9 (RC-L1 E2@30k) = **about 137.1-144.2 h** (plus 2 C) |
| Amir | 17.3 (E5 trend) + 18.9 (main tier seed 2) + 2.9 (A1 seed 2) + 13.1 (RC-L1 E7@30k) = **about 52.2 h** |
| **[team]** (no owner) | 18.9 (main tier seed 1) + 2.9 (A1 seed 1) + 3.5 (RC-E9) + 11.2 (E7, E8 trend) + 7.9 (RC-E10) = **about 44.4 h** |

In all about **234-241 GPU h** plus the budget searches S and 2 C (not measured): about 190-197
h of seed-0 work and 43.6 h for seeds 1 and 2 of the main tier and A1, which a futility stop
saves.
Against about 360 h or more with 3 seeds for every arm at B2. (Soheil's summary figure for the
whole plan was about 190 h; the itemised total is higher mainly by the ablations' evaluation
passes, 15.3 h, RC-L1, 29.0 h, and now E10, 7.9 h.) The loads are uneven as the owners stand
(Soheil about 137-144 h, between four and five weeks at 30 h per week): **[team]**, how to
share the ablations and the unowned runs.
