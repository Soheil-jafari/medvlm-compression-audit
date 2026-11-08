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

Known total (recomputed 2026-11-08, the current design: 3 seeds at B2, seed 0 only at B1 and
B3): about **329 to 351 GPU hours**, not counting RC-TT1 and RC-E0a/E0b (done). It is E1 7.4 h
(measured) + E2 14.2 h (3 x 4.72 h) + E3 9 x under 1 h + E4 3 x 7.1-14.1 h + E9 1-2 h + 25
recoveries x 1.2 h (29.6 h; 15 at B2, 10 in seed 0's trend) + E7 scoring 3 x 7.9 h + E8 5 x
under 1 h + the ablations 192.5 h + RC-L1 25.6 h. On top come C, 2 C, G, the student
evaluation passes (V, T), 4 teacher passes and the budget searches. The ablations are about
**193 h** of the known total (3 seeds x 64.2 h: 53.5 h of scoring and 9 recoveries x 1.2 h;
was 177 h at 0.6 h per recovery). About 48 h of that is A1 and A2 scoring again what E7 scored;
the team decides whether to add the reuse D07 describes (decisions log). The total written
here before (285 to 297 h) had 3 seeds at every budget and 13.6 s per update.

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
                          RC-E2 seed 0 ─ RC-E2 seed 1 ─ RC-E2 seed 2 <───────┤
                               │ (hand-off: E2 seed k adapter)               └──> RC-E4 (cache only)
                          RC-B0 (E2 seed 0) ──> B1-B3 ──> FREEZE (all three approve)
                                                               │
         ┌───────────────┬───────────────────┬────────────────┼─────────────────┐
       RC-E3 (E2 k)    RC-E9            RC-G2 (B2, seed 0)   RC-R1 (E0, E1, E2 test)
                                             │
                         RC-E5, RC-E6, RC-E7 by seed (G2's runs reused at seed 0)
                                             │ (hand-off: E7 pruned, seed k)
                                   RC-E8 by seed ──> RC-C1 (CPU) ──> paper
                         RC-A1..A6 by seed (after E7 B2) ──> appendix, paper
                         RC-L1 (code first; after E7 B2 seed 0, RC-B0, freeze)
```

Each run of seed k reads the E2 adapter of seed k (checked in the code: scoring, removal,
recovery, evaluation, E3 and E8). E4 starts from the pretrained student and reads only the
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
| 3 | RC-E2 seed 0 (done 2026-11-08) | RC-E2 seed 1 | RC-E2 seed 2; RC-B0 (minutes, E2 seed 0 hand-off). Then B1-B3 and the **freeze** | 4.72 + 0.53 (measured) / 4.72 + V / 4.72 + V |
| 4 | RC-E3 seed 0; RC-E4 seed 0; RC-R1 | RC-E9 (T4 x2) | RC-G2 (5 runs) | 9.1-16.1 + 4 V + 11 T / 1-2 + V + T (teacher) / 13.8 + 5 V + 5 S |
| 5 | RC-E3 seeds 1, 2; RC-E4 seeds 1, 2 | RC-E5, E6, E7 seed 1 (15 runs) | RC-E5, E6, E7 seed 0 (10 runs; G2's 5 reused) | 18.1-32.3 + 8 V + 8 T / 25.7 + 15 V + 15 T + 15 S / 11.8 + 10 V + 15 T + 10 S |
| 6 | RC-A1..A6 seed 0 (start) | RC-E8 seed 1; RC-A1..A6 seed 1 (start) | RC-E5, E6, E7 seed 2 (15 runs); RC-E8 seed 0 | part of 64.2 / 3 + part of 64.2 / 28.7 + 18 V + 18 T + 15 S |
| 7 | RC-A1..A6 seed 0 | RC-A1..A6 seed 1 | RC-E8 seed 2; RC-A1..A6 seed 2 (start) | each ablation seed: 64.2 + 9 V + 9 T + 9 S over weeks 6 to 8 |
| 8 | RC-A1..A6 seed 0 (end); RC-C1 (CPU) once E8 seed 2 is in | RC-A1..A6 seed 1 (end) | RC-A1..A6 seed 2 | as above |
| 9 | RC-L1: cache extension (T4 x2) and E2@30k (code first) | spare | RC-L1: E7@30k; RC-T1 on the rented GPU (not Kaggle) | 14.2 + 2 C + 2 V + 2 T / 0 / 11.4 + S + 2 V + 2 T |

Weeks 4 to 8 still show the plan of 2026-10-17 (3 seeds at every budget); section 7 is the
current proposal for E3-E8 and its hours. The hours in the table were recomputed on 2026-11-08
with E2's 27.2 s per update (E2, E4, recoveries, RC-L1).

Known totals per person, all weeks, as the table stands (without RC-TT1): Soheil about 113 to
134 h (plus G, C, 2 C), Farzan about 101 to 102 h, Amir about 137 h, before the evaluation
passes (were 87 to 98, 85 and 113 h at 13.6 s per update). Weeks 5 to 8 are the tight ones: if
V and T turn out to be an hour or more each, the grid and the ablations each stretch by about a
week. Soheil's week 2 held RC-TT1 (15.17 h measured) and E1 seed 0 (2.34 h measured).

## 5. What runs in parallel

- Week 2: the three E1 seeds run while Soheil runs Gate 1 and the cache. E1 needs only the
  shared data.
- Week 3: the three E2 seeds run, one per account, as soon as the cache hand-off is shared.
- After the freeze: quantization (Soheil), E9 (Farzan) and RC-G2 (Amir) run at once. Then the
  three grid seeds run on three accounts, and then the three ablation seeds.

## 6. Critical path

RC-D1-embed -> the pair labels (two raters) -> RC-D1-label -> RC-D1 pass 1 (run 4) -> label values ->
RC-D1 pass 2 -> finding list ->
RC-E0a (Gate 1, E0 validation) -> RC-E0b (the teacher cache, **C unknown**) -> RC-E2 seed 0 -> RC-B0 -> B1-B3
and the freeze -> RC-G2 -> the E7 grid of each seed -> RC-E8 -> RC-C1. The ablations and RC-L1 are needed for the paper but
not for the comparisons (H1-H4).

Things that change this schedule when they become known:
- C (RC-E0b's first 100 answers);
- V (RC-E1's first validation step);
- the session time limit (Kaggle; the T4 x2 quota rule is settled: counts once);
- the D07 scoring method;
- the A1/A2 reuse decision.

The agent updates this file when each of them arrives.

## 7. Compression runs E3-E8 across the three accounts (proposal; recomputed 2026-11-06)

**Seeds** (pre-registration section 11 item 11, before the freeze): 3 seeds at the primary
budget B2 for every arm (and E3, E4 without a budget); **seed 0 only at B1 and B3**, the trend.
So a seed-1 or seed-2 session of RC-E5 to RC-E8 runs B2 only (the plan does it by itself).

**Split** (proposal; the team confirms): each account runs every B2 arm of the seed whose E2 it
trained (seed 0 Soheil, 1 Farzan, 2 Amir). Seed 0's trend runs (B1, B3) are shared out so that
the three loads are even: they read Soheil's `e2-seed0-adapter` (a cross-account hand-off) and
seed 0's committed pruning scores (`unit_scores.csv` and its settings file; commit them before
the trend runs, or the scoring runs again).

**Before any of them can run:** the freeze (`preregistration.frozen_on`); RC-B0 and B1-B3;
`budgets.measurement.hardware` (Amir); `recovery.steps` and `recovery.data_subset`; D07 section
1 (`pruning.clinical.ablation_method`); the three E2 adapters.

| Account | Runs | Card commands (`scripts/run_card.py`, inputs found by their manifests) | Attach | Hands back |
|---|---|---|---|---|
| Soheil | RC-B0; RC-G2; seed 0 at B2 and without a budget: E3, E4, E5, E6, E7, E8; the E6 trend (B1, B3); E0's validation again (RC-E0a section 3c); RC-R1; RC-C1 | `--card RC-E3 --seed 0`; `--card RC-E4 --seed 0`; `--card RC-E5 --seed 0 --budget B2`; `--card RC-E6 --seed 0`; `--card RC-E7 --seed 0 --budget B2`; `--card RC-E8 --seed 0 --budget B2` | `medcompress-data`, `e2-seed0-adapter`, `e0-teacher-cache`, `g2-b2-seed0-pruned` (optional), `e7-seed0-pruned` (E8); for RC-R1 `e0-teacher-adapter` and the six E1/E2 adapters; for RC-C1 every `*_outcomes` | results (git), including seed 0's `unit_scores.csv`; `e2-seed0-adapter` to Amir and Farzan |
| Farzan | seed 1 at B2 and without a budget: E3, E4, E5, E6, E7, E8; RC-E9; the E7 and E8 trend of seed 0 (B1, B3) | `--card RC-E3 --seed 1` ... `--card RC-E8 --seed 1`; `--card RC-E9`; `--card RC-E7 --seed 0 --budget B1 --budget B3`; `--card RC-E8 --seed 0 --budget B1 --budget B3` | `medcompress-data`, `e2-seed1-adapter`, `e2-seed0-adapter` (trend), `e0-teacher-cache`, `e7-seed1-pruned`, the trend's own `e7-seed0-pruned` (E8 trend), `e0-teacher-adapter` (E9) | results (git); the `*_outcomes` of seed 1 and of the trend runs, and `e1-`/`e2-seed1-adapter`, to Soheil |
| Amir | seed 2 at B2 and without a budget: E3, E4, E5, E6, E7, E8; the E5 trend of seed 0 (B1, B3) | `--card RC-E3 --seed 2` ... `--card RC-E8 --seed 2`; `--card RC-E5 --seed 0 --budget B1 --budget B3` | `medcompress-data`, `e2-seed2-adapter`, `e2-seed0-adapter` (trend), `e0-teacher-cache`, `e7-seed2-pruned` | results (git); the `*_outcomes` of seed 2 and of the trend runs, and `e1-`/`e2-seed2-adapter`, to Soheil |

Make every folder under `/kaggle/working/handoff/` ONE private dataset named like the folder
(`e7-seed1-pruned`), shared with the team.

**Hours** (estimates). One evaluation of a run on validation, the test and the unfiltered test
is about **1.7 h**: the validation pass of E1 took 0.47-0.53 h and E2's 0.53 h (measured), the
test is the same size, the unfiltered test about 1.4 times the probes (an assumption). A
recovery is 157 updates x 27.2 s = **1.2 h** (E2's measured step, 2026-11-08; was 0.6 h).

| Block | Runs | Known part | Evaluation | In all |
|---|---|---|---|---|
| One seed at B2 and without a budget: E3 (3), E4 (1), E5 (3), E6 (1), E7 (1), E8 (1) | 10 | E3 3 x under 1 h; E4 7.1-14.1 h; recoveries 5 x 1.2 h; E7 scoring 7.9 h; E8 under 1 h | 10 x 1.7 h | **41.9-49.0 h** + 5 S |
| Seed 0's trend at B1 and B3: E5 (6), E6 (2), E7 (2), E8 (2) | 12 | 10 x 1.2 h recovery; E8 2 x under 1 h | 12 x 1.7 h | **34.3 h** + 10 S (E5 17.3, E6 5.8, E7 5.8, E8 5.4) |
| RC-R1 | 7 test pairs | | E0 about 2 h (T4 x2), six students about 1.2 h each | about **9 h** |
| E0's validation again | 1 | | 0.8 h (measured first pass) | **0.8 h** |
| RC-E9 | 1 | 1-2 h | the teacher's two tests, about 2 h | about **3.5 h** |

| Account | Load (known part, estimates) |
|---|---|
| Soheil | 41.9-49.0 (seed 0) + 5.8 (E6 trend) + 9 (RC-R1) + 0.8 (E0) = **about 57.5-64.6 h** |
| Farzan | 41.9-49.0 (seed 1) + 3.5 (E9) + 11.2 (E7, E8 trend) = **about 56.6-63.7 h** |
| Amir | 41.9-49.0 (seed 2) + 17.3 (E5 trend) = **about 59.2-66.3 h** |

In all about **173-194 GPU h** plus the budget searches S (not measured), against about 242-263
h with 3 seeds at every budget: about 68 h saved (recomputed 2026-11-08 with 27.2 s per update;
it was 148-159 h at 13.6 s). At 30 h per person per week that is about two to two and a quarter
weeks of each account. Seed 0's B2 runs of RC-G2 (five) are reused in its grid.
