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

## 2. Hours per run (estimates)

| Run | Known part | Unknown part |
|---|---|---|
| RC-TT1, our teacher (RC-TT0, 2026-10-19) | 38,000 examples x 1.39 s = **14.7 h** on T4 x2, counted once: 14.7 GPU hours (2,375 updates) | model load, saves, the validation pass |
| E1 or E2 training, one seed | 625 updates x 13.6 s = **2.4 h** | model load, saves |
| Recovery of one pruned run | 157 updates x 13.6 s = **0.6 h** (rounding of 156.25 still open) | |
| Clinical scoring, proposed mix (E7, and as the code stands A1, A2, A5) | 108 units x 640 probes x 0.414 s = **7.9 h** | the method (D07 section 1) and the probe count are not frozen |
| A4 at n32 / n64 / n256 | **2.0 / 4.0 / 15.9 h** (probes assumed to scale with the calibration images) | |
| A6 layers only / KV and channel groups | **2.6 / 5.3 h** | |
| E5, E6, A3 scoring | minutes (not measured) | |
| GPTQ, one variant (E3 w4, w8; E8 w4) | **under 1 h** (card estimate from RC-Q0) | |
| E4 training, one seed | 1.5 to 3 x E2 = **3.5 to 7.1 h** (card estimate) | E4's memory at 1 x 16 |
| E9 (the 7B at 4 bit) | **1 to 2 h** (card estimate) | |
| RC-L1: E2@30k / E7@30k recovery / E7@30k scoring | **7.1 / 1.8 / 7.9 h** | the cache for 20,000 more answers: 2 C |
| Teacher cache, 10,000 answers (RC-E0b) | | **C**: unknown until RC-E0b times its first 100 answers (they are written to `results/E0/nobudget/seed0/metrics.csv` at once) |
| Gate 1 (RC-E0a) | | **G**: teacher probes x 0.584 s x conditions, counts from RC-D1 |
| One evaluation pass of the student, validation or test | | **V**, **T**: probe rows x conditions x 0.414 s, plus 1,000 free-text answers. One answer at a time runs at about 5 tokens per second; the speed of the batched answers of stage 06 is not measured. RC-E1's first validation step gives the first real V |
| One budget search (each pruned run) | | **S**: several footprint measurements, not measured |

Known total: about **285 to 297 GPU hours**. On top come C, 2 C, G, about 200 student
evaluation passes (V, T), 4 teacher passes and about 70 budget searches. The ablations are
about **177 h** of the known total. About 48 h of that is A1 and A2 scoring again what E7 scored; the
team decides whether to add the reuse D07 describes (decisions log).

## 3. Dependency chain

```
RC-D1 pass 1 (shared data) ──> label values (agent, team) ──> RC-D1 pass 2 (counts)
   │                                                                 │
   ├──> RC-E1 training, 3 seeds (needs only the shared data)         ├──> finding list, max_new_tokens
   │        └──> RC-E1 validation (needs the finding list) <──────────┤
   │                                                                 └──> RC-E0a Gate 1
   └──> RC-X1 part A (CPU)                                                   │
                                                                     RC-E0b teacher cache + E0 val
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
| 1 | RC-D1-embed done (2026-10-19, T4, 143 s of embedding); labelling 298 pairs with a second rater (about 50 to 100 min each, an estimate), the third member's decisions, and a top-up of 100 pairs if fewer than 30 lie near the chosen threshold; RC-D1-label's scoring (CPU, minutes); RC-D1 run 4 = pass 1 (CPU, D02 amendment 3); share the dataset; RC-D1 pass 2 (CPU) once the label values are confirmed. RC-TT0 is done | account setup; RC-X1 part A (CPU) once the shared data is shared | account setup; attach and verify the shared data | 0 / 0 / 0 |
| 2 | RC-TT1, our teacher's training (14.7 h on T4 x2, counted once; more than one session if the session limit is shorter; RC-TT0 done 2026-10-19); then RC-E0a Gate 1 and RC-E0b cache with our adapter (T4 x2); RC-E1 seed 0 | RC-E1 seed 1 | RC-E1 seed 2 | 14.7 + 2.4 + G + C + 2 V / 2.4 + V / 2.4 + V |
| 3 | RC-E2 seed 0 | RC-E2 seed 1 | RC-E2 seed 2; RC-B0 (minutes, E2 seed 0 hand-off). Then B1-B3 and the **freeze** | 2.4 + V / 2.4 + V / 2.4 + V |
| 4 | RC-E3 seed 0; RC-E4 seed 0; RC-R1 | RC-E9 (T4 x2) | RC-G2 (5 runs) | 5.5-9.1 + 4 V + 11 T / 1-2 + V + T (teacher) / 10.9 + 5 V + 5 S |
| 5 | RC-E3 seeds 1, 2; RC-E4 seeds 1, 2 | RC-E5, E6, E7 seed 1 (15 runs) | RC-E5, E6, E7 seed 0 (10 runs; G2's 5 reused) | 11.1-18.2 + 8 V + 8 T / 16.8 + 15 V + 15 T + 15 S / 5.9 + 10 V + 15 T + 10 S |
| 6 | RC-A1..A6 seed 0 (start) | RC-E8 seed 1; RC-A1..A6 seed 1 (start) | RC-E5, E6, E7 seed 2 (15 runs); RC-E8 seed 0 | part of 59.0 / 3 + part of 59.0 / 19.8 + 18 V + 18 T + 15 S |
| 7 | RC-A1..A6 seed 0 | RC-A1..A6 seed 1 | RC-E8 seed 2; RC-A1..A6 seed 2 (start) | each ablation seed: 59.0 + 9 V + 9 T + 9 S over weeks 6 to 8 |
| 8 | RC-A1..A6 seed 0 (end); RC-C1 (CPU) once E8 seed 2 is in | RC-A1..A6 seed 1 (end) | RC-A1..A6 seed 2 | as above |
| 9 | RC-L1: cache extension (T4 x2) and E2@30k (code first) | spare | RC-L1: E7@30k; RC-T1 on the rented GPU (not Kaggle) | 7.1 + 2 C + 2 V + 2 T / 0 / 9.7 + S + 2 V + 2 T |

Known totals per person, all weeks: Soheil about 87 to 98 h (plus G, C, 2 C), Farzan about
85 h, Amir about 113 h, before the evaluation passes. Weeks 5 to 8 are the tight ones: if V
and T turn out to be an hour or more each, the grid and the ablations each stretch by about a
week. Soheil's week 2 holds RC-TT1 (about 14.7 h) and E1 seed 0 (2.4 h): if G, C and the two
validation passes need more than the remaining 12.9 h, it spills into week 3 and everything
after it moves with it.

## 5. What runs in parallel

- Week 2: the three E1 seeds run while Soheil runs Gate 1 and the cache. E1 needs only the
  shared data.
- Week 3: the three E2 seeds run, one per account, as soon as the cache hand-off is shared.
- After the freeze: quantization (Soheil), E9 (Farzan) and RC-G2 (Amir) run at once. Then the
  three grid seeds run on three accounts, and then the three ablation seeds.

## 6. Critical path

RC-D1-embed -> the pair labels (two raters) -> RC-D1-label -> RC-D1 pass 1 (run 4) -> label values ->
RC-D1 pass 2 -> finding list ->
RC-E0a (Gate 1) -> RC-E0b (the teacher cache, **C unknown**) -> RC-E2 seed 0 -> RC-B0 -> B1-B3
and the freeze -> RC-G2 -> the E7 grid of each seed -> RC-E8 -> RC-C1. The ablations and RC-L1 are needed for the paper but
not for the comparisons (H1-H4).

Things that change this schedule when they become known:
- C (RC-E0b's first 100 answers);
- V (RC-E1's first validation step);
- the session time limit (Kaggle; the T4 x2 quota rule is settled: counts once);
- the D07 scoring method;
- the A1/A2 reuse decision.

The agent updates this file when each of them arrives.
