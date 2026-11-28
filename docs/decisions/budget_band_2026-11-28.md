# Note of 2026-11-28 (research lead): stage-07 benchmark peaks just above target x 1.03

**Status:** recorded 2026-11-28, before any accuracy result of G4 (E5 activation B3), G6 (E5
Taylor B3) or G10 (E7 B3) is read, and before E5 magnitude B3 (G2) and the A-series run. Memory
numbers only. No setting, rule or analysis of the pre-registration changes.

## What the pre-registration says

- Section 4: "Metric | peak GPU memory during inference (weights, activations, KV cache)";
  "Tolerance | 3 % of the target, either side".
- D07 (section 4): "The compared quantity is the **measured** footprint after removal, never a
  parameter count."
- Amendment of 2026-11-17, item 2: "a search over k (0-85) keeps the smallest k whose measured
  footprint is at or below target x 1.03 (the band of +-3 % and the selection rule of
  2026-11-16, unchanged)".
- Decisions log, 2026-11-16 (the selection rule, stated): "The search keeps the smallest removal
  whose measured footprint is at or below target x 1.03 ... the results report each arm's
  measured footprint and its gap to the target (`paper/tables/budget_match.tex`)".
- Section 6, exclusions: "a budget-matched run outside the 3 % tolerance is reported, marked, and
  the primary comparison is repeated without such runs as a sensitivity analysis".

The band is applied to the footprint the budget search measures for the chosen removal (stage
05); the stage-07 benchmark is not named for it. `budget_match.tex` has computed the gap and
"within tolerance" from the search footprint since 2026-11-16, with the benchmark peak beside it.
On that measure every pruning arm so far is inside the band, so the sensitivity clause is not
triggered.

## What was found

The stage-07 benchmark peak (the recovered model, loaded from its saved checkpoint) is above
target x 1.03 for three arms, by at most 653,683 bytes (0.0124 %): E7 B1 (G9), E5 activation B3
(G4) and E5 Taylor B3 (G6). E5 magnitude B3 (G2, not run yet) is expected to remove the same 1485
units and match G4 and G6 (an expectation, not a result).

Both numbers come from the same function and protocol (`evaluation.efficiency.measure_footprint`,
the same 32 images, the same subtraction of GPU bytes that are not the model's), on different
model states: the search measures E2 with the units removed in-process, before recovery; stage 07
measures the recovered model loaded from its checkpoint. The difference (benchmark minus search)
depends on the removal set, from -1,896,448 to +1,844,224 bytes (-1.9 to +1.8 MB) over the runs so
far, and is identical for identical removal sets (every 957-unit B2 arm +355,840; both 396-unit B1
arms -292,352; both 1485-unit B3 arms +1,844,224). Its cause beyond the model state is not
established. The unpruned model's level check is -0.08 % in every run.

| Run | Arm | Budget | Units | Search footprint | Benchmark peak | Benchmark - search | Upper bound (target x 1.03) | Benchmark - bound |
|---|---|---|---|---|---|---|---|---|
| G1 | E5 magnitude | B1 | 396 | 6,987,762,848 | 6,987,470,496 | -292,352 | 6,989,621,635 | -2,151,139 |
| G3 | E5 activation | B1 | 396 | 6,987,762,848 | 6,987,470,496 | -292,352 | 6,989,621,635 | -2,151,139 |
| G9 | E7 | B1 | 371 | 6,989,309,600 | 6,990,113,952 | +804,352 | 6,989,621,635 | **+492,317 (+0.0070 %)** |
| RC-G2 | E5 magnitude | B2 | 957 | 6,105,386,144 | 6,105,741,984 | +355,840 | 6,132,217,176 | -26,475,192 |
| RC-G2 | E5 activation | B2 | 957 | 6,105,386,144 | 6,105,741,984 | +355,840 | 6,132,217,176 | -26,475,192 |
| RC-G2 | E5 Taylor | B2 | 957 | 6,105,386,144 | 6,105,741,984 | +355,840 | 6,132,217,176 | -26,475,192 |
| RC-G2 | E5R | B2 | 957 | 6,105,386,144 | 6,105,741,984 | +355,840 | 6,132,217,176 | -26,475,192 |
| RC-G2 | E7U | B2 | 957 | 6,105,386,144 | 6,105,741,984 | +355,840 | 6,132,217,176 | -26,475,192 |
| RC-G2 | E6 | B2 | 10 | 6,063,998,624 | 6,063,672,992 | -325,632 | 6,132,217,176 | -68,544,184 |
| RC-G2 | E7 | B2 | 885 | 6,132,020,384 | 6,130,149,536 | -1,870,848 | 6,132,217,176 | -2,067,640 |
| G4 | E5 activation | B3 | 1485 | 5,273,622,176 | 5,275,466,400 | +1,844,224 | 5,274,812,717 | **+653,683 (+0.0124 %)** |
| G6 | E5 Taylor | B3 | 1485 | 5,273,622,176 | 5,275,466,400 | +1,844,224 | 5,274,812,717 | **+653,683 (+0.0124 %)** |
| G10 | E7 | B3 | 1067 | 5,216,048,800 | 5,215,492,768 | -556,032 | 5,274,812,717 | -59,319,949 |
| superseded | E5 activation (inner units removable) | B2 | 281 | 6,131,026,592 | 6,129,130,144 | -1,896,448 | 6,132,217,176 | -3,087,032 |
| superseded | E5 magnitude (inner units removable) | B2 | 220 | 6,041,150,112 | 6,041,026,208 | -123,904 | 6,132,217,176 | -91,190,968 |

Targets and bounds: E2's committed peak 7,618,471,936 minus 15 / 30 / 45 % of its decoder-layer
bytes 5,549,543,424 (`configs/budgets.yaml`), times 1.03. Every search footprint is at or below
its bound.

## Decision

1. The search rule stays unchanged for every remaining run (E5 magnitude B3 and the A-series),
   so the whole grid follows one rule. No rerun: one more unit per layer would break "the smallest
   removal at or below target x 1.03" on the search's measure (about 12 GPU h).
2. The three arms are kept. They are flagged, not marked as outside the tolerance: "benchmark
   peak above target x 1.03 (by at most 0.012 %, 0.65 MB); inside the band on the search
   footprint the band is defined on". The flag appears in `budget_match.tex` (a column), in the
   main results table (a dagger on the arm) and in the methods text. Section 6's sensitivity
   analysis is not run for them (they are within the tolerance on the registered measure); the
   primary B2 comparison is not affected either way (all three are B1 or B3).
3. If any arm's benchmark peak exceeds target x 1.03 by more than 0.1 %, the research lead is
   told before anything else is done with it. The aggregation prints a line for such an arm,
   and every results commit checks it.

**Correction (2026-11-28, same day; the decision is unchanged).** "Identical for identical removal
sets" is imprecise: the arms named there remove different units (different criteria) but the same
number of channel groups per layer (k = 29 at B2, 12 at B1, 45 at B3). The difference between the
two measures is identical for the same removal shape, whichever units are removed.

