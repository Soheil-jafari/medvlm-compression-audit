# Pruning design, locked 2026-11-17

**Status:** decided 2026-11-17 (research lead; agreed by Soheil and Amir). Pre-registration
section 10, amendment of 2026-11-17 (post hoc, disclosed). Keeps the protection rule of the
amendment of 2026-11-16 and supersedes the clarification of 2026-11-15.
**Depends on:** D07 (pruning settings), the seed plan (pre-registration section 11 item 12).

## The design

| Arm | Role | Units | Allocation | Scores | Budgets, seeds |
|---|---|---|---|---|---|
| E5 magnitude | comparator candidate | MLP channel groups of 128 | `uniform_mlp` | committed (session 1) | B1-B3; 3 seeds at B2 if chosen |
| E5 activation (Wanda-sp) | comparator candidate | MLP channel groups of 128 | `uniform_mlp` | committed (session 1) | as above |
| E5 Taylor | comparator candidate | MLP channel groups of 128 | `uniform_mlp` | session 2 | as above |
| E6 | comparator candidate | whole layers | `global` (one-shot output similarity) | session 2 | as above |
| E7 (ours) | main arm | layers, KV groups, channel groups | `global` | committed (session 3) | B1-B3; 3 seeds at B2 |
| E5R (Random) | exploratory | MLP channel groups of 128 | `uniform_mlp` | seeded random (`random`) | B2, seed 0 |
| E7U (E7-U) | exploratory | MLP channel groups of 128 | `uniform_mlp` | E7's committed | B2, seed 0 |

The A-series (A1-A6) stay ablations of E7 with the `global` allocation. E5R and E7U are not
comparator candidates.

**Protection, every arm.** No unit in decoder layers 0, 1 and 35 is ever removed: not the layer,
not a KV group, not an MLP channel group. Stage 05 refuses a removal set that touches one.

**`uniform_mlp`.** The same number k of MLP channel groups is removed from every unprotected layer
(2-34); within a layer, the k lowest-scoring groups go (ties by unit id). Each layer has 86 groups
of 128 channels (11,008 channels), so k runs from 0 to 85. One step of k removes 33 groups
(128 x 2,048 x 3 = 786,432 parameters each), about 52 MB in fp16.

**Budget.** The same rule as every arm: the smallest k whose measured footprint is at or below
target x 1.03 (the +-3 % band and the selection rule of 2026-11-16). The search bisects over k
with the same measurer and the same 1 % level check as the global search.

**E7U.** E7's scores per parameter, within each layer, lowest first; the same k per layer as the
search finds. In a layer every channel group has the same size, so dividing by the size changes
no order. Stage 04 of E7U only checks and reuses E7's committed file; it never scores and never
writes it (`pruning.scores_from: E7`).

**Methods text (paper).** "Magnitude, activation (Wanda-sp) and first-order Taylor importance of
128-channel MLP groups, removed in equal number from every unprotected layer (LLM-Pruner's
allocation)."

## Why

The figures from the literature below are as the research lead supplied them on 2026-11-17;
they were not re-checked against the papers in this session (`TODO: verify` page or table
numbers before they go into the paper). The one-line readings of each source (what it shows for
this study, why B3 sits at the edge of Huang et al.'s range, why SliceGPT, Sheared LLaMA and
Minitron were not chosen) are the agent's wording, to be checked by the team with the figures.

1. **Global rankings across layers are unreliable.** Scores of different layers are on different
   scales. Molchanov et al. (ICLR 2017) report a cross-layer rank correlation of 0.14 for raw
   Taylor scores against 0.73 after normalisation. FLAP (An et al., AAAI 2024) and ECoFLaP (Sung
   et al., ICLR 2024) report that global magnitude pruning fails on vision-language models. RC-G2
   session 1 showed the same: the global per-parameter ranking put 171 of 220 (magnitude) and 172
   of 281 (activation) removed units inside layers 0, 1 and 35, and validation op macro balanced
   accuracy fell to 0.598 and 0.572 (E2 0.836), with small-finding recall 0.
2. **The method E5 follows allocates per layer.** LLM-Pruner (Ma et al., NeurIPS 2023) protects
   the first 3 and the last layer and removes the same ratio from every other layer; its local
   allocation scored 56.69 against 56.49 for global.
3. **Protection matters.** Without it, Taylor pruning reached a perplexity of 3,631.7 in Kim et al.
   2024 (Shortened LLaMA). With LoRA recovery at 20 %, LLM-Pruner scored 61.8 against FLAP's 59.5
   there.
4. **A random baseline on the same units.** Blalock et al. (MLSys 2020) ask every pruning study
   to report random and magnitude baselines. A3 is random on E7's global design; E5R is random on
   E5's design, so E5's criteria can be read against chance on the same units and allocation.
5. **Criterion against allocation.** E7 differs from E5 in both its scores and its allocation.
   E7U keeps E7's scores and takes E5's allocation, so the two can be separated.
6. **E6 is unchanged.** Whole-layer removal by output similarity is one-shot. Gromov et al. 2024,
   ShortGPT and SLEB (ICML 2024) describe the limits of such one-shot layer removal; E6 is kept
   as pre-registered and these are stated as its limitation.
7. **Budgets stay where they are.** Huang et al. (structural pruning of LLaVA, InternVL and Bunny)
   report collapse beyond 45 %; B3 removes 45 % of the decoder-layer bytes, the edge of that range.
8. **Rare classes go first.** Hooker et al., Holste et al. (MICCAI 2023) and Tran et al.
   (NeurIPS 2022) report that pruning forgets rare classes first. This is why the primary
   endpoints are small-finding recall and the lesion-mask effect, and why a design that empties
   whole layers is a poor comparator.

## Not chosen

- **2:4 or unstructured sparsity.** The T4 has no sparse tensor cores, so neither saves memory
  or time here; the budgets are measured peak memory.
- **SliceGPT, Sheared LLaMA, Minitron.** Other structured families (rotation-based slicing,
  targeted structured pruning with continued pre-training, pruning with large-scale
  distillation). They need training budgets or model changes outside the shared recovery recipe
  that every arm uses.

## Disclosure

The change was made after E5 session 1's validation results, before any pruned E7 model and
before any test result. It applies to every generic arm alike and makes the comparison arms
stronger, which works against H1. E5R and E7U were added before any of their results. Session
1's results stay committed in `results/superseded/inner_units_removable/` as "the original
pre-registered E5 (global per-parameter ranking, inner units of protected layers removable);
appendix; outside H1-H4". Session 2 (stopped at `d9039dc`) produced nothing to keep.

## Offline estimate (not a result)

`scripts/replay_removal.py` (CPU) runs stage 05's own search (`recovery.search_budget`) on the
committed scores with an **estimated** footprint: E2's measured peak minus 2 bytes per removed
parameter. On session 1's two runs this estimate is within 0.5 % of the measured drop (ratios
1.0050 and 1.0041). Output: `results/replay/removal_replay.md` (tables) and two CSV files.

| Arm | B1 | B2 | B3 |
|---|---|---|---|
| E5 magnitude, E5 activation, E5R, E7U (k of 86 per layer) | 13 (15.1 %) | 29 (33.7 %) | 46 (53.5 %) |
| E7 (global; layers / KV groups / channel groups) | 0 / 5 / 370 | 0 / 11 / 879 | 4 / 13 / 1,050 |

Every arm reaches B3 and no arm removes a protected unit. The search keeps the smallest k in the
band, so k sits at the band's upper edge (+2.1 to +2.7 % above the target for the uniform arms):
k = 33 at B2 would be the first k at or below the target itself (k = 29 to 35 lie in the band).
The measured search on the T4 can differ from this estimate by a step.

## Implementation

- `configs/pruning.yaml`: `pruning.removal.allocation: global`; `uniform_mlp` in
  `configs/experiments/E5.yaml`, `E5R.yaml`, `E7U.yaml`.
- `budgets.uniform_groups`, `uniform_removal`, `search_uniform_budget`; `recovery.search_budget`
  (both allocations; stage 05 and the replay call it).
- `stage.scores_from` / `scores_path`: E7U reads E7's scores; stage 04 refuses to score for it.
- Every pruning run writes `removal_by_layer.csv` (36 rows), the units removed by type, and for
  `uniform_mlp` the metric `uniform_k` and `uniform_k` in `removal.json`.
- A recovery resume state saved for another removal set is refused (`removal_digest.txt`).
- RC-G2 parts `E5R` and `E7U` (session 5); smoke jobs `E5R` and `E7U`.
- E7's 30-group check (session 4) adds the fit of the exact effect on the first-order estimate
  (slope and intercept with 95 % intervals) and E7's B2 selection again with the 30 exact
  effects in place (`first_order_check_b2.csv`): how many of the 30 switch between kept and
  removed, and whether any KV-group or layer decision changes. `--followup-only` reruns this part
  on CPU from the committed files.
- Tests: `tests/test_pruning_design.py`, `tests/test_recovery.py`.
