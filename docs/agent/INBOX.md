# Inbox: notes from the team to the research agent

Add one dated line per note at the top of **Open**. Decisions, answers to decision
memos, priorities, "run card RC-xx done", "please stop", and so on. Notes here
override `STATUS.md`. The agent moves each handled note to **Handled** with the
iteration number.

Example: `- 2026-10-20 (Soheil): D01 decided: endoscopy, Qwen2.5-VL-7B -> 3B. Log it.`

## Open

<!-- newest first -->
- 2026-10-17 (agent, for Soheil): **pre-registration freeze list.** Every item still open,
  with a recommended answer and one reason each. `results/data/summary.csv` is not committed yet,
  so items A1-A10 show their counts as `[RC-D1]` placeholders; the agent fills them by the stated
  rule when RC-D1's two passes are in. Reply with one line:
  `- 2026-10-.. (Soheil): freeze list: approve all` or `... approve all except <numbers>: <change>`.
  The freeze itself needs Amir's and Farzan's approval too (pre-registration section 9).

  **Can be decided now**
  1. Fixed training pool. Recommended: one pool from `seeds.data_split` for E1, E2, E4 and every
     recovery, with the teacher cache holding only that pool. Reason: the cache is 10,000
     answers instead of up to 30,000, and the seeds still differ in order and starting point.
  2. Finding inclusion rule. Recommended: at least 20 positive training images and at least 10
     in each of validation and test. Reason: enough positives for a recall interval in each split.
  3. Prevalence groups. Recommended: common at 10 % or more, rare below 2 %. Reason: fixed cut
     points chosen before the counts, unlike tertiles, which move with the data.
  4. Rarity cap. Recommended: 20. Reason: keeps one very rare finding from dominating the
     criterion.
  5. Weight normalisation. Recommended: none. Reason: the criterion only ranks units, so a
     common scale changes nothing.
  6. Tiers. Recommended: multipliers 2.0 / 1.5 / 1.0 (findings that call for action, for
     follow-up, others), assigned from the finding names only. Reason: the clinical priority is
     fixed before any result.
  7. Comparator "best of E5/E6". Recommended: the generic arm with the highest validation
     small-finding recall at B2, averaged over 3 seeds, the same for both endpoints. Reason: it
     is picked on validation only.
  8. Headline accuracy for H1 and H4. Recommended: macro balanced accuracy. Reason: with few
     positives, plain accuracy mostly measures specificity.
  9. H2 matching. Recommended: E3 `w8` against the best generic arm at the budget nearest to
     `w8`'s measured peak memory. Reason: it compares at equal memory.
  10. McNemar test. Recommended: per finding, on each probe's majority answer over the 3 seeds.
      Reason: one paired test per finding instead of three.
  11. A3 ranking. Recommended: random scores not divided by the unit size. Reason: divided,
      random removal would take whole layers first (about 98 % of the removed parameters at a
      30 % cut).
  12. Reuse E7 for `A4:n128` and `A5:logprob_drop`. Recommended: yes. Reason: identical
      settings, 6 runs fewer.
  13. Scoring method (D07 section 1). Recommended: exact for layers and KV groups, first-order
      for channel groups. Reason: about 7.9 h per seed at side 672, against about 236 h for exact
      scoring of every unit (estimates); RC-G2's 30-group check tests the approximation.
  14. Recovery length. Recommended: 157 steps (`ceil(2,500 / 16)`) on the first 2,500
      examples. Reason: the same rounding as the training steps.
  15. A1 and A2 reuse of E7's effects. Recommended: the agent adds the reuse (keeps E7's
      per-probe effects). Reason: D07 already says so, the results are the same, and it saves
      about 48 GPU hours (estimate).
  16. Budget hardware. Recommended: the budgets are matched on the Kaggle T4 (as RC-B0 measures);
      the rented GPU is only for RC-T1's final timing. Reason: every pruned run searches its
      budget on the T4 in its own session.
  17. Timing GPU and precision for RC-T1. No recommendation from here: the team chooses the
      rented GPU (cost and availability). Not needed before the grid.
  18. Rare-finding pooling. Recommended: pooled within a prevalence group. Reason: rare findings
      have too few positives each for an interval.
  19. Stale config notes (A1 at B2 only; A6 `heads_channels` without whole layers).
      Recommended: confirm as the pre-registration states. Reason: they already match its text.

  **Need the RC-D1 counts** (placeholders until `summary.csv` is in)
  - A1. Finding list C: findings that pass rule 2. Depends on: `positives:<f>:train/val/test`
    [RC-D1 pass 2]. For the record (no action): in run 2's split, the rarest answer values had
    these image counts (train / validation / test): barretts 15 / 4 / 3, short-segment barretts
    8 / 4 / 1, hemorrhoids 1 / 0 / 0. Run 3's counts replace them.
  - A2. Display names. Depends on: the answer values as written (`answer_count:*`, pass 1).
  - A3. Prevalence per finding. Depends on: `prevalence:<f>` [RC-D1 pass 2].
  - A4. Common and rare groups. Depends on: A3 and rule 3.
  - A5. Lesion-size strata: quartiles of the training lesion areas; "small" is below q1.
    Depends on: `lesion_area_threshold:q1..q3` [RC-D1].
  - A6. Mask findings: `polyp`, if it passes rule 2. Depends on: `positives:polyp:*`.
  - A7. Tier of each finding. Depends on: A1 and rule 6.
  - A8. `max_new_tokens` = ceil(1.5 x p99 answer words) + 8, the same value in three configs.
    Depends on: `answer_words:p99` [RC-D1 pass 1]. Needed before RC-E0b.
  - A9. Data facts in the document. Depends on: the group rows (`groups:*`,
    `val_test_confirmed_duplicates_of_train`), `masks:linked/ambiguous/unmatched` and the printed
    `split_sha256` [RC-D1 run 3]; and the chosen `min_correlation` [RC-D1 calibration].
  - A10. Enough masked validation images for Gate 1's interval (a check). Depends on:
    `masked_images:val` [RC-D1].
  - Before pass 2: the label values of the abnormality question
    (`configs/data/kvasir_vqa_x1.yaml`). The agent proposes them from `answer_count:*` of pass 1.

  **Need another run**
  - B1. Budgets B1-B3: about 15 / 30 / 45 % of the decoder's bytes removed, written as fractions
    of E2's measured peak. Depends on: RC-B0 (E2 seed 0, two readings within 3 %).
  - B2. E4's memory at 1 x 16 without gradient checkpointing. Depends on: RC-E4 seed 0 (after
    the freeze; it stops rather than change the setting).

  Already decided or filled: image side 672, G2 0.10, micro-batch 1 x 16 (2026-10-17); the
  student revision is pinned. Still for Soheil, outside the document: tell Amir and Farzan
  about the one-student amendment.

## Handled

<!-- - YYYY-MM-DD (name): note -> iteration N, what was done -->
- 2026-10-18 (Soheil, in the session): RC-D1 run 2 result; D02 amendment 2 (two-stage near-duplicate detection, calibration step, run 3) -> iteration 52, code, tests, RC-D1, RC-TT0, RC-TT1, data card, D02, log, pre-registration.
- 2026-10-18 (Soheil, in the session): RC-D1 run 1 findings; D02 and D01 amendments, rule 3, the appendix item -> iteration 51.
- 2026-10-17 (Soheil, in the session): decisions on the image side (672), G2 (0.10), the D06 amendment (1 x 16, no gradient checkpointing), the GPU budget (90 h per week, 30 per person), RC-D1 owner (Soheil); Tasks 1-5 (shared data, hand-offs, unattended runs, work split and schedule, freeze list) -> iteration 50.
- 2026-10-16 (Soheil, in the session): RC-S0 proposals answered: data_subset 10,000 accepted; image side not decided (RC-S0b first); training-length check added (30,000, E2 and E7 at B2, seed 0); G2 unchanged -> iteration 48, configs, D05, D06, pre-registration, RC-S0b, RC-L1 and the log.
- 2026-10-12 (Soheil): D08 answered in the session (GPTQ 4/8 + LLM.int8, E4 option B, 128 calibration images, keep w8a8) -> iteration 37, D08, decisions log and quantization configs.
- 2026-10-12 (Soheil): D05 amendment accepted, pick one student (Qwen2.5-VL-3B, fallback Qwen3-VL-2B) -> iteration 34, configs, D05, RC-S0 and the log updated.
- 2026-10-12 (Soheil): D04 answered in the session (moderate Gate 1 rule, mid-grey fill, top_k 20) -> iteration 26, E0 config, D04 memo and decisions log updated.
- 2026-10-12 (Soheil): D05 decided: Q1 teacher family only, Q2 keep size, Q3 weights accepted -> iteration 4, recorded in D05, `configs/student_selection.yaml` and the decisions log.
- 2026-10-12 (Soheil): no scheduled runs; iterations run back to back in the interactive session until stopped -> iteration 4, scheduled task removed, protocol updated.
- 2026-10-12 (Soheil): D02 decided (Q1 A, Q3 EndoBench, Q5 retinal later; Q2, Q4 defaults) and D03 decided (MIT) -> iteration 8, configs, LICENSE, pyproject, README, decisions log.
- 2026-10-12 (Soheil): D01 decided: T1 (Qwen2.5-VL-7B + Kvasir LoRA) now; optional Qwen3-VL-8B teacher check at the end if budget remains -> iteration 6, teacher config DECIDED, student pool fixed to 10 Qwen-family models.
- 2026-10-12 (Soheil): asked why retinal is not the main domain -> answered in the session; retinal stays the alternative / second domain, to revisit in D02.
