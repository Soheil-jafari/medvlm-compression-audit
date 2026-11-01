# Pre-registration

**Status: DRAFT** (agent, 2026-10-12). Not frozen. Nothing in it may be changed after the
freeze without approval from all three members and a dated entry in `docs/decisions_log.md`.

Every compression run (E3-E9, A1-A6) starts only after the freeze: the stage scripts refuse
to run while `preregistration.frozen_on` in `configs/preregistration/finding_weights.yaml` is
open (`cli.require_frozen_preregistration`), and the test split is locked until then
(`evaluation/runner.py`).

How to read this draft:

- plain text = decided (the source is named);
- **[RC-D1]**, **[RC-S0]**, **[RC-B0]** = filled in by the stated rule from that run card's
  committed numbers, by the agent, before the team review;
- **[team]** = a decision for the team; the draft gives a proposal and the reason.

**Frozen on:** ____ &nbsp;&nbsp; **Commit:** ____ &nbsp;&nbsp; **Approved by:** ____

## 1. Domain, models and data

| Item | Value | Source |
|---|---|---|
| Domain | GI endoscopy | D02 |
| Teacher | Qwen2.5-VL-7B-Instruct, revision `cc594898`, with **our own** LoRA adapter, trained on our training split only at image side 672 (RC-TT1), merged. The published SimulaMet adapter (revision `79e77755`) was trained on the official train split, which holds almost every image of our test split: it is used only for the appendix item of section 6 | D01 amendment 2026-10-18 |
| Teacher training (rule 3) | recipe from the SimulaMet `args.json` where it applies (LoRA on the language model, vision tower frozen, lr 2e-5, cosine, 3 % warm-up, weight decay 0.1), 16 examples per update as micro-batch 1 x 16, fp16 with loss scaling. Setup: fp16 over 2 x T4 unless it runs out of memory or is more than 2x slower per example than QLoRA (NF4) on 1 x T4. N: the largest multiple of 1,000 training examples within 15 GPU hours from RC-TT0's warm step time, capped at one pass, drawn stratified by question type with seed 20261012. Quality: Gate 1 passes and validation macro balanced accuracy >= E1 seed 0 + 0.02, else one extension of +50 % of N, accepted and reported either way | D01 amendment, `configs/teacher_training.yaml` |
| Student | Qwen2.5-VL-3B-Instruct, revision `66285546d2b821cf421d4f5eb2576359d3770cd3` (pinned 2026-10-16 after RC-S0 passed); fallback Qwen3-VL-2B-Instruct; image side 672 for every model (decided 2026-10-17, RC-S0b) | D05 amendment |
| Tokenizer equality | `tokenizers_identical = 1` in `results/smoke/RC-M1.csv` | RC-M1 |
| Main data | Kvasir-VQA-x1, revision `61e41148`: all 6,449 images, with the questions of both official splits, re-split **by image** into train / validation / test = 70 / 15 / 15 % of the images, seed 20261012, stratified by finding answer value and Kvasir-SEG mask (`data/splits.py: assign_groups_stratified`). The official split is by question, so it is not used (data card `docs/data_cards/kvasir_vqa_x1.md`). After the split, the held-out filter of D02 amendment 6 removes from validation and test the images too similar to a training image (row Test sets) | D02 amendments 2026-10-18 and 6, RC-D1 run 4b |
| Near-duplicates | D02 amendment 3 (2026-10-19, before any model result): a pair of images is joined when the cosine of their embeddings reaches the rule chosen on human labels. Embeddings: DINOv2-small (`facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, licence apache-2.0 per the hub record; RC-D1-embed, 2026-10-19), two views (whole image; central 60 %), 224 x 224, L2-normalised. 298 candidate pairs (stratified, border zones oversampled, seed 20261012) are labelled by two raters on sheets without scores; each label file's SHA-256 is committed (`docs/runs/RC-D1-label_commitments.csv`) and the scoring refuses a changed file. Two raters, no third member (D02 amendment 4, 2026-10-22, before scoring; it records the deviations from blinding: rater 2 labelled pairs 1-10 after seeing rater 1's labels for them, rater 1 labelled pairs 1-10 and 291-298 after the files were exchanged). Cohen's kappa on pairs 11-290 (both blind; as scored) **[RC-D1-label]** `labels:kappa`, reported only; no kappa threshold triggers anything. Amendment 4's primary truth (positive = `same_scene` for both raters; `unsure` is not `same_scene`) chose `full:0.93`; RC-D1 run 4 with it passed the automatic checks but failed the deciding check (same-scene pairs across splits on the final contact sheet: about 17, 4 and 13 of 24 in the three bands), so it is **not accepted** (D02 amendment 5, 2026-10-28). **Post hoc** (amendment 5, after that check): truth = `same_scene` for either rater; its rule by the same criterion (`labels:either_rater:rule:*`, `full:0.92` OR crop >= 0.80) starts a sweep fixed before any new sheet: `full:T` OR crop >= 0.80, T in {0.92, 0.90, 0.88, 0.86, 0.84}, each split built as `prepare` builds it, rules over the 10 % guard marked without a sheet; the rule used is the highest T within the guard whose three cross-split bands show no same-scene pair (judged by the team; counts per band per T recorded) **[RC-D1 sweep]** `results/data/near_duplicate_sweep.csv`; if none is clean, stop. **Sweep result** (2026-10-28, D02 amendment 6): only `full:0.92` OR crop >= 0.80 stayed within the guard (largest group 447, 6.9 %), and its sheet was not clean (about 12, 3 and 1 of 24 same-scene pairs per band), so the sweep stopped. **Post hoc** (amendment 6, fixed before any new sheet): the groups of `full:0.92` OR crop >= 0.80, unchanged split, then the held-out filter (row Test sets). The paper reports the label-chosen rule, its failure, the sweep, the filter and its selection. Top-up (recorded 2026-10-19, before any label): if fewer than 30 labelled pairs lie within +-0.03 of the chosen threshold (chosen view), 100 more candidate pairs from that band (seed 20261013, labelled pairs left out) are labelled by both raters (no third member there either) and the rule is chosen again once on all labels by the same criterion. The rule, fixed in advance: of 450 rules (cosine 0.50-0.99 by 0.01; full, centre or larger view; alone or OR crop >= 0.65 / 0.80), the highest sensitivity with precision >= 0.5 on the labels and the largest group <= 10 % of the images; ties to the higher precision, the simpler rule, the lower threshold. Its sensitivity and precision with Wilson 95 % intervals, and weighted to all candidates: **[RC-D1-label]**. Applied to all pairs; groups whole to one split. No validation or test image has a joined pair with a training image (checked: **[RC-D1]** `val_test_confirmed_duplicates_of_train` = 0). The deciding check is a contact sheet of the closest cross-split pairs (by cosine, crop and dHash). HyperKvasir has no patient IDs, so the splits are not patient-disjoint (a limitation) | D02 amendments, RC-D1-embed, RC-D1-label, RC-D1 run 4 (not accepted), the sweep and run 4b |
| Test sets (D02 amendments 6 and 7) | **Two co-primary tests** (D02 amendment 7, 2026-10-28, post hoc, before any model result): (1) the internal test after the held-out filter, reported as a **reduced-leakage** test, and (2) an **external test from another centre** (PolypGen and/or GastroVision; licences to be checked, run card RC-X2; which of them, and whether it has the lesion masks the primary endpoints need, **[RC-X2]**). The held-out filter removes from validation and from test every image whose nearest training image has a full-view cosine >= F, or a crop correlation >= 0.70, or lies within 4 dHash bits with a crop correlation >= 0.50; it never moves images between splits and never touches train; validation/test pairs are counted, not filtered. **F = 0.86**, the lowest F of the sweep that keeps >= 50 % of the test images (no F was clean; the residual same-examination pairs at F = 0.86 are about 4-6 of the 24 most similar by cosine, 0 by crop, 0 by dHash): the test keeps 512 of 968 images and 12,067 of 21,728 QA rows (`results/data/near_duplicate_filter_sweep.csv`). Limitation: same-examination pairs cannot be removed without patient or video ids. **Every model is scored on three tests**, and the conclusions (H1-H4) are reported for each: the filtered test, the unfiltered test (the filtered test with the removed test images added back, a second private test file, scored as `test_unfiltered`; comparisons in `results/comparisons_unfiltered/`), and the external test **[S11]**. The strict subset is no longer a separate analysis (2026-10-29): at F = 0.86 it is 100 % of the filtered test (its margin, 0.05 below 0.92, is above F). The split is RC-D1 run 4b's: `split_sha256` `54dfa70a9aa99ea86262447acfd16fe4d4e871272781a31f39c9f330f4f7d090`, `test_unfiltered_sha256` `adf50c68ee84f727329860ddbd17a6747d3b2e0e37e9e33dd4dcd6b36dc036cb` (pinned in `configs/data/kvasir_vqa_x1.yaml`). How the two co-primary tests combine in a verdict (support needed on both, and the multiplicity correction across them) **[team]**, before the freeze | D02 amendments 6 and 7 |
| Unsure sensitivity analysis | for the chosen rule, `unsure` counted as not `same_scene` for the precision and as `same_scene` for the sensitivity (with two raters: positive when both say `same_scene` or `unsure`), and the rule this reading would choose: **[RC-D1-label]** `labels:unsure_analysis:*`. Reported only; run 4 uses the primary rule | D02 amendments 3 (additions of 2026-10-19) and 4 |
| Strict-test sensitivity analysis | **dropped** (2026-10-29, D02 amendment 7 additions): at F = 0.86 the strict subset (most similar training image below 0.92 - 0.05 = 0.87 by cosine and 0.80 - 0.05 = 0.75 by crop) is 100 % of the filtered test, because the held-out filter already removes every test image at a cosine >= 0.86 or a crop >= 0.70; it would repeat the filtered-test results | D02 amendments 3 and 7 |
| Masks | Kvasir-SEG, linked to VQA images within 4 dHash bits; ambiguous links dropped (counts **[RC-D1]**) | D02, RC-D1 |
| Split hash | `54dfa70a9aa99ea86262447acfd16fe4d4e871272781a31f39c9f330f4f7d090` (RC-D1 run 4b, the filtered splits); the unfiltered test `adf50c68ee84f727329860ddbd17a6747d3b2e0e37e9e33dd4dcd6b36dc036cb`; pinned in `configs/data/kvasir_vqa_x1.yaml` and checked whenever the shared data are attached | RC-D1 run 4b, D02 amendments 6 and 7 |
| External test | **co-primary** (D02 amendment 7): PolypGen and/or GastroVision, from another centre than Kvasir, deduplicated against our images; licences, centres, masks and format to be checked (run card RC-X2; nothing downloaded yet). EndoBench (RC-X1) is no longer among the test sets | D02 amendment 7 |
| Training pool | one draw of `training.data_subset` training samples with seed 20261012 (`offline_kd.training_pool`), shared by E1, E2, E4 and every recovery (its first 25 %); the teacher cache holds exactly this pool. **[team]** confirm (proposed in iteration 36) | iteration 36 |

## 2. Findings

| Item | Rule | Value |
|---|---|---|
| Finding list C | every answer value of the source question "are there any abnormalities in the image" (label rule `abnormalities`, `configs/data/kvasir_vqa_x1.yaml`) with at least **20** positive training images and at least **10** positive images in each of validation and test. Instruments and landmarks are control questions, not findings (D02) | **Frozen 2026-10-31** (amendment, section 11): `polyp`, `ulcerative_colitis`, `oesophagitis`, images with the value train / validation / test 810 / 105 / 106, 691 / 85 / 98, 546 / 37 / 25 (`answer_count:abnormalities:<value>:<split>`, RC-D1 run 4b). Excluded, too few to test (a limitation): barretts 14 / 2 / 0, short-segment barretts 8 / 0 / 1, hemorrhoids 1 / 0 / 0 |
| Display names | the words put into the probe template; the finding's name in the source answers, lower case, with its own article where English needs one | **Frozen 2026-10-31** (wording fix the same day, section 11): "a polyp", "ulcerative colitis", "oesophagitis". Free-text answers are searched for the dataset's wording without the article (`domain.finding_names`: "polyp", "ulcerative colitis", "oesophagitis") |
| Prevalence | images with the value / training images (4,513) | **2026-10-31:** 17.9 %, 15.3 %, 12.1 % (computed from the committed counts, `data/summary.py`) |
| Prevalence groups | the draft cut points were common >= 10 %, rare < 2 % | **Frozen 2026-10-31:** all three findings are common; **no rare or medium group exists** in this dataset, so no cut point is set (`thresholds: null`). The rare-group analysis is dropped from the internal data (a limitation); the per-finding results stay. (The draft cut points would give the same groups.) |
| Mask findings | the findings Kvasir-SEG delineates (`image_reliance_check.mask_findings`) | **Frozen 2026-10-31:** `[polyp]` |
| Lesion-size strata | quartiles of the polyp mask area of training images with a linked mask (`lesion_area_threshold:q1..q3`); "small" = below q1 | **Frozen 2026-10-31:** q1 0.05676, q2 0.11336, q3 0.19932 (the committed values, full precision in `configs/domain/endoscopy.yaml`); the same cut points for the external test |
| Probe | "Is there {finding} in this image? Answer yes or no." (wording fix of 2026-10-31, before any model result; it was "Is there a {finding} ..."), one forward pass, log P(yes) against log P(no) | D02, S04, section 11 |

## 3. Finding weights (clinical criterion E7, E8, A2, A4-A6)

`w_c = min(1 / prevalence_c, cap) * tier_multiplier(tier_c)`, prevalence from the training
images. The table is `configs/preregistration/finding_weights.yaml`; the loader refuses an
incomplete or unfrozen table for a final run.

| Item | Proposal | Reason |
|---|---|---|
| Cap **[team]** | 20 | every finding below 5 % prevalence gets the same rarity factor, so a finding with a handful of positives cannot dominate the score |
| Tiers **[team]** | three tiers, multipliers 2.0 (findings that call for action at the same examination, for example a polyp or bleeding), 1.5 (findings that change follow-up), 1.0 (others) | a simple, explainable split; the clinical assignment of each finding is the team's judgement, made from the finding names before any result |
| Normalisation | none | multiplying every weight by the same number does not change the removal order; reported weights are also shown rescaled to sum to the size of C |

| finding | prevalence | importance tier | cap | final weight |
|---|---|---|---|---|
| **[RC-D1]** | **[RC-D1]** | **[team]** | **[team]** | computed |

## 4. Budgets

| Item | Value | Source |
|---|---|---|
| Metric | peak GPU memory during inference (weights, activations, KV cache) | D07 |
| Tolerance | 3 % of the target, either side | D07 |
| Protocol | batch 1, free generation with `evaluation.max_new_tokens` (46 by the D04 rule, section 11), 3 warm-up and 5 timed passes, 32 validation images drawn with seed 0, the same for every arm | D07 |
| Reference | E2 seed 0, measured twice in fresh processes (RC-B0) | D07, RC-B0 |
| B1 / B2 / B3 | remove about 15 / 30 / 45 % of the decoder's bytes, written as fractions of E2's measured peak: **[RC-B0]** | D07 section 4 |
| Timing hardware **[team]** | the rented GPU type (final numbers); development numbers on the Kaggle T4 are labelled as such | D07 |

## 5. Endpoints

**Primary** (at B2, 3 seeds, on each of the two co-primary tests of D02 amendment 7: the reduced-leakage internal test, i.e. the filtered test split at F = 0.86, and the external test from another centre; how the two combine **[team]**, before the freeze):

1. **Small-finding recall:** recall of positive probes of the mask findings in the smallest
   lesion-area stratum.
2. **Lesion-mask effect:** flip rate with the lesion masked minus flip rate with an
   area-matched random region masked (mid-grey fill 128), on positive mask-finding probes.

**Comparator ("best of E5 and E6") [team]:** proposal: among E5:magnitude, E5:activation,
E5:taylor and E6, the arm with the highest **validation** small-finding recall at B2, averaged
over its 3 seeds (ties broken by name). The same arm is the comparator for both endpoints.
Reason: it is the strongest generic competitor on the primary endpoint, chosen without the test
split, so H3 cannot win against a weak comparator.

**Headline accuracy for H1 and H4 [team]:** the overall probe accuracy is dominated by absent
findings (most images lack most findings), so a model that drifts towards "no" keeps a high
accuracy while losing recall, which would make H1 easy to support. Proposal: H1 and H4 use the
**macro balanced accuracy** (per finding, the mean of sensitivity and specificity, averaged over
findings; `macro_balanced_accuracy`); the overall accuracy is still reported.

**Secondary:** per-finding sensitivity and specificity, macro-F1, overall accuracy,
blank-image accuracy drop and other-image tests, ECE (15 equal-width bins), Brier score, AURC, hallucination /
omission / negation rates on 1,000 free-text answers, exact match and token F1, efficiency
(parameters, file size, peak memory, cold and warm latency, throughput, energy where measurable,
kernel check). The rare-finding recall by prevalence group is dropped (2026-10-31): the dataset
has no rare group (section 2). The flip analysis of section 11 is a secondary analysis too.

## 6. Analysis plan

- **Bootstrap:** paired percentile bootstrap with 10,000 resamples (D06) that resamples
  **images**: all probes of one image are drawn together (`statistics.resample_indices` with
  `clusters`; iteration 35). Random seed of the bootstrap: 0. 95 % intervals.
- **Seeds:** for each arm, the per-image outcomes are averaged over its 3 seeds, then images are
  bootstrapped (`evaluation/comparisons.py`); seed means and SDs are reported next to them.
- **Tests:** McNemar per finding (exact binomial below 25 discordant pairs, otherwise
  chi-square with continuity correction), Holm correction across findings; the two primary
  endpoints are Holm-corrected together.
- **Verdicts** (`evaluation/verdicts.py`, written to `docs/results/VERDICTS.md`):
  H1 supported if every arm's CI of (relative drop in small-finding recall minus relative drop
  in the headline accuracy) is above 0, not supported if any is below 0; H2 supported if the CI
  of (best pruning arm minus E3 `w8`) small-finding recall is below 0, not supported if above 0;
  H3 supported if both primary differences are above 0 with Holm-adjusted p < 0.05, not
  supported if either CI lies at or below 0; H4 supported if any Kendall tau between the
  accuracy ranking and a clinical ranking has a CI upper bound below 1. Otherwise inconclusive.
  The verdicts are reported for each of the three tests (D02 amendment 7, with the strict
  subset dropped on 2026-10-29): the filtered test, the external test and the unfiltered
  test.
- **Comparison step** (`scripts/09_compare_arms.py`, `evaluation/compare_stage.py`): one
  sequence of image-level resamples (seed 0) is shared by every arm and every comparison. An
  arm with a missing seed is left out and listed, never compared on fewer seeds.
  - H1 for every arm of E3-E8 (each variant and budget): the relative drop against E2 of the
    small-finding recall minus that of the headline accuracy. (The same with the rare-group
    recall is not computed: no rare group, 2026-10-31.)
  - H2 **[team]** proposal: E3 `w8` is compared at the budget whose target is nearest to its
    measured peak memory, against the generic pruning arm (E5 criteria, E6) with the highest
    validation small-finding recall at that budget (mean of 3 seeds).
  - H4: the compressed students E3-E8 at every budget, ranked by the headline accuracy and by
    small-finding recall and ECE (negated, so higher is better in both); the rare-group ranking
    is dropped with the rare group (2026-10-31).
  - McNemar per finding **[team]** proposal: on each probe's majority answer over the 3 seeds
    (right in at least 2), for E7 against the comparator and for H2.
- **Answer parsing:** probes are scored, never parsed, so they cannot be unparseable. Free-text
  answers count as "yes"/"no" only if they start with that word after leading punctuation
  (`clinical.parse_yes_no`); anything else is an error.
- **Exclusions:** a quantized run that fails the low-precision kernel check is not reported as
  quantized; a budget-matched run outside the 3 % tolerance is reported, marked, and the primary
  comparison is repeated without such runs as a sensitivity analysis; a run that crashes is
  rerun once with the same seed and otherwise reported as failed.
- **Appendix item: the published adapter** (secondary, pre-registered 2026-10-18). The
  published SimulaMet adapter (`configs/models/qwen2_5_vl_7b_teacher_simulamet.yaml`) is
  evaluated on our test split with the same suites as E0. It is labelled "has seen these images
  in training", because it was trained on the official train split, which holds almost every
  image of our test split. It is reported next to our teacher (E0), with the official split's
  overlap counts from the data card. It has no other use in the study.
- **Teacher quality rule** (rule 3, D01 amendment). Our teacher must pass Gate 1 and, once E1
  seed 0 exists, reach a validation macro balanced accuracy of at least E1 seed 0's + 0.02. If
  either fails, it is trained once more for +50 % of N (the next examples of the same order,
  from the trained adapter), re-checked, and accepted and reported either way. After that one
  extension, a failed Gate 1 stops the study as before (`docs/FINAL_GOAL.md`).
- **Gate 2** (reading only): E7 against the comparator at B2 on the validation split, seed 0
  (RC-G2), after the freeze. It decides which article shape is drafted first; it changes no
  setting.
- **Training-length check** (secondary, **not confirmatory**; added 2026-10-16, decision 3 of
  Soheil). E2 and E7 are run once more with **30,000** training examples, at one budget (**B2**)
  and seed 0, next to the main 10,000-example runs:
  - *Data:* the first 30,000 of the same seeded pool (`offline_kd.training_pool`, seed
    20261012). Every pool is a prefix of one shuffle, so the 30,000 contain the 10,000 of the
    main runs (`tests/test_training_config.py`): only the amount of data changes, together with
    what follows from it, the number of steps (`ceil(30,000 / 16)` = 1,875) and the order in
    which the examples are visited. Everything else is the D06 recipe.
  - *How E7 depends on E2, and what changes:* every pruning arm starts from the E2 student of
    its seed (adapter merged) for the scoring (stage 04), the removal and the recovery
    (stage 05) and the evaluation (the pruned model is rebuilt from it); the recovery runs on
    the first 25 % of the pool for 25 % of the E2 steps. For the check, E7 starts from the
    30,000-example E2 (E2@30k) and recovers for 25 % of the E2@30k steps on the first 25 % of
    its pool (7,500 examples, which lie inside the main 10,000), with the same rounding as the
    main recovery. Its B2 target is the same absolute footprint as in the main runs (a fraction
    of the main E2 reference, RC-B0; the merged 30k student has the same architecture).
  - *Reported (appendix), whatever it shows:* at 10,000 (the main seed-0 runs) and at 30,000,
    the two primary endpoints of E2 and of E7 at B2, the paired difference E7 - E2 with the
    image-level paired bootstrap 95 % interval (10,000 resamples, seed 0; seed 0 against seed 0
    at both sizes), and whether the order of E2 and E7 on each primary endpoint is the same at
    both sizes. It is not part of H1-H4, cannot change the main analysis, the comparator or a
    verdict, and no setting is changed because of it.

## 7. Compression settings (decided, frozen with this document)

| Area | Settings | Source |
|---|---|---|
| Training (E1, E2, E4) | LoRA rank 16, alpha 32, dropout 0.05 on all decoder projections; AdamW 1e-4, linear decay, 20 warm-up steps, 16 examples per update as micro-batch 1 x 16 accumulation steps without gradient checkpointing (D06 amendment, 2026-10-17), KD weight 0.5, temperature 2; one pass over the pool: 10,000 examples, 625 steps (decided 2026-10-16); the training-length check adds E2 and E7 at 30,000 examples (section 6) | D06 |
| Recovery (E5-E7, A1-A6) | the same adapter, optimiser and loss; 25 % of the E2 steps on the first 25 % of the pool; identical for every arm (fingerprint logged) | D06 |
| Pruning | units: decoder layers, KV groups, MLP channel groups of 128; score per parameter; first 2 and last 1 layers protected; 128 calibration images; one-shot removal; 1 negative per positive; signed effects; probe batch 4; how a unit is switched off while scored **[team]** (D07 section 1) | D07 |
| Quantization | E3 `w4`, `w8`: GPTQ, groups of 128, 128 calibration images, decoder linear layers only; `w8a8`: LLM.int8; E8, E9: the `w4` recipe; kernel class and versions from RC-Q0 | D08 |
| E4 | 4-bit learned step sizes (groups of 128), LoRA through the rounding, GRACE's confidence-gated decoupled distillation on the cached top-20 (alpha 1, beta 4, T 2) | D08 |
| Ablations (B2, 3 seeds) | A1 weights 1; A2 rarity only; A3 random order over E7's units, ranked **without** the per-parameter division **[team]** confirm (with it, whole layers would go first: an estimate from the student's unit sizes gives about 98 % of the removed parameters at a 30 % cut, so A3 would be random layer removal; without it about half); A4 32 / 64 / 128 / 256 nested calibration images; A5 KL effect (scored exactly; if D07 section 1 uses the first-order estimate for some units, both A5 variants run on the exactly scored unit types only); A6 layers only / KV and channel groups only | S10, iteration 42 |
| Reuse of E7 **[team]** | proposal: `A4:n128` and `A5:logprob_drop` have exactly E7's settings (a test checks it), so E7's B2 runs are reported for them (6 runs fewer) | iteration 42 |

## 8. Deviations from the methods we follow (declared in advance)

- E6 scores layers once and removes them in one shot like every other arm; the method it
  follows removes layers iteratively (D07).
- E4 is a reduced GRACE: no relational alignment of visual tokens, no adaptive loss weight, no
  full fine-tuning; the gate and the decoupled terms use the cached top-20 (D08).
- `w8a8` is LLM.int8 (dynamic, 16-bit outliers), not a calibrated static W8A8 (D08).

## 9. Freeze procedure

1. The agent fills every **[RC-D1]**, **[RC-S0]** and **[RC-B0]** item from the committed
   numbers by the rules above.
2. The team decides every **[team]** item (the tiers from the finding names only).
3. All three members approve; the agent writes the date, the commit hash and the approvers
   above and into `configs/preregistration/finding_weights.yaml` (`frozen_on`, `frozen_by`,
   `commit`), and logs the freeze in `docs/decisions_log.md`.

## 10. Deviations after the freeze

List any deviation from this document after the freeze, with date and reason.

## 11. Amendments before the freeze

### Amendment of 2026-10-31 (Soheil): evaluation settings frozen, before any model result

Recorded before any model result: RC-TT1 trained the teacher, and its validation and Gate 1
waited on these settings, so no model has been scored. Every count comes from the committed
`results/data/summary.csv` (RC-D1 run 4b, `split_sha256` `54dfa70a...`).

1. **Finding list** (`domain.findings`, label rule values): `polyp`, `ulcerative_colitis`,
   `oesophagitis`. Images with the abnormality answer value, train / test: 810 / 106,
   691 / 98, 546 / 25 (validation 105, 85, 37). They meet the draft rule of section 2 (at
   least 20 training and 10 validation and test images). Probe display names come from the
   dataset's own wording (see the wording fix, item 8).
   - Excluded: barretts, short-segment barretts and hemorrhoids (test images 0, 1, 0): too few
     to test. Stated as a limitation. An image whose only abnormality value is excluded is a
     negative for the three findings.
   - Negatives: the source answers hold no "none" value for this question (no
     `answer_count:abnormalities:none` row), so only images with at least one abnormality
     value are labelled; a finding's negatives are the images with another abnormality. The
     first validation run (E0) counts the probes; the team reads that count before the freeze.
2. **Prevalence groups:** all three findings are common (training prevalence 17.9 %, 15.3 %,
   12.1 % of the 4,513 training images). No rare group exists in this dataset.
   - The rare-finding analysis is dropped from the internal data (a limitation). The
     per-finding results stay (sensitivity, specificity and McNemar per finding).
   - The groups are set so the code reflects this: `common` holds the three findings,
     `medium` and `rare` are empty, `thresholds: null`. No cut point is invented. The
     evaluation no longer runs the rare-finding suite. The comparison step writes no
     `diff:drop_gap_rare` and no `tau:rare_recall` row when the rare group is empty, and says
     so in `choices.json` (`rare_group`).
3. **Small findings** (`small_findings.thresholds`): the committed training quartiles of the
   polyp mask area, q1 0.05676, q2 0.11336, q3 0.19932 (`lesion_area_threshold:q1..q3`);
   small = below q1.
   - The filtered test holds 86 masked images. If their areas follow the training quartiles,
     about 21 are small (86 / 4, an estimate). The real count is written by every test
     evaluation (`n_stratum:0`, with `n_mask_positives`), and is reported next to the
     small-finding recall and its image-bootstrap 95 % interval.
   - With about 21 small polyps that interval is wide: about +-0.17 at a recall of 0.8 (normal
     approximation, an estimate). The paired differences between arms are expected to be
     narrower, by an amount not known before the runs.
4. **Generation limit** (`evaluation.max_new_tokens`, E0's `teacher_cache.max_new_tokens`, and
   `budgets.measurement.max_new_tokens`): the D04 rule, `ceil(1.5 x p99 answer words) + 8`.
   The configs hold the rule's name (`d04_rule`); the code computes the number from the
   committed summary each time a run reads it (`data/summary.py`). With p99 = 25 words it is
   ceil(37.5) + 8 = **46**. No number is typed by hand.
5. **Mask findings** (`image_reliance_check.mask_findings`): `[polyp]`.
6. **External test** (D02 amendment 7, co-primary): **PolypGen first**, because it has masks
   and many polyps, so the small-polyp stratum also has enough cases there. Its card (RC-X2)
   checks the licence first (the exact licence, or stop), and plans:
   - image-level probes for polyp present / absent with the same probe template;
   - lesion-size strata from its masks, with **our** training quartiles above;
   - the same near-duplicate overlap check against our training images.

   Nothing is downloaded yet.
7. **Secondary analysis: what compression forgets** (after Hooker, Courville, Clark, Dauphin
   and Frome, arXiv:1911.05248). For each method and budget, the test items (probes) whose
   answer flips between E0 and the compressed model:
   - an item flips when the arm's majority answer over its 3 seeds differs from E0's answer;
   - reported: the number and rate of flips, and how they are distributed over the findings
     and over the lesion-size strata of the positive polyps (rate, share and the ratio of each
     group's flip rate to the overall one), and the flip rate of small polyps minus that of
     larger ones;
   - question: are small polyps over-represented among the flips (the interval of
     `flip_ratio:stratum0` above 1)?
   - 95 % image-level percentile bootstrap, the comparison step's resamples (seed 0);
   - descriptive: not part of H1-H4, no verdict, no multiplicity correction;
   - E3-E8 start from E2, which is distilled from E0, so their flips against E0 include those
     of the distillation. E2's own row (`E2_vs_E0`) is reported next to them;
   - no extra GPU: it reads the per-probe answers stage 06 already saves
     (`evaluation/flips.py`, run by the comparison step; RC-R1 also hands off E0's test
     outcomes).

8. **Probe wording fix** (Soheil, 2026-10-31, the same day, after the items above and
   **before any model result**: RC-E0a had not run, so no model had answered a probe). The
   template "Is there a {finding} in this image? Answer yes or no." made two of the three
   probes ungrammatical ("Is there a ulcerative colitis", "Is there a oesophagitis").
   - Template: **"Is there {finding} in this image? Answer yes or no."**
   - Display names carry their own article: `polyp` -> "a polyp", `ulcerative_colitis` ->
     "ulcerative colitis", `oesophagitis` -> "oesophagitis". The polyp probe reads as before.
   - Free-text answers are still searched for the dataset's wording without an article
     (`domain.finding_names`), so the hallucination, omission and negation rates do not change
     with the probe wording ("no polyps" is still found).
   - The external test (PolypGen) uses the same template and the display name "a polyp".

Still open before `preregistration.frozen_on` is set: how the two co-primary tests combine in a
verdict, the finding weights (cap, tiers), the budgets and the other **[team]** items above.
The test split stays locked until then.

