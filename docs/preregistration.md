# Pre-registration

**Status: FROZEN on 2026-11-08** by Soheil and Amir (section 9), at commit `18f1b47` (drafted
by the agent from 2026-10-12; last decisions on 2026-11-08, section 11 item 14). Nothing in it may be changed after the
freeze without approval from both authors (Soheil and Amir) and a dated entry in
`docs/decisions_log.md`.

Every compression run (E3-E9, A1-A6) starts only after the freeze: the stage scripts refuse
to run while `preregistration.frozen_on` in `configs/preregistration/finding_weights.yaml` is
open (`cli.require_frozen_preregistration`), and the test split is locked until then
(`evaluation/runner.py`).

How to read this draft:

- plain text = decided (the source is named);
- **[RC-D1]**, **[RC-S0]**, **[RC-B0]** = filled in by the stated rule from that run card's
  committed numbers, by the agent, before the team review;
- **[team]** = a decision for the team; the draft gives a proposal and the reason.

**Frozen on:** 2026-11-08 &nbsp;&nbsp; **Commit:** `18f1b47658eb188975dbbf3caa4b76e2e4d611fb` &nbsp;&nbsp; **Approved by:** Soheil and Amir (Amir's written approval received)

## 1. Domain, models and data

| Item | Value | Source |
|---|---|---|
| Domain | GI endoscopy | D02 |
| Teacher | Qwen2.5-VL-7B-Instruct, revision `cc594898`, with **our own** LoRA adapter, trained on our training split only at image side 672 (RC-TT1), merged. The published SimulaMet adapter (revision `79e77755`) was trained on the official train split, which holds almost every image of our test split: it is used only for the appendix item of section 6 | D01 amendment 2026-10-18 |
| Teacher training (rule 3) | recipe from the SimulaMet `args.json` where it applies (LoRA on the language model, vision tower frozen, lr 2e-5, cosine, 3 % warm-up, weight decay 0.1), 16 examples per update as micro-batch 1 x 16, fp16 with loss scaling. Setup: fp16 over 2 x T4 unless it runs out of memory or is more than 2x slower per example than QLoRA (NF4) on 1 x T4. N: the largest multiple of 1,000 training examples within 15 GPU hours from RC-TT0's warm step time, capped at one pass, drawn stratified by question type with seed 20261012. Quality: Gate 1 passes and validation macro balanced accuracy >= E1 seed 0 + 0.02, else one extension of +50 % of N, accepted and reported either way | D01 amendment, `configs/teacher_training.yaml` |
| Student | Qwen2.5-VL-3B-Instruct, revision `66285546d2b821cf421d4f5eb2576359d3770cd3` (pinned 2026-10-16 after RC-S0 passed); fallback Qwen3-VL-2B-Instruct; image side 672 for every model (decided 2026-10-17, RC-S0b) | D05 amendment |
| Tokenizer equality | `tokenizers_identical = 1` in `results/smoke/RC-M1.csv` | RC-M1 |
| Main data | Kvasir-VQA-x1, revision `61e41148`: all 6,449 images, with the questions of both official splits, re-split **by image** into train / validation / test = 70 / 15 / 15 % of the images, seed 20261012, stratified by finding answer value and Kvasir-SEG mask (`data/splits.py: assign_groups_stratified`). The official split is by question, so it is not used (data card `docs/data_cards/kvasir_vqa_x1.md`). After the split, the held-out filter of D02 amendment 6 removes from validation and test the images too similar to a training image (row Test sets) | D02 amendments 2026-10-18 and 6, RC-D1 run 4b |
| Near-duplicates | D02 amendment 3 (2026-10-19, before any model result): a pair of images is joined when the cosine of their embeddings reaches the rule chosen on human labels. Embeddings: DINOv2-small (`facebook/dinov2-small`, revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`, licence apache-2.0 per the hub record; RC-D1-embed, 2026-10-19), two views (whole image; central 60 %), 224 x 224, L2-normalised. 298 candidate pairs (stratified, border zones oversampled, seed 20261012) are labelled by two raters on sheets without scores; each label file's SHA-256 is committed (`docs/runs/RC-D1-label_commitments.csv`) and the scoring refuses a changed file. Two raters, no third member (D02 amendment 4, 2026-10-22, before scoring; it records the deviations from blinding: rater 2 labelled pairs 1-10 after seeing rater 1's labels for them, rater 1 labelled pairs 1-10 and 291-298 after the files were exchanged). Cohen's kappa on pairs 11-290 (both blind; as scored) **0.428** (280 pairs, `labels:kappa`), reported only; no kappa threshold triggers anything. Amendment 4's primary truth (positive = `same_scene` for both raters; `unsure` is not `same_scene`) chose `full:0.93`; RC-D1 run 4 with it passed the automatic checks but failed the deciding check (same-scene pairs across splits on the final contact sheet: about 17, 4 and 13 of 24 in the three bands), so it is **not accepted** (D02 amendment 5, 2026-10-28). **Post hoc** (amendment 5, after that check): truth = `same_scene` for either rater; its rule by the same criterion (`labels:either_rater:rule:*`, `full:0.92` OR crop >= 0.80) starts a sweep fixed before any new sheet: `full:T` OR crop >= 0.80, T in {0.92, 0.90, 0.88, 0.86, 0.84}, each split built as `prepare` builds it, rules over the 10 % guard marked without a sheet; the rule used is the highest T within the guard whose three cross-split bands show no same-scene pair (judged by the team; counts per band per T recorded in `results/data/near_duplicate_sweep.csv`; no T was clean, so the sweep stopped and amendment 6 applies); if none is clean, stop. **Sweep result** (2026-10-28, D02 amendment 6): only `full:0.92` OR crop >= 0.80 stayed within the guard (largest group 447, 6.9 %), and its sheet was not clean (about 12, 3 and 1 of 24 same-scene pairs per band), so the sweep stopped. **Post hoc** (amendment 6, fixed before any new sheet): the groups of `full:0.92` OR crop >= 0.80, unchanged split, then the held-out filter (row Test sets). The paper reports the label-chosen rule, its failure, the sweep, the filter and its selection. Top-up (recorded 2026-10-19, before any label): if fewer than 30 labelled pairs lie within +-0.03 of the chosen threshold (chosen view), 100 more candidate pairs from that band (seed 20261013, labelled pairs left out) are labelled by both raters (no third member there either) and the rule is chosen again once on all labels by the same criterion. The rule, fixed in advance: of 450 rules (cosine 0.50-0.99 by 0.01; full, centre or larger view; alone or OR crop >= 0.65 / 0.80), the highest sensitivity with precision >= 0.5 on the labels and the largest group <= 10 % of the images; ties to the higher precision, the simpler rule, the lower threshold. Its sensitivity and precision with Wilson 95 % intervals, and weighted to all candidates: sensitivity 0.927 (0.806-0.975), precision 0.50 (0.39-0.61); weighted 0.980 and 0.421 (`labels:chosen:*`). Applied to all pairs; groups whole to one split. No validation or test image has a joined pair with a training image (checked: `val_test_confirmed_duplicates_of_train` = 0, RC-D1 run 4b). The deciding check is a contact sheet of the closest cross-split pairs (by cosine, crop and dHash). HyperKvasir has no patient IDs, so the splits are not patient-disjoint (a limitation) | D02 amendments, RC-D1-embed, RC-D1-label, RC-D1 run 4 (not accepted), the sweep and run 4b |
| Test sets (D02 amendments 6 and 7) | **Two co-primary tests** (D02 amendment 7, 2026-10-28, post hoc, before any model result): (1) the internal test after the held-out filter, reported as a **reduced-leakage** test, and (2) an **external test from another centre** (**PolypGen**, decided 2026-10-31 and 2026-11-08: CC BY 4.0 as stated in Ali et al., Sci Data 2023, research use allowed, citation and acknowledgement required; polyp masks; run card RC-X2; GastroVision not planned). The held-out filter removes from validation and from test every image whose nearest training image has a full-view cosine >= F, or a crop correlation >= 0.70, or lies within 4 dHash bits with a crop correlation >= 0.50; it never moves images between splits and never touches train; validation/test pairs are counted, not filtered. **F = 0.86**, the lowest F of the sweep that keeps >= 50 % of the test images (no F was clean; the residual same-examination pairs at F = 0.86 are about 4-6 of the 24 most similar by cosine, 0 by crop, 0 by dHash): the test keeps 512 of 968 images and 12,067 of 21,728 QA rows (`results/data/near_duplicate_filter_sweep.csv`). Limitation: same-examination pairs cannot be removed without patient or video ids. **Every model is scored on three tests**, and the conclusions (H1-H4) are reported for each: the filtered test, the unfiltered test (the filtered test with the removed test images added back, a second private test file, scored as `test_unfiltered`; comparisons in `results/comparisons_unfiltered/`), and the external test **[S11]**. The strict subset is no longer a separate analysis (2026-10-29): at F = 0.86 it is 100 % of the filtered test (its margin, 0.05 below 0.92, is above F). The split is RC-D1 run 4b's: `split_sha256` `54dfa70a9aa99ea86262447acfd16fe4d4e871272781a31f39c9f330f4f7d090`, `test_unfiltered_sha256` `adf50c68ee84f727329860ddbd17a6747d3b2e0e37e9e33dd4dcd6b36dc036cb` (pinned in `configs/data/kvasir_vqa_x1.yaml`). How the two co-primary tests combine in a verdict (decided 2026-11-08): **supported** only if supported on both (intersection-union, no alpha split), **partly supported** if on one; reported per test either way (section 6) | D02 amendments 6 and 7 |
| Unsure sensitivity analysis | for the chosen rule, `unsure` counted as not `same_scene` for the precision and as `same_scene` for the sensitivity (with two raters: positive when both say `same_scene` or `unsure`), and the rule this reading would choose: sensitivity 0.879 and precision 0.50 for the chosen rule, and the same threshold 0.93 (`labels:unsure_analysis:*`). Reported only; run 4 uses the primary rule | D02 amendments 3 (additions of 2026-10-19) and 4 |
| Strict-test sensitivity analysis | **dropped** (2026-10-29, D02 amendment 7 additions): at F = 0.86 the strict subset (most similar training image below 0.92 - 0.05 = 0.87 by cosine and 0.80 - 0.05 = 0.75 by crop) is 100 % of the filtered test, because the held-out filter already removes every test image at a cosine >= 0.86 or a crop >= 0.70; it would repeat the filtered-test results | D02 amendments 3 and 7 |
| Masks | Kvasir-SEG, linked to VQA images within 4 dHash bits; ambiguous links dropped (1,010 linked, 5 ambiguous dropped, 12 unmatched; `masks:*`) | D02, RC-D1 |
| Split hash | `54dfa70a9aa99ea86262447acfd16fe4d4e871272781a31f39c9f330f4f7d090` (RC-D1 run 4b, the filtered splits); the unfiltered test `adf50c68ee84f727329860ddbd17a6747d3b2e0e37e9e33dd4dcd6b36dc036cb`; pinned in `configs/data/kvasir_vqa_x1.yaml` and checked whenever the shared data are attached | RC-D1 run 4b, D02 amendments 6 and 7 |
| External test | **co-primary** (D02 amendment 7): **PolypGen** (Ali et al., Sci Data 2023; CC BY 4.0, research use allowed, citation and acknowledgement required; the paper's commercial-use wording is inconsistent, irrelevant for academic use; confirming the release page's terms is Amir's step at the download and does not block the freeze). Probes: polyp present / absent per image, the same template; size strata from its masks with **our** training q1-q3 (unchanged); the held-out filter of D02 amendment 6 against our training images with the same settings (F 0.86; `scripts/polypgen.py overlap`), removed images counted per centre; centre C4 (Oslo) reported apart, and the external test without C4 is a sensitivity analysis; the bootstrap clusters by **video sequence**, not by frame (a single frame is its own cluster). EndoBench (RC-X1) is no longer among the test sets | D02 amendment 7; section 11 item 14 |
| Training pool | one draw of `training.data_subset` training samples with seed 20261012 (`offline_kd.training_pool`), shared by E1, E2, E4 and every recovery (its first 2,500 examples); the teacher cache holds exactly this pool. Decided (iteration 36; confirmed 2026-11-08) | iteration 36 |

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

Decided 2026-11-08 (Soheil and Amir; section 11 item 14):

| Item | Value | Reason |
|---|---|---|
| Cap | 20 | every finding below 5 % prevalence gets the same rarity factor, so a finding with a handful of positives cannot dominate the score (none of the three reaches it) |
| Tiers | assigned **by rule from the finding names only**: **action** = 2.0, a treatment performed during the same examination (e.g. resection); **follow_up** = 1.5, changes treatment or follow-up but is not treated during the examination; **other** = 1.0 | a simple, explainable split fixed before any compression result |
| Normalisation | none | multiplying every weight by the same number does not change the removal order; reported weights are also shown rescaled to sum to the size of C |

Prevalence = training images with the value / 4,513 (`results/data/summary.csv`, RC-D1 run 4b):

| finding | prevalence | importance tier | cap | final weight |
|---|---|---|---|---|
| polyp | 0.179481 (810) | action (removed during the examination) | 20 | 5.5716 x 2.0 = **11.1432** |
| ulcerative_colitis | 0.153113 (691) | follow_up | 20 | 6.5311 x 1.5 = **9.7967** |
| oesophagitis | 0.120984 (546) | follow_up | 20 | 8.2656 x 1.5 = **12.3983** |

**A1 (uniform weights) and A2 (rarity only) are the pre-registered check of this choice**: if
the weights matter, E7 differs from them (E7 against A1 is the pre-specified secondary
comparison, section 5).

## 4. Budgets

| Item | Value | Source |
|---|---|---|
| Metric | peak GPU memory during inference (weights, activations, KV cache) | D07 |
| Tolerance | 3 % of the target, either side | D07 |
| Protocol | batch 1, free generation with `evaluation.max_new_tokens` (46 by the D04 rule, section 11), 3 warm-up and 5 timed passes, 32 validation images drawn with seed 0, the same for every arm | D07 |
| Reference | E2 seed 0, measured twice in fresh processes (RC-B0) | D07, RC-B0 |
| B1 / B2 / B3 | **remove 15 / 30 / 45 % of E2's decoder-layer bytes**: target = E2's measured peak minus that share of its `decoder_layers_bytes` (frozen 2026-11-08; RC-B0 only measures the two numbers, which turns the fractions into GB; `configs/budgets.yaml` `decoder_bytes_removed`) | D07 section 4; section 11 item 14 |
| Budget hardware | **Kaggle Tesla T4** (`budgets.measurement.hardware`, 2026-11-08): RC-B0 and every budget search | section 11 item 14 |
| Timing hardware (RC-T1) | the rented GPU type for the final timing numbers: **open, and it does not block the freeze** (timing is secondary; development numbers on the Kaggle T4 are labelled as such) | D07 |

## 5. Endpoints

**Primary** (at B2, 3 seeds, on each of the two co-primary tests of D02 amendment 7: the reduced-leakage internal test, i.e. the filtered test split at F = 0.86, and the external test from another centre; how the two combine: decided 2026-11-08, section 6 "Verdicts"):

1. **Small-finding recall:** recall of positive probes of the mask findings in the smallest
   lesion-area stratum, **at the model's operating point** (section 11 item 10: the probe-score
   threshold fixed on validation at specificity 0.95 for that finding, applied unchanged to
   every test; `op_small_finding_recall`).
2. **Lesion-mask effect:** flip rate with the lesion masked minus flip rate with an
   area-matched random region masked (mid-grey fill 128), on positive mask-finding probes.

**Comparator ("best of E5 and E6")** (section 11 item 12, amendment of 2026-11-08, Soheil and
Amir): chosen on **validation, seed 0, at B2**, among E5:magnitude, E5:activation, E5:taylor and
E6, by this ordering: (i) `op_small_finding_recall`; (ii) ties: the recall at the operating
point over all mask-positive polyps (86 on validation, `op_mask_positive_recall`); (iii) ties:
the mean, over the small-polyp probes, of the probe margin minus that arm's own
`op_threshold:polyp` (`op_small_margin_mean`); (iv) then the name (`seed_plan.rank_key`,
`scripts/choose_comparator.py`). Only the chosen arm runs seeds 1 and 2. The same arm is the
comparator for both endpoints. Reason: it is the strongest generic competitor on the primary
endpoint, chosen without the test split, so H3 cannot win against a weak comparator; (ii) and
(iii) exist because E1 and E2 both score 13 of 13 on the validation small-polyp recall, so (i)
alone can tie at the ceiling and leave the choice to the name. (Replaces "averaged over its 3
seeds".)

**Pre-specified secondary comparison** (section 11 item 13, 2026-11-08): **E7 against A1**
(uniform finding weights) at B2, on 3 seeds each, on the two primary endpoints (difference with
its image-bootstrap interval and unadjusted p) and McNemar per finding (`E7_vs_A1`). It asks
whether the clinical weights matter beyond probe effects alone. It is **not part of any
confirmatory verdict** and not in the Holm family of the primary endpoints. A1 re-weights E7's
per-finding effects of the same seed (no scoring of its own), and its seeds 1 and 2 run under
the same futility rule as the main tier.

**Headline accuracy for H1 and H4** (decided 2026-11-08): the overall probe accuracy is dominated by absent
findings (most images lack most findings), so a model that drifts towards "no" keeps a high
accuracy while losing recall, which would make H1 easy to support. H1 and H4 use the
**macro balanced accuracy** (per finding, the mean of sensitivity and specificity, averaged over
findings), **at the operating point** since 2026-11-06 (`op_macro_balanced_accuracy`, section
11 item 10); the plain yes/no macro balanced accuracy and the overall accuracy are still
reported, as secondary. (Rule 3's quality check of the teacher keeps the plain macro balanced
accuracy it was pre-registered with, and was read with it.)

**At the operating point** (section 11 item 10): per-finding sensitivity and specificity, the
small-finding recall and its strata, the headline accuracy, the flip analysis and the McNemar
answers. The AUROC is reported next to them.

**Secondary:** the plain yes/no metrics (the answer the model gives, "yes" when log P(yes) >
log P(no)): per-finding sensitivity and specificity, and next to them the per-finding
AUROC from the probe scores (image-bootstrap 95 % interval) and the per-finding "yes" rate
(added 2026-11-02, section 11 item 9), macro-F1, overall accuracy,
blank-image accuracy drop and other-image tests, ECE (15 equal-width bins), Brier score, AURC, hallucination /
omission / negation rates on 1,000 free-text answers, exact match and token F1, efficiency
(parameters, file size, peak memory, cold and warm latency, throughput, energy where measurable,
kernel check). The rare-finding recall by prevalence group is dropped (2026-10-31): the dataset
has no rare group (section 2). The flip analysis of section 11 is a secondary analysis too.

**Secondary analyses added 2026-11-08** (section 11 item 14; not in any confirmatory verdict):
(a) the recall at the operating point over **all mask-positive polyps**
(`op_mask_positive_recall`, 86 on validation), next to the small-polyp recall; (b) the
**headline accuracy without ulcerative colitis** (`op_macro_balanced_accuracy_without:
ulcerative_colitis`), because its scores barely separate the classes (E2 seed 0: operating-
point sensitivity 0.27, AUROC 0.80); (c) **E0 against E2**: the small-finding recall E2 minus
E0 (`E2_vs_E0`, paired image bootstrap) and the flips of E2 against E0: what distillation itself
forgets, which every compressed arm inherits.

## 6. Analysis plan

- **Bootstrap:** paired percentile bootstrap with 10,000 resamples (D06) that resamples
  **images**: all probes of one image are drawn together (`statistics.resample_indices` with
  `clusters`; iteration 35). Random seed of the bootstrap: 0. 95 % intervals.
- **Seeds** (section 11 item 12, amendment of 2026-11-08; replaces item 11): the **main-claim
  tier** has 3 seeds (0, 1, 2): E2, E7 at B2 and the comparator at B2. **Every other arm has
  seed 0 only** and is reported as exploratory / descriptive: the other E5 / E6 arms, every E3
  variant (`w8`, `w4`, `w8a8`), E4, E8, E9, every arm at B1 and B3, the ablations A1-A6 and the
  training-length check. E1 keeps the 3 seeds it ran before the amendment. A1 at B2 has 3
  seeds for the pre-specified secondary comparison E7 against A1 (item 13). **No arm has 2
  seeds** (the majority answer is "right in more than half of the seeds", so with 2 a 1-1
  split counts as wrong; `seed_plan.check_seed_count`). For a 3-seed arm the per-image outcomes
  are averaged over its seeds, then images are bootstrapped (`evaluation/compare_stage.py`);
  seed means and SDs are reported next to them. A single-seed arm is read against a yardstick:
  the spread of E7 at B2 over its 3 seeds; a single-seed difference smaller than that spread is
  reported as within run-to-run noise. **Futility-only stopping rule:** after seed 0, seeds 1
  and 2 of the main tier run unless E7 at B2 ranks **below** the comparator by the ordering of
  section 5 (validation, seed 0); then the study stops there and reports seed 0 as exploratory,
  with no confirmatory claim. Otherwise seeds 1 and 2 run, seed 0 stays in the final analysis
  and the test is unchanged.
- **Tests:** McNemar per finding (exact binomial below 25 discordant pairs, otherwise
  chi-square with continuity correction), Holm correction across findings; the two primary
  endpoints are Holm-corrected together.
- **Verdicts** (`evaluation/verdicts.py`, written to `docs/results/VERDICTS.md`):
  H1 supported if every arm's CI of (relative drop in small-finding recall minus relative drop
  in the headline accuracy, both at the operating point) is above 0, not supported if any is
  below 0, over the 3-seed arms of the main tier (E7 and the comparator at B2; confirmatory);
  the same rows of every seed-0 arm (`<arm>_vs_E2_seed0`, against E2 seed 0) are exploratory
  and not part of the verdict. H2 supported if the CI
  of (best pruning arm minus E3 `w8`) small-finding recall is below 0, not supported if above 0;
  **H2 is confirmatory** (decided 2026-11-08): E3 `w8` joins the 3-seed tier (seeds 0, 1, 2;
  each E2 seed quantized deterministically; about 5.4 GPU h more), compared with the best
  pruning arm at the budget nearest E3 `w8`'s measured peak (after RC-B0). If that budget is
  not B2, that arm needs seeds 1 and 2 too: the comparison step **flags it**
  (`choices.json` `h2.flag`), and until they exist H2 is exploratory
  (`best_pruning_vs_E3:w8_seed0`). E3 `w4` stays seed 0.
  H3 supported if both primary differences are above 0 with Holm-adjusted p < 0.05, not
  supported if either CI lies at or below 0; after a futility stop H3 is exploratory
  (`E7_vs_best_seed0`). **H4 is exploratory** (amendment of 2026-11-08): the Kendall taus
  between the accuracy ranking and each clinical ranking (both at the operating point), over
  seed 0 of every compressed arm at B2 and without a budget, are reported with their
  image-bootstrap CIs and no confirmatory claim. Otherwise inconclusive. The small-finding
  recall of H2 and H3 is at the operating point too.
  The verdicts are reported for each of the three tests (D02 amendment 7, with the strict
  subset dropped on 2026-10-29): the filtered test, the external test and the unfiltered
  test. **How the two co-primary tests combine** (decided 2026-11-08): a hypothesis is
  **supported** only if it is supported on both the internal filtered test and the external
  test (intersection-union: each test at the full alpha, no alpha split); **partly supported**
  if supported on one; reported per test either way (`verdicts.combine_tests`). The unfiltered
  test is a sensitivity analysis and does not enter the combination.
- **Comparison step** (`scripts/09_compare_arms.py`, `evaluation/compare_stage.py`): one
  sequence of image-level resamples (seed 0) is shared by every arm and every comparison. A
  main-tier arm is compared on its 3 seeds; with a seed missing it is read at seed 0 and
  reported as exploratory; an arm with 2 seeds is refused; every other arm is read at seed 0.
  - H1 for every arm of E3-E8 (each variant and budget): the relative drop against E2 of the
    small-finding recall minus that of the headline accuracy, at the operating point. The 3-seed
    arms of the main tier against the 3-seed E2 (confirmatory); every seed-0 arm against E2
    seed 0, reported apart (`_seed0`, exploratory). (The same with the rare-group
    recall is not computed: no rare group, 2026-10-31.)
  - H2 (decided 2026-11-08): E3 `w8` is compared at the budget whose target is nearest to its
    measured peak memory, against the generic pruning arm (E5 criteria, E6) that comes first at
    that budget by the comparator's ordering (section 5; validation, seed 0); both on 3 seeds
    (above).
  - H4 (exploratory): seed 0 of the compressed students E3-E8 at B2 and without a budget,
    ranked by the headline accuracy and by
    small-finding recall and ECE (negated, so higher is better in both); the rare-group ranking
    is dropped with the rare group (2026-10-31).
  - McNemar per finding (decided 2026-11-08): on each probe's majority answer over the 3 seeds
    (right in at least 2) at the operating point, for the 3-seed arms (E7 against the
    comparator, H2, E7 against A1); for a seed-0 arm the answer of seed 0. Never on 2 seeds.
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
| Recovery (E5-E7, A1-A6) | the same adapter, optimiser and loss; **157 updates on the first 2,500 examples of the fixed pool** (25 % of the E2 examples, `ceil(2,500 / 16)` as the training steps; decided 2026-11-08); identical for every pruning arm (fingerprint logged): **the identical recovery is the fairness rule**, so the arms differ only in what they remove | D06; section 11 item 14 |
| Pruning | units: decoder layers, KV groups, MLP channel groups of 128; score per parameter; first 2 and last 1 layers protected; 128 calibration images; one-shot removal; 1 negative per positive; signed effects; probe batch 4; how a unit is switched off while scored: **exact** for decoder layers and KV groups (switched off and scored again), **first-order** for MLP channel groups (the gate derivative; basis: Molchanov et al., CVPR 2019, a correlation above 93 % between the Taylor estimate and the true importance), decided 2026-11-08; RC-G2's check on 30 channel groups is a **report**: below a rank correlation of 0.7 the agent stops and asks the team; channel groups are never scored exactly as a fallback (about 236 h per seed) | D07 |
| Quantization | E3 `w4`, `w8`: GPTQ, groups of 128, 128 calibration images, decoder linear layers only; `w8a8`: LLM.int8; E8, E9: the `w4` recipe; kernel class and versions from RC-Q0 | D08 |
| E4 | 4-bit learned step sizes (groups of 128), LoRA through the rounding, GRACE's confidence-gated decoupled distillation on the cached top-20 (alpha 1, beta 4, T 2) | D08 |
| Ablations (B2, seed 0 only, exploratory; amendment of 2026-11-08; A1 also seeds 1 and 2 for the secondary comparison, item 13) | A1 weights 1; A2 rarity only; A3 random order over E7's units, ranked **without** the per-parameter division (decided 2026-11-08; with it, whole layers would go first: an estimate from the student's unit sizes gives about 98 % of the removed parameters at a 30 % cut, so A3 would be random layer removal; without it about half); A4 32 / 64 / 128 / 256 nested calibration images; A5 KL effect (scored exactly; if D07 section 1 uses the first-order estimate for some units, both A5 variants run on the exactly scored unit types only); A6 layers only / KV and channel groups only | S10, iteration 42 |
| Reuse of E7 | decided 2026-11-08: `A4:n128` and `A5:logprob_drop` have exactly E7's settings (a test checks it), so E7's B2 runs are reported for them, and the methods say so (fewer runs). **A1 and A2 re-weight E7's per-finding effects** of the same seed (D07; implemented 2026-11-08, `pruning/stage.py`): the importance is linear in the weights, so no scoring of their own | iteration 42; item 13 |

## 8. Deviations from the methods we follow (declared in advance)

- E6 scores layers once and removes them in one shot like every other arm; the method it
  follows removes layers iteratively (D07).
- E4 is a reduced GRACE: no relational alignment of visual tokens, no adaptive loss weight, no
  full fine-tuning; the gate and the decoupled terms use the cached top-20 (D08).
- `w8a8` is LLM.int8 (dynamic, 16-bit outliers), not a calibrated static W8A8 (D08).

## 9. Freeze procedure

1. The agent fills every **[RC-D1]**, **[RC-S0]** and **[RC-B0]** item from the committed
   numbers by the rules above.
2. The team decides every **[team]** item (the tiers from the finding names only): all decided
   on 2026-11-08 (section 11 item 14).
3. **Soheil and Amir approve** (the authors; Farzan left the project on 2026-11-08 and is not an
   author); the agent writes the date, the commit hash and the approvers above and into
   `configs/preregistration/finding_weights.yaml` (`frozen_on`, `frozen_by`, `commit`), and
   logs the freeze in `docs/decisions_log.md`.
4. **Done on 2026-11-08** at commit `18f1b47`, approved by Soheil and Amir (Amir's written
   approval received). Accepted at the freeze, the three items the agent had decided: the C4
   sensitivity analysis (section 11 item 14.10), the estimate of about 239-246 GPU h
   (`docs/schedule.md`), and Soheil starting account 3's sessions (`docs/team.md`).

## 10. Deviations after the freeze

The document was frozen on 2026-11-08 at commit `18f1b47`. From now on **every change needs a
dated amendment here** (date, what changes, why, approved by both authors) and an entry in
`docs/decisions_log.md`; the freeze record itself is the only edit made at the freeze.

**Amendment of 2026-11-13 (Soheil and Amir; before any compression result).** Calibration images whose fp16 forward through the model gives any non-finite value are replaced by the next eligible image in the same seeded order, keeping the nested subsets (A4) and the finding stratification. The rule is applied once per seed, before any criterion or quantization, identically for every arm that uses the calibration set (pruning criteria, GPTQ, E4, E8, E11/E12). The replaced ids are listed in a committed file. *Why:* RC-E3 seed 0 stopped on a NaN Hessian; 2 of its 128 calibration images give a non-finite vision tower output in fp16 (the frozen Qwen2.5-VL vision encoder overflows on them; their pixel values are finite; `scripts/check_gptq_inputs.py`, Amir, commit 9113632). The saved validation outcomes of E2 seed 0 and E1 seed 2 have no NaN margin (681 each). No other setting changes; the vision tower stays fp16. *How:* `scripts/check_calibration.py` checks each candidate with the fp16 forward of the model being calibrated (E2 of the seed for the student arms, the teacher for E9) and writes `results/calibration/seed<k>_replaced.csv`; every stage that draws calibration images reads the cleaned set (`data/calibration.py`) and refuses to run without it. E4 trains on the teacher cache and draws no calibration set, so the rule does not change it.

**Amendment of 2026-11-13, second (Soheil and Amir; before any test evaluation and before the
RC-CAL0 scan results).** Any test image (the filtered or the unfiltered internal test, and later
PolypGen) whose fp16 vision tower output is non-finite for the student **or** the teacher is
dropped from that test for **every** arm before any test evaluation; the list is committed and its
count is reported next to the test sizes. *Why:* the fp16 vision encoder overflows on some images
(the amendment above); a non-finite probe has no defined answer. No student arm changes the vision
tower, so the student list holds for every student arm, and the union with the teacher's keeps one
test set for all arms. Dropping is decided before any result, from the encoders alone (no model
output of any arm is read). Scoring such an answer as wrong would treat the students and the
teacher differently (different encoders); running the encoder in fp32 for those images would change
the numerics the study measures. *How:* `scripts/scan_vision_finite.py` (RC-CAL0) finds the images;
the agent commits the list; the test evaluation leaves those images out (code before RC-R1).

**Clarification of 2026-11-13 (Soheil and Amir; before any compression result).** A5
`logprob_drop` runs on layers and KV groups only (exactly scored), as section 7 and D07 already
state for both A5 variants, as **its own B2 run on seed 0**. The reuse of E7 for A5
`logprob_drop` (freeze item 12, section 7 row "Reuse of E7") assumed identical settings, which is
false: E7 also scores and removes first-order channel groups. That reuse is withdrawn for A5 only;
`A4:n128` stays E7. A5 `logprob_drop` takes E7's exact scores of the layers and KV groups of the
same seed (an exact score does not depend on which other units are scored), so it needs no
scoring of its own; its removal, recovery and evaluation are its own (about 3 GPU h plus a budget
search, an estimate).

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
   - an item flips when the arm's answer differs from E0's answer: the majority over its 3
     seeds for a main-tier arm, seed 0's answer for every other arm (amendment of 2026-11-08);
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

9. **Per-finding AUROC and "yes" rate** (Soheil, 2026-11-02; added after E0's validation was
   seen and **before any student or compressed result**). E0's validation showed a very
   conservative teacher: specificity about 1.0, sensitivity 0.68 (polyp), 0.46 (ulcerative
   colitis), 0.70 (oesophagitis). Sensitivity and specificity depend on the yes/no threshold,
   so a compression method that only shifts the bias cannot be told from one that loses
   discrimination. So, for **every model** (E0-E9, A1-A6) and every split, a secondary metric:
   - `auroc:<finding>`: the AUROC of the probe score (log P(yes) - log P(no), saved for every
     probe) against the label, with the image-level percentile bootstrap 95 % interval (10,000
     resamples, the run's seed, as for the other clinical metrics); left out when a finding has
     only one class;
   - `yes_rate:<finding>`: the share of that finding's probes answered "yes"; with
     `n_positives:<finding>` and `n_negatives:<finding>` to read it against;
   - reported next to sensitivity and specificity (`appendix_auroc.tex`,
     `appendix_yes_rate.tex`). Reading: the same AUROC with another "yes" rate is a shift of
     the bias; a lower AUROC is lost discrimination;
   - descriptive: not part of H1-H4, no verdict, no multiplicity correction;
   - no new GPU run: every later evaluation writes them itself. E0's validation rows could not
     be recomputed (2026-11-04): its saved outcomes were removed with the repository clone at
     the end of RC-E0a's session. They come with the next E0 evaluation (RC-R1 on the test
     splits; validation only if E0 is evaluated there again). `scripts/rescore_probes.py`
     stays for runs whose outcomes are kept.

10. **Probe operating point** (Soheil, 2026-11-06; decided after seeing E0 and E1 on
    validation, **before any compression result and before the freeze**). The students answer
    "yes" very rarely (E1 sensitivity: ulcerative colitis 0-6 %, oesophagitis 0-5 %) while
    their scores still rank the probes well (AUROC: polyp about 0.995, oesophagitis 0.92-0.93;
    ulcerative colitis only 0.58-0.67). The plain yes/no answer therefore mostly measures each
    model's bias, not what it can tell apart.
    - For each model (every run: each arm, variant, budget and seed) and each finding, a
      threshold on the probe score (log P(yes) - log P(no)) is fixed on **validation** at
      **specificity 0.95** on that finding's validation negatives: the k-th smallest negative
      score, k = ceil(0.95 x negatives); "yes" when the score is above it
      (`probe_suites.fit_thresholds`, `op_threshold:<f>`).
    - The thresholds are applied **unchanged** to every test (filtered, unfiltered, external):
      a test evaluation reads them from the run's committed validation rows and refuses to run
      without them.
    - Sensitivity, the small-finding recall (the clinical primary endpoint), the headline
      balanced accuracy, the flip analysis and the McNemar answers use this operating point
      (`op_*` rows, `probe_yes_op`); the AUROC is reported next to it; the plain yes/no metrics
      stay, as secondary. On validation the specificity is 0.95 by construction, so the
      validation op rows are descriptive; the comparisons are on the tests.
    - The lesion-mask effect (primary endpoint 2) and Gate 1 keep the plain answers and the
      margin effect, as pre-registered.
    - E0 and E1 get the new rows from their next evaluation: E0 needs its validation pass again
      (its saved outcomes were lost, 2026-11-04; about 0.8 h on T4 x2), E1 from its saved
      validation outcomes on CPU if they were kept (`scripts/rescore_probes.py`), else a
      validation pass per seed (about 0.5 h each). No training is repeated.
11. **Seeds per budget** (Soheil, 2026-11-06, before the freeze, to save GPU hours): 3 seeds
    (0, 1, 2) at the primary budget B2 for every compressed arm (and E3, E4 without a budget);
    **seed 0 only at B1 and B3**, reported as the trend and stated so in the paper. E1 and E2
    keep 3 seeds. The trend arms are compared with E2 seed 0, and they are not part of the
    verdicts (H1, H4). **Replaced by item 12** (2026-11-08).
12. **Seed plan** (pre-freeze amendment, **approved by Soheil and Amir on 2026-11-08**; written
    before any compression result, seed 0 included; `medcompress.seed_plan`).
    1. Main-claim tier, 3 seeds (0, 1, 2): E2, E7 at B2 and the comparator at B2.
    2. Everything else, seed 0 only, reported as exploratory / descriptive: the other E5 / E6
       arms, all E3 variants (`w8`, `w4`, `w8a8`), E4, E8, E9, every arm at B1 and B3, the
       ablations A1-A6 and the training-length check. (Item 14 since moved E3 `w8` into the
       3-seed tier for H2; E3 `w4` stays seed 0, decided 2026-11-08.)
    3. Never 2 seeds for any arm: the majority McNemar answer uses hits > 0.5, so a 1-1 split
       counts as wrong.
    4. Comparator: chosen on validation, seed 0, at B2, among E5:magnitude, E5:activation,
       E5:taylor and E6, by (i) `op_small_finding_recall`; (ii) ties: the operating-point
       recall over all mask-positive polyps (86 on validation); (iii) ties: the mean, over the
       small-polyp probes, of (probe margin minus that arm's own `op_threshold:polyp`); (iv)
       then the name. Reason: E1 and E2 both score 13 of 13 on the validation small-polyp
       recall, so (i) alone can tie at the ceiling and leave the choice to the name. Only the
       chosen arm runs seeds 1 and 2. Replaces "averaged over its 3 seeds" (section 5). The
       evaluation writes (ii) and (iii) as `op_mask_positive_recall` and
       `op_small_margin_mean` from the saved margins and the run's own thresholds;
       `scripts/choose_comparator.py` applies the ordering to the committed rows and writes
       `results/seed_plan/comparator_B2.json`.
    5. Stopping rule (futility only, written before any seed-0 compression result): after seed
       0, seeds 1 and 2 of the main tier run unless E7 at B2 ranks **below** the comparator by
       the ordering of item 4 (validation, seed 0). If it ranks below: stop, report seed 0 as
       exploratory, with no confirmatory claim. Otherwise seeds 1 and 2 run, seed 0 stays in
       the final analysis and the test is unchanged. A rule that can only stop for futility
       removes chances to claim an effect and never adds one, so it does not raise the
       false-positive rate. (A full tie on (i)-(iii) goes to the name, where E7 comes after
       every candidate, so it stops; written as the rule reads.)
    6. Single-seed arms are read against a yardstick: the spread of E7 at B2 across its 3
       seeds. A single-seed difference smaller than that spread is reported as within
       run-to-run noise.
    7. Hypotheses: H1-H3 (E7 against E2 and against the comparator, the primary endpoints)
       stay confirmatory on the 3-seed tier. H4 (the ranking of E3-E8) becomes exploratory, on
       seed 0, with image-bootstrap CIs and no confirmatory claim. (H2 stays confirmatory:
       E3 `w8` joined the 3-seed tier on 2026-11-08, item 14.)

    **Rationale** (Soheil's analysis and study doc, 2026-11-08):
    - E1's 3 seeds at the **default** cutoff: the small-polyp recall spread (SD 0.118 on 13
      polyps) is about what the 13 images alone produce (about 0.126). The AUROC spread is far
      below the sampling SE (polyp 0.0004 against 0.0048; oesophagitis 0.007 against 0.030;
      ulcerative colitis 0.049 against 0.039). The yes/no rate varies more (SD 0.104 against
      0.055), which the per-model operating point re-fits. The operating-point seed spread is
      pending E1 seeds 1 and 2's rescoring; seed 0 is at the ceiling (13 of 13).
    - Power simulation: more small polyps (the external test) does more than more seeds. One
      seed at 100 small polyps detects about 18-22 points; 3 seeds at 21 polyps about 22-23.
    - Sources: the NeurIPS paper checklist (answering "no" to the statistical-significance item
      with a compute justification is acceptable), Bouthillier et al. 2021, Jordan (ICLR 2024),
      Varoquaux and Cheplygina 2022, Hooker et al. 2020.
    - GPU cost (estimates; the recovery hours assume the distillation loop's measured 27.2 s
      per update): the extra seeds of the main tier are about 38 h (per seed: E2 5.2 h
      measured, E7 about 10.8 h, the comparator about 2.9 h). The whole plan is about 190 h,
      against about 360 h or more for 3 seeds everywhere at B2.
    - Agent's check against the committed E1 rows (2026-11-08): the SDs reproduce (small-polyp
      recall 0.1175; AUROC polyp 0.0004, oesophagitis 0.0074, ulcerative colitis 0.0491). The
      sampling SEs taken from the committed bootstrap intervals (width / 3.92) are 0.131-0.145
      for the small-polyp recall, 0.0023-0.0029 (polyp), 0.021-0.024 (oesophagitis) and
      0.037-0.038 (ulcerative colitis); the binomial SE at E1's mean recall (0.59, 13 polyps)
      is 0.136. Which "yes/no rate" the 0.104 refers to is not stated (the `yes_rate:polyp` SD
      is 0.048, `original_yes_rate` 0.115). The conclusion (the seed spread is within the
      sampling noise, except the ulcerative colitis AUROC) is the same either way.

13. **Pre-freeze additions** (approved by Soheil and Amir on 2026-11-08, delegated to the
    research lead; before any compression result).
    1. **Correction to the freeze list, item 2:** "if the check fails, channel groups are scored
       exactly" is dropped (about 236 h per seed, not feasible). D07's rule stays: RC-G2's
       check on 30 channel groups is a report; if its rank correlation with the estimate is
       below 0.7, the agent stops and asks the team. The basis of the first-order scoring of
       channel groups: Molchanov et al., CVPR 2019, report a correlation above 93 % between
       the Taylor estimate and the true importance.
    2. **A1 joins the 3-seed tier at B2** as a **pre-specified secondary** comparison, E7
       against A1 (section 5). Not part of the confirmatory verdicts. It re-weights E7's
       per-finding effects: recovery and evaluation only, about 6 GPU h for seeds 1 and 2 (an
       estimate). It runs under the same futility rule as the main tier.
    3. **E10, an exploratory baseline:** a natively small model at matched memory.
       Qwen3-VL-2B (the D05 fallback), trained once with the E2 recipe, seed 0, then evaluated
       like every arm. First, its tokenizer is checked against the teacher cache's top-20
       token ids (`scripts/check_tokenizer_match.py`: every id the cache uses maps to the same
       token, and at least 99 % of the cached answers are re-encoded exactly). If it fails, it
       is trained with the E1 recipe (answers only) and the paper says so. Compared at matched
       peak memory with the pruned 3B arm nearest its footprint (after RC-B0), and at 4 bit
       against E8. Run card RC-E10 (about 4 GPU h, an estimate). Descriptive, no verdict.
    4. **Positioning** (`docs/related_work.md`): the closest prior work, what is new here, and
       which methods are left out and why. The generic arms are re-implementations in one
       framework with identical recovery, named as such (magnitude, Taylor in the
       LLM-Pruner style, activation, and E6 in the ShortGPT / Khan et al. style), never as the
       original methods. A1 and A2 are the in-framework analogues of task-aware and
       long-tail-aware (LTAP-style) weighting.

14. **Pre-freeze decisions** (approved by Soheil and Amir on 2026-11-08; Soheil delegated the
    choices to the research lead and both approve; Farzan is not an author). Before any
    compression result. Each with its reason:
    1. **Finding weights** (section 3): `w_c = min(1 / prevalence_c, cap) * tier`, cap 20,
       normalisation none; tiers by rule from the finding names only (action 2.0: treated in
       the same examination, e.g. resection; follow_up 1.5: changes treatment or follow-up,
       not treated in the examination; other 1.0): polyp action, oesophagitis follow_up,
       ulcerative colitis follow_up. Reason: an explainable clinical priority fixed before any
       compression result. A1 and A2 are its pre-registered check.
    2. **Scoring method** (D07 section 1): exact for layers and KV groups, first-order for
       channel groups; RC-G2's 30-group check stays a report (below 0.7: stop and ask the
       team). Reason: about 7.9 h per seed, against about 236 h for exact scoring of every
       unit (estimates); Molchanov et al., CVPR 2019 support the first-order estimate.
    3. **Recovery:** 157 updates on the first 2,500 examples of the fixed pool, identical for
       every pruning arm. Reason: the identical recovery is the fairness rule; the same rounding
       as the training steps.
    4. **A3:** random scores not divided by the unit size (otherwise whole layers go first).
       **Reuse:** E7 stands for `A4:n128` and `A5:logprob_drop` (identical settings), stated in
       the methods.
    5. **Headline accuracy** for H1 and H4: the macro balanced accuracy at the operating point
       (already used). **McNemar:** per finding, on the majority answer over the 3 seeds for
       3-seed arms. Reason: with few positives the overall accuracy measures specificity.
    6. **H2 stays confirmatory:** E3 `w8` joins the 3-seed tier (seeds 0, 1, 2; deterministic
       quantization of each E2 seed; about 5.4 GPU h more); the matched pruning arm is the best
       pruning arm at the budget nearest E3 `w8`'s measured peak (after RC-B0); if that budget
       is not B2, that arm needs seeds 1 and 2 too, and the comparison step flags it. E3 `w4`:
       seed 0 only. Reason: H2 is a headline question and needs the same seeds as H1 and H3.
    7. **The two co-primary tests:** a hypothesis is supported only if supported on both the
       internal filtered test and the external test (intersection-union, no alpha split);
       partly supported if supported on one; reported per test either way. Reason: the
       internal test is reduced-leakage only, so a claim must also hold at another centre.
    8. **Budgets** (section 4): the fractions are frozen now (15 / 30 / 45 % of the decoder
       bytes removed); `budgets.measurement.hardware` = "Kaggle Tesla T4"; RC-B0 only converts
       the fractions to GB. The timing GPU (RC-T1) stays open and does not block the freeze.
       Reason: the rule does not depend on a measurement, only its conversion does.
    9. **Secondary analyses** (section 5; not in the confirmatory verdicts): (a) the
       operating-point recall over all mask-positive polyps; (b) the headline accuracy without
       ulcerative colitis (E2 seed 0: operating-point sensitivity 0.27); (c) E0 against E2,
       small-finding recall and flips (what distillation itself forgets).
    10. **PolypGen** (RC-X2, co-primary): licence read in Ali et al., Sci Data 2023: CC BY 4.0,
        research use allowed, citation and acknowledgement required (the paper's commercial-use
        wording is inconsistent; irrelevant for academic use); confirming the Synapse page
        terms is Amir's step at the download and does not block the freeze. The loader and the
        overlap check against our Kvasir images (centre C4 is Oslo) are written before any
        download is used (`data/polypgen.py`, `scripts/polypgen.py`); the bootstrap clusters by
        video / sequence, not by frame; the small-polyp threshold is our training q1, unchanged.
        Agent's addition, recorded before any result and accepted by Soheil and Amir at the
        freeze: C4 is reported apart, and the external test without C4 is a sensitivity
        analysis.
    11. **E1 seed 1** (final): its outputs were lost; it has no operating-point rows; E1's test
        evaluation (RC-R1) uses seed 0 only. E1 is a control outside the 3-seed tier; seed 1
        keeps its committed default-cutoff validation rows. Not waiting on anyone.
    12. **Team and approval:** section 9's freeze approval is Soheil and Amir; Farzan left the
        project. GPU quota: three Kaggle accounts x 30 h = 90 h per week (Soheil, Amir, and one
        lent by Farzan); every hand-off dataset is shared to Soheil's account.
    13. **Freeze-list item 19** (confirmed as this document states): A1 runs at B2 only; A6
        `heads_channels` holds no whole layers. Already settled: the fixed training pool
        (iteration 36), the finding list and no rare group (item 1-2 of the 2026-10-31
        amendment), the comparator (item 12), the headline accuracy (item 5 above).
    14. **Exploratory arms E11 and E12** (docs only; section 12).

## 12. Exploratory arms not yet built (E11, E12)

Documented 2026-11-08 (section 11 item 14.14); no code, config, registry entry or run card yet.

- **E11, clinical mixed-precision quantization** (exploratory, seed 0, from E2): for each decoder
  layer, quantize that layer alone to 4 bit and measure the clinically weighted drop in the
  correct answer (the same probes, weights and calibration images as E7). Keep the most
  sensitive layers at 8 bit and the rest at 4 bit, until the memory target is reached: the
  midpoint between E3 `w4` and E3 `w8`, measured like the budgets.
- **E12, its generic control:** the same procedure with an unweighted sensitivity.
- **Compared:** E11 against E12, and both against E3 `w4` and `w8`, at measured memory; no
  confirmatory claim.
- **Condition:** both run only if a later feasibility check shows that our evaluation stack
  (transformers, the Kaggle T4, the same evaluation and timing code) runs per-layer bit widths
  with real low-precision kernels (D08 notes that GPTQModel's "dynamic" per-layer setting may
  not be supported in transformers). If not, they are reported as "not run" and listed as
  future work.
- **Background:** HAQ and HAWQ (mixed-precision quantization, 2019).

**Frozen on 2026-11-08** (section 9). Open outside the document, none of it part of the
pre-registration: the timing GPU (RC-T1), and code the plan needs before its runs (the
external-test evaluation split, E10's 4-bit run and comparisons). The test split is open from
the freeze on; it is read only by the steps this document names.
