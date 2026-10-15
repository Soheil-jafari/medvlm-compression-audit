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
| Teacher | Qwen2.5-VL-7B-Instruct, revision `cc594898`, with the SimulaMet Kvasir-VQA-x1 LoRA, revision `79e77755`, merged | D01 |
| Student | Qwen2.5-VL-3B-Instruct; revision pinned from `sha_seen` `66285546` once **[RC-S0]** passes; fallback Qwen3-VL-2B-Instruct | D05 amendment |
| Tokenizer equality | `tokenizers_identical = 1` in `results/smoke/RC-M1.csv` | RC-M1 |
| Main data | Kvasir-VQA-x1, revision `61e41148`: official train file = training; official test split halved **by image** into validation and test with seed 20261012 | D02, RC-D1 |
| Near-duplicates | validation / test images within 4 of 64 dHash bits of a training image are removed (count: **[RC-D1]** `duplicates_removed:val_test`) | RC-D1 |
| Masks | Kvasir-SEG, linked to VQA images within 4 dHash bits; ambiguous links dropped (counts **[RC-D1]**) | D02, RC-D1 |
| Split hash | **[RC-D1]** `split_sha256` (printed by `scripts/prepare_data.py`) | RC-D1 |
| External test | EndoBench, deduplicated against training images; licence still to be verified by Farzan (TODO: verify) | D02 |
| Training pool | one draw of `training.data_subset` training samples with seed 20261012 (`offline_kd.training_pool`), shared by E1, E2, E4 and every recovery (its first 25 %); the teacher cache holds exactly this pool. **[team]** confirm (proposed in iteration 36) | iteration 36 |

## 2. Findings

| Item | Rule | Value |
|---|---|---|
| Finding list C | every answer value of the source question "are there any abnormalities in the image" (label rule `abnormalities`, `configs/data/kvasir_vqa_x1.yaml`) with at least **20** positive training images and at least **10** positive images in each of validation and test. **[team]** confirm the two numbers. Instruments and landmarks are control questions, not findings (D02) | **[RC-D1]** from `answer_count:*` and `positives:<f>:<split>` |
| Display names | the answer value as written in the source answers, lower case | **[RC-D1]** |
| Prevalence | positive training images / training images (`prevalence:<f>`) | **[RC-D1]** |
| Prevalence groups | common: prevalence >= 10 %; rare: < 2 %; medium: in between. **[team]** confirm the cut points (the alternative is tertiles of the prevalences of C) | **[RC-D1]** |
| Mask findings | the findings Kvasir-SEG delineates: `polyp` (`image_reliance_check.mask_findings`) | D02 |
| Lesion-size strata | quartiles of the lesion area of training images with a linked mask (`lesion_area_threshold:q1..q3`); "small" = below q1 | **[RC-D1]** |
| Probe | "Is there a {finding} in this image? Answer yes or no.", one forward pass, log P(yes) against log P(no) | D02, S04 |

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
| Protocol | batch 1, free generation with `evaluation.max_new_tokens`, 3 warm-up and 5 timed passes, 32 validation images drawn with seed 0, the same for every arm | D07 |
| Reference | E2 seed 0, measured twice in fresh processes (RC-B0) | D07, RC-B0 |
| B1 / B2 / B3 | remove about 15 / 30 / 45 % of the decoder's bytes, written as fractions of E2's measured peak: **[RC-B0]** | D07 section 4 |
| Timing hardware **[team]** | the rented GPU type (final numbers); development numbers on the Kaggle T4 are labelled as such | D07 |

## 5. Endpoints

**Primary** (at B2, test split, 3 seeds):

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
rare-finding recall by prevalence group (positives pooled within a group), blank-image accuracy
drop and other-image tests, ECE (15 equal-width bins), Brier score, AURC, hallucination /
omission / negation rates on 1,000 free-text answers, exact match and token F1, efficiency
(parameters, file size, peak memory, cold and warm latency, throughput, energy where measurable,
kernel check).

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
- **Comparison step** (`scripts/09_compare_arms.py`, `evaluation/compare_stage.py`): one
  sequence of image-level resamples (seed 0) is shared by every arm and every comparison. An
  arm with a missing seed is left out and listed, never compared on fewer seeds.
  - H1 for every arm of E3-E8 (each variant and budget): the relative drop against E2 of the
    small-finding recall minus that of the headline accuracy; the same with the rare-group
    recall is reported next to it.
  - H2 **[team]** proposal: E3 `w8` is compared at the budget whose target is nearest to its
    measured peak memory, against the generic pruning arm (E5 criteria, E6) with the highest
    validation small-finding recall at that budget (mean of 3 seeds).
  - H4: the compressed students E3-E8 at every budget, ranked by the headline accuracy and by
    small-finding recall, rare-group recall and ECE (negated, so higher is better in both).
  - McNemar per finding **[team]** proposal: on each probe's majority answer over the 3 seeds
    (right in at least 2), for E7 against the comparator and for H2.
- **Answer parsing:** probes are scored, never parsed, so they cannot be unparseable. Free-text
  answers count as "yes"/"no" only if they start with that word after leading punctuation
  (`clinical.parse_yes_no`); anything else is an error.
- **Exclusions:** a quantized run that fails the low-precision kernel check is not reported as
  quantized; a budget-matched run outside the 3 % tolerance is reported, marked, and the primary
  comparison is repeated without such runs as a sensitivity analysis; a run that crashes is
  rerun once with the same seed and otherwise reported as failed.
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
| Training (E1, E2, E4) | LoRA rank 16, alpha 32, dropout 0.05 on all decoder projections; AdamW 1e-4, linear decay, 20 warm-up steps, 4 x 4 examples per update, KD weight 0.5, temperature 2; one pass over the pool: 10,000 examples, 625 steps (decided 2026-10-16); the training-length check adds E2 and E7 at 30,000 examples (section 6) | D06 |
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
