# Pre-registration

Fill in every section, review as a team, then record the freeze below **before any
compression run (E3-E9, A1-A6)**. After freezing, any change needs approval from all
three members and a dated entry in `docs/decisions_log.md`.

**Frozen on:** ____ &nbsp;&nbsp; **Commit:** ____ &nbsp;&nbsp; **Approved by:** ____

## 1. Domain, model pair and data

- Domain: TODO
- Teacher / student: TODO (tokenizer equality verified: TODO)
- Datasets and roles: TODO
- Split files and hashes: TODO
- External test set and deduplication result: TODO

## 2. Findings

- Finding list C: TODO
- Prevalence groups (common / medium / rare) and the rule used to form them: TODO
- Lesion-size strata thresholds (quartiles of lesion area): TODO
- Probe template and display names: TODO

## 3. Finding weights

Table in `configs/preregistration/finding_weights.yaml`:
`w_c = min(1 / prevalence_c, cap) * tier_multiplier(tier_c)`.

| finding | prevalence | importance tier | cap | final weight |
|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO |

Tier definitions and clinical rationale: TODO. Normalisation: TODO.

## 4. Budgets

- Metric (peak memory / latency): TODO
- Measurement protocol (hardware, batch size, resolution, prompt length, max new
  tokens, warm-up, repeats, sample list): TODO
- B1 / B2 / B3 target fractions of E2: TODO / TODO / TODO
- Tolerance: TODO

## 5. Endpoints

**Primary** (at B2, E7 vs best of E5 and E6):

1. Small-finding recall (smallest lesion-area quartile).
2. Lesion-mask effect: flip rate with lesion masked minus flip rate with an
   area-matched random region masked.

Rule for choosing "best of E5 and E6": TODO.

**Secondary:** per-finding sensitivity/specificity, macro-F1, overall accuracy,
rare-finding recall by prevalence group, blank-image and other-patient tests, ECE,
Brier score, risk-coverage, hallucination / omission / negation rates, efficiency.

## 6. Analysis plan

- Paired bootstrap 95% CIs over test images (resamples: TODO).
- McNemar test per finding; Holm correction across findings.
- Holm correction across the two primary endpoints.
- Mean and SD over 3 seeds for training arms; how seeds enter the bootstrap: TODO.
- Answer parsing rules and handling of unparseable answers: TODO.
- Exclusion criteria (e.g. failed runs, kernels not running): TODO.

**Metric definitions as implemented (proposals until frozen).** The evaluation stage
(`evaluation/runner.py`, `evaluation/probe_suites.py`) computes the following; the
freeze either accepts them or replaces them before the first compression run.

- Probe set: one yes/no probe per labelled (image, finding) pair of the split; an image
  with several QA rows counts once per finding. Scored by one forward pass (log
  probability of "yes" against "no"); answers are never unparseable. The same probe set
  is used for every arm and seed (optional cap per finding, same for all).
- Overall accuracy: fraction of probes answered correctly. Macro-F1: unweighted mean of
  per-finding F1 (undefined F1 left out). Sensitivity and specificity per finding.
- Small-finding recall: recall of positive probes of the findings that have lesion masks
  (`image_reliance_check.mask_findings`), in the smallest lesion-area stratum; area cut
  points taken from the training split and frozen in `configs/domain/endoscopy.yaml`.
- Rare-finding recall: positives pooled within each prevalence group
  (`evaluation.rare_finding_pooling`).
- Calibration: confidence is the larger of P(yes) and P(no) renormalised over the two
  answers; ECE with equal-width bins (`evaluation.calibration_bins`), Brier score on
  P(yes), AURC from the risk-coverage curve by confidence.
- Confidence intervals: percentile bootstrap over probes (`evaluation.bootstrap_resamples`
  resamples, level `evaluation.ci_level`); an interval that does not contain its point
  estimate is widened to contain it.
- Free-text suites: a seeded sample of the split's QA rows (`evaluation.generation_samples`,
  the same rows for every arm), greedy generation with `evaluation.max_new_tokens`.
- Test split: `evaluation.split=test` is refused until `preregistration.frozen_on` is set.

## 7. Deviations

List any deviation from this document after the freeze, with date and reason.
