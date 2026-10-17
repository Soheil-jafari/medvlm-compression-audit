# D04: Gate 1 (does the teacher really look at the image?) and the teacher cache

**Status:** answered 2026-10-12 (Soheil): moderate Gate 1 rule, mid-grey fill, `top_k` 20.
The `max_new_tokens` rule stays a proposal until the RC-D1 counts arrive. **Stage:** S05. **Prepared:** 2026-10-12.
**Depends on:** D01 (teacher), D02 (data). The numbers below are set **before** any
result exists, so they cannot be tuned to a result.

## Question

Before the teacher's answers are cached and every later experiment starts from it,
the team must fix (1) the pass/fail rule of Gate 1, (2) the grey value used to hide
image parts, and (3) the size of the teacher cache. All of them are fixed now so that
no result can influence them.

## What Gate 1 is, in plain words

Everything we measure later (what compression "forgets") only means something if the
teacher actually uses the picture. A model can answer "is there a polyp?" from the
words alone (for example by always saying "no"). If the teacher does that, then
pruning cannot hurt its "vision", and the whole study would be measuring nothing.

Gate 1 checks this on the **validation** half only, with three tests:

1. **Is the teacher better than chance on real images?** Balanced accuracy of the
   yes/no finding probes (average of "finds the positives" and "clears the
   negatives", averaged over findings), so a model that always says "no" scores 0.5.
2. **Does blanking the image hurt?** Same questions, but the image is replaced by a
   plain grey picture. A model that looks at the image must get clearly worse.
3. **Does hiding the lesion change the answer?** For images where a lesion mask
   exists (Kvasir-SEG polyps), grey out the lesion, and separately grey out a
   region of exactly the same shape somewhere else that does not touch the lesion
   (the control). The **lesion-mask effect** is: how often the answer flips when the
   lesion is hidden, minus how often it flips for the control. A model that looks at
   the lesion has a clearly positive effect.

A fourth, non-gating check is reported too: show the model an image with the
opposite label and see whether its answer follows the new image.

## Options

| Topic | Option A (proposed) | Option B | Option C |
|---|---|---|---|
| Gate 1 strictness | 0.65 / 0.10 / CI above 0 | stricter: 0.75 / 0.15 / CI above 0 (risk: a good teacher fails on small validation data) | looser: 0.60 / 0.05 / CI above 0 (risk: lets a weak image user through) |
| Fill | flat mid-grey (128) | black (0) | blurred copy of the image (needs more code, adds a second effect) |
| Cache `top_k` | 20 | 10 (smaller, loses the tail) | 50 (larger, little gain) |

## Recommendation: proposed thresholds (all must hold)

| Check | Metric | Proposed rule | Why |
|---|---|---|---|
| Better than chance | `original_balanced_accuracy` | at least **0.65** | 0.5 is chance; 0.65 leaves room to lose accuracy later |
| Uses the image | `blank_accuracy_drop` (real minus blank) | at least **0.10** | a blind model drops about 0; 0.10 is a visible, non-noise drop |
| Looks at the lesion | `lesion_mask_effect`, 95 % paired bootstrap CI | lower end **above 0** | the same significance rule as the primary endpoint |

If any check fails, the loop stops (`docs/FINAL_GOAL.md`, section 7) and the team
picks another teacher, task or domain. The three numbers are in
`configs/experiments/E0.yaml` (`image_reliance_check.gate1`) and stay `TODO` until
approved.

## Other settings that need a decision

- **Grey fill.** Blank image and both mask conditions use the same flat mid-grey
  (value 128 of 255). Because the lesion and the control are filled identically,
  any "a grey patch confuses the model" effect cancels in the difference.
  Alternatives (black, blur, the image's mean colour) are possible but add nothing
  for this comparison.
- **Teacher cache: `top_k`.** For every generated token the cache keeps the **20**
  most likely tokens and their log-probabilities (roughly 0.5 GB for 144k answers of
  ~20 tokens; the run prints the exact size). Distillation then matches the
  student to the teacher on these 20 instead of one hard answer.
- **Teacher cache: `max_new_tokens`.** Set from the RC-D1 answer-length counts as
  `ceil(1.5 x p99 answer words) + 8`, so almost no answer is cut and the rule is
  fixed in advance. (RC-D1 now records the p50, p90, p99 and maximum answer length.)
- **Image size.** The teacher and every student must see the same picture size, so
  `inputs.image_longest_side` is one value, fixed at the student-selection freeze
  (after RC-S0 measured what fits a T4). Gate 1 and the cache wait for it.
- **Masks delineate which finding?** Kvasir-SEG masks outline polyps, so the mask
  audit only uses the "polyp" probe (`image_reliance_check.mask_findings`). The id
  must match the frozen finding list.

## Risks

- **Thresholds too strict:** a good teacher could fail on a small validation set.
  The CI rule already accounts for sample size; the 0.65 and 0.10 bounds are
  deliberately modest. A failed gate is reported as it is, never re-run with other
  numbers.
- **Few mask-annotated images:** the lesion-mask effect needs positives with a
  mask. RC-D1 reports how many exist in validation; if there are too few for a
  CI, the agent reports it before RC-E0a runs.
- **Grey patch is not natural:** both conditions share it, so it cancels in the
  difference; the absolute flip rates are not read as clinical numbers.

## What changes once decided

`configs/experiments/E0.yaml` gets the three numbers, the fill and `top_k`;
`max_new_tokens` is added after RC-D1; the decision goes into
`docs/decisions_log.md`; RC-E0a changes from `blocked` towards `ready`.

## What blocks running RC-E0a

1. Team approves this memo (inbox note).
2. RC-D1 done (counts for the finding list; label rules filled and the finding list
   frozen).
3. RC-M1 done (the teacher loads and its adapter works).
4. The student-selection freeze set `inputs.image_longest_side`.

## Decision (2026-10-12)

- Gate 1: moderate rule (0.65 / 0.10 / CI above 0). Written into `configs/experiments/E0.yaml`.
- Fill: flat mid-grey (128).
- Cache `top_k`: 20.
- `max_new_tokens`: rule `ceil(1.5 x p99 answer words) + 8`, applied when the RC-D1
  counts arrive unless the team objects then.

## Amendment (2026-10-18): Gate 1 with our teacher; the teacher's quality rule

- Gate 1 is run on **our** teacher (D01 amendment): the 7B with the LoRA RC-TT1 trains on our
  training split. The thresholds of the decision of 2026-10-12 are unchanged.
- Rule 3, its quality part (D01 amendment): the teacher must pass Gate 1 and, once E1 seed 0
  exists, have a validation macro balanced accuracy of at least E1 seed 0's + 0.02. If either
  fails, it is trained once more for +50 % of N, re-checked, and accepted and reported either
  way. After that one extension, the study's stop rule for a failed Gate 1 applies as before.
- Gate 1 now has masked validation images. The image-disjoint split (D02 amendment) puts the
  Kvasir-SEG-linked images into validation in proportion; run 1's old split had none.

## Questions that were asked

1. Approve the three Gate 1 thresholds (0.65 / 0.10 / CI above 0)?
2. Approve mid-grey (128) as the fill?
3. Approve `top_k = 20` and the `max_new_tokens` rule?
