# Non-finite masked images in the lesion-mask audit (2026-11-25)

**Status:** proposed by the agent at the research lead's request (2026-11-25) and recorded
**before** any effect size of the sensitivity version below is computed. The pre-registered
audit and its reported values stay as they are. The research lead may change the rule before the
analysis runs; a change after that would be a post hoc change and reported as such.

## What was found

`lesion_mask_margin_effect` is "undefined and left out" in every 3B-student arm's results (E2,
E3 `w4` / `w8` / `w8a8` / `awq_w4`, E5 x 3, E5R, E7 and their B2 test files), and present for E0,
E9 and E10. In E5R's saved outcomes (Amir, 2026-11-24) every original margin is finite, but some
**masked** images give a non-finite margin (checked here on the arrays):

| Split | Audit probes | Lesion-masked non-finite | Random-masked non-finite | Either |
|---|---|---|---|---|
| Validation | 86 | 1 | 0 | 1 |
| Test | 83 | 2 | 2 | 3 |
| Unfiltered test | 144 | 3 | 14 | 15 |

Most likely the same fp16 overflow in the 3B student's vision encoder that the non-finite drop
list handles for original images (second amendment of 2026-11-13), now triggered by the masked
versions, which the drop list does not cover. The teacher (E0, E9) and the 2B (E10) do not show
it. The margin effect is a mean over probes, so one non-finite margin makes it NaN.

## How such a probe is scored now (the pre-registered version)

A probe's answer is "yes" when log p(yes) >= log p(no) (`YesNoScore.predicted`). A non-finite
forward pass gives NaN for both, the comparison is false, and the masked answer is "no". A flip
is a masked answer that differs from the original answer. So **a non-finite masked pass is
counted as a flip exactly when the original answer was "yes"**, whatever the image shows (checked
on every non-finite masked probe of E5R's three splits). Such flips are spurious:

- on the lesion side they add to the lesion flip rate (the effect looks larger);
- on the random-region side they add to the control's flip rate (the effect looks smaller);
  the unfiltered test's 14 random-masked cases in E5R matter most;
- an arm that rarely answers "yes" (the collapsed arms) gets few of them; the count differs by
  arm, so arms are not affected equally.

## The rule (a labelled sensitivity analysis; the pre-registered version stays as reported)

1. **Scope:** the lesion-mask audit of every arm on every split (validation, filtered test,
   unfiltered test, and the external test when it exists): the flip effect (primary endpoint 2)
   and the margin effect.
2. **Exclusion set, per split:** every audit probe (image, finding) whose lesion-masked or
   random-masked margin is non-finite **in any arm** of the analysis (the union over arms), or
   whose original margin is non-finite. The same probes are dropped for every arm, both of its
   components (lesion and random) together.
3. **Recomputed on the remaining probes:** the lesion and random flip rates, the flip effect and
   the margin effect, with the same paired image-cluster bootstrap as the pre-registered audit
   (the evaluation's resample count and seed).
4. **Reported** next to the pre-registered values, labelled "sensitivity: audit probes with a
   non-finite masked image dropped (union over arms)", with the number of probes dropped per
   split and, per arm, its own non-finite count and how many of them the pre-registered version
   counted as flips. Verdicts and pre-registered tables use the pre-registered values.
5. **Arms without audit margins** (evaluated before 2026-11-20): a non-finite masked image cannot
   be seen from flips alone, so such an arm is "not evaluable" in the sensitivity analysis until
   a mask-audit-only rerun (`save_outcomes.py --audit-only`) adds its margins; its probes still
   enter the union once it has them.
6. **The union is recomputed** whenever arms are added; the final report uses the union over
   every arm of the final analysis (the set is written as counts only; the probes are image ids
   and are not committed).

## Related paths checked

- **FBR:** its pilot evaluation uses the same audit, so the same rule applies to it. Its training
  probes and E2's margin cache use original training images: a non-finite margin there (student
  or E2) makes that update non-finite, and the update is skipped and logged as the registered
  recovery skips one (until now a non-finite E2 margin stopped the run before training; cell (ii)
  now counts them).
- **The plain-answer collapse:** unaffected (original images only).
