# Gate 2: E7 against the comparator at B2, validation, seed 0

- Runs included: `results/E5/{magnitude,activation,taylor}/B2/seed0/`, `results/E6/B2/seed0/`,
  `results/E7/B2/seed0/` (RC-G2 sessions 1, 2 and 4, committed 2026-11-20); reference
  `results/E2/nobudget/seed0/`. Run card RC-G2
- Split: validation (227 labelled images; the lesion-mask audit uses the 86 mask-positive polyp
  probes; 13 small polyps, stratum 0)
- Status: complete for the comparator choice and the futility rule. E5R and E7U (exploratory,
  RC-G2 session 5) are not in yet. The paired image-level intervals (E7 minus the comparator)
  need the saved per-image outcomes, which are not in the repository: they come from RC-C1
  with the outcome hand-offs attached. The intervals below are each arm's own 95 %
  image-bootstrap interval from its `metrics.csv`.

## The pre-registered ordering (`scripts/choose_comparator.py`)

Ordering of pre-registration section 5 / section 11 item 12: (i) `op_small_finding_recall`,
(ii) `op_mask_positive_recall`, (iii) `op_small_margin_mean`, (iv) the name. Written to
`results/seed_plan/comparator_B2.json` (byte-identical on pandas 2.3.3 and 3.1.1).

| Place | Arm | (i) small-finding recall | (ii) mask-positive recall | (iii) small margin |
|---|---|---|---|---|
| 1 | E5 magnitude | 1.000 | 1.000 (86/86) | 10.72 |
| 2 | E5 activation | 1.000 | 1.000 | 7.75 |
| 3 | E5 Taylor | 1.000 | 0.988 | 11.54 |
| 4 | E6 | 1.000 | 0.977 | 9.23 |
| - | **E7** | 1.000 | **0.977** | 6.21 |

- **Comparator: E5 magnitude**, decided by (iii): the four candidates tie on (i) at the ceiling,
  E5 magnitude and E5 activation tie on (ii), and E5 magnitude has the larger margin.
- **Futility rule: stop.** E7 ties the comparator on (i) and ranks below it on (ii) (0.977
  against 1.000: 2 of the 86 mask-positive polyp probes). By the rule written before any seed-0
  compression result, seeds 1 and 2 of the main tier (E2, E7 and the comparator at B2, E3 `w8`
  for H2, A1 for the secondary comparison) do not run; seed 0 is reported as exploratory, with
  no confirmatory claim (H3 becomes `E7_vs_best_seed0`). The run cards refuse seeds 1 and 2.

## Numbers (validation, seed 0, B2)

| Metric | E2 | E5 magnitude | E5 activation | E5 Taylor | E6 | E7 |
|---|---|---|---|---|---|---|
| `op_macro_balanced_accuracy` | 0.836 (0.812-0.859) | 0.721 (0.693-0.751) | 0.784 (0.757-0.808) | 0.748 (0.719-0.775) | 0.632 (0.617-0.645) | 0.745 (0.715-0.773) |
| `op_small_finding_recall` (13 small polyps) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `op_mask_positive_recall` (86) | - | 1.000 | 1.000 | 0.988 | 0.977 | 0.977 |
| `lesion_mask_effect` | 0.058 (0.012-0.116) | 0.000 (0.000-0.000) | 0.000 (0.000-0.000) | 0.058 (0.012-0.116) | 0.000 (0.000-0.000) | 0.395 (0.279-0.512) |
| `original_yes_rate` (mask audit, plain yes/no) | 0.209 | 0.000 | 0.000 | 0.128 | 0.000 | 0.709 |
| `auroc:polyp` | 0.999 | 0.994 | 0.996 | 0.996 | 0.990 | 0.994 |
| `ece` | 0.162 | 0.274 | 0.321 | 0.254 | 0.326 | 0.081 |
| `macro_balanced_accuracy` (plain yes/no) | 0.553 | 0.500 | 0.500 | 0.521 | 0.500 | 0.688 |
| Units removed | - | 957 channel groups | 957 | 957 | 10 layers | 874 channel groups, 11 KV groups |
| Footprint (target 5.954e9) | - | 6.105e9 (+2.55 %) | 6.105e9 (+2.55 %) | 6.105e9 (+2.55 %) | 6.064e9 (+1.85 %) | 6.132e9 (+3.00 %) |

Every arm is within the 3 % band; no arm removes a unit of layers 0, 1 or 35; each stage-07 peak
equals its search footprint within 0.03 %. E7's first-order check: Spearman 0.972 over 30
channel groups (rule: at least 0.7), slope of the exact effect on the estimate 0.958
(0.891-1.025), intercept -0.006 (-0.076 to 0.065); with the 30 exact effects in place, 2 of the
30 switch at B2 and no KV-group or layer decision changes.

## Reading

Small-finding recall at the operating point is 1.000 for every arm, the reference included, so
the first criterion cannot separate them on 13 small polyps; the choice and the futility
decision rest on the tie-breaks, and E7 stops on two mask-positive probes. By the rule, Gate 2
points to **article shape B** (audit and negative result; `docs/FINAL_GOAL.md` section 1): H3
cannot be supported confirmatorily.

Two observations for the team, not used in any decision above. (1) The lesion-mask effect is
measured on the plain yes/no answer (`image_reliance.lesion_mask_effect`), not at the operating
point. E5 magnitude, E5 activation and E6 answer "no" to every mask-positive probe in the plain
answer (`original_yes_rate` 0), so no answer can flip and their effect is 0 by construction; E7
answers "yes" to 71 % of them (E2 21 %). The effect therefore mixes reliance on the lesion with
the plain-answer yes rate. (2) E7 has the lowest calibration error of all six (0.081, E2 0.162)
and the highest plain-answer balanced accuracy (0.688, E2 0.553), while its operating-point
accuracy (0.745) sits among the E5 arms. All of this is seed 0, validation, descriptive.
