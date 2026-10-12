# S07: Pre-registration freeze

**Goal.** Everything that could be tuned to flatter the clinical criterion is fixed
in writing before any compression run.

**Depends on.** S03 (training-split statistics) and S06 (E2 footprint). Must finish
before the first E3-E9 or A1-A6 run.

## Tasks

- [x] [agent] Fill a complete draft of `docs/preregistration.md` (iteration 40: decided items
  filled, data-dependent items given as rules, team items as proposals): finding list,
  prevalence groups and their rule, lesion-size thresholds, probe display names,
  weight table (training-split prevalence and the team's tiers), cap, normalisation,
  budget metric, protocol, B1-B3 targets and tolerance, the rule for "best of E5/E6"
  (validation split), bootstrap resamples, answer parsing and exclusion rules,
  Gate 2 definition, and how seeds enter the bootstrap.
- [x] [agent] Fill `configs/preregistration/finding_weights.yaml` and
  `configs/budgets.yaml` from the draft, leaving `frozen_on: TODO` (rules and proposals in the
  table; values after RC-D1 and the team's answers).
- [x] [agent] A guard (and its test) that refuses final compression runs while
  `frozen_on` is `TODO` (`cli.require_frozen_preregistration`, called by every stage of
  E3-E9 and A1-A6).
- [ ] [agent] After RC-D1: fill the finding list, prevalences, groups, size thresholds and
  display names by the rules of the draft.
- [ ] [team:Amir] [team:Soheil] [team:Farzan] Assign clinical importance tiers;
  review the draft; record the freeze date, commit and approvers.
- [ ] [agent] After the freeze: decisions-log entry with the freeze commit hash.

## Outputs

Frozen `docs/preregistration.md`, finding-weight table and budgets.

## Exit gate

Freeze recorded by all three members, freeze commit hash logged.
