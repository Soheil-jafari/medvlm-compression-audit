# Decision memos

The agent prepares a memo for every decision the team has to make. The team answers
in `docs/agent/INBOX.md`; the agent then records the decision in
`docs/decisions_log.md` and applies it to the configs.

| Memo | Decision | Stage | Status |
|---|---|---|---|
| [D01](D01_teacher.md) | Teacher (the student is chosen by D05) | S01 | decided 2026-10-12: T1, Qwen2.5-VL-7B + Kvasir-VQA-x1 LoRA |
| [D02](D02_datasets.md) | Datasets and their roles; first finding list | S01 | decided 2026-10-12: Kvasir-VQA-x1, Kvasir-SEG, EndoBench; val/test from the official test split |
| [D03](D03_licence.md) | Code licence | S01 | decided 2026-10-12: MIT |
| [D04](D04_gate1.md) | Gate 1 thresholds, mask fill, teacher-cache settings | S05 | answered 2026-10-12: moderate rule, mid-grey fill, top_k 20 |
| [D05](D05_student_selection.md) | Student selection protocol, 10 -> 5 -> 3 -> 1 | S05b | answered 2026-10-12: teacher family only, keep size, weights accepted |
| [D06](D06_training_recipe.md) | LoRA and optimiser recipe, recovery length, evaluation numbers | S05b, S06, S09 | answered 2026-10-12: all recommendations accepted |
| [D07](D07_pruning_settings.md) | Scoring method and cost, removal order, grid, budgets | S09, S10 | partly answered 2026-10-12: removal order and grid accepted; scoring method and budgets open |

Memo layout: question; options; evidence for each option with verified sources
(links and access dates); risks; recommendation; what changes in the repository
once decided. Unverified facts are marked `TODO: verify`, never guessed.
