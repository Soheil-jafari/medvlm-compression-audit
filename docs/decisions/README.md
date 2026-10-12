# Decision memos

The agent prepares a memo for every decision the team has to make. The team answers
in `docs/agent/INBOX.md`; the agent then records the decision in
`docs/decisions_log.md` and applies it to the configs.

| Memo | Decision | Stage | Status |
|---|---|---|---|
| [D01](D01_teacher.md) | Teacher (the student is chosen by D05) | S01 | written 2026-10-12, waiting on the team |
| D02 | Datasets and their roles; first finding list | S01 | not written yet |
| [D03](D03_licence.md) | Code licence | S01 | written 2026-10-12, waiting on the team |
| D04 | Gate 1 thresholds | S05 | not written yet |
| [D05](D05_student_selection.md) | Student selection protocol, 10 -> 5 -> 3 -> 1 | S05b | answered 2026-10-12: teacher family only, keep size, weights accepted |

Memo layout: question; options; evidence for each option with verified sources
(links and access dates); risks; recommendation; what changes in the repository
once decided. Unverified facts are marked `TODO: verify`, never guessed.
