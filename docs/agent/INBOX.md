# Inbox: notes from the team to the research agent

Add one dated line per note at the top of **Open**. Decisions, answers to decision
memos, priorities, "run card RC-xx done", "please stop", and so on. Notes here
override `STATUS.md`. The agent moves each handled note to **Handled** with the
iteration number.

Example: `- 2026-10-20 (Soheil): D01 decided: endoscopy, Qwen2.5-VL-7B -> 3B. Log it.`

## Open

<!-- newest first -->

## Handled

<!-- - YYYY-MM-DD (name): note -> iteration N, what was done -->
- 2026-10-12 (Soheil): D05 decided: Q1 teacher family only, Q2 keep size, Q3 weights accepted -> iteration 4, recorded in D05, `configs/student_selection.yaml` and the decisions log.
- 2026-10-12 (Soheil): no scheduled runs; iterations run back to back in the interactive session until stopped -> iteration 4, scheduled task removed, protocol updated.
- 2026-10-12 (Soheil): asked why retinal is not the main domain -> answered in the session; retinal stays the alternative / second domain, to revisit in D02.
