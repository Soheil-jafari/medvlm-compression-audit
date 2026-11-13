# Inbox: notes from the team to the research agent

Add one dated line per note at the top of **Open**. Decisions, answers to decision
memos, priorities, "run card RC-xx done", "please stop", and so on. Notes here
override `STATUS.md`. The agent moves each handled note to **Handled** with the
iteration number.

Example: `- 2026-10-20 (Soheil): D01 decided: endoscopy, Qwen2.5-VL-7B -> 3B. Log it.`

## Open

<!-- newest first -->
- 2026-11-08 (agent): what is left of the freeze list after the decisions of 2026-11-08 (none
  of it blocks the freeze):
  - item 17, the timing GPU and precision for RC-T1: open (team; timing is secondary);
  - B2, E4's memory at 1 x 16 without gradient checkpointing: RC-E4 seed 0 measures it after
    the freeze (it stops rather than change the setting).

## Handled

<!-- - YYYY-MM-DD (name): note -> iteration N, what was done -->
- 2026-11-09 (Soheil): freeze the pre-registration, approved by Soheil and Amir on 2026-11-08
  at commit 18f1b47; accept the C4 sensitivity analysis, the 239-246 h estimate and Soheil
  starting account 3's sessions -> iteration 71: `frozen_on` / `frozen_by` / `commit` set,
  decisions log, STATUS, RC-B0 `ready`; every later change needs a dated amendment.
- 2026-11-08 (Soheil and Amir; choices delegated to the research lead): **the freeze list of
  2026-10-17, decided** -> iteration 70, pre-registration section 11 item 14, configs, decisions
  log. Item by item, with where it was decided:
  1 fixed training pool: decided in iteration 36, confirmed (pre-registration section 2);
  2 finding inclusion rule, 3 prevalence groups, A1-A6 finding list, display names, prevalence,
  groups (no rare group), size strata, mask finding: frozen in the amendment of 2026-10-31
  (section 11 items 1-5); 4 cap 20 and 5 normalisation none, 6 tiers (action 2.0 / follow_up
  1.5 / other 1.0; polyp action, oesophagitis and ulcerative colitis follow_up), A7 the tier of
  each finding: decided 2026-11-08 (section 3, `finding_weights.yaml`); 7 comparator: decided
  2026-11-08 in the seed plan (section 5); 8 headline accuracy: macro balanced accuracy at the
  operating point (section 5); 9 H2 matching and 10 McNemar: decided, H2 confirmatory with E3
  `w8` in the 3-seed tier (section 6); 11 A3 ranking without the division, 12 reuse of E7 for
  `A4:n128` and `A5:logprob_drop` (section 7); 13 scoring method: exact for layers and KV groups,
  first-order for channel groups, the 30-group check a report (section 7; corrected
  2026-11-08: no exact fallback); 14 recovery 157 updates on 2,500 examples (section 7); 15 A1
  and A2 re-weight E7's effects (implemented 2026-11-08); 16 budget hardware Kaggle Tesla T4
  (section 4); 18 rare-finding pooling: moot, no rare group (2026-10-31); 19 the two config
  notes confirmed (A1 at B2 only; A6 `heads_channels` without whole layers); A8 `max_new_tokens`
  46 by the D04 rule (2026-10-31); A9 data facts and A10 masked validation images: filled from
  RC-D1 run 4b; B1 the budgets frozen as 15 / 30 / 45 % of the decoder bytes (RC-B0 converts
  them). Still open: 17 and B2 (Open above).
- 2026-10-19 (Soheil, in the session): RC-D1 run 3 not accepted; D02 amendment 3 (embeddings, the rule chosen on human labels, run 4, the strict test subset) -> iteration 54.
- 2026-10-19 (Soheil, in the session): calibration results and D02 amendment 2 final values (0.65, no automatic join, guard 10 %, the final sheet as the deciding check, the embedding fallback); T4 x2 counts once (N = 38,000); RC-TT1 on the published target modules -> iteration 53.
- 2026-10-18 (Soheil, in the session): RC-D1 run 2 result; D02 amendment 2 (two-stage near-duplicate detection, calibration step, run 3) -> iteration 52, code, tests, RC-D1, RC-TT0, RC-TT1, data card, D02, log, pre-registration.
- 2026-10-18 (Soheil, in the session): RC-D1 run 1 findings; D02 and D01 amendments, rule 3, the appendix item -> iteration 51.
- 2026-10-17 (Soheil, in the session): decisions on the image side (672), G2 (0.10), the D06 amendment (1 x 16, no gradient checkpointing), the GPU budget (90 h per week, 30 per person), RC-D1 owner (Soheil); Tasks 1-5 (shared data, hand-offs, unattended runs, work split and schedule, freeze list) -> iteration 50.
- 2026-10-16 (Soheil, in the session): RC-S0 proposals answered: data_subset 10,000 accepted; image side not decided (RC-S0b first); training-length check added (30,000, E2 and E7 at B2, seed 0); G2 unchanged -> iteration 48, configs, D05, D06, pre-registration, RC-S0b, RC-L1 and the log.
- 2026-10-12 (Soheil): D08 answered in the session (GPTQ 4/8 + LLM.int8, E4 option B, 128 calibration images, keep w8a8) -> iteration 37, D08, decisions log and quantization configs.
- 2026-10-12 (Soheil): D05 amendment accepted, pick one student (Qwen2.5-VL-3B, fallback Qwen3-VL-2B) -> iteration 34, configs, D05, RC-S0 and the log updated.
- 2026-10-12 (Soheil): D04 answered in the session (moderate Gate 1 rule, mid-grey fill, top_k 20) -> iteration 26, E0 config, D04 memo and decisions log updated.
- 2026-10-12 (Soheil): D05 decided: Q1 teacher family only, Q2 keep size, Q3 weights accepted -> iteration 4, recorded in D05, `configs/student_selection.yaml` and the decisions log.
- 2026-10-12 (Soheil): no scheduled runs; iterations run back to back in the interactive session until stopped -> iteration 4, scheduled task removed, protocol updated.
- 2026-10-12 (Soheil): D02 decided (Q1 A, Q3 EndoBench, Q5 retinal later; Q2, Q4 defaults) and D03 decided (MIT) -> iteration 8, configs, LICENSE, pyproject, README, decisions log.
- 2026-10-12 (Soheil): D01 decided: T1 (Qwen2.5-VL-7B + Kvasir LoRA) now; optional Qwen3-VL-8B teacher check at the end if budget remains -> iteration 6, teacher config DECIDED, student pool fixed to 10 Qwen-family models.
- 2026-10-12 (Soheil): asked why retinal is not the main domain -> answered in the session; retinal stays the alternative / second domain, to revisit in D02.
