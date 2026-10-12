# Run cards

GPU jobs are run by team members on their own dataset access (Kaggle, Colab or a
rented GPU). The agent prepares each job as a run card (`TEMPLATE.md`); the person
who runs it commits the result CSVs and the completed card. The agent picks up the
results in its next iteration.

## Queue

| Card | Stage | What | Who | Status |
|---|---|---|---|---|
| [RC-D1](RC-D1_data_preparation.md) | S03 | Data preparation: download, val/test halves, dedup, mask links, aggregate counts | Farzan | ready |
| [RC-M1](RC-M1_model_smoke_test.md) | S04 | GPU smoke test: load pair, score probes, remove one layer | Soheil | ready |
| [RC-E0a](RC-E0a_gate1_teacher_image_reliance.md) | S05 | Gate 1: teacher image-reliance checks (validation) | Soheil | blocked (D04, RC-D1, RC-M1, image size) |
| [RC-E0b](RC-E0b_teacher_cache_and_E0_evaluation.md) | S05 | Teacher cache (train) and E0 validation metrics | Soheil | blocked (Gate 1 verdict, RC-D1, open settings) |
| [RC-S0](RC-S0_student_feasibility.md) | S05b | Student candidates: feasibility of all 10 on one T4 | all three | blocked (RC-M1) |
| [RC-S1](RC-S1_round1_screen.md) | S05b | Round 1 screen, 10 candidates | all three | blocked (RC-S0, RC-D1, Round 1 settings) |
| [RC-S2](RC-S2_round2_recipe.md) | S05b | Round 2, top 5 rebuilt with the full recipe | all three | blocked (RC-S1, RC-E0b) |
| [RC-S3](RC-S3_round3_e2_recipe.md) | S05b | Round 3, top 3 with the E2 recipe, 2 seeds | all three | blocked (RC-S2, E2 recipe) |
| [RC-E1](RC-E1_student_answer_finetune.md) | S06 | Student on ground-truth answers, 3 seeds | Soheil | blocked (student choice, training length, RC-D1, open settings) |
| [RC-E2](RC-E2_student_offline_distillation.md) | S06 | Student with offline distillation, 3 seeds | Soheil | blocked (RC-E1 settings, RC-E0b) |
| [RC-B0](RC-B0_e2_reference_footprint.md) | S06 | E2 reference footprint and byte split for the budgets | Amir | blocked (RC-E2 seed 0) |
| RC-E3 / RC-E4 / RC-E9 | S08 | Quantization arms | Soheil | not written yet |
| [RC-G2](RC-G2_gate2_pilot.md) | S09 | Gate 2 pilot at B2 (validation): E5 (3 criteria), E6, E7 | Amir | blocked (RC-E2, RC-B0, D07 scoring method, finding weights) |
| RC-E5 ... RC-E8 | S09 | Pruning grid | Amir | not written yet |
| RC-A1 ... RC-A6 | S10 | Ablations at B2 | Amir | not written yet |
| RC-X1 | S11 | External test | Farzan | not written yet |
| RC-T1 | S11 | Final timing on the rented GPU | Amir | not written yet |

Statuses: `not written yet` -> `ready` -> `running` -> `done` (or `failed`, with the
reason in the card).
