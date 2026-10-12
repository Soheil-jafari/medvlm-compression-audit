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
| [RC-S0](RC-S0_student_feasibility.md) | S05b | Student checks: feasibility of Qwen2.5-VL-3B and the fallback Qwen3-VL-2B on one T4 | Soheil | blocked (RC-M1) |
| [RC-S1](RC-S1_round1_screen.md) | S05b | Round 1 screen, 10 candidates | all three | not used (D05 amendment) |
| [RC-S2](RC-S2_round2_recipe.md) | S05b | Round 2, top 5 rebuilt with the full recipe | all three | not used (D05 amendment) |
| [RC-S3](RC-S3_round3_e2_recipe.md) | S05b | Round 3, top 3 with the E2 recipe, 2 seeds | all three | not used (D05 amendment) |
| [RC-E1](RC-E1_student_answer_finetune.md) | S06 | Student on ground-truth answers, 3 seeds | Soheil | blocked (RC-S0 numbers for the training length, RC-D1, open settings) |
| [RC-E2](RC-E2_student_offline_distillation.md) | S06 | Student with offline distillation, 3 seeds | Soheil | blocked (RC-E1 settings, RC-E0b) |
| [RC-B0](RC-B0_e2_reference_footprint.md) | S06 | E2 reference footprint and byte split for the budgets | Amir | blocked (RC-E2 seed 0) |
| [RC-Q0](RC-Q0_quantization_smoke_test.md) | S08 | Quantization toolchain smoke test: kernel classes, GPTQ and LLM.int8 on the student, versions | Soheil | ready |
| [RC-E3](RC-E3_ptq.md) | S08 | E3: GPTQ w4, w8 and LLM.int8 w8a8 of E2, 3 seeds | Soheil | blocked (RC-Q0, RC-E2, RC-D1, freeze) |
| [RC-E4](RC-E4_qakd.md) | S08 | E4: quantization-aware distillation (D08 option B), 3 seeds | Soheil | blocked (RC-Q0, RC-E0b, E2 settings, freeze) |
| [RC-E9](RC-E9_teacher_ptq.md) | S08 | E9: the teacher at 4 bit (supplementary) | Soheil | blocked (RC-Q0, RC-D1, freeze) |
| [RC-G2](RC-G2_gate2_pilot.md) | S09 | Gate 2 pilot at B2 (validation): E5 (3 criteria), E6, E7 | Amir | blocked (RC-E2, RC-B0, D07 scoring method, finding weights) |
| [RC-E5](RC-E5_generic_pruning_grid.md) | S09 | E5: magnitude, activation, Taylor x B1-B3 x 3 seeds (27 runs) | Amir | blocked (freeze, RC-E2, RC-B0, D07 section 1, RC-G2) |
| [RC-E6](RC-E6_output_similarity_grid.md) | S09 | E6: output-similarity layer pruning x B1-B3 x 3 seeds (9 runs) | Amir | blocked (as RC-E5) |
| [RC-E7](RC-E7_clinical_pruning_grid.md) | S09 | E7: clinical criterion x B1-B3 x 3 seeds (9 runs) | Amir | blocked (as RC-E5, weights frozen) |
| [RC-E8](RC-E8_clinical_pruning_plus_ptq4.md) | S09 | E8: E7 + 4-bit GPTQ x B1-B3 x 3 seeds (9 runs) | Amir | blocked (RC-E7, RC-Q0, freeze) |
| [RC-A1](RC-A1_unweighted_criterion.md) | S10 | A1: unweighted clinical criterion, B2 x 3 seeds | Amir | blocked (freeze, RC-E2, RC-B0, D07 section 1, RC-E7 at B2) |
| [RC-A2](RC-A2_rarity_only_weights.md) | S10 | A2: rarity-only weights, B2 x 3 seeds | Amir | blocked (as RC-A1) |
| [RC-A3](RC-A3_random_removal.md) | S10 | A3: random removal, B2 x 3 seeds | Amir | blocked (as RC-A1) |
| [RC-A4](RC-A4_calibration_size.md) | S10 | A4: 32 / 64 / 256 calibration images, B2 x 3 seeds (128 = E7) | Amir | blocked (as RC-A1) |
| [RC-A5](RC-A5_delta_definition.md) | S10 | A5: KL effect, B2 x 3 seeds (log-prob drop = E7) | Amir | blocked (as RC-A1; unit set depends on D07 section 1) |
| [RC-A6](RC-A6_unit_granularity.md) | S10 | A6: layers only / heads and channel groups, B2 x 3 seeds | Amir | blocked (as RC-A1) |
| RC-X1 | S11 | External test | Farzan | not written yet |
| RC-T1 | S11 | Final timing on the rented GPU | Amir | not written yet |

Statuses: `not written yet` -> `ready` -> `running` -> `done` (or `failed`, with the
reason in the card).
