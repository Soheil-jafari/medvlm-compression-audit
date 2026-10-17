# Run cards

GPU jobs are run by team members on their own dataset access (Kaggle, Colab or a
rented GPU). The agent prepares each job as a run card (`TEMPLATE.md`); the person
who runs it commits the result CSVs and the completed card. The agent picks up the
results in its next iteration.

Since 2026-10-17 every card after RC-D1 reads the **shared data** (one private Kaggle dataset
written by RC-D1) with `--data-root`, and the outputs of earlier runs (**hand-offs**, one private
dataset each) with `--from`. Both are verified by their SHA-256 manifests before anything runs.
A card runs unattended with `notebooks/run_card.ipynb` (Save & Run All), which calls
`scripts/run_card.py`. Section 2b of each card says what to attach and what it hands off. How to
set up an account: `docs/team_setup.md`. Who runs what, week by week: `docs/schedule.md`.

## Queue

| Card | Stage | What | Who | Status |
|---|---|---|---|---|
| [RC-D1](RC-D1_data_preparation.md) | S03 | Data preparation: download, image-disjoint re-split of all images (70/15/15 by near-duplicate group), mask links, aggregate counts, the shared-data dataset; pass 2 recounts after the label values | Soheil | **ready (run 2)**; run 1 used the old split |
| [RC-M1](RC-M1_model_smoke_test.md) | S04 | GPU smoke test: load pair, score probes, remove one layer | Soheil | done 2026-10-15 |
| [RC-TT0](RC-TT0_teacher_feasibility.md) | S05 | Our teacher: feasibility of one LoRA step of the 7B (fp16 on 2 x T4, QLoRA on 1 x T4), then rule 3 (setup and N) | Soheil | **ready** |
| [RC-TT1](RC-TT1_teacher_training.md) | S05 | Our teacher: LoRA training on our training split (resumable), hand-off, validation | Soheil | ready (needs RC-D1 run 2 and RC-TT0's result file) |
| [RC-E0a](RC-E0a_gate1_teacher_image_reliance.md) | S05 | Gate 1: teacher image-reliance checks (validation), with our teacher adapter | Soheil | blocked (RC-TT1, RC-D1 both passes, finding list; image side and RC-M1 met) |
| [RC-E0b](RC-E0b_teacher_cache_and_E0_evaluation.md) | S05 | Teacher cache (train; its first 100 answers are timed) and E0 validation metrics, with our teacher adapter | Soheil | blocked (RC-TT1, Gate 1 verdict, RC-D1, open settings; image side met) |
| [RC-S0](RC-S0_student_feasibility.md) | S05b | Student checks: feasibility of Qwen2.5-VL-3B and the fallback Qwen3-VL-2B on one T4 | Soheil | done 2026-10-16 (both pass G1) |
| [RC-S0b](RC-S0b_image_size_measurement.md) | S05b | Image side: student and teacher at 448 / 560 / 672 on synthetic images, one process per step, LoRA step at micro-batch 1 / 2 / 4 with and without gradient checkpointing, stated adapter and processor settings | Soheil | done 2026-10-16 (run 2); side 672 decided 2026-10-17 |
| [RC-S1](RC-S1_round1_screen.md) | S05b | Round 1 screen, 10 candidates | all three | not used (D05 amendment) |
| [RC-S2](RC-S2_round2_recipe.md) | S05b | Round 2, top 5 rebuilt with the full recipe | all three | not used (D05 amendment) |
| [RC-S3](RC-S3_round3_e2_recipe.md) | S05b | Round 3, top 3 with the E2 recipe, 2 seeds | all three | not used (D05 amendment) |
| [RC-E1](RC-E1_student_answer_finetune.md) | S06 | Student on ground-truth answers, 3 seeds | Farzan (card); seeds 0 Soheil, 1 Farzan, 2 Amir | blocked (RC-D1; the validation pass needs the finding list; image side, G2, training length met) |
| [RC-E2](RC-E2_student_offline_distillation.md) | S06 | Student with offline distillation, 3 seeds | Soheil (card); seeds 0 Soheil, 1 Farzan, 2 Amir | blocked (RC-E1 settings, RC-E0b) |
| [RC-L1](RC-L1_training_length_check.md) | S06/S09 | Training-length check: E2 and E7 at B2 with 30,000 examples, seed 0 (pre-registration section 6) | Amir and Soheil (E2@30k Soheil, E7@30k Amir) | blocked (RC-E2, RC-E7 B2 seed 0, RC-B0, freeze; agent code first) |
| [RC-B0](RC-B0_e2_reference_footprint.md) | S06 | E2 reference footprint and byte split for the budgets | Amir | blocked (RC-E2 seed 0) |
| [RC-Q0](RC-Q0_quantization_smoke_test.md) | S08 | Quantization toolchain smoke test: kernel classes, GPTQ and LLM.int8 on the student, versions | Soheil | done 2026-10-15 |
| [RC-E3](RC-E3_ptq.md) | S08 | E3: GPTQ w4, w8 and LLM.int8 w8a8 of E2, 3 seeds | Soheil | blocked (RC-E2, RC-D1, freeze; RC-Q0 met) |
| [RC-E4](RC-E4_qakd.md) | S08 | E4: quantization-aware distillation (D08 option B), 3 seeds | Soheil | blocked (RC-E0b, E2 settings, freeze; RC-Q0 met) |
| [RC-E9](RC-E9_teacher_ptq.md) | S08 | E9: the teacher at 4 bit (supplementary) | Farzan | blocked (RC-TT1, RC-D1, freeze; RC-Q0 met) |
| [RC-G2](RC-G2_gate2_pilot.md) | S09 | Gate 2 pilot at B2 (validation): E5 (3 criteria), E6, E7 | Amir | blocked (RC-E2, RC-B0, D07 scoring method, finding weights) |
| [RC-E5](RC-E5_generic_pruning_grid.md) | S09 | E5: magnitude, activation, Taylor x B1-B3 x 3 seeds (27 runs) | Amir (card); seeds 0 Amir, 1 Farzan, 2 Amir | blocked (freeze, RC-E2, RC-B0, D07 section 1, RC-G2) |
| [RC-E6](RC-E6_output_similarity_grid.md) | S09 | E6: output-similarity layer pruning x B1-B3 x 3 seeds (9 runs) | Amir (card); seeds 0 Amir, 1 Farzan, 2 Amir | blocked (as RC-E5) |
| [RC-E7](RC-E7_clinical_pruning_grid.md) | S09 | E7: clinical criterion x B1-B3 x 3 seeds (9 runs) | Amir (card); seeds 0 Amir, 1 Farzan, 2 Amir | blocked (as RC-E5, weights frozen) |
| [RC-E8](RC-E8_clinical_pruning_plus_ptq4.md) | S09 | E8: E7 + 4-bit GPTQ x B1-B3 x 3 seeds (9 runs) | Amir (card); seeds 0 Amir, 1 Farzan, 2 Amir | blocked (RC-E7, freeze; RC-Q0 met) |
| [RC-A1](RC-A1_unweighted_criterion.md) | S10 | A1: unweighted clinical criterion, B2 x 3 seeds | Amir (card); seeds 0 Soheil, 1 Farzan, 2 Amir (proposal) | blocked (freeze, RC-E2, RC-B0, D07 section 1, RC-E7 at B2) |
| [RC-A2](RC-A2_rarity_only_weights.md) | S10 | A2: rarity-only weights, B2 x 3 seeds | Amir (card); seeds 0 Soheil, 1 Farzan, 2 Amir (proposal) | blocked (as RC-A1) |
| [RC-A3](RC-A3_random_removal.md) | S10 | A3: random removal, B2 x 3 seeds | Amir (card); seeds 0 Soheil, 1 Farzan, 2 Amir (proposal) | blocked (as RC-A1) |
| [RC-A4](RC-A4_calibration_size.md) | S10 | A4: 32 / 64 / 256 calibration images, B2 x 3 seeds (128 = E7) | Amir (card); seeds 0 Soheil, 1 Farzan, 2 Amir (proposal) | blocked (as RC-A1) |
| [RC-A5](RC-A5_delta_definition.md) | S10 | A5: KL effect, B2 x 3 seeds (log-prob drop = E7) | Amir (card); seeds 0 Soheil, 1 Farzan, 2 Amir (proposal) | blocked (as RC-A1; unit set depends on D07 section 1) |
| [RC-A6](RC-A6_unit_granularity.md) | S10 | A6: layers only / heads and channel groups, B2 x 3 seeds | Amir (card); seeds 0 Soheil, 1 Farzan, 2 Amir (proposal) | blocked (as RC-A1) |
| [RC-R1](RC-R1_reference_arms_test_split.md) | S11 | E0, E1, E2 on the test split (after the freeze) | Soheil | blocked (freeze, RC-TT1, RC-E0b, RC-E1, RC-E2) |
| [RC-C1](RC-C1_paired_comparisons.md) | S11 | Paired comparisons (H1-H4), tables, figures, verdicts; CPU only | Soheil | blocked (every test-split evaluation) |
| [RC-X1](RC-X1_external_test_endobench.md) | S11 | EndoBench part A: terms, format, overlap with our images (no GPU); part B written after it | Farzan | blocked (RC-D1) |
| [RC-T1](RC-T1_final_timing.md) | S11 | Final timing of every arm on the rented GPU (T5) | Amir | blocked (GPU and precision chosen, every arm done) |

Statuses: `not written yet` -> `ready` -> `running` -> `done` (or `failed`, with the
reason in the card).
