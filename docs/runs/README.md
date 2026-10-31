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

Since 2026-10-29 (D02 amendments 6 and 7) every model is scored on three tests:
- the test split: the filtered test, a "reduced-leakage" test (F = 0.86), co-primary;
- the unfiltered test: the filtered test with the removed test images added back;
- the external test from another centre, co-primary (RC-X2; not built yet).

The strict subset was dropped: at F = 0.86 it is the whole filtered test. The shared data of
every card are RC-D1 run 4b's private dataset `medcompress-data`: its `split_sha256` is pinned
in `configs/data/kvasir_vqa_x1.yaml` and checked when the data are attached.

Today the first two are scored as separate splits. In the runner, each step `test evaluation
<run>` is followed by `unfiltered test evaluation <run>`. Section 2b of a card may still list only the first. By hand, run each
`06_evaluate.py ... -o evaluation.split=test` again with `-o evaluation.split=test_unfiltered`.
Its rows go to `metrics_test_unfiltered.csv`, its outcomes to `outcomes_test_unfiltered.npz`.

## Queue

| Card | Stage | What | Who | Status |
|---|---|---|---|---|
| [RC-D1](RC-D1_data_preparation.md) | S03 | Data preparation: download, image-disjoint re-split of all images (70/15/15 by near-duplicate group), mask links, aggregate counts, the shared-data dataset; pass 2 recounts after the label values | Soheil | **done** (run 4b, 2026-10-29: shared data `medcompress-data`, `split_sha256` 54dfa70a...); pass 2 recounts after the label values; runs 1-4 not used |
| [RC-D1-embed](RC-D1-embed_embeddings_and_labelling_sheets.md) | S03 | Embeddings (proposal DINOv2-small, verified on the hub), candidate pairs, about 300 pairs to label on score-free sheets (CPU) | Soheil | **ready** |
| [RC-D1-label](RC-D1-label_rater_instructions.md) | S03 | Two raters label the pairs; kappa, every rule's sensitivity and precision, the joining rule chosen by the rule fixed in advance (CPU) | two raters; scoring Soheil | ready after RC-D1-embed |
| [RC-M1](RC-M1_model_smoke_test.md) | S04 | GPU smoke test: load pair, score probes, remove one layer | Soheil | done 2026-10-15 |
| [RC-TT0](RC-TT0_teacher_feasibility.md) | S05 | Our teacher: feasibility of one LoRA step of the 7B (fp16 on 2 x T4, QLoRA on 1 x T4), then rule 3 (setup and N) | Soheil | done 2026-10-19 (result file committed) |
| [RC-TT1](RC-TT1_teacher_training.md) | S05 | Our teacher: LoRA training on our training split (resumable), hand-off, validation | Soheil | done (2026-10-31: two sessions, 15.17 h of training; adapter `e0-teacher-adapter`) |
| [RC-E0a](RC-E0a_gate1_teacher_image_reliance.md) | S05 | Gate 1 (teacher image-reliance checks) and the E0 validation, with our teacher adapter | Soheil | ready (2026-10-31; about 2 h on T4 x2 plus setup, an estimate) |
| [RC-E0b](RC-E0b_teacher_cache_and_E0_evaluation.md) | S05 | Teacher cache of the training pool (its first 100 answers are timed; stops and resumes between shards) | Soheil | ready after RC-E0a's `gate1_pass` = 1 is committed (the cache checks it); C unknown, about 12.5 h one answer at a time (an estimate), so two sessions |
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
| [RC-X1](RC-X1_external_test_endobench.md) | S11 | EndoBench part A: terms, format, overlap with our images (no GPU); part B written after it | Farzan | set aside (D02 amendment 7) |
| [RC-X2](RC-X2_external_test_other_centre.md) | S11 | External test from another centre, co-primary: **PolypGen first** (2026-10-31), part A: the exact licence (or stop), centres, polyp masks and negatives, overlap (no GPU); part B (probes, strata with our quartiles, the held-out filter's check) planned in section 7 | [team] (proposal: Farzan) | blocked (licence) |
| [RC-T1](RC-T1_final_timing.md) | S11 | Final timing of every arm on the rented GPU (T5) | Amir | blocked (GPU and precision chosen, every arm done) |

Statuses: `not written yet` -> `ready` -> `running` -> `done` (or `failed`, with the
reason in the card).
