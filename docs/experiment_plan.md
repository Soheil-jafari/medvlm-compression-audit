# Experiment plan

The experiment IDs below are used everywhere: configs, code, results and the paper.
The registry (`src/medcompress/registry.py`) is the source of truth;
`tests/test_configs.py` checks that configs agree with it.

## Ladder

| ID | What | Starts from | Needs | Budgets | Seeds | Scripts | Main outputs |
|---|---|---|---|---|---|---|---|
| E0 | Teacher, full precision | pretrained | - | - | 1 | 00, 01, 06, 07 | teacher cache, metrics |
| E1 | Student fine-tuned on correct answers only | pretrained | - | - | 3 | 02 (sft), 06, 07 | adapters, metrics |
| E2 | Student + offline distillation | pretrained | E0 cache | - | 3 | 02 (offline_kd), 06, 07 | adapters, metrics, **budget reference** |
| E3 | E2 + PTQ: `w8`, `w4` (GPTQ/AWQ), `w8a8` where supported | E2 | - | measured | 3 (per E2 seed) | 03, 06, 07 | quantized models, metrics |
| E4 | Quantization-aware distillation (GRACE-style) | pretrained | E0 cache | measured | 3 | 02 (qakd), 06, 07 | adapters, metrics |
| E5 | Generic structured pruning: `magnitude`, `activation`, `taylor` + recovery | E2 | E0 cache | B1-B3 | 3 | 04, 05, 06, 07 | unit scores, metrics |
| E6 | Output-similarity layer pruning (Khan et al. style) + recovery | E2 | E0 cache | B1-B3 | 3 | 04, 05, 06, 07 | unit scores, metrics |
| E7 | **Clinical pruning criterion (ours)** + recovery | E2 | E0 cache | B1-B3 | 3 | 04, 05, 06, 07 | unit scores, metrics, mask audit |
| E8 | E7 + 4-bit PTQ | E7 | - | B1-B3 | 3 (per E7 seed) | 03, 06, 07 | metrics |
| E9 | Teacher at 4-bit (supplementary, matched memory) | E0 | - | measured | 1 | 03, 06, 07 | metrics |

Scripts: `00_check_teacher_image_reliance`, `01_cache_teacher_outputs`,
`02_train_student`, `03_quantize`, `04_score_pruning_units`, `05_prune_and_recover`,
`06_evaluate`, `07_benchmark_efficiency`; `08_aggregate_results` runs once at the end.
`scripts/run_experiment.py --exp <ID> [--variant V] [--budget B] --seed S` runs an
experiment's stages in order.

"measured" = not budget-matched: the footprint is measured with the budget protocol
and reported relative to B1-B3.

## Student selection (before E1)

**Amended 2026-10-12 (D05 amendment):** the study uses one student, Qwen2.5-VL-3B, after
Round 0 and the image-use checks (RC-S0); Qwen3-VL-2B is the named fallback and the optional
second student. The staged protocol below is kept for the record and is not run.

The staged protocol chooses the student by successive halving on the validation split
(`docs/decisions/D05_student_selection.md`, `configs/student_selection.yaml`,
stage S05b): R0 feasibility of 10 architectures on one T4; R1 the same short LoRA
fine-tune for all 10, keep 5; R2 the top 5 rebuilt as new models with the full
student recipe, keep 3; R3 the top 3 with the exact E2 recipe and 2 seeds, keep 1.
Score: 0.4 probe macro-F1 + 0.3 small-finding recall + 0.3 image reliance (weights
to be confirmed). Selection runs are not part of the E/A ladder; results go to
`results/selection/` and the appendix.

| Block | Runs |
|---|---|
| Student selection (R1: 10, R2: 5, R3: 3 x 2 seeds) | 21 (plus R0 smoke tests) |

## Criterion ablations (variants of E7, at B2)

| ID | Change vs E7 | Config hook | Variants |
|---|---|---|---|
| A1 | Unweighted, w_c = 1 | `pruning.clinical.weighting: uniform` | - |
| A2 | Rarity-only weights | `pruning.clinical.weighting: rarity_only` | - |
| A3 | Random removal | `pruning.criterion: random` | - |
| A4 | Calibration set size | `pruning.calibration.size_by_variant` | `n32`, `n64`, `n128`, `n256` |
| A5 | Delta: correct-answer drop vs prediction change (KL) | `pruning.clinical.delta_by_variant` | `logprob_drop`, `kl` |
| A6 | Layers only vs heads + channel groups | `pruning.units.types_by_variant` | `layers_only`, `heads_channels` |

`A4:n128` and `A5:logprob_drop` have exactly E7's settings (checked by
`tests/test_pruning_ablations.py`). Proposal (team decision in the pre-registration): report
E7's B2 runs for them instead of running them again (`registry.SAME_AS_E7`). A3 ranks its
random scores without dividing by the unit size (otherwise whole layers go first).

## Budgets

B1, B2, B3 are fractions of E2's **measured** peak memory (or latency), placeholders
about 75%, 60% and 50%. The metric, protocol, tolerance and timing hardware are in
`configs/budgets.yaml` and are frozen in `docs/preregistration.md` before any
compression run. E5-E8 run at every budget; ablations at B2.

## Primary endpoints (pre-registered)

Small-finding recall and lesion-mask effect at **B2**, comparing **E7** against the
**best of E5 and E6**, chosen by a pre-registered rule on the validation split (draft proposal:
the generic arm with the highest validation small-finding recall at B2; see
`docs/preregistration.md` section 5).

## Run count (for compute planning)

| Block | Runs |
|---|---|
| Student training (E1, E2, E4) | 3 x 3 = 9 |
| PTQ (E3: 3 variants x 3 seeds; E8: 3 budgets x 3 seeds; E9: 1) | 9 + 9 + 1 = 19 |
| Pruning + recovery (E5: 3 x 3 x 3; E6, E7: 3 x 3 each) | 27 + 9 + 9 = 45 |
| Ablations (A1-A3: 3 each; A4: 12; A5: 6; A6: 6; 6 fewer if `A4:n128` and `A5:logprob_drop` reuse E7) | 33 (27) |

Every run is followed by evaluation (06) and benchmarking (07).

## Order of work

1. Data: manifests, patient-level splits, masks, deduplication (Farzan).
2. E0: image-reliance gate, then teacher cache (Soheil).
3. E1, E2 training; E2 budget reference measurement (Soheil, Amir).
4. Freeze preregistration (finding list, weights, budgets, endpoints, analysis plan).
5. E3, E4, E9 (Soheil); E5, E6, E7, E8 (Amir); ablations (Amir).
6. Evaluation and aggregation (Farzan); final timing on rented GPU (Amir).
