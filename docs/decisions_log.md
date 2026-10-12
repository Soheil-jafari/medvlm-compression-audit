# Decisions log

Dated, append-only. Newest entries at the bottom. Changes to pre-registered items
must be logged here.

## Decisions

- **2026-10-12:** chose the merged A+B direction.
- **2026-10-12:** created the initial repository scaffolding (structure, configs,
  documented stubs, docs, paper outline). No methods implemented.
- **2026-10-12:** fixed the `.gitignore` rule that had kept `src/medcompress/data/` and
  `configs/data/` out of git (CI was failing on import); restored both.
- **2026-10-12:** set up the research loop: definition of done in
  `docs/FINAL_GOAL.md`, stages S00-S12 in `docs/stages/`, live state in
  `docs/agent/STATUS.md`. The loop works only on the `research-loop` branch; the
  team reviews it and merges into `main` through pull requests.
- **2026-10-12 (Soheil):** the student will be chosen by a staged selection after the
  teacher is fixed: 10 architectures compared in a simplified common screen, the top
  5 rebuilt as new models, then top 3, then one. Protocol drafted in
  `docs/decisions/D05_student_selection.md` (answers pending); D01 now covers the
  teacher only.
- **2026-10-12 (Soheil):** D05 answered: students only from the teacher's family
  (same tokenizer), each winner keeps its size, score weights 0.4 / 0.3 / 0.3.
- **2026-10-12 (Soheil):** D01 decided: teacher = Qwen2.5-VL-7B-Instruct
  (revision cc594898) + SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft LoRA (revision 79e77755).
  Optional robustness check at the end: a fine-tuned Qwen3-VL-8B teacher.
  Consequence (D05): the student pool is 10 Qwen-tokenizer models.
- **2026-10-12 (Soheil):** D02 decided: domain = GI endoscopy; main QA = Kvasir-VQA-x1
  (revision 61e41148), validation and test = the two halves of its official test
  split by image; masks = Kvasir-SEG matched by fingerprint; rare findings from
  Kvasir-VQA-x1 classes; external test = EndoBench (licence to verify); retinal kept
  as an optional later second domain.
- **2026-10-12 (Soheil):** D03 decided: MIT licence for the code.
- **2026-10-12 (Soheil):** D04 answered: Gate 1 moderate rule (balanced accuracy at
  least 0.65, blank-image drop at least 0.10, lesion-mask effect CI above zero), mid-grey
  fill (128), teacher cache `top_k` 20. `max_new_tokens` follows the rule in D04 once
  the RC-D1 answer lengths are known.
- **2026-10-12 (Soheil):** no scheduled runs; the research loop runs back to back in
  an interactive session until the team stops it.
- **2026-10-12 (Soheil):** D06 answered, all recommendations: LoRA rank 16 / alpha 32 /
  dropout 0.05; AdamW lr 1e-4, linear decay, warm-up 20, micro-batch 4 x 4, KD weight 0.5,
  temperature 2; recovery = 25 % of the E2 steps; 10 000 bootstrap resamples, 15 ECE bins,
  1 000 free-generation rows. `steps` and `data_subset` follow at the selection freeze.
- **2026-10-12 (Soheil):** D07 partly answered: rank by score per parameter for every
  criterion; first 2 and last 1 decoder layers protected for every arm; MLP channel groups of
  128, 128 calibration images, one-shot removal, 1 negative per positive, signed effects,
  probe batch 4. Open: how units are switched off while scored (cost question, waits for the
  RC-M1 speed) and the budgets. Soheil also asked that nothing heavy runs in the agent session:
  all GPU work stays in run cards for the team.

## Open decisions

| Topic | Options / notes | Owner | Due |
|---|---|---|---|
| Student selection freeze | Set `train_examples`, `compute_unit_b`, G2 threshold and image side from RC-S0, then freeze (memo D05) | team | after RC-S0 |
| EndoBench terms | Confirm licence and download before it is used as the external test | Farzan | before S11 |
| Finding list and weights | Pre-registered table, frozen before E7 | team | TODO |
| Budgets and timing hardware | B1-B3 fractions, metric (memory vs latency), protocol, rented GPU type | Amir | TODO |
| Quantization backends | GPTQ / AWQ / W8A8 tooling that runs on T4 and the timing GPU | Soheil | TODO |
| Venue | e.g. MICCAI (Springer LNCS template) | team | TODO |
