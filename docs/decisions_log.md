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
- **2026-10-12 (Soheil):** D07 section 4 partly answered: the budget metric is peak GPU
  memory during inference, the tolerance is 3 % of the target, and the protocol is batch 1,
  3 warm-up and 5 timed passes, 32 validation images drawn by a fixed seed (the same for every
  arm). Written to `configs/budgets.yaml`. Still open: the timing hardware, B1 to B3 (rule after
  RC-B0), `max_new_tokens` (rule of D04) and the scoring method (not needed before E2 exists).
- **2026-10-12 (Soheil):** asked whether the 10 -> 5 -> 3 -> 1 selection is needed, because the
  paper asks what compression does to a student, not which student is best. Agreed that it is
  not needed for the research question. The numbers 1 to 10 in D05 are the order of listing,
  not a ranking: nothing has been measured. An amendment is proposed in D05 (Qwen2.5-VL-3B
  with Round 0 and the image checks, Qwen3-VL-2B as the named fallback, no ranking); it is
  applied only after Soheil confirms it and Amir and Farzan are told.
- **2026-10-12 (Soheil):** D05 amendment accepted: pick one student. Qwen2.5-VL-3B-Instruct
  with Round 0 (RC-S0) and the image-use checks; Qwen3-VL-2B-Instruct is the named fallback
  and the optional second student. Rounds R1 to R3 are not run. Applied to the base and
  training configs; the training length and the image size follow from the RC-S0 numbers. Amir
  and Farzan are to be told.
- **2026-10-12 (Soheil):** D08 answered: engine `transformers` for every arm; E3 `w4` and `w8`
  with GPTQ (GPTQModel, group size 128), `w8a8` with bitsandbytes LLM.int8 (reported as 8-bit
  with 16-bit outliers); only the decoder's linear layers are quantized; calibration = the same
  128 images as the pruning criteria; E8 and E9 use the `w4` recipe; E4 = option B (4-bit
  learned-step-size quantization trained with LoRA and GRACE's confidence-gated decoupled
  distillation on the cached top-20). A smoke test (RC-Q0) pins the library versions first.

## Open decisions

| Topic | Options / notes | Owner | Due |
|---|---|---|---|
| Student checks | Run RC-S0 for the two models, then set the image side, the G2 threshold and the training length from its numbers; tell Amir and Farzan about the amendment | Soheil | after RC-M1 |
| Fixed training pool (agent proposal, iteration 36) | E1, E2, E4 and every recovery draw their examples from one pool fixed by `seeds.data_split` (20261012, the RC-D1 seed), not from the run seed. The teacher cache holds only that pool (`teacher_cache.pool_size` = `training.data_subset`) instead of all 143,594 training rows; the three seeds vary the order and the adapter's start. Alternative: a different pool per seed (the cache then needs the union of three pools) | Soheil | before RC-E0b |
| EndoBench terms | Confirm licence and download before it is used as the external test | Farzan | before S11 |
| Finding list and weights | Pre-registered table, frozen before E7. Draft rules (iteration 40): a finding enters C with at least 20 positive training images and 10 in each of validation and test; groups common >= 10 %, rare < 2 %; cap 20; three tiers with multipliers 2.0 / 1.5 / 1.0 assigned from the finding names | team | after RC-D1 |
| Comparator "best of E5/E6" (draft proposal) | the generic arm with the highest validation small-finding recall at B2 (mean of 3 seeds), the same for both primary endpoints | team | before the freeze |
| Headline accuracy for H1 and H4 (draft proposal) | macro balanced accuracy instead of the overall probe accuracy, which mostly measures specificity; overall accuracy still reported | team | before the freeze |
| A3 ranking (agent proposal, iteration 42) | A3 orders its random scores without dividing by the unit size (`score_normalisation: none`). Divided by size, a random score sends whole layers first (estimate: about 98 % of the removed parameters at a 30 % cut), so A3 would test random layer removal, not random choice of units. D07's per-parameter rule stays for every other arm | team | before the freeze |
| Reuse of E7 for identical ablation arms (agent proposal) | `A4:n128` and `A5:logprob_drop` have exactly E7's settings (test), so E7's B2 runs are reported for them (27 ablation runs instead of 33). If D07 section 1 uses the first-order estimate for some units, A5 moves to the exactly scored units and `A5:logprob_drop` is run | team | before the freeze |
| Budgets and timing hardware | B1-B3 fractions (rule after RC-B0) and the rented GPU type; metric, tolerance and protocol are decided | Amir | TODO |
| Venue | e.g. MICCAI (Springer LNCS template) | team | TODO |
