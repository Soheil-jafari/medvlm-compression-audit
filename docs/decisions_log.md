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
- **2026-10-12 (Soheil):** no scheduled runs; the research loop runs back to back in
  an interactive session until the team stops it.

## Open decisions

| Topic | Options / notes | Owner | Due |
|---|---|---|---|
| Medical domain | GI endoscopy (lead) vs retinal fundus | team | TODO |
| Teacher | Qwen2.5-VL-7B (+ published Kvasir-VQA-x1 adapter); Qwen3-VL-8B; MedGemma family (memo D01) | Soheil | TODO |
| Student selection protocol | 10 candidates, 10 -> 5 -> 3 -> 1 on validation; tokenizer rule, size rule, score weights (memo D05) | team | TODO |
| Datasets | Kvasir-VQA-x1, Kvasir-SEG, ColonVQA/ColonEval, ColonPert, EndoBench; IDRiD, FGADR (retinal) | Farzan | TODO |
| Finding list and weights | Pre-registered table, frozen before E7 | team | TODO |
| Budgets and timing hardware | B1-B3 fractions, metric (memory vs latency), protocol, rented GPU type | Amir | TODO |
| Quantization backends | GPTQ / AWQ / W8A8 tooling that runs on T4 and the timing GPU | Soheil | TODO |
| Venue | e.g. MICCAI (Springer LNCS template) | team | TODO |
| Licence | Repository is already public, so this is urgent (memo D03) | team | TODO |
