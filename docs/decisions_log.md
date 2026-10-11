# Decisions log

Dated, append-only. Newest entries at the bottom. Changes to pre-registered items
must be logged here.

## Decisions

- **2026-10-12:** chose the merged A+B direction.
- **2026-10-12:** created the initial repository scaffolding (structure, configs,
  documented stubs, docs, paper outline). No methods implemented.

## Open decisions

| Topic | Options / notes | Owner | Due |
|---|---|---|---|
| Medical domain | GI endoscopy (lead) vs retinal fundus | team | TODO |
| Teacher / student | Qwen2.5-VL-7B (+ published Kvasir-VQA-x1 adapter) -> Qwen2.5-VL-3B; Qwen3-VL-8B -> Qwen3-VL-2B; MedGemma family. Must share a tokenizer. | Soheil | TODO |
| Datasets | Kvasir-VQA-x1, Kvasir-SEG, ColonVQA/ColonEval, ColonPert, EndoBench; IDRiD, FGADR (retinal) | Farzan | TODO |
| Finding list and weights | Pre-registered table, frozen before E7 | team | TODO |
| Budgets and timing hardware | B1-B3 fractions, metric (memory vs latency), protocol, rented GPU type | Amir | TODO |
| Quantization backends | GPTQ / AWQ / W8A8 tooling that runs on T4 and the timing GPU | Soheil | TODO |
| Venue | e.g. MICCAI (Springer LNCS template) | team | TODO |
| Licence | Before making the repository public | team | TODO |
