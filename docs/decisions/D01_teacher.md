# D01: Teacher

**Status:** decided 2026-10-12 (Soheil): **T1**, Qwen2.5-VL-7B-Instruct + SimulaMet
Kvasir-VQA-x1 LoRA. Optional at the end, if budget remains: rerun the key comparison
with a fine-tuned Qwen3-VL-8B teacher to show the conclusion does not depend on the
teacher.
**Stage:** S01. **Prepared:** 2026-10-12 (iteration 5).
**Facts:** read from Hugging Face API records, `config.json`, `args.json` and model
cards on 2026-10-12. Links in the tables.

## Question

Which model is the teacher? The teacher answers every training question once; its
answers and confidence are saved and used to train the student (E2) and to repair
pruned models (recovery). It only has to run, never train, on Kaggle's 2x T4.
Because of D05, the teacher's family also decides which students can compete.

## Options

| | T1 Qwen2.5-VL-7B + Kvasir adapter | T2 Qwen3-VL-8B | T3 Lingshu-7B | T4 MedGemma 27B | T5 Qwen3.5-9B |
|---|---|---|---|---|---|
| Hub ID | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/api/models/Qwen/Qwen2.5-VL-7B-Instruct) + [SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft](https://huggingface.co/SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft) | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct) | [lingshu-medical-mllm/Lingshu-7B](https://huggingface.co/api/models/lingshu-medical-mllm/Lingshu-7B) | google MedGemma 27B multimodal (TODO: verify ID) | Qwen/Qwen3.5-9B |
| Stored params | 8.29B (+ LoRA) | 8.77B | 8.29B (Qwen2.5-VL architecture) | about 27B | about 9B |
| Already tuned on our endoscopy data | **Yes**: LoRA rank 16, alpha 64, all linear layers, vision encoder frozen, 3 epochs, lr 2e-5, bf16, trained on the Kvasir-VQA-x1 **train** file (`args.json`) | No (none found on the hub) | No (general medical model) | No | No |
| Licence | apache-2.0 base, mit adapter | apache-2.0 | mit | Health AI Developer Foundations terms | apache-2.0 |
| Fits Kaggle 2x T4 for inference | fp16 split over both GPUs (about 16.6 GB of weights), or 8-bit on one | Same as T1 | Same as T1 | Only at 4-bit, tight | Probably, like T2 |
| Students left by D05 (same family) | Qwen tokenizer family: 6 of 10 now, top-up possible | Qwen tokenizer family (same as T1) | Same as T1 | Gemma family only (about 1-3 small models) | Qwen3.5 only (0.8B, 2B, 4B, one architecture) |
| What we would have to do | Nothing before Gate 1 | Fine-tune an 8B model on T4s first | Fine-tune or use it untuned | Fine-tune a 27B model: not feasible | Fine-tune first |

### Tokenizer check (file hashes from the hub, `?blobs=true`)

Same hash = byte-identical file.

| Model | `vocab.json` | `merges.txt` | Reading |
|---|---|---|---|
| Qwen2.5-VL-7B (T1) | 4783fe10 | 20024bfe | reference |
| Qwen2.5-VL-3B | 4783fe10 | 20024bfe | identical tokenizer files (all four files match) |
| Qwen3-VL-2B | 4783fe10 | 20024bfe | same vocabulary and merges; extra special tokens |
| InternVL3.5-2B | 4783fe10 | 31349551 | same vocabulary; merges differ only by a `#version: 0.2` header line; extra special tokens |
| Ovis2.5-2B | 4783fe10 | 31349551 | as InternVL3.5 |
| Hulu-Med-4B | 6bce3a0a | 31349551 | merges as InternVL3.5; vocabulary file differs byte-wise (TODO: compare contents) |
| FastVLM-1.5B | 6bce3a0a | 8488ce08 | both files differ byte-wise (TODO: compare contents) |

The final check is done at runtime in S04 (encode a fixed text with both tokenizers
and compare token IDs). Qwen2.5-VL-7B has 152,064 embedding rows and the 3B has
151,936; only rows that real tokens use are compared during distillation.

## Recommendation

**T1: Qwen2.5-VL-7B-Instruct with the SimulaMet Kvasir-VQA-x1 LoRA adapter.**

- It is the only teacher already trained on our main dataset, so no teacher training
  is needed on Kaggle; the next GPU step can be Gate 1 directly.
- It was trained on the **train** file only, so the official test split stays
  unseen (to confirm against the dataset's split files in D02).
- Its family has the most student candidates (D05).
- It runs on Kaggle's 2x T4 for inference.

T2 is the fallback if T1 fails Gate 1 (newer model, same student pool, but it must
be fine-tuned first).

## Risks

- **The reported score is not independent.** The adapter card reports 85.78%
  accuracy (graded by another LLM) on a 1% slice held out from its own training
  data. Our E0 evaluation on the validation split and Gate 1 are the real checks.
- **Card inconsistency.** The hub's model tree names the text-only
  `Qwen/Qwen2.5-7B-Instruct`, but the card's table and `args.json` say
  `Qwen/Qwen2.5-VL-7B-Instruct`. `args.json` is the training record, so the
  vision-language model is taken as the base; RC-M1 confirms the adapter loads.
- **bf16 training, fp16 hardware.** T4s have no bf16. RC-M1 checks for overflow in
  fp16; fallback is 8-bit loading or fp32 for the unstable layers.
- **Prompt format.** The adapter was trained with ms-swift (`ms-swift==3.8.0` in
  the card) and the `qwen2_5_vl` template; the same chat template must be used, or
  quality drops.
- **Sharp outputs on training images.** Three epochs on the train split may make the
  teacher very confident there, which carries less information for distillation.
  A softening temperature can be set in the distillation recipe (before the freeze).

## What changes in the repository once decided

- `configs/models/qwen2_5_vl_7b_teacher.yaml`: status `DECIDED`, pinned revisions of
  the base model and the adapter.
- `configs/base.yaml: components.teacher` points at it.
- D05: the student pool keeps only teacher-family models and is topped up to 10.
- Run card RC-M1 (load test) can be written.

## Answer format for the inbox

`- YYYY-MM-DD (name): D01 decided: <T1 | T2 | T3 | T4 | T5>.`
