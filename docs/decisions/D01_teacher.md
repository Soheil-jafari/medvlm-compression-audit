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

## Amendment (2026-10-18, Soheil): our own teacher adapter, and rule 3

**Why.** The published SimulaMet adapter was trained on the official train split. That split
holds 3,821 of the 4,058 official test images and almost every image of any image-disjoint
split (D02 amendment, same day). Teaching the students with it would leak test images through
the teacher.

**Decision.**
1. **The teacher.** Qwen2.5-VL-7B-Instruct at the same pinned revision, with a LoRA adapter
   **we** train on our training split only, at image side 672 (the students' input).
   - Recipe from the SimulaMet `args.json` where it applies: LoRA on the language model, vision
     tower frozen, lr 2e-5, cosine schedule, 3 % warm-up, weight decay 0.1.
   - LoRA rank, alpha and dropout: D06's 16 / 32 / 0.05, unless the published
     `adapter_config.json` differs and can be read on Kaggle. Then RC-TT0 prints it and its
     values are used.
   - 16 examples per update (micro-batch 1 x 16, as the students); fp16 on T4 with loss
     scaling and fp32 LoRA weights.
   - `configs/teacher_training.yaml`; run cards RC-TT0 and RC-TT1.
2. **Rule 3** (pre-registered, fixed before any validation result):
   - **Setup:** prefer (a), the fp16 base split over 2 x T4. Use (b), QLoRA with an NF4 base on
     1 x T4, only if (a) runs out of memory or is more than 2x slower per example. If (b), the
     teacher is still evaluated in fp16 with the trained adapter, and the card reports the
     validation balanced accuracy under both NF4 and fp16 for the record.
   - **Size:** N is the largest multiple of 1,000 training examples whose projected time, from
     RC-TT0's warm step time, is at most 15 GPU hours, capped at one pass over the training
     split's QA rows. A T4 x2 hour counts as 2 GPU hours until the team has checked Kaggle's
     rule; RC-TT0 records the other reading next to it.
   - **Examples:** drawn from the training split with seed 20261012, stratified by question
     type, so rare types are included: one seeded order, of which N is the first part
     (`models/teacher_training.py: teacher_order`). The students' 10,000 / 30,000 pools stay as
     they are.
   - **Quality:** the teacher must pass Gate 1 (D04 thresholds unchanged) and, once E1 seed 0
     exists, have a validation macro balanced accuracy of at least E1 seed 0's + 0.02. If
     either fails, it is trained **once** more from the last checkpoint for +50 % of N:
     - the next N / 2 examples of the same order;
     - starting from the trained adapter;
     - with its own 3 % warm-up and cosine schedule;
     - then re-checked, and accepted and reported either way.

     The study's stop rule for a failed Gate 1 stays as it is after that one extension.
3. **Appendix item** (secondary, pre-registered): the published SimulaMet adapter
   (`configs/models/qwen2_5_vl_7b_teacher_simulamet.yaml`) is evaluated on our new test split,
   labelled "has seen these images in training", next to our teacher. The overlap counts of
   the official split go into the data card. The published adapter has no other use.

## Amendment 2 (2026-10-19, Soheil): RC-TT0's results; the published targets; N

From RC-TT0 (`results/smoke/RC-TT0/teacher_training.csv`, commit 3afb035):
- `lora_rank` 16, `lora_alpha` 64.0, `lora_dropout` 0.05, `lora_from_published` 1: the
  published `adapter_config.json` was read. D06's defaults were rank 16, alpha 32, dropout 0.05,
  so only alpha changes (32 -> 64).
- `lora_targets_match` 0: the published `target_modules` value is neither the string
  `all-linear` nor exactly the list of D06's targets, the seven projections of the language
  model (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). The check
  of that commit did not parse a regular expression, so a regular expression over those seven
  also gives 0. The `args.json` read on 2026-10-12 says "all-linear" with the vision encoder
  frozen. The value itself is in RC-TT0's printout of the `lora_config` step, which is not
  committed, and the hub cannot be reached from the agent's session: **which modules it names
  is open until RC-TT1 prints it** (or the RC-TT0 printout is pasted).
- Setup (a), fp16 over two T4s, without gradient checkpointing: 1.39 s per example (warm).
- A T4 x2 hour counts once against the Kaggle quota (measured 2026-10-19: a 17-minute T4 x2
  session used 16 minutes of quota).

**Decision.**
1. RC-TT1 uses the **published target modules**, with the published rank 16, alpha 64 and
   dropout 0.05, in line with the amendment of 2026-10-18 (item 2 of that day in the decisions
   log, item 1 above): the recipe follows the published one where it applies. The run reads `adapter_config.json` at the pinned revision,
   prints `target_modules` as written, and adapts those module names inside the language
   model's decoder layers; the vision tower and the projector stay frozen, as the published
   `args.json` froze the vision encoder. It records the adapted layers per name, and stops
   before any training if a name matches no linear layer there (a decision is then needed),
   or if the file's rank, alpha or dropout differ from RC-TT0's.
2. N = **38,000** examples (RC-TT0's `rule_n_examples_if_t4x2_counts_once`): rule 3 with a
   T4 x2 hour counted once. 2,375 updates of 16; about 14.7 h of wall time and of quota
   (**estimate** from RC-TT0's warm step time), within the 15 GPU hours of rule 3.

Applied in `configs/teacher_training.yaml` (`target_modules: published`,
`rule.t4x2_counts_twice: false`), `models/teacher_training.py` (`published_target_names`,
`adapted_counts`, `t4x2_counts_once`), `compression/lora.py` (`layer_modules`, `layer_regex`),
`scripts/train_teacher.py` and run card RC-TT1.

## Answer format for the inbox

`- YYYY-MM-DD (name): D01 decided: <T1 | T2 | T3 | T4 | T5>.`
