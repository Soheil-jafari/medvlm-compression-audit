# D08: Quantization arms (E3, E4, E8, E9): tools, settings and calibration

**Status:** answered 2026-10-12 (Soheil): E3 with GPTQ 4/8 bit (GPTQModel) and LLM.int8 for
`w8a8`; E4 option **B** (reduced GRACE with confidence-gated decoupled distillation on the cached
top-20); 128 calibration images shared with pruning; `w8a8` kept as LLM.int8. **Stage:** S08.
**Depends on:** D05 (student), D06 (training recipe, used again by E4), D07 (calibration
images). All numbers are fixed **before** any quantization result; after the answer they go
into `configs/experiments/E3.yaml`, `E4.yaml`, `E8.yaml`, `E9.yaml` and are frozen with the
pre-registration.

## Decision (2026-10-12)

| Question | Answer |
|---|---|
| E3 tools | `w4`, `w8`: GPTQ with GPTQModel, group size 128; `w8a8`: bitsandbytes LLM.int8 |
| E4 scope | B: 4-bit group-wise learned-step-size quantization (group 128) of the decoder's linear layers, trained with LoRA through the quantizer, with GRACE's confidence-gated decoupled distillation computed on the cached top-20 (details below) |
| Calibration | 128 images, the same nested set as the pruning criteria (D07) |
| `w8a8` | kept, reported as "8-bit weights and activations with 16-bit outliers" (LLM.int8) |

**Toolchain fixed by RC-Q0 (2026-10-15, Kaggle 1 x T4).** Kernel class
`gptqmodel.nn_modules.qlinear.torch.TorchLinear` (`quantization.kernel_class`): the fastest at
`w4` (0.238 ms; `w8` 0.201 ms) among the classes that passed both widths (TritonV2Linear also
did, at 0.240 ms; TorchAtenLinear passed `w4` only). The kernel check found TorchLinear on all
252 decoder linear layers for `w4` and `w8`, and LLM.int8 on all 252 for `w8a8`; the answers on
synthetic images matched the 16-bit model for all three recipes. Pinned in
`requirements/train.txt`: gptqmodel 7.6.0 and bitsandbytes 0.50.2; tested with torch
2.11.0+cu128, transformers 5.19.0, peft 0.21.2, accelerate 1.14.0 and torchao 0.18.0
(`quantization.library_versions`). Kaggle's preinstalled torchao 0.10.0 breaks peft 0.21.2,
which needs 0.16.0 or newer; the requirements now ask for that.

**E4 loss (option B), taken from GRACE's code** (`qwen-vl-finetune/qwenvl/train/grace_modules.py`,
read 2026-10-12) and adapted to the cache:

- per answer token, teacher-forced on the teacher's cached answer, the decoupled distillation
  term `(alpha * TCKD + beta * NCKD) * T^2` with GRACE's defaults `alpha = 1`, `beta = 4`,
  `T = 2`. TCKD is the KL divergence between teacher and student of the two-way split
  {target token, everything else}; NCKD the KL divergence of the distributions over the
  non-target tokens. Target = the teacher's cached token at that step.
- the token's weight (the "confidence gate") is `exp(-H / log V)`, where `H` is the entropy of
  the teacher's distribution at temperature 1 and `V` the vocabulary size; positions are
  averaged with these weights.
- **Adaptation to the cache:** the teacher distribution is known only on its top-20 tokens.
  The student is restricted to the same 20 tokens (as in E2's KL), and the entropy treats the
  probability mass outside the top 20 as one extra outcome. This is an approximation of GRACE,
  reported as a deviation.
- total loss = `(1 - kd_weight) * CE + kd_weight * GDKD` with the E2 weight 0.5 (GRACE adapts the
  weight online; we keep D06's fixed weight, also a deviation). No relational CKA term (needs
  the teacher's visual tokens live).

## Question

Which software quantizes the student (E3, E4, E8) and the teacher (E9), with which settings,
so that every arm still runs through **the same evaluation and timing code**?

In plain words: quantization stores each weight with fewer bits (8 or 4 instead of 16), like
saving a photo with fewer colours. Two kinds exist:

- **after training** ("post-training quantization", E3, E8, E9): take the finished model,
  look at a few calibration examples, and round the weights so the outputs change as little
  as possible. GPTQ and AWQ are two recipes for this rounding.
- **during training** ("quantization-aware training", E4): train the model while it already
  sees its rounded weights, so it learns to cope with them.

## The constraint that decides most of it

Every arm is evaluated by the same Hugging Face `transformers` code (yes/no probes, mask
audit, free text) and timed by the same code. A quantized model only counts if it runs
**real low-precision kernels** inside `transformers` (otherwise the memory and speed numbers
are those of a 16-bit model). The development GPU is a Kaggle T4 (Turing architecture,
"sm_75"); the final timing GPU is still open.

## What the sources say (read 2026-10-12)

1. **`transformers` quantization overview** (docs, `quantization/overview.md`): GPT-QModel
   supports 2/3/4/8 bits on CUDA, can be saved and fine-tuned with adapters; bitsandbytes
   supports 4/8 bits on the fly (no calibration); AWQ 4 bits; compressed-tensors 1 to 8 bits.
2. **GPTQModel README** (github.com/ModelCloud/GPTQModel; latest release 7.5.0, 2026-09-15):
   methods "GPTQ, AWQ, ParoQuant, GGUF, FP8, EXL3, QQQ, and more"; NVIDIA needs "`Turing+`
   (`sm_75+`)"; "`Marlin` and JIT CUDA kernels now support NVIDIA `Turing+` (`sm_75+`) GPUs";
   Qwen 2 / 2.5 / 3 VL are in the model table (type `qwen2_5_vl`); its example uses
   `bits=4, group_size=128` and 1,024 calibration samples; models "are inference compatible
   with HF Transformers (minus `dynamic`)".
3. **AutoAWQ** is deprecated and archived; the vLLM documentation says its functionality was
   adopted by llm-compressor. It was last tested with Torch 2.6.0 and Transformers 4.51.3, so
   it is not an option next to our pinned `transformers` 5.19.
4. **llm-compressor README** (github.com/vllm-project/llm-compressor): W8A8 (int8 and fp8),
   W4A16, W8A16, with GPTQ, AWQ and SmoothQuant; checkpoints are "compatible with vLLM". It
   does not say that `transformers` runs them with low-precision kernels.
5. **`transformers` compressed-tensors docs** (`quantization/compressed_tensors.md`): by
   default the model "decompresses the model on the first forward pass" (16-bit matrix
   products); the kernel path exists only for "W8A8 fp8" and needs "Ada Lovelace (L4, L40),
   Hopper (H100), Blackwell and newer"; int8 is a storage format there, not a kernel path.
   So **int8 W8A8 from llm-compressor would not run int8 kernels in `transformers`**, and fp8
   cannot run on a T4.
6. **bitsandbytes README**: NVIDIA "SM60+ minimum", "SM75+ recommended"; LLM.int8() is
   "based on vector-wise quantization" and treats outliers "with 16-bit matrix
   multiplication". The paper (Dettmers et al., *LLM.int8(): 8-bit Matrix Multiplication for
   Transformers at Scale*, arXiv:2208.07339) states that more than 99.9 % of values are still
   multiplied in 8-bit, so it quantizes the activations of those products as well.
7. **GRACE** (Chen, Habibian, Benini, Li, *Gated Relational Alignment via Confidence-based
   Distillation for Efficient VLMs*, ICML 2026, arXiv:2601.22709; code
   github.com/ForeverBlue816/GRACE, Apache-2.0): loss = cross-entropy + confidence-gated
   decoupled distillation + relational alignment (CKA) of the penultimate-layer **visual
   tokens** of teacher and student, with an adaptive weight; quantizer "Group-wise LSQ QAT"
   (learned step size per group of 128 weights) at 4 or 8 bits, vision encoder frozen;
   trained on "16 × A100 (64 GB)".

## Options and proposal

### 1. Engine

**Proposal: `transformers` for every arm** (the same probes, audit and timing code). vLLM is
not used: running only the quantized arms in another engine would make their latency
incomparable with the pruned arms.

### 2. E3 variants (post-training quantization of E2)

| Variant | Proposal | Alternatives | Why |
|---|---|---|---|
| `w4` | **GPTQ, 4 bit, group size 128**, with GPTQModel (source 2) | AWQ 4 bit with GPTQModel; bitsandbytes NF4 (no calibration) | GPTQModel is maintained, lists Qwen2.5-VL, runs on Turing and inside `transformers`; AutoAWQ is archived (3) |
| `w8` | **GPTQ, 8 bit, group size 128**, with GPTQModel | bitsandbytes 8 bit | same tool and recipe as `w4`, so the two differ only in bits |
| `w8a8` | **bitsandbytes LLM.int8()** (6), reported as "8-bit weights and activations with 16-bit outliers" | drop `w8a8` (the plan allows "where supported") | the only 8-bit activation path that runs real kernels in `transformers` on a T4 (5, 6). It is dynamic, not a calibrated static W8A8; this is stated as a deviation |

What is quantized, for every quantized arm: the **decoder's linear layers** (attention q, k, v,
o and MLP gate, up, down). The vision encoder, the projector, the embeddings and the output
head stay at 16 bit. Reason: the pruning arms only touch the decoder, so both families
compress the same part of the model; the published llm-compressor recipes for Qwen2.5-VL also
leave the vision tower and the output head out.

Calibration (`w4`, `w8`): **the same 128 calibration images as the pruning criteria** (D07,
nested set from the training pool), each with one of its training questions and the
ground-truth answer in the chat template. Reason: both families then see the same data. The
alternatives are 512 (the size of public Qwen2.5-VL recipes) or 1,024 (the GPTQModel example);
more calibration data can help GPTQ, but would give quantization more data than pruning.

### 3. E8 and E9

- **E8** = each E7 checkpoint (every budget and seed) + the E3 `w4` recipe, calibrated on the
  same 128 images.
- **E9** = the teacher (7B, adapter merged) + the E3 `w4` recipe. Supplementary (one run).

### 4. E4 (quantization-aware distillation)

A faithful GRACE needs the teacher running **live** during training (the CKA term compares
the teacher's visual tokens) and full training of all weights; it was trained on 16 A100s.
Our teacher is only available as a cache of its top-20 answer probabilities, and a T4 cannot
hold a full fine-tune of a 3B model. So:

| Option | What | Cost and risk |
|---|---|---|
| **A (proposed)** | "GRACE-style, reduced to what fits": 4-bit group-wise learned-step-size quantization (group 128) of the decoder's linear layers, trained with LoRA *through* the quantizer (the forward pass uses the rounded version of weight + LoRA, so the final merged and rounded model is exactly the trained one), with the **E2 loss** (cross-entropy + 0.5 x top-20 KL, temperature 2) on the same training pool and steps as E2. Exported in the GPTQ format so E3 `w4` and E4 run the same kernel | about the cost of E2 plus the rounding in every step (an **estimate** until RC-Q0); leaves out the CKA term and the adaptive controller (reported as deviations) |
| B | A + GRACE's confidence-gated decoupled distillation, computed on the cached top-20 | closer to GRACE, more code; an approximation, because the cache holds 20 tokens per step |
| C | drop E4 | saves 3 runs; the paper loses the "quantization during training" family |

### 5. The low-precision kernel check

`check_low_precision_kernels` passes only if every targeted decoder linear layer has been
replaced by the backend's quantized module (not a plain 16-bit `Linear`) and stores its weights
in an integer type, and if the decoder's measured bytes fell by at least half (8 bit) or three
quarters (4 bit) minus a margin. It records the module class per layer next to the numbers.
A model that fails the check is not reported as quantized.

### 6. First, a smoke test (RC-Q0)

Before the 13 quantization runs: one Kaggle T4 session that installs GPTQModel and
bitsandbytes next to our pinned `transformers`, quantizes the student with `w4`, `w8` and
`w8a8` on 32 calibration examples, loads each in `transformers`, runs the kernel check, scores
64 probes and measures peak memory. It also pins the library versions. Runtime: **estimate**
under an hour; the card reports the real number.

## Risks

- **Versions:** GPTQModel and bitsandbytes must work with `transformers` 5.19 and torch 2.8;
  not verified here (the session has no GPU). RC-Q0 checks this before anything else.
- **The timing GPU:** if it is an Ada or Hopper GPU, fp8 W8A8 would become possible there,
  but the T4 cannot evaluate it, so it is not proposed. The checkpoints quantized on the T4
  are the ones timed on the rented GPU (Marlin covers newer GPUs as well, source 2).
- **LLM.int8 speed:** LLM.int8 often runs slower than 16 bit because of the outlier split;
  the paper reports what is measured, not what is expected.

## Recommendation

Engine `transformers`; E3 `w4` and `w8` with GPTQ (GPTQModel, group 128); `w8a8` with
LLM.int8; decoder linear layers only; 128 calibration images shared with pruning; E8 and E9
with the `w4` recipe; E4 option A; RC-Q0 first.

## What changes once decided

`configs/experiments/E3.yaml`, `E4.yaml`, `E8.yaml`, `E9.yaml` (backends, bits, group size,
calibration size, quantized modules), `requirements/train.txt` (pinned versions after RC-Q0),
`compression/quantization/ptq.py` and `qakd.py`, run cards RC-Q0, RC-E3, RC-E4, RC-E9.

## Sources (accessed 2026-10-12)

1. `transformers` documentation, `docs/source/en/quantization/overview.md` (main branch, GitHub).
2. ModelCloud, GPTQModel README, github.com/ModelCloud/GPTQModel (release 7.5.0).
3. vLLM documentation, "AutoAWQ" page (docs.vllm.ai, features/quantization/auto_awq).
4. vLLM project, llm-compressor README, github.com/vllm-project/llm-compressor.
5. `transformers` documentation, `docs/source/en/quantization/compressed_tensors.md`.
6. bitsandbytes README, github.com/bitsandbytes-foundation/bitsandbytes (tag 0.50.0);
   Dettmers, Lewis, Belkada, Zettlemoyer, arXiv:2208.07339 (NeurIPS 2022).
7. Chen, Habibian, Benini, Li, arXiv:2601.22709, ICML 2026 (PMLR 306); github.com/ForeverBlue816/GRACE.
