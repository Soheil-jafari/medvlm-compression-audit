# Run card RC-S0b: image size measurement (synthetic images, no dataset)

## 1. What and why

- Stage: S05b (decision on `inputs.image_longest_side`, D05)
- Experiments: none trained; measurements only, at the image sides **448, 560 and 672**
  (multiples of 28, the pixels per visual token of Qwen2.5-VL).
- Purpose: the study is about small findings. At side 448 a polyp of 5 % of the image width is
  about 22 pixels, smaller than one 28 x 28 visual-token patch, so a reviewer could say that the
  input resolution, not the compression, removed the small findings. The side is therefore
  chosen (decision 2 of 2026-10-16) from a cheap measurement of what each side costs, before
  Gate 1 or any real-image run. Nothing here looks at a real image or a result of the study.
- Status: `ready`

## 2. Requirements

- Platform: Kaggle notebook, **GPU T4 x2**, Internet **on**. The student uses **one** T4
  (`CUDA_VISIBLE_DEVICES=0` on its command only); the teacher uses both. Do not `export
  CUDA_VISIBLE_DEVICES` for the whole notebook, or the teacher step sees one GPU only.
- Disk: `export HF_HOME=/kaggle/temp/hf`. Each model downloads once and is reused for all
  three sides (each side loads it again from the local cache).
- Dataset access: none. The images are the synthetic images of RC-M1 and RC-S0 (same
  generator, same seed, same 8 probes), drawn at 1344 x 1008 so that every side shrinks them:
  448 x 336 (exactly the RC-S0 size), 560 x 420 and 672 x 504. The RC-S0 images (448 x 336)
  could not be used as they are: the loader only shrinks images, so they would look the same
  at every side.
- The installs as in RC-S0: if Kaggle's torchao is below 0.16.0, uninstall it first (peft
  0.21.2 needs 0.16.0 or newer); `requirements/train.txt` brings `optimum>=1.24.0`, without
  which peft fails on every LoRA layer next to gptqmodel.
- Expected runtime (an **estimate** from the RC-M1 and RC-S0 timings): the install takes a few
  minutes (not measured before); the student downloads in about 40 s, loads in about 3 s and is
  loaded 4 times per side; its probes take 0.18 s each at 448 and a LoRA step 0.74 s at one
  example, so the student part is a few minutes; the teacher downloads in about 1 min 16 s and
  loads in about 41 s per side, so the teacher part is about 5 minutes. Total about 15 to 25
  minutes, well under one hour. Larger sides take longer per probe; the run measures by how much.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip show torchao 2>/dev/null | grep Version  # Kaggle ships 0.10.0; peft 0.21.2 needs 0.16.0 or newer
pip uninstall -y torchao                     # only if the version above is below 0.16.0 (RC-M1)
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
python -m pytest -q tests/test_image_sides.py tests/test_results.py   # CPU; must pass first

# (1) what the published teacher adapter and the Qwen processors state about image size
#     (print only; paste the whole output into section 6)
python scripts/measure_image_sides.py settings 2>&1 | tee rc_s0b_settings.txt

# (2) the student, ONE T4: scoring, visual tokens, peak memory, one LoRA step at micro-batch
#     1, 2 and 4 (an out-of-memory error is recorded as a result and the run goes on)
CUDA_VISIBLE_DEVICES=0 python scripts/measure_image_sides.py student \
    --config configs/models/qwen2_5_vl_3b_student.yaml --allow-candidate 2>&1 | tee rc_s0b_student.txt

# (3) the teacher (7B + the Kvasir LoRA), both T4s: scoring, visual tokens, peak memory,
#     generation speed
python scripts/measure_image_sides.py teacher \
    --config configs/models/qwen2_5_vl_7b_teacher.yaml 2>&1 | tee rc_s0b_teacher.txt

python -m pytest -q tests/test_results.py                  # must pass before you commit
pip freeze | grep -E "torch|transformers|peft|accelerate|optimum"
```

Each side is written to the CSV as soon as it is done, so a session that ends early keeps the
sides it finished; run the same command again to redo the rest. If a step stops with an error
that is not an out-of-memory of a LoRA step, keep the last 30 lines of its log for section 6
and do not change any setting to get past it.

## 4. What to commit back

- `results/smoke/RC-S0b/student.csv` and `results/smoke/RC-S0b/teacher.csv` (rows
  `RC-S0b:side<N>`, one per metric and side; they never touch the RC-M1, RC-S0 or R0 files)
- This card with section 6 filled, including the whole output of step (1).

Check before committing: `git status` shows no images, models or caches.

## 5. How the result is read

Per side the CSVs hold, for the student: `student_visual_tokens`,
`student_scoring_seconds_per_probe`, `student_peak_gpu_bytes_scoring`,
`student_generate_tokens_per_second`, and for each micro-batch `student_lora_step_ok_mb<b>`
(1 or 0), `student_lora_step_seconds_mb<b>` (if it worked) and
`student_peak_gpu_bytes_lora_step_mb<b>` (for a failed step, the peak reached before the
failure); for the teacher the same scoring rows with the prefix `teacher_`, including
`teacher_generate_tokens_per_second`, which sets the cost of the teacher cache.

Checks: at side 448 the student shows 194 visual tokens, as in RC-S0 (same 448 x 336 image).
Whole 28-pixel patches would give 302 at 560 and 434 at 672; the measurement, not this
arithmetic, is what counts.

**Decision rule (written 2026-10-16, before the run; D05).** For each side the agent computes
the projected GPU hours of the whole study from these numbers and the RC-D1 counts: the teacher
cache, Gate 1, E0 evaluation, E1/E2 training at 10,000 examples, the scoring of D07, the
recoveries, every evaluation, and the training-length check (pre-registration section 6). The
team states the GPU hours it has (`TODO: team GPU hours available`) and picks the **largest
side whose projected total fits**. If the D06 micro-batch of 4 does not fit one T4 at that side,
the agent proposes micro-batch 2 (or 1) with gradient accumulation 8 (or 16), which keeps 16
examples per update: the same recipe up to floating-point rounding (the model has no batch
normalisation). That is a D06 amendment **proposal** (D06 memo); D06 stays as it is until the
team decides.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft|accelerate|optimum"`):
- Step (1) output (paste all of it; "not stated" lines included):
- Step (2): wall time, and for each side whether micro-batch 4 fitted (with the error line if not):
- Step (3): wall time:
- Session wall time:
- Anything unusual:
