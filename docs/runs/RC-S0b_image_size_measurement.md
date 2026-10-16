# Run card RC-S0b: image size measurement (synthetic images, no dataset)

## 1. What and why

- Stage: S05b (decision on `inputs.image_longest_side`, D05)
- Experiments: none trained; measurements only, at the image sides **448, 560 and 672**
  (multiples of 28, the pixels per visual token of Qwen2.5-VL). Every (side, step) runs in a
  child process of its own, so the GPU memory of one step is fully freed before the next.
- Purpose: the study is about small findings. At side 448 a polyp of 5 % of the image width is
  about 22 pixels, smaller than one 28 x 28 visual-token patch, so a reviewer could say that the
  input resolution, not the compression, removed the small findings. The side is therefore
  chosen (decision 2 of 2026-10-16) from a cheap measurement of what each side costs, before
  Gate 1 or any real-image run. Nothing here looks at a real image or a result of the study.
- Status: `done` (run 1 on 2026-10-16 was stopped by a script bug, section 6; run again)

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
- Expected runtime (an **estimate** from the RC-M1, RC-S0 and run 1 timings; the import time
  of a child was not measured): the install takes a few minutes (not measured). The student is
  downloaded once (about 40 s, RC-M1) and then measured in **21 child processes** (7 steps x
  3 sides); each child imports torch and transformers (assumed 10 to 30 s) and loads the
  student from the local cache (about 3 s, RC-M1), then scores 17 probes and generates 16
  tokens (0.18 s per probe at 448) or makes two LoRA optimizer steps (the first took 6.68 s at
  micro-batch 1 in run 1). That is about 0.5 to 1 minute per child, 10 to 20 minutes for the
  student. The teacher is downloaded once (about 1 min 16 s) and measured in 3 children, each
  loading it in about 41 s (RC-M1), about 3 to 6 minutes. Total about **20 to 40 minutes**,
  under one hour. Larger sides and gradient checkpointing take longer per step; the run
  measures by how much. A child that runs longer than 30 minutes is stopped and its step
  recorded as failed (`--child-timeout`).

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip show torchao 2>/dev/null | grep Version  # Kaggle ships 0.10.0; peft 0.21.2 needs 0.16.0 or newer
pip uninstall -y torchao                     # only if the version above is below 0.16.0 (RC-M1)
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
python -m pytest -q tests/test_image_sides.py tests/test_results.py   # must pass first

# (1) revisions (in the configs and on the hub now) and what the published teacher adapter
#     and the Qwen processors state about image size; print only, paste all of it in section 6
python scripts/measure_image_sides.py settings 2>&1 | tee rc_s0b_settings.txt

# (2) the student, ONE T4: 7 steps per side, each in its own process: score (scoring, visual
#     tokens, peak memory, generation) and two LoRA optimizer steps at micro-batch 1, 2 and 4,
#     without (lora_mb<b>) and with gradient checkpointing (lora_gc_mb<b>); a failure, out of
#     memory included, is written as ..._ok = 0 and the run goes on
CUDA_VISIBLE_DEVICES=0 python scripts/measure_image_sides.py student \
    --config configs/models/qwen2_5_vl_3b_student.yaml --allow-candidate 2>&1 | tee rc_s0b_student.txt

# (3) the teacher (7B + the Kvasir LoRA), both T4s: one scoring process per side
python scripts/measure_image_sides.py teacher \
    --config configs/models/qwen2_5_vl_7b_teacher.yaml 2>&1 | tee rc_s0b_teacher.txt

python -m pytest -q tests/test_results.py                  # must pass before you commit
pip freeze | grep -E "torch|transformers|peft|accelerate|optimum"
```

Each (side, step) is written to the CSV as soon as its process ends, so a session that ends
early keeps what it finished. To redo only some of it, name them, for example `--sides 672
--steps lora_mb4 lora_gc_mb4`; a step run again replaces its earlier rows for that side. The
last lines of each command list the steps that failed. For a failure that is not an
out-of-memory error, keep the last 30 lines of that step's log for section 6 and do not change
any setting to get past it.

## 4. What to commit back

- `results/smoke/RC-S0b/student.csv` and `results/smoke/RC-S0b/teacher.csv` (rows
  `RC-S0b:side<N>`, one per metric and side; they never touch the RC-M1, RC-S0 or R0 files)
- This card with section 6 filled, including the whole output of step (1). Its revision lines
  name, for the teacher, the student and the fallback Qwen3-VL-2B, the revision in the config
  and the one the hub serves now. The fallback's revision stays TODO (not pinned) until the
  team uses it; this line records which revision the hub serves.

Check before committing: `git status` shows no images, models or caches.

## 5. How the result is read

Per side the CSVs hold one `..._ok` row per step (1 if the step worked, 0 if it failed).
**Read a value only where its step's `_ok` row is 1**; with 0 the only other row is the peak
memory reached before the failure, when it could be read.

- `score` (student and teacher, prefix `student_` or `teacher_`): `_scoring_ok`,
  `_visual_tokens`, `_scoring_seconds_per_probe`, `_peak_gpu_bytes_scoring`,
  `_generate_tokens_per_second` (for the teacher it sets the cost of the teacher cache),
  `_peak_gpu_bytes_generate`, `_allocated_gpu_bytes_after_load`.
- `lora_mb<b>` (student): `student_lora_step_ok_mb<b>`, `student_lora_step_seconds_mb<b>` (the
  second, warm optimizer step alone, measured like RC-S0's 0.74 s), `..._first_seconds_mb<b>`
  (the whole first call, adapter attach and warm-up included) and
  `student_peak_gpu_bytes_lora_step_mb<b>` (over both steps).
- `lora_gc_mb<b>` (student): the same with gradient checkpointing, named
  `student_lora_gc_step_...`. Measurement only: D06 does not use it.

Checks: at side 448 the student shows 194 visual tokens, as in RC-S0 (same 448 x 336 image);
its micro-batch-1 step should be near the 9,377,473,536 bytes of run 1 (RC-S0:
9,140,717,056). Whole 28-pixel patches would give 302 tokens at 560 and 434 at 672; the
measurement, not this arithmetic, is what counts.

**Decided 2026-10-17 (Soheil): side 672**, with micro-batch 1 x 16 (D05, D06).

**Decision rule (written 2026-10-16, before the run; D05).** For each side the agent computes
the projected GPU hours of the whole study from these numbers and the RC-D1 counts: the teacher
cache, Gate 1, E0 evaluation, E1/E2 training at 10,000 examples, the scoring of D07, the
recoveries, every evaluation, and the training-length check (pre-registration section 6). The
team states the GPU hours it has (`TODO: team GPU hours available`) and picks the **largest
side whose projected total fits**. If the D06 micro-batch of 4 does not fit one T4 at that side,
the agent proposes micro-batch 2 (or 1) with gradient accumulation 8 (or 16), which keeps 16
examples per update: the same recipe up to floating-point rounding (the model has no batch
normalisation). The alternative is gradient checkpointing at micro-batch 4; its time cost is
read from `student_lora_gc_step_seconds_mb<b>` against `student_lora_step_seconds_mb<b>`, and
the projection uses the time of the option proposed. That is a D06 amendment **proposal** (D06
memo); D06 stays as it is until the team decides.

## 6. Run report (filled by the person who ran it)

### Run 1: stopped by a script bug (2026-10-16, Soheil; recorded by the agent from his report)

- Commit used: 5613b7f. Kaggle, GPU T4 x2. Nothing was committed from this run.
- Tests: 11 passed, 1 failed: `test_an_out_of_memory_step_is_a_result_not_a_crash` assumed no
  GPU; with CUDA the code also writes the peak-memory row, as designed. The test now passes
  with and without CUDA.
- Step (1), valid facts: the adapter's `args.json` has `"max_pixels": null`, bf16 training,
  `gradient_checkpointing=True`, `per_device_train_batch_size=32`. The teacher and student
  processors are `Qwen2VLImageProcessor` with `patch_size` 14, `merge_size` 2,
  `longest_edge=12845056`, `shortest_edge=3136`; `min_pixels` and `max_pixels` are not set (so,
  as Soheil notes, the processor does not shrink Kvasir images by default). Then the step
  crashed on the 3B student: its config had `revision: TODO` and the step asked the hub for the
  revision "TODO" (404). Fixed: the student is pinned (below) and a model whose revision is
  still TODO is printed as "revision not pinned" and skipped.
- Student revision downloaded on Kaggle (as main, with `--allow-candidate`):
  `66285546d2b821cf421d4f5eb2576359d3770cd3`, the `sha_seen` of 2026-10-12; now pinned in
  `configs/models/qwen2_5_vl_3b_student.yaml`. RC-M1 and RC-S0 did not record theirs
  (`load_model` asks for `sha_seen` when no revision is set, which is this hash).
- Step (2), side 448 (17 rows): scoring peak 7,843,901,952 bytes, 0.177 s per probe, 194 visual
  tokens; LoRA step at micro-batch 1 ok, 6.68 s (the whole call, adapter attach included, so
  not comparable with RC-S0's 0.74 s), peak 9,377,473,536 bytes; micro-batch 2 out of memory at
  a peak of 15,445,145,088 bytes; micro-batch 4 out of memory at 15,439,432,192 bytes. Then the
  scoring at side 560 raised an out-of-memory error that was not caught (14.55 GiB already in
  use) and the script stopped. Step (3) was not run.
- Diagnosis: every step loaded a fresh model in the same process, and a second 7.5 GB copy
  stayed on the GPU; 7.5 + 7.5 GB matches every failure at about 15.4 GB. What the code shows:
  after each step the script deleted its own reference to the model and called
  `gc.collect()` and `torch.cuda.empty_cache()`, after a success and after a failure (the
  exception variable is cleared at the end of the `except` block). So no variable of the
  script held the copy; something below it kept a reference, and it happened after a
  successful step too (micro-batch 1 passed, then micro-batch 2 failed at 15.4 GB). Which
  object held it cannot be found without a GPU; a child process per step makes it irrelevant,
  because the GPU memory of a process is freed when it ends.
- Valid from run 1: the step (1) facts, the side-448 scoring rows and the micro-batch-1 peak
  (9,377,473,536 bytes: only one model was on the GPU then). **Not valid:** the micro-batch 2
  and 4 out-of-memory results and the side-560 crash. The run is repeated in full.


### Run 2 (2026-10-16, Kaggle T4 x2, same session as run 1)
- Commit: e044d749e45a1847a96270e96bcfef6ae9e3398d (repo reset to it; run-1 CSVs deleted first)
- Versions: torch 2.11.0+cu128, transformers 5.19.0, peft 0.21.2, accelerate 1.14.0, optimum 2.3.0,
  torchao 0.18.0, gptqmodel 7.6.0, bitsandbytes 0.50.2. Tests: test_image_sides + test_results 16 passed.
- Settings step (completed): adapter SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft @ 79e77755...: args.json
  "max_pixels": null; training_args: bf16=True, gradient_checkpointing=True,
  per_device_train_batch_size=32, learning_rate=2e-05, num_train_epochs=3.0, cosine schedule;
  adapter_config.json, additional_config.json, trainer_state.json: not stated.
  Teacher Qwen2.5-VL-7B-Instruct: config revision cc594898...; hub main now cc594898....
  Student Qwen2.5-VL-3B-Instruct: config revision 66285546...; sha_seen 66285546...; hub main now 66285546....
  2B Qwen3-VL-2B-Instruct: revision not pinned; sha_seen 89644892e4d85e24eaac8bacfd4f463576704203;
  hub main now 89644892e4d85e24eaac8bacfd4f463576704203.
  Teacher and student processors: Qwen2VLImageProcessor, patch_size 14, merge_size 2,
  longest_edge=12845056, shortest_edge=3136, min_pixels/max_pixels not set. This repository
  shrinks images only to inputs.image_longest_side; no config sets max_pixels.
- Student (1 T4): 14:20:14 to 14:29:31 UTC (9 min 17 s). Failed (recorded as _ok = 0):
  side 560 lora_mb4 and side 672 lora_mb4, both CUDA out of memory (14.5 GiB in use).
  All other 19 steps ok, including all gradient-checkpointing steps.
- Teacher (2 T4): 14:31:25 to 14:36:00 UTC (4 min 35 s, incl. download). All three sides ok.
- Results test: 6 passed.
- Anything unusual: none in run 2. Run 1's mb2/mb4 failures were caused by the script
  (now fixed: one process per step); at side 448 mb2 and mb4 fit in run 2.
