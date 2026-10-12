# Run card RC-M1: model smoke test (teacher and one student)

## 1. What and why

- Stage: S04
- Experiments: none (feasibility only)
- Purpose: find out on a real GPU whether the teacher and a student load in fp16,
  share a tokenizer, answer yes/no probes in one forward pass with finite scores,
  and get smaller in **measured** memory when one decoder layer is removed. No
  dataset is used: the probes run on synthetic images.
- Status: `ready`

## 2. Requirements

- Platform: Kaggle notebook with **2 x T4** (GPU T4 x2), Internet **on**.
  The teacher needs about 16.6 GB in fp16 (8,292,166,656 stored parameters x 2
  bytes), more than one T4 holds, so it is split over both GPUs. With a single T4
  add `--teacher-bits 8` (needs `pip install bitsandbytes`), which needs roughly half of that.
- Disk for downloads: about 16.6 GB (teacher) + 7.5 GB (3B student) + the small
  adapter. Point the cache at a volume with room, for example
  `export HF_HOME=/kaggle/temp/hf`.
- Repository commit to run from: latest `research-loop` (write the commit in section 5).
- Dataset access: none. The models are downloaded from the Hugging Face hub (licences
  as recorded in the model configs: teacher Apache-2.0, Kvasir adapter MIT, 3B student
  `qwen-research`, a research-only flag already noted in `D05`).
- Expected runtime: model downloads dominate (tens of minutes); the measurements take
  a few minutes.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 5
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
# restart the kernel here if pip replaced torch, then continue:

python -m pytest -q tests/test_models.py tests/test_surgery.py    # CPU tests; must pass first
python scripts/smoke_models.py \
    --teacher-config configs/models/qwen2_5_vl_7b_teacher.yaml \
    --student-config configs/models/qwen2_5_vl_3b_student.yaml \
    --allow-candidate --out results/smoke/RC-M1.csv 2>&1 | tee rc_m1_log.txt
python -m pytest -q tests/test_results.py                           # must pass before you commit
pip freeze | grep -E "torch|transformers|peft|accelerate|bitsandbytes"
```

If the script fails, do not change settings to get past it: copy the last 30 lines of
the error into section 5 and set the status to `failed`.

## 4. What to commit back

- `results/smoke/RC-M1.csv`
- This card with section 5 filled (paste the printed hardware block, the two
  "generation" lines and the `pip freeze` lines; they contain no data).

## 5. Pass criteria (decided before the run)

Read from `results/smoke/RC-M1.csv`. These are checks, not targets to tune towards.

| Check | Metric | Pass if |
|---|---|---|
| Same tokenizer | `tokenizers_identical` | equals 1 |
| Teacher is stable in fp16 | `teacher_scores_finite` | equals 1 |
| Student is stable in fp16 | `student_scores_finite`, `student_scores_finite_after_removal` | equal 1 |
| Teacher follows the yes/no format | `teacher_yes_no_probability_mass` | close to 1 (report the value; a value far below 1 means the answer position or prompt format is wrong) |
| Removal is physical | `student_parameter_drop_matches` | equals 1 |
| Memory really drops | `student_removal_measured_drop_bytes` vs `student_removal_expected_drop_bytes` | measured within 10 % of expected |
| Removal changes the model | `student_scores_change_after_removal` | equals 1 |
| Teacher fits the plan | `teacher_peak_gpu_bytes_scoring` | below the total memory of the GPUs used |

Record in section 6 whether the teacher's published adapter loaded without error (the
loader fails by itself if any language-model adapter module stays at zero).

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware (GPU, driver, library versions from `pip freeze`):
- Wall time and peak GPU memory (`teacher_peak_gpu_bytes_scoring`, `student_peak_gpu_bytes_scoring`):
- Pass criteria: all met / not met (list):
- Adapter loaded without error: yes / no
- Anything unusual (errors, retries, changed settings):
