# Run card RC-M1: model smoke test (teacher and one student)

## 1. What and why

- Stage: S04
- Experiments: none (feasibility only)
- Purpose: find out on a real GPU whether the teacher and a student load in fp16,
  share a tokenizer, answer yes/no probes in one forward pass with finite scores,
  and get smaller in **measured** memory when one decoder layer is removed. No
  dataset is used: the probes run on synthetic images.
- Status: `done`

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
pip show torchao 2>/dev/null | grep Version  # Kaggle ships 0.10.0; peft 0.21.2 needs 0.16.0 or newer
pip uninstall -y torchao                     # only if the version above is below 0.16.0 (RC-M1)
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
# restart the kernel here if pip replaced torch, then continue:

python -m pytest -q tests/test_models.py tests/test_surgery.py    # CPU tests; must pass first
python -m pytest -q 2>&1 | tail -3   # the whole suite with torch installed: write the last line
                                     # in section 5 (the agent session has no torch, so the
                                     # torch tests only run here)
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

- Run by / date: Soheil, 2026-10-15
- Commit actually used: 309c41924c7d23bbe560fc2e63ced157890ee2ed
- Hardware (GPU, driver, library versions from `pip freeze`): Kaggle, 2 x Tesla T4 (15,636,037,632 bytes each, compute capability 7.5), CUDA 12.8, driver version not printed; Python 3.13.15, Linux 6.18.48. torch 2.11.0+cu128, transformers 5.19.0, peft 0.21.2, accelerate 1.14.0; bitsandbytes not installed.
- Wall time and peak GPU memory: wall time <WRITE THE TIME FROM THE KAGGLE TIMER HERE>; teacher download 1 min 16 s + load 41 s, student download 40 s + load 3 s. teacher_peak_gpu_bytes_scoring = 17,181,603,328 (teacher split over both GPUs); student_peak_gpu_bytes_scoring = 7,851,811,328.
- Pass criteria: all met (tokenizers_identical 1; teacher and student scores finite 1; teacher_yes_no_probability_mass 0.99998; student_parameter_drop_matches 1; measured drop 154,153,984 bytes = expected 154,153,984; scores change after removal 1; teacher peak 17.18 GB is below the 31.3 GB of the two GPUs).
- Adapter loaded without error: yes. peft printed one long "missing adapter keys" warning on the first attempt (the published adapter uses older module names); the loader remapped the keys and its check that no language-model adapter module stays at zero passed.
- Anything unusual: (1) Kaggle's preinstalled torchao 0.10.0 made peft 0.21.2 raise ImportError (it needs 0.16.0 or newer) in every test that builds a LoRA model; fixed with `pip uninstall -y torchao` and a re-run. (2) pip kept Kaggle's torch 2.11.0; the code was tested with torch 2.8. (3) Full test suite on this GPU session after the fix: 880 passed, 6 failed, 3 skipped. tests/test_efficiency.py (2 tests) and tests/test_models.py::test_measure_model_and_layer_removal_on_cpu assert that no GPU numbers exist (written for a CPU-only session). tests/test_selection_rounds.py::test_round_0_records_numbers_and_cannot_pass_the_gpu_gate_on_a_cpu has the same CPU assumption (from the test code; its output was not read). tests/test_ptq_torch.py::test_the_decoder_is_quantized_saved_and_reloaded and tests/test_qakd_torch.py::test_the_e4_stage_trains_exports_and_reloads_the_same_model differ by 1e-4 to 4e-4 in a log-probability after save and reload, against a tolerance of 1e-4 (cause not confirmed, probably GPU arithmetic). (4) Student scoring time was 0.199 s per probe on synthetic images, above the 0.12 s assumed in D07 section 1.
