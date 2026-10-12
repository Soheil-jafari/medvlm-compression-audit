# Run card RC-Q0: quantization toolchain smoke test

## 1. What and why

- Stage: S08
- Experiments: none (feasibility only)
- Purpose: before the 13 quantization runs (E3, E8, E9), find out on a T4 which GPTQModel
  kernel class runs our 4- and 8-bit weights correctly and fastest, whether the student can be
  quantized end to end (GPTQ `w4` and `w8`, LLM.int8 `w8a8`), whether it passes the
  low-precision kernel check and still answers the probes, how much memory it saves, and which
  library versions work together. No dataset is used: calibration and probes run on synthetic
  images.
- Status: `ready` (can run right after RC-M1, in the same kind of session)

## 2. Requirements

- Platform: Kaggle notebook with **one T4**, Internet **on**.
- Disk: the 3B student (about 7.5 GB). `export HF_HOME=/kaggle/temp/hf`.
- Repository commit to run from: latest `research-loop` (write it in section 6).
- Dataset access: none.
- Expected runtime: **estimate** 30 to 60 minutes (model download, three quantizations of the
  student on 16 synthetic images, a few probe passes). The card reports the real number.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
pip install -q gptqmodel bitsandbytes          # versions are printed below and pinned after this card
# restart the kernel here if pip replaced torch, then continue:

python -m pytest -q tests/test_quantization.py tests/test_ptq_torch.py   # includes the GPU kernel tests
python scripts/smoke_quantization.py --allow-candidate --out results/smoke/RC-Q0.csv 2>&1 | tee rc_q0_log.txt
python -m pytest -q tests/test_results.py                               # must pass before you commit
pip freeze | grep -E "torch|transformers|peft|accelerate|gptqmodel|bitsandbytes"
```

If `pip install gptqmodel` tries to replace torch or transformers, stop and write what it asked
for in section 6: the pinned versions must not change silently. If the script fails, do not
change settings to get past it: copy the last 30 lines into section 6 and set the status to
`failed`.

## 4. What to commit back

- `results/smoke/RC-Q0.csv`
- This card with section 6 filled: the printed `versions` block, the "chosen kernel class"
  line, the lines that say which kernel classes failed, the three "kernel check" lines and the
  `pip freeze` lines (they contain no data).

## 5. Pass criteria (decided before the run)

Read from `results/smoke/RC-Q0.csv`. These are checks, not targets to tune towards.

| Check | Metric | Pass if |
|---|---|---|
| A kernel class reproduces our 4-bit and 8-bit codes | `kernel_ok:<class>:w4`, `kernel_ok:<class>:w8` | at least one class with both equal to 1 |
| GPTQ models use the kernels | `w4_kernels_ok`, `w8_kernels_ok` | equal 1 |
| LLM.int8 uses its kernel | `w8a8_kernels_ok` | equals 1 |
| Quantized weights are smaller | `w4_state_bytes`, `w8_state_bytes`, `w8a8_state_bytes` vs `fp16_state_bytes` | each below the fp16 value |
| Answers survive | `w4_agreement_with_fp16`, `w8_agreement_with_fp16`, `w8a8_agreement_with_fp16` | reported (synthetic images: no target value; a value near 0.5 or below on `w8` would point to a broken path) |

After the run the agent sets `quantization.kernel_class` (the chosen class) and
`quantization.library_versions` in `configs/quantization.yaml`, pins the versions in
`requirements/train.txt`, and sets RC-E3 and RC-E9 to `ready` when their other inputs exist.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and the printed `versions` block:
- Chosen kernel class (printed) and the classes that failed (with their error lines):
- Wall time and peak GPU memory (`fp16_peak_bytes`, `w4_peak_bytes`, `w8_peak_bytes`, `w8a8_peak_bytes`):
- Pass criteria: all met / not met (list):
- Anything unusual (errors, retries, changed settings, what pip tried to change):
