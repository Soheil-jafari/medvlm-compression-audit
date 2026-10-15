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
- Status: `done` (can run right after RC-M1, in the same kind of session)

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
pip show torchao 2>/dev/null | grep Version  # Kaggle ships 0.10.0; peft 0.21.2 needs 0.16.0 or newer
pip uninstall -y torchao                     # only if the version above is below 0.16.0 (RC-M1)
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
pip install -q gptqmodel bitsandbytes          # as run on 2026-10-15; train.txt now pins 7.6.0 and 0.50.2
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

- Run by / date: Soheil, 2026-10-15, Kaggle
- Commit actually used: 9cafa2bcdb3c4a634a83b528ee5c4fb550a6739a
- Hardware and the printed `versions` block: 1 x Tesla T4 (15,636,037,632 bytes, capability 7.5),
  4 CPUs, Python 3.13.15, CUDA 12.8. Versions: torch 2.11.0+cu128, transformers 5.19.0,
  peft 0.21.2, gptqmodel 7.6.0, bitsandbytes 0.50.2 (also accelerate 1.14.0, torchao 0.18.0,
  numpy 2.2.6, protobuf 7.36.2, Triton 3.6.0). 55 GPTQModel kernel classes found.
- Chosen kernel class (printed) and the classes that failed (with their error lines):
  chosen `gptqmodel.nn_modules.qlinear.torch.TorchLinear` (fastest at w4: 0.238 ms; w8 0.201 ms).
  Also passed both: TritonV2Linear (w4 0.240 ms, w8 0.244 ms). TorchAtenLinear passed w4 only
  (0.672 ms). All other classes failed, for example: BitBLAS classes "bitblas is not installed";
  GPTQModel BitsAndBytes classes "has no buffer 'qweight' to load"; ExllamaV2Linear w4
  "post_init() missing 1 required positional argument: 'scratch_space'", w8 "only supports [4]
  bits"; Marlin/Machete need a newer GPU than T4; Mlx classes are for Apple hardware.
  Kernel checks: "w4: kernel check True, classes {TorchLinear: 252}", "w8: kernel check True,
  classes {TorchLinear: 252}", "w8a8: kernel check True, classes {bitsandbytes Linear8bitLt: 252}".
- Wall time and peak GPU memory: wall time about __ minutes (GPTQ w4 508 s, w8 504 s).
  Peak: fp16 7,843,649,024; w4 3,797,138,944; w8 5,197,114,880; w8a8 5,073,484,288 bytes.
  State size: fp16 7,509,246,624; w4 3,405,009,312; w8 4,803,105,312; w8a8 4,734,714,528 bytes.
- Pass criteria: all met. A class with w4 and w8 both 1 (TorchLinear, TritonV2Linear);
  w4/w8/w8a8 kernels_ok = 1; all three state sizes below fp16; agreement with fp16 = 1.0
  for w4, w8 and w8a8 (synthetic images only).
- Anything unusual: a fresh Kaggle session had transformers 5.16.1 and peft 0.20.0
  preinstalled; installing requirements/train.txt restored the pinned 5.19.0 and 0.21.2,
  and GPTQModel 7.6.0 accepted them (no conflict reported). torch was never replaced.
  Tests: 22 passed, 1 failed (test_ptq_torch.py::test_the_decoder_is_quantized_saved_and_reloaded:
  reload differs by up to 2.5e-4, tolerance 1e-4; same kind as the RC-M1 reload failures).
  torchao printed that its Hopper-only kernels (_C_cutlass_90a, _C_mxfp8) cannot load on T4.
  The kernel choice rule only compares w4 speed; here the chosen class also passed w8.

## 7. Agent notes after the run (2026-10-15)

- Kernel choice rule fixed: `scripts/smoke_quantization.py` compared only the w4 speed, so it
  could have picked a class that fails w8 (TorchAtenLinear passed w4 only). It now chooses the
  fastest class at w4 **among the classes that passed both w4 and w8**
  (`quantization/smoke.py`, `choose_kernel_class`). This does not change this run's result:
  TorchLinear passed both widths and was the fastest of the two that did (TritonV2Linear
  0.240 ms at w4); a test applies the new rule to the committed `RC-Q0.csv` and gets
  TorchLinear.
- The first line of the committed `RC-Q0.csv` was a pasted `==== RC-Q0.csv` label, which the
  results check rejects; the agent removed that line (no number changed).
- `quantization.kernel_class` and `quantization.library_versions` are set from this run, and
  gptqmodel 7.6.0 and bitsandbytes 0.50.2 are pinned in `requirements/train.txt`.
