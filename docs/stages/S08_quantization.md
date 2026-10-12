# S08: Quantization arms E3, E4, E9

**Goal.** Post-training quantization of E2 (E3), quantization-aware distillation
(E4) and the 4-bit teacher (E9), evaluated and measured.

**Depends on.** S07 (frozen), S06.

## Tasks

- [x] [agent] D08 memo (tools, settings, calibration, scope of E4); answered 2026-10-12.
- [x] [agent] `compression/quantization/{grid,gptq,kernels,ptq,smoke}.py`: the symmetric
  group-wise grid and its packed storage, GPTQ (NumPy reference and torch), sequential
  layer-by-layer GPTQ of the decoder in memory (so E8 can quantize pruned models), LLM.int8,
  GPTQModel kernel modules built from our codes and checked numerically, the low-precision
  kernel check (`efficiency.check_low_precision_kernels`, `low_precision_kernels_ok` row), the
  E3/E8/E9 stage (`scripts/03_quantize.py`) and model loading in the runner.
- [x] [agent] Run cards **RC-Q0** (toolchain smoke test, ready), **RC-E3** (3 variants x 3
  seeds) and **RC-E9** (1 run), both blocked on RC-Q0, the data and the freeze.
- [x] [agent] `compression/quantization/qakd.py` (D08 option B: learned step-size 4-bit
  quantization with LoRA trained through the rounding, GRACE's confidence-gated decoupled
  distillation on the cached top-20, exact export in the E3 `w4` format) and run card **RC-E4**.
- [ ] [agent] After RC-Q0: set `quantization.kernel_class` and the library versions, pin them in
  `requirements/train.txt`.
- [ ] [team:Soheil] Execute the run cards.
- [ ] [agent] `docs/results/E3.md`, `E4.md`, `E9.md` from the committed CSVs.

## Outputs

13 runs with metrics and footprints: E3 x 9, E4 x 3, E9 x 1 (E8 is in S09).

## Exit gate

All runs committed with metrics and footprints; kernel check recorded for each
quantized arm.
