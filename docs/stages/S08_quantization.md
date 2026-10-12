# S08: Quantization arms E3, E4, E9

**Goal.** Post-training quantization of E2 (E3), quantization-aware distillation
(E4) and the 4-bit teacher (E9), evaluated and measured.

**Depends on.** S07 (frozen), S06.

## Tasks

- [ ] [agent] `compression/quantization/ptq.py` with the backend chosen in S01
  (W8, W4 via GPTQ/AWQ; W8A8 where supported) and a check that low-precision
  kernels actually run; `compression/quantization/qakd.py` (GRACE-style).
- [ ] [agent] Run cards **RC-E3** (3 variants x 3 seeds), **RC-E4** (3 seeds),
  **RC-E9** (1 run). Each run evaluates on validation and test and writes
  `metrics.csv` and `efficiency.csv` in the `results/` layout.
- [ ] [team:Soheil] Execute the run cards.
- [ ] [agent] `docs/results/E3.md`, `E4.md`, `E9.md` from the committed CSVs.

## Outputs

13 runs with metrics and footprints: E3 x 9, E4 x 3, E9 x 1 (E8 is in S09).

## Exit gate

All runs committed with metrics and footprints; kernel check recorded for each
quantized arm.
