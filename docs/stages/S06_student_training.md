# S06: Students E1, E2 and the budget reference

**Goal.** Train the student baselines (E1 supervised, E2 distilled) with 3 seeds and
measure E2's footprint, which defines the absolute budgets B1-B3.

**Depends on.** S05 and S05b (the student chosen by the selection).

## Tasks

- [x] [agent] `compression/lora.py`, `compression/distillation/{losses,batching,offline_kd}.py`
  (cross-entropy on the answer plus KL on the cached top-k teacher probabilities,
  one shared trainer for E1, E2 and recovery) and the `sft` and `offline_kd` paths of
  `scripts/02_train_student.py`. Loss tests and tiny-model tests (done in iteration
  26); the hyper-parameters in `configs/training.yaml` are still `TODO` (see below).
- [ ] [agent] Training recipe memo (rank, alpha, learning rate, steps, batch size,
  KD weight and temperature): proposed from RC-S0/RC-S1 results, not tuned on the
  result of any compression arm; team approves before RC-E1/RC-E2.
- [ ] [agent] `evaluation/efficiency.py` and `scripts/07_benchmark_efficiency.py`:
  the measurement protocol from `configs/budgets.yaml` (peak memory, cold and warm
  latency, throughput, file size, parameter count).
- [ ] [agent] Run cards **RC-E1** (3 seeds), **RC-E2** (3 seeds) and **RC-B0** (E2
  reference footprint on the development GPU).
- [ ] [team:Soheil] Execute RC-E1 and RC-E2. [team:Amir] Execute RC-B0.
- [ ] [agent] `docs/results/E1.md` and `E2.md` (validation metrics, mean and SD over
  seeds); absolute B1-B3 targets proposed from RC-B0 for S07.

## Outputs

E1 and E2 adapters on team storage, validation metrics, E2 reference footprint.

## Exit gate

3 seeds each of E1 and E2 with committed metrics; E2 footprint measured with the
budget protocol.
