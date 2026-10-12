# S06: Students E1, E2 and the budget reference

**Goal.** Train the student baselines (E1 supervised, E2 distilled) with 3 seeds and
measure E2's footprint, which defines the absolute budgets B1-B3.

**Depends on.** S05 and S05b (the student chosen by the selection).

## Tasks

- [ ] [agent] `compression/distillation/losses.py` and `offline_kd.py` (KL on the
  cached top-k teacher probabilities), the `sft` and `offline_kd` paths of
  `scripts/02_train_student.py` with LoRA. Pure loss tests plus heavy tiny-model
  tests.
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
