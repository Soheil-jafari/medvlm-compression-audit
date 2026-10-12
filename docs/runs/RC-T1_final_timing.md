# Run card RC-T1: final timing of every arm on the rented GPU

## 1. What and why

- Stage: S11
- Experiments: E0, E1, E2, E3 (`w8`, `w4`, `w8a8`), E4, E5 (3 criteria), E6, E7 and E8 at B2,
  and E9; seed 0 (seeds 1 and 2 only if the team decides the credits allow it).
- Runs: 14 benchmark runs (42 with all seeds). Nothing is trained or evaluated.
- Purpose: table T5 reports parameters, file size, peak memory, cold and warm latency,
  throughput and the low-precision kernel check, all measured on one rented GPU with the
  pre-registered protocol. The Kaggle T4 numbers of the other cards are development numbers;
  they matched the budgets but never enter T5.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- The team chose the GPU and the precision: `hardware.final_timing` and `hardware.precision`
  in `configs/base.yaml` (pre-registration section 4).
- Every arm above is done and its checkpoints are kept: the E1 and E2 adapters, the quantized
  codes of E3, E4, E8 and E9, and the pruned models of E5 to E8 at B2.
- `budgets.measurement` is complete (it is the same protocol as in development).

Platform: one rented GPU of the chosen type, the **same machine for every run**, nothing else
running on it. Dataset access: your own copy of Kvasir-VQA-x1 and the private RC-D1 outputs,
downloaded there with your own access (the 32 timing questions are validation images). The
checkpoints are copied there from the private datasets. Nothing from those goes into the
repository, and the machine is wiped when the runs are done.

Runtime: each run loads one model and answers the 32 timing questions 3 times to warm up and
5 times timed. The agent fills in the estimate from the RC-B0 wall time when it sets the card
to `ready`; until then any number is a guess. Rent the GPU only for the measured time.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
nvidia-smi --query-gpu=name,driver_version --format=csv       # write this in section 6
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=$HOME/data                      # your own copy
export MEDCOMPRESS_PRIVATE_ROOT=$HOME/private                # your RC-D1 outputs
export MEDCOMPRESS_CHECKPOINT_ROOT=$HOME/checkpoints         # every arm's checkpoints
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results

SEED=0
T="-o hardware.timing_role=final"
B="scripts/07_benchmark_efficiency.py"
python $B --exp E0 --seed $SEED $T
python $B --exp E1 --seed $SEED $T
python $B --exp E2 --seed $SEED $T
for v in w8 w4 w8a8; do python $B --exp E3 --variant $v --seed $SEED $T; done
python $B --exp E4 --seed $SEED $T
for v in magnitude activation taylor; do python $B --exp E5 --variant $v --budget B2 --seed $SEED $T; done
for e in E6 E7 E8; do python $B --exp $e --budget B2 --seed $SEED $T; done
python $B --exp E9 --seed 0 $T
python -m pytest -q tests/test_results.py                  # must pass before you commit
```

With `hardware.timing_role=final` each run writes `efficiency_final.csv` and
`efficiency_final_setup.json` next to the development files and never touches them. If a run
stops with "configuration incomplete", tell the agent through the inbox instead of filling a
setting in.

## 4. What to commit back

For every run: `results/<run dir>/efficiency_final.csv` and `efficiency_final_setup.json`, and
this card with section 6 filled.

Check before committing: `git status` shows no images, checkpoints or data.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| The rented GPU was used | `hardware` in every `efficiency_final_setup.json` | the GPU named in `hardware.final_timing`, the same for every run |
| Same protocol everywhere | `protocol` in every `efficiency_final_setup.json` | identical in every file |
| Quantized arms really run low-precision code | `final:low_precision_kernels_ok` for E3, E4, E8, E9 | 1 (a 0 is reported and the arm is not called quantized, pre-registration section 6) |
| Peak memory measured | `final:peak_memory_bytes` | present in every file |

## 6. Run report (filled by the person who ran it)

- Run by / date / GPU provider:
- Commit actually used:
- GPU name and driver (`nvidia-smi`), library versions (`pip freeze | grep -E "torch|transformers|peft|gptqmodel|bitsandbytes"`):
- Wall time per run and total GPU hours (the real cost):
- Anything unusual (errors, retries, other load on the machine):
