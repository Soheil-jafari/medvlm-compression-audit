# Run card RC-E4: quantization-aware distillation (E4), seed 0

## 1. What and why

- **Seeds (seed plan, pre-registration section 11 item 12, Soheil and Amir, 2026-11-08):** seed 0 only, reported as exploratory / descriptive (no confirmatory test); the plan (`scripts/run_card.py`) refuses seeds 1 and 2. This replaces the seed counts written below.
- Stage: S08
- Experiments: E4, seeds 0, 1, 2 (3 runs). The student is trained from its pretrained weights
  with 4-bit weights already in the loop: learned step sizes per group of 128 weights, LoRA
  trained through the rounding, and GRACE's confidence-gated decoupled distillation on the
  cached top-20 teacher tokens (D08 option B). Same training pool, steps and optimiser as E2.
  The result is exported in the same 4-bit format as E3 `w4` and evaluated and timed the same
  way.
- Purpose: the "quantize during training" family of the audit.
- Status: `ready` for seed 0 (2026-11-11): every requirement of section 2 is met (frozen,
  RC-E0b done, the E2 training values set); its peak at 1 x 16 is not measured, so the
  out-of-memory rule of section 2 applies

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- **Memory (2026-10-17):** E4 now trains at micro-batch 1 x 16 **without** gradient
  checkpointing, like every training run (D06 amendment). It had checkpointing on for memory
  only, and its peak at 1 x 16 is not measured. Run seed 0 first. If it stops with "CUDA out of
  memory", stop there, keep the last 30 lines of the log and tell the agent: do not turn
  checkpointing back on yourself (the team decides).
- RC-Q0 passed; `quantization.kernel_class` and the library versions are set and pinned. **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
- RC-E0b done (the teacher cache of the training pool on team storage; the manifest hash matches
  the logged one).
- `training.steps`, `training.data_subset` and `teacher_cache.pool_size` are set (the E2 values;
  E4 trains on exactly the E2 pool).
- The pre-registration is frozen before the first E4 run; `inputs.image_longest_side`,
  `evaluation.max_new_tokens` and `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, one T4, Internet on. Dataset access: your own copy of Kvasir-VQA-x1
and the private RC-D1 outputs; the teacher cache attached as a private dataset.

Runtime per seed: **estimate** 1.5 to 3 times an E2 run of the same length (every step rounds
the weights, and the layers are recomputed in the backward pass to save memory). E2 seed 0's
training took 4.72 h (RC-E2 section 6, 27.2 s per update), so **about 7.1 to 14.1 h** of
training per seed (an estimate), plus the validation pass (about 0.5 h).

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** account 3 (lent by Farzan; Soheil starts it), seed 0, after RC-E9 (reassigned 2026-11-11; seed plan of 2026-11-08: seeds 1 and 2 are not run).
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher cache.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E4"` and `SEED = <k>` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-E4 --seed <k>`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E4 seed<k>
  - test evaluation E4 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E4_seed<k>_quantized`, `E4_seed<k>_outcomes`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

One cell (Kaggle settings and inputs of section 2b; **Save Version -> Save & Run All**). It finds the shared data and the attached hand-offs by their manifests (`scripts/find_inputs.py`), stops with a clear message if the shared data are missing, and runs the steps of section 2b with `scripts/run_card.py`, which also checks that every hand-off the card reads is there. Then it writes the hand-offs under `/kaggle/working/handoff/` and the files to commit under `/kaggle/working/results_to_commit/`.

```bash
%%bash
SEED=0   # this session's seed: 0, 1 or 2 (section 2b says who runs which)
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
python - <<'PY' || pip uninstall -y -q torchao   # peft needs torchao 0.16.0 or newer, or none
import sys
from importlib import metadata
try:
    version = tuple(int(x) for x in metadata.version("torchao").split(".")[:2])
except metadata.PackageNotFoundError:
    sys.exit(0)
sys.exit(0 if version >= (0, 16) else 1)
PY
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
python scripts/find_inputs.py data > /dev/null || exit 1        # stops if the shared data are not attached
python scripts/find_inputs.py handoffs                          # the hand-offs found by their manifests
python scripts/run_card.py --card RC-E4 --seed $SEED 2>&1 | tee /kaggle/working/run_card_RC-E4_seed${SEED}.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

The runner runs the training (02), the evaluation (06) and the benchmark (07). The
codes go to `checkpoints/E4/nobudget/seed<k>/quantized/` (never committed); keep them as a
private dataset.

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

- `results/E4/nobudget/seed<k>/metrics.csv`, `metrics_test.csv`, `efficiency.csv`, `efficiency_setup.json`
- This card with section 6 filled.

Check before committing: `git status` shows no images, checkpoints, caches or `outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Trained as planned | `train_steps` in `metrics.csv` | equals `training.steps` |
| Every decoder matrix exported | `quantized_matrices` | 7 x the number of decoder layers |
| Real low-precision kernels | `low_precision_kernels_ok` in `efficiency.csv` | equals 1 |
| Training did not diverge | `train_loss_final` | finite (report the value) |

## 5b. RC-SMOKE out of memory (9113632) and the fix (2026-11-13)

The smoke's training stage ran out of memory (14.24 GiB on one T4) in `qakd.lsq_quantize`: for
every target layer autograd kept, until the backward pass, the float32 ratio `w / s` and the
16-bit quantized weight, about 6 bytes per decoder weight (2.78 billion of them, about 16.7 GB).
The settings stay (micro-batch 1 x 16, no gradient checkpointing). Fix in code only
(`qakd._qat_linear_function`): one autograd function per QAT layer that keeps the layer's input
(as the linear layer does anyway) and references to its LoRA matrices, step sizes and frozen
base; the quantized weight is built under `no_grad` in the forward and built again, with the
same operations, in the backward, layer by layer, and freed. Output and gradients are identical
bit for bit to the old code (test), and the graph keeps no weight-sized tensor per layer (test).
**Expected peak (estimate): about 13-14 GB of the T4's 15.6 GB** (E2's measured training peak
12.66 GB, plus about 0.35 GB of step sizes with their Adam state and gradients, plus at most
about 0.8 GB built and freed per layer for the largest MLP matrix). RC-E4 runs after RC-SMOKE
passed for it.

## 6. Run report (filled by the person who ran it)

- Run by / date / seeds:
- Commit actually used:
- Hardware and library versions:
- Wall time and `train_peak_gpu_bytes`:
- Teacher cache manifest hash:
- Pass criteria: all met / not met (list):
- Anything unusual:
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
