# Run card RC-E3: post-training quantization of E2 (w4, w8, w8a8): seed 0, and w8 at seeds 1 and 2 (H2)

## 1. What and why

- **Seeds (seed plan, pre-registration section 11 items 12 and 14, Soheil and Amir, 2026-11-08):** `w8` joins the 3-seed tier so that **H2 stays confirmatory**: seeds 0, 1, 2 (each E2 seed quantized deterministically; about 5.4 GPU h for seeds 1 and 2), and seeds 1 and 2 run under the futility rule (the plan refuses them before `results/seed_plan/comparator_B2.json` is committed, or when it says stop; a seed 1 or 2 session runs `w8` only). `w4` and `w8a8`: seed 0 only, exploratory. This replaces the seed counts written below.
- Stage: S08
- Experiments: E3, variants `w4` and `w8` (GPTQ, groups of 128, GPTQModel kernel) and `w8a8`
  (LLM.int8), one run per E2 seed (0, 1, 2): 9 runs. Each run is evaluated on the
  **validation** split now and on the **test** split after the pre-registration freeze, and
  timed with the budget protocol.
- Purpose: the "quantize after training" family of the audit (D08).
- Status: `ready` for seed 0 (2026-11-11): every requirement of section 2 is met (frozen,
  RC-E2 seed 0 done). Seeds 1 and 2 (`w8` only) wait for seed 0's futility check. Seed 0's
  measured `w8` peak decides H2's matched budget (RC-Q0 measured about 5.20 GB, nearest B3;
  docs/schedule.md section 7)

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- RC-Q0 passed; `quantization.kernel_class` and `quantization.library_versions` are set and
  the versions are pinned in `requirements/train.txt`. **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
- RC-E2 done for the seed you run (the E2 adapter on team storage).
- RC-D1 done (the calibration images are 128 training images, drawn with the run seed, the
  same ones the pruning criteria use).
- The pre-registration is frozen before the first E3 run (compression runs come after the
  freeze); `inputs.image_longest_side`, `evaluation.max_new_tokens` and
  `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, one T4, Internet on. Dataset access: your own copy of Kvasir-VQA-x1
and the private RC-D1 outputs; the E2 adapter attached as a private dataset. Nothing from those
goes into the repository.

Runtime per run: **estimate** under an hour for GPTQ on 128 calibration images (RC-Q0 reports
the speed on 16 images), plus the evaluation and timing; LLM.int8 needs no calibration. Each
run is independent; one session per seed is fine.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Amir, seed 0 (reassigned 2026-11-11); `w8` seed 1 on account 3 (Soheil), seed 2 Amir (docs/schedule.md section 7).
- **Kaggle settings:** GPU T4 x2 (the runner uses one T4); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; E2 seed<k> adapter.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E3"` and `SEED = <k>` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-E3 --seed <k>`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E3 w4 seed<k>
  - test evaluation E3 w4 seed<k>
  - run E3 w8 seed<k>
  - test evaluation E3 w8 seed<k>
  - run E3 w8a8 seed<k>
  - test evaluation E3 w8a8 seed<k>
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E3_seed<k>_quantized`, `E3_seed<k>_outcomes`.
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
python scripts/run_card.py --card RC-E3 --seed $SEED 2>&1 | tee /kaggle/working/run_card_RC-E3_seed${SEED}.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

The runner runs quantize (03), evaluate (06) and benchmark (07) in order and stops at
the first stage that fails. If a stage stops with "configuration incomplete", a setting above
is still open: do not fill it in yourself, tell the agent through the inbox. The quantized
codes go to `checkpoints/E3/<variant>/nobudget/seed<k>/quantized/` (never committed); keep them
as a private dataset.

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

For each variant `<v>` and seed `<k>`:

- `results/E3/<v>/nobudget/seed<k>/metrics.csv`, `metrics_test.csv` and `efficiency.csv`
- `results/E3/<v>/nobudget/seed<k>/efficiency_setup.json` (hardware, protocol, kernel classes)
- This card with section 6 filled.

Check before committing: `git status` shows no images, checkpoints, caches or `outcomes*.npz`.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Real low-precision kernels | `low_precision_kernels_ok` in `efficiency.csv` | equals 1 for every run; a run with 0 is not reported as quantized |
| Calibration as planned | `quantization_calibration_images` in `metrics.csv` (w4, w8) | equals 128 |
| Every decoder matrix quantized | `quantized_matrices` (w4, w8) | 7 x the number of decoder layers |

## 5b. Seed 0 run at commit 262a4f1: the offload check (2026-11-13)

At that commit `run_experiment.py` ran each variant's stages (quantize, validation evaluation,
benchmark) in one process, and the evaluation and benchmark loaded the model with
`device_map="auto"`. If the quantize stage's student was still on GPU 0, the next load was
split onto the CPU (as in RC-G2 session 1). Then either the load crashed in the adapter check
("Tensor.item() cannot be called on meta tensors"), or, when only modules without LoRA weights
were moved (e.g. `lm_head`), it ran on silently, slower and with wrong memory. The test and
unfiltered test evaluations are separate steps (their own process) and are not affected.

**Check** (in the session's output): `grep -n -i -E "offload|meta device|meta tensors"
/kaggle/working/run_card_RC-E3*.txt /kaggle/working/logs/RC-E3_seed0.log`.
- No match: the evaluation and benchmark were on the GPU; keep the results.
- Any match: the validation metrics and `efficiency.csv` of that variant are invalid; run
  `--card RC-E3 --seed 0` again on the fix commit. A second check: each variant's
  `peak_memory_bytes` should be near RC-Q0's (w4 3,797,138,944; w8 5,197,114,880; w8a8
  5,073,484,288 bytes, synthetic images) and `warm_latency_s` not far above E2's 0.987 s; an
  offloaded run shows a much lower peak and a much higher latency.

## 6. Run report (filled by the person who ran it)

Seed 0 at `d2322fa` failed about 2.2 minutes into `w4` at the first Cholesky factorization
(leading minor 1). That commit already cast activations to FP32, so the traceback alone
cannot establish FP16 overflow. Rerun seed 0 in full after the Hessian validation fix.
For a read-only diagnostic, use the setup in section 3 and replace the runner command and
final echo with:

```bash
DATA=$(python scripts/find_inputs.py data)
ADAPTER=$(python scripts/find_inputs.py handoff E2_seed0_adapter)
CUDA_VISIBLE_DEVICES=0 python scripts/check_gptq_inputs.py --exp E3 --seed 0 \
  --variant w4 --layers 0-2 --data-root "$DATA" --from "$ADAPTER"
```

The diagnostic prints per-batch statistics for each linear on the unquantized E2 parent.
A finite FP32 Hessian and a non-finite FP16 Hessian demonstrate overflow in the diagnostic's
FP16 path for that batch. The full rerun checks the actual accumulated Hessians. If finite
Hessians require damping retries, report the retry count and maximum effective damping
from the metrics, and the per-matrix entries from the checkpoint manifest.

- Run by / date / seeds:
- Commit actually used:
- Hardware and `pip freeze | grep -E "torch|transformers|peft|gptqmodel|bitsandbytes"`:
- Wall time per variant:
- Pass criteria: all met / not met (list):
- Anything unusual (errors, retries, changed settings):
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
