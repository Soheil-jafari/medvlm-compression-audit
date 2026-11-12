# Run card RC-E9: the teacher at 4 bit (supplementary)

## 1. What and why

- Stage: S08
- Experiments: E9, the teacher (Qwen2.5-VL-7B with the Kvasir adapter merged) quantized with
  the E3 `w4` recipe (GPTQ, groups of 128), one run (seed 0), evaluated on validation now and on
  test after the freeze, and timed with the budget protocol.
- Purpose: a supplementary comparison at matched memory: a quantized big model against the
  compressed small ones (D08).
- Status: `ready` (2026-11-12, again): the first attempt (commit 262a4f1) ran out of GPU memory in
  GPTQ (section 6); fixed without changing the result. Run it again on the fix commit

## 2. Requirements

Blocked until all of these are true; the agent sets the status to `ready` when they are:

- RC-TT1 done: E9 quantizes the teacher with **our** adapter (D01 amendment 2026-10-18); attach its hand-off with `--from`.
- RC-Q0 passed; `quantization.kernel_class` and the library versions are set and pinned. **Met 2026-10-15:** RC-Q0 chose `TorchLinear`; gptqmodel 7.6.0 and bitsandbytes 0.50.2 are
  pinned.
- RC-D1 done (calibration: the 128 training images of seed 0, the same as the pruning criteria).
- The pre-registration is frozen; `inputs.image_longest_side`, `evaluation.max_new_tokens` and
  `budgets.measurement.hardware` are set.

Platform: Kaggle notebook, **2 x T4** (the 16-bit teacher does not fit one T4 while it is being
quantized), Internet on. Dataset access as in RC-E3.

Runtime: **estimate** one to two hours (GPTQ on the 7B decoder; its MLP matrices are the
largest). If a GPU runs out of memory during the rounding of the MLP matrices, write the error
in section 6 and stop: the agent will move that step to the CPU.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil, on account 3 (lent by Farzan; 2026-11-08); runs before RC-E4 seed 0 there (2026-11-11).
- **Kaggle settings:** GPU T4 x2 (both T4s); Internet on.
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher adapter.
- **Checks before anything runs:** every stage command takes `--data-root $DATA` and one `--from` per hand-off, found by their manifests, never by fixed paths: `DATA=$(python scripts/find_inputs.py data) || exit 1` (the folder of a `bundle_manifest.json` with `private/` next to it) and `X=$(python scripts/find_inputs.py handoff <name>) || exit 1` (the folder of a `bundle_manifest.json` whose path contains the hand-off's name); a missing input stops the cell with a clear message. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E9"` and `SEED = None` (`DATA` and `FROM` stay empty); it runs `python scripts/run_card.py --card RC-E9`, which finds the shared data and every attached hand-off by their manifests. The runner checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of `requirements/`) and the GPUs, runs the steps below, and prints section 6 (also saved as `/kaggle/working/run_report_<card>.md`). Steps:
  - run E9 seed0
  - test evaluation E9 seed0
- **Hand-offs it writes** (each folder under `/kaggle/working/handoff/` becomes ONE private dataset, shared with the team only; manifest with SHA-256, commit, config hash, experiment, seed, budget): `E9_seed0_quantized`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown here (TODO: the team checks on Kaggle). This card uses T4 x2. A training run saves its state every 50 updates and continues from it in a new session (`docs/team_setup.md`, "If a session stops").

## 3. Commands

One cell (Kaggle settings and inputs of section 2b; **Save Version -> Save & Run All**). It finds the shared data and the attached hand-offs by their manifests (`scripts/find_inputs.py`), stops with a clear message if the shared data are missing, and runs the steps of section 2b with `scripts/run_card.py`, which also checks that every hand-off the card reads is there. Then it writes the hand-offs under `/kaggle/working/handoff/` and the files to commit under `/kaggle/working/results_to_commit/`.

```bash
%%bash
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
python scripts/run_card.py --card RC-E9 2>&1 | tee /kaggle/working/run_card_RC-E9.txt
echo "run_card exit code: ${PIPESTATUS[0]}"
```

Keep `outputs/` (the per-probe `outcomes*.npz` files) as a private dataset too: the paired
comparisons (RC-C1) need the test-split ones. Write its name in section 6.

## 4. What to commit back

- `results/E9/nobudget/seed0/metrics.csv`, `metrics_test.csv`, `efficiency.csv` and `efficiency_setup.json`
- This card with section 6 filled.

## 5. Pass criteria (decided before the run)

| Check | Where | Pass if |
|---|---|---|
| Real low-precision kernels | `low_precision_kernels_ok` | equals 1 |
| Calibration as planned | `quantization_calibration_images` | equals 128 |

## 6. Run report (filled by the person who ran it)

**Attempt 1 (reported by Soheil, 2026-11-12; commit 262a4f1, T4 x2): failed.**
`torch.OutOfMemoryError` in `compression/quantization/gptq.py` line 161 (the dampening, `h +=
damping * mean(diag(h)) * eye(n)`): tried to allocate 1.34 GiB on GPU 0 with 13.34 GiB in use.
The teacher's `down_proj` Hessian is 18,944 x 18,944 in float32 (1.34 GiB); the code held the
caller's Hessian, its copy, the identity and its scaled copy at once (4 x 1.34 GiB on top of
about 7.7 GiB of weights), and would have held five such matrices at its Cholesky step.

**The fix (2026-11-12; the results are unchanged):**
- the dampening adds to the diagonal in place (no identity matrix), and each (in x in) matrix is
  freed as soon as the next one is made: at most three of them are alive (the caller's Hessian
  and two), 4.33 GiB for `down_proj` with the weight copy and the codes
  (`gptq.gptq_peak_bytes`); each Hessian is freed after its last user;
- the 128 calibration inputs stay on the CPU and go to the GPU one at a time (about a gigabyte
  of float32 pixel values had been kept on GPU 0, an estimate);
- the teacher is split over the two T4s under per-GPU caps that keep the GPTQ room plus 1.5 GiB
  free on each (`ptq.gptq_max_memory`): caps 9,379,446,784 bytes (8.74 GiB) per T4, which the
  balanced split of the 15.45 GiB of weights stays under (about 7.7 GiB each).
- Tests: the codes and scales are identical to the old code's, bit for bit, for single matrices
  (4 and 8 bit, with a dead input) and for the whole decoder of the tiny model (the E3 / E8
  path).
- **Expected peak per GPU (estimate):** about 7.7 GiB of weights + 4.33 GiB of GPTQ buffers
  (only while that GPU's `down_proj` is rounded) + up to about 0.5 GiB (the layer's other
  Hessians, one example's activations, its inputs): **about 12.0-12.6 GiB of the T4's 14.56
  GiB**. By construction even a GPU filled to its cap keeps 4.33 + 1.5 GiB free.

**Attempt 2:**

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- Wall time and peak GPU memory:
- Pass criteria: all met / not met (list):
- Anything unusual:
- Names of the private datasets with the checkpoints and `outcomes*.npz` files:
