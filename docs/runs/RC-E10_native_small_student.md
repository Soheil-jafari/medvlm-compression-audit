# Run card RC-E10: native small student at matched memory (E10), seed 0

## 1. What and why

- **Seeds (seed plan, pre-registration section 11 item 12, Soheil and Amir, 2026-11-08):** seed 0
  only, reported as exploratory / descriptive (no confirmatory test).
- Stage: S06 (training) and S11 (the comparison)
- Experiment: E10, no budget, seed 0 (pre-freeze addition of 2026-11-08, pre-registration
  section 11 item 13).
- Purpose: is a natively small model as good as a compressed large one? Qwen3-VL-2B-Instruct
  (the D05 fallback, `configs/models/qwen3_vl_2b_student.yaml`) is trained once with the E2
  recipe (offline distillation from the teacher cache, D06), then evaluated like every arm. It
  is compared:
  - at **matched peak memory** with the pruned 3B arm whose measured footprint is nearest its
    own (after RC-B0, which fixes the budgets);
  - **at 4 bit** against E8 (the clinically pruned 3B at 4 bit).
- Status: `ready` (2026-11-12) for the training, the validation and the benchmark (steps
  2-4). Step 1 passed on Kaggle (commit 262a4f1): `training.mode: offline_kd`, the 2B pinned at
  revision 89644892e4d85e24eaac8bacfd4f463576704203 (section 6). The 4-bit E10 and the two
  comparisons are a later card step (agent code still to come); they do not block the training

## 2. Requirements

- RC-E0b done (the teacher cache, pinned; done 2026-11-01).
- **Step 1, the tokenizer check** (CPU, minutes): the E2 recipe compares the student's logits
  with the teacher's cached top-20 token ids, so the student tokenizer must agree on every id
  the cache uses. D01 found the same `vocab.json` and `merges.txt` as the teacher, with extra
  special tokens; the check reads the cache itself
  (`scripts/check_tokenizer_match.py`):
  - every token id in the cache (answer ids and top-20 ids) maps to the same token string in
    both tokenizers; and
  - the student re-encodes at least 99 % of the cached answers into the cached ids (the rule
    of the student selection rounds, `selection.rounds.MIN_TOKENIZER_MATCH`).
  - **Both pass:** `training.mode: offline_kd` (the E2 recipe) in `configs/experiments/E10.yaml`.
  - **Either fails:** the E1 recipe (`sft`, `kd_weight: 0`, `kd_temperature: null`,
    `teacher_cache_from: null`, as `E1.yaml`), and the report and the paper say so.
  The agent sets the value from the committed `results/smoke/RC-E10/tokenizer_check.csv`; until
  then the training stops with "configuration incomplete". **Met 2026-11-12:** both pass,
  `training.mode: offline_kd`.
- The 2B's revision pinned in its model config (`sha_seen` 89644892..., recorded 2026-10-12;
  pinned from what step 1 prints, never typed from memory). **Met 2026-11-12:** `revision:
  89644892e4d85e24eaac8bacfd4f463576704203` in `configs/models/qwen3_vl_2b_student.yaml`, read
  by the model loader and, since then, by the tokenizer check too.
- The freeze, for the test split (as every arm).
- **Code still to come (agent), before the 4-bit step (not before the training):** the 4-bit E10 (GPTQ `w4` of the
  merged E10 student, like E3 `w4`) and the two exploratory comparisons in the comparison step
  (E10 against the pruned 3B arm nearest its peak memory; E10 at 4 bit against E8). Not
  tested on the real 2B here: its LoRA targets, image processing (patch 16) and memory on one
  T4 are untested.

GPU hours (estimate, about **4 h** on T4 x2, counted once): the training of 625 updates on the
smaller model (E2's measured 3B update is 27.2 s; the 2B's is not measured) and one validation
pass (about 0.5 h), plus the setup. The 4-bit run and the test evaluations come on top (GPTQ
under 1 h, about 1.7 h of evaluation per model, the schedule's figures). If the 2B's update is as slow as
the 3B's 27.2 s, the training is 4.72 h and the session about 5.3 h (2026-11-12); either way it
fits one session under the 12-hour limit, and the training saves its state every 50 updates.

## 2b. Shared data, hand-offs and the unattended run (2026-11-08)

- **Who:** Soheil (account 1; 2026-11-08).
- **Kaggle settings:** GPU T4 x2 (the training uses one T4); Internet on (the 2B download and
  the tokenizer files).
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher cache.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E10"` and
  `SEED = None`; it runs `python scripts/run_card.py --card RC-E10`, which finds the shared data
  and the cache by their manifests. Steps:
  - tokenizer check (E2 or E1 recipe)
  - train E10 seed0 (`training.mode: offline_kd`, set 2026-11-12)
  - validation E10 seed0
  - benchmark E10 seed0 (waits if a setting is open)
- **Hand-offs it writes:** `E10_seed0_adapter` (ONE private dataset, `e10-seed0-adapter`).
- **Results to commit:** `results/smoke/RC-E10/tokenizer_check.csv` after step 1;
  `results/E10/nobudget/seed0/metrics.csv` and `environment.json` after the training.

## 3. Commands

Written when the card is set to `ready` (after step 1 and the code above). Step 1 alone, on a
CPU notebook (Internet on, `e0-teacher-cache` attached):

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
CACHE=$(python scripts/find_inputs.py handoff E0_teacher_cache) || exit 1
python scripts/check_tokenizer_match.py --config experiments/E10.yaml --from $CACHE
cat results/smoke/RC-E10/tokenizer_check.csv
```

It prints both revisions, the ids that differ, the share of answers re-encoded exactly and
`PASS` / `FAIL` with the training mode the rule gives. Commit the CSV and paste the lines.

**Steps 2-4 (training, validation, benchmark; RC-SMOKE passed for RC-E10 at 9113632).** Kaggle
GPU T4 x2 (the training uses one T4), Internet on (the 2B download); attach `medcompress-data`
and `e0-teacher-cache`. Save Version -> Save & Run All:

```bash
%%bash
set -euo pipefail
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
python - <<'PY' || pip uninstall -y -q torchao
import sys
from importlib import metadata
try:
    version = tuple(int(x) for x in metadata.version("torchao").split(".")[:2])
except metadata.PackageNotFoundError:
    sys.exit(0)
sys.exit(0 if version >= (0, 16) else 1)
PY
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
python scripts/find_inputs.py data > /dev/null
python scripts/find_inputs.py handoffs
python scripts/run_card.py --card RC-E10 2>&1 | tee /kaggle/working/run_card_RC-E10.txt
```

Afterwards: make `/kaggle/working/handoff/E10_seed0_adapter` the private dataset
`e10-seed0-adapter`; send `results_to_commit/` and the run report. About 4-5.3 GPU h
(estimate); the disk holds the adapter only (tens of MB).

## 4. What to commit back

- `results/smoke/RC-E10/tokenizer_check.csv`
- `results/E10/nobudget/seed0/metrics.csv`, `efficiency.csv`, `environment.json`
- This card with section 6 filled.

## 5. How the result is read

Exploratory: the comparisons are reported with image-bootstrap intervals and no verdict. A
difference smaller than E7's spread over its 3 seeds at B2 is reported as within run-to-run
noise (seed plan item 6). If the E1 recipe was used, E10 is compared with the 3B as trained
without distillation in mind, and the text says so.

## 6. Run report (filled by the person who ran it)

**Step 1 (the tokenizer check), reported by Soheil on 2026-11-12:**

- Commit actually used: 262a4f1.
- Tokenizer check (`results/smoke/RC-E10/tokenizer_check.csv`, committed as reported): 5,238
  cache token ids checked, 0 differ; 9,991 of 10,000 cached answers re-encoded exactly (0.9991,
  the rule needs 0.99); `tokenizer_check_passed` = 1. Recipe: the E2 recipe
  (`training.mode: offline_kd`).
- Revisions printed: the student tokenizer "@ None" (unpinned at that commit: the check read
  only `revision`, while the model loader falls back to `sha_seen`). Pinned since then at
  89644892e4d85e24eaac8bacfd4f463576704203 (the hash Soheil gave, equal to the `sha_seen` of
  2026-10-12); the tokenizer check now reads the same revision as the loader. The teacher's
  printed revision was not reported.

**Steps 2-4 (training, validation, benchmark):**

- Run by / date:
- Commit actually used:
- Wall time, peak GPU memory, final training loss:
- Name of the private dataset with the adapter:
- Anything unusual:
