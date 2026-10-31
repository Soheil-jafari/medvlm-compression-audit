# Run card RC-E0a: Gate 1 (does the teacher use the image?) and E0 validation

## 1. What and why

- Stage: S05
- Experiments: E0, no budget, seed 0 (the seed only drives which random control regions are
  drawn)
- Purpose:
  1. run the three Gate 1 checks of `docs/decisions/D04_gate1.md` on the **validation**
     split with our teacher (Qwen2.5-VL-7B-Instruct with the RC-TT1 adapter), and write the
     metrics and the verdict;
  2. score the teacher on the validation split with the full evaluation suite (the reference
     row of every comparison, and rule 3's quality check against E1 seed 0). This needs no
     cache, so it runs here, in the same session (moved from RC-E0b on 2026-10-31).
- Status: `ready` (2026-10-31): RC-TT1 done (the adapter is the private dataset
  `e0-teacher-adapter`); the evaluation settings are frozen (pre-registration section 11).

## 2. Requirements

All met (2026-10-31):
- D04 approved (2026-10-12); its numbers are in `configs/experiments/E0.yaml`
  (`image_reliance_check.gate1`), the fill and `mask_findings: [polyp]` in `configs/base.yaml`.
- RC-D1 done (run 4b, 2026-10-29; shared data `medcompress-data`, `split_sha256` 54dfa70a...
  pinned and checked when attached).
- The finding list frozen (2026-10-31): polyp, ulcerative colitis, oesophagitis; display names
  from the dataset's wording; `evaluation.max_new_tokens` computed by the D04 rule (46).
- RC-M1 passed (2026-10-15); image side 672 (2026-10-17).
- RC-TT1 done (2026-10-31): our teacher adapter, trained on our training split only.

Platform: Kaggle notebook, **2 x T4**, Internet **on** (the 7B base model is downloaded from
the hub at its pinned revision). Inputs, all private: `medcompress-data` and
`e0-teacher-adapter`.

Hours (estimates, from the measured teacher speeds of RC-S0b run 2 at side 672 and the
committed counts):
- Gate 1: at most 229 labelled validation images (the sum of the validation answer counts;
  fewer if an image has two values), so at most 687 probes. Original, blank and other-image
  conditions on each, and the lesion and random masks on the at most 88 masked images: at most
  about 2,240 scored probes x 0.584 s = **about 0.36 h**.
- E0 validation: the same probe conditions (about 0.36 h) plus 1,000 free-text answers. One
  answer at a time the teacher generates 2.66 tokens per second; at about 12 tokens per answer
  (the median answer is 9 words; about 1.3 tokens per word is an assumption) that is about
  **1.3 h**. The answers are generated in batches of 8, whose speed is not measured, so it is
  probably less. Every answer at the 46-token cap would be 4.8 h (upper bound).
- Setup (install, the 17 GB model download, loading): not measured.
- In all: **about 2 h of GPU time, plus the setup**. A T4 x2 hour counts once against the
  quota.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil.
- **Kaggle settings:** GPU T4 x2 (both T4s); Internet on.
- **Attach (Add Input), all private datasets:** `medcompress-data` (the shared data of
  RC-D1); `e0-teacher-adapter` (the teacher adapter).
- **Checks before anything runs:** every stage command takes `--data-root /kaggle/input/<shared-data>` and one `--from /kaggle/input/<dataset>` per hand-off. The SHA-256 of every file is compared with the dataset's manifest; if one differs, or a hand-off the card needs is missing, the run stops before a model is loaded.
- **Unattended (Save & Run All):** the cell of section 3 (preferred), or `notebooks/run_card.ipynb` with `CARD = "RC-E0a"`, `SEED = None`, `DATA` and `FROM` filled in; it runs `python scripts/run_card.py --card RC-E0a --data-root /kaggle/input/medcompress-data --from /kaggle/input/e0-teacher-adapter`. Steps:
  - gate 1
  - E0 validation
- **Hand-offs it writes:** none (its results reach the others through git).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/` (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured 2026-10-19). The session time limit is not verified here; RC-TT1's 11-h session ran.

## 3. Commands

Kaggle settings: **GPU T4 x2**, Internet **on**, **Save Version -> Save & Run All**. Add
Input: `medcompress-data` and `e0-teacher-adapter`. One cell; nothing to edit:

```bash
%%bash
cd /kaggle/working && rm -rf medvlm-compression-audit
git clone -q -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && echo "commit: $(git log -1 --format=%H)"
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
DATA=/kaggle/input/medcompress-data
ADAPTER=/kaggle/input/e0-teacher-adapter
python scripts/prepare_data.py verify --data-root $DATA        # must print OK (8,511 files)
python scripts/handoff.py verify $ADAPTER                       # must print OK
python scripts/00_check_teacher_image_reliance.py --exp E0 --seed 0 \
    --data-root $DATA --from $ADAPTER 2>&1 | tee /kaggle/working/rc_e0a_gate1.txt
echo "gate 1 exit code: ${PIPESTATUS[0]} (0 = ran; read gate1_pass)"
python scripts/06_evaluate.py --exp E0 --seed 0 \
    --data-root $DATA --from $ADAPTER 2>&1 | tee /kaggle/working/rc_e0a_validation.txt
echo "validation exit code: ${PIPESTATUS[0]}"
python -m pytest -q tests/test_results.py                       # must pass before you commit
mkdir -p /kaggle/working/results_to_commit/results
cp -r results/E0 /kaggle/working/results_to_commit/results/
grep -E "gate1|n_probes|n_mask_positives|n_stratum|macro_balanced_accuracy|evaluation_seconds" \
    results/E0/nobudget/seed0/metrics.csv
```

What it prints:
- `OK: 8511 files match the manifest ...` and `OK: ... files match the manifest ...` (the
  adapter);
- Gate 1: `wrote <n> rows to .../results/E0/nobudget/seed0/metrics.csv; gate1_pass=<0|1>`;
- the validation rows; the last command shows the counts to read: `n_probes_scored` (how many
  probes; only images with an abnormality value are labelled), `n_mask_positives` and
  `n_stratum:0` (the small polyps of validation), and the time.

If a script stops with "configuration incomplete", a setting is open: do not fill it in
yourself, tell the agent through the inbox.

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics.csv` (from `results_to_commit/`): the `teacher_*` rows
  stay, the Gate 1 and validation rows are added.
- This card with section 6 filled.
- Commit before RC-E0b: the cache checks `gate1_pass` in the committed file.

## 5. How the result is read

`gate1_pass` equals 1 only if all three checks hold (`gate1_accuracy_ok`,
`gate1_blank_drop_ok`, `gate1_lesion_effect_ok`). The agent writes the verdict proposal into
`docs/results/E0.md`; the team confirms it in `docs/decisions_log.md`. A failed gate is not a
reason to change thresholds: rule 3's one extension of the teacher training comes first, and
after it a failed gate stops the study (pre-registration section 6).

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Wall time (Gate 1; validation):
- `gate1_pass` and the three checks:
- `n_probes_scored`, `n_mask_positives`, `n_stratum:0`:
- Anything unusual (errors, retries, changed settings):
