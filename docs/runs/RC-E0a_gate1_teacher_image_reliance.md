# Run card RC-E0a: Gate 1, does the teacher use the image?

## 1. What and why

- Stage: S05
- Experiments: E0, no budget, seed 0 (the seed only drives which random control
  regions are drawn)
- Purpose: run the three Gate 1 checks of `docs/decisions/D04_gate1.md` on the
  **validation** split with the teacher, and write the metrics and the verdict.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when
they are:

- D04 approved (done 2026-10-12) and its numbers written into `configs/experiments/E0.yaml`
  (`image_reliance_check.gate1`, `fill`, `mask_findings`).
- RC-D1 done, label rules and the finding list frozen (`configs/domain/endoscopy.yaml`,
  `configs/data/kvasir_vqa_x1.yaml`), so the validation probes exist.
- RC-M1 passed. **Met 2026-10-15.**
- `inputs.image_longest_side` set at the student-selection freeze.

Platform: Kaggle notebook, **2 x T4**, Internet **on**. Dataset access: your own copy of
Kvasir-VQA-x1 and Kvasir-SEG (from RC-D1) and the **private** RC-D1 outputs
(`splits.json`, `mask_links.json`) attached as a private Kaggle dataset. Nothing from
those goes into the repository.

Expected runtime: model download (about 17 GB) plus a few minutes of scoring.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data               # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>   # splits.json, mask_links.json
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results

python scripts/00_check_teacher_image_reliance.py --exp E0 --seed 0
python -m pytest -q tests/test_results.py                           # must pass before you commit
```

If the script stops with "configuration incomplete", a setting above is still open:
do not fill it in yourself, tell the agent through the inbox.

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics.csv`
- This card with section 6 filled.

## 5. How the result is read

`gate1_pass` equals 1 only if all three checks hold (`gate1_accuracy_ok`,
`gate1_blank_drop_ok`, `gate1_lesion_effect_ok`). The agent writes the verdict
proposal into `docs/results/E0.md`; the team confirms it in `docs/decisions_log.md`.
A failed gate stops the loop; it is not a reason to change thresholds.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Wall time:
- `gate1_pass` and the three checks:
- Anything unusual (errors, retries, changed settings):
