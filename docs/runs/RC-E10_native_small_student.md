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
- Status: `blocked` (see section 2)

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
  then the training stops with "configuration incomplete".
- The 2B's revision pinned in its model config (`sha_seen` 89644892..., recorded 2026-10-12;
  pinned from what step 1 prints, never typed from memory).
- The freeze, for the test split (as every arm).
- **Code still to come (agent), before this card is `ready`:** the 4-bit E10 (GPTQ `w4` of the
  merged E10 student, like E3 `w4`) and the two exploratory comparisons in the comparison step
  (E10 against the pruned 3B arm nearest its peak memory; E10 at 4 bit against E8). Not
  tested on the real 2B here: its LoRA targets, image processing (patch 16) and memory on one
  T4 are untested.

GPU hours (estimate, about **4 h** on T4 x2, counted once): the training of 625 updates on the
smaller model (E2's measured 3B update is 27.2 s; the 2B's is not measured) and one validation
pass (about 0.5 h), plus the setup. The 4-bit run and the test evaluations come on top (GPTQ
under 1 h, about 1.7 h of evaluation per model, the schedule's figures).

## 2b. Shared data, hand-offs and the unattended run (2026-11-08)

- **Who:** **[team]** (no owner yet).
- **Kaggle settings:** GPU T4 x2 (the training uses one T4); Internet on (the 2B download and
  the tokenizer files).
- **Attach (Add Input), all private datasets:** the shared data of RC-D1; teacher cache.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-E10"` and
  `SEED = None`; it runs `python scripts/run_card.py --card RC-E10`, which finds the shared data
  and the cache by their manifests. Steps:
  - tokenizer check (E2 or E1 recipe)
  - train E10 seed0 (stops until `training.mode` is set from the check)
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

- Run by / date:
- Commit actually used:
- Tokenizer check: ids checked / differing, answers matching, the recipe chosen:
- Revisions printed (teacher, 2B):
- Wall time, peak GPU memory, final training loss:
- Name of the private dataset with the adapter:
- Anything unusual:
