# Run card RC-S0: student feasibility on one T4 (Round 0)

> **Amended 2026-10-12 (D05 amendment):** this card covers two models only, the chosen student
> Qwen2.5-VL-3B and its fallback Qwen3-VL-2B, and one person runs it. The split of ten
> candidates below is kept for the record and is not used. Neither family is on the loader's
> list of families that need downloaded model code, so `--allow-remote-code` is not needed.

## 1. What and why

- Stage: S05b (decision D05, Round 0)
- Experiments: none (selection runs; no dataset, no budget)
- Purpose: find out for each of the two models (originally 10 candidates) whether it loads in fp16 on one T4,
  scores 20 yes/no probes on synthetic images with finite values and takes one LoRA
  step (gate G1). The same run records the numbers that the team needs to fix the
  open Round 1 settings: visual tokens per image, scoring time per probe, peak memory,
  and the time of one LoRA step.
- Status: `ready` (2026-10-15: RC-M1 passed all its checks)

## 2. Requirements

Needed RC-M1 to pass first: Round 0 reuses the loader, the yes/no scoring and the LoRA code
that RC-M1 exercises on the 3B student, so a failure there would be mistaken for a failure of
every candidate. **Met 2026-10-15:** RC-M1 passed all 8 checks.

- Platform: Kaggle notebook, GPU T4 x2, Internet **on**. Use **one** T4 only
  (`CUDA_VISIBLE_DEVICES=0`): the gate asks whether the candidate trains on one T4.
- Disk: models are downloaded one at a time; the loop below deletes each one after
  its run. `export HF_HOME=/kaggle/temp/hf`.
- Dataset access: none. No image of the study is used.
- Split of the work (each person runs their own share, in their own session):
  - Soheil: `qwen2_5_vl_3b`, `qwen3_vl_2b` (the other eight are not run)
- Five candidates ship their own model code (`ovis2_5`, `fastvlm`, `hulu_med`,
  `sail_vl2`, `internvl3_5`): the script refuses to load them without
  `--allow-remote-code`. **Before** passing it, open the model repository at the
  revision named in the candidate's config (`sha_seen`), read the `modeling_*.py` and
  `configuration_*.py` files, and check that they do no network access, no file writes
  outside the cache and no `exec`/`eval` of downloaded strings. Write what you read in
  section 6. If you are not sure, do not pass the flag and record the candidate as
  not run.
- Image size: `inputs.image_longest_side` is still open (it is decided from these
  numbers). For this run only, pass `-o inputs.image_longest_side=448` on the command
  line. Do not edit a config file.
- Expected runtime (an **estimate** from RC-M1, same image side 448): the 3B student
  downloaded in 40 s and loaded in 3 s; the 20 probes of Round 0 take about 20 x 0.199 s = 4 s
  (RC-M1 measured 0.199 s per probe for the 3B student, not the 0.12 s assumed in D07); the
  LoRA step takes seconds. With the install and the CPU test, plan about 30 minutes of the
  session for both models; the 2B fallback has not been timed yet.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip show torchao 2>/dev/null | grep Version  # Kaggle ships 0.10.0; peft 0.21.2 needs 0.16.0 or newer
pip uninstall -y torchao                     # only if the version above is below 0.16.0 (RC-M1)
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export CUDA_VISIBLE_DEVICES=0
python -m pytest -q tests/test_selection_ranking.py                  # CPU; must pass first
python -m pytest -q 2>&1 | tail -3   # the whole suite with torch: write the last line in
                                     # section 6 (checks the 2026-10-15 test fixes on a GPU machine)

# neither model ships its own code, so --allow-remote-code is not used (see the note at the top)
for c in qwen2_5_vl_3b qwen3_vl_2b; do       # the student and its fallback (D05 amendment)
  python scripts/select_student.py --round R0 --candidate $c \
      -o inputs.image_longest_side=448 2>&1 | tee rc_s0_$c.log
  rm -rf $HF_HOME/hub/models--*
done
python -m pytest -q tests/test_results.py                            # must pass before you commit
pip freeze | grep -E "torch|transformers|peft|accelerate"
```

If a candidate fails (does not load, out of memory, non-finite scores), keep its log,
do not change any setting to get it through, and note the last 20 lines of the error in
section 6. A failure is a result: it takes the candidate out of the pool.

When both models are done, run

```bash
python scripts/select_student.py --rank R0      # prints pass/FAIL per candidate
```

which writes `results/selection/R0/ranking.csv`.

## 4. What to commit back

- `results/selection/R0/<candidate>/metrics.csv` for each of your candidates
- `results/selection/R0/ranking.csv` (the person who merges)
- This card with section 6 filled, one line per candidate. Paste the "generation
  sample" line of each log; it contains no data.

## 5. How the result is read

Gate G1 is `gate_g1_hardware` = 1 (loads, scores finite values and trains one LoRA step
on one T4 in fp16). A candidate that fails is replaced by the next reserve in
`configs/student_selection.yaml` (`reserves`, in that order); the reserve then needs its
own config file and its own Round 0 run, in a new card. Gates G4 (licence) and G5 (no
test exposure) are checked by the team from the model cards and recorded under
`manual_gate_failures`.

The agent turns the numbers into proposals for the open Round 1 settings
(`student_selection.compute_unit_b`, `inputs.image_longest_side`, the validation
subset size, the G2 threshold) and puts them in the inbox; the team confirms them and
sets `frozen_on` before RC-S1.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft|accelerate"`):
- Per candidate: passed G1 (yes/no), wall time, remote code read (files and what you checked),
  visual tokens, peak memory, anything unusual:
- Candidates not run or failed, with the last lines of the error:
