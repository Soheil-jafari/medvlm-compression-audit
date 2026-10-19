# Run card RC-TT0: teacher feasibility (one LoRA step of the 7B per setup, then rule 3)

## 1. What and why

- Stage: S05 (our own teacher, D01 amendment of 2026-10-18)
- Experiments: none trained; measurements on synthetic images at side 672.
- Purpose: the published SimulaMet adapter was trained on the official train split, which
  holds almost every image of any image-disjoint test split (RC-D1 run 1). We therefore train
  our own LoRA on the 7B teacher (RC-TT1). This card:
  - measures one LoRA training step of the 7B base at micro-batch 1, with warm timing and
    peak memory, in two setups, each with and without gradient checkpointing:
    - **(a) `fp16_2gpu`:** the fp16 base split over two T4s;
    - **(b) `nf4_1gpu`:** QLoRA, the base in 4-bit NF4 on one T4;
  - reads and prints the published `adapter_config.json` (rank, alpha, dropout, targets);
  - applies **rule 3** (`configs/teacher_training.yaml`, pre-registered 2026-10-18) and prints
    the chosen setup, `N` and the projected hours, all **estimates**.

  Each step runs in a process of its own, as in RC-S0b. An out-of-memory error is a result.
- Who: **Soheil**.
- Status: `done` (result file committed 2026-10-19, commit 3afb035; the run report is not
  committed). Read in section 5.

## 2. Requirements

- Platform: Kaggle notebook, **GPU T4 x2**, Internet **on**.
- Dataset access: none (synthetic images).
- **Does not wait for RC-D1.** It uses synthetic images only and attaches no dataset, so it can
  run before, or while, RC-D1's calibration step and run 3 run. RC-TT1 waits for RC-D1 run 3.
- Installs as in RC-S0b: if Kaggle's torchao is below 0.16.0, uninstall it first;
  `requirements/train.txt` brings `optimum>=1.24.0` and `bitsandbytes==0.50.2`, which NF4 needs.
- Rule 3, as written before any result:
  - **Setup:** (a) unless (a) runs out of memory or is more than 2x slower per example than
    (b). Within a setup, the faster variant that ran.
  - **N:** the largest multiple of 1,000 training examples whose projected time (warm step time
    x N) is at most **15 GPU hours**. The run counted a T4 x2 hour as 2 GPU hours, and printed
    and stored the other reading next to it. Measured 2026-10-19: a T4 x2 hour counts **once**,
    so RC-TT1 uses the other reading (38,000).
  - **Cap:** one pass over the training split's QA rows; RC-TT1 applies it, because it knows
    the split.
- LoRA settings: rank 16, alpha 32, dropout 0.05 (D06), unless the published
  `adapter_config.json` can be read here. Then its rank, alpha and dropout are printed, stored
  and used by RC-TT1. The targets stay the language model's linear layers, with the vision
  tower frozen.
- Expected runtime (**estimate**): about 10 to 20 minutes.
  - The 16.6 GB download: RC-M1 downloaded the teacher in about 1 min 16 s.
  - Four children, each loading the 7B: the fp16 load took about 41 s in RC-M1; the NF4 load
    quantizes while it loads and was not measured.
  - Two LoRA steps per child.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil.
- **Kaggle settings:** GPU T4 x2 (both T4s); Internet on.
- **Attach (Add Input):** nothing.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-TT0"`,
  `SEED = None`, `DATA = None`, `FROM = []`. It runs `python scripts/run_card.py --card RC-TT0`:
  the version check, then the step below, then section 6 printed. Steps:
  - feasibility, rule 3
- **Hand-offs it writes:** none (its result file reaches RC-TT1 through git).
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/`
  (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured
  2026-10-19). The session time limit is unknown here (TODO: the team checks on Kaggle).

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip show torchao 2>/dev/null | grep Version  # Kaggle ships 0.10.0; peft 0.21.2 needs 0.16.0 or newer
pip uninstall -y torchao                     # only if the version above is below 0.16.0
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
python -m pytest -q tests/test_teacher_training.py tests/test_results.py   # CPU; must pass first

python scripts/teacher_feasibility.py run 2>&1 | tee rc_tt0.txt    # all five steps, then rule 3

python -m pytest -q tests/test_results.py                  # must pass before you commit
pip freeze | grep -E "torch|transformers|peft|accelerate|optimum|bitsandbytes"
```

To redo one step: `python scripts/teacher_feasibility.py run --steps nf4_1gpu`. Its earlier
rows are replaced, and rule 3 is applied again. To reprint the rule from the committed file:
`python scripts/teacher_feasibility.py rule`.

## 4. What to commit back

- `results/smoke/RC-TT0/teacher_training.csv`. It holds:
  - the four setups (`<setup>_lora_step_ok_mb1`, `_seconds_mb1`, `_peak_gpu_bytes_..._mb1`, with
    `lora_gc_step` for gradient checkpointing);
  - the LoRA settings (`lora_rank`, `lora_alpha`, `lora_dropout`, `lora_from_published`,
    `lora_targets_match`);
  - rule 3's rows (`rule_*`).

  RC-TT1 reads this file: nobody types a number.
- The run report, with the whole printout of the `lora_config` step and the rule 3 lines.

## 5. How the result is read

- `rule_setup_fp16_2gpu` is 1 for setup (a) and 0 for (b). `rule_gradient_checkpointing` is 0
  or 1.
- `rule_n_examples` is N with a T4 x2 hour counted twice (19,000 in the committed run).
  `rule_n_examples_if_t4x2_counts_once` is the other reading (38,000). A T4 x2 hour counts once
  (measured 2026-10-19), so RC-TT1 uses 38,000 (`rule.t4x2_counts_twice: false`).
- The committed run: setup (a) `fp16_2gpu` without gradient checkpointing, 1.39 s per example;
  peak GPU memory 22.9 GB over the two T4s (18.1 GB with checkpointing); (b) ran too (1.59 s,
  11.8 GB). `lora_from_published` 1: rank 16, alpha 64, dropout 0.05 from the published
  `adapter_config.json`. `lora_targets_match` 0: its `target_modules` value is neither the
  string `all-linear` nor exactly the list of the seven decoder projections (the check of that
  commit did not parse a regular expression). The value itself is in the run's printout of the
  `lora_config` step, which is not committed; RC-TT1 reads, prints and uses it (D01,
  2026-10-19).
- `rule_projected_wall_hours` and `rule_projected_gpu_hours` are **estimates**: the warm step
  time at micro-batch 1 x N. The real run steps the optimizer only every 16 examples, so it
  should be slightly faster.
- If (b) is chosen, the teacher is still evaluated in fp16 with the trained adapter. RC-TT1
  also records the NF4 validation result.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- The printout of the `lora_config` step (all of it):
- For each setup: ok, warm step seconds, peak memory (or the error line):
- Rule 3 lines (setup, N, projected hours):
- Session wall time:
- Anything unusual:
