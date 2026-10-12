# Run card RC-S1: Round 1 screen of the candidate pool

## 1. What and why

- Stage: S05b (decision D05, Round 1)
- Experiments: none (selection runs on the validation split; no budget)
- Purpose: put every candidate that passed Round 0 through the same short test. Each
  candidate is scored zero-shot, fine-tuned with the same short LoRA run on
  ground-truth answers (`compute_unit_b` examples, same rows for all), and scored again
  on the same fixed subset of validation images. The result is the score S of D05, a
  rank per candidate and the five that advance. Gate G2 (does the model use the image)
  removes candidates that answer as well on a blank image.
- Status: `blocked` (see section 2)

## 2. Requirements

Blocked until all of these are true; the agent changes the status to `ready` when they are:

- RC-S0 done and its ranking (`results/selection/R0/ranking.csv`) committed; failed
  candidates replaced by reserves (each reserve needs its own Round 0 run).
- RC-D1 done; the finding list, label rules, prevalence groups and lesion-size
  thresholds written into `configs/domain/endoscopy.yaml`;
  `image_reliance_check.mask_findings` set.
- The team confirmed the Round 1 settings (proposals come from the RC-S0 numbers) and
  `student_selection.frozen_on` is set: `compute_unit_b`, `rounds.R1.validation_subset.n_images`,
  `gates.G2_uses_image.threshold`, `inputs.image_longest_side`, the training recipe
  (`training.lora.*`, `learning_rate`, `batch_size`, `grad_accumulation`) and
  `evaluation.bootstrap_resamples`, `evaluation.calibration_bins`.
- Platform: Kaggle notebook, GPU T4 x2, Internet **on**. One candidate per T4: you can
  run two at once with `CUDA_VISIBLE_DEVICES=0` and `=1`.
- Dataset access: your own copy of Kvasir-VQA-x1 and Kvasir-SEG and the **private**
  RC-D1 outputs (`splits.json`, `mask_links.json`) attached as a private Kaggle
  dataset. Everyone must use the same RC-D1 outputs, because the validation subset and
  the training rows are drawn from them by seed. Nothing from these goes into the repository.
- Split of the work: the same three shares as RC-S0 (the list in
  `configs/student_selection.yaml` minus the candidates that failed Round 0).
- The remote-code rule of RC-S0 applies: read the model code before passing
  `--allow-remote-code`; the files you read are already written down in the RC-S0 card
  for your candidates, so only repeat the check if the pinned revision changed.
- Expected runtime: decided from the RC-S0 numbers; the agent fills it in before the
  card is set to `ready`.

## 3. Commands

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit && git log -1 --format=%H        # write this in section 6
export HF_HOME=/kaggle/temp/hf
pip install -q -r requirements/base.txt -r requirements/train.txt && pip install -q -e . --no-deps
export MEDCOMPRESS_DATA_ROOT=/kaggle/working/data                # your own copy (from RC-D1)
export MEDCOMPRESS_PRIVATE_ROOT=/kaggle/input/<your-rc-d1-private>
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results

# your candidates only; --allow-remote-code only for the five families of RC-S0
for c in <candidate> <candidate>; do
  CUDA_VISIBLE_DEVICES=0 python scripts/select_student.py --round R1 --candidate $c
  rm -rf $HF_HOME/hub/models--*
done
python -m pytest -q tests/test_results.py                            # must pass before you commit
```

If a script stops with "configuration incomplete", a setting is still open: do not fill
it in yourself, tell the agent through the inbox. If a candidate fails with an error
other than that, keep the log and put the last 20 lines in section 6.

**Per-probe answers.** Each run also writes `outputs/selection/R1/<candidate>/seed0/outcomes.npz`
(true/false per probe; no images). They are never committed. Upload your
`outputs/selection` folder as a **private** Kaggle dataset (name it
`<yourname>-selection-r1`); the person who ranks needs all three.

**Ranking (one person, Soheil).** After the other two have pushed their
`results/selection/R1/` files to `research-loop` and `git pull` has brought them in, attach the
three private datasets to a CPU notebook (no GPU, no dataset, no internet needed) and run:

```bash
mkdir -p /kaggle/working/outputs/selection
for d in /kaggle/input/*-selection-r1; do cp -r $d/selection/. /kaggle/working/outputs/selection/; done
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_RESULTS_ROOT=$PWD/results
python scripts/select_student.py --rank R1
python scripts/select_student.py --report
```

The command stops with a list of missing files if any candidate's answers are absent.

## 4. What to commit back

- `results/selection/R1/<candidate>/metrics.csv` for each of your candidates
- `results/selection/R1/ranking.csv` and `docs/results/SELECTION.md` (the person who ranks)
- This card with section 6 filled.

## 5. How the result is read

`ranking.csv` has, per candidate, the score S with a 95 % interval, the rank, the raw
and scaled components (macro-F1, small-finding recall, blank-image drop), the gap to the
next rank with its interval and an `advance` flag. Components are min-max scaled
across the candidates that passed the gate. The five highest points estimates advance
to Round 2; where the interval of a gap contains zero the report says the order is not
separated, and the agent mentions that in the iteration note. Candidates that fail G2
are listed with `gate_failed:G2_uses_image` and do not take part in the scaling.
Selection never uses test images.

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions (`pip freeze | grep -E "torch|transformers|peft"`):
- Per candidate: wall time, peak GPU memory (`train_peak_gpu_bytes`), errors or retries:
- Name of the private dataset with your `outcomes.npz` files:
- Anything unusual (changed settings, candidates not run):
