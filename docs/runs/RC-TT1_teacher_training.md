# Run card RC-TT1: train our teacher adapter (resumable, Save & Run All)

## 1. What and why

- Stage: S05 (D01 amendment of 2026-10-18)
- Experiment: E0's adapter. Qwen2.5-VL-7B-Instruct at its pinned revision, with a new LoRA
  trained on **our training split only**, at image side 672 (the students' input).
- What the run does:
  1. Reads the setup, gradient checkpointing, N and the LoRA rank, alpha and dropout from
     RC-TT0's committed `results/smoke/RC-TT0/teacher_training.csv` (commit 3afb035). No number
     is typed by hand. A T4 x2 hour counts **once** against the Kaggle quota (measured
     2026-10-19), so N is RC-TT0's `rule_n_examples_if_t4x2_counts_once`: **38,000**.
     `configs/teacher_training.yaml` (`rule.t4x2_counts_twice: false`) says so; no flag is
     needed.
  2. Reads the published adapter's `adapter_config.json` at its pinned revision, prints its
     `target_modules` as written, and puts the LoRA on those modules inside the language
     model's decoder layers; the vision tower stays frozen (D01, 2026-10-19). It counts the
     adapted linear layers per target and **stops before training** if a published target
     matches none there, or if the file's rank, alpha or dropout differ from RC-TT0's.
  3. Takes the first N examples of one seeded order of the training split, stratified by
     question type, so rare types are included. N is capped at one pass over the split.
  4. Trains, saving its state about every 30 minutes (every 81 updates; an estimate from
     RC-TT0's step time).
  5. When started again, continues from the furthest saved state it finds, also one in a
     previous notebook version's output attached as input.
  6. Writes the adapter as a hand-off: ONE private dataset, whose manifest holds the SHA-256
     of every file, the commit, the config hash, N, the steps and the setup.
  7. Runs the teacher's validation evaluation and Gate 1. Both wait cleanly while the finding
     list is still open; RC-E0a and RC-E0b then run them.
- Who: **Soheil**.
- Status: `ready` (2026-10-29): RC-D1 run 4b's shared data are the private dataset
  `medcompress-data`; RC-TT0's result file is committed. Run it with the cells of section 3.

## 2. Requirements

- RC-D1 **run 4b** done (2026-10-29): its shared-data dataset **`medcompress-data`** attached.
  It is the split of D02 amendments 5-7: `full:0.92:or_crop:0.80` and the held-out filter at
  F 0.86, `split_sha256` `54dfa70a9aa99ea86262447acfd16fe4d4e871272781a31f39c9f330f4f7d090`.
  - That hash is pinned in `configs/data/kvasir_vqa_x1.yaml`. The script refuses any other
    shared data (exit code 5) when it attaches them, before the model is loaded, and it rebuilds
    the training split and hashes it again.
  - Not the shared data of runs 2, 3 or 4: their splits are not accepted.
- RC-TT0 done: `results/smoke/RC-TT0/teacher_training.csv` is committed (commit 3afb035). It
  gives setup (a), fp16 over two T4s, without gradient checkpointing; 1.39 s per example (warm
  step); LoRA rank 16, alpha 64, dropout 0.05, read from the published `adapter_config.json`.
- Platform: Kaggle notebook, **GPU T4 x2** (setup b uses one of them), Internet **on** (model
  download).
- Recipe:
  - from the SimulaMet `args.json` where it applies: lr 2e-5, cosine schedule, 3 % warm-up,
    weight decay 0.1;
  - LoRA on the published adapter's target modules inside the language model, vision tower
    frozen; rank 16, alpha 64, dropout 0.05 from RC-TT0 (the published values);
  - 16 examples per update (micro-batch 1 x 16, like the students);
  - fp16 with loss scaling and fp32 LoRA weights;
  - answers only, no distillation.

  The settings are in `configs/teacher_training.yaml`.
- Runtime (**estimates** from RC-TT0's warm step time): N = 38,000 examples, 2,375 updates of
  16, about **14.7 hours** of training on T4 x2. That is within rule 3's 15 GPU hours (rule 3
  bounds the projected training time). The real run steps the optimizer only every 16
  examples, so it may be a little faster.
- **Two sessions, one resume.** The first session stops itself on a saved step after 11 hours
  (`--stop-after-hours 11`, counted from the script's start). The version then ends normally
  and its output keeps the state; the second session resumes from it and finishes.
  - 11 is an hour below a 12-hour session limit. Not verified here (open item "Kaggle session
    time limit"): if Kaggle shows another limit, set the flag an hour below it.
  - The stop does not rely on a version stopped by Kaggle keeping its output (also not known
    here).
- **GPU quota** (**estimate**). A T4 x2 session counts once (measured 2026-10-19), so the quota
  used is the wall time of the two sessions:
  - about 11 h for the first;
  - the rest of the training, about 4 h, plus its setup, for the second;
  - **about 15 hours in all**: 14.7 h of training plus each session's setup (the clone, the
    installs, two checks of the 8,511 files, the model download and load; not measured
    here).

  That is about half of one person's 30 GPU hours per week.
- The LoRA targets were timed by RC-TT0 as D06's seven projections of the language model. If the
  published targets are a different set, the step time may differ from the estimate above.
- **Quality rule** (rule 3, pre-registered):
  - The teacher must pass Gate 1 (D04 thresholds unchanged).
  - Once E1 seed 0 exists, its validation macro balanced accuracy must be at least E1 seed 0's
    + 0.02.
  - If either fails, the teacher is trained **once** more with `--extend`, then re-checked and
    accepted and reported either way. The extension adds 50 % of N: the next N / 2 examples of
    the same order, starting from the trained adapter, with its own 3 % warm-up and cosine
    schedule.
  - After that one extension, the study's stop rule for a failed Gate 1 stays as it is.

## 2b. Shared data, hand-offs and the unattended run (2026-10-17)

- **Who:** Soheil.
- **Kaggle settings:** GPU T4 x2; Internet on.
- **Attach (Add Input), all private datasets:** `medcompress-data` (RC-D1 run 4b). To resume,
  also the output of the stopped notebook version. For `--extend`, also the teacher adapter
  hand-off.
- **Checks before anything runs:** `--data-root /kaggle/input/<shared-data>` is verified
  (the SHA-256 of every file), and the run stops if a file differs. The plan is read from
  RC-TT0's committed file, and the run stops if that file is missing.
- **Use the cells of section 3** (Save & Run All). `notebooks/run_card.ipynb` with
  `CARD = "RC-TT1"` (`python scripts/run_card.py --card RC-TT1 --data-root
  /kaggle/input/medcompress-data`) also runs it, but it has no time budget: it is only safe if
  one session is longer than the whole training. Steps:
  - train the teacher (and, at the end, the validation evaluation and Gate 1 when they can run)
- **Hand-offs it writes:** `E0_teacher_adapter` under `/kaggle/working/handoff/`. Make it ONE
  private dataset, for example `e0-teacher-adapter`, shared with the team only. RC-E0a,
  RC-E0b, RC-E9 and RC-R1 attach it with `--from`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/`
  (`docs/team_setup.md`).
- **Session limits:** a T4 x2 session counts **once** against the weekly quota (measured
  2026-10-19: a 17-minute T4 x2 session used 16 minutes). The session time limit is unknown
  here (TODO: the team checks on Kaggle). The training saves its state about every 30 minutes
  and continues from it (section 3).

## 3. Commands

Kaggle settings for both sessions: **GPU T4 x2**, Internet **on**, **Save Version -> Save & Run
All**, so that `/kaggle/working` (and the saved state in it) becomes the version's output.

**Session 1** (Add Input: `medcompress-data`). One cell; nothing to edit:

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
python scripts/prepare_data.py verify --data-root $DATA        # must print OK (8,511 files)
python -m pytest -q tests/test_training.py -k "time_budget or stopped_training"   # tiny model, CPU
python scripts/train_teacher.py --data-root $DATA --dry-run      # the plan; loads no model
python scripts/train_teacher.py --data-root $DATA --stop-after-hours 11 \
    2>&1 | tee /kaggle/working/rc_tt1_session1.txt
echo "train_teacher exit code: ${PIPESTATUS[0]} (3 = stopped on a saved step, resume; 0 = finished)"
mkdir -p /kaggle/working/results_to_commit/results
cp -r results/E0 /kaggle/working/results_to_commit/results/ 2>/dev/null || true
```

What it prints, in order:
- `OK: 8511 files match the manifest ...` (the shared data, every file's SHA-256);
- the two training tests passing;
- the dry run:
  - `split: split_sha256 54dfa70a... (pinned, checked); 106627 training rows`;
  - `N = 38000 of 106627 training rows (a T4 x2 hour counted once)`;
  - `2375 updates of 16 ... a save every 81 updates; projected about 14.7 h`;
- the training:
  - `published target_modules (as written): ...` and `LoRA targets inside the decoder layers:
    ...`, before the model is loaded;
  - then `adapted linear layers per target: ...` before the first update;
  - then `starting from step 0`;
- after 11 hours: `STOPPED for a resume ... step <k> of 2375 is saved in
  /kaggle/working/checkpoints/E0/teacher_resume`, and exit code 3.

  A save also comes every 81 updates (about 30 minutes, an estimate), so a crash loses at most
  that.

If the hub cannot be reached for the published `adapter_config.json`, the training stops with
exit code 5 before the model is loaded. Give a downloaded copy with
`--published-adapter-config <path>`.

**Session 2, the resume** (a new version of the same notebook; Add Input: `medcompress-data`
**and** the output of the session-1 version: Your Work -> the notebook -> version 1 -> Output).
One cell; nothing to edit:

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
python scripts/prepare_data.py verify --data-root $DATA        # must print OK (8,511 files)
find /kaggle/input -path '*teacher_resume/current/progress.json'   # the state of session 1
python scripts/train_teacher.py --data-root $DATA --stop-after-hours 11 \
    2>&1 | tee /kaggle/working/rc_tt1_session2.txt
echo "train_teacher exit code: ${PIPESTATUS[0]} (3 = stopped on a saved step, resume; 0 = finished)"
mkdir -p /kaggle/working/results_to_commit/results
cp -r results/E0 /kaggle/working/results_to_commit/results/ 2>/dev/null || true
ls /kaggle/working/handoff                                      # E0_teacher_adapter when done
```

- It prints `continuing from step <k> saved in /kaggle/input/...` and finishes the remaining
  updates.
- Then it writes the adapter hand-off `E0_teacher_adapter` under `/kaggle/working/handoff/` and
  the `teacher_*` rows. The validation and Gate 1 wait cleanly on the finding list.
- It must use the same commit's settings as session 1. If a training setting changed in
  between, the resume refuses ("belongs to another run"): tell the agent rather than removing
  the state.
- If session 2 also prints `STOPPED for a resume` (exit code 3), run the same cell once more,
  with session 2's output attached instead.
- To name the folder with the state yourself, add `--resume-from /kaggle/input/<session-1-output>`.

**The one extension** (only if the quality rule fails; decided after Gate 1 and E1 seed 0):

```bash
python scripts/train_teacher.py --data-root /kaggle/input/medcompress-data --extend \
    --from /kaggle/input/<teacher-adapter> --stop-after-hours 11
```

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics.csv`: the `teacher_*` training rows (with
  `teacher_lora_rank`, `_alpha`, `_dropout`, `teacher_lora_adapted_layers`,
  `teacher_lora_target:<name>` and `teacher_t4x2_counts_once`), and the E0 validation and
  Gate 1 rows if they ran.
- The run report (`docs/runs/reports/`).
- The hand-off folder `E0_teacher_adapter`, made ONE private dataset. Write its name in the
  report. Never commit the adapter.

## 5. Pass criteria

| Check | Where | Pass if |
|---|---|---|
| The pinned split | the dry run's `split:` line | `split_sha256 54dfa70a... (pinned, checked)` |
| Plan followed | `teacher_train_steps`, `teacher_n_examples` | 2,375 and 38,000 (`ceil(N / 16)`, N as RC-TT0's `rule_n_examples_if_t4x2_counts_once`, capped at the training rows) |
| Published targets used | `teacher_lora_target:<name>` | every published target adapts at least one layer (a projection of the decoder: 28, one per decoder layer) |
| Training sane | `teacher_train_loss_final` | finite |
| Hand-off intact | `python scripts/handoff.py verify /kaggle/input/<teacher-adapter>` | prints OK |

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- Sessions used (one line each: the notebook version, start step, end step, wall time, the exit
  code):
- N, steps, final loss, peak memory:
- The printed `published target_modules (as written)` line and `adapted linear layers per
  target` line:
- Name of the private teacher-adapter dataset:
- Validation and Gate 1 (ran, or waited on the finding list):
- Anything unusual:
