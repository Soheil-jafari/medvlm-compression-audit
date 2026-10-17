# Run card RC-TT1: train our teacher adapter (resumable, Save & Run All)

## 1. What and why

- Stage: S05 (D01 amendment of 2026-10-18)
- Experiment: E0's adapter. Qwen2.5-VL-7B-Instruct at its pinned revision, with a new LoRA
  trained on **our training split only**, at image side 672 (the students' input).
- What the run does:
  1. Reads the setup, gradient checkpointing, N and the LoRA settings from RC-TT0's committed
     `results/smoke/RC-TT0/teacher_training.csv`. No number is typed by hand.
  2. Takes the first N examples of one seeded order of the training split, stratified by
     question type, so rare types are included. N is capped at one pass over the split.
  3. Trains, saving its state about every 30 minutes (an estimate from RC-TT0's step time).
  4. When started again, continues from the furthest saved state it finds, also one in a
     previous notebook version's output attached as input.
  5. Writes the adapter as a hand-off: ONE private dataset, whose manifest holds the SHA-256
     of every file, the commit, the config hash, N, the steps and the setup.
  6. Runs the teacher's validation evaluation and Gate 1. Both wait cleanly while the finding
     list is still open; RC-E0a and RC-E0b then run them.
- Who: **Soheil**.
- Status: `ready` (it stops with a clear message until RC-TT0's result file is committed).

## 2. Requirements

- RC-D1 run 2 done: the shared-data dataset (image-disjoint split) attached.
- RC-TT0 done, and its `results/smoke/RC-TT0/teacher_training.csv` committed.
- Platform: Kaggle notebook, **GPU T4 x2** (setup b uses one of them), Internet **on** (model
  download).
- Recipe:
  - from the SimulaMet `args.json` where it applies: lr 2e-5, cosine schedule, 3 % warm-up,
    weight decay 0.1;
  - LoRA on the language model, vision tower frozen; rank, alpha and dropout from RC-TT0;
  - 16 examples per update (micro-batch 1 x 16, like the students);
  - fp16 with loss scaling and fp32 LoRA weights;
  - answers only, no distillation.

  The settings are in `configs/teacher_training.yaml`.
- Runtime: at most 15 GPU hours by rule 3, as RC-TT0 projected (an **estimate**). Not known
  here: the Kaggle session time limit, and whether a stopped version keeps its output
  (TODO: check on Kaggle). The run resumes either way (section 3).
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
- **Attach (Add Input), all private datasets:** the shared data of RC-D1. To resume, also the
  output of the stopped notebook version. For `--extend`, also the teacher adapter hand-off.
- **Checks before anything runs:** `--data-root /kaggle/input/<shared-data>` is verified
  (the SHA-256 of every file), and the run stops if a file differs. The plan is read from
  RC-TT0's committed file, and the run stops if that file is missing.
- **Unattended (Save & Run All):** `notebooks/run_card.ipynb` with `CARD = "RC-TT1"`,
  `SEED = None`, `DATA = "/kaggle/input/<shared-data>"`, `FROM = []`. It runs
  `python scripts/run_card.py --card RC-TT1 --data-root /kaggle/input/<shared-data>`: the
  version check, then the step below, then section 6 printed. Steps:
  - train the teacher (and, at the end, the validation evaluation and Gate 1 when they can run)
- **Hand-offs it writes:** `E0_teacher_adapter` under `/kaggle/working/handoff/`. Make it ONE
  private dataset, for example `e0-teacher-adapter`, shared with the team only. RC-E0a,
  RC-E0b, RC-E9 and RC-R1 attach it with `--from`.
- **Results to commit:** the copies under `/kaggle/working/results_to_commit/`
  (`docs/team_setup.md`).
- **Session limits (unknown here; TODO: the team checks on Kaggle):** the session time limit,
  and whether a T4 x2 session counts once or twice against the weekly quota. This card uses
  T4 x2. The training saves its state about every 30 minutes and continues from it (section 3).

## 3. Commands

First a dry run, which prints N, the updates and the projected hours and loads no model:

```bash
python scripts/train_teacher.py --data-root /kaggle/input/<shared-data> --dry-run
```

**Start** (Save & Run All with the notebook above, or by hand in a T4 x2 session):

```bash
python scripts/train_teacher.py --data-root /kaggle/input/<shared-data> 2>&1 | tee rc_tt1.txt
```

**If the session stops:**
1. Open a new session of the same notebook.
2. **Add Input:** the output of the stopped version (Your Work -> the notebook -> its output),
   plus the shared data.
3. Run it again, the same way. The script finds `checkpoints/E0/teacher_resume` under
   `/kaggle/input` and continues from the furthest step saved. It prints "continuing from step
   ... saved in ...". To name the folder yourself, add
   `--resume-from /kaggle/input/<stopped-output>`.

**The one extension** (only if the quality rule fails; decided after Gate 1 and E1 seed 0):

```bash
python scripts/train_teacher.py --data-root /kaggle/input/<shared-data> --extend \
    --from /kaggle/input/<teacher-adapter>
```

## 4. What to commit back

- `results/E0/nobudget/seed0/metrics.csv`: the `teacher_*` training rows, and the E0
  validation and Gate 1 rows if they ran.
- The run report (`docs/runs/reports/`).
- The hand-off folder `E0_teacher_adapter`, made ONE private dataset. Write its name in the
  report. Never commit the adapter.

## 5. Pass criteria

| Check | Where | Pass if |
|---|---|---|
| Plan followed | `teacher_train_steps` | `ceil(N / 16)`, N as RC-TT0's `rule_n_examples` (capped at the training rows) |
| Training sane | `teacher_train_loss_final` | finite |
| Hand-off intact | `python scripts/handoff.py verify /kaggle/input/<teacher-adapter>` | prints OK |

## 6. Run report (filled by the person who ran it)

- Run by / date:
- Commit actually used:
- Hardware and library versions:
- Sessions used (one line each: start step, end step, wall time):
- N, steps, final loss, peak memory:
- Name of the private teacher-adapter dataset:
- Validation and Gate 1 (ran, or waited on the finding list):
- Anything unusual:
