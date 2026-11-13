# Team setup: running a card from your own Kaggle account (Soheil, Amir, account 3)

Since 2026-11-08 the quota is three Kaggle accounts x 30 h = 90 h per week: Soheil's, Amir's,
and one lent by Farzan (account 3; Soheil starts its sessions). Farzan left the project and is
not an author. **Share every hand-off dataset to Soheil's account** (Kaggle: dataset settings,
Sharing, add his user), so each account can attach it.

Read this once before your first card. Who runs what each week is in `docs/schedule.md`. What
each card needs and writes is in section 2b of the card (`docs/runs/`).

## 1. What you need

- **A Kaggle account** with GPU and Internet access. If Kaggle asks you to verify your phone
  number for them, do so.
- **Collaborator access to the GitHub repository**, to push results to `research-loop`. Soheil
  adds you: repository **Settings -> Collaborators -> Add people**, with your GitHub username.
  Accept the invitation from your email or GitHub notifications.
- **Access to the shared datasets on Kaggle.** Soheil (or whoever made a dataset) adds your
  Kaggle username as a collaborator on each private dataset. You never need your own copy of
  Kvasir: the shared data is the copy RC-D1 made.
- Nothing secret goes into the repository or into a notebook cell: no tokens, no passwords.

## 2. Which commit

The notebook clones the latest `research-loop`. If `docs/schedule.md` or the inbox names a
commit for your run, put it in `COMMIT` in the notebook's first cell. The runner writes the
commit it used into the run report either way.

On your own machine (for committing results):

```bash
git clone -b research-loop https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs          # path roots outside the clone,
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints  # so removing the clone keeps
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache              # every output
```

**Path roots outside the clone (2026-11-04).** The configs' default output, checkpoint and
cache roots are relative paths, so they land inside the clone, and a cell that removes the
clone at the end (`rm -rf medvlm-compression-audit`, to keep the version output small) removes
them too. That deleted RC-E0a's `outcomes.npz`. So every cell sets, before any script runs:

```bash
export MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs
export MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints
export MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache
```

On Kaggle, every stage script prints a `WARNING` when one of these roots still resolves inside
the clone (`src/medcompress/cli.py`), and `scripts/run_card.py` sets them itself.

## 3. Attach the shared datasets

In the Kaggle notebook: **Add Input -> Datasets**, find the dataset (it is listed among the
datasets shared with you) and add it. It appears read-only somewhere under `/kaggle/input/`
(for our private datasets `/kaggle/input/datasets/<user>/<slug>/...`, sometimes a folder or two
deeper). So no cell uses a fixed path: every input is **found by its manifest**
(`scripts/find_inputs.py`, `src/medcompress/inputs.py`):

- the shared data (RC-D1, `medcompress-data`): the folder of a `bundle_manifest.json` that has
  a `private/` folder next to it: `DATA=$(python scripts/find_inputs.py data) || exit 1`;
- a hand-off (`E0_teacher_adapter`, `E0_teacher_cache`, `E2_seed1_adapter` ...): the folder of
  a `bundle_manifest.json` whose path contains the hand-off's name (case and `-` against `_` do
  not matter, so the slug `e2-seed1-adapter` matches `E2_seed1_adapter`):
  `CACHE=$(python scripts/find_inputs.py handoff E0_teacher_cache) || exit 1`;
- `python scripts/find_inputs.py handoffs` lists every attached hand-off.

If an input is missing (or attached twice), the command prints `STOP: ...` with what to attach
and the cell stops. In `notebooks/run_card.ipynb`, leave `DATA = None` and `FROM = []`: the
runner finds the shared data and every attached hand-off the same way.

Before anything runs, the runner checks the SHA-256 of every file against the dataset's
manifest. It stops if a file differs or if a hand-off the card needs is missing, so a wrong or
incomplete dataset costs minutes, not GPU hours. You can check a dataset by hand:

```bash
python scripts/prepare_data.py verify --data-root "$(python scripts/find_inputs.py data)"
python scripts/handoff.py verify "$(python scripts/find_inputs.py handoff E2_seed1_adapter)"
```

## 4. Run a card with Save & Run All

1. Open `notebooks/run_card.ipynb` in Kaggle: **File -> Import Notebook**, from the GitHub URL
   of the file or from a downloaded copy.
2. Notebook settings:
   - **Accelerator** as section 2b of the card says (GPU T4 x2; a one-GPU card uses one T4);
   - **Internet on**;
   - the inputs of section 3 attached.
3. Fill in the first cell: `CARD`, `SEED` (for cards split by seed), `DATA`, `FROM`, and
   `EXTRA` if the card needs it.
4. **Save Version -> Save & Run All.** The run needs no one at the keyboard. The runner, in order:
   - checks the library versions (torchao 0.16.0 or newer, optimum 1.24.0 or newer, the pins of
     `requirements/`) and the GPUs;
   - verifies the inputs;
   - runs the card's steps;
   - writes the hand-offs;
   - prints the report.
5. When the version has finished, open it and its **Output** tab.

## 5. What to take from the output

| Output | What to do with it |
|---|---|
| `run_report_<card>[_seed<k>].md` | Fill in your name and anything unusual. It is also inside `results_to_commit/` (next row). |
| `results_to_commit/` | Download it and copy its contents into your clone, keeping the paths. It holds the results files of the run and the report under `docs/runs/reports/`. Then commit (section 6). |
| `handoff/<name>/` | Each folder becomes **one private** Kaggle dataset: **New Dataset** from the output folder, visibility **Private**, then add the other two team members as collaborators. Write the dataset's name into the report. |
| `logs/<card>.log` | Keep it. Paste the last 30 lines into the report if a step failed. |

Name the hand-off datasets after the folder, in lower case, for example `e2-seed1-adapter` or
`e0-teacher-cache`. Never make one public: they hold copies of Kvasir images or models trained
on them (Kvasir-VQA/x1: CC BY-NC 4.0; Kvasir-SEG: research and educational use only).

## 6. Commit the results to research-loop

```bash
cd medvlm-compression-audit
git checkout research-loop && git pull --ff-only origin research-loop
cp -r ~/Downloads/results_to_commit/. .          # same paths as in the repository
git status                                       # only results/... and docs/runs/reports/...
python -m pytest -q tests/test_results.py        # must pass
git add results docs/runs/reports
git commit -m "RC-E2 seed 1 results"
git pull --rebase origin research-loop && git push origin research-loop
```

Each run writes its own files: `results/<ID>/.../seed<k>/` and its own report file. Two
people's commits therefore never touch the same file, and the `pull --rebase` before the push
only puts your commit on top of theirs. Never commit:
- data, images, masks;
- the teacher cache, checkpoints, `outcomes*.npz` (git ignores them);
- a Kaggle or GitHub token.

## 7. If a session stops

Two things are not known here (TODO: the team checks on Kaggle): Kaggle's session time limit,
and whether a stopped version keeps its output.

A training run (E1, E2, E4, every recovery) saves its state every 50 updates under
`checkpoints/<ID>/seed<k>/resume/` and continues from there when started again with the same
settings. If the stopped version kept its output:

1. In a new session, add that output as an input (**Add Input -> Notebook output**).
2. Turn the saved state into a resume hand-off:

   ```bash
   STATE=$(find /kaggle/input -type d -path '*checkpoints/E2/seed1/resume' | head -1)
   [ -n "$STATE" ] || { echo "STOP: no saved state of E2 seed 1 attached"; exit 1; }
   python scripts/handoff.py export --kind resume --exp E2 --seed 1 \
       -o paths.checkpoint_root="${STATE%/E2/seed1/resume}" \
       --out /kaggle/working/handoff/E2_seed1_resume
   ```

3. Run the card again with `/kaggle/working/handoff/E2_seed1_resume` added to `FROM`. The
   training continues from its last saved step.

For E1 and E2, a training step whose adapter is attached is skipped. So if the training
finished but the validation did not, attach the adapter hand-off and run the card again: only
the rest runs.

## 8. GPU hours

Each person has 30 Kaggle GPU hours per week (decision of 2026-10-17). Kaggle shows what is
left. A T4 x2 session counts **once** against the quota (measured 2026-10-19: a 17-minute
T4 x2 session used 16 minutes). Write the GPU hours a session used into its report;
`docs/schedule.md` is corrected from these numbers.

## 9. Questions

Write a dated line in `docs/agent/INBOX.md` (or tell Soheil). The agent reads it at its next
iteration.
