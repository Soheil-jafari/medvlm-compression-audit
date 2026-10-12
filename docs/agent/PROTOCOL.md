# Research loop protocol

The research agent advances the project in short, repeated **iterations**. Each
iteration does one useful, verified piece of work, records it, and stops. The
team stays in charge: the agent proposes, implements and analyses; the team
decides, runs GPU jobs on its own dataset access, and merges.

Target: `docs/FINAL_GOAL.md`. Live state: `docs/agent/STATUS.md`.

**Run mode (since 2026-10-12):** iterations run back to back in an interactive
session started by the team, until the team stops it. There is no scheduled run.
When an iteration needs a team decision and a team member is present, the agent
asks in the session and records the answer in the inbox; otherwise it moves on to
the next doable item.

## Branches

- The agent works only on the branch **`research-loop`**. It never pushes to
  `main`, never force-pushes, never opens pull requests and never posts comments.
  The team reviews `research-loop` and opens pull requests into `main` itself.
- Team members commit results, run cards and inbox notes to `main` (via a pull
  request) or directly to `research-loop`. The agent merges `origin/main` into
  `research-loop` at the start of every iteration.
- If the session created any branch other than `research-loop`, the agent does not
  push it, and deletes it from `origin` if it was pushed.

## One iteration

### 1. Sync

```bash
git fetch origin
git checkout research-loop            # create from origin/main if it does not exist
git pull --ff-only origin research-loop
git merge --no-edit origin/main       # stop and log a blocker on conflicts
pip install -r requirements/base.txt -r requirements/dev.txt && pip install -e . --no-deps
```

### 2. Orient (read, in this order)

1. `docs/agent/STATUS.md`: current stage, next action, blockers.
2. `docs/agent/INBOX.md`: new notes from the team. These override STATUS.
3. The last three entries of `docs/agent/ITERATION_LOG.md`.
4. The current stage file in `docs/stages/`.
5. `git log origin/main --since="<date of last iteration>"`: what the team changed.
6. New or changed files under `results/` and `docs/runs/`.

### 3. Choose exactly one work item

Take the first that applies:

1. **CI is red** on `research-loop` or `main`: fix it.
2. **New results arrived** (a run card marked `done` with CSVs under `results/`):
   validate them against the schema in `results/README.md`, write or update
   `docs/results/<ID>.md`, tick the run in `docs/runs/README.md`, check the stage
   exit gate.
3. **Inbox item** that the agent can act on.
4. **Next agent task in the current stage** (unticked `[agent]` box in its stage
   file).
5. **Work ahead:** if the current stage waits on the team, the next unticked
   `[agent]` task in a later stage whose inputs already exist (for example CPU
   infrastructure, run cards, paper method and related-work sections).
6. **Nothing doable:** do a quality pass (docs consistency, test gaps, references
   check) or, at most once a month, a novelty re-check. Log the iteration as idle
   if even that is not useful.

Keep the item small: one module with its tests, one run card, one report, one
paper section. If it is bigger, split it in the stage file and do the first part.

### 4. Do the work

- Code follows `CONTRIBUTING.md`: no heavy imports at module top level, docstrings
  and return annotations on public functions, `ruff` clean.
- Every implemented function gets unit tests on small **synthetic** inputs. When a
  stub listed in `test_stubs_raise_not_implemented_with_todo` is implemented,
  replace its entry with real tests.
- Code that needs a GPU, model weights or real data is finished up to a **run card**
  (`docs/runs/TEMPLATE.md`) that a team member can execute on Kaggle or Colab. The
  agent never claims a GPU result it did not receive as a committed CSV.
- Analyses read only committed CSVs under `results/`. Every number the agent
  writes into a report cites the CSV path it came from.

### 5. Verify

```bash
ruff check . && ruff format --check . && pytest
```

All three must pass before committing. For documents: links resolve, IDs match
`src/medcompress/registry.py`, nothing contradicts `docs/preregistration.md`.

### 6. Record

- `docs/agent/STATUS.md`: stage table, current stage, next action, the
  "waiting on the team" queue and the date.
- `docs/agent/ITERATION_LOG.md`: append one entry (format at the top of that file).
- The stage file: tick finished boxes; add new boxes if work was split.
- `docs/decisions_log.md`: only to record decisions the **team** made (from the
  inbox), with the date. The agent may add proposals to the open-decisions table.

### 7. Commit and push

- Use the git identity given in the task instructions. Plain commit messages
  (subject line plus a short body); no trailers of any kind.
- Run the attribution check from the task instructions over all tracked files and
  over the new commit messages. Nothing may mention any AI tool, assistant, model
  vendor or session link.
- `git push origin research-loop`, then check CI with
  `gh run list --branch research-loop --limit 1`.

## Hard rules

1. **No data in git or in this session.** No datasets, images, masks, patient
   identifiers, checkpoints, caches or secrets. Never download restricted data into
   a cloud session. Synthetic fixtures only.
2. **No invented facts.** No guessed hub IDs, URLs, licence terms, dataset sizes or
   citation details: verify from the primary source or leave `TODO`. No estimated
   or hand-typed results.
3. **Team decisions stay with the team:** domain, model pair, datasets, finding list,
   finding weights, budgets, primary-endpoint rules, venue, licence, pivots. The
   agent writes decision memos with options and evidence (`docs/decisions/`) and
   waits for the answer in the inbox.
4. **Pre-registration is binding.** After the freeze the agent never edits a
   pre-registered item. Before the freeze it never looks at test-split results
   (there should be none).
5. **No result-driven tuning.** Never change weights, thresholds, budgets or the
   criterion after seeing test results. If H3 fails, follow shape B in
   `docs/FINAL_GOAL.md`.
6. **Small and reversible.** One work item per iteration; no force-push; no history
   rewriting; no deleting team files.
7. **Stop when done or blocked.** If `docs/FINAL_GOAL.md` is complete, set STATUS to
   `DONE`. If everything is blocked on the team, say exactly what is needed in the
   "waiting on the team" queue and end the iteration.

## How the team talks to the agent

Add a dated line to `docs/agent/INBOX.md` (decisions, answers, priorities, "run card
X done"). The agent moves handled items to the "Handled" section with the iteration
number.
