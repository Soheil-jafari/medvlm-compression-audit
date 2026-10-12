# Documentation

## Start here

| Document | What it is | Maintained by |
|---|---|---|
| [FINAL_GOAL.md](FINAL_GOAL.md) | Definition of done: the hypotheses, tables, figures and quality bars the article needs | team (agent ticks boxes with evidence) |
| [agent/STATUS.md](agent/STATUS.md) | Where the project is now, what is next, what waits on the team | agent |
| [agent/INBOX.md](agent/INBOX.md) | Notes and decisions from the team to the agent | team |

## Project

| Document | What it is |
|---|---|
| [project_definition.md](project_definition.md) | The idea, the clinical pruning criterion, evaluation axes |
| [experiment_plan.md](experiment_plan.md) | Experiment ladder E0-E9, ablations A1-A6, budgets, run counts |
| [preregistration.md](preregistration.md) | Items frozen before any compression run |
| [decisions_log.md](decisions_log.md) | Dated, append-only decisions and the open-decisions table |
| [related_work.md](related_work.md) | Prior work, verified references, novelty checks |
| [team.md](team.md) | Owners and review rules |
| [data_policy.md](data_policy.md) | What may never enter the repository or a cloud session |

## Research loop

| Folder / document | What it is |
|---|---|
| [agent/PROTOCOL.md](agent/PROTOCOL.md) | What one iteration of the loop does, and its hard rules |
| [agent/ITERATION_LOG.md](agent/ITERATION_LOG.md) | One entry per iteration |
| [stages/](stages/README.md) | S00-S12: goal, tasks, outputs and exit gate per stage |
| [decisions/](decisions/README.md) | Decision memos the team answers (D01, D02, ...) |
| [runs/](runs/README.md) | Run cards for GPU jobs the team executes, and the queue |
| [results/](results/README.md) | Per-experiment reports written from committed result CSVs |
| [data_cards/](data_cards/TEMPLATE.md), [model_cards/](model_cards/TEMPLATE.md) | Cards for every dataset and model used |

## How work flows

1. The agent runs one iteration at a time on the `research-loop` branch
   (`agent/PROTOCOL.md`): it implements and tests code, writes memos and run cards,
   analyses committed results and drafts the paper.
2. The team answers memos in the inbox, runs the GPU jobs from the run cards on
   their own dataset access, commits the small result CSVs, and merges
   `research-loop` into `main` through pull requests.
3. The loop ends when every box in `FINAL_GOAL.md` is ticked and signed off.
