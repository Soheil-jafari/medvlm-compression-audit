# What Compression Forgets

**Pruning vs. Quantization vs. Distillation in Medical Vision-Language Models, and a
Clinically Weighted Pruning Criterion for Preserving Small Findings** (working title)

We take one generative medical vision-language model (image + question -> answer)
and compress it in several ways (post-training quantization, quantization-aware
distillation, generic structured pruning, output-similarity layer pruning, and our
clinically weighted pruning criterion) to the same **measured** memory or latency
budget. We then ask which compression method silently damages small-lesion recall,
rare-finding recall, image reliance, calibration and hallucination behaviour, even
when average accuracy holds. Our one new method is a structured-pruning criterion
that scores units by how much fixed yes/no finding probes depend on them, weighted
by capped inverse prevalence and clinical importance.

**Positioning** ([`docs/related_work.md`](docs/related_work.md)). The closest prior work weights
importance by patient group (FairPrune), by class (class-aware pruning), by the long tail
(LTAP) or by domain (D-Pruner, unstructured), or changes the calibration data for a domain
(MixCal, unstructured / 2:4); the closest medical study (Khan et al. 2025) removes layers of
LLaVA by output similarity and adds AWQ, scored by an LLM judge on 39 test images with no
per-finding or small-lesion analysis. What is new here: structured pruning of a medical VLM;
importance = the drop in the correct answer per clinical finding, weighted by a pre-registered
clinical-priority rule; evaluation on small-lesion recall and lesion-mask reliance; and a
matched-memory comparison of pruning, quantization and distillation. A1 and A2 are the
in-framework analogues of task-aware and long-tail-aware (LTAP-style) weighting. Our generic
arms are re-implementations in one framework with identical recovery (magnitude, Taylor in the
LLM-Pruner style, activation, and E6 in the ShortGPT / Khan et al. style), never the original
methods.

> **Status: scaffolding only. No methods are implemented.** Every algorithm is a
> documented stub that raises `NotImplementedError`. Only infrastructure (config
> loading, experiment registry, CLI parsing, small utilities) runs. No data, model
> weights or results are in this repository.
>
> **Progress:** [`docs/agent/STATUS.md`](docs/agent/STATUS.md). **Definition of
> done:** [`docs/FINAL_GOAL.md`](docs/FINAL_GOAL.md). Work is planned in stages
> S00-S12 ([`docs/stages/`](docs/stages/README.md)) and advanced in small
> iterations on the `research-loop` branch ([`docs/agent/PROTOCOL.md`](docs/agent/PROTOCOL.md)).

## Decided vs. undecided

| Topic | Status | Where |
|---|---|---|
| Study design: one teacher/student pair, all arms matched on measured budget | Decided | `docs/project_definition.md` |
| Experiment ladder E0-E9 and ablations A1-A6 | Decided | `src/medcompress/registry.py`, `docs/experiment_plan.md` |
| Clinical pruning criterion (formula, procedure) | Decided | `src/medcompress/compression/pruning/criteria/clinical.py` |
| Shared recovery for all pruning arms (LoRA + offline KD) | Decided | `configs/recovery.yaml` |
| Evaluation axes and statistics plan | Decided | `docs/experiment_plan.md`, `docs/preregistration.md` |
| Primary endpoints: small-finding recall and lesion-mask effect at B2, E7 vs best of E5/E6 | Decided | `docs/preregistration.md` |
| Seed plan: 3 seeds for the main-claim tier (E2, E7 and the comparator at B2), seed 0 for every other arm; futility-only stopping rule (amendment of 2026-11-08, Soheil and Amir) | Decided | `docs/preregistration.md` section 11 item 12, `src/medcompress/seed_plan.py` |
| Medical domain: GI endoscopy | Decided (D02) | `configs/domain/` |
| Teacher Qwen2.5-VL-7B with our Kvasir LoRA; student Qwen2.5-VL-3B (shared tokenizer) | Decided (D01, D05) | `configs/models/` |
| Data: Kvasir-VQA-x1 re-split by image; external co-primary test PolypGen | Decided (D02 amendments) | `configs/data/`, `docs/runs/RC-X2_*.md` |
| Finding list (polyp, ulcerative colitis, oesophagitis) and finding weights (cap 20; tiers action 2.0 / follow-up 1.5 / other 1.0, by rule from the names) | Decided (2026-10-31, 2026-11-08) | `configs/preregistration/finding_weights.yaml`, pre-registration section 3 |
| Budgets B1 / B2 / B3: 15 / 30 / 45 % of E2's decoder bytes removed, matched on the Kaggle T4 | Decided (2026-11-08) | `configs/budgets.yaml` |
| Scoring method, recovery (157 updates on 2,500 examples, identical for every pruning arm: the fairness rule), A3 ranking, reuse of E7 | Decided (2026-11-08) | pre-registration section 7 |
| How the two co-primary tests combine: supported only if supported on both (intersection-union), partly supported on one | Decided (2026-11-08) | pre-registration section 6 |
| Quantization backends (GPTQ w4 / w8, LLM.int8) | Decided (D08) | `configs/quantization.yaml` |
| Calibration images not finite in fp16 are replaced by the next eligible image of the same seeded order (nested, stratified; committed list) | Decided (amendment of 2026-11-13, Soheil and Amir) | pre-registration section 10, `results/calibration/` |
| Timing GPU for the final numbers (RC-T1) | **Open** (does not block the freeze) | `configs/base.yaml` |
| Venue | **Open** | `docs/decisions_log.md` |

All open items are listed with dates in `docs/decisions_log.md`.

## Repository map

```
configs/                 YAML configs (plain YAML + _base_ inheritance)
  base.yaml              paths, seeds, logging, hardware, evaluation defaults
  budgets.yaml           B1-B3 placeholders and how budgets are measured
  training.yaml          shared student-training defaults (E1, E2, E4)
  pruning.yaml           shared pruning settings (E5-E7, A1-A6)
  recovery.yaml          shared recovery recipe, identical for all pruning arms
  domain/                endoscopy / retinal: finding list, prevalence groups, probe template
  preregistration/       pre-registered finding weight table
  models/                candidate teacher / student configs (CANDIDATE - NOT DECIDED)
  data/                  candidate dataset configs (CANDIDATE - NOT DECIDED)
  experiments/           E0-E9 and ablations/A1-A6
src/medcompress/         Python package (src layout)
  config.py registry.py budgets.py cli.py
  data/ models/ compression/ evaluation/ utils/
scripts/                 thin entry points 00-08 and run_experiment.py
notebooks/               Kaggle / Colab template
results/                 results CSV schema (only small summaries are tracked)
docs/                    goal, status, stages, run cards, memos, plan, preregistration, policies
paper/                   LaTeX outline (sections, placeholder tables, references.bib)
tests/                   import, config, registry, script and utility tests
```

## Experiments

| ID | Description | Starts from | Budgets | Seeds |
|---|---|---|---|---|
| E0 | Teacher, full precision | pretrained | - | 1 |
| E1 | Student fine-tuned on correct answers only, full precision | pretrained | - | 3 |
| E2 | Student + offline distillation, full precision (start of all compression arms) | pretrained (+E0 cache) | - | 3 |
| E3 | E2 + post-training quantization (GPTQ `w8` / `w4`; LLM.int8 `w8a8`) | E2 | measured | `w8`: 3 (H2); `w4`, `w8a8`: 1 |
| E4 | Student trained with quantization-aware distillation (GRACE-style) | pretrained (+E0 cache) | measured | 1 (seed 0) |
| E5 | E2 + generic structured pruning (magnitude, activation/Wanda-style, Taylor) + recovery | E2 | B1-B3 | 1; 3 for the comparator at B2 |
| E6 | E2 + output-similarity layer pruning (Khan et al. style) + recovery | E2 | B1-B3 | 1; 3 if it is the comparator (B2) |
| E7 | E2 + clinical pruning criterion (ours) + recovery | E2 | B1-B3 | 3 at B2; 1 at B1, B3 |
| E8 | E7 + 4-bit post-training quantization | E7 | B1-B3 | 1 (seed 0) |
| E9 | Teacher quantized to 4-bit (supplementary, matched memory) | E0 | measured | 1 |
| E10 | Native Qwen3-VL-2B (the D05 fallback), E2 recipe (E1 if its tokenizer does not match the cache); exploratory, at matched memory and at 4 bit | pretrained (+E0 cache) | measured | 1 (seed 0) |
| A1 | Unweighted criterion (w_c = 1); also the pre-specified secondary comparison E7 vs A1 | E2 | B2 | 3 (secondary) |
| E11 | *Exploratory, not built:* clinical mixed-precision quantization (per-layer 4 / 8 bit by the clinically weighted sensitivity), only if per-layer bit widths run with real kernels | E2 | measured | 1 (if feasible) |
| E12 | *Exploratory, not built:* its generic control (unweighted sensitivity) | E2 | measured | 1 (if feasible) |
| A2 | Rarity-only weights | E2 | B2 | 1 (seed 0) |
| A3 | Random removal | E2 | B2 | 1 (seed 0) |
| A4 | Calibration set size 32 / 64 / 128 / 256 | E2 | B2 | 1 (seed 0) |
| A5 | Delta = correct-answer drop vs prediction change (KL) | E2 | B2 | 1 (seed 0) |
| A6 | Layers only vs heads + channel groups | E2 | B2 | 1 (seed 0) |

"measured" = not budget-matched; the footprint is measured and placed relative to B1-B3.
Run `python scripts/run_experiment.py --list` for the live registry.

### Seeds (pre-registration section 11 item 12, 2026-11-08)

- **Three seeds (0, 1, 2) only where the main claim is made:** E2, E7 at B2, the comparator
  at B2, and E3 `w8` with its matched pruning arm for H2 (H2 stays confirmatory; decided
  2026-11-08). H1-H3 (E7 against E2 and against the comparator, the primary endpoints) are
  confirmatory on these runs only. E1 keeps the 3 seeds it already ran. **A1** (uniform
  weights) at B2 also has 3 seeds, for the pre-specified **secondary** comparison E7 against A1
  (no verdict); it re-weights E7's per-finding effects, so its seeds 1 and 2 cost about 6 GPU h.
- **Seed 0 only for everything else**, reported as exploratory: the other E5 / E6 arms, E3
  `w4` and `w8a8`, E4, E8, E9, E10, every arm at B1 and B3, the other ablations and the
  training-length check.
- **Verdicts on the two co-primary tests:** a hypothesis is supported only if it is supported
  on both the internal filtered test and the external test (PolypGen), partly supported if on
  one; each test is reported either way.
  H4 (the ranking of E3-E8) is exploratory too. A single-seed difference smaller than E7's
  spread over its 3 seeds at B2 is reported as within run-to-run noise.
- **Why:** on E1's 3 seeds the seed-to-seed spread of the small-polyp recall and of the AUROC
  is about what the 13 validation polyps alone produce, so more seeds buy little; more small
  polyps (the external test) buy more. **Never 2 seeds:** the majority answer of the McNemar
  tests counts a 1-1 split as wrong.
- **The comparator** is picked on validation, seed 0, at B2, among E5:magnitude,
  E5:activation, E5:taylor and E6: highest `op_small_finding_recall`; ties by the recall over
  all mask-positive polyps; then by the small polyps' mean margin above the arm's own polyp
  threshold; then by name (`scripts/choose_comparator.py`). Only that arm runs seeds 1 and 2.
- **Stopping rule (futility only):** after seed 0, seeds 1 and 2 of the main tier run unless
  E7 at B2 ranks below the comparator by the same ordering. Then the study stops and reports
  seed 0 as exploratory. Stopping only for futility does not raise the false-positive rate.
- **GPU hours** (estimates, with E2's measured 27.2 s per distillation update): about 234-241 h
  for the whole remaining plan, plus 5.4 h for E3 `w8`'s seeds 1 and 2 (about 239-246 h), of
  which about 49 h are seeds 1 and 2 of the 3-seed tier, and about 8 h E10; at 90 h per week
  (three Kaggle accounts) about 2.7 weeks of GPU time (`docs/schedule.md` section 7), against about 360 h or
  more with 3 seeds for every arm at B2. (A1 and A2 no longer score on their own: they
  re-weight E7's effects, which saves about 16 h of scoring.)

## Quickstart

### Kaggle / Colab

Open `notebooks/kaggle_template.ipynb`. It clones the repo, installs it, sets paths
through environment variables and runs a script. Attach your **own** dataset access;
see `docs/data_policy.md`.

### Local (light dependencies, no GPU)

```bash
git clone https://github.com/Soheil-jafari/medvlm-compression-audit.git
cd medvlm-compression-audit
python -m venv .venv && source .venv/bin/activate
make install-dev          # base + dev requirements, editable install, pre-commit hooks
make lint test
python scripts/run_experiment.py --list
python scripts/run_experiment.py --exp E7 --budget B2 --seed 0 --plan-only
```

GPU sessions additionally run `make install-train`. Paths are set with
`MEDCOMPRESS_DATA_ROOT`, `MEDCOMPRESS_CACHE_ROOT`, `MEDCOMPRESS_CHECKPOINT_ROOT`,
`MEDCOMPRESS_OUTPUT_ROOT` and `MEDCOMPRESS_RESULTS_ROOT`, or per run with
`-o paths.data_root=/path`.

## Team

Authors: **Soheil** (lead; teacher/student setup, distillation, quantization, recovery, data,
evaluation and statistics) and **Amir** (pruning, the clinical criterion, efficiency). Farzan
left the project on 2026-11-08 and lends one Kaggle account. The pre-registration freeze needs
both authors' approval. See `docs/team.md` for review rules.

## Policies

- Data: [docs/data_policy.md](docs/data_policy.md). No data, images, checkpoints or
  secrets are ever committed.
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).
- Licence: code under the [MIT License](LICENSE) (decided 2026-10-12, memo D03).
  Datasets and model weights keep their own licences and are never in this
  repository.
