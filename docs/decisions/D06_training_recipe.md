# D06: Training recipe, recovery recipe and evaluation numbers

**Status:** answered 2026-10-12 (Soheil): every recommendation accepted (rank 16 / alpha 32 / dropout 0.05; lr 1e-4, warm-up 20, batch 4 x 4, KD weight 0.5, temperature 2; recovery = 25 % of the E2 steps; 10 000 resamples, 15 bins, 1 000 generation rows). **Stage:** S05b, S06, S09.
**Depends on:** D01 (teacher), D02 (data), D05 (selection protocol).
All numbers are fixed **before** any result exists. After the team answers they are
written to `configs/training.yaml`, `configs/recovery.yaml` and `configs/base.yaml` and
frozen with the pre-registration; changing one later needs a logged decision.

## Question

The code for training, distillation, recovery and evaluation is finished, but the numbers
it needs are still `TODO`. Three groups of them need a decision:

1. the LoRA adapter and the optimiser (one recipe for E1, E2, E4 and, with the same
   adapter and optimiser, for the recovery after pruning),
2. how long each run trains (E1/E2) and how long the recovery after pruning lasts,
3. the numbers of the evaluation (bootstrap resamples, calibration bins, free-generation
   rows).

The student-selection rounds (D05) train with the same adapter and optimiser, so the first
group has to be fixed before Round 1.

## Why one recipe

The study compares *what compression forgets*. If the arms were trained differently, the
differences could come from the training instead of the compression. So E1 and E2 share
every number except the training signal (answers only vs answers + teacher), and every
pruning arm shares the recovery recipe exactly (`recovery_fingerprint` is written with each
run so a reviewer can check it).

## Options

Each block lists the proposal and its alternatives.

### 1. Adapter and optimiser

| Setting | Proposal | Reason |
|---|---|---|
| Where | LoRA [1] on all decoder projections (attention q/k/v/o and MLP gate/up/down); vision encoder and projector frozen | Already fixed in the selection config. Adapters on all linear layers were needed to match full fine-tuning in the QLoRA study [2] |
| Rank / alpha | **16 / 32** | The feasibility step of Round 0 already uses it; large enough for a 2-4B decoder, small enough for a T4 |
| Dropout | **0.05** | Common LoRA default; regularises a short run on a few thousand examples |
| Precision | fp16 weights, fp32 adapter weights, loss scaling | T4 has no bf16 (implemented and tested) |
| Optimiser | AdamW, learning rate **1e-4**, weight decay 0 | The value of the Round 0 step; LoRA fine-tuning usually runs at learning rates of the order 1e-4 [2] |
| Schedule | linear decay to 0 with **20** warm-up steps | Supported by the trainer; avoids a large first step |
| Batch | **4** examples per micro-batch, **4** accumulation steps (16 per update) | Fits a T4 at 448 px with fp16; the update size stays the same if the micro-batch shrinks |
| Clipping | gradient norm 1.0 | Trainer default |
| Distillation (E2 and recovery) | loss = cross-entropy on the ground-truth answer + **0.5** x KL to the cached teacher top-20 log-probabilities, **temperature 2** | Soft targets from the teacher are the point of E2 [4]; a weight of 0.5 keeps the ground-truth answer as an equal anchor. A temperature above 1 softens the teacher distribution so the cached tokens below the top one carry signal [4] |
| E1 (answers only) | the same, with the KD weight 0 | The only difference between E1 and E2 |

Alternatives: rank 8 / alpha 16 (cheaper, a little less capacity), rank 64 / alpha 128
(more capacity, more memory and a higher risk of over-fitting a small subset).

### 2. How long to train

The absolute size of a run depends on the speed of one T4, which RC-S0 measures. So the
memo fixes **rules**, and the numbers follow from the throughput:

- **E1 and E2:** `data_subset` and `steps` are those of Round 3 of the selection (the exact
  E2 recipe, `compute_units` 1.7 x b examples); `steps = ceil(data_subset / 16)`, one pass
  over the subset.
- **Recovery after pruning:** **25 %** of the E2 steps with the same learning rate, adapter
  and loss; the examples are drawn from the training split by the seed alone, so every arm
  sees the same ones.
  *Note (iteration 36, agent proposal awaiting Soheil):* the code now draws one training pool
  with the data seed (`seeds.data_split`) instead of the run seed: E1, E2, E4 and every
  recovery use it (recovery = its first 25 %), and the teacher cache holds only this pool.
  The run seed changes the order and the adapter's start. See the open-decisions table.
- Options for the recovery share: 10 % (cheaper, recovers less after a deep cut), 50 %
  (recovers more, doubles the cost of the 45 pruning runs).

Cost check: 45 main pruning runs (5 arms x 3 budgets x 3 seeds; E5 has three criteria) plus
about 33 ablation runs make about 80 recoveries; at 25 % of an E2 run this is about 20
E2-equivalents of training, on top of 6 for E1/E2.
The exact hours are filled in from RC-S0 (**estimate until then**).

### 3. Evaluation numbers

| Setting | Proposal | Reason |
|---|---|---|
| `bootstrap_resamples` | **10 000** | Paired image-level bootstrap is cheap (arrays of a few thousand probes); 10 000 resamples keep the Holm-adjusted tail p-values stable down to about 0.0005 |
| `calibration_bins` | **15** equal-width | The ECE of [3] with 15 equal-width bins; the code default |
| `generation_samples` | **1 000** QA rows of the validation split, drawn by seed, the same rows for every arm | Free generation of the whole split costs about an hour per arm; 1 000 rows give word-level metrics with narrow intervals |
| `max_new_tokens` | `ceil(1.5 x p99 answer words) + 8` (D04), computed when RC-D1 reports the answer lengths | unchanged |

## Risks

- A short run may under-train a candidate that needs a longer schedule. Mitigation: all
  candidates get the same budget, and the selection ranks them on the same footing.
- Temperature 2 is a choice among 1, 2 and 4; changing it later would be result-driven.
  It is fixed here, before E2 exists.
- 25 % recovery may leave the deepest budget (B3) clearly below the teacher. That is a
  finding about compression, not a reason to change the recipe after the fact.

## Recommendation

Accept the table in section 1, the rules in section 2 (25 % recovery) and the numbers in
section 3.

## What changes once decided

`configs/training.yaml` (adapter, optimiser, KD), `configs/recovery.yaml` (the same adapter
and optimiser, steps = 25 % of E2, KD), `configs/base.yaml` (`bootstrap_resamples`,
`calibration_bins`, `generation_samples`); `steps` and `data_subset` are filled in at the
selection freeze from the RC-S0 throughput.

## Sources (accessed 2026-10-12)

1. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, arXiv:2106.09685.
2. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314
   (adapters on all linear layers of the transformer blocks).
3. Guo et al., *On Calibration of Modern Neural Networks*, arXiv:1706.04599.
4. Hinton et al., *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531.
