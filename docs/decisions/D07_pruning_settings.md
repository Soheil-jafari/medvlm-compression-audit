# D07: Pruning settings, scoring cost and the budget protocol

**Status:** partly answered 2026-10-12 (Soheil): sections 2 and 3 accepted as proposed (score per parameter; first 2 and last 1 layers protected for every arm; group 128, 128 calibration images, one shot, 1 negative per positive, signed effects, probe batch 4). In section 4 the metric (peak GPU memory), the tolerance (3 %) and the protocol (batch 1, 3 warm-up and 5 timed passes, 32 validation images drawn by a fixed seed) are accepted. Open: section 1 (how a unit is switched off while scored; RC-M1 measured the T4 speed on 2026-10-15, so the cost table below is updated, and nothing needs the answer before E2 exists), the timing hardware, and B1 to B3 (set by the rule after RC-B0). **Stage:** S09, S10.
**Depends on:** D02 (data and findings), D05 (student), D06 (recovery recipe).
Every number below is fixed **before** the first pruning result, so none of it can be tuned
to a result. After the team answers, the values go into `configs/pruning.yaml`,
`configs/experiments/E6.yaml`, `E7.yaml` and `configs/budgets.yaml`, and are frozen with the
pre-registration.

## Question

The code for the six criteria (magnitude, activation, Taylor, output similarity, random and
our clinical criterion), the matched-budget search and the shared recovery is finished and
tested on a small stand-in model. Several settings that the code cannot choose for us are
still open. They fall into four groups:

1. how a unit is switched off while it is scored, which decides whether scoring takes hours
   or weeks (section 1),
2. how scores become a removal order (section 2),
3. the size of the unit grid and of the calibration set (section 3),
4. the budgets: what is measured, how close is close enough, and where B1 to B3 lie
   (section 4).

## Options

### 1. How a unit is switched off while it is scored (the clinical criterion)

The clinical criterion measures, for each unit, how much the log-probability of the right
yes/no answer drops on the calibration probes when the unit is removed, weighted by finding.
Two ways to get that number:

- **Exact:** remove the unit (a hook that gives the same output as deleting its weights; the
  tests check that the two agree), run every probe, record the change. One pass over all
  probes **per unit**.
- **First-order estimate:** give each unit a gate (1 = present, 0 = removed) and read the
  derivative of the objective with respect to every gate from **one** backward pass. This
  is the Taylor estimate of removing the unit, the idea behind the gradient criterion of structured
  pruning [2]. It is cheap, but it approximates the removal
  by a straight line, which is good for a small unit (the tests show a correlation of about
  0.9997 with exact removal for channel groups on the stand-in model) and poor for a whole
  layer.

Grid size depends on the student (hub configs read 2026-10-12):

| Student candidate | Layers | KV groups per layer | MLP width | Units with 128-channel groups |
|---|---|---|---|---|
| Qwen2.5-VL-3B | 36 | 2 | 11 008 | 36 + 72 + 3 096 = **3 204** |
| Qwen3-VL-2B | 28 | 8 | 6 144 | 28 + 224 + 1 344 = **1 596** |

Cost on one T4 for one seed, with 128 calibration images and about 640 probes. This memo first
**assumed** 0.12 s per probe forward pass. **Measured 2026-10-15 (RC-M1):** the 3B student takes
**0.199 s per probe** (`student_scoring_seconds_per_probe` = 0.1993 in `results/smoke/RC-M1.csv`,
synthetic images at the image side 448). **Measured 2026-10-16 (RC-S0, same side):** the 3B at
0.179 s and the 2B fallback at **0.074 s per probe** (`r0_scoring_seconds_per_probe` in
`results/selection/R0/`). The 3B column keeps RC-M1's larger 0.199 s; the 2B column now uses
0.074 s instead of the assumed 0.12 s. Hours = units x 640 probes x seconds per probe; they
remain estimates, because the real number of probes depends on the frozen finding list.

| Method | Qwen2.5-VL-3B, measured 0.199 s | Qwen2.5-VL-3B, assumed 0.12 s (old) | Qwen3-VL-2B, measured 0.074 s | Qwen3-VL-2B, assumed 0.12 s (old) |
|---|---|---|---|---|
| Exact for every unit | about 114 h | about 68 h | about 21 h | about 34 h |
| Exact for layers and KV groups, first-order for channel groups (**proposed**) | about 3.8 h + minutes | about 2.3 h + minutes | about 3.3 h + minutes | about 5.4 h + minutes |
| First-order for every unit | minutes | minutes | minutes | minutes |

(The first version of this table gave about 33 h for the 2B at 0.12 s; 1,596 x 640 x 0.12 s is
about 34 h. With RC-S0's 0.179 s the 3B would need about 102 h and 3.4 h.)
With the measured speed the proposed mix costs about 3.8 h of scoring per seed of E7, about
11.5 h for its three seeds (A1 and A2 reuse these effects); exact scoring of every unit, about
114 h per seed, is out of reach. For the 2B fallback the proposed mix would cost about 3.3 h per
seed: its 252 KV groups make up for its faster probes.

Exact for every unit does not fit the budget of three Kaggle accounts for three seeds plus
ablations. First-order for every unit would score whole layers with a straight line, which is
the weakest case of the approximation. The proposed mix uses exact removal where the unit is
large (layers, KV groups: few of them) and the estimate where the units are many and small.

Two consequences, both stated openly:

- The estimate is a **linear** function of the per-probe effects, so it cannot clip negative
  effects at zero (`negative_delta: clip0`); with the mix the setting must be `keep`. Likewise
  the KL variant of the effect (ablation A5) has a zero derivative at "unit present" and
  needs exact removal; A5 will therefore run on layers and KV groups only unless the team
  accepts its cost.
- A1 and A2 (uniform and rarity-only finding weights) use the **same** per-probe effects
  as E7 and differ only in the weights, so their effects are computed once and re-weighted.

To check the approximation on the real model, RC-G2 also computes the exact effect for 30
randomly chosen channel groups (about 30 x 640 x 0.199 s = 64 minutes, an estimate from the
measured speed) and reports its rank correlation with the estimate. It is a
report, not a gate; if the correlation is below 0.7 the agent stops and asks the team.

### 2. From scores to a removal order

| Setting | Proposal | Alternative | Reason |
|---|---|---|---|
| Rank by | **score per parameter** (`per_parameter`) for every criterion | the raw score (`none`) | To free a given number of bytes at the smallest total loss, remove the units with the smallest *loss per freed byte* first (the greedy rule for a budgeted removal when effects add up). A raw score ranks by loss alone and ignores how many bytes a unit frees, so it is not the cheapest way to reach a byte target. The scores of magnitude, activation [1], Taylor [2] and clinical are all sums over weights, so the division is meaningful for each. The output-similarity criterion only scores layers, which are all the same size, so for E6 both give the same order |
| Protected layers | the first **2** and last **1** decoder layers are never removed, **for every arm** | only E6 (as in the method it follows); or none | The method E6 follows never prunes blocks 1, 2 and 32 of its 32-block model ("as their removal leads to significant degradation") [3]; layer-removal studies find the outer layers matter most [4]. Applying the same rule to all arms keeps the comparison fair. Heads and channels inside a protected layer can still be removed |
| Schedule | **one shot**: remove everything the budget needs, then recover | iterative (several small steps with recovery between) | One shot is the same for all arms and costs one recovery. Iterative pruning can recover better but multiplies the recovery cost and is not implemented; it can be a later extension. **Deviation to report:** the method E6 follows removes its layers iteratively [3]; E6 here scores once and removes in one shot like every other arm |

### 3. Grid and calibration

| Setting | Proposal | Reason |
|---|---|---|
| MLP channel-group size | **128** | Divides both MLP widths above (86 and 48 groups), a multiple of the tensor-core tile, and keeps the grid at 1 600 to 3 200 units. 64 doubles the exact-scoring cost, 256 is coarse |
| Calibration images | **128** from the training split, stratified so that every finding with positives appears first; nested subsets so that A4 (32 / 64 / 128 / 256) compares like with like. One draw per seed, the same images for every criterion | Middle of the A4 sweep. The data-driven baselines (activation, Taylor, output similarity) use the same images |
| Negatives per positive (clinical probes) | **1** | With all negatives, a rare finding's mean effect is dominated by images without the finding and the clinically important positives are diluted. One negative per positive weights both classes equally within a finding (the draw is seeded) |
| Negative effects | **keep** (signed) | The only option compatible with the estimate in section 1 |
| Probe batch | **4** | The same micro-batch as training; safe on a T4 with gradients |
| Loss scale (fp16) | automatic | The estimate needs loss scaling in fp16; reduced automatically if a gradient is not finite |

### 4. The budgets

The compared quantity is the **measured** footprint after removal, never a parameter count.

| Setting | Proposal | Alternative | Reason |
|---|---|---|---|
| Metric | **peak GPU memory** during inference (weights + activations + KV cache) | warm latency | Memory is deterministic; latency on shared Kaggle GPUs is noisy, which breaks the bisection of the search |
| Tolerance | **3 %** of the target | 2 % or 5 % | One decoder layer of the 3B candidate is about 154 MB, about 2 % of its peak; the search overshoots by at most one unit, so 3 % of a target near 0.8 x peak (about 2.4 % of the peak) covers a single-layer step |
| Protocol | batch 1, free generation with the D04 `max_new_tokens`, 3 warm-up and 5 timed passes, 32 validation images drawn by a fixed seed, the same for every arm | other batch size | A batch of 1 is the deployment case for a single image; the timing samples are fixed so arms are comparable |
| Hardware | development on Kaggle T4, **final numbers on the rented GPU** (type still open); every row records the hardware | - | Numbers from different GPUs are never mixed |
| B1 / B2 / B3 | set by a **rule** after RC-B0 has measured E2: remove about **15 % / 30 % / 45 %** of the decoder's bytes, expressed as fractions of E2's measured peak | the placeholders 0.75 / 0.60 / 0.50 of the peak | The vision encoder, the embeddings and the activations cannot be pruned. For the 3B candidate the decoder body is roughly three quarters of the weights (an **estimate** from the hub config, until RC-B0). Taking a decoder body of about 62 % of the peak as the working assumption, 0.50 of the peak would need about 81 % of the decoder removed, which no method survives; 0.70 needs about 48 % |

## Risks

- **Score per parameter** changes which units go first for every criterion; it must be fixed
  now, not tried both ways after seeing results. A side effect: small channel groups are not
  automatically removed first any more, so layers can appear earlier in the order.
- **Mixed ablation methods** put exact and estimated numbers on one scale in one ranking.
  The estimate is a first-order approximation of the same quantity, so the scales agree to
  the extent the approximation holds; this is why the check on 30 groups is part of RC-G2.
- **Tolerance 3 %** with discrete units: an arm can land below the band when the last step
  is a whole layer; such runs are reported as outside the tolerance and listed.
- **Fractions from a rule** are only as good as the measured decomposition of E2's memory;
  RC-B0 must report the decoder's share before the numbers are frozen.

## Recommendation

Accept the proposals of sections 1 to 4.

## What changes once decided

`configs/pruning.yaml` (group size, calibration size, schedule, normalisation, protected
layers), `configs/experiments/E7.yaml` (negatives per positive, negative effects, ablation
method per unit type, probe batch), `configs/budgets.yaml` (metric, tolerance, protocol; the
fractions after RC-B0), the first-order check in RC-G2, cached per-probe effects for A1 and A2.

## Sources (accessed 2026-10-12)

1. Sun et al., *A Simple and Effective Pruning Approach for Large Language Models* (Wanda),
   arXiv:2306.11695 (the activation criterion).
2. Ma et al., *LLM-Pruner: On the Structural Pruning of Large Language Models*,
   arXiv:2305.11627 (structured pruning with recovery fine-tuning; first-order importance).
3. Khan et al., *Compression Strategies for Efficient Multimodal LLMs in Medical Contexts*,
   arXiv:2507.21976 (layer pruning for a medical multimodal model; E6 follows it).
4. Men et al., *ShortGPT: Layers in Large Language Models are More Redundant Than You Expect*,
   arXiv:2403.03853.
5. Model configs: `huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct` and `Qwen/Qwen3-VL-2B-Instruct`
   (`config.json`, read 2026-10-12).
