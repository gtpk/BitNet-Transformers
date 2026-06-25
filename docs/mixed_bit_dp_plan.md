# Mixed-Bit DP Plan (RT-123..125)

Document position: [Index](./index.md) -> after [G5 Baseline Plan](./g5_baseline_plan.md)
and [Paper Skeleton](./paper_skeleton.md).

Status note after RT-123: this is no longer the primary next path. The sensitivity
scan showed strong layer interaction and a weak additive-DP premise, so this document
is kept as a useful mixed-bit selector/reference. The primary next experiment is now
[Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md).
The old "RT-124 DP selector" and "RT-125 hybrid artifact" labels below are therefore
historical/frozen unless a future result revives mixed-bit allocation.

## Why This Exists

RT-121 and RT-122 changed the research claim:

```text
all-I2_S b1.58 is smallest and fastest, and CE recovery is real,
but it does not beat Q2_K on PPL and 1.1B greedy generation is not usable yet.
```

The next question is not "can we train all-I2_S longer?" only. The better systems
question is:

```text
Can we keep most of the I2_S memory-traffic win while spending a small extra bit
budget on the most sensitive parts of the model?
```

This is a mixed-bit allocation problem. Solving it exactly is combinatorial, but the
model is small enough at the layer/group level that a sensitivity scan plus
multiple-choice knapsack DP is practical.

## Problem Formulation

Start from a baseline artifact:

```text
all target linears = I2_S b1.58
embedding / lm_head / norms = f16
```

For each item `i` and precision choice `b`:

```text
b in {I2_S, Q2_K, Q3_K_M, Q4_0}
cost(i,b) = bytes(i,b) - bytes(i,I2_S)
gain(i,b) = CE(all-I2_S) - CE(item i upgraded to b)
```

Then solve:

```text
maximize    sum_i sum_b gain(i,b) * x(i,b)
subject to  sum_i sum_b cost(i,b) * x(i,b) <= budget
            sum_b x(i,b) = 1 for every item i
            x(i,b) in {0,1}
```

This is a **multiple-choice knapsack** approximation. It assumes additive gains.
Interactions are real, so the DP result is a candidate generator, not the final proof.
Every DP-selected artifact must be re-measured end to end.

## Item Granularity

Do not start at individual tensors. The first scan should use two groups per block:

```text
attn group = q_proj + k_proj + v_proj + o_proj
mlp group  = gate_proj + up_proj + down_proj
```

For a 12-layer Llama-160M:

```text
items = 24 groups
choices = {I2_S, Q2_K, Q3_K_M}
```

This is small enough for scanning and DP, and it keeps the result interpretable:

```text
which layers need attention precision?
which layers need MLP precision?
```

Only after this works should we consider per-tensor granularity.

## Metrics

Use PPL/CE, but do not stop there. RT-122 showed that CE recovery can coexist with
bad greedy generation.

Primary optimization metric:

```text
residual_gap = CE(candidate) - CE(FP)
```

Secondary deployment metrics:

```text
whole MB
target-linear MB
token-gen t/s
```

Generation risk metrics:

```text
loop_rate
salad_rate
repeated_ngram_rate
unique_token_ratio
```

The DP value can start as `gain = delta CE`, but the validation table must report all
of the above.

## RT-123: Sensitivity Scan

Goal:

```text
Measure which groups recover CE most efficiently when upgraded from I2_S to Q2_K/Q3_K_M.
```

Input:

- model: start with `JackFram/llama-160m`
- baseline: adapted all-I2_S from RT-116/RT-121
- eval: same `eval.txt` and same `llama-perplexity` tool as RT-121
- groups: 24 layer groups (`attn`, `mlp`)
- choices: `Q2_K`, `Q3_K_M` first; add `Q4_0` only if needed

For each group/choice:

```text
1. create hybrid GGUF with only that group upgraded
2. run llama-perplexity
3. record CE, PPL, MB
4. compute delta_ce and delta_ce_per_mb
```

Output JSON shape:

```json
{
  "base": {"ppl": 114.14, "ce": 4.737},
  "fp": {"ppl": 43.21, "ce": 3.766},
  "items": [
    {
      "group": "blk.03.mlp",
      "choice": "Q2_K",
      "cost_mb": 0.84,
      "delta_ce": 0.031,
      "delta_ce_per_mb": 0.0369
    }
  ]
}
```

Pass/report rule:

```text
At least a few groups have positive delta_ce_per_mb large enough to justify a hybrid.
If all deltas are noise or negative, mixed-bit is unlikely to help via simple allocation.
```

## RT-124: DP Selector

Goal:

```text
Turn the sensitivity scan into candidate hybrid policies under byte budgets.
```

Use coarse integer MB units:

```text
unit = 0.1 MB for 160M
unit = 1.0 MB for 1.1B
```

DP:

```text
dp[i][c] = best additive gain using first i groups with cost <= c
```

Produce a Pareto set, not one answer:

```text
tiny-fast      : +5% whole MB or less
balanced       : +10% whole MB or less
quality-heavy  : +20% whole MB or less
```

For each policy, output:

```json
{
  "budget_mb": 12.0,
  "predicted_delta_ce": 0.42,
  "selected": [
    {"group": "blk.00.mlp", "choice": "Q3_K_M"},
    {"group": "blk.05.attn", "choice": "Q2_K"}
  ]
}
```

Pass/report rule:

```text
DP must find at least one policy predicted to close a meaningful fraction of
the all-I2_S vs Q2_K gap at lower MB than full Q2_K.
```

## RT-125: Hybrid Artifact Validation

Goal:

```text
Build real hybrid artifacts from DP policies and measure the actual result.
```

Compare:

- FP f16
- full Q2_K
- full Q3_K_M
- all-I2_S OURS
- DP tiny-fast hybrid
- DP balanced hybrid
- DP quality-heavy hybrid

Metrics:

```text
PPL / CE
residual_gap vs FP
whole MB
target MB
llama-bench tg t/s
prompt loop/salad panel
```

Pass criterion:

```text
At least one hybrid has:
  - lower PPL / residual_gap than all-I2_S OURS
  - lower MB and/or faster tg than full Q2_K
  - no new runtime parity issue
```

Strong result:

```text
hybrid approaches Q2_K PPL while staying meaningfully smaller/faster.
```

Fail result:

```text
hybrid does not improve generation or CE enough; pure I2_S remains a systems-only
artifact and usability needs data/objective changes instead.
```

## Why DP Instead Of Greedy

Greedy by `delta_ce_per_mb` is a useful baseline but is too local. DP is still cheap at
group granularity and handles uneven group costs better:

```text
greedy can miss a combination of medium-score small items
DP finds the best additive set under each byte budget
```

However, because group interactions exist:

```text
DP selected policy must be validated by real PPL/prompt runs.
```

If DP prediction and real validation diverge badly, use DP as the initialization for
forward selection:

```text
select DP policy -> remeasure -> add/drop one group at a time with real CE feedback
```

## Implementation Notes

The hardest engineering part is not DP; it is producing hybrid GGUFs where some groups
stay I2_S and other groups use K-quants. The first implementation can be pragmatic:

1. produce all candidate full GGUFs from the same adapted HF source:
   - I2_S
   - Q2_K
   - Q3_K_M
   - Q4_0 if used
2. copy selected tensor byte blocks from the higher-bit GGUF into the I2_S GGUF
3. update tensor type/offset metadata if required
4. run `llama-perplexity` as the validator

If direct GGUF surgery is too brittle, use a Python-side validation first:

```text
replace selected groups with dense higher-precision approximations in PyTorch
measure CE to rank policies
only then build GGUF for the top policies
```

## Relationship To Other Ideas

This plan does not replace:

- activation-aware gamma
- low-rank residual adapters
- sparse outlier correction
- unlikelihood/repetition-aware training

It answers a narrower systems question first:

```text
Can a small amount of higher-bit allocation fix the biggest all-I2_S quality loss?
```

If yes, use DP hybrid as the new systems/quality trade-off. If no, the next branch is
data/objective repair rather than more bit allocation.

## Decision

Proceed in this order:

```text
RT-123 sensitivity scan on 160M
RT-124 DP selector
RT-125 validate 2-3 hybrid artifacts
then decide whether to promote the best policy to TinyLlama-1.1B
```

Do not run G6 seed variance before RT-123 unless the goal is paper hygiene only.
The scientific bottleneck is now usability under a memory budget, not variance of the
old all-I2_S recipe.

## RT-123 RESULT (2026-06-25): sensitivity is interaction-dominated; additive-DP premise is weak

Gate finding: the pinned bitnet.cpp `llama-quantize` has **no `--tensor-type`** override,
so real per-group hybrid GGUFs require byte surgery. Per the plan's fallback, RT-123 ran
a PyTorch per-group **FP-restore** sensitivity scan on the FP/PTQ Llama-160M
(`scripts/rt123_sensitivity_scan.py`, `reports/rt123_sensitivity_160m.json`).

```text
CE_fp = 3.147 (PPL 23)   CE_all-ternary(PTQ) = 11.66 (PPL 115,808)
sensitivity(g) = CE_allT - CE(g restored to FP)   [FP upper bound]
```

| sign | groups |
| --- | --- |
| positive (FP-restore helps) | blk.11.attn +1.07, blk.11.mlp +0.95, blk.10.attn +0.53, blk.01.mlp +0.39, blk.02.mlp +0.32, blk.03.mlp +0.06 (6/24) |
| negative (FP-restore HURTS) | the other 18/24 (e.g. blk.00.attn -1.44, blk.10.mlp -1.43, blk.09.mlp -1.10) |

Honest findings:
1. **Heavy non-additivity.** 18/24 single-group FP-restores make CE *worse* than
   all-ternary: the all-ternary PTQ model is a self-consistent (bad) fixed point, and
   perturbing one layer to FP mismatches the downstream ternary stack. The
   additive-knapsack DP assumption is therefore unreliable on this baseline.
2. **Clean signal is output-proximal.** Precision helps most at the last layers
   (blk.11, blk.10) plus a few early MLPs — interpretable (final-layer error is not
   averaged out by later layers).
3. **Low ceiling.** The best single group recovers ~1 of the ~8.5-nat PTQ->FP gap; even
   summing the positives (optimistic, non-additive) leaves PPL in the thousands.
   **Bit-allocation alone cannot approach FP** on the PTQ model.

Caveat: this is a proxy — FP-restore on the *un-adapted* PTQ model, not the ADAPTED
model with real Q2_K. A faithful per-(group,choice) scan needs the adapted model + real
hybrid GGUFs (surgery).

```text
VERDICT (by the plan's own rule): borderline-negative. A few positive groups exist
(output layers), but 18/24 negatives break the additive premise, so a full RT-124
knapsack DP on these deltas would be DP-on-noise. Bit allocation cannot close the
PTQ->FP gap; usability remains recovery/data-bound (consistent with RT-122).
RECOMMENDATION: do NOT build the full additive DP. Instead test ONE intuition-backed
hybrid — last 1-2 layers higher-bit, rest I2_S — on the ADAPTED model via GGUF surgery
(RT-125), and otherwise treat usability as a recovery/data problem, not a bit-allocation
problem.
```
