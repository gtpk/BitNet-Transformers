# Weight-Only Sync Plan

Document position: [Index](./index.md) -> after factual-recovery / quantization-aware
branches, before launching more data-heavy adaptation.

Status: proposal / implementation plan. No experiment has been run yet.

## Question

The current FACT line exposed a painful pattern:

```text
Small non-representative post-training data can become an adaptation trap.
Hard factual replay with 291 facts is learned, but it does not transfer.
```

So this document asks:

```text
Can we make the FP model and the b1.58 model agree better using weights only,
before using any representative PTQ/QAT/adaptation data?
```

This is not a replacement for PopQA blend or later adaptation. It is a possible
**initialization / preprocessing** stage:

```text
FP checkpoint
  -> weight-only sync transform
  -> b1.58 / I2_S materialization
  -> evaluate
  -> if needed, run PopQA/instruction blend from a better starting point
```

## Core Math

For one linear layer:

```text
y_fp = W x
y_q  = Wq x
error = E_x ||(W - Wq)x||^2
      = Tr((W - Wq) Sigma_x (W - Wq)^T)
```

where:

```text
Sigma_x = E[x x^T]
```

Representative calibration data estimates `Sigma_x`. Without data, we do not know it.
The simplest weight-only assumption is:

```text
Sigma_x ~= I
```

Then:

```text
error ~= ||W - Wq||_F^2
```

This is the honest limitation of weight-only sync:

```text
It can make weights geometrically closer.
It cannot know which activation directions matter unless we add assumptions.
```

The practical goal is therefore narrower:

```text
Reduce obvious coordinate/pathology mismatch before any data-based adaptation.
```

## Why This May Still Help

b1.58 ternary projection is coordinate-dependent:

```text
Q(W) != R Q(R^T W)
```

If weight information is concentrated in a few awkward coordinates, a rotation,
equalization, or scale-balancing transform can make the same FP function easier to
approximate with ternary weights.

This is the shared intuition behind:

- data-free equalization / bias correction
- QuaRot / SpinQuant style rotations
- QuIP / QuIP# incoherence processing
- channel-wise scale balancing
- low-rank or multi-plane residuals

Sources to anchor this branch:

- Data-Free Quantization Through Weight Equalization and Bias Correction:
  <https://arxiv.org/abs/1906.04721>
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs:
  <https://arxiv.org/abs/2404.00456>
- SpinQuant:
  <https://arxiv.org/abs/2405.16406>
- QuIP#:
  <https://arxiv.org/abs/2402.04396>

## Transformer-Specific Constraint

Do not assume rotations can be inserted anywhere.

Dense orthogonal rotations preserve ordinary matrix products, but LLaMA-style blocks
also have:

```text
RMSNorm with learned per-channel gamma
residual additions
SwiGLU / elementwise nonlinearities
RoPE attention structure
lm_head / tied or untied output basis
```

Consequences:

| transform | safe without retraining? | note |
| --- | --- | --- |
| sign flip | mostly safe if consistently folded | cheap and deployment-friendly |
| permutation | mostly safe if gamma / connected weights are permuted consistently | tensor-name bookkeeping required |
| diagonal scale | possible but interacts with RMSNorm/gamma and SiLU | equalization candidate |
| Hadamard / block rotation | useful as diagnostic; exact folding across RMSNorm/gamma is tricky | may need runtime op or approximate fold |
| dense learned rotation | upper bound only | likely violates memory-traffic goal |

Therefore the first implementation must distinguish:

```text
deployable transforms:
  sign / permutation / diagonal equalization

diagnostic transforms:
  Hadamard / block orthogonal / learned rotations
```

## Candidate Families

### WSYNC-001: Weight-Only Baselines

Purpose:

```text
Establish the no-data floor using existing conversion code.
```

Arms:

```text
FP
per-tensor b1.58
row-scale b1.58
group/block-scale b1.58
```

Metrics:

```text
weight MSE
row-wise MSE
PyTorch CE/PPL on existing fixed eval text
FACT panel
I2_S export parity if materialized
```

This repeats known baselines in a single WSYNC table, so later gains are comparable.

### WSYNC-002: Data-Free Diagonal Equalization

Idea:

For adjacent linear maps:

```text
W2 W1 x = (W2 S^-1)(S W1) x
```

Choose diagonal `S` to reduce channel range imbalance before ternarization.

Candidate objective:

```text
min_S  ||S W1 - Q(S W1)||_F^2
     + ||W2 S^-1 - Q(W2 S^-1)||_F^2
```

Practical simplified objective:

```text
balance per-channel absmax / absmean between producer and consumer weights
clip S to a safe range, e.g. [1/4, 4]
```

Transformer caution:

```text
Do this only across foldable boundaries first.
Avoid crossing RMSNorm/SwiGLU/RoPE until a local equivalence proof is written.
```

First safe targets:

```text
MLP internal gate/up/down statistics as diagnostic
attention output/input row-column balancing as diagnostic
```

Pass signal:

```text
CE improves by >= 0.2 nats vs same scale baseline
or FACT panel improves without adaptation
```

### WSYNC-003: Sign / Permutation Incoherence

Idea:

A signed permutation is the cheapest rotation family:

```text
R = P D,  D_i in {-1, +1}
```

It is cheap because it is just reindexing and sign flips. It can be folded into
weights and sometimes into normalization parameters.

Search options:

```text
random signed permutations
sort channels by norm and interleave large/small
pair high-error channels with low-error channels
```

Why test:

```text
If even signed/permutation incoherence helps, there is a deployable path.
If only dense rotations help, it is probably a diagnostic insight, not product-ready.
```

Pass signal:

```text
best of N cheap random/sign/permutation trials beats row-scale baseline by >= 0.2 nats
```

### WSYNC-004: Hadamard / Block Rotation Diagnostic

Idea:

Apply fixed block Hadamard rotations to make weights more incoherent:

```text
W_rot = W H
Wq_rot = Q(W_rot)
```

Then either:

```text
diagnostic only:
  compare reconstruction / CE under a local folded approximation

or runtime candidate:
  insert cheap Hadamard in activation path and fold inverse into neighbors
```

This is related to QuaRot / QuIP#.

Pass signal:

```text
Hadamard improves CE by >= 0.5 nats vs row-scale b1.58.
```

Fail signal:

```text
Hadamard improves reconstruction but not CE/FACT.
```

That would mean the no-data geometry objective is not aligned with model behavior.

### WSYNC-005: RMSNorm / Scale Correction

Idea:

Quantization changes row norms and therefore layer output scale. Without activation
data, approximate the output variance shift as:

```text
Var[y_i] ~= ||W_i||_2^2 / d
Var[yq_i] ~= ||Wq_i||_2^2 / d
```

Correct row scale:

```text
c_i = ||W_i||_2 / (||Wq_i||_2 + eps)
Wq_i <- c_i Wq_i
```

This is not pure I2_S if `c_i` is per-row. But it can be treated as:

```text
diagnostic upper bound
or hybrid row-scale format candidate
```

Pass signal:

```text
large CE improvement from row-norm correction
```

Interpretation:

```text
I2_S per-tensor scale is too rigid; row-scale/hybrid storage may be the necessary compromise.
```

### WSYNC-006: Multi-Plane / Residual Upper Bound

If pure b1.58 cannot match the weight geometry:

```text
W ~= alpha_1 T_1 + alpha_2 T_2
```

or:

```text
W ~= alpha T + U V^T
```

This is not pure I2_S, but it answers a useful question:

```text
How much additional capacity is needed before data adaptation becomes easy?
```

This should remain an upper-bound / hybrid-candidate experiment, not the first
deployable path.

## Experiment Ladder

### RT-WSYNC-001: 160M Weight-Only Table

Model:

```text
Felladrin/Llama-160M-Chat-v1
```

Run:

```text
FP
per-tensor b1.58
row-scale b1.58
block/group-scale b1.58
row-norm corrected b1.58
```

Metrics:

```text
CE/PPL
FACT panel
weight MSE
row norm ratio
target storage estimate
```

Pass:

```text
any weight-only transform improves CE by >=0.5 nats or FACT by >=0.05
```

### RT-WSYNC-002: Equalization Screen

Run:

```text
diagonal equalization on safe/foldable boundaries only
clip range sweep: [1/2,2], [1/4,4], [1/8,8]
```

Pass:

```text
consistent improvement over row-scale baseline without CE/FACT regression.
```

### RT-WSYNC-003: Cheap Rotation Screen

Run:

```text
signed permutation random trials N=16
norm-sort/interleave permutation
pairwise sign/swap phases
block Hadamard diagnostic
```

Pass:

```text
deployable cheap transform gives measurable CE/FACT improvement.
```

### RT-WSYNC-004: Combine With PopQA Blend

Only if RT-WSYNC-001..003 show a positive initialization.

Run:

```text
best WSYNC init
  -> b1.58 materialize
  -> FACT-003H PopQA blend
```

Question:

```text
Does weight-only sync reduce the amount of data needed for factual recovery?
```

Pass:

```text
same PopQA blend budget beats non-WSYNC initialization by >=0.05 FACT
or reaches same FACT with fewer steps.
```

## Decision Tree

### S1: Weight-only improves CE/FACT before any data

Conclusion:

```text
Use WSYNC as default initialization before all adaptation.
```

Next:

```text
run 1.1B WSYNC + PopQA blend.
```

### S2: Weight-only improves weight MSE but not CE/FACT

Conclusion:

```text
Sigma_x ~= I is the wrong assumption.
Need at least tiny representative calibration or activation-aware objective.
```

Next:

```text
use PopQA / representative blend; keep WSYNC only as diagnostic.
```

### S3: Only dense/Hadamard rotation helps

Conclusion:

```text
Incoherence helps, but product runtime may not support it cheaply.
```

Next:

```text
consider structured Hadamard runtime or multi-plane hybrid, not pure I2_S.
```

### S4: No weight-only transform helps

Conclusion:

```text
Do not spend more time on weight-only sync.
The lever is representative data/objective or capacity/hybrid.
```

Next:

```text
FACT-003H PopQA blend, then Qwen/Gemma target ladder.
```

## Implementation Notes

Suggested script:

```text
scripts/rt_wsync_weight_only.py
```

Recommended API:

```bash
python scripts/rt_wsync_weight_only.py \
  --model-id Felladrin/Llama-160M-Chat-v1 \
  --arms fp,pt,row,row_norm,perm,hadamard \
  --eval-tokens 60000 \
  --json-out reports/rt_wsync_160m.json
```

Reuse:

```text
bitnet_llama/conversion.py
scripts/rt116_quality_recovery.py load_wikitext
scripts/fact004a_160m_smoke.py scoring helpers
```

Do not start with 1.1B. Weight-only screens are cheap enough to falsify at 160M first.

## Claim Discipline

Allowed if positive:

```text
Weight-only preprocessing reduces b1.58 conversion damage and improves the starting
point for later representative-data adaptation.
```

Not allowed:

```text
Weight-only sync solves factual recovery.
```

The current project evidence says factual behavior is sensitive to objective/data.
WSYNC can only be a better initialization unless a very strong result proves otherwise.
