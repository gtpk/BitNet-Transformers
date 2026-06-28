# Entropy-Guided I2_S Growth Plan

Status: proposal / next PC-side instrumentation.

This note answers the question:

```text
If a layer keeps moving back and forth during STE adaptation, can we treat that
as a sign that one I2_S plane cannot express the needed function, then grow
channels/planes/sidecars only there?
```

Short answer:

```text
yes, but only after separating "optimizer noise" from "capacity bottleneck."
```

The useful object is not entropy alone. It is:

```text
temporal ternary instability
+ output residual
+ task saliency
+ positive improvement per added byte
```

This keeps the idea rooted in I2_S. We do not replace the model with a generic
LoRA/Q2/Q3 model. We use instability to choose where the I2_S trunk deserves the
smallest auxiliary organ.

## Literature Map

The exact "b1.58 layer confusion entropy" idea is not a standard named method,
but the surrounding pieces are real:

| line | relevant idea | what we borrow |
| --- | --- | --- |
| Cascade-Correlation | add hidden units to chase residual error | grow only when residual remains structured |
| Net2Net | widen/deepen a net with function-preserving transformations | initialize growth so the model does not jump |
| AdaNet | adaptive structure learning with complexity regularization | reward improvement only if it beats complexity cost |
| Dynamically Expandable Networks | expand/split units when capacity or drift requires it | "semantic drift" is analogous to unstable ternary codes |
| HAQ / HAWQ | assign precision per layer using hardware or Hessian sensitivity | use sensitivity, not uniform capacity |
| Post-Training Model Expansion | expansion can improve quantized model quality after training | small expansion is a valid quantization co-design move |

Closest new clue:

```text
Post-Training Model Expansion explicitly argues that post-training optimization
does not only shrink models; it may selectively expand them to improve quality
inside a quantization co-design space.
```

Our twist:

```text
expand only inside an I2_S-rooted artifact, using STE instability as a locator.
```

## Corrected Projection View

It is useful to describe b1.58 conversion as projection, but this is only a
modeling frame. I2_S is not a convex linear subspace. It is a discrete codebook:

```text
M_I2S = { gamma * T | T_ij in {-1,0,+1}, gamma > 0 }
```

So:

```text
W -> Q_I2S(W)
```

is better described as:

```text
discrete constrained approximation
```

not a clean Euclidean projection with a unique solution.

The product-relevant problem is also not pure weight MSE. The layer matters only
through its input distribution:

```text
Q*(W) = argmin_Q E_x || W x - Q(W) x ||^2
```

and the model matters only through final task behavior:

```text
min  L_task(f_Q)
```

Therefore the more honest objective is:

```text
min_{G, q, a}
  L_val(f_{G,q,a})
  + lambda_mem * Bytes(q,a)
  + lambda_lat * Latency(q,a)
  + lambda_aux * AuxCost(a)
```

where:

```text
G = cheap coordinate transform
q = I2_S-compatible ternary trunk
a = small auxiliary capacity
```

with constraints:

```text
q stays I2_S
a stays small
I2_S remains the trunk
```

Coordinate transforms are only allowed if they are cheap or foldable:

```text
diagonal scale
signed permutation
block-Hadamard / RHT
small structured orthogonal transform
```

Dense learned rotation is only an upper-bound diagnostic unless it can be
rewritten into a cheap I2_S-rooted transform.

Current evidence:

```text
WSYNC scaling failed.
Fixed block-Hadamard H-I2S failed.
Therefore data-free projection tricks are not trusted until they beat behavior,
not only reconstruction.
```

## Why Entropy Alone Is Not Enough

A layer can have high code entropy for three different reasons:

1. healthy learning,
2. noisy optimizer / too high learning rate,
3. true representation shortage.

So this is wrong:

```text
high entropy => add capacity
```

The safer rule is:

```text
high temporal instability
+ high output residual
+ high task saliency
+ validation improvement when grown
=> capacity bottleneck
```

If growth does not improve held-out FACT/CE, the instability was not useful
evidence. It was either optimizer noise, data mismatch, or STE mismatch.

This is the central falsification rule:

```text
top-k layers by bottleneck_score must beat random-k layers.
```

If top-k does not beat random-k, the entropy score is just a nice-looking
diagnostic, not a capacity locator.

## EGROW-001 Result: Sensitivity, Not Instability

EGROW-001 has now been run on 160M with two seeds.

Result:

```text
top-8 bottleneck overlap across seeds 41/42: 7/8
top blocks: {0, 3, 9, 10, 11}
not last-layer-only
```

Shared top layers:

```text
layers.0.mlp.down_proj
layers.3.mlp.down_proj
layers.9.mlp.down_proj
layers.10.mlp.down_proj
layers.11.mlp.down_proj
layers.9.self_attn.o_proj
layers.11.self_attn.o_proj
```

Honest correction:

```text
flip_rate ~= 0.000
temporal_entropy ~= 0.002
```

The ternary codes settle quickly. They do not keep flipping. Therefore the
original intuition:

```text
frequent ternary flipping => capacity bottleneck
```

is not supported at 160M.

What did work:

```text
B_l is driven by output_residual x task_saliency.
```

So EGROW is no longer "entropy proves the layer is confused." It is now:

```text
use sensitivity-weighted residual to locate where I2_S loses important function.
```

The down-proj-heavy result also matches quantization literature intuition:

```text
MLP down_proj maps expanded intermediate channels back to hidden size,
so outlier/salient modes can be concentrated there.
```

Keep the temporal-instability metrics in the logger because they are useful
false-positive controls, but do not expect them to drive the ranking.

Archived result:

```text
reports/egrow_160m_layer_instability.md
```

## Signals To Log

For each target linear layer `l`, maintain a rolling window over training steps.

### 1. Ternary Flip Rate

```text
F_l(t) = mean_ij 1[T_l^t[i,j] != T_l^{t-1}[i,j]]
```

Interpretation:

```text
many weights keep changing ternary buckets
```

This is the direct version of "the layer keeps coming back and forth."

### 2. Temporal Code Entropy

For each weight element, look at the empirical distribution of its recent ternary
states:

```text
p_ij(k) = Pr(T_l^{t-w:t}[i,j] = k),  k in {-1,0,+1}
H_ij = - sum_k p_ij(k) log p_ij(k)
H_time_l = mean_ij H_ij
```

High `H_time_l` means the ternary assignment itself is unstable.

### 3. Gradient Conflict / Reversal

```text
C_l(t) = 1 - cos(g_l^t, g_l^{t-1})
R_l(t) = 1[ <Delta theta_l^t, Delta theta_l^{t-1}> < 0 ]
```

High values mean optimization is pushing the same layer in alternating
directions.

### 4. Output Residual

On a small representative batch:

```text
E_l = E_x || W_l x - Q_l(W_l) x ||_2^2 / E_x || W_l x ||_2^2
```

This prevents a noisy but unimportant layer from getting capacity.

### 5. Task Saliency

Use FACT / PopQA / instruction batches:

```text
S_l = || partial L_task / partial h_l ||
```

or a cheaper proxy:

```text
S_l = activation_norm_l * gradient_norm_l
```

This prevents us from growing layers that matter for WikiText CE but not for the
behavior gap we care about.

## Bottleneck Score

Normalize each scalar across layers, then compute:

```text
I_l = a * norm(H_time_l)
    + b * norm(F_l)
    + c * norm(C_l)
    + d * norm(R_l)

B_l = I_l * norm(E_l) * norm(S_l)
```

`I_l` is instability.

`B_l` is capacity-bottleneck suspicion.

The multiplicative form is intentional:

```text
unstable but low residual => ignore
unstable but low FACT saliency => ignore
high residual but stable => quantizer error, not growth location
all high => candidate for growth
```

## Growth Actions

All actions must keep I2_S as the trunk.

| action | form | when to try |
| --- | --- | --- |
| low-rank sidecar | `gamma*T + B A` | first candidate; cheapest and already planned |
| selected extra ternary plane | `gamma1*T1 + gamma2*T2` | if residual has structured modes beyond rank sidecar |
| selected Q2/Q3 pocket | tiny top-k sensitive slice | diagnostic only; must stay smaller than broad Q2_K |
| train-from-start hybrid | initialize auxiliary branch at step 0 | if post-hoc growth causes co-adaptation mismatch |

Do not use:

```text
whole-layer FP restore after training
```

HYBRID-001A already showed that post-hoc FP restore breaks co-adaptation.

## Evolution Loop

This is the evolutionary part.

```text
1. Train/adapt I2_S model for a short window.
2. Log B_l for all target linear layers.
3. Propose mutations on top-k layers:
   - +rank 2 sidecar
   - +rank 4 sidecar
   - +one extra ternary plane
   - +tiny Q2 pocket
4. Run short evaluation.
5. Keep mutation only if:
     Delta FACT / Delta bytes > threshold
     and CE/tags do not collapse
     and i2_s ~= f16 still holds for the I2_S trunk.
6. Prune auxiliary branches whose learned norm stays near zero.
```

Objective:

```text
min_theta  L_adapt(theta)
         + lambda_bytes * bytes(theta)
         + lambda_ops   * ops(theta)
         + lambda_aux   * auxiliary_branch_penalty(theta)
```

Selection rule:

```text
accept mutation m if

  score(m) = Delta FACT
           - alpha * Delta CE
           - beta  * Delta bytes
           - gamma * Delta token_latency

is positive on held-out data.
```

## Decision Table

| observation | interpretation | next |
| --- | --- | --- |
| high `B_l`, sidecar helps | local I2_S capacity bottleneck | keep I2_S-rooted growth |
| high `B_l`, sidecar does not help | STE/optimizer/data issue, not capacity | improve objective/data |
| low `B_l`, bad FACT | factual gap is not in target linears | data/objective, embedding/lm_head diagnostic |
| train facts high, held-out flat | replay memorization | broader representative blend |
| sidecar helps only at high rank | one-plane I2_S lacks too much capacity | compare PTQTP-lite / Q2 pocket honestly |
| extra branch carries most bytes/ops | no longer I2_S-rooted product | demote to upper-bound diagnostic |

## PC / Colab Split

### PC First

Use RTX 3080 for cheap 160M probes:

```text
EGROW-001: instrumentation only
  log F_l, H_time_l, C_l, R_l, E_l, S_l during 160M adaptation

EGROW-002: top-k sidecar smoke
  add rank-2 or rank-4 sidecar only to top 1/3/5 layers by B_l
  compare with random-k layers

EGROW-003: false-positive test
  lower learning rate or change seed
  if B_l disappears, it was optimizer noise
```

### Colab Only After PC Signal

Use Colab/L4 for 1.1B only if PC shows:

```text
top-k B_l sidecar > random-k sidecar
and FACT moves
and overhead is small
```

Then:

```text
EGROW-004: 1.1B top-k confirmation
EGROW-005: export/runtime accounting
```

## Why This Is Different From Sidecar Alone

Plain sidecar asks:

```text
does extra capacity help anywhere?
```

Entropy-guided growth asks:

```text
where is the I2_S trunk visibly failing to settle,
and does the smallest local auxiliary branch fix that exact failure?
```

This distinction matters. If we add sidecars everywhere, the method becomes just
another LoRA adapter. If we add them only where `B_l` says one I2_S plane is
unstable, it remains an I2_S-rooted conversion method.

## Immediate Next Work

Do not start by adding capacity. Implement only the logger first.

Required output:

```text
reports/egrow_160m_layer_instability.json
```

Minimum fields:

```text
layer_name
module_type
flip_rate
temporal_entropy
gradient_cosine_conflict
update_reversal_rate
output_residual
task_saliency
bottleneck_score
rank
```

First success condition:

```text
the top layers by bottleneck_score are stable across 2 seeds
and are not identical to "last layers only" by construction.
```

Only then should SIDE-001 use the ranking instead of all-layer sidecars.

## Detailed Experiment Ladder

### EGROW-001: Instrumentation Only

Status:

```text
DONE: first success condition met.
```

Goal:

```text
find whether layer instability is measurable and stable
```

Run on PC / 160M:

```text
model: Llama-160M
recipe: current content-KL / PopQA-compatible adaptation recipe
no sidecar
no extra plane
log every N steps
```

Outputs:

```text
reports/egrow_160m_layer_instability_seed41.json
reports/egrow_160m_layer_instability_seed42.json
reports/egrow_160m_layer_instability_summary.md
```

Pass:

```text
MET:
  top-8 overlap 7/8 across seeds
  not last-layer-only
  high residual + FACT saliency drives the ranking
```

Fail:

```text
rankings are random across seeds
or the score is dominated by a single noisy metric
```

If fail:

```text
do not add sidecars from entropy;
return to data/objective or plain SIDE-001 all-layer smoke.
```

Decision:

```text
proceed to EGROW-002,
but interpret the locator as sensitivity/residual-guided, not flip-guided.
```

### EGROW-002: Top-k Sidecar Versus Random-k

Goal:

```text
test whether the score actually identifies useful growth locations
```

Arms:

| arm | layers | sidecar |
| --- | --- | --- |
| base | none | rank 0 |
| top1 | highest `B_l` shared layer | rank 4 |
| top3 | top 3 shared `B_l` layers, down_proj-heavy | rank 4 |
| top7 | all shared top layers from EGROW-001 | rank 4 or rank 2 |
| random3 | random 3 matched module types | rank 4 |
| random7 | random 7 matched module types | rank 2 or rank 4 |
| last3 | last 3 target modules | rank 4 |
| all-small | all target modules | rank 2 |

Initial top-k candidate set:

```text
layers.0.mlp.down_proj
layers.3.mlp.down_proj
layers.9.mlp.down_proj
layers.10.mlp.down_proj
layers.11.mlp.down_proj
layers.9.self_attn.o_proj
layers.11.self_attn.o_proj
```

Pass:

```text
top-k > random-k and top-k > last-k on held-out FACT/PopQA,
with small byte/ops overhead,
and no CE/tag collapse.
```

Fail:

```text
top-k does not beat random-k,
or only all-layer sidecar works,
or required rank is too large.
```

Interpretation:

```text
top-k wins      => entropy-guided local capacity is real
random-k ties   => sidecar helps generically, score is not useful
all-small wins  => broad low-rank adaptation, not layer locator
none wins       => capacity not the current bottleneck
```

### EGROW-003: False-Positive Controls

Run cheap controls before Colab:

```text
lower LR
different seed
shuffle FACT saliency batch
random-k matched by layer type
```

If the same layers stay top-ranked under real FACT saliency but not under
shuffled saliency, the score is more credible.

### EGROW-004: 1.1B Confirmation

Launch on Colab only if EGROW-002 passes on PC.

Arms:

```text
base content-KL / PopQA blend
top-k sidecar rank 4
random-k sidecar rank 4
```

Pass:

```text
top-k improves FACT/PopQA over base and random-k,
i2_s ~= f16 remains,
sidecar overhead remains below the predeclared budget.
```

### EGROW-005: Runtime Accounting

Only after quality passes:

```text
report I2_S target bytes
report sidecar bytes
report sidecar ops proxy
compare against Q2_K/Q3_K whole-model bytes and token-gen
```

If the sidecar cost approaches Q2_K/Q3_K while quality does not beat them, the
branch is demoted to analysis, not product.

## Sources To Read First

- Dynamically Expandable Networks: https://arxiv.org/abs/1708.01547
- AdaNet: https://arxiv.org/abs/1607.01097
- Net2Net: https://arxiv.org/abs/1511.05641
- HAWQ: https://arxiv.org/abs/1905.03696
- HAQ: https://arxiv.org/abs/1811.08886
- Post-Training Model Expansion: https://arxiv.org/abs/2503.17513
- Cascade-Correlation: https://proceedings.neurips.cc/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html
