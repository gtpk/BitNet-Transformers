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

Implement only the logger first.

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

## Sources To Read First

- Dynamically Expandable Networks: https://arxiv.org/abs/1708.01547
- AdaNet: https://arxiv.org/abs/1607.01097
- Net2Net: https://arxiv.org/abs/1511.05641
- HAWQ: https://arxiv.org/abs/1905.03696
- HAQ: https://arxiv.org/abs/1811.08886
- Post-Training Model Expansion: https://arxiv.org/abs/2503.17513
- Cascade-Correlation: https://proceedings.neurips.cc/paper/1989/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html
