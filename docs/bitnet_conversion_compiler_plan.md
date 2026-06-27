# BitNet Conversion Compiler Plan

Document position: [Index](./index.md) -> synthesis layer above WSYNC,
PopQA blend, PTQTP-lite, and hybrid capacity.

Status: modeling + execution plan. This is the current unifying hypothesis for the
project, not a completed result.

## One-Line Thesis

Converting an existing FP LLM to b1.58 is not a scalar quantizer problem.
It is a constrained compiler problem:

```text
choose a coordinate system,
preserve salient directions,
allocate extra capacity only where needed,
then adapt behavior with representative data.
```

The target artifact is:

```text
mostly I2_S / b1.58,
small and fast,
with enough factual/instruction behavior to be useful on low-resource hardware.
```

## Why The Model Changed

Early project framing:

```text
W -> ternary(W)
then short adaptation
```

Evidence rejected that as the whole story:

| observation | result | implication |
| --- | --- | --- |
| one-shot b1.58 PTQ | PPL/factual collapse | direct projection is too lossy |
| scale/objective/GPTQ/signed-eps sweeps | only small gains | quantizer tweaks are not the main lever |
| I2_S runtime parity | f16 ~= i2_s when artifact is correct | runtime is not the quality bottleneck |
| content-KL | improves facts without empty collapse | objective matters |
| small hard factual replay | train facts learned, eval transfer fails | tiny non-representative data creates memorization shortcuts |
| PopQA blend 160M smoke | train ~= heldout ~= eval, no memorization signature | representative distributions are healthier |
| post-hoc FP restore | worsens quality | capacity changes are coupled and cannot be patched after co-adaptation |
| external literature | AWQ/SmoothQuant/QuaRot/SpinQuant/PTQTP/TWLA | modern low-bit conversion is transform + saliency + capacity allocation |

Therefore the new abstraction is:

```text
BitNet Conversion Compiler
```

not:

```text
one ternary quantizer.
```

## Core Layer Model

For one linear layer:

```text
y_l = W_l x_l
```

The true output error of a quantized weight is:

```text
E_l
  = E_x ||(W_l - Wq_l) x_l||^2
  = Tr((W_l - Wq_l) Sigma_l (W_l - Wq_l)^T)
```

where:

```text
Sigma_l = E[x_l x_l^T]
```

Weight-only methods implicitly assume:

```text
Sigma_l ~= I
```

Then:

```text
E_l ~= ||W_l - Wq_l||_F^2
```

This assumption is often false in LLMs because activation distributions are anisotropic:

```text
outlier channels,
salient heads,
residual stream directions,
format/factual answer token directions.
```

## Coordinate Transform Model

Introduce an invertible transform `G_l`:

```text
x'_l = G_l x_l
W'_l = W_l G_l^{-1}
```

In FP:

```text
W_l x_l = W'_l x'_l
```

After quantization:

```text
Q(W_l) x_l
```

and:

```text
Q(W_l G_l^{-1}) G_l x_l
```

can behave very differently.

The compiler therefore searches:

```text
min_{G_l, Q_l}
  E_x || W_l x_l - Dec(Q_l(W_l G_l^{-1})) G_l x_l ||^2
```

Candidate transforms:

| symbol | family | literature | deployment status |
| --- | --- | --- | --- |
| `S_l` | diagonal scale / equalization | SmoothQuant, data-free equalization | plausible if folded safely |
| `P_l D_l` | signed permutation | rotation/incoherence family | cheapest deployable candidate |
| `H_l` | Hadamard / block rotation | QuaRot, QuIP | diagnostic or structured runtime |
| `R_l` | learned orthogonal rotation | SpinQuant | upper bound / expensive candidate |

## Representation / Capacity Model

Current I2_S-compatible one-plane representation:

```text
Wq_l = gamma_l T_l
T_l in {-1, 0, +1}
```

Capacity-expanded representation:

```text
Wq_l = sum_{k=1}^{K_l} alpha_{l,k} T_{l,k}
```

or hybrid residual:

```text
Wq_l = gamma_l T_l + U_l V_l^T
```

The compiler chooses:

```text
C_l in {
  I2_S one-plane,
  row-scale,
  two ternary planes,
  Q2/Q3 pocket,
  low-rank residual,
  FP pocket
}
```

subject to:

```text
memory budget,
latency budget,
runtime availability.
```

## Saliency Model

Let channel or layer saliency be:

```text
s_{l,j} = sensitivity of output/task loss to error in channel j
```

A simple activation-aware proxy:

```text
s_{l,j} ~= E[|x_{l,j}|] * ||W_{l,:,j}||
```

The stronger version uses the covariance-weighted residual:

```text
s_{l,j} ~= contribution of column j to
Tr((W_l - Wq_l) Sigma_l (W_l - Wq_l)^T)
```

Compiler rule:

```text
protect high-saliency channels first;
do not add capacity uniformly.
```

This is the AWQ/SmoothQuant lesson translated to b1.58 conversion.

## Inter-Layer Interaction Model

Layer choices are not independent.

Naive selector:

```text
min sum_l cost_l(C_l)
```

Observed failure:

```text
post-hoc FP restore worsened quality;
single-layer changes can break downstream co-adaptation.
```

Better selector:

```text
min sum_l cost_l(C_l)
  + sum_l psi_l(C_l, C_{l+1})
```

where `psi_l` measures adjacent interaction cost:

```text
activation distribution shift,
residual stream mismatch,
attention/MLP handoff mismatch.
```

This is the TWLA-style inter-layer-aware lesson.

## Adaptation / Data Model

Let `A` be the adaptation objective and `D_rep` the representative data distribution.

Current objective family:

```text
L_adapt =
  L_answer_CE
  + lambda * L_content_KL(base, student)
  + rho * L_representative_QA
```

Rejected form:

```text
mu * L_small_fact_CE
```

when the fact set is tiny and separate. Empirically it creates:

```text
train facts high,
held-out/eval facts flat or worse.
```

Compiler rule:

```text
Small hard facts are diagnostics.
Representative QA blend is adaptation data.
```

## Unified Optimization Problem

The project's current target can be written as:

```text
min_{G, C, T, alpha, A}
  E_{x ~ D_rep} [
    L_task(
      f_FP(x),
      f_{G,C,T,alpha,A}(x)
    )
  ]
  + lambda_mem * Memory(C)
  + lambda_lat * Latency(C)
  + lambda_int * sum_l psi_l(C_l, C_{l+1})
```

Layer-local proxy:

```text
min
  sum_l Tr((W_l - Wq_l) Sigma_l (W_l - Wq_l)^T)
  + sum_l Omega_capacity(C_l)
  + sum_l psi_l(C_l, C_{l+1})
  + L_adapt(D_rep)
```

where:

```text
G_l       coordinate transform
C_l       representation/capacity choice
Sigma_l   activation covariance / saliency
T_l       ternary code plane(s)
alpha_l   scale(s)
A         adaptation objective
```

## Critique-Driven Refined Form

The broad `G-C-S-A` form is useful as a map, but too broad as an optimizer.
The refined version treats `S` as an estimator rather than a free variable, restricts
`G` to valid model-preserving transforms, and decomposes `A` into data/loss/budget.

### S is an estimator, not a free knob

Instead of:

```text
min_{G,C,S,A}
```

use:

```text
S_l = phi_l(W_l, Sigma_l(D_cal), task)
```

where `phi_l` can be chosen from:

| estimator | literature hint | meaning |
| --- | --- | --- |
| weight norm | weight-only fallback | no calibration needed, weakest signal |
| activation magnitude | AWQ | salient channel proxy |
| Hessian trace/eigenvalue | HAWQ | second-order layer sensitivity |
| block reconstruction residual | BRECQ / GPTQ | local output-preservation signal |
| PopQA / instruction covariance | our FACT/PopQA line | representative factual/instruction directions |

Then:

```text
(G*, C*, A*) =
argmin_{G in G_valid, C in C_budget, A in A_budget}
  J(G, C, phi(W, Sigma_D, task), A)
```

This avoids double-counting `Sigma_l` and `S_l`.

### G must be valid

Use:

```text
G_l in G_valid
```

where `G_valid` contains only transforms that either preserve the FP function exactly
or have an explicitly measured approximation error:

```text
signed permutations,
foldable diagonal equalization,
structured Hadamard/Kronecker rotations with known insertion points,
learned rotations only as an upper bound.
```

This prevents the formula from pretending arbitrary rotations are always legal inside
RMSNorm/RoPE/SwiGLU/residual paths.

### C is discrete and budgeted

Use:

```text
C in C_budget
```

with:

```text
C_l in {I2_S, row-scale, two-plane, Q2/Q3, low-rank residual, FP pocket}
```

This is a mixed discrete-continuous optimization problem. The practical solver is not
gradient descent over all variables; it is staged search:

```text
screen -> rank -> allocate -> adapt -> verify runtime.
```

### A should be decomposed

Use:

```text
A = (D_rep, L_adapt, B_train)
```

where:

| part | meaning |
| --- | --- |
| `D_rep` | representative data distribution, e.g. PopQA/instruction blend |
| `L_adapt` | CE, content-KL, distillation, answer mask, replay rules |
| `B_train` | steps, trainable params, optimizer, hardware budget |

This prevents "adaptation" from hiding the whole learning process inside one symbol.

### Resource is memory traffic, not only file size

Replace:

```text
lambda_mem * Memory(C) + lambda_lat * Latency(C)
```

with:

```text
R(G,C) =
  a * bytes_moved(G,C)
  + b * ops(G,C)
  + c * kernel_overhead(G,C)
  + d * unsupported_runtime_penalty(G,C)
```

This follows the project's memory-traffic-first goal and the HAQ lesson: the right
policy depends on actual hardware/runtime feedback, not only nominal bitwidth.

### Fidelity is a vector

Do not collapse quality into one scalar too early. Track:

```text
Fidelity =
  (CE/PPL, logit KL, FACT score, degeneration tags, instruction behavior, runtime parity)
```

Scalarization is allowed only after the reporting table exists:

```text
J = w_ce CE + w_fact FACT_loss + w_deg Degeneration + ...
```

This guards against a repeated failure mode:

```text
CE improves while factual behavior stays weak.
```

## Literature-Based Fixes To The Critique

The critique of `G-C-S-A` maps to known method families:

| critique | fix | literature anchor | project translation |
| --- | --- | --- | --- |
| `S` too vague | estimate sensitivity from Hessian / activation / residual | HAWQ, AWQ, GPTQ | `S = phi(...)`, not free |
| discrete `C` hard to optimize | hardware-aware or mixed-precision search | HAQ, HAWQ | budgeted beam/DP/greedy over layer choices |
| layer independence false | reconstruct blocks, not isolated layers | BRECQ, OmniQuant | block-level WSYNC / PTQTP-lite |
| arbitrary rotations invalid | restrict to foldable/equivalent transforms | SmoothQuant, QuaRot, SpinQuant | `G_valid` only |
| fidelity target unclear | block output / logits / task metrics | BRECQ, GPTQ, OmniQuant | vector scorecard before scalarization |
| data-free too weak | use generated or representative calibration cautiously | LLM-QAT, PopQA blend | data as sensor before data as trainer |

References:

- HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision:
  <https://arxiv.org/abs/1905.03696>
- HAQ: Hardware-Aware Automated Quantization with Mixed Precision:
  <https://arxiv.org/abs/1811.08886>
- BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction:
  <https://arxiv.org/abs/2102.05426>
- OmniQuant: Omnidirectionally Calibrated Quantization for LLMs:
  <https://arxiv.org/abs/2308.13137>
- AdaRound: Up or Down? Adaptive Rounding for PTQ:
  <https://arxiv.org/abs/2004.10568>
- GPTQ:
  <https://arxiv.org/abs/2210.17323>
- LLM-QAT: Data-Free Quantization Aware Training for LLMs:
  <https://arxiv.org/abs/2305.17888>

## Practical Solver Sketch

Do not solve the full compiler problem at once.

### Stage 1: Geometry screen

```text
Fix C = I2_S / row-scale.
Fix A = none.
Search small G_valid.
```

Goal:

```text
Does coordinate/equalization help before adaptation?
```

### Stage 2: Saliency estimation

```text
Fix best G from Stage 1.
Estimate S with:
  weight-only proxy,
  PopQA activation proxy,
  Hessian/block residual proxy if affordable.
```

Goal:

```text
Do top-saliency channels/layers predict conversion damage better than random?
```

### Stage 3: Capacity allocation

```text
Fix G and S.
Search C under byte/latency budget:
  one-plane,
  two-plane,
  row-scale,
  Q2/Q3/FP pockets.
```

Use:

```text
greedy / beam search / adjacent-state DP
```

not a full independent knapsack.

### Stage 4: Representative adaptation

```text
Fix G,C,S.
Run A = PopQA/instruction blend + content-KL.
```

Goal:

```text
Does better geometry/capacity reduce adaptation budget or improve FACT score?
```

### Stage 5: Runtime verification

```text
Export only representations with a plausible runtime path.
Measure file size, bytes moved, token-gen speed, and parity.
```

This keeps the compiler aligned with the low-resource product goal.

## Current Assumptions

### A1. Activation anisotropy

```text
Sigma_l is not close to I.
```

Therefore weight MSE alone is insufficient.

### A2. Gauge matters

```text
There exists a transform G_l such that ternary projection is easier in that basis.
```

This is the QuaRot/SpinQuant/SmoothQuant direction.

### A3. Saliency is sparse or at least uneven

```text
Some channels/layers matter much more than others.
```

Therefore uniform capacity is wasteful.

### A4. Capacity shortage is local

```text
One-plane b1.58 is not equally insufficient everywhere.
```

Therefore adaptive trit-plane or hybrid allocation should beat uniform expansion.

### A5. Layer decisions are coupled

```text
C_l changes the distribution seen by C_{l+1}.
```

Therefore additive knapsack is only a first approximation.

### A6. Adaptation data must be representative

```text
Tiny hard fact sets create memorization shortcuts.
Representative QA streams create behavior alignment.
```

## Hypotheses To Test

### H1. WSYNC improves the no-data start

Prediction:

```text
weight-only equalization/rotation lowers CE or FACT before any adaptation.
```

Pass:

```text
CE improves by >= 0.5 nats
or FACT improves by >= 0.05
on 160M.
```

### H2. Saliency predicts where capacity is needed

Prediction:

```text
top-saliency layers/channels explain most conversion error.
```

Pass:

```text
protecting top-k saliency targets beats random top-k under same bytes.
```

### H3. Plane capacity beats codebook tweaks

Prediction:

```text
two-plane residual fitting beats signed-epsilon / threshold / GPTQ-lite.
```

Pass:

```text
PTQTP-lite improves one-plane FACT/CE under comparable target bytes.
```

### H4. Representative blend beats small hard replay

Prediction:

```text
PopQA/instruction blend shows no train/eval memorization gap and improves 1.1B behavior.
```

Pass:

```text
train PopQA, heldout PopQA, and FACT panel move together;
no train=1.0 / eval-flat signature.
```

### H5. Combined compiler beats each component

Prediction:

```text
WSYNC init + representative blend + selective extra capacity
beats any single component.
```

Pass:

```text
same adaptation budget yields higher FACT/CE than PopQA blend alone,
or same FACT with fewer steps/tokens.
```

## Execution Plan

### COMP-000: Keep Systems Substrate Fixed

Do not reopen solved runtime questions unless a new representation requires it.

Known:

```text
I2_S x86 runtime works.
Path A' Wq=gamma*T export works.
Mac M5 bitnet.cpp ternary runtime is toolchain-broken, not algorithm-broken.
```

### COMP-001: Finish Active FACT Branches

Inputs:

```text
FACT-003D mu=0.25 final result
FACT-003H PopQA blend readiness
```

Decision:

```text
if mu=0.25 <= content-KL baseline:
  small hard replay is demoted
  PopQA blend becomes main adaptation branch
```

### COMP-002: Run WSYNC-001 / 002 On 160M

Goal:

```text
test whether weight-only geometry improves the starting point.
```

Arms:

```text
per-tensor I2_S
row-scale
row-norm correction
diagonal equalization
signed permutation
Hadamard diagnostic
```

Output:

```text
reports/rt_wsync_160m.json
docs/weight_only_sync_plan.md result section
```

### COMP-003: Add PopQA Calibration For Saliency

Goal:

```text
estimate Sigma_l / saliency with representative QA rather than tiny facts.
```

Arms:

```text
AWQ-lite scale
SmoothQuant-lite scale migration
saliency-ranked protected channels
```

Output:

```text
saliency table by layer/channel group
top-k protection vs random-k comparison
```

### COMP-004: PTQTP-Lite / Adaptive Plane Probe

Goal:

```text
test whether one-plane capacity is the remaining bottleneck.
```

Arms:

```text
one-plane I2_S
two-plane all layers
two-plane top-saliency layers only
two-plane top residual layers only
```

Measure:

```text
target bytes,
CE/FACT,
runtime feasibility,
comparison to Q2_K.
```

### COMP-005: Combine Best Geometry + PopQA Blend

Goal:

```text
does better initialization reduce adaptation data/steps and improve factual quality?
```

Arms:

```text
PopQA blend from current I2_S init
PopQA blend from WSYNC init
PopQA blend from WSYNC + selective plane init
```

### COMP-006: Scale Ladder

Only scale once the 160M/1.1B branch shows a real component.

Suggested ladder:

```text
160M -> TinyLlama 1.1B -> Qwen/Gemma small audit -> Qwen/Gemma 7B-class goalpost
```

Do not run 7B before:

```text
runtime/export audit,
target linear policy audit,
memory budget estimate,
representative eval panel,
one smaller model success.
```

## Expected Outcomes

### Outcome O1: WSYNC works

Then:

```text
make WSYNC the default initialization before adaptation.
```

### Outcome O2: WSYNC only improves weight MSE

Then:

```text
data-free geometry is misaligned with task directions;
move to PopQA-calibrated saliency.
```

### Outcome O3: PopQA blend works but WSYNC does not

Then:

```text
representative data is the main lever;
focus on blend size/objective and scale-up.
```

### Outcome O4: PopQA blend also plateaus

Then:

```text
one-plane same-topology is likely capacity-limited;
run PTQTP-lite / selective planes.
```

### Outcome O5: Selective planes work

Then:

```text
project becomes adaptive ternary-capacity compiler.
```

### Outcome O6: Nothing moves FACT near Q2_K

Then:

```text
all-I2_S conversion remains a systems result;
product path requires larger base model, hybrid precision, or external PTQ method.
```

## Product Claim Guardrail

Allowed now:

```text
We have a faithful and fast I2_S substrate and a growing compiler plan for conversion.
```

Not allowed yet:

```text
We can convert arbitrary LLaMA/Qwen/Gemma models to useful b1.58 with near-Q2 quality.
```

The product claim becomes plausible only when:

```text
representative blend or compiler components improve factual/instruction behavior
on a model that already has useful base ability,
while retaining I2_S or mostly-I2_S speed/storage advantages.
```
