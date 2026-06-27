# Current Theory, Hypotheses, And Experiment Plan

Document position: [Index](./index.md) -> compact control-room summary above
[BitNet Conversion Compiler Plan](./bitnet_conversion_compiler_plan.md).

Status: living synthesis. Update this file when a result changes the decision tree,
not after every intermediate log line.

## One-Line Project Goal

```text
Make existing public LLMs run like low-resource models by reducing memory traffic,
while preserving enough useful behavior to matter.
```

This is not the same as:

```text
prove that pure 1.58-bit PTQ always works.
```

The product target is:

```text
mostly b1.58 / I2_S,
fast on commodity hardware,
with a controlled amount of adaptation,
and with fallback capacity only where it buys real behavior.
```

## Current Evidence Snapshot

| axis | current status | evidence |
| --- | --- | --- |
| I2_S runtime/export | solved on x86/Linux | RT-111/112: f16/f32 ~= i2_s when `Wq=gamma*T` is exported |
| storage | solved | target linears are 16x smaller than f32, whole-file ratio improves with scale |
| speed | solved for dense LLaMA | token-generation speedup grows from tiny -> 160M -> 1.1B |
| one-shot b1.58 PTQ | failed | PTQ PPL/factual collapse |
| quantizer tweaks | mostly ruled out | scale/objective/AWQ/GPTQ/signed-eps all too small |
| CE/PPL recovery | works | short target-linear adaptation recovers large CE fraction |
| decoding usability | rescued | repetition penalty/sampling avoids greedy attractors |
| factual quality | still open | content-KL best fact is modest; small hard replay overfits |
| post-hoc capacity restore | failed | late FP restore worsens behavior after all-ternary co-adaptation |
| representative blend | promising mechanism, not yet final | PopQA blend avoids train/eval memorization signature in 160M smoke |

## The Main Theory

### T0. b1.58 Conversion Is A Compiler Problem

The old mental model was:

```text
FP weight -> ternary weight -> short tuning
```

The current model is:

```text
choose a valid coordinate system,
estimate salient directions,
allocate representation capacity,
adapt on representative behavior,
verify the real runtime.
```

So the problem is not one scalar quantizer. It is:

```text
BitNet conversion compiler = G + C + S + A + runtime
```

where:

| symbol | meaning |
| --- | --- |
| `G` | valid coordinate transform / gauge |
| `C` | representation and capacity choice |
| `S` | saliency estimate, not a free knob |
| `A` | adaptation data, loss, and training budget |

### T1. The Correct Layer Error Is Activation-Weighted

For one layer:

```text
y = W x
yq = Wq x
```

The expected output error is:

```text
E_l = E_x ||(W_l - Wq_l)x_l||^2
    = Tr((W_l - Wq_l) Sigma_l (W_l - Wq_l)^T)
```

where:

```text
Sigma_l = E[x_l x_l^T]
```

Weight-only PTQ assumes:

```text
Sigma_l ~= I
```

Then:

```text
E_l ~= ||W_l - Wq_l||_F^2
```

Our results say this assumption is too weak for final quality. It may help
initialization, but it cannot know factual/instruction directions by itself.

### T2. Coordinate Choice Matters

If `G_l` is a valid transform:

```text
x'_l = G_l x_l
W'_l = W_l G_l^{-1}
```

then in full precision:

```text
W_l x_l = W'_l x'_l
```

But after quantization:

```text
Q(W_l)x_l != Q(W_l G_l^{-1})G_l x_l
```

Therefore a transform can make the same FP function easier or harder to
represent as ternary.

This is the shared lesson from:

```text
SmoothQuant / AWQ      -> scale or protect important channels
QuaRot / SpinQuant     -> rotate into a quantization-friendly basis
PTQTP / TWLA           -> add structured capacity or layer-aware choices
TurboQuant-style RHT   -> make runtime vectors/codebooks easier to quantize
```

### T3. One-Plane I2_S Has A Real Capacity Limit

The current I2_S-compatible form is:

```text
Wq_l = gamma_l T_l
T_l in {-1, 0, +1}
```

Capacity expansions are:

```text
Wq_l = sum_{k=1}^{K_l} alpha_{l,k} T_{l,k}
```

or:

```text
Wq_l = gamma_l T_l + residual_l
```

But capacity should not be added blindly. Post-hoc FP restore failed because the
all-ternary adapted model became co-adapted:

```text
early ternary layers learned to feed later ternary layers.
```

Changing only late layers after training broke that system. Capacity changes must
be trained from the start or inserted through a valid compiler step.

### T4. Adaptation Data Is Not Just "More Data"

We saw three regimes:

```text
small hard facts    -> memorization shortcut
WikiText/Dolly CE   -> fluency recovery, weak factual recovery
content-KL          -> first useful factual lever
representative QA   -> likely needed to avoid memorization and forgetting
```

The current adaptation loss family is:

```text
L_adapt =
  L_answer_CE(D_rep)
  + lambda * L_content_KL(base, student; exclude EOS/special)
  + optional representative QA blend
```

Small separate hard replay:

```text
mu * L_small_fact_CE
```

is now demoted unless the dataset is large and representative. With 291 facts, it
can drive train facts to 1.0 while held-out/eval facts stay flat or worsen.

### T4b. Why PopQA Blend Can Work

Let:

```text
D_lm      = general language / instruction stream
D_fact    = representative factual QA stream, e.g. PopQA
D_eval    = fixed held-out factual panel
D_small   = tiny protected fact set
```

Small hard replay optimizes:

```text
L_small(theta) =
  L_answer_CE(D_lm)
  + lambda * L_content_KL(base || theta)
  + mu * L_CE(D_small)
```

PopQA blend instead optimizes:

```text
D_rep = (1 - rho) D_lm + rho D_fact

L_blend(theta) =
  L_answer_CE(D_rep)
  + lambda * L_content_KL(base || theta)
```

The important difference is not only size. It is gradient geometry.

For a factual evaluation risk:

```text
F_eval(theta) = E_{(q,a) ~ D_eval}[-log p_theta(a | q)]
```

one small gradient step gives:

```text
Delta F_eval
  ~= - eta < grad F_eval, grad L_train >_{H^{-1}}
```

where `H^{-1}` is the local preconditioner / curvature geometry. Training helps
factual behavior only when:

```text
< grad F_eval, grad L_train >_{H^{-1}} > 0
```

Tiny hard replay has:

```text
grad L_CE(D_small) = g_fact + noise
Var(noise) ~= sigma^2 / n_eff
```

with very small `n_eff`. Repeating the same 291 facts many times lowers train loss
but does not create new coverage:

```text
n_eff does not grow like repeated tokens.
```

So the optimizer can find a cheap item-specific shortcut:

```text
train facts up,
held-out / FACT flat or worse.
```

Representative blend has larger support:

```text
n_eff(PopQA) >> n_eff(D_small)
Var(noise) lower
grad L_train closer to grad F_eval
```

if `D_fact` and `D_eval` share enough factual QA structure. A useful bound is the
usual domain-adaptation intuition:

```text
F_eval(theta)
  <= F_fact(theta)
   + discrepancy(D_eval, D_fact)
   + generalization_error(n_eff)
```

FACT-003H is testing whether PopQA makes both the discrepancy and the
generalization error small enough to move real held-out factual behavior.

Expected good signature:

```text
train PopQA, tight held-out PopQA, and FACT panel move together.
```

Bad signatures:

```text
train up, held-out flat      -> memorization shortcut
held-out up, FACT flat       -> distribution mismatch
CE up, facts down            -> objective still misaligned
```

### T5. Runtime Correctness Is A Separate Theorem

For export:

```text
Wq = gamma * T
```

If we materialize `Wq` before GGUF conversion, upstream I2_S stores:

```text
scale = max(|Wq|) = gamma
```

So:

```text
Q_I2S(Wq) ~= Wq
```

on x86/Linux runtime. This solved the export problem. Quality failures after that
are model/adaptation failures, not I2_S byte-layout failures.

## Unified Objective

The current high-level objective is:

```text
min_{G, C, A}
  FidelityLoss(
    f_FP,
    f_{G,C,A}
  )
  + lambda_R * R(G,C)
  + lambda_I * Interaction(G,C)
  + lambda_B * Budget(A)
```

with:

```text
S_l = phi_l(W_l, Sigma_l(D_cal), task)
```

and:

```text
FidelityLoss =
  w_ce   * CE
  + w_fact * FACT_loss
  + w_deg  * degeneration
  + w_kl   * logit_KL
```

Resource is not just file size:

```text
R(G,C) =
  a * bytes_moved
  + b * ops
  + c * kernel_overhead
  + d * unsupported_runtime_penalty
```

This keeps the project aligned with the original product goal:

```text
reduce token-time memory traffic, not only checkpoint bytes.
```

## Hypotheses And Current Verdicts

| id | hypothesis | current verdict | next use |
| --- | --- | --- | --- |
| H1 | I2_S can faithfully run our b1.58 weights | solved | fixed substrate |
| H2 | one-shot pure b1.58 PTQ is enough | false | do not spend more on pure PTQ |
| H3 | quantizer tweaks are the main lever | false so far | keep only row/block scale as init candidates |
| H4 | content-KL improves factual retention | true but weak | default objective component |
| H5 | tiny hard factual replay generalizes | false on 1.1B | use only as diagnostic |
| H6 | representative blend should beat tiny replay | open, promising | PopQA blend 1.1B is the key next run |
| H7 | post-hoc FP layer restore fixes capacity | false | train-from-start hybrid only if needed |
| H8 | valid rotations/equalization can improve ternary fit | false data-free (RT-WSYNC-001 + H-I2S: row/group/row-norm scaling AND block-Hadamard rotation all fail at 160M -- ternary stays collapsed, FACT 0.0) | demote data-free weight-only sync (plan S4); revisit only combined with STE/adaptation |
| H9 | selective extra planes/capacity can close the remaining gap | open | PTQTP-lite after representative data is tested |
| H10 | mostly-I2_S + tiny low-rank residual can recover missing behavior | open | SIDE-001 160M rank 2/4/8 smoke |

## TurboQuant-Style Projection As Part Of `G`

The TurboQuant-like idea is not separate from the compiler. It is a concrete
`G + kernel` candidate.

For KV cache, if `R` is orthogonal:

```text
q^T k = (R q)^T (R k)
```

So the runtime can store:

```text
Q(R k)
```

and compare against:

```text
R q
```

This is attractive because the transform can make directions more isotropic before
codebook quantization.

For weights, a related I2_S-safe diagnostic is:

```text
W x = (W H^T)(H x)
```

then test:

```text
Wq_H = Q_I2S(W H^T)
yq = Wq_H (H x)
```

If this helps, it becomes a candidate runtime kernel:

```text
FWHT/RHT on activation -> I2_S matmul
```

If it only helps with a non-I2_S Lloyd/codebook quantizer, it becomes a custom
kernel idea, not a bitnet.cpp-compatible default.

## Two-Track Roadmap

The project now splits into two explicit tracks.

```text
Track A: I2_S-preserving mainline
Track B: non-I2_S exploration / upper-bound track
```

Track A is the product track. Track B is allowed to be more speculative, but it
must not silently replace Track A claims.

### Track A: I2_S-Preserving Mainline

Philosophy:

```text
Keep the final target linears as bitnet.cpp-compatible I2_S whenever possible.
Do not win by inventing a representation that loses the runtime/storage result.
```

Allowed:

| component | allowed form | why |
| --- | --- | --- |
| weights | `Wq = gamma*T`, exported through I2_S | proven faithful on x86 |
| adaptation | content-KL, answer mask, representative blend | changes behavior, not runtime format |
| data | PopQA/instruction/WikiText blend, de-leaked panels | avoids tiny replay shortcut |
| geometry | foldable equalization, signed permutation, row/scale init | may improve I2_S start |
| projection | H-I2S only if final matmul remains I2_S | keeps kernel philosophy |
| KV cache | TurboQuant-style KV compression as independent runtime add-on | does not change weight format |

Not allowed in Track A:

```text
custom non-I2_S codebooks,
extra FP layers as the main win,
Q2/Q3 pockets counted as pure b1.58,
multi-plane weights counted as one-plane I2_S,
kernel claims without x86/runtime verification.
```

Track A success criterion:

```text
Mostly I2_S artifact
+ factual/instruction behavior improves over content-KL baseline
+ i2_s ~= f16 parity
+ storage/speed advantage remains visible.
```

### Track B: Non-I2_S Exploration / Upper Bounds

Philosophy:

```text
Break the I2_S rule only to learn what capacity or geometry is missing.
```

Allowed:

| component | example | purpose |
| --- | --- | --- |
| multi-plane ternary | PTQTP-lite, two `alpha*T` planes | test capacity shortage |
| hybrid precision | Q2/Q3/FP pockets, low-rank residual | measure minimum extra capacity |
| custom codebook | Lloyd/codebook after RHT, signed-epsilon variants | test representation mismatch |
| custom kernels | fused RHT+matmul, custom ternary planes | upper-bound systems path |
| train-from-start hybrid | fixed FP/Q2 pockets during adaptation | avoid post-hoc co-adaptation break |
| architecture changes | BitNet-specific layer surgery, expanded strips/channels | long-range design candidate |

Track B success criterion:

```text
It must answer a diagnostic question:
  "what does I2_S lack?"
or produce a path that can be partially migrated back into Track A.
```

Track B failure criterion:

```text
quality improves but memory traffic/speed collapses,
or implementation requires a custom runtime with small quality gain.
```

### Promotion Rule: B -> A

A Track B idea can move into Track A only if it can be rewritten as:

```text
folded transform + I2_S
or
small runtime pre/post transform + I2_S
or
representative adaptation with unchanged I2_S weights
```

Examples:

| Track B finding | promotion path |
| --- | --- |
| dense rotation helps | try signed permutation / block-Hadamard H-I2S |
| custom Lloyd codebook helps | test whether I2_S plus RHT captures most of it |
| two planes help | identify top-saliency layers; maybe one-plane + data is enough |
| FP pocket helps | train-from-start Q2/I2_S hybrid, then measure byte trade-off honestly |

### Sidecar Candidate

The most practical Track B idea right now is:

```text
y = I2_S(x) + low_rank_residual(x)
```

or:

```text
W_side = gamma*T + B A
```

This keeps the main matrix in I2_S while adding a tiny low-rank correction. It is
not pure I2_S, but it may remain product-relevant if the sidecar is much smaller
than moving the whole model to Q2/Q3.

Plan:

```text
docs/i2s_lora_sidecar_plan.md
```

PC should test rank `2/4/8` on 160M first. Colab should only confirm on 1.1B if the
160M smoke moves FACT.

## Hardware Split: Colab vs RTX 3080 PC

### Colab / L4 / A100 Role

Use Colab for runs that need either memory, Linux/x86 runtime, or long 1.1B
training.

Track A Colab jobs:

```text
1.1B FACT-003H PopQA blend
1.1B content-KL / answer-mask variants
I2_S GGUF export + llama.cpp / bitnet.cpp runtime parity
storage/latency measurement on Linux/x86
Qwen/Gemma small audit once the 1.1B gate is positive
```

Track B Colab jobs:

```text
1.1B train-from-start hybrid
two-plane/PTQTP-lite model-level probes
Q2/Q3 pocket adaptation
larger representative data adaptation
```

Colab rules:

```text
one expensive hypothesis at a time;
Drive checkpoint and metrics logging required;
no 7B-class run before a 1.1B component works;
every run must produce JSON/MD artifacts before changing the plan.
```

### RTX 3080 PC Role

Use the 3080 box as a fast predictor and tooling machine, not as the final judge
for 1.1B quality.

Track A PC jobs:

```text
160M seed sweeps
160M PopQA/length-mix smoke
scoring pipeline validation
FACT panel rescoring / parser checks
data generation and de-leak checks
WSYNC-001/002 cheap geometry probes
Turbo projection PyTorch reference on small layers
```

Track B PC jobs:

```text
two-plane small-layer probes
random/top-k saliency checks
custom codebook reconstruction tests
H-I2S linear probes before any kernel work
small hybrid ablations on 160M
```

PC rules:

```text
use PC to kill bad branches cheaply;
do not over-trust 160M magnitude for 1.1B;
direction and failure mode matter more than exact score;
if PC result only improves weight MSE, do not launch a Colab training run yet.
```

### Parallel Execution Policy

When Colab is busy:

```text
PC should run:
  data prep,
  de-leak,
  160M predictor,
  eval pipeline checks,
  WSYNC/Turbo projection probes.
```

When PC finds a positive direction:

```text
Colab confirms it on 1.1B.
```

When Colab finds a failure:

```text
PC should reproduce the failure cheaply on 160M if useful,
then search for the next low-cost branch.
```

## Next Experiment Order

### Step 0: Close The Active 1.1B Branch

Wait for current 1.1B `mu=0.25` factual replay result.

Decision:

```text
if fact_rate > 0.185 and no memorization/degeneration:
  weak hard replay is a small auxiliary lever.
else:
  demote small hard replay and move to representative blend.
```

Do not infer too much from 160M magnitude. It is a direction predictor only.

### Step 1: FACT-003H PopQA Blend On 1.1B

This is the most important near-term run.

Why:

```text
small facts overfit;
PopQA blend has enough unique facts to behave like a distribution.
```

Pass signal:

```text
train PopQA, held-out PopQA, and FACT panel move together.
No train=1.0 / eval-flat signature.
fact_rate improves beyond content-KL baseline 0.185.
```

Fail signal:

```text
PopQA train improves but held-out/FACT stays flat,
or FACT drops while CE improves.
```

Then the issue is either objective mismatch or same-topology capacity.

### Step 2: WSYNC-001 / WSYNC-002 On 160M

Run only cheap geometry probes first:

```text
per-tensor I2_S
row-scale
diagonal equalization
signed permutation
Hadamard diagnostic
```

Goal:

```text
Does a better coordinate system improve the no-adaptation start?
```

Pass:

```text
CE improves by >= 0.5 nats or FACT improves by >= 0.05.
```

If it only improves weight MSE, keep it as analysis, not product path.

### Step 3: Turbo Projection Probe

Run two narrow probes:

```text
KV: RHT/sphere/codebook reference on collected Q/K
Weight: H-I2S linear probe Q(W H^T) Hx
```

Decision:

| result | action |
| --- | --- |
| KV improves memory/attention error | keep as independent KV-cache track |
| H-I2S improves CE/FACT | add to WSYNC init |
| only custom codebook improves | custom kernel / research-only branch |
| no gain | do not spend kernel time |

### Step 4: Saliency And Selective Capacity

If representative blend improves but still plateaus below practical quality:

```text
estimate saliency S_l from PopQA/instruction activations
rank layers/channels
compare top-k protection vs random-k
```

Then test:

```text
two-plane top layers,
row-scale top layers,
Q2/Q3 pockets,
low-rank residual pockets.
```

Do not use independent additive knapsack as the final solver. It already failed
as an assumption. Use it only as a ranking index.

### Step 5: Scale Ladder

Only scale once the component moves 1.1B behavior.

Order:

```text
160M fast predictor
-> TinyLlama 1.1B confirmation
-> Gemma/Qwen small audit
-> Qwen 7B-class goalpost
```

For Qwen/Gemma:

```text
first audit architecture and target linear names;
then one-forward replacement smoke;
then storage/runtime estimate;
then adaptation.
```

## Stop Rules

Stop or demote a branch when:

| branch | stop condition |
| --- | --- |
| small hard replay | train facts high but eval flat/worse |
| WSYNC | MSE improves but CE/FACT does not |
| rotation/projection | PyTorch reference does not beat baseline |
| custom kernel | benefit requires non-I2_S codebook but quality gain is small |
| hybrid capacity | added bytes do not improve FACT/CE under fair budget |
| 7B scale-up | 1.1B does not show the component works |

## Claim Guardrail

Allowed now:

```text
We have a faithful, fast I2_S substrate and strong evidence that b1.58 conversion
needs a compiler: valid transforms, saliency, representative adaptation, and
possibly selective capacity.
```

Not allowed yet:

```text
We can convert any useful FP model to all-I2_S b1.58 with near-Q2_K factual quality.
```

The next claim upgrade requires:

```text
1.1B representative blend or compiler component improves factual score beyond 0.185,
without losing I2_S runtime parity and without memorization signature.
```
