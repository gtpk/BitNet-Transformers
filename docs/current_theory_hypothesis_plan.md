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

The final evaluation question is:

```text
Under the same user constraint, which deployed model is better?
```

So I2_S purity is not the final metric. It is a product constraint and an explanation
for speed/storage. The final comparison is Pareto-style:

```text
quality + speed + size + post-training cost + runtime support.
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
| pure one-shot I2_S PTQ | failed | `gamma*T` PPL/factual collapse |
| simple quantizer tweaks | mostly ruled out | scale/objective/AWQ/GPTQ/signed-eps all too small |
| PT2-style asymmetric PTQ | newly open | `mu + alpha*T`, AGA, SSR not covered by our negative PTQ track |
| CE/PPL recovery | works | short target-linear adaptation recovers large CE fraction |
| decoding usability | rescued | repetition penalty/sampling avoids greedy attractors |
| factual quality | still open | content-KL/DINO move distribution but do not yet preserve assistant-level facts |
| post-hoc capacity restore | failed | late FP restore worsens behavior after all-ternary co-adaptation |
| representative blend | failed for TinyLlama 1.1B | PopQA blend avoids 160M memorization but collapses at TinyLlama 1.1B |
| scale/model collapse | revised | Pythia-160M/410M/1B recover; TinyLlama-1.1B generation also recovers by 1600, but factual readout stays weak |

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

### T3b. PT2-LLM Narrows The Negative PTQ Claim

The old shortcut was:

```text
PTQ ternary conversion failed.
```

This is now too broad. The correct statement is:

```text
pure gamma*T I2_S PTQ failed.
```

PT2-LLM adds a different local family:

```text
Wq = mu + alpha*T
```

with iterative fitting, activation-aware grid alignment, and structural column
reordering. That family was not tested by RT-124..127. Therefore PT2-style fitting
is now a first-class branch:

```text
PT2-lite init
  -> project to pure I2_S if possible
  -> otherwise measure a minimal mu correction as I2_S-rooted auxiliary capacity
```

The product guardrail remains:

```text
pure I2_S wins only if the gain survives Wq = gamma*T.
mu correction is allowed only as an explicitly labeled auxiliary branch.
```

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
| H2 | one-shot pure b1.58/I2_S PTQ (`gamma*T`) is enough | false | do not spend more on naive pure PTQ |
| H3 | simple quantizer tweaks are the main lever | false so far | keep only row/block scale as weak init candidates |
| H4 | content-KL improves factual retention | true but weak | default objective component |
| H5 | tiny hard factual replay generalizes | false on 1.1B | use only as diagnostic |
| H6 | representative blend should beat tiny replay | open, promising | PopQA blend 1.1B is the key next run |
| H7 | post-hoc FP layer restore fixes capacity | false | train-from-start hybrid only if needed |
| H8 | valid rotations/equalization can improve ternary fit | false data-free (RT-WSYNC-001 + H-I2S: row/group/row-norm scaling AND block-Hadamard rotation all fail at 160M -- ternary stays collapsed, FACT 0.0) | demote data-free weight-only sync (plan S4); revisit only combined with STE/adaptation |
| H9 | selective extra planes/capacity can close the remaining gap | open | only as I2_S-rooted auxiliary capacity, not a replacement |
| H10 | I2_S-rooted tiny low-rank residual can recover missing behavior | open | SIDE-001 160M rank 2/4/8 smoke |
| H11 | layers that repeatedly flip ternary states during STE reveal local I2_S capacity bottlenecks | flip part false; sensitivity locator true at 160M (EGROW-001: top-8 overlap 7/8, but flip_rate ~= 0; residual x saliency drives ranking) | EGROW-002: top-k sensitive layers vs random-k sidecar |
| H12 | 160M cheap geometry/capacity probes can identify the next product lever | mostly exhausted / negative (WSYNC, H-I2S, SIDE-001, EGROW-002); EGROW-004/005 are gated off, not run | wait for FACT-003H; reopen 1.1B capacity only if representative data plateaus or a new locator/growth action appears |
| H13 | natural-system analogies can produce new I2_S-rooted smoke candidates | open | start with RDT-001 ledger, HOME-001 homeostasis, then SIGMA-001/RHT-002 references; no Colab until PC smoke passes |
| H14 | DINO-style no-label self-distillation can preserve base factual behavior during I2_S conversion | partially true | logit-DINO moves gold logprob/rank; hidden alignment overconstrains; exact-match depends on model/schedule |
| H15 | generation collapse is a dynamic phenomenon, not a final-score event | true / active | Pythia shows recoverable degenerate transients; log teacher-relative degen_gap/gold_rank_ratio |
| H16 | ~1B model scale itself causes collapse | false | Pythia-1B is stable; TinyLlama-1.1B generation recovers by 1600 |
| H17 | TinyLlama-1.1B collapse may be an unresolved transient, not hard impossibility | true for generation / false for factual exact | do not call 800-step failure hard collapse; next bottleneck is readout/format |
| H18 | PT2-style asymmetric fitting can shorten the transient by improving the ternary initializer | open / high-priority | run PT2-I2S-001..005 as initializer/competitor, not as emergency rescue |
| H19 | factual knowledge is reachable but not decoded into concise answers | open / supported by TL1B-1600 | gold_rank 375 but FACT 0.111 -> test answer-token/format-aware readout objectives |

## Collapse Dynamics Reframe

The project question is shifting from:

```text
Is this objective good?
```

to:

```text
When and why does this objective trigger generation collapse?
```

The core warning from FACT and DINO diagnostics is:

```text
loss can improve while generation quality collapses.
```

Therefore final PPL/fact scores are not enough. Future DINO/FACT runs should
emit step-level telemetry:

```text
total loss,
content KL,
DINO loss,
gradient norm,
parameter update norm,
hidden activation variance,
logit entropy,
top-1 probability,
gold logprob/rank,
salad/empty/loop rates.
```

The goal is to locate the collapse onset:

```text
the first training window where generation tags degrade together with entropy /
confidence / rank dynamics.
```

Detailed plan:

```text
docs/collapse_dynamics_research_plan.md
```

## DINO-I2S As Objective Branch

DINO-I2S is not a new precision format and not a non-I2_S track. It is an
adaptation objective:

```text
I2_S student follows a frozen FP/base teacher on broad unlabeled text.
```

The reason to keep it on the table is that FACT-003D showed the failure mode of
small hard factual replay:

```text
train facts go up,
held-out/eval facts do not,
so the model found a memorization shortcut.
```

DINO-style self-distillation replaces the table-memorization pressure with:

```text
preserve content logits + selected hidden geometry of the base model.
```

First-smoke objective:

```text
L =
  L_answer_CE
  + lambda_c * KL_content(p_teacher || p_student)
  + beta_h * hidden_alignment
```

The project already learned one important constraint:

```text
raw KL can copy EOS/stop behavior and cause empty answers.
```

Therefore DINO-I2S must use content-only KL:

```text
V_content = V \ {EOS, BOS, PAD, special/control tokens}
```

TL1B-1600 adds the current DINO boundary:

```text
1600 steps recover generation stability and CE,
but factual exact answers remain below content-KL baseline.
```

Observed signature:

```text
degen_gap -> clean,
gold_rank improves strongly,
FACT exact remains low,
generations become fluent base-LM rambling instead of short Q/A answers.
```

Therefore DINO is no longer mainly a "collapse rescue" objective. Its next role was
initially framed as:

```text
move factual probability mass, then combine with answer-format/readout pressure
so the model actually emits the reachable answer.
```

Qwen-1.5B RFIT adds a newer and more specific role:

```text
DINO as anti-overfit consistency regularizer.
```

The Qwen-1.5B minimal content-KL recipe undertrained at 800 but over-trained by
1600: FACT rose from 0.111 to 0.222, yet eval CE worsened from 4.85 to 5.68 while
train CE collapsed to 0.37. This suggests weak late logit-DINO may be useful not
as a direct factual booster, but as a brake against answer-CE stream memorization.

Detailed plan:

```text
docs/qwen_rfit_dino_anti_overfit_plan.md
```

Detailed plan:

```text
docs/dino_i2s_self_distillation_plan.md
```

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

If it only helps with a Lloyd/codebook quantizer that cannot be rewritten as an
I2_S-rooted branch, it becomes an upper-bound diagnostic, not a bitnet.cpp-compatible
default.

## I2_S Trunk And Auxiliary Branches

The project should not drift away from I2_S. The correct mental model is:

```text
I2_S trunk
  -> I2_S-preserving branches
  -> bounded auxiliary branches
  -> upper-bound diagnostics, clearly labeled
```

The trunk is always the current product philosophy. Branches are allowed only if
they explain, protect, or minimally extend the I2_S artifact. They must not silently
become a different quantization project.

### Trunk: I2_S-Preserving Mainline

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

Not allowed on the trunk:

```text
custom non-I2_S codebooks,
extra FP layers as the main win,
Q2/Q3 pockets counted as pure b1.58,
multi-plane weights counted as one-plane I2_S,
kernel claims without x86/runtime verification.
```

Trunk success criterion:

```text
Mostly I2_S artifact
+ factual/instruction behavior improves over content-KL baseline
+ i2_s ~= f16 parity
+ storage/speed advantage remains visible.
```

### Auxiliary Branches: Still Rooted In I2_S

Philosophy:

```text
Keep the I2_S base as the main carrier, then add the smallest possible helper.
```

Allowed:

| component | example | purpose |
| --- | --- | --- |
| tiny sidecar | `gamma*T + BA`, low-rank residual | compensate missing modes while keeping I2_S base |
| selected extra plane | second ternary plane on top-saliency layers | add capacity locally, not globally |
| selected Q2/Q3 pocket | very small top-saliency pocket | measure minimum auxiliary precision without replacing the I2_S trunk |
| I2_S-compatible transform | folded scale, signed permutation, H-I2S if useful | improve the I2_S projection |
| KV branch | TurboQuant-style KV compression | reduce runtime memory outside weights |

Auxiliary branch success criterion:

```text
It must improve behavior while I2_S remains the trunk:
  most target-linear bytes stay I2_S,
  sidecar/pocket overhead is small and reported,
  speed/storage still beat broad Q2/Q3/Q4 alternatives.
```

Auxiliary branch failure criterion:

```text
quality improves only by letting the auxiliary branch carry most behavior,
or overhead approaches Q2/Q3 everywhere,
or runtime advantage collapses.
```

### Upper-Bound Diagnostics

Some ideas do not belong to the current I2_S trunk. Keep them as diagnostics or
literature upper bounds unless they can be rewritten back into an I2_S-rooted form:

```text
dense learned rotation,
large sidecar rank,
full Q2/Q3 restore,
custom Lloyd/codebook kernels.
```

These do not become the product path unless they can be rewritten as:

```text
I2_S trunk + small auxiliary branch
```

Examples:

| diagnostic finding | I2_S-rooted rewrite |
| --- | --- |
| dense rotation helps | try signed permutation / block-Hadamard H-I2S |
| custom Lloyd codebook helps | test whether I2_S plus RHT captures most of it |
| two planes help | restrict extra plane to top-saliency layers |
| FP pocket helps | shrink to sidecar / Q2 pocket / low-rank residual |

### Sidecar Candidate

The most practical auxiliary branch right now is:

```text
y = I2_S(x) + low_rank_residual(x)
```

or:

```text
W_side = gamma*T + B A
```

This keeps the main matrix in I2_S while adding a tiny low-rank correction. It is
not all-I2_S, but still I2_S-rooted. It remains product-relevant only if the sidecar
stays much smaller than moving the whole model to Q2/Q3.

Plan:

```text
docs/i2s_lora_sidecar_plan.md
```

PC should test rank `2/4/8` on 160M first. Colab should only confirm on 1.1B if
the 160M smoke moves FACT without letting the sidecar dominate bytes/ops.

### Entropy-Guided Growth Candidate

The sidecar should not blindly attach everywhere. A better I2_S-rooted version is:

```text
monitor where the ternary model cannot settle,
then grow the smallest auxiliary branch only there.
```

The signal is not raw entropy. Raw entropy can mean healthy learning, high
learning rate, data noise, or true capacity shortage. The proposed bottleneck
score is:

```text
B_l =
  instability_l
  * output_residual_l
  * task_saliency_l
```

where:

```text
instability_l =
  temporal ternary entropy
  + ternary flip rate
  + gradient conflict
  + update reversal rate
```

Only layers with high `B_l` are allowed to receive an auxiliary branch:

```text
I2_S trunk + rank-2/4 sidecar
I2_S trunk + selected second ternary plane
I2_S trunk + tiny top-k Q2/Q3 pocket as diagnostic
```

This turns the user's "the layer keeps going back and forth" observation into a
testable rule:

```text
oscillation alone is not capacity shortage;
oscillation + residual + FACT saliency + held-out gain is.
```

Plan:

```text
docs/entropy_guided_i2s_growth_plan.md
```

## Hardware Split: Colab vs RTX 3080 PC

### Colab / L4 / A100 Role

Use Colab for runs that need either memory, Linux/x86 runtime, or long 1.1B
training.

I2_S trunk Colab jobs:

```text
1.1B FACT-003H PopQA blend
1.1B content-KL / answer-mask variants
I2_S GGUF export + llama.cpp / bitnet.cpp runtime parity
storage/latency measurement on Linux/x86
Qwen/Gemma small audit once the 1.1B gate is positive
```

I2_S-rooted auxiliary Colab jobs:

```text
1.1B I2_S + tiny sidecar confirmation
I2_S-rooted two-plane/PTQTP-lite model-level probes
small top-saliency Q2/Q3 pocket adaptation
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

Closed PC branches are tracked here:

```text
docs/pc_negative_branch_map.md
```

I2_S trunk PC jobs:

```text
160M seed sweeps
160M PopQA/length-mix smoke
scoring pipeline validation
FACT panel rescoring / parser checks
data generation and de-leak checks
WSYNC-001/002 cheap geometry probes
Turbo projection PyTorch reference on small layers
```

I2_S-rooted auxiliary / diagnostic PC jobs:

```text
SIDE-001 160M rank 2/4/8 smoke
SIDE-002 sidecar co-adaptation smoke
EGROW-001 160M layer-instability logger
EGROW-002 top-k sidecar by bottleneck_score vs random-k
two-plane small-layer probes
random/top-k saliency checks
custom codebook reconstruction tests only as an upper-bound diagnostic
H-I2S linear probes before any kernel work
small hybrid ablations on 160M
```

PC rules:

```text
use PC to kill bad branches cheaply;
do not over-trust 160M magnitude for 1.1B;
direction and failure mode matter more than exact score;
if PC result only improves weight MSE, do not launch a Colab training run yet.
if WSYNC/SIDE/EGROW-002 already killed a PC branch, do not reopen it without a
new 1.1B result or a genuinely new mechanism. EGROW-004/005 are conditional
stages, not failed stages.
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
  RDT-001 cost ledger,
  HOME-001 activation-homeostasis smoke if FACT-003H is still running.
  SIGMA-001 residual-feedback reference,
  RHT-002 dithered/randomized Hadamard reference.
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
| custom kernel | benefit requires a non-I2_S-rooted codebook but quality gain is small |
| I2_S-rooted auxiliary capacity | added bytes do not improve FACT/CE under fair budget |
| 7B scale-up | 1.1B does not show the component works |

## Claim Guardrail

Allowed now:

```text
We have a faithful, fast I2_S substrate and strong evidence that b1.58 conversion
needs a compiler: valid transforms, saliency, representative adaptation, and
possibly I2_S-rooted auxiliary capacity.
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
