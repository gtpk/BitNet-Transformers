# Literature Deep Dive: Collapse Dynamics

Status: research synthesis / active.

Document position: [Index](./index.md) -> [Generation Collapse Dynamics Plan](./collapse_dynamics_research_plan.md).

## Why This Deep Dive Exists

The project's factual-recovery track found a repeated pattern:

```text
loss can improve while generation quality collapses.
```

The literature says this is not surprising. Adjacent fields rarely treat collapse
as a single final metric. They track dynamics:

```text
entropy,
confidence,
diversity,
gradient balance,
loss-landscape sharpness,
residual-stream changes,
teacher/student stability.
```

This document maps those ideas into I2_S / b1.58 conversion experiments.

## Source Map

| area | papers | what they contribute |
| --- | --- | --- |
| neural text degeneration | Holtzman et al. 2019; Fu et al. 2020 | repetition can arise from decoding/probability dynamics, not only model quality |
| self-supervised collapse | DINO; BYOL/SimSiam dynamics | no-label teacher/student needs collapse prevention, asymmetry, centering, stop-gradient |
| catastrophic forgetting | Luo et al. 2023; Li et al. 2024 | fine-tuning can erase prior knowledge; scale may worsen visible forgetting; loss landscape flatness matters |
| RLHF/DPO alignment collapse | InstructGPT; AdaDPO; FPO for iterative RLHF | KL/reference terms, gradient balance, and feedback loops are central |
| low-bit/QAT collapse | StableQAT; HiF8 QAT; DynamicPTQ | standard training loss can hide quantization failure; residual/scale dynamics must be logged |

## 1. Text Degeneration Is A Distribution/Decoding Problem

Holtzman et al., **The Curious Case of Neural Text Degeneration**
([arXiv:1904.09751](https://arxiv.org/abs/1904.09751)), showed a key mismatch:

```text
likelihood training can produce good models,
but likelihood-style decoding can produce repetitive/bland text.
```

Project translation:

```text
Do not equate CE/PPL recovery with generation recovery.
```

This matches RT-122/RT-129:

```text
greedy looked collapsed,
rep-penalty/sampling rescued non-degenerate generation.
```

### Metric To Borrow

Track decoding-time entropy and top-1 probability:

```text
if entropy drops and top1 spikes before loop_rate rises,
collapse is a decoding-confidence phenomenon.
```

Fu et al., **A Theoretical Analysis of the Repetition Problem in Text Generation**
([arXiv:2012.14660](https://arxiv.org/abs/2012.14660)), define repetition via
transition dynamics and "high inflow": many histories point back to the same
token. This suggests:

```text
loop collapse may be detected before text visibly loops,
by monitoring whether a few tokens receive excessive incoming probability mass.
```

### I2_S Experiment Translation

Add to generation probe:

```text
top1_prob
token entropy
distinct-1/2
repeated n-gram rate
average repetition probability proxy
```

If DINO improves gold rank but also increases top1 concentration:

```text
use entropy/top1 guard or curriculum schedule.
```

## 2. DINO/BYOL Collapse Prevention Is About Training Asymmetry

DINO ([arXiv:2104.14294](https://arxiv.org/abs/2104.14294)) is self-distillation
without labels. It uses teacher/student structure and view consistency; DINO's
success depends on nontrivial dynamics rather than simply minimizing a symmetric
matching loss.

Tian et al., **Understanding self-supervised Learning Dynamics without
Contrastive Pairs** ([arXiv:2102.06810](https://arxiv.org/abs/2102.06810)),
study why BYOL/SimSiam-like methods avoid trivial collapse. Their abstract
highlights:

```text
predictors,
stop-gradient,
EMA,
weight decay,
and learning dynamics
```

as active ingredients.

Project translation:

```text
DINO-I2S should not be symmetric hidden matching.
```

DINO-DIAG already supports this:

```text
dino_logit helps,
dino_hidden erases the benefit.
```

### Rule For Our Runs

Default:

```text
frozen FP teacher
student learns content logits
hidden alignment OFF
```

Only add hidden alignment if:

```text
logit DINO improves CE but not gold rank,
and hidden drift is proven by telemetry.
```

### Possible Collapse Controls

Borrow cautiously:

```text
teacher centering
teacher temperature
student temperature
stop-gradient teacher
view asymmetry
EMA teacher only after frozen-teacher smoke
```

Do not add all of these at once. They are diagnostics.

## 3. Catastrophic Forgetting Literature Explains Fluent-But-Wrong

Luo et al., **An Empirical Study of Catastrophic Forgetting in Large Language
Models During Continual Fine-tuning** ([arXiv:2308.08747](https://arxiv.org/abs/2308.08747)),
evaluate forgetting across domain knowledge, reasoning, and reading
comprehension. They report forgetting in 1B-7B LLMs and note that larger models
can show more severe visible forgetting in that range.

Project translation:

```text
1.1B failing while 160M looks harmless is plausible.
```

Our FACT-003D/H results fit:

```text
160M can show a positive mechanism,
1.1B can expose a different forgetting/collapse regime.
```

Li et al., **Revisiting Catastrophic Forgetting in Large Model Tuning**
([arXiv:2406.04836](https://arxiv.org/abs/2406.04836)), link forgetting to loss
landscape flatness and propose sharpness-aware mitigation.

Project translation:

```text
record update norm / grad norm / sharpness proxy.
```

The question is not just:

```text
which loss term?
```

It is:

```text
did the update move into a sharp basin where small generation perturbations explode?
```

### Practical Low-Cost Proxy

Full SAM may be expensive. Start with:

```text
grad_norm
update_norm
update_to_param
loss on same mini-probe before/after optimizer step
```

If collapse onset aligns with update spikes:

```text
try lower LR,
warmup,
gradient clipping,
or SAM-like small perturbation smoke.
```

## 4. RLHF/DPO Teaches Reference Anchors And Gradient Balance

InstructGPT ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) is not a
collapse paper, but it matters because RLHF keeps a reference model via KL and
mixes objectives to preserve broad behavior.

Project translation:

```text
content-KL is our reference anchor.
```

But FACT-003B taught:

```text
raw KL can copy stop behavior.
```

So the anchor must be selective:

```text
content tokens yes,
EOS/special tokens no.
```

AdaDPO ([arXiv:2605.28440](https://arxiv.org/abs/2605.28440)) argues that DPO
can have asymmetric gradient behavior: suppressing bad responses faster than it
promotes good ones. It balances gradients using model probabilities.

Project translation:

```text
measure gradient contribution per objective term.
```

If DINO or content-KL dominates CE too early:

```text
curriculum or adaptive weighting, not static lambda.
```

FPO for iterative RLHF ([arXiv:2605.04266](https://arxiv.org/abs/2605.04266))
frames alignment collapse as a feedback-loop problem where a policy exploits
blind spots and reinforces errors.

Project translation:

```text
avoid letting generated/sampled outputs become the only training distribution
without monitoring diversity and blind spots.
```

For us, this mainly warns against:

```text
self-generated factual replay without filtering.
```

## 5. Low-Bit/QAT Papers Say Training Loss Can Hide Failure

StableQAT ([arXiv:2601.19320](https://arxiv.org/abs/2601.19320)) argues that
ultra-low-bit QAT suffers from gradient mismatch/instability and proposes a
Fourier-derived surrogate for rounding.

Project translation:

```text
STE mismatch may be part of collapse dynamics.
```

We should log:

```text
ternary code flip rate,
weight residual,
gradient norm,
and update norm
```

not because EGROW flip-rate was high (it was not), but because future failures
may still show gradient mismatch at different scale/objective.

Cheng et al., **Max-Window Scale Estimation for Near-Lossless HiF8 W8A8 QAT**
([arXiv:2605.26189](https://arxiv.org/abs/2605.26189)), explicitly report two
failure modes invisible to training loss:

```text
amax saturation / forward clipping,
catastrophic forgetting from aggressive LR.
```

They use a scale history window and a warmup/low-LR schedule.

Project translation:

```text
loss can look fine while knowledge-sensitive representations are corrupted.
```

Add:

```text
activation max/rms history,
hidden variance,
scale/quantization residual history,
LR and warmup logging.
```

DynamicPTQ ([arXiv:2606.12487](https://arxiv.org/abs/2606.12487)) argues that
static smoothing is insufficient because residual-stream dynamics are phase-wise
across depth. It introduces Jump Ratio and Historical Feature SNR.

Project translation:

```text
we should inspect residual stream dynamics over layers and steps,
not only final weight reconstruction.
```

This directly supports the new collapse plan:

```text
hidden activation variance,
residual jump,
and historical signal/noise proxies should be logged.
```

## Updated Model Of Collapse

For I2_S adaptation, collapse can arise from at least four interacting channels:

```text
1. Decode distribution channel:
   entropy drops, top1 spikes, loops/empty outputs rise.

2. Knowledge-retention channel:
   gold logprob/rank falls for factual/entity tokens while CE still improves.

3. Optimization channel:
   gradient/update spikes or sharpness increase before collapse.

4. Quantization/residual channel:
   hidden variance, activation range, or residual-stream SNR shifts phase-wise.
```

So each run should be read as:

```text
objective -> training dynamics -> generation dynamics -> final score
```

not:

```text
objective -> final score
```

## Concrete Changes To Our Next Experiments

### DINO-I2S-002 Must Be Diagnostic

Do not run only:

```text
final FACT score.
```

Run:

```text
gold logprob/rank trajectory
category-level facts
entropy/top1 trajectory
generation tag trajectory
grad/update norms
hidden variance
```

### Curriculum Is Now First-Class

Try static DINO only after establishing telemetry. If collapse appears:

```text
Stage 1: content-KL only
Stage 2: content-KL + small DINO-logit
Stage 3: increase DINO only if entropy/top1 is stable
```

### Hidden Matching Is Demoted

Because DINO-DIAG found:

```text
dino_hidden erases dino_logit benefit.
```

Use hidden telemetry first. Do not use hidden loss by default.

### Entity-Attribute Gap Is A Coverage Problem

DINO-DIAG found:

```text
simple_fact improves,
entity_attr flat/worse.
```

This suggests:

```text
entity-rich unlabeled prompts,
entity-token weighted KL,
or category-specific diagnostics
```

but only after the 1.1B DINO-logit gate confirms the effect scales.

## Practical Checklist

Before launching a major run:

```text
[ ] metrics.jsonl exists
[ ] TensorBoard or equivalent scalar logs exist
[ ] generation probes saved at fixed steps
[ ] FACT category table saved
[ ] gold rank/logprob saved
[ ] f16 vs i2_s parity planned
[ ] collapse-onset rule defined before seeing final result
```

If any box is missing, the run is a score run, not a dynamics run.

## Bottom Line

The literature supports the user's reframe:

```text
collapse is usually handled through dynamics, regularization, scheduling, and
adaptive weighting, not by judging final loss alone.
```

For this project, the most useful next research claim is:

```text
I2_S generation collapse can be localized and decomposed into entropy,
confidence, gold-rank, hidden/residual, and optimizer-update dynamics.
```

That claim is more actionable than:

```text
DINO works / DINO fails.
```

