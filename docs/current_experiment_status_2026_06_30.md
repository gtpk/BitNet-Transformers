# Current Experiment Status -- 2026-06-30

Status: latest synthesis after PYTHIA-LADDER pythia-1b.

Document position: [Index](./index.md) -> current checkpoint.

## One-Line State

```text
I2_S systems are solved, one-shot conversion is false, adaptation can recover CE
and fluent generation, but factual retention depends on model-specific training
dynamics rather than a simple 1B scale wall.
```

## What Is Solved

| axis | status | evidence |
| --- | --- | --- |
| I2_S runtime/export | solved on x86/Linux | Path A' `Wq=gamma*T` -> GGUF -> I2_S parity |
| storage | solved | target linears are 16x smaller than f32, 8x vs f16 |
| token generation speed | solved for dense LLaMA family | speedup grows with scale; TinyLlama 1.1B showed ~7.5x vs f32 |
| pure one-shot I2_S PTQ | false | `gamma*T` PTQ collapses; simple quantizer tweaks cannot rescue behavior |
| PT2-style ternary PTQ | newly open | asymmetric `mu + alpha*T`, AGA, SSR were not covered by our negative PTQ track |
| short adaptation | works for CE/fluency | CE/PPL and non-degenerate generation recover under content-KL style recipes |
| I2_S runtime faithfulness after adaptation | solved | i2_s tracks adapted f16; collapse is model/training, not runtime |

## What Failed

The strong original product claim is not proven:

```text
existing FP model -> all-I2_S -> almost same factual assistant
```

Failed or demoted branches:

| branch | result |
| --- | --- |
| one-shot pure I2_S PTQ / RTN | collapse |
| scale/threshold/GPTQ/AWQ/signed-eps | too small; simple quantizer tweaks are not the main lever |
| WSYNC / H-I2S / SIGMA / RHT | data-free geometry cannot leave the collapsed regime |
| small hard factual replay | memorizes train facts; held-out/eval drops |
| PopQA blend on TinyLlama 1.1B | generation collapse |
| lm_head unfreeze | more forgetting |
| post-hoc FP restore | breaks all-ternary co-adaptation |
| sidecar / EGROW targeted sidecar at 160M | no actionable behavior gain |
| hidden-state DINO | overconstrains; erases logit-DINO benefit |

## DINO Reinterpretation

DINO-logit should not be read only by exact-match.

DINO-DIAG showed:

```text
gold logprob rises,
gold rank improves,
simple_fact benefits strongly,
entity_attr remains weak.
```

So DINO-logit is a real distribution-level retention mechanism. Its failure mode
is not "no signal"; it is:

```text
signal does not always reach rank-1 exact generation, especially for entity
attribute facts and some larger/model-specific regimes.
```

Hidden alignment is currently demoted:

```text
dino_logit positive,
dino_hidden flat/worse.
```

## Collapse Dynamics Reframe

The active research question is no longer:

```text
Which objective wins at final step?
```

It is:

```text
When does generation collapse begin,
how long is the degenerate transient,
and does the model consolidate before the training budget ends?
```

Required telemetry for major runs:

```text
degenerate_rate,
gold_rank / gold_logp,
entropy / top1,
hidden_var_mid / hidden_var_last,
grad_norm / update_norm,
teacher-relative degen_gap and gold_rank_ratio.
```

## Pythia Ladder Result So Far

Pythia controls model family/tokenizer/pretraining better than TinyLlama-vs-small-model comparisons.

| rung | transient | result |
| --- | --- | --- |
| Pythia-160M | none | stable |
| Pythia-410M | step ~50-250, then recovery | stable |
| Pythia-1B | step ~0-250, recovery by ~225-300 | stable |
| TinyLlama-1.1B | unresolved within 800 steps | collapse |

Revised thesis:

```text
~1B scale itself is not the collapse wall.
Pythia-1B recovers under the same recipe.
TinyLlama-1.1B collapse is model-specific or schedule-specific.
```

The best current model of the dynamics:

```text
scale/model complexity can lengthen a degenerate transient.
Collapse happens when the transient does not resolve within the training budget.
```

Pythia-1B is the key correction: it shows that the transient can still resolve at
roughly the TinyLlama scale.

## Current Decision Point

PT2-LLM changes the next-step ordering.

Old next experiment:

```text
TinyLlama-1.1B longer-budget run (1600 steps first, extend to 2400 only if
trajectory suggests recovery).
```

New first experiment:

```text
PT2-lite I2_S initializer smoke.
```

Reason:

```text
If PT2-style ITF/AGA/SSR starts closer to FP, it may shorten the degenerate
transient before we spend another long TinyLlama run.
```

So the revised order is:

```text
1. PT2-I2S-001/002 on PC: ITF + activation-aware alpha-only projection.
2. If pure I2_S projection helps behavior, use it as the new TinyLlama initializer.
3. If only mu+alpha*T helps, evaluate the I2_S-rooted mu correction sidecar.
4. If PT2-lite does not help, return to TinyLlama longer-budget.
```

Still valuable, but second in line:

```text
TinyLlama-1.1B longer-budget run (1600 steps first, extend to 2400 only if
trajectory suggests recovery).
```

Question:

```text
Is TinyLlama a hard model-specific collapse,
or does it simply need a longer consolidation schedule?
```

Decision table:

| TinyLlama longer-budget result | interpretation | next |
| --- | --- | --- |
| degen_gap -> 0 and gold_rank improves | original product path partially reopens | schedule/curriculum recipe |
| gold_rank improves but degen stays high | readout/generation stabilization issue | entropy/top1 guard, decoding-aware schedule |
| gold_rank does not improve | model-specific hard collapse | chat-vs-base ablation, different base |
| worsens | objective mismatch | stop DINO/FACT for TinyLlama |

Secondary options:

```text
Pythia-1.4B / 2.8B for academic ladder completeness.
TinyLlama chat-vs-base ablation to isolate chat tuning.
Qwen/Gemma base ladder for product direction.
```

## Current Claim Discipline

Allowed:

```text
I2_S is a strong systems substrate.
One-shot b1.58 conversion fails.
Adaptation can recover CE/fluency.
DINO-logit moves factual probability mass.
Generation collapse is a dynamic transient/consolidation phenomenon.
Pythia shows no generic 1B scale wall.
```

Not allowed:

```text
all-I2_S preserves factual assistant quality.
content-KL/DINO solves factuality.
TinyLlama collapse proves all 1B models collapse.
```
