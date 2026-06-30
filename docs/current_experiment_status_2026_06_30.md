# Current Experiment Status -- 2026-06-30

Status: latest synthesis after TL1B-1600 final.

Document position: [Index](./index.md) -> current checkpoint.

## One-Line State

```text
I2_S systems are solved, one-shot conversion is false, and TinyLlama-1.1B
generation collapse was a budget-limited transient rather than a hard impossibility.
The remaining gap is factual readout / answer-format quality.
```

## What Is Solved

| axis | status | evidence |
| --- | --- | --- |
| I2_S runtime/export | solved on x86/Linux | Path A' `Wq=gamma*T` -> GGUF -> I2_S parity |
| storage | solved | target linears are 16x smaller than f32, 8x vs f16 |
| token generation speed | solved for dense LLaMA family | speedup grows with scale; TinyLlama 1.1B showed ~7.5x vs f32 |
| pure one-shot I2_S PTQ | false | `gamma*T` PTQ collapses; simple quantizer tweaks cannot rescue behavior |
| PT2-style ternary PTQ | newly open | asymmetric `mu + alpha*T`, AGA, SSR were not covered by our negative PTQ track |
| short/medium adaptation | works for CE/fluency | TL1B-1600 recovers generation stability and CE after the 800-step transient |
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
| PopQA blend on TinyLlama 1.1B | generation collapse in tested runs |
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

TL1B-1600 adds the 1.1B interpretation:

```text
DINO-logit does not produce factual exact-match win on TinyLlama-1.1B,
but the 800-step collapse was a budget-limited transient.
At 1600 steps, generation is stable and CE recovers, while factual exact still lags.
```

Final TL1B-1600 numbers:

| metric | result |
| --- | --- |
| recovered_fraction | 0.806 |
| CE_adapted | 4.08 |
| final degen_gap | -0.20 |
| final gold_rank | 375 (from ~2006 near step 800) |
| generation tags | ok 27/27 |
| FACT exact | 0.111, below content-KL 0.185 |

Interpretation:

```text
generation stability recovered;
factual answer readout / Q-A format did not.
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
| TinyLlama-1.1B | unresolved at 800; recovers generation by 1600 | stable generation, weak factual readout |

Revised thesis:

```text
~1B scale itself is not the collapse wall.
Pythia-1B recovers within 800 steps.
TinyLlama-1.1B needs longer budget for generation recovery, but still has a
factual readout / answer-format gap.
```

The best current model of the dynamics:

```text
scale/model complexity can lengthen a degenerate transient.
What looked like collapse can be a transient that does not resolve within the
short training budget.
```

Pythia-1B is the key correction: it shows that the transient can still resolve at
roughly the TinyLlama scale.

## Current Decision Point

TL1B-1600 changes the DINO/TinyLlama interpretation, while PT2-LLM changes the
initializer roadmap.

No longer the right first question:

```text
TinyLlama-1.1B longer-budget run (1600 steps first, extend to 2400 only if
trajectory suggests recovery).
```

It has been answered:

```text
1600 steps recover generation stability but not factual exact answers.
```

Current first experiment:

```text
PT2-lite I2_S initializer smoke.
```

Reason:

```text
If PT2-style ITF/AGA/SSR starts closer to FP, it may shorten the degenerate
transient and improve the starting ternary code before more long runs.
```

So the revised order is:

```text
1. PT2-I2S-001/002 on PC: ITF + activation-aware alpha-only projection.
2. If pure I2_S projection helps behavior, use it as the new TinyLlama initializer.
3. If only mu+alpha*T helps, evaluate the I2_S-rooted mu correction sidecar.
4. If PT2-lite does not help, focus on answer-format/readout objectives rather
   than simply extending to 2400.
```

Lower priority:

```text
TinyLlama-1.1B 2400-step extension.
```

Question:

```text
Can the model turn improved gold_rank into short factual answers instead of
fluent base-LM rambling?
```

Decision table:

| next result | interpretation | next |
| --- | --- | --- |
| FACT exact rises above 0.185 while tags stay ok | DINO/readout recipe beats content-KL | promote recipe |
| gold_rank improves but FACT remains low | readout/answer-format bottleneck | answer-token-weighted or format-aware objective |
| tags regress again | stability bottleneck | entropy/top1 guard or staged schedule |
| PT2 projected-I2_S improves old init | better initializer | rerun shorter TinyLlama schedule |

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
TinyLlama-1.1B generation collapse at 800 steps was a budget-limited transient.
```

Not allowed:

```text
all-I2_S preserves factual assistant quality.
content-KL/DINO solves factuality.
TinyLlama generation collapse is a hard impossibility.
```
