# Experiment History And Paper Opportunities

Document position: [Index](./index.md) -> compact historical map and paper-topic router.

Status: living history after TL1B-1600, PT2-I2S-001, and ANS-001 launch.

## Why This Document Exists

The project has crossed several thesis reversals. Reading only the latest status can
hide why earlier failures mattered. This document keeps the historical arc visible:

```text
what we believed,
what experiment changed it,
what remains open,
and what paper/report topic it creates.
```

Raw numbers remain in [Evidence Ledger](../reports/EVIDENCE_LEDGER.md). Current
decision state remains in [Current Experiment Status](./current_experiment_status_2026_06_30.md).

## One-Line Current Story

```text
I2_S is a strong systems substrate.
Pure gamma*T one-shot conversion fails.
Adaptation can recover fluent generation.
TL1B-1600 proves TinyLlama 1.1B generation collapse was a budget-limited transient.
The current open bottleneck is factual readout / answer format, not runtime or capacity.
```

## Phase History

### Phase 1: Systems Substrate Was Proven

Question:

```text
Can b1.58-compatible weights actually run in a real ternary runtime?
```

Result:

```text
yes, on x86/Linux bitnet.cpp I2_S.
```

Key evidence:

| run | result |
| --- | --- |
| RT-111 | official bitnet.cpp I2_S parity on x86 |
| RT-112 | our `Wq = gamma*T` Path A' parity |
| RT-113 | target-linear 16x vs f32, token-gen speedup |
| RT-114 | Llama-160M whole-file compression 0.196, tg 5.69x vs f32 |
| RT-115 | TinyLlama-1.1B whole-file compression 0.1149, tg 7.51x vs f32 |

Permanent lesson:

```text
If we can make good b1.58-compatible weights, I2_S can carry them.
```

Paper topic:

```text
Paper 1: I2_S Systems.
```

### Phase 2: Pure One-Shot `gamma*T` Conversion Failed

Question:

```text
Can existing FP models be converted like ordinary quantization?
```

Result:

```text
not with our pure gamma*T / absmean I2_S family.
```

Key evidence:

| lever | result |
| --- | --- |
| RTN / one-shot ternary | PPL/factual collapse |
| row/block scale | real but insufficient |
| AWQ/SmoothQuant-style diagonal | too small |
| GPTQ/Hessian assignment | only small fraction of the gap |
| signed-epsilon 2-bit | did not beat ternary |
| WSYNC / RHT / SIGMA / HOME | data-free geometry stayed collapsed |

Important correction after PT2-LLM:

```text
This does NOT prove "PTQ ternary is impossible".
It proves "our simple pure gamma*T one-shot path failed".
```

Paper topic:

```text
Paper 2: Conversion Limits.
```

### Phase 3: Content-KL Became The First Factual Lever

Question:

```text
If one-shot fails, can short adaptation recover behavior?
```

Result:

```text
yes for CE/fluency; partially for factual behavior.
```

Key evidence:

| run | result |
| --- | --- |
| FACT-001 | FP/Q2_K know facts; adapted I2_S is fluent but fact-poor |
| FACT-002 | data swap recovers fluency, not facts |
| FACT-003A | answer mask moves facts and fixes empty collapse |
| FACT-003B | raw KL copies EOS/stop and fails |
| FACT-003C | content-KL `lambda=0.2` gives fact 0.185, recovery 0.845, ok 27/27 |

Permanent lesson:

```text
The question is not just "how much KL".
It is "which distribution is copied".
```

Paper topic:

```text
Paper 3: Content-KL / Factual Recovery.
```

### Phase 4: Objective Add-Ons Hit A Wall

Question:

```text
Can we push factual exact above content-KL 0.185 with more objective terms?
```

Result:

```text
not yet.
```

Key evidence:

| branch | result |
| --- | --- |
| lm_head unfreeze | fluent but facts washed out |
| hard atomic replay `mu=1.0/0.25` | memorization / net-negative |
| PopQA blend | avoids tiny memorization at 160M but collapses at TinyLlama 1.1B tested runs |
| DINO 800-step | distribution signal exists but 1.1B run looked collapsed at 800 |
| sidecar / EGROW | 160M cheap capacity screens did not produce actionable behavior gain |

At this point the project could have looked dead. It was not.

### Phase 5: Collapse Became A Dynamics Problem

Question:

```text
Was 1.1B failure a hard wall or a transient that exceeded the training budget?
```

Result:

```text
budget-limited transient.
```

Key evidence:

| model/run | result |
| --- | --- |
| Pythia-160M | stable, no transient |
| Pythia-410M | transient step ~50-250, then recovery |
| Pythia-1B | transient step ~0-250, then recovery |
| TL1B-1600 | TinyLlama generation recovers by ~850-1600 |

TL1B-1600 final:

| metric | value |
| --- | --- |
| recovered_fraction | 0.806 |
| CE_adapted | 4.08 |
| degen_gap | -0.20 |
| tags | ok 27/27 |
| gold_rank | 375 |
| FACT exact | 0.111, below content-KL 0.185 |

Permanent lesson:

```text
"collapse" can be a time-local training transient.
Final loss alone is not enough; step-level telemetry is central.
```

New paper topic:

```text
Paper 5: Generation Collapse Dynamics In Low-Bit Adaptation.
```

### Phase 6: The Current Bottleneck Is Readout / Answer Format

Question:

```text
If generation is stable and gold_rank improves, why is FACT exact still low?
```

Current answer:

```text
the answer is reachable but not emitted as a short answer.
```

Evidence:

```text
TL1B-1600 gold_rank moved strongly (near step 800 ~2006 -> final 375),
but generations became fluent base-LM rambling instead of concise Q/A answers.
```

Active test:

```text
ANS-001: answer-token-weighted CE
L = CE_answer + 0.2 * content-KL + beta * CE_answer_token
beta = 4, k = 3
```

160M safety signal:

| arm | FACT | CE | tags |
| --- | ---: | ---: | --- |
| beta=0 | 0.000 | 3.955 | ok26/empty1 |
| beta=4 | 0.074 | 3.972 | ok27 |

Interpretation:

```text
beta=4 is not harmful at 160M, but 160M does not reproduce the 1.1B readout bottleneck.
The decisive test is 1.1B ANS-001.
```

New paper topic:

```text
Paper 6: Readout And Answer-Format Bottlenecks After Low-Bit Adaptation.
```

### Phase 7: PT2-LLM Became Competitor And Donor

Question:

```text
Are we behind PT2-style post-training ternarization?
```

Current answer:

```text
PT2 is a real competitor, but also a donor.
```

Important correction:

```text
PT2 exact may win as a model.
But the product question is whether PT2 projected-I2_S or PT2-init+adapt sits on
the Pareto frontier.
```

First smoke:

| arm | result |
| --- | --- |
| PT2-I2S-001 ITF by weight-MSE | lowers MSE but worsens behavior/gold-rank |

Lesson:

```text
weight reconstruction alone is still the wrong objective.
PT2-I2S-002 activation-aware grid alignment is the next meaningful PT2 smoke.
```

New paper topic:

```text
Paper 7: PT2 Meets I2_S -- Ternary PTQ As Initializer, Competitor, And Runtime Tradeoff.
```

## Major Thesis Revisions

| old belief | new belief | evidence |
| --- | --- | --- |
| I2_S runtime may be the blocker | runtime is solved on x86/Linux | RT-111..115 |
| PTQ ternary is impossible | pure `gamma*T` PTQ failed; PT2-style remains open | RT-124..127 + PT2 |
| quantizer tweaks are the lever | simple tweaks are too small | RT-124..127 |
| 1.1B is hard collapse | 800-step failure was budget-limited transient | TL1B-1600 |
| DINO failure means no signal | DINO moves gold probability/rank but may miss exact readout | DINO-DIAG + TL1B |
| capacity is the next obvious fix | current bottleneck is answer readout/format | TL1B-1600 |

## New Paper / Report Opportunities

### Paper 1: I2_S Systems

Status:

```text
strongest / most ready.
```

Claim:

```text
b1.58-compatible weights can be exported to bitnet.cpp I2_S with faithful runtime,
scaling storage reduction, and growing token-generation speedup.
```

### Paper 2: Conversion Limits

Status:

```text
ready, but claim must exclude PT2-style asymmetric PTQ.
```

Claim:

```text
same-shape pure gamma*T conversion is not ordinary quantization.
Simple PTQ tricks are insufficient at b1.58.
```

### Paper 3: Content-KL Objective

Status:

```text
good tech-report/workshop story.
```

Claim:

```text
factual recovery is objective-sensitive; content-KL fixes raw-KL stop-copying and
is the first stable factual lever.
```

### Paper 5: Collapse Dynamics

Status:

```text
new and promising.
```

Claim:

```text
generation collapse during low-bit adaptation is a dynamic transient, not only a
final-score event. Pythia and TL1B show consolidation timing can decide whether a
run appears failed.
```

Needed:

```text
clean plots: degen_gap, gold_rank, entropy/top1, hidden_var, train_ce vs step.
```

### Paper 6: Readout / Answer-Format Bottleneck

Status:

```text
active, depends on ANS-001/002.
```

Claim candidate:

```text
after low-bit adaptation, factual tokens can become reachable before they become
emittable. Answer-token or format-aware objectives target that gap.
```

Needed:

```text
ANS-001 1.1B final,
ANS-002 if ANS-001 is flat,
first-token / gold-rank / exact-answer analysis.
```

### Paper 7: PT2-I2S Model Competition

Status:

```text
new comparative track.
```

Claim candidate:

```text
PT2-style asymmetric ternary is a strong competitor and a useful initializer donor,
but final value depends on whether gains survive I2_S projection or justify a cheap
mu correction.
```

Needed:

```text
PT2-I2S-002 activation-aware output-error smoke,
PT2 exact vs projected-I2_S vs ours adapted scorecard,
runtime/size/cost Pareto table.
```

## Immediate Action Map

| priority | action | why |
| --- | --- | --- |
| 1 | finish ANS-001 1.1B | directly tests current readout bottleneck |
| 2 | if ANS-001 passes, update Paper 6 and model scorecard | turns TL1B into factual-readout story |
| 3 | if ANS-001 is flat, run ANS-002 short-answer format curriculum | next direct treatment for rambling |
| 4 | run PT2-I2S-002 on PC | tests PT2 as initializer/donor after ITF-only failed |
| 5 | build final Pareto comparison table | avoids arguing by hidden metric |

## Guardrails

Do not claim:

```text
all-I2_S has factual parity with Q2_K or FP.
DINO solves factuality.
PT2 is beaten before PT2-I2S-002/scorecard exists.
TinyLlama collapse is impossible.
```

Do claim:

```text
I2_S systems are strong.
pure gamma*T one-shot conversion fails.
generation collapse can be a budget-limited transient.
the current bottleneck is factual readout / answer format.
final comparison must be model-artifact Pareto comparison.
```
