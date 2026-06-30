# Paper Series Plan

Document position: [Index](./index.md) -> publication/report roadmap after the project
split into multiple papers.

Related:

- [Paper 1: I2_S Systems](./paper_1_i2s_systems.md)
- [Paper 2: Conversion Limits](./paper_2_conversion_limits.md)
- [Paper 3: Content-KL Factual Recovery](./paper_3_content_kl_factual_recovery.md)
- [Paper 4: Hybrid Capacity Candidate](./paper_4_hybrid_capacity_candidate.md)
- [Paper 5: Collapse Dynamics](./paper_5_collapse_dynamics.md)
- [Paper 6: Factual Readout / Answer Format](./paper_6_factual_readout_answer_format.md)
- [Paper 7: PT2-I2S Model Competition](./paper_7_pt2_i2s_model_competition.md)
- [Paper Draft Manuscripts Index](./paper_draft_manuscripts_index.md)
- [Named Rules And Principles](./paper_named_rules.md)
- [Paper Evidence Matrix](./paper_evidence_matrix.md)
- [Literature Positioning Map](./literature_positioning_map.md)
- [Fair Comparison Framework](./fair_comparison_framework.md)
- [Paper Draft](./paper_draft.md)
- [Paper Skeleton](./paper_skeleton.md)

## Why Split The Work

The project no longer fits one clean paper. We now have at least seven separable
stories:

```text
1. The runtime/export systems path works.
2. Same-shape one-shot b1.58 conversion fails for principled reasons.
3. Factual recovery needs the right objective; content-KL is the first working lever.
4. If content-KL plateaus, the product path may need variable/hybrid capacity.
5. Generation collapse is a dynamic transient; 800-step TinyLlama failure was not final.
6. Factual tokens can be reachable internally but not emitted as concise answers.
7. PT2-style ternary PTQ is both a competitor and a donor for better I2_S initializers.
```

Forcing all four into one paper would blur the strongest results and make the open
parts look like weaknesses in the solved parts. The correct structure is a paper
series.

## Series Map

| paper | working title | status | main claim | should not claim |
| --- | --- | --- | --- | --- |
| 1 | I2_S Systems | nearly ready | b1.58 weights can be faithfully exported and run with scaling storage/speed gains | quality parity |
| 2 | Conversion Limits | nearly ready, PT2 caveat added | one-shot/same-shape `gamma*T` b1.58 conversion fails; simple quantizer tricks are not enough | all PTQ ternary is impossible |
| 3 | Content-KL Factual Recovery | active | factual gap is objective-sensitive; content-KL fixes raw-KL EOS failure and moves facts | Q2_K/FP parity |
| 4 | Hybrid Capacity | candidate | mostly-I2_S plus selective capacity may be the product path if content-KL plateaus | evidence before HYBRID-001 |
| 5 | Collapse Dynamics | new strong topic | low-bit adaptation collapse is a transient/consolidation phenomenon; schedule decides apparent failure | DINO solves factuality |
| 6 | Factual Readout / Answer Format | active, ANS-001 pending | facts may be reachable by rank but not emitted as concise answers | solved before ANS results |
| 7 | PT2-I2S Model Competition | new comparative track | PT2 exact/projected/adapted variants must be compared as final artifacts | ours wins before scorecard |

The current `paper_draft.md` is an integrated historical draft. It should stop
accumulating new results. New work should be routed into the right paper skeleton
and then, once stable, into the matching short manuscript draft listed in
[Paper Draft Manuscripts Index](./paper_draft_manuscripts_index.md).

Use [Paper Evidence Matrix](./paper_evidence_matrix.md) as the canonical table of
known results and blank cells. Individual papers may summarize those numbers, but
the matrix is the first place to update when a new run completes.

Use [Named Rules And Principles](./paper_named_rules.md) as the shared vocabulary
for the empirical laws discovered across the project. Individual drafts may cite
the rule names, but this document is the canonical place to define or rename them.

Use [Literature Positioning Map](./literature_positioning_map.md) to decide which
external papers are direct competitors, which ideas should be borrowed, and where our
claims must be narrowed.

Use [Experiment History And Paper Opportunities](./experiment_history_and_paper_opportunities.md)
to understand why the paper list expanded and which result caused each thesis revision.

## Shared Scorecard

Every paper that compares methods must use the same fair axes:

```text
pretraining tokens / pretraining cost
post-training tokens / post-training GPU time
parameter count
storage size
token-generation speed
PPL / CE
factual score
generation behavior
runtime parity
```

The canonical table is [Fair Comparison Framework](./fair_comparison_framework.md).

## Paper 1: I2_S Systems

Use [Paper 1: I2_S Systems](./paper_1_i2s_systems.md).

Core evidence:

```text
RT-111 official bitnet.cpp x86 parity
RT-112 our Path A' parity
RT-113 storage/latency
RT-114 Llama-160M scale-up
RT-115 TinyLlama-1.1B scale-up
Mac M5 negative as runtime/toolchain caveat
```

Key hint we found:

```text
Do not send latent FP weights directly to upstream I2_S.
Materialize Wq = gamma*T first.
Then max(abs(Wq)) = gamma, so upstream absmax quantization preserves our scale.
```

Possible conclusion:

```text
The systems substrate is ready: b1.58-compatible weights can use existing bitnet.cpp
I2_S without custom writer/kernel on x86.
```

## Paper 2: Conversion Limits

Use [Paper 2: Conversion Limits](./paper_2_conversion_limits.md).

Core evidence:

```text
RT-121 Q2_K beats ours on PPL
RT-123 additive mixed-bit DP assumption is weak
RT-124 scale granularity only partially helps
RT-125 GPTQ/Hessian assignment recovers only a small fraction
RT-127 signed-epsilon 2-bit does not beat ternary
FACT-001 shows runtime is not the factual culprit
```

Key hints we found:

```text
Native BitNet success does not imply same-shape post-hoc conversion works.
Layer effects are non-additive.
The all-ternary model can be a self-consistent but weak fixed point.
Quantizer/codebook tweaks do not rebuild knowledge.
```

Possible conclusion:

```text
Existing-model b1.58 conversion is not ordinary quantization. It is a function
reconstruction / adaptation problem.
```

## Paper 3: Content-KL Factual Recovery

Use [Paper 3: Content-KL Factual Recovery](./paper_3_content_kl_factual_recovery.md).

Core evidence so far:

```text
FACT-001: FP/Q2_K know facts; adapted I2_S is fluent but fact-poor.
FACT-002: data swap recovers fluency, not facts.
FACT-003A: answer-only mask moves facts to ~0.15.
FACT-003B: raw KL lambda=1.0 copies EOS/empty behavior and fails.
FACT-003C: content-KL lambda=0.2 gives fact 0.185, recovery 0.845, ok 27/27.
FACT-003C lambda=0.1 fails as salad; lambda=0.5 pending.
```

Key hint we found:

```text
The issue is not simply "KL strength".
It is "which distribution is copied".
Raw KL copies the base chat model's stop decision.
Content-KL copies content while masking EOS/special stop mass.
```

Possible conclusion:

```text
Factual recovery is objective-sensitive. The first real lever is content anchoring,
not more data and not another quantizer.
```

## Paper 4: Hybrid Capacity Candidate

Use [Paper 4: Hybrid Capacity Candidate](./paper_4_hybrid_capacity_candidate.md).

Core evidence still needed:

```text
HYBRID-001A late-layer/attention/MLP restore
HYBRID-001B Q2/Q3 replacement if F16 helps
HYBRID-001C multi-strip ternary
HYBRID-001D low-rank residual
```

Key hints we have:

```text
Native BitNet is not just "plain LLaMA + ternary weights".
Public BitNet uses BitLinear, SubLN, relu2, no bias, native training.
BitNet Reloaded suggests capacity expansion can matter for small models.
Our same-shape all-I2_S model may be under-capacitized for facts.
```

Possible conclusion:

```text
If all-I2_S plus content-KL plateaus, the product path is probably a mostly-I2_S
model with small capacity/precision pockets.
```

## What Goes Where

| result / idea | destination |
| --- | --- |
| CAT-Q / TWLA / PTQTP comparison | Paper 2 related work, Paper 4 method candidates |
| x86 I2_S parity | Paper 1 |
| storage/speed scale law | Paper 1 |
| Mac M5 runtime failure | Paper 1 appendix |
| one-shot PTQ collapse | Paper 2 |
| Q2_K negative baseline | Paper 2 |
| RT-124..127 quantizer sweep | Paper 2 |
| native BitNet architecture audit | Paper 2 background, Paper 4 motivation |
| FACT-001 factual gap | Paper 3 |
| FACT-002 data swap failure | Paper 3 |
| FACT-003A/B/C | Paper 3 |
| lambda sweep | Paper 3 |
| HYBRID-001 | Paper 4 |
| phase rotation | optional appendix / future candidate |
| gpt-oss MXFP4 negative | Paper 1 or Paper 2 appendix |

## Current Priority

```text
1. Finish ANS-001 1.1B and decide whether answer-token weighting fixes readout.
2. If ANS-001 passes, update Paper 6 and the final model scorecard.
3. If ANS-001 is flat, move to ANS-002 short-answer format curriculum.
4. Run PT2-I2S-002 on PC to test activation-aware PT2 as initializer/donor.
5. Keep Paper 1 stable; keep Paper 2 caveated against PT2-style asymmetric PTQ.
```

## Drafting Rule

Every paper draft must include three visible blocks:

```text
Draft Abstract
Result Table For The Paper
Blank Cells Before Submission
```

This prevents the project from turning into a loose pile of experiments. If a
future result does not fill a blank cell or change a claim, it should go to an
appendix or a runbook, not the main paper story.
