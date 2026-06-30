# Paper 6 Skeleton: Factual Readout And Answer Format

Working title:

```text
Reachable But Not Emitted: Factual Readout Bottlenecks In b1.58 LLM Adaptation
```

Status: active. This becomes a result paper only after ANS-001/ANS-002.

Central tables:

- [Paper Evidence Matrix](./paper_evidence_matrix.md)
- [Evidence Ledger](../reports/EVIDENCE_LEDGER.md)
- [TL1B-1600 Summary](../reports/pythia_ladder/tl1b_long_metrics_summary.md)

## Draft Abstract

After longer training resolves generation collapse in TinyLlama-1.1B, factual exact
answer quality remains below the content-KL baseline. The model is fluent and stable,
and gold answer tokens rise substantially in rank, but generation drifts into
base-LM-style continuations instead of short Q/A answers. This suggests a distinct
readout bottleneck: factual tokens become reachable in the distribution before they
become emitted as concise answers. We test answer-token-weighted CE, which increases
gradient pressure on the first answer tokens while preserving content-KL anchoring.
The 160M smoke is safe and directionally positive; the decisive 1.1B experiment is
running.

## Thesis Candidate

```text
Low-bit adaptation can recover fluent generation before it recovers answer readout.
Gold-rank improvement is not enough; the model must be trained to emit the answer
in the desired format.
```

## Core Evidence

| run | evidence | link |
| --- | --- | --- |
| TL1B-1600 | generation recovered; gold_rank 375; FACT 0.111; fluent rambling | [TL1B summary](../reports/pythia_ladder/tl1b_long_metrics_summary.md) |
| FACT-003C content-KL | current factual best 0.185 | [Evidence Ledger](../reports/EVIDENCE_LEDGER.md) |
| ANS-001 160M beta=0 | FACT 0.000, ok26/empty1 | [Evidence Ledger](../reports/EVIDENCE_LEDGER.md#h-ans-readout-track----active) |
| ANS-001 160M beta=4 | FACT 0.074, ok27, harmless | [Evidence Ledger](../reports/EVIDENCE_LEDGER.md#h-ans-readout-track----active) |
| ANS-001 1.1B | running | pending |

## Current Result Table

| arm | model | FACT | gold_rank / first-token | CE | tags | verdict |
| --- | --- | ---: | --- | ---: | --- | --- |
| content-KL baseline | TinyLlama-1.1B | 0.185 | not final readout-focused | ~4.10 | ok | current factual best |
| TL1B-1600 DINO | TinyLlama-1.1B | 0.111 | gold_rank 375 | 4.08 | ok27 | fluent but off-format |
| ANS beta=0 | 160M | 0.000 | rank=1 / hit=0.778 | 3.955 | ok26/empty1 | baseline |
| ANS beta=4 | 160M | 0.074 | rank=1 / hit=0.778 | 3.972 | ok27 | safe, positive |
| ANS beta=4 | TinyLlama-1.1B | running | running | running | running | decisive |

## Method Sketch

```text
L = CE_answer
  + lambda * KL_content(base || student)
  + beta * CE_answer_token
```

Current default:

```text
lambda = 0.2
beta = 4
k = first 3 answer tokens
DINO = off
```

## Figures

1. Gold-rank vs exact answer: reachable-but-not-emitted scatter.
2. TL1B-1600 sample generations showing fluent rambling.
3. ANS-001 beta=0 vs beta=4 table.
4. If positive: answer-token weighting improves exact answers without destabilizing tags.

## Possible Outcomes

| outcome | interpretation | next |
| --- | --- | --- |
| FACT > 0.185, tags ok | readout bottleneck confirmed and improved | promote ANS-001 |
| FACT flat, tags ok | first-token weighting insufficient | ANS-002 short-answer format curriculum |
| FACT up, tags/CE worse | beta too high | beta 1/2 sweep |
| FACT down | answer weighting harmful | demote ANS-001 |

## Missing Before Submission

```text
ANS-001 1.1B final table.
First-token metric standardization.
ANS-002 if ANS-001 is flat.
```

## Related Docs

- [Current Theory](./current_theory_hypothesis_plan.md)
- [Factual Gap Experiment Plan](./factual_gap_experiment_plan.md)
- [Experiment History](./experiment_history_and_paper_opportunities.md)
