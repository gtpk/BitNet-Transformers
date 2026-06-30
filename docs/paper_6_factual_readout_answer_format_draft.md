# Paper 6 Draft: Reachable But Not Emitted

Status: short 2-3 page draft. ANS-001 1.1B is pending.

Related skeleton: [Paper 6 Skeleton](./paper_6_factual_readout_answer_format.md)  
Central evidence: [Evidence Ledger ANS section](../reports/EVIDENCE_LEDGER.md#h-ans-readout-track----active)

## Abstract

TL1B-1600 shows that generation collapse and factual answer failure are not the same
problem. After longer adaptation, TinyLlama-1.1B recovers fluent generation and CE,
and gold answer tokens move substantially upward in rank. Yet exact factual answers
remain below the content-KL baseline, and outputs drift into fluent base-LM
continuations instead of concise Q/A answers. We call this a readout bottleneck:
factual tokens are reachable in the distribution but not emitted in the desired
format. We test answer-token-weighted CE, which increases gradient pressure on the
first answer tokens. A 160M smoke shows the method is harmless and directionally
positive, but the decisive 1.1B experiment is required because 160M does not
reproduce the readout bottleneck.

## 1. Problem

Factual recovery was previously measured by exact answer rate. TL1B-1600 adds a more
subtle failure:

```text
gold_rank improves strongly
generation tags are clean
FACT exact remains low
outputs are fluent but off-format
```

This suggests the knowledge is not completely erased. Instead, the generation policy
does not select or stop on the concise answer.

## 2. ANS-001 Objective

We add answer-token weighting:

```text
L = CE_answer
  + 0.2 * KL_content
  + beta * CE_answer_token
```

Current smoke:

```text
beta = 4
k = first 3 answer tokens
DINO = off
```

DINO is off because DINO solved the transient but worsened factual exact answers in
TL1B-1600.

## 3. Evidence So Far

| arm | model | FACT | first-token / rank | CE | tags |
| --- | --- | ---: | --- | ---: | --- |
| content-KL baseline | TinyLlama-1.1B | 0.185 | n/a | ~4.10 | ok |
| TL1B-1600 DINO | TinyLlama-1.1B | 0.111 | gold_rank 375 | 4.08 | ok27 |
| beta=0 | 160M | 0.000 | rank=1, hit=0.778 | 3.955 | ok26/empty1 |
| beta=4 | 160M | 0.074 | rank=1, hit=0.778 | 3.972 | ok27 |

The 160M result is a safety signal, not a full predictor. Its answer token is already
rank 1, so it cannot test the 1.1B readout bottleneck.

## 4. Decision Table

| 1.1B ANS-001 result | interpretation | next |
| --- | --- | --- |
| FACT > 0.185, tags ok | answer-token readout works | promote ANS-001 |
| FACT flat, tags ok | token weight insufficient | ANS-002 short-answer format curriculum |
| FACT up, CE/tags worse | beta too strong | beta sweep |
| FACT down | answer weighting harmful | demote ANS-001 |

## References

- Neural text degeneration / decoding context: <https://arxiv.org/abs/1904.09751>
- DINO self-distillation background: <https://arxiv.org/abs/2104.14294>
- Internal evidence: [TL1B-1600 Summary](../reports/pythia_ladder/tl1b_long_metrics_summary.md), [Evidence Ledger](../reports/EVIDENCE_LEDGER.md)
