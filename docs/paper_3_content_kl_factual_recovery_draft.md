# Paper 3 Draft: Content-KL Anchoring For Factual Recovery

Status: short 2-3 page draft. Evidence supports a narrow objective claim.

Related skeleton: [Paper 3 Skeleton](./paper_3_content_kl_factual_recovery.md)  
Central evidence: [Paper Evidence Matrix](./paper_evidence_matrix.md#paper-3-content-kl-factual-recovery-evidence)

## Abstract

Once I2_S export is faithful, factual failures must be attributed to the adapted
model rather than the runtime. We show that short b1.58 adaptation can recover
fluency and CE while still losing factual answers. FP and Q2_K references answer a
small factual panel, while one-shot I2_S collapses and WikiText/Dolly-style
adaptation produces fluent but fact-poor generations. Raw KL anchoring to the base
model fails because it copies stop/EOS behavior, producing empty outputs. We introduce
content-KL, which removes EOS and special tokens from the KL support and anchors only
content mass. At `lambda=0.2`, content-KL becomes the first stable factual lever:
FACT 0.185, CE recovery 0.845, and ok 27/27 generations. This is not factual parity
with FP or Q2_K, but it isolates the first objective that moves facts without
degeneration.

## 1. Setup

The factual panel is deliberately small but diagnostic. It asks whether the adapted
model still emits short factual answers. FP and Q2_K references are sane; PTQ I2_S is
not. This separates runtime failure from adaptation failure.

## 2. Failure Modes

Data-only adaptation recovered language modeling but not facts. Raw KL was worse:
the base chat model often places high probability on short or stopped answers under
the Q/A prompt, and the student copied that behavior. The model became empty rather
than factual.

Content-KL changes the copied distribution:

```text
remove EOS/BOS/PAD/special ids
renormalize teacher and student content logits
compute KL on content support only
```

This keeps the useful base-model content field while avoiding stop-token imitation.

## 3. Results

| arm | fact_i2s | CE recovery | behavior | interpretation |
| --- | ---: | ---: | --- | --- |
| FP f16 | 0.815 | n/a | ok | base knows facts |
| Q2_K | 0.741 | n/a | ok | ordinary quantization preserves facts |
| PTQ i2_s | 0.000 | n/a | salad | no adaptation fails |
| FACT-002 mixed | 0.074 | 0.814 | fluent | data recovers fluency, not facts |
| FACT-003B raw KL | 0.000 | 0.474 | empty | copied stop behavior |
| FACT-003C content-KL 0.1 | 0.037 | 0.484 | salad | too weak |
| FACT-003C content-KL 0.2 | 0.185 | 0.845 | ok 27/27 | current best |
| FACT-003C content-KL 0.5 | 0.037 | high | over-anchored | too strong |

The sweep forms an inverted-U: too little content anchor does not stabilize the
student, too much washes out specific answers, and `lambda=0.2` is the sweet spot.

## 4. Limitations

Content-KL does not solve factuality. The best observed FACT score remains far below
Q2_K. Later TL1B-1600 experiments show that generation stability and factual readout
can separate: a model can be fluent, have improved gold-rank, and still fail exact
answers.

## 5. Named Rules

This paper uses four named rules from
[Named Rules And Principles](./paper_named_rules.md):

- **Runtime Exoneration Rule**: if adapted F16 and adapted I2_S agree, the failure
  is objective/model-side, not runtime-side.
- **Stop-Mass Contamination Rule**: raw KL can copy the teacher's EOS/stop
  behavior instead of content.
- **Content-Anchor Sweet Spot**: content-KL has an inverted-U; too weak and too
  strong both fail.
- **Fluent-But-Fact-Poor Rule**: CE/PPL recovery and factual recovery are separate.

## References

- DINO self-distillation inspiration: <https://arxiv.org/abs/2104.14294>
- Neural text degeneration / decoding context: <https://arxiv.org/abs/1904.09751>
- Internal evidence: [Evidence Ledger](../reports/EVIDENCE_LEDGER.md#a-tinyllama-11b-factual-recovery-ladder-content-kl-is-the-lever-ceiling-0185), [Factual Gap Plan](./factual_gap_experiment_plan.md)
