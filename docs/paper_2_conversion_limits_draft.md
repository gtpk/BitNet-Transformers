# Paper 2 Draft: Why Post-Training b1.58 Conversion Is Not Ordinary Quantization

Status: short 2-3 page draft. Evidence is strong, with PT2 caveat.

Related skeleton: [Paper 2 Skeleton](./paper_2_conversion_limits.md)  
Central evidence: [Paper Evidence Matrix](./paper_evidence_matrix.md#paper-2-conversion-limits-evidence)

## Abstract

Native b1.58 LLMs show that ternary models can work, but this does not imply that a
trained FP checkpoint can be converted by ordinary post-training rounding. We study
same-shape post-training conversion into the pure I2_S-compatible form
`Wq = gamma*T` and find that one-shot ternarization collapses behavior. Standard PTQ
tools help only locally: row/block scale, AWQ/SmoothQuant-style diagonal scaling,
GPTQ-style assignment, and signed-epsilon codebooks all improve some reconstruction
or CE metrics but do not recover usable factual behavior. Q2_K remains a stronger
quality baseline, while our adapted I2_S models win primarily on size and speed.
The correct conclusion is not that b1.58 is impossible, nor that all ternary PTQ is
dead. It is that pure same-shape `gamma*T` conversion is not ordinary quantization;
it is a function reconstruction and adaptation problem. PT2-LLM narrows this claim
further by showing that asymmetric `mu + alpha*T` ternarization remains a serious
competitor.

## 1. Problem Definition

A native BitNet model is trained inside a ternary-friendly function class. A
post-training converter inherits a full-precision function and tries to represent it
with only three symbols per weight. This is a fundamentally different problem. In
our pure I2_S form, each target linear becomes:

```text
Wq = gamma*T,  T in {-1,0,+1}
```

This is much more constrained than standard 4-bit or even 2-bit quantization.

## 2. Evidence Against Pure One-Shot Conversion

The strongest one-shot or calibration-light levers were not enough:

| lever | best observed effect | interpretation |
| --- | --- | --- |
| RTN ternary | PPL above 100k in 160M baseline panel | collapse |
| row scale | +1.84 nats over per-tensor | real but insufficient |
| block/group scale | +2.36 nats | strongest scale lever, still far from usable |
| activation diagonal | +0.14 nats | too small |
| GPTQ/Hessian assignment | +0.51 nats, about 6% of gap | useful but not enough |
| signed-epsilon 2-bit | worse than ternary | zero is not the blocker |
| WSYNC/H-I2S | FACT 0.0 in collapsed regime | data-free geometry insufficient |

The most important negative result is conceptual: weight reconstruction, and even
some output-aware assignment, can improve a proxy while behavior remains collapsed.

## 3. What This Does Not Prove

This paper must be careful after PT2-LLM. PT2-LLM uses an asymmetric ternary family:

```text
Wq = mu + alpha*T
```

with iterative fitting, activation-aware grid alignment, and structural reordering.
Our RT-124..127 sweep did not test that family. Therefore the safe claim is:

```text
pure gamma*T I2_S PTQ failed;
simple PTQ tools were insufficient;
PT2-style asymmetric PTQ remains open.
```

## 4. Implication

The conversion problem must be treated as a compiler problem:

```text
coordinate choice + saliency + capacity + adaptation + runtime
```

If quality matters, a same-shape one-shot conversion is not enough. If memory and
speed matter, I2_S remains valuable, but it needs a better initializer or adaptation
schedule.

## 5. Named Rules

This paper uses four named rules from
[Named Rules And Principles](./paper_named_rules.md):

- **Same-Shape Ternary Gap**: native BitNet success does not imply post-hoc
  same-topology conversion works.
- **Quantizer Lever Ceiling**: PTQ knobs move loss but cannot close the b1.58 gap.
- **Non-Additive Restore Principle**: layer/group restores can hurt because the
  all-ternary model is co-adapted.
- **Adaptation Dominance Rule**: at 1.58 bits, objective/data adaptation dominates
  pure quantizer design.

## References

- BitNet: <https://arxiv.org/abs/2310.11453>
- BitNet b1.58: <https://arxiv.org/abs/2402.17764>
- GPTQ: <https://arxiv.org/abs/2210.17323>
- AWQ: <https://arxiv.org/abs/2306.00978>
- SmoothQuant: <https://arxiv.org/abs/2211.10438>
- PT2-LLM: <https://arxiv.org/abs/2510.03267>
- Internal evidence: [Evidence Ledger](../reports/EVIDENCE_LEDGER.md), [PT2-I2S-001 result](../reports/pt2_i2s_001_result.md)
