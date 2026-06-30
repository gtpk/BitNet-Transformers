# Paper 7 Draft: PT2 Meets I2_S

Status: short 2-3 page draft. Comparative track, not yet result-complete.

Related skeleton: [Paper 7 Skeleton](./paper_7_pt2_i2s_model_competition.md)  
Central plan: [PT2-I2S Initializer Plan](./pt2_i2s_initializer_plan.md)

## Abstract

PT2-LLM changes the interpretation of our negative one-shot experiments. Our pure
I2_S family `Wq = gamma*T` fails as a post-training ternary converter, but PT2-LLM
uses a richer asymmetric grid `Wq = mu + alpha*T`, iterative fitting, activation-aware
grid alignment, and structural reordering. Therefore PT2 is both a competitor and a
donor. As a competitor, PT2 exact must be compared against Q2_K and our adapted I2_S
artifacts as a final model. As a donor, PT2 components may produce a better I2_S
initializer or a cheap `mu*(sum x)` correction. Our first smoke, PT2-I2S-001, shows
that weight-MSE iterative fitting is not enough: it lowers reconstruction error but
worsens behavior. The next meaningful test is activation-aware output alignment.

## 1. Why PT2 Matters

Our earlier conclusion was too broad:

```text
PTQ ternary does not work.
```

The correct conclusion is:

```text
pure gamma*T I2_S PTQ did not work.
```

PT2-LLM changes the representation:

```text
Wq = mu + alpha*T
```

The `mu` term is not a normal bias. It contributes:

```text
y = alpha*(T x) + mu*(1^T x)
```

So exact PT2 is not automatically pure I2_S, but its cost may be small enough to
evaluate as an I2_S-rooted auxiliary correction.

## 2. Required Comparison

The final question is not purity. It is model quality under constraints:

| artifact | role |
| --- | --- |
| FP16 | quality ceiling |
| Q2_K | practical quantization baseline |
| PT2 exact | asymmetric ternary quality competitor |
| PT2 projected-I2_S | pure I2_S compatibility test |
| old I2_S PTQ | known collapsed baseline |
| old I2_S + adaptation | current product path |
| PT2-init + adaptation | possible frontier |

Every row must report size, effective bits, calibration/adaptation cost, tok/s,
PPL/QA/FACT, generation tags, and runtime class.

## 3. First Result

PT2-I2S-001 tested weight-MSE iterative fitting:

| arm | weight MSE | CE | FACT | gold_rank |
| --- | ---: | ---: | ---: | ---: |
| absmean I2_S | 0.282 | 11.64 | 0.0 | 1996 |
| ITF pure I2_S | 0.215 | 12.10 | 0.0 | 5590 |
| ITF asymmetric | 0.191 | 11.64 | 0.0 | 3929 |

The result is instructive: lower weight MSE worsened gold-rank. This agrees with
the broader project lesson that reconstruction alone is the wrong target.

## 4. Next Test

PT2-I2S-002 should use activation-aware grid alignment:

```text
min ||(W - Wq) X||^2
```

rather than weight MSE. It should separately measure exact `mu+alpha*T`, pure
projected I2_S, and adapted PT2-initialized I2_S.

## 5. Observed Rule And Open Hypothesis

This paper currently has one observed rule from
[Named Rules And Principles](./paper_named_rules.md):

- **Reconstruction-Is-Not-Behavior Rule**: better weight reconstruction can still
  worsen behavior.

PT2 as a competitor/donor and projection survival are comparison strategies or
open hypotheses, not discovered laws. They should stay out of the named-rule list
until the exact/projected/adapted scorecard is measured.

## References

- PT2-LLM: <https://arxiv.org/abs/2510.03267>
- BitNet b1.58: <https://arxiv.org/abs/2402.17764>
- bitnet.cpp: <https://arxiv.org/abs/2502.11880>
- GPTQ: <https://arxiv.org/abs/2210.17323>
- QuIP / incoherence processing: <https://arxiv.org/abs/2307.13304>
- Internal evidence: [PT2-I2S-001 Result](../reports/pt2_i2s_001_result.md), [Fair Comparison Framework](./fair_comparison_framework.md)
