# Paper 7 Skeleton: PT2-I2S Model Competition

Working title:

```text
PT2 Meets I2_S: Asymmetric Ternary PTQ As Competitor And Initializer
```

Status: new comparative track. Not draftable until PT2-I2S-002 and scorecard rows
exist.

Central tables:

- [PT2-I2S Initializer Plan](./pt2_i2s_initializer_plan.md)
- [Fair Comparison Framework](./fair_comparison_framework.md)
- [Paper Evidence Matrix](./paper_evidence_matrix.md)
- [PT2-LLM Deep Dive](./literature_deep_dive_pt2_llm.md)
- [PT2-I2S-001 Result](../reports/pt2_i2s_001_result.md)

## Draft Abstract

PT2-LLM shows that post-training ternarization of LLMs can work when the quantizer
uses an asymmetric grid, activation-aware fitting, and structural reordering. This
directly challenges the earlier interpretation that b1.58 PTQ is impossible. We
reframe the comparison around final deployable artifacts: PT2 exact, PT2 projected
back to pure I2_S, old I2_S, adapted I2_S, and PT2-initialized adapted I2_S. Our
first smoke shows that weight-MSE iterative fitting improves reconstruction but not
behavior, so activation-aware output fitting is the next meaningful test. The paper
will ask whether PT2 can beat our models directly, or whether its components can be
borrowed to improve I2_S adaptation.

## Thesis Candidate

```text
PT2-style ternary PTQ must be judged as both a competitor and a donor:
exact PT2 may win quality, projected PT2 tests pure I2_S compatibility, and
PT2-init+adapt tests product relevance.
```

## Artifact Rows To Compare

| artifact | role | runtime class |
| --- | --- | --- |
| FP16 | quality ceiling | dense |
| Q2_K | practical low-bit baseline | k-quant |
| PT2 exact | strongest asymmetric ternary competitor | non-I2_S or custom |
| PT2 projected-I2_S | pure I2_S compatibility test | bitnet.cpp I2_S |
| old I2_S PTQ | known collapsed baseline | bitnet.cpp I2_S |
| old I2_S + adaptation | current product path | bitnet.cpp I2_S |
| PT2-init + adaptation | possible next frontier | pure/A+ I2_S |

## Current Evidence

| run | result | link |
| --- | --- | --- |
| RT-124..127 | simple `gamma*T` PTQ levers fail | [Evidence Matrix](./paper_evidence_matrix.md) |
| PT2-I2S-001 | ITF lowers weight MSE but worsens gold-rank/behavior | [report](../reports/pt2_i2s_001_result.md) |
| PT2-I2S-002 | pending activation-aware output-error fitting | [plan](./pt2_i2s_initializer_plan.md) |

PT2-I2S-001 summary:

| arm | weight MSE | CE | FACT | gold_rank | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| absmean I2_S | 0.282 | 11.64 | 0.0 | 1996 | old baseline |
| ITF pure I2_S | 0.215 | 12.10 | 0.0 | 5590 | MSE improves, behavior worsens |
| ITF asymmetric | 0.191 | 11.64 | 0.0 | 3929 | `mu` not enough under weight-only fitting |

## Required Scorecard

```text
method | size_MB | bits | calib/adapt GPU_h | tok/s | W2_PPL | C4_PPL |
QA_avg | FACT | gen_ok | runtime | Pareto?
```

## Key Claims To Test

Do claim now:

```text
PT2-LLM narrows our negative PTQ claim.
Weight reconstruction alone is insufficient in our first PT2-lite smoke.
```

Do not claim yet:

```text
ours beats PT2.
PT2 exact is I2_S-compatible.
PT2 projected-I2_S works.
```

## Next Experiments

| id | goal | pass signal |
| --- | --- | --- |
| PT2-I2S-002 | activation-aware grid alignment with frozen T | output/FACT improves, not just MSE |
| PT2-I2S-003 | measure `mu*(sum x)` correction value | exact `mu` gain large enough to justify A+ |
| PT2-I2S-004 | SSR column reordering | pure I2_S behavior improves |
| PT2-I2S-005 | TinyLlama schedule test | transient shortens or FACT improves |

## Related Docs

- [PT2-I2S Initializer Plan](./pt2_i2s_initializer_plan.md)
- [Literature Deep Dive 04: PT2-LLM](./literature_deep_dive_pt2_llm.md)
- [Fair Comparison Framework](./fair_comparison_framework.md)
