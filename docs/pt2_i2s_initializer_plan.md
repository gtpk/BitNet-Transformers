# PT2-I2S Initializer Plan

Document position: [Index](./index.md) -> PT2-derived direction update.

Status: proposed next branch after PT2-LLM deep dive, 2026-06-30.

## One-Line Purpose

```text
Before spending more GPU on longer TinyLlama adaptation, test whether PT2-style
asymmetric, activation-aware ternarization gives a better I2_S starting point.
```

This plan exists because PT2-LLM changes our negative PTQ interpretation.

Old conclusion:

```text
PTQ ternary conversion is dead.
```

Corrected conclusion:

```text
pure gamma*T I2_S PTQ is dead; PT2-style alpha*T + mu fitting is untested.
```

## Core Distinction

Our current pure I2_S artifact is:

```text
Wq = gamma*T
T in {-1, 0, +1}
```

PT2-LLM's asymmetric ternary family is:

```text
Wq = mu + alpha*T
T in {-1, 0, +1}
```

The forward pass decomposes as:

```text
y = (mu + alpha*T)x
  = alpha*(T x) + mu*(1^T x)
```

This creates three compatibility classes.

| class | form | I2_S status | allowed claim |
| --- | --- | --- | --- |
| A. pure I2_S | `gamma*T` | native bitnet.cpp I2_S | full systems claim |
| A+. I2_S-rooted correction | `alpha*T + mu*(sum x)` | I2_S trunk + tiny dynamic correction | mostly-I2_S / auxiliary claim |
| B. non-I2_S diagnostic | exact PT2 dense/asymmetric kernel | not bitnet.cpp I2_S | upper-bound only |

The experiment must report which class wins. Do not silently count A+ or B as pure
I2_S.

## Hypothesis

PT2-style fitting may reduce the collapse transient by improving the starting code:

```text
bad init gamma*T
  -> long degenerate transient
  -> sometimes no consolidation within budget

better init T/alpha/mu
  -> shorter transient
  -> same adaptation budget may recover behavior
```

If this is true, the next expensive TinyLlama run should use a PT2-lite initializer
instead of the old absmean initializer.

## Experiment Ladder

### PT2-I2S-001: ITF Row Grid, No Activations

Goal:

```text
Separate "better ternary assignment T" from "mu correction".
```

For each target linear row, run iterative ternary fitting:

```text
initialize mu = mean(W_row)
initialize T = nearest((W_row - mu) / alpha)
repeat K <= 10:
  solve alpha, mu for fixed T by least squares
  update T by nearest among {-1,0,+1} grid points
```

Evaluate three weights:

```text
E0 current absmean I2_S:     gamma*T_abs
E1 exact PT2 row grid:       mu + alpha*T_itf
E2 projected pure I2_S:      gamma_proj*T_itf
```

Pass signals:

| signal | interpretation |
| --- | --- |
| E1 improves, E2 also improves | better `T` survives I2_S projection -> pure initializer candidate |
| E1 improves, E2 loses gain | `mu` carries the win -> A+ correction branch |
| neither improves | ITF alone not useful |

### PT2-I2S-002: Activation-Aware Grid Alignment, Frozen T

Goal:

```text
Move grid parameters using calibration activations without changing T.
```

Given calibration activations `X` for a layer and fixed `T` from PT2-I2S-001:

```text
min_{alpha, mu} ||(W - (mu + alpha*T)) X||_F^2
```

Also test the pure I2_S projection:

```text
min_gamma ||(W - gamma*T) X||_F^2
```

Pass signals:

| signal | interpretation |
| --- | --- |
| output error and FACT improve for pure `gamma*T` | activation-aware alpha-only init belongs on Track A |
| only `mu+alpha*T` improves | use A+ dynamic correction or keep as upper bound |
| train/calibration improves but eval degrades | calibration overfit; freeze-T rule insufficient |

### PT2-I2S-003: `mu` Dynamic Correction Sidecar

Goal:

```text
Measure the exact cost/value of the missing PT2 offset.
```

Runtime formula:

```text
y = alpha*(T x) + mu*s
s = sum_j x_j
```

This is not an ordinary bias. It is input dependent, but cheap:

```text
one scalar reduction per token/group
one row-vector multiply-add
```

Report:

```text
bytes for mu,
extra ops,
latency estimate,
FACT/CE recovery,
whether i2_s trunk still dominates traffic.
```

Decision:

| result | decision |
| --- | --- |
| small `mu` cost, large behavior gain | A+ becomes a serious I2_S-rooted product branch |
| large cost or tiny gain | reject; return to pure I2_S / adaptation |

### PT2-I2S-004: SSR Column Reordering

Goal:

```text
Try PT2's cheap structural reordering before arbitrary rotations.
```

For block size 128:

```text
group columns with similar row statistics / activation-weighted signatures
apply permutation P
ternarize W P
apply P^T to activation path or fold consistently into adjacent layer
```

This is safer than Hadamard/RHT because permutation is lossless and cheap. It still
must respect transformer wiring; do not permute channels that cannot be folded
through normalization, attention heads, or residual joins.

### PT2-I2S-005: TinyLlama Transient Test

Only after 001-004:

```text
old absmean init vs best PT2-lite init
same content-KL/DINO schedule
same 800-step telemetry
```

Success means:

```text
degenerate transient shortens,
gold_rank improves earlier,
or 800-step TinyLlama no longer collapses.
```

If PT2-lite helps at 800 steps, then a 1600-step TinyLlama run should use it as the
new default initializer.

## Metrics

Every arm must record both reconstruction and behavior:

```text
weight MSE,
activation-weighted output MSE,
CE / PPL,
FACT eval,
gold logprob/rank,
degenerate/salad/empty/loop rate,
runtime class A/A+/B,
storage and estimated extra traffic.
```

The key warning from earlier work remains:

```text
lower reconstruction loss is not enough.
Behavior must move.
```

## PC / Colab Split

PC / RTX 3080:

```text
PT2-I2S-001 on 160M / Pythia-160M
PT2-I2S-002 small calibration activation smoke
SSR toy-layer and 160M smoke
mu correction dense-reference upper bound
```

Colab / L4:

```text
PT2-I2S-005 TinyLlama gate only after PC smoke shows behavior signal
Pythia-1B/1.4B if needed for dynamics comparison
```

## Decision Table

| outcome | meaning | next |
| --- | --- | --- |
| pure projection improves behavior | replace absmean init; rerun TinyLlama longer-budget with PT2-lite |
| exact `mu+alpha*T` improves but projection does not | pursue I2_S-rooted `mu` sidecar; do not call it pure I2_S |
| SSR improves pure I2_S | add SSR before adaptation; compare against rotations |
| only reconstruction improves | reject as another WSYNC-style dead branch |
| nothing improves | return to collapse dynamics / longer schedule; PT2 does not explain our failures |

## Source Notes

Primary sources:

- PT2-LLM arXiv: https://arxiv.org/abs/2510.03267
- PT2-LLM GitHub: https://github.com/XIANGLONGYAN/PT2-LLM

Related internal docs:

- [Literature Deep Dive 04: PT2-LLM](./literature_deep_dive_pt2_llm.md)
- [Current Theory, Hypotheses, And Experiment Plan](./current_theory_hypothesis_plan.md)
- [Current Experiment Status -- 2026-06-30](./current_experiment_status_2026_06_30.md)
