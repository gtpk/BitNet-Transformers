# Fair Comparison Framework

Document position: [Index](./index.md) -> current scorecard for comparing this project
against native BitNet, normal quantization, and future larger LLaMA runs.

Related:

- [Paper Draft](./paper_draft.md)
- [Paper Evidence Matrix](./paper_evidence_matrix.md)
- [Factual Gap Experiment Plan](./factual_gap_experiment_plan.md)
- [Hybrid / Variable BitNet Conversion Plan](./hybrid_variable_bitnet_conversion_plan.md)
- [G5 Baseline Comparison Plan](./g5_baseline_plan.md)
- [Native BitNet Architecture Audit](./native_bitnet_architecture_audit.md)

## Why This Exists

Comparing only PPL or only file size is unfair. The project is not trying to beat
native BitNet on quality after using a tiny fraction of its training compute, and it
is not trying to beat mature Q2_K on pure PPL while using fewer bits and a different
runtime.

The product goal is:

```text
turn existing public models into small, fast, useful low-resource artifacts
without paying native-from-scratch training cost.
```

So every comparison must include:

```text
pretraining cost
post-training/adaptation cost
parameter count
storage
inference speed
quality
```

## Required Scorecard

Every serious result should report this table shape:

| field | meaning |
| --- | --- |
| method | FP base, Q2_K, native BitNet, ours all-I2_S, ours hybrid |
| base model | model name and architecture family |
| dense params | total dense parameters |
| active params | MoE active params if applicable |
| pretrain tokens | from-scratch training tokens, if known |
| pretrain GPU hours | if known; otherwise state unknown |
| post-train tokens | adaptation/calibration tokens used by the method |
| post-train GPU hours | actual hardware/time used for conversion/adaptation |
| trainable params | which params are updated during adaptation |
| whole-file size | final artifact GB/MB |
| target-linear bits | bits/weight for converted target linears |
| token-gen speed | `llama-bench` tg t/s on same hardware |
| prompt speed | pp t/s on same hardware |
| PPL/CE | same tool, same eval set |
| factual score | fixed panel or benchmark subset |
| generation tags | ok / repetitive / loop / salad / empty |
| runtime parity | adapted I2_S vs adapted F16 delta |

## Baseline Families

### A. Native BitNet

```text
train from scratch inside b1.58 constraints
```

Fair interpretation:

```text
quality can be high, but training cost is enormous.
Do not compare its benchmark score to ours without pretraining tokens/GPU hours.
```

### B. FP model + normal quantization

Examples:

```text
Q2_K
Q3_K_M
Q4_0
GPTQ/AWQ/AQLM/QuIP-style methods
```

Fair interpretation:

```text
post-training cost is near zero or calibration-only;
quality preservation is strong;
storage/speed may be worse than I2_S.
```

This is the strongest practical baseline for users.

### C. Ours: all-I2_S conversion + adaptation

```text
existing FP model -> per-tensor b1.58 target linears -> I2_S
short adaptation with answer-only CE + content-KL
```

Fair interpretation:

```text
small and fast;
much cheaper than native training;
quality still below Q2_K on facts;
currently best FACT recipe is content-KL lambda=0.2.
```

### D. Ours: hybrid / variable capacity

```text
mostly I2_S, with selected Q2/Q3/F16/multi-strip/residual pockets
```

Fair interpretation:

```text
the likely product path if all-I2_S cannot reach factual quality.
Compare under a fixed bytes/token budget, not under a pure-bit ideology.
```

## Current Known Numbers

| method / run | pretrain | post-train | size/speed | quality |
| --- | --- | --- | --- | --- |
| Native BitNet 2B4T | 4T-token native training | SFT/DPO | small/fast BitNet runtime | high, public benchmark model |
| Q2_K TinyLlama | base pretrain only | quant only | bigger/slower than I2_S | fact ~0.74 on our panel |
| all-I2_S RT-120 | base pretrain only | ~4.92M tokens | 1.1B: 0.1149 whole i2_s/f32, tg 7.51x vs f32 | recovered 0.698, facts weak |
| FACT-003A | base pretrain only | same budget + answer mask | I2_S parity | fact ~0.15 |
| FACT-003B raw KL 1.0 | base pretrain only | answer CE + raw KL | I2_S parity | fact 0.00, empty collapse |
| FACT-003C content-KL 0.2 | base pretrain only | answer CE + content KL | I2_S parity | fact 0.185, ok 27/27 |
| FACT-003C content-KL 0.1 | base pretrain only | answer CE + weak content KL | I2_S parity | fact 0.037, salad; too weak |

Pending:

```text
FACT-003C content-KL lambda=0.5
```

## Claim Discipline

Do claim:

```text
I2_S runtime/export is faithful on x86.
Storage/speed gains scale with model size.
Short adaptation makes collapsed b1.58 usable-tier under sane decoding.
content-KL fixes the raw-KL EOS failure and is the first factual lever.
```

Do not claim:

```text
quality parity with Q2_K or FP.
best PPL-per-bit.
native BitNet-level quality.
that teacher-free is required.
that pure all-I2_S is enough for product quality.
```

## Next Scorecard Update

When `lambda=0.5` finishes, update:

```text
FACT-003C sweep table
best lambda
whether content-KL plateaus below 0.3
whether HYBRID-001 should start immediately
```

If a larger LLaMA run starts, report the scorecard before any qualitative claim.
