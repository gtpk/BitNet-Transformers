# I2_S v0 Recipe (locked) + Closed Branches

Document position: [Index](./index.md). This freezes the **best working b1.58 / I2_S
conversion recipe** as `I2_S v0`, and closes the factual-recovery branches that have been
empirically exhausted. Written after FACT-003H (PopQA blend) was confirmed to fail at 1.1B
across two seeds. See [Current Theory](./current_theory_hypothesis_plan.md) and
[DINO-I2S plan](./dino_i2s_self_distillation_plan.md) for the one remaining open mechanism.

## What is solved

| axis | status |
| --- | --- |
| I2_S runtime / format | solved (i2_s == f16 parity, small/fast) |
| storage + speed | solved |
| CE / PPL recovery | solved (recovered_fraction ~0.8 at 1.1B) |
| non-degenerate generation | solved (no empty/loop/salad under v0 recipe) |
| **factual quality** | **low — content-KL `0.185` is the current ceiling** |

Honest framing (user): `I2_S v0` is a usable, cheaply-runnable b1.58 model ("흙수저용
usable b1.58"). It is **not yet** a "facts-you-can-trust assistant". The gap is factual
quality, and every CE/replay-based lever to raise it past `0.185` has failed.

## I2_S v0 recipe (LOCKED)

The best recovery recipe, frozen as the v0 baseline:

```text
base            : TinyLlama/TinyLlama-1.1B-Chat-v1.0 (frozen FP self-teacher)
quant           : per-tensor b1.58, ALL target linears ternary (PerTensorBitLinear STE)
                  Wq = gamma * T,  gamma = mean|W|,  T = clamp(round(W/gamma), -1, 1)
objective       : answer-only CE  +  lambda * content-KL(base || student)
lambda (KL)     : 0.2          (FACT-003C; inverted-U, 0.2 is the sweet spot)
kl-content-only : YES          (drop EOS/BOS/PAD/special from KL -> anchor CONTENT not stop)
lm_head         : FROZEN       (unfreezing washes facts out -> FACT-004A)
embeds          : FROZEN
hard replay     : NONE         (small hard-CE replay memorizes -> FACT-003D)
PopQA blend     : NONE         (collapses generation at 1.1B -> FACT-003H)
optimizer       : adamw
export          : Wq = gamma * T  (per_tensor_b158_approx), i2_s == f16
```

Result on the held-out factual panel (`data/factual_panel_v1.jsonl`, 27 prompts, never trained on):

```text
fact_rate        ~0.185   (best achieved; content-KL 0.2)
recovered_frac   ~0.845
tags             ok 27/27 (no empty/loop/salad)
i2_s vs f16      parity (within ~0.0015 nats)
```

`0.185` is documented as the **same-topology factual ceiling**: target-linear-only,
frozen lm_head/embeds, all-I2_S adaptation at 1.1B caps factual recall here.

## Closed branches (do not re-tread)

Each of these was tried and failed; the cost ledger (RDT-001) exists to stop us re-running them.

| branch | what it tried | result | why it fails |
| --- | --- | --- | --- |
| **lm_head unfreeze** (FACT-004A) | adapt lm_head too | fact 0.185 -> 0.04 | CE/KL minimized by generic-fluent text, abandons factual tokens |
| **small hard replay** (FACT-003D, mu sweep 1.0/0.25) | mu * answer-CE on 291 atomic facts | eval 0.185 -> 0.111 (mu1.0) / 0.037 (mu0.25) | bigger model memorizes the tiny table, crowds out general recall — net-negative at ANY mu |
| **PopQA blend** (FACT-003H, v1 + v2 seed1) | blend 12.7k PopQA at 5%, answer-only CE | v1 recovered 0.474 fact 0.0 (loops); v2 recovered 0.706 fact 0.037 (salad), popqa_tight 0.013 | 1.1B overfits short-QA blend into degenerate generation; **160M+blend recovered 0.885, so harm is 1.1B-specific** |
| **post-hoc FP restore** (HYBRID-001A) | un-quantize late layers after I2_S training | all arms worse on fact AND CE | model co-adapted early layers to feed ternary late layers; breaks when partially un-quantized |
| **WSYNC / H-I2S** (data-free weight-only) | scaling + Hadamard rotation, no data | fact 0.0 | weight-only geometry cannot escape ternary collapse |
| **SIGMA-001 / RHT-002** | residual-feedback / dithered-Hadamard reference | FAIL, fact 0.0 | movement only inside the collapsed (CE 11-13) regime |
| **sidecar / EGROW** (SIDE-001, EGROW-001/002) | tiny I2_S+LoRA auxiliary, bottleneck-targeted | no clear lever; top-k <= random-k | small added capacity does not move FACT at 160M; localization buys nothing |
| **HOME-001** (activation homeostasis) | match student/base hidden mean+RMS | eval flat 0.111 | aligning activation stats is not a factual lever |

Cross-cutting lesson (user): **"factual을 CE/replay로 고치는 단계"는 끝났다** — the
CE/replay objective family is exhausted. More QA, harder replay, or more small sidecars
will not move the factual ceiling.

## The one remaining open mechanism: DINO-I2S

Rationale: every failed branch above tries to **inject answers** (memorize-style). DINO-I2S
is the opposite pressure — **representation-preserving, no-label self-distillation**: keep the
I2_S student inside the FP teacher's content distribution + hidden geometry on broad unlabeled
text. It "forgets less" rather than "learns facts harder". See
[dino_i2s_self_distillation_plan.md](./dino_i2s_self_distillation_plan.md).

Gate (user, strict):

```text
1. PC 160M smoke FIRST (DINO-I2S-000 code path, then DINO-I2S-001 objective).
2. If FACT or PopQA-tight does NOT move >= +0.05 at 160M -> CLOSE DINO too.
   Then accept: "1.1B LLaMA-family same-topology I2_S factual ceiling is low",
   and the next move is a better/larger base model (goalpost shift), not more objective tuning.
3. Only if 160M passes -> DINO-I2S-002 1.1B Colab gate (target fact > 0.185, ideally >= 0.25).
```

One-line status: the project has a usable `I2_S v0`; raising factual quality is now a
single bet on representation-preserving self-distillation (DINO-I2S), or a base-model
goalpost shift — not more CE/replay.
