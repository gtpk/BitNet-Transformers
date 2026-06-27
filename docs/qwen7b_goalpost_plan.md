# Qwen 7B Goalpost Plan

Document position: [Index](./index.md) -> long-range product target for the b1.58
conversion project.

Last updated: 2026-06-27.

## Why This Document Exists

The short-term work is currently on TinyLlama-1.1B and 160M predictors. That is useful
for method discovery, but it can hide the real product question:

```text
Can this pipeline make a genuinely useful, popular open model smaller and faster?
```

Qwen 7B is the current goalpost for that question.

This document does **not** say "jump to Qwen 7B now." It says:

```text
Keep Qwen 7B visible while we run smaller experiments,
so the smaller experiments answer the right questions.
```

## Target Definition

Primary executable target:

```text
Qwen/Qwen2.5-7B-Instruct
```

Why this target:

- Apache 2.0 public model.
- Very widely used in Hugging Face / local inference ecosystems.
- Qwen2.5-7B-Instruct model card reports:
  - 7.61B parameters, 6.53B non-embedding parameters.
  - 28 layers.
  - GQA: 28 Q heads and 4 KV heads.
  - RoPE, SwiGLU, RMSNorm, QKV bias.
  - multilingual support including Korean.
  - context length up to 131,072 tokens in the model-card description.
- There are many existing quantized variants, so Q2_K/GGUF comparison is socially
  meaningful.

Secondary watch target:

```text
Qwen/Qwen3-8B
```

Reason:

```text
Qwen3 is newer and may become the more relevant public target,
but Qwen2.5-7B-Instruct has a more stable 7B target identity and wide deployment.
```

## Why Qwen 7B Is The Right Goalpost

The project's purpose is not to produce a toy 1.1B result. The desired final shape is:

```text
existing popular public model
-> b1.58-friendly adaptation
-> I2_S or mostly-I2_S artifact
-> smaller/faster than Q2_K
-> good enough factual/instruction quality to be useful
```

Qwen 7B is a good product-scale target because:

| reason | meaning |
| --- | --- |
| popular | users actually care about running it locally |
| capable | a 7B instruction model has enough knowledge to make factual recovery meaningful |
| dense-ish transformer | closer to our LLaMA pipeline than gpt-oss MXFP4 MoE |
| multilingual | Korean/user-local usefulness is plausible |
| ecosystem | GGUF/Q2_K baselines should exist or be easy to build |

## What Must Be True Before Full Qwen 7B Training

Do not spend A100/H100 time until these are true:

| gate | pass condition |
| --- | --- |
| G1: 1.1B factual mechanism | protected replay or successor moves FACT score beyond mechanism-only tier |
| G2: adaptation scaling | factual score improves with data/step/replay size, not immediate plateau |
| G3: Qwen compatibility audit | target linears, tokenizer, GGUF conversion, and I2_S path are mapped |
| G4: Qwen small smoke | Qwen 1.5B or 3B passes the same ladder |
| G5: environment | A100 80GB/H100 or equivalent plan is available for the 7B run |

The point:

```text
Qwen 7B is the goalpost, not the debugging ground.
```

## Qwen Ladder

### Q0. Metadata / Architecture Audit

No weight download required if possible.

Check:

```text
config.json
model_type
target linear names
lm_head / embedding tie
vocab size and embedding floor
attention dimensions / GQA
QKV bias handling
GGUF converter support
bitnet.cpp / llama.cpp support status
```

Output:

```text
docs/qwen_architecture_audit.md
reports/qwen7b_metadata_audit.json
```

Decision:

```text
If tensor names map cleanly to our target-linears, proceed to Qwen small smoke.
If not, write adapter plan before any training.
```

### Q1. Qwen 1.5B / 3B Smoke

Target candidates:

```text
Qwen2.5-1.5B-Instruct
Qwen2.5-3B-Instruct
```

Purpose:

```text
test recipe transfer inside the Qwen family before 7B cost.
```

Run ladder:

```text
FP reference
Q2_K reference
one-shot I2_S collapse
content-KL baseline
protected factual replay
I2_S vs F16 parity
storage/speed
```

Pass:

```text
protected replay or successor moves factual score materially over all-I2_S baseline,
without degeneration, and I2_S preserves F16 behavior.
```

### Q2. Qwen 7B Baseline Audit

Before adaptation:

```text
build or fetch FP16/F16 GGUF
build Q2_K GGUF
build one-shot I2_S Wq artifact
measure:
  file size
  target-linear ratio
  token-gen speed
  FACT/simple factual panel
  small instruction panel
  PPL/CE if feasible
```

This gives the actual product target:

```text
Qwen 7B Q2_K quality/speed/size
```

The b1.58 artifact must be compared to that, not to TinyLlama numbers.

### Q3. Qwen 7B Adaptation

Only after Q0-Q2 and 1.1B/Qwen-small evidence.

Start with parameter-efficient variants, not full naive AdamW:

```text
freeze ternary codes, train scales/residuals
LoRA/residual strip on selected target linears
cached top-k teacher logits for content anchor
protected factual replay
answer-only CE
content-KL or content-AKL
activation checkpointing
8-bit optimizer / FSDP / DeepSpeed if needed
```

Full target-linear adaptation is allowed only if hardware budget is explicit.

## Success Metrics For Qwen 7B

Use **relative-to-Q2_K** targets, not TinyLlama absolute targets.

| tier | goal | interpretation |
| --- | --- | --- |
| Qwen mechanism | improves over all-I2_S baseline and PTQ collapse | method transfers, not product yet |
| Qwen product minimum | >= 50% of Q2_K factual score, no degeneration, much smaller/faster | first usable-ish candidate |
| Qwen strong | 70-80% of Q2_K factual/instruction score | serious product trade-off |
| Qwen parity | close to Q2_K with smaller/faster artifact | major result |

For current small-panel language:

```text
Q2_K is the practical baseline.
FP is the upper reference.
I2_S must justify itself by speed/storage.
```

Do not claim product success from a model that only says fluent wrong things.

## Hardware Forecast

### Qwen 1.5B / 3B

Likely workable on:

```text
L4 24GB
A10 24GB
RTX 4090 24GB with reduced settings
A100 40GB comfortable
```

### Qwen 7B

Naive full adaptation likely wants:

```text
A100 80GB
H100 80GB
2x A100 40GB
```

Possibly workable but tight with:

```text
L40S 48GB + PEFT/cached teacher/top-k logits
```

Not recommended for full training:

```text
RTX 3080 10GB
single 24GB GPU without PEFT/offload
```

Reason:

```text
student forward/backward
teacher/content-KL forward or cached logits
optimizer states
activations
factual replay
```

This is heavier than ordinary inference and often heavier than simple LoRA SFT.

## What The RTX 3080 Can Do For Qwen

The 3080 should still help:

```text
metadata/config audit
tokenizer/panel formatting tests
small Qwen 0.5B/1.5B smoke if memory allows
FACT panel scoring
generation panel
result parsing
GGUF/Q2_K/I2_S load smoke on smaller artifacts
```

It should not be the main Qwen 7B training box.

## Why This Is Still Different From Q2_K

Q2_K asks:

```text
How close can we keep Qwen 7B with near-zero training?
```

Our b1.58 path asks:

```text
Can we pay some adaptation cost to get a much smaller/faster artifact
that is still useful enough?
```

For Qwen 7B the comparison table must include:

```text
Q2_K quality
Q2_K size
Q2_K token-gen speed
I2_S/adapted quality
I2_S/adapted size
I2_S/adapted token-gen speed
adaptation tokens and GPU hours
```

If the adaptation cost is too high or quality too low, Q2_K wins.

## Decision Tree

```text
1.1B FACT-003D succeeds to 0.40+ or shows strong scaling
  -> Qwen Q0 audit now, Qwen 1.5B/3B next.

1.1B FACT-003D reaches only 0.25-0.35
  -> mechanism confirmed; do Qwen audit, but improve data/objective before 7B training.

1.1B FACT-003D fails to beat 0.185
  -> do not train Qwen 7B; use Qwen only for architecture audit and rethink objective/data.

Qwen small smoke transfers
  -> prepare A100/H100 7B run.

Qwen small smoke fails while TinyLlama works
  -> Qwen architecture/tokenizer/data mismatch; debug small target first.
```

## Concrete Near-Term Actions

Even before Qwen training:

```text
Q0-A: metadata-only Qwen2.5-7B-Instruct audit
Q0-B: estimate target-linear vs embedding/lm_head storage floor
Q0-C: verify llama.cpp/GGUF conversion support with a tiny/no-weight smoke if possible
Q0-D: define Qwen factual/instruction eval panel
Q0-E: decide Qwen small target: 1.5B or 3B
```

These are cheap and help keep the final goal visible.

## Bottom Line

Qwen 7B should be the visible goalpost:

```text
not because we should train it immediately,
but because a poor-resource LLM project needs a real popular target.
```

The current 1.1B work is valuable only if it tells us whether the Qwen 7B jump is
worth the hardware cost.

## Source List

- Qwen2.5-7B-Instruct model card — https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen3-8B model card — https://huggingface.co/Qwen/Qwen3-8B
