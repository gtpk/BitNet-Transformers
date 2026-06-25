# Scale-Up Target Roadmap

Document position: [Index](./index.md) -> after RT-113, before large public-model work.

Related docs:

- [GGUF / bitnet.cpp Export Scoping Plan](./bitnet_cpp_export_scoping.md)
- [Memory-Traffic-First BitNet Plan](./memory_traffic_first_plan.md)
- [Colab Validation Summary](./colab_validation_summary.md)
- [Research Signal Note](./research_signal_note.md)

## Purpose

RT-112 and RT-113 closed the tiny artifact:

```text
per-tensor-native b1.58 -> Wq=gamma*T -> GGUF F32 -> bitnet.cpp I2_S
```

is both faithful and efficient on x86. The next risk is not the tiny path. The
next risk is target selection.

This roadmap fixes the order:

```text
1. Llama-160M first  -> fast, LLaMA-shaped, linear-dominated scale-up demo
2. Quality Recovery -> prove the small/fast artifact can regain useful answers
3. gpt-oss-20b next -> socially useful public target, MoE/runtime audit required
4. gpt-oss-120b     -> projection only until 20b is understood
```

The goal is not to chase the biggest model first. The goal is to keep the
evidence ladder clean: mechanics, scale, then socially meaningful deployment.

## Why Llama-160M First

Target: `JackFram/llama-160m`.

This is the right immediate target because it isolates the exact question left
by RT-113:

```text
Does the whole-file storage ratio move toward the target-linear ratio when the
model is no longer embedding-dominated?
```

Reasons:

- It is LLaMA-shaped, so the existing bitnet.cpp converter path is already
  relevant.
- It is small enough to download, materialize, convert, quantize, and benchmark
  quickly on x86 Colab/Linux.
- It is linear-dominated: target linears should outweigh the embedding floor,
  unlike the tiny RT-112/113 artifact.
- It tests storage/latency scale-up without mixing in MoE routing, custom tensor
  maps, or architecture support risk.

Non-goal:

```text
Good absolute PPL is not expected from one-shot b1.58 PTQ.
```

SCALE-001 judges runtime faithfulness and efficiency:

```text
i2_s ~= f16 on the same Wq artifact
i2_s whole-file ratio improves vs tiny
i2_s tg throughput stays ahead of f32/f16
```

## RT-114 / SCALE-001 Plan

Driver: `scripts/rt114_scaleup.py`.

Path:

```text
download JackFram/llama-160m
materialize Wq = gamma*T on target linears
save ternary-dense HF dir
convert HF -> F32/F16 GGUF
llama-quantize F32 -> I2_S
run f16 vs i2_s PPL parity
delegate storage/latency to scripts/rt113_storage_latency.py
```

Test cases:

| ID | Check | Pass/report rule |
| --- | --- | --- |
| SCALE-001a | whole-file ratio | I2_S whole/f32 is far below tiny `0.45`; expected around linear-dominated projection |
| SCALE-001b | latency | I2_S token-generation throughput beats f32 and f16 under the same llama-bench settings |
| SCALE-001c | plumbing | 160M HF -> GGUF -> I2_S converts and loads without tensor-name surgery |
| SCALE-001d | runtime faithfulness | I2_S PPL is close to F16 PPL on the same materialized `Wq` eval stream |

Interpretation:

- PASS means RT-113 was not a tiny-only artifact.
- FAIL in PPL parity means an encoding/runtime issue at scale.
- FAIL in latency but PASS in parity means the algorithm/export path is sound,
  but runtime/kernel/context settings need diagnosis.
- Poor absolute PPL with I2_S ~= F16 is not a SCALE-001 failure.

## Quality Recovery Track

See [Quality Recovery Plan](./quality_recovery_plan.md).

RT-114 shows the systems claim scales: storage improves and token-generation
speedup grows on a pretrained LLaMA-shaped model. That still does not prove
answer quality. One-shot PTQ quality is expected to be poor.

The next quality question is:

```text
Can short, teacher-free CE adaptation recover enough quality while keeping the
same b1.58 -> I2_S runtime path?
```

Run this before any gpt-oss quality claim:

| ID | Target | Question |
| --- | --- | --- |
| QR-001 | Llama-160M | how bad is one-shot PTQ vs FP? |
| QR-002 | Llama-160M | does CE-only STE adaptation recover loss? |
| QR-003 | Llama-160M | does I2_S preserve adapted F16 Wq quality? |
| QR-004 | Llama-160M | are outputs usable on a small prompt panel? |

This track evaluates **same-quality output**, not identical strings.

## Why gpt-oss Later

Target after Llama-160M: `gpt-oss-20b`.

This is the right public target because it represents the practical problem this
project is trying to solve: a strong open-weight model that many users want to
run locally, but that still pressures low-resource hardware.

Public model-card and deployment analyses describe gpt-oss as:

- open-weight models released under Apache 2.0
- MoE transformer models
- `gpt-oss-20b` around 20.9B total parameters and about 3.6B active parameters
- designed for local or consumer-hardware deployment, but still memory-sensitive

Sources:

- [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925)
- [GPT-OSS-20B deployment analysis](https://arxiv.org/abs/2508.16700)

Why not jump directly:

- It is MoE, so target-linears are not just attention/MLP dense matrices.
- Expert tensor naming and routing must be audited before conversion.
- bitnet.cpp/GGUF support may not match the model architecture directly.
- Runtime improvements may be split across resident total weights, active
  expert weights, router overhead, and KV/cache traffic.

So gpt-oss belongs after Llama-160M, not before it.

## GPT-OSS Track

### RT-115 / OSS-001: Architecture Audit

Question:

```text
Can the existing Path A' export idea even map cleanly onto gpt-oss-20b?
```

Checks:

| ID | Check | Output |
| --- | --- | --- |
| OSS-001a | config + architecture | dense/MoE/attention/router fields summarized |
| OSS-001b | tensor map | target linears, expert linears, router tensors, embedding/lm_head separated |
| OSS-001c | runtime support | existing GGUF/llama.cpp/bitnet.cpp path classified as direct/adapt/blocked |
| OSS-001d | storage projection | total, active, target-linear, expert-linear bytes estimated |

Pass condition:

```text
We can say exactly which tensors would become I2_S and which must remain fp16/f32.
```

### RT-116 / OSS-002: Storage Projection

Question:

```text
If gpt-oss-20b target/expert linears become I2_S, does the resident model size
drop enough to matter for low-resource users?
```

Report:

- fp16/bf16 resident size
- existing common quantized size if available
- I2_S projected size for target/expert linears
- non-compressible floor: embeddings, norms, router, metadata, KV cache
- expected whole-file ratio

This is still projection. No quality claim yet.

### RT-117 / OSS-003: Minimal Export Smoke

Question:

```text
Can one small gpt-oss shard or reduced tensor subset survive Wq=gamma*T ->
I2_S-style conversion without layout surprises?
```

Pass:

- tensor-level materialization works
- scale finite, no NaN/Inf
- packed size law matches expectation
- if a runtime path exists, load smoke succeeds

### RT-118 / OSS-004: Runtime Benchmark

Run only if RT-115..117 are clean.

Metrics:

- storage: fp16/bf16 vs I2_S artifact/projection
- prompt and token-generation throughput
- TTFT vs TPOT if MoE tooling exposes it
- active-expert vs resident-weight interpretation
- PPL/logprob parity between F16 `Wq` and I2_S `Wq`

Pass:

```text
I2_S is faithful to the same Wq artifact and improves the memory-traffic story
without hiding MoE/router overhead.
```

## Decision Tree

```text
Llama-160M SCALE-001 passes
  -> proceed to gpt-oss-20b architecture audit

Llama-160M parity fails
  -> fix Path A' scale/runtime at LLaMA scale before touching MoE

Llama-160M parity passes but latency fails
  -> diagnose bitnet.cpp bench settings/kernel/context; do not blame b1.58 yet

gpt-oss audit direct
  -> implement storage projection and minimal export smoke

gpt-oss audit adapt
  -> write tensor-map adapter plan before touching weights

gpt-oss audit blocked
  -> keep gpt-oss as projection target, choose another LLaMA/Qwen/Mistral public
     dense model for the next executable benchmark
```

## What We Should Not Claim Yet

- Do not claim GPT-OSS quality after b1.58 conversion before adaptation or
  parity checks.
- Do not compare absolute PPL of one-shot ternary PTQ to trained fp16 models as
  a quality result.
- Do not claim Mac/on-device speed from x86 bitnet.cpp results.
- Do not claim MoE deployment benefit until resident size, active expert traffic,
  router overhead, and KV/cache traffic are separated.

## Current Recommendation

Execute in this order:

1. Treat `JackFram/llama-160m` SCALE-001 as the completed systems scale-up gate.
2. Start QR-001..004 on Llama-160M to test useful quality recovery.
3. Optionally run TinyLlama-1.1B as SCALE-002 to confirm storage/latency scaling
   at 1B before MoE.
4. Open `gpt-oss-20b` RT-115 architecture audit.
5. Only after the audit, decide whether gpt-oss is executable now or should
   remain a projection/north-star target.
