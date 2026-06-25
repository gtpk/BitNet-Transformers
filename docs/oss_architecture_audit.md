# gpt-oss-20b Architecture Audit (RT-117A / OSS-001)

Document position: [Index](./index.md) -> after the LLaMA systems+quality tracks
closed; decides whether gpt-oss-20b is a direct conversion or a projection target.

Related: [Scale-Up Target Roadmap](./scaleup_target_roadmap.md),
[Quality Recovery Plan](./quality_recovery_plan.md),
[GGUF / bitnet.cpp Export Scoping Plan](./bitnet_cpp_export_scoping.md).

## Method (metadata-only, no weight download)

`scripts/rt117_oss_audit.py` fetches ONLY `config.json` and per-tensor safetensors
metadata (dtype/shape via HTTP range reads) for `openai/gpt-oss-20b` — no multi-GB
weight download. 459 tensors. Result JSON: `reports/rt117_oss_audit.json`.

## Architecture (GptOssForCausalLM)

```text
hidden 2880 | 24 layers | vocab 201,088 (o200k) | base dtype BF16
attn: GQA — q_proj 4096x2880, k_proj/v_proj 512x2880, o_proj 2880x4096, + biases
      + per-layer `self_attn.sinks` (64) attention-sink params
MoE : 32 local experts, num_experts_per_tok = 4 (top-4 routing)
      per-layer router.weight 32x2880 (BF16)
```

### The decisive finding: experts are ALREADY 4-bit (MXFP4)

```text
model.layers.N.mlp.experts.gate_up_proj_blocks  U8  (32, 5760, 90, 16)   <- packed 4-bit
model.layers.N.mlp.experts.gate_up_proj_scales  U8  (32, 5760, 90)
model.layers.N.mlp.experts.gate_up_proj_bias    BF16(32, 5760)
model.layers.N.mlp.experts.down_proj_blocks     U8  (32, 2880, 90, 16)   <- packed 4-bit
model.layers.N.mlp.experts.down_proj_scales     U8  (32, 2880, 90)
model.layers.N.mlp.experts.down_proj_bias       BF16(32, 2880)
```

The expert FFN weights ship as **MXFP4** — U8-packed 4-bit values (16 bytes = 32 fp4
per block, 90 blocks = 2880 contracted dim) plus U8 block scales. They are NOT
full-precision; the 8x ternary win does not apply to them as-is.

## Storage breakdown (current resident ~13.8 GB)

| class | size | dtype | note |
| --- | ---: | --- | --- |
| expert-ffn | 10,166 MB | U8 (MXFP4 4-bit) | already quantized; dominates |
| embed/lm_head | 2,316 MB | BF16 | 201,088 vocab x 2880 — large floor |
| attn | 1,274 MB | BF16 | the only genuine 16-bit I2_S target |
| router/norm/bias | ~5 MB | BF16 | keep |
| **total** | **~13,761 MB** | | matches gpt-oss-20b's ~14 GB |

NOTE: `rt117_oss_audit.py`'s naive I2_S projection (whole ratio 0.365) assumed the
target linears were 16-bit and is therefore an OVER-estimate of the win — the
experts are already 4-bit, so ternary buys ~2x on them, not 8x. Corrected picture:

- **attn (BF16, 1.27 GB)** is the only clean 8x target -> ~0.16 GB (save ~1.1 GB).
- **experts (4-bit, 10.2 GB)** -> ternary 2-bit is only ~2x (~5.1 GB) AND requires
  de-MXFP4 -> b1.58 ternarize -> CE recovery (lossy) -> a ternary-MoE runtime kernel.
- **embed/lm_head (2.3 GB)** is a hard floor (huge vocab); not a target.

### Per-token weight traffic (the memory-traffic-first metric)

Only 4 of 32 experts fire per token, so per-token expert read is 10.2 GB x 4/32 ≈
1.27 GB (already 4-bit). With attn 1.27 GB BF16, current per-token weight traffic ≈
**2.5 GB/token**. Ternary projection (attn 8x, experts 2x): attn ~0.16 GB + experts
~0.64 GB ≈ **~0.8 GB/token** — a real ~3x traffic cut, but entirely gated on a
ternary-MoE runtime that does not exist in our stack yet.

## Runtime feasibility

- **Architecture support**: our pinned bitnet.cpp fork (commit `01eb415`, vendored
  llama.cpp ~b3639, 2024) PREDATES gpt-oss (2025) — it cannot load
  `GptOssForCausalLM`. Mainline llama.cpp added gpt-oss + an MXFP4 path in 2025, but
  the bitnet.cpp fork we rely on for I2_S has not merged it.
- **Converter**: `convert-hf-to-gguf-bitnet.py` (our fork) has no gpt-oss/MXFP4 path.
- **MoE ternary matmul**: bitnet.cpp I2_S is a DENSE GEMM kernel. gpt-oss needs
  top-4 expert gather + grouped GEMM; there is no ternary-MoE I2_S kernel in the fork.

## Risk classification: BLOCKED -> projection target (not a direct conversion)

```text
direct  : NO  — runtime can't load the arch; experts are MXFP4; no ternary-MoE kernel
adapt   : LARGE — needs ALL of:
          (1) gpt-oss GGUF support in the runtime (rebase bitnet.cpp onto a
              gpt-oss-capable llama.cpp, or port the I2_S kernel forward)
          (2) a de-MXFP4 -> per-tensor b1.58 path for the expert blocks + CE recovery
              at MoE scale (RT-116 recipe, but per-expert and on top of a 4-bit start)
          (3) a ternary-MoE matmul kernel (expert gather + 2-bit grouped GEMM)
blocked : the runtime/kernel pieces (1)+(3) are missing today
```

## Verdict & recommendation

gpt-oss-20b is a **projection target**, not a same-day conversion. The LLaMA recipe
(per-tensor b1.58 -> I2_S, teacher-free CE recovery, faithful I2_S runtime) is proven
on DENSE models; gpt-oss breaks three assumptions at once: MoE routing, natively-4-bit
experts (MXFP4), and an architecture our pinned runtime cannot load. The honest move
is to keep the systems+quality claims on the dense LLaMA family and treat gpt-oss as a
future track whose unlock order is:

1. **Runtime first** — get gpt-oss to RUN at all in a bitnet-capable stack (rebase the
   I2_S kernel onto a gpt-oss-supporting llama.cpp, or evaluate whether mainline
   llama.cpp's MXFP4 already gives the storage/speed we want without ternary).
2. **Then the cheap slice** — I2_S the BF16 attn linears only (8x on ~1.3 GB), measure.
3. **Then the hard slice** — de-MXFP4 experts -> b1.58 + CE recovery + a ternary-MoE
   kernel, only if (1) and the projected ~3x token-traffic cut justify the effort.

RT-117B (one-shard weight smoke) is NOT worth running yet: the block is at the
runtime/kernel/format level, which downloading shards would not change. Revisit RT-117B
once a gpt-oss-capable runtime path exists.

## Open question worth a cheap check next

Does mainline llama.cpp already run gpt-oss-20b (MXFP4) fast/small enough on x86 that
the ternary-MoE effort is unnecessary? A metadata/runtime check of mainline llama.cpp's
gpt-oss support (no bitnet.cpp) would tell us whether the value is in "ternary-MoE" or
just "use the existing MXFP4 path". That is the cheapest, highest-information next step.
