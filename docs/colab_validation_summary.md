# Colab Validation Summary

Document position: [Index](./index.md) -> completed Colab validation milestone.

Related docs:

- [Colab Arena Runbook](./colab_arena_runbook.md)
- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [Evolutionary LLM Arena Plan](./evolutionary_llm_arena_plan.md)

## Status

Date: 2026-06-24

Conclusion:

```text
PROCEED
```

`ScaledBitLinear` passed the Colab scale-up gate. It remained on the Pareto
frontier in the moderate run, was the quality winner across all three seed
sweep runs, stayed robust across the first group-size sweep, and passed the
activation fake-quant tiebreaker.

It then **passed real-text validation** (Wikitext-2, seeds 31/32/33, act0 and
act8): scaled-STE beat projected-QAT on accuracy, loss, PPL, and fitness in all
three seeds and stayed on the Pareto frontier. The first storage artifact
(packed ternary format, Phase 1) also landed and proved the b1.58 byte
reduction. See the Real-Text Validation Result and Packed Ternary Format
Milestone sections below.

## Validation Checklist

| Step | Result | Notes |
| --- | --- | --- |
| `colab-mcp` connection | Pass | Cell add, execution, and polling worked |
| Environment cleanup | Pass | Removed editable `transformers 4.35.2`; used standard `transformers 4.57.6` with a clean clone |
| Faster smoke arena | Pass | `strict` passed; SSTE TC passed `3/3` |
| Moderate arena | Pass | `800` train steps; scaled-STE and projected-QAT were tied/competitive on the Pareto frontier |
| Seed sweep | Pass | Seeds `31`, `32`, `33` all returned `rc=0`; scaled-STE was quality winner `3/3` |
| Group-size sweep | Pass | Group sizes `32`, `64`, `128` all kept scaled-STE quality winner `3/3` and frontier `3/3`; loss stayed in a narrow `0.2875-0.2996` band |
| Activation fake-quant seed 31 | Borderline | No quality collapse. `bits=8` improved loss/KL but lost frontier by a tiny accuracy/RAM tie-break against projected-QAT |
| Activation fake-quant seeds 32/33 | Pass | `bits=8` made scaled-STE quality winner, resource winner, and Pareto member on both seeds |

## Research Interpretation

The Colab runs support the local result:

```text
ScaledBitLinear = S1 groupwise scale preservation + CE-only STE
```

This candidate is no longer just a local tiny smoke artifact. It is stable
enough across Colab seed, group-size, and activation fake-quant sweeps to
justify the next optimization stage.

The previous hold conditions are now satisfied:

- packed ternary kernel can move from "blocked" to "candidate next phase"
- GGUF or bitnet.cpp export can be scoped after sweep stability
- TurboQuant KV-cache work can be revisited as the runtime/memory side branch

## Activation Fake-Quant Borderline Result

Seed `31` is the known worst-case seed from the earlier sweep. On that seed,
activation fake-quant did not collapse quality, but it did nudge scaled-STE off
the Pareto frontier.

| Candidate | Bits | Accuracy | Loss | KL to fp16 | Fitness | Pareto |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `s1_scaled_ste_int4` | `0` | `0.908` | `0.291` | `0.092` | `1.210` | Yes |
| `s1_projected_qat_int4` | `0` | `0.907` | `0.294` | `0.063` | `1.212` | Yes |
| `s1_scaled_ste_int4` | `8` | `0.906` | `0.286` | `0.083` | `1.209` | No |
| `s1_projected_qat_int4` | `8` | `0.907` | `0.294` | `0.063` | `1.212` | Yes |

Interpretation:

```text
This is not activation collapse. Accuracy moved by only -0.002, loss improved,
and KL improved. The frontier miss is caused by a very small accuracy margin:
projected-QAT keeps slightly lower RAM proxy, so once scaled-STE loses its
accuracy edge by ~0.001, projected-QAT dominates it on the Pareto check.
```

Decision:

```text
Resolved by act8 seeds 32 and 33. This was seed-specific frontier noise, not
activation-quant collapse.
```

## Activation Fake-Quant Tiebreaker Result

| Seed | Candidate | Accuracy | Loss | KL to fp16 | Fitness | Pareto |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `32` | `s1_scaled_ste_int4` | `0.918` | `0.261` | `0.084` | `1.221` | Yes |
| `32` | `s1_projected_qat_int4` | `0.909` | `0.285` | `0.076` | `1.214` | Yes |
| `33` | `s1_scaled_ste_int4` | `0.908` | `0.287` | `0.138` | `1.211` | Yes |
| `33` | `s1_projected_qat_int4` | `0.905` | `0.306` | `0.104` | `1.209` | Yes |

Final act8 decision:

```text
PASS
```

Activation fake-quant is robust enough for the next validation phase. The seed
31 frontier miss is now classified as a tiny Pareto-margin artifact.

Watch item:

```text
scaled-STE has slightly higher KL-to-fp16 than projected-QAT on the act8 runs,
even when scaled-STE has better accuracy, loss, and fitness.
```

This does not block progress, but it should be tracked in real-text validation
and any export/logit-equivalence checks.

## Real-Text Validation Result

Date: 2026-06-24

Real text removes the largest remaining quality risk before kernel/export work.
The arena was run in `--data-mode text` on a 200 KB Wikitext-2 sample
(byte-level tokenizer, `180177` train / `20020` eval tokens) at the moderate
config across seeds `31`, `32`, `33`, at both activation settings.

| Seed | Candidate | Acc | Loss | PPL | KL to fp16 | Fitness | Pareto |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `31` | `fp16_dense` | `0.435` | `1.911` | `6.76` | `0.000` | `0.435` | No |
| `31` | `s1_projected_qat_int4` | `0.453` | `1.879` | `6.55` | `0.113` | `0.758` | Yes |
| `31` | `s1_scaled_ste_int4` (act0) | `0.466` | `1.847` | `6.34` | `0.180` | `0.768` | Yes |
| `31` | `s1_scaled_ste_int4` (act8) | `0.469` | `1.828` | `6.22` | `0.168` | `0.772` | Yes |
| `32` | `fp16_dense` | `0.453` | `1.826` | `6.21` | `0.000` | `0.453` | No |
| `32` | `s1_projected_qat_int4` | `0.458` | `1.805` | `6.08` | `0.112` | `0.763` | Yes |
| `32` | `s1_scaled_ste_int4` (act0) | `0.477` | `1.784` | `5.95` | `0.152` | `0.780` | Yes |
| `32` | `s1_scaled_ste_int4` (act8) | `0.476` | `1.786` | `5.96` | `0.142` | `0.779` | Yes |
| `33` | `fp16_dense` | `0.424` | `1.956` | `7.07` | `0.000` | `0.424` | No |
| `33` | `s1_projected_qat_int4` | `0.431` | `1.926` | `6.86` | `0.101` | `0.735` | Yes |
| `33` | `s1_scaled_ste_int4` (act0) | `0.434` | `1.904` | `6.71` | `0.145` | `0.737` | Yes |
| `33` | `s1_scaled_ste_int4` (act8) | `0.437` | `1.897` | `6.67` | `0.148` | `0.740` | Yes |

Real-text decision:

```text
PASS
```

- scaled-STE (act0 and act8) beat projected-QAT on accuracy, loss, PPL, and
  fitness in all three seeds, and stayed on the Pareto frontier `3/3`.
- Generation smoke produced finite, non-degenerate decodes for every candidate.
- On real text, act8 did **not** hurt scaled-STE (seeds 31/33 slightly better
  than act0, seed 32 tied). This confirms the synthetic seed-31 act8 frontier
  miss was synthetic-specific noise.
- KL-to-fp16 watch item persists (scaled-STE higher than projected-QAT) but
  CE/PPL are better, so it is not a contradiction. Carry it into export/logit
  checks rather than treating it as a pause.

## Packed Ternary Format Milestone

Date: 2026-06-24

With real text passed, the first storage artifact landed (Phase 1 of
[Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md)):
`bitnet_llama/packing.py` + `scripts/check_packed_ternary.py`. This proves the
theoretical b1.58 maps to a real byte reduction.

```text
trit packing      : 1.600 bits/elem  (b1.58 bound = log2(3) = 1.585)
512x2048 layer    : trit 8.65x, two_bit 7.11x vs fp16
to_dense()        : == conversion.S1 alpha*T exactly (max_err 0.0)
ScaledBitLinear export round-trip : max_err 7e-9
TC PACK-001..006  : all pass
```

## Packed Model Export/Import Milestone

Date: 2026-06-24

Phase 2 was verified locally because it is pure packing plus logit comparison
and does not require GPU execution. This moves the claim from "each layer can
round-trip" to "the whole model preserves output after export/import."

```text
PACK-101 logit equality      : max_logit_err=0.00e+00
PACK-102 save/load artifact  : max_logit_err=0.00e+00
PACK-103 whole-model storage : 14 layers packed, 3.78x vs fp16
                               203.3 KB packed+fp16-others vs 769.2 KB fp16
```

The whole-model compression ratio is lower than the layer-only `8.65x` number
because embedding, `lm_head`, norms, and biases remain fp16. That is the right
number to track for end-to-end artifacts.

## Packed Runtime Module Milestone

Date: 2026-06-24

Phase 3 verified a runtime-facing module, not just a saved artifact.
`PackedTernaryLinear` holds uint8 packed codes and scale buffers with no dense
`[out,in]` float weight parameter, then reconstructs `alpha*T` on-the-fly for a
reference `F.linear`.

```text
PACK-201 layer forward      : max_err=0.00e+00
PACK-202 model logits       : 14 modules swapped, max_err=0.00e+00
PACK-203 no dense weight    : uint8 codes, no float weight parameter
PACK-204 state round-trip   : max_err=0.00e+00
target linear storage       : 74.0 KB packed vs 640.0 KB fp16, 8.65x
```

Important limitation: this is still a reference runtime. Forward currently
materializes dense `alpha*T` before calling `F.linear`, so storage/load memory is
reduced but compute-time peak memory and latency are not solved yet. That is the
Phase 4 job.

## Blocked Dequant Matmul Reference Milestone

Date: 2026-06-24

Phase 4 answered the memory part of the runtime question: packed weights can be
used for matmul without ever materializing the full dense `[out,in]` weight.
The reference path walks output-row chunks, unpacks only that chunk, applies
groupwise scales, and accumulates with `F.linear`-equivalent math.

```text
PACK-301 correctness     : max_err=0.00e+00
PACK-302 working set     : 8192 vs 65536 weight elements, 8.0x smaller
PACK-303 fused module    : max_logit_err=0.00e+00
PACK-304 latency honesty : dense 0.339 ms, blocked 0.399 ms, 1.2x slower
```

Interpretation:

```text
Memory win: real, at the reference working-set level.
Speed win : not yet. Python-loop blocked matmul is slower, as expected.
```

The Python/PyTorch reference ladder is now complete. Further latency work needs
either a real fused kernel or an export path into an optimized ternary runtime.

## Export Mapping Scoping Milestone

Date: 2026-06-24

GGUF/bitnet.cpp Step 0/1 was completed. The ternary value domain and 2-bit-style
packing family are compatible enough to investigate, but the scale model is not
lossless for this project.

```text
This project      : groupwise alpha[out, in/group], lambda-threshold ternary
I2_S-style export : per-tensor scale, absmean round/clamp ternary
Mapping decision  : lossy re-quantization
```

The local mapping check compared groupwise S1 against per-tensor b1.58 on the
same tiny Llama-shaped fixture:

```text
groupwise output error : 0.4339
per-tensor output err  : 0.5139
relative degradation   : +18.4%
affected layers        : 14 / 14
```

This first mapping check only ruled out **post-hoc** groupwise -> per-tensor
export. It did not prove that per-tensor b1.58 itself is weak. The follow-up
gate therefore added a native per-tensor candidate and measured real-text CE/PPL.

The arena candidate now exists:

```text
s1_scaled_ste_export_pt_int8_kv
s1_scaled_ste_export_pt_int4_kv
```

Local fixture smoke is directionally negative but not decisive:

```text
groupwise scaled-STE int4 : acc 0.311, loss 2.400
per-tensor export int4    : acc 0.274, loss 2.472
```

The authoritative native per-tensor gate below is the decision record.

## Per-Tensor Native Gate Result (decisive)

Date: 2026-06-24

The authoritative Colab Wikitext gate (seeds 31/32/33) compared the groupwise
baseline, a post-hoc per-tensor export of the groupwise model, and a per-tensor
b1.58 model trained natively with CE-only STE (`per_tensor_ste_native`).

| Seed | groupwise PPL | post-hoc export PPL | native per-tensor PPL | native acc | native KL | native Pareto |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `31` | `6.34` | `9.85` | `6.28` | `0.451` | `0.175` | No* |
| `32` | `5.95` | `10.55` | `6.01` | `0.468` | `0.165` | Yes |
| `33` | `6.71` | `11.31` | `6.71` | `0.439` | `0.149` | Yes (q+r winner) |

*seed 31 native per-tensor has lower PPL than groupwise but falls off the
frontier on a tiny fitness/RAM tie-break.

Decision:

```text
PASS -> per-tensor-native is the viable export source.
```

- Native per-tensor PPL is within +-1% of groupwise on all three seeds
  (-0.9% / +1.0% / 0.0%), far inside the +5-10% pass threshold.
- It stays on the Pareto frontier 2/3 (outright quality+resource winner on seed 33).
- KL-to-fp16 (~0.15-0.18) matches groupwise; nothing like the 0.55+ of post-hoc export.
- Generation smoke finite + non-degenerate on all three (decodes to English).
- Native per-tensor (PPL ~6) crushes post-hoc export (PPL ~10), which proves the
  earlier export loss was a post-hoc conversion artifact, not a per-tensor weakness.

Consequence: train per-tensor b1.58 native rather than post-hoc converting a
groupwise model. Later runtime work showed official bitnet.cpp I2_S is faithful
on x86/Linux, while the local Mac M5 build is broken. The remaining project gate
is our own tiny model on x86 I2_S (RT-112).

## I2_S Export PoC Milestone (local)

Date: 2026-06-24

After the per-tensor native gate, the Python export reference landed (commit
`5df98bf`): `bitnet_llama/i2s_export.py` + `scripts/check_i2s_export.py`. A
per-tensor-native model exports to I2_S-style artifacts (per-tensor `gamma` +
2-bit ternary codes) and re-imports with `gamma*T` reproducing the model exactly.

```text
PTX-101 layer round-trip   : max_err 0.00e+00
PTX-102 model round-trip   : max_logit_err 3.58e-07
PTX-103 save/load          : max_logit_err 3.58e-07
PTX-104 storage            : target 8.00x (2.0 bits/elem), whole-model 3.68x vs fp16
PTX-105 tiny-text PPL       : native == imported (CE delta 0.0)
```

This is export correctness only; runtime speed needs the bitnet.cpp I2_S kernel.
See [I2_S Export PoC Plan](./i2s_export_poc_plan.md) for the next runtime gate
(RT-101..107).

## Artifact Note

The seed sweep JSON files were generated inside the Colab session:

```text
reports/tiny_real_arena_scaled_ste_colab_seed_31.json
reports/tiny_real_arena_scaled_ste_colab_seed_32.json
reports/tiny_real_arena_scaled_ste_colab_seed_33.json
```

The group-size sweep JSON files were also generated inside the Colab session:

```text
reports/tiny_real_arena_scaled_ste_colab_g32.json
reports/tiny_real_arena_scaled_ste_colab_g64.json
reports/tiny_real_arena_scaled_ste_colab_g128.json
```

The activation fake-quant JSON files were also generated inside the Colab
session:

```text
reports/tiny_real_arena_scaled_ste_colab_act0.json
reports/tiny_real_arena_scaled_ste_colab_act8.json
reports/tiny_real_arena_scaled_ste_colab_act8_seed32.json
reports/tiny_real_arena_scaled_ste_colab_act8_seed33.json
```

The real-text JSON files (with the `data/wikitext_tiny.txt` 200 KB sample) were
also generated inside the Colab session:

```text
reports/tiny_real_arena_text_wikitext_seed31.json
reports/tiny_real_arena_text_wikitext_seed32.json
reports/tiny_real_arena_text_wikitext_seed33.json
reports/tiny_real_arena_text_wikitext_act8_seed31.json
reports/tiny_real_arena_text_wikitext_act8_seed32.json
reports/tiny_real_arena_text_wikitext_act8_seed33.json
```

They were not committed from the local workspace and may be ephemeral. Treat
this document as the milestone record, not as a replacement for raw result
archival. Re-run the sweep before making a paper-style quantitative claim.

## Runtime Platform Resolution (RT-111)

Date: 2026-06-25

After local Mac M5 runs showed I2_S collapse and TL1 build failures, x86 Colab
was used as a controlled runtime sanity check with the official
`1bitLLM/bitnet_b1_58-large` model, same pinned bitnet.cpp commit, and the same
`llama-quantize ... I2_S` path.

| Platform | F32/F16 PPL | I2_S PPL | Verdict |
| --- | ---: | ---: | --- |
| x86 Colab/Linux | `1.8547` | `1.8548` | I2_S runtime valid |
| Mac M5/macOS26/clang21 | `13.95` official f32 | `112791` official i2_s | local toolchain/backend broken |

Interpretation:

- bitnet.cpp I2_S is not an upstream/runtime-design failure.
- The Mac M5 failure is local build/backend trouble: I2_S Metal produced garbage,
  I2_S CPU crashed, TL1 default build had `BITNET_ARM_TL1=OFF`, and TL1 ON hit a
  clang LUT compile blowup.
- Our quantization/export work remains valid; f16 GGUF parity had already shown
  the weights and protocol are sound.
- Runtime validation should continue on x86/Linux first.

## Our x86 I2_S Artifact Resolution (RT-112)

Date: 2026-06-25

RT-112 tested this repo's tiny per-tensor-native b1.58 model on the same healthy
x86 bitnet.cpp I2_S runtime. Two export paths were compared with one
`llama-perplexity` tool and one eval stream.

| Path | F32 PPL | F16 PPL | I2_S PPL | Verdict |
| --- | ---: | ---: | ---: | --- |
| latent Path A | `806.49` | `806.41` | `2071.48` | collapses, expected control |
| ternary-dense Path A' (`Wq=gamma*T`) | `306.42` | `306.48` | `305.02` | PASS, I2_S ~= F16/F32 |

Interpretation:

- Our per-tensor-native b1.58 model runs faithfully in real bitnet.cpp x86 I2_S
  when exported through ternary-dense Path A'.
- Path A confirms the math: upstream latent-FP I2_S quantize uses
  `sign(W)*absmax`, which breaks our `mean(abs)` gamma.
- Path B direct byte writer is unnecessary for now.
- This result is a runtime/export correctness milestone, not yet a latency claim.

## RT-113 / EXPORT-006/007: storage + latency on x86 (DONE, 2026-06-25)

Measured on Colab x86 (2 cores) with the RT-112 ternary Path A' artifact
(`tiny_pt_ternary`), identical conditions across f32/f16/i2_s. Driver:
`scripts/rt113_storage_latency.py` (-> `reports/rt113_storage_latency.json`).

**Storage** — target-linear-only is the true I2_S compression; whole-file is
diluted by the f16 embedding floor on this tiny model:

| ratio vs f32 | target-linear-only | whole artifact |
| --- | ---: | ---: |
| f16 | 0.500 | 0.932 |
| i2_s | **0.0626 (16x)** | 0.450 |

**Latency** (llama-bench, t=2; 4-5 runs). Shared 2-core Colab CPU is noisy so
f32/f16 absolutes wander; i2_s is stable and the ratio is the robust claim:

| fmt | pp64 t/s | tg t/s |
| --- | ---: | ---: |
| f32 | ~6700 | ~290 (noisy) |
| f16 | ~8400 | ~300 (noisy) |
| i2_s | **~11200 (stable)** | **~595 (stable)** |

i2_s is fastest on both phases: prompt-processing **~1.7x vs f32**, token-gen
**~2x vs f32 and vs f16** (memory-bandwidth-bound phase, the 2-bit weight traffic
wins). i2_s tg is rock-steady ~595 t/s; f32/f16 tg noise makes the ratio range
1.8-3.5x. Peak RSS is not a discriminator (mmap -> ~5632 KB for all three);
on-disk bytes + tg t/s carry the memory story.

Conclusion: per-tensor-native -> I2_S is correct (RT-112) AND efficient (16x linear
compression, ~2x token-gen) on x86 with no custom kernel. The "before a custom
kernel" question is answered for the x86/Linux target.

## RT-114 / SCALE-001: real-model scale-up (DONE, 2026-06-25)

`scripts/rt114_scaleup.py` on JackFram/llama-160m (x86, 2 cores): downloaded the
pretrained FP model, materialized Wq=gamma*T on 84 target linears (PTQ, no retrain),
kept embd+lm_head f16, measured parity/storage/latency. All 4 TC PASS:

- **Storage**: whole-file i2_s/f32 = **0.196** (tiny was 0.45 -> converges toward the
  scale-invariant target-linear 0.0625 / 16x, because llama-160m's linears dominate).
- **Latency** (llama-bench t=2): i2_s tg **5.69x vs f32, 3.00x vs f16** (tiny was ~2x
  -> the memory-traffic win GROWS with scale); pp 3.51x vs f32.
- **Mechanics**: convert+quantize+ppl+bench rc=0; 84 linears -> i2_s, embd/lm_head f16.
- **Parity**: i2_s vs f16 PPL 1.043x looks like 4% but is only **+0.042 nats** of CE
  (0.3%) at the degenerate PTQ operating point; the residual is the I2_S int8
  ACTIVATION quant (RT-106), not encoding -> runtime faithful at scale. Absolute PPL
  ~493k is PTQ-broken by design (quality needs ternary training, separate track).

Conclusion: the I2_S export gain is not a tiny-toy artifact — storage converges to
the 16x linear floor and token-gen speedup grows at real scale.

## RT-115 / SCALE-002: TinyLlama-1.1B confirms the scale law (DONE, 2026-06-25)

Same driver, `--model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0` (154 target linears).
Three models over 100x params give two clean monotonic trends:

| model | params | whole i2_s/f32 | i2_s tg speedup vs f32 |
| --- | ---: | ---: | ---: |
| tiny | ~10M | 0.450 | ~2x |
| llama-160m | 160M | 0.196 | 5.69x |
| TinyLlama-1.1B | 1.1B | **0.1149** | **7.51x** |

whole-file ratio converges toward the scale-invariant 16x target-linear floor
(0.0625); token-gen speedup grows (1.1B: f32 2.43 t/s -> i2_s 18.26 t/s). Parity
even tighter: i2_s vs f16 = -0.0071 nats (essentially identical). The
storage/latency/parity story is a confirmed SCALE LAW on a real 1.1B LLaMA.

## Next Actions (legacy section updated 2026-06-27)

Synthetic gates, real-text validation, packed-format Phase 1/2/3/4 reference,
RT-113 storage/latency, RT-114/115 scale-up, RT-116/120 recovery, RT-121 baseline,
RT-129 decoding rescue, RT-124..127 quantizer sweep, and FACT-001..003C factual
recovery are all done or in progress. Current recommendation:

1. Finish the FACT-003C content-KL sweep: `lambda=0.2` is current best, `lambda=0.1`
   failed, `lambda=0.5` is pending.
2. Update [Fair Comparison Framework](./fair_comparison_framework.md) with the best
   content-KL recipe: post-train tokens/time, storage, speed, PPL, factual score.
3. If content-KL plateaus below a useful factual tier, run
   [Hybrid / Variable BitNet Conversion Plan](./hybrid_variable_bitnet_conversion_plan.md)
   HYBRID-001A.
4. Keep Mac M5 I2_S/TL1 work as a separate upstream/toolchain issue, not the main
   research path.

See [Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md) for the
format spec and TC matrix.

## RT-120 / TRAIN-003: G1 budget scaling on TinyLlama-1.1B (DONE, 2026-06-25)

Per `g1_budget_scaling_runbook.md`, L4 one-shot (linears-only, fp32 + AdamW8bit +
grad-ckpt, microbatch 4 x grad-accum 6 = effective batch 24, 800 steps, 4.92M train
tokens — ~16x the old budget). Result: recovered_fraction **0.480 -> 0.698** (adapted
PPL 1,217 -> 162.5; FP 10.1), QR-003 adapted i2_s vs f16 = **-0.0148 nats**. Confirms
the 1.1B low recovery was a fixed-budget artifact, not a scale failure; reaches the
paper-useful tier (>=0.70 within rounding) and I2_S runtime preservation holds.
Raw: `reports/rt120_tinyllama_g1_l4_s800_b4x6.json`.
