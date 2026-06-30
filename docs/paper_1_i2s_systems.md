# Paper 1 Skeleton: I2_S Systems

Working title:

```text
Faithful b1.58 Export and Memory-Traffic Scaling for Post-Training LLaMA Conversion
```

Status: nearly ready. This is the cleanest and strongest paper.

Central table: [Paper Evidence Matrix](./paper_evidence_matrix.md).

## Draft Abstract

Post-training conversion to b1.58 is useful only if the resulting weights can be
served by a real low-bit runtime. We show that b1.58-compatible LLaMA weights can be
faithfully exported to bitnet.cpp's I2_S format without a custom byte writer or a
custom kernel, provided the converter materializes the ternary dense weight
`Wq = gamma*T` before upstream quantization. This turns upstream absmax scaling into
the desired BitNet mean-scale representation because `max(abs(Wq)) = gamma`. On
x86/Linux, I2_S preserves both official BitNet GGUF behavior and our converted
weights, while target-linear storage is reduced by 16x versus f32. Across tiny,
160M, and 1.1B LLaMA models, whole-file compression moves from 0.450 to 0.196 to
0.1149 of f32 as linear layers dominate, and token-generation speedup grows from
about 2x to 5.69x to 7.51x versus f32. These results establish the systems
substrate for b1.58 conversion, while leaving model quality to separate adaptation
work.

## Result Table For The Paper

| run | model | artifact path | storage result | tg speed | parity | claim supported |
| --- | --- | --- | --- | ---: | --- | --- |
| RT-111 | official BitNet | upstream I2_S | n/a | n/a | f32 1.8547 vs i2_s 1.8548 | runtime works on x86 |
| RT-112 | tiny Path A' | Wq -> f32 GGUF -> I2_S | i2_s 15.94MB vs f32 36.26MB | n/a | i2_s 305.02 ~= f16 306.48 ~= f32 306.42 | materialized Wq is faithful |
| RT-113 | tiny ~10M | same | whole 0.450, target 0.0626 | ~2x vs f32/f16 | pass | tiny storage/speed proof |
| RT-114 | Llama-160M | same | whole 0.196, target 0.0625 | 5.69x vs f32 | +0.0418 nats vs f16 | scale-up holds |
| RT-115 | TinyLlama-1.1B | same | whole 0.1149, target 0.0625 | 7.51x vs f32 | -0.0071 nats vs f16 | scale law continues |

## Blank Cells Before Submission

| blank | current value | why blank | next action |
| --- | --- | --- | --- |
| latency error bars | Colab 2-core noisy | acceptable for trend, weak for final systems paper | rerun RT-114/115 on quiet x86 |
| ARM runtime | Mac M5 broken | toolchain caveat, not product proof | test known-good ARM or cite as limitation |
| end-to-end app latency | not measured | llama-bench only | run fixed prompt workload |

## Thesis

If b1.58-compatible weights are available, we can export them to the existing
bitnet.cpp I2_S runtime without a custom writer or custom kernel, and the resulting
artifact gives real storage and token-generation speed advantages that grow with
model size.

## Do Claim

```text
Wq = gamma*T can be faithfully represented by upstream I2_S if Wq is materialized first.
I2_S runtime parity holds on x86.
Target-linear storage is 16x vs f32.
Whole-file compression approaches the target-linear floor as models grow.
Token-generation speedup grows with scale.
```

## Do Not Claim

```text
The converted model has Q2_K/FP-level quality.
The method solves factual recovery.
Mac M5 local runtime is reliable.
```

## Core Results

| result | evidence |
| --- | --- |
| official I2_S parity | RT-111: official model f32 1.8547 vs i2_s 1.8548 |
| our Path A' parity | RT-112: ternary-dense i2_s ~= f16/f32 |
| latent Path A failure | RT-112: latent FP -> I2_S collapses |
| storage | RT-113/114/115: target-linear i2_s/f32 = 0.0625 |
| speed | RT-114/115: tg speedup grows to 7.51x vs f32 at 1.1B |
| Mac caveat | RT-107..109: Mac M5/clang21 runtime/toolchain failure |

## Evidence Links

| evidence | link |
| --- | --- |
| central matrix | [Paper Evidence Matrix](./paper_evidence_matrix.md#paper-1-i2_s-systems-evidence) |
| systems ledger | [Evidence Ledger](../reports/EVIDENCE_LEDGER.md) |
| I2_S export TC | [reports/i2s_export_tc.json](../reports/i2s_export_tc.json) |
| bitnet.cpp layout/runtime audit | [bitnet_cpp_i2s_layout_audit.md](./bitnet_cpp_i2s_layout_audit.md) |
| export scoping | [bitnet_cpp_export_scoping.md](./bitnet_cpp_export_scoping.md) |

## Key Insight

```text
Q_absmax(Q_mean(W)) preserves Q_mean(W) if we materialize Wq first.

Wq = gamma*T
max(abs(Wq)) = gamma
sign(Wq) = T
```

The naive export path failed because it let upstream I2_S quantize latent FP weights
with `sign(W)*absmax(W)`, which is not BitNet's `round(W/mean|W|)*mean|W|`.

## Figures

1. Path A vs Path A' parity table.
2. Storage scale law: tiny -> 160M -> 1.1B.
3. Token-generation speedup scale law.
4. Runtime portability table: x86 OK, Mac M5 broken.

## Missing Before Final

```text
Archive all rt111..115 JSONs if not already committed.
Add exact hardware/compiler table.
Optionally rerun latency on quieter x86 host for error bars.
```
