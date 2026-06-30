# Paper 1 Draft: Faithful I2_S Export And Memory-Traffic Scaling

Status: short 2-3 page draft. Evidence is mostly complete.

Related skeleton: [Paper 1 Skeleton](./paper_1_i2s_systems.md)  
Central evidence: [Paper Evidence Matrix](./paper_evidence_matrix.md#paper-1-i2_s-systems-evidence)

## Abstract

Ternary LLMs promise large reductions in memory traffic, but that promise matters
only if the weights can be served by a real low-bit runtime. We show that
b1.58-compatible LLaMA-family weights can be exported to bitnet.cpp's I2_S format
faithfully on x86/Linux, without a custom byte writer or custom kernel, provided
the converter first materializes the ternary dense weight `Wq = gamma*T`. This
small change turns upstream absmax quantization into the desired scale-preserving
representation because `max(abs(Wq)) = gamma`. Across tiny, 160M, and 1.1B models,
the target-linear storage ratio remains at the theoretical 2-bit floor relative to
f32, while whole-file compression improves as linear layers dominate. Token
generation speedup grows with scale, reaching 7.51x versus f32 in our TinyLlama-1.1B
runtime measurement. These results establish I2_S as a viable systems substrate for
post-training b1.58 conversion; quality recovery is a separate modeling problem.

## 1. Motivation

The project goal is not simply to make checkpoints smaller. The deployment bottleneck
for low-resource LLMs is token-time memory traffic: every generated token repeatedly
streams weight matrices from memory. BitNet and BitNet b1.58 argue that ternary
weights can preserve model quality when trained natively in a suitable architecture,
while bitnet.cpp provides optimized runtime support for ternary LLMs. The open
systems question for us was narrower: if a post-training conversion pipeline produces
ternary-compatible weights, can an existing runtime carry them faithfully and quickly?

## 2. Method: Path A'

The naive export path sends latent FP weights into upstream I2_S quantization. That
path is wrong for our learned quantizer, because upstream I2_S uses an absmax/sign
semantics for latent weights. Our working path materializes:

```text
Wq = gamma*T,  T in {-1, 0, +1}
```

before GGUF conversion. Since `max(abs(Wq)) = gamma`, upstream absmax scaling stores
the same scale we intend. This is the key systems identity:

```text
Q_absmax(Q_mean(W)) preserves gamma when the input is already gamma*T.
```

We call this Path A'. It avoids custom I2_S byte writing and lets bitnet.cpp perform
the final quantize/load path.

## 3. Results

The official bitnet.cpp runtime passed first: the official I2_S model matched the
F32 reference on x86. Our own Path A' artifact then matched F16/F32 references in
PPL. Storage and latency scaled as expected:

| run | model | storage result | speed result | parity |
| --- | --- | --- | --- | --- |
| RT-112 | tiny Path A' | i2_s 15.94MB vs f32 36.26MB | n/a | i2_s ~= f16/f32 |
| RT-113 | tiny ~10M | whole 0.450 of f32 | ~2x tg vs f32/f16 | pass |
| RT-114 | Llama-160M | whole 0.196, target 0.0625 | 5.69x tg vs f32 | +0.0418 nats vs f16 |
| RT-115 | TinyLlama-1.1B | whole 0.1149, target 0.0625 | 7.51x tg vs f32 | -0.0071 nats vs f16 |

The trend is exactly the memory-traffic story: target linears always compress to the
same floor, but whole-file savings improve as non-linear components become a smaller
fraction of total model size.

## 4. Limitations

This paper does not claim quality parity with Q2_K, FP16, or native BitNet. It proves
that I2_S can faithfully serve compatible weights. It also does not solve ARM
deployment: our Mac M5 local bitnet.cpp build showed toolchain/runtime problems,
while x86/Linux remained the authoritative runtime.

## 5. Named Rules

This paper uses four named rules from
[Named Rules And Principles](./paper_named_rules.md):

- **Gamma Repack Law**: materialized `Wq = gamma*T` lets upstream absmax I2_S
  preserve our scale.
- **Linear-Dominance Compression Law**: whole-model compression approaches the
  target-linear `1/16` floor as linear layers dominate.
- **Memory-Traffic Amplification Rule**: generation speedup grows with scale when
  weights dominate memory traffic.
- **Runtime Fidelity Rule**: once weights are ternary-compatible, I2_S is not the
  quality bottleneck.

## References

- Wang et al., "BitNet: Scaling 1-bit Transformers for Large Language Models", arXiv 2310.11453: <https://arxiv.org/abs/2310.11453>
- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", arXiv 2402.17764: <https://arxiv.org/abs/2402.17764>
- Wang et al., "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs", arXiv 2502.11880: <https://arxiv.org/abs/2502.11880>
- Internal evidence: [I2_S export TC](../reports/i2s_export_tc.json), [bitnet.cpp layout audit](./bitnet_cpp_i2s_layout_audit.md), [Evidence Ledger](../reports/EVIDENCE_LEDGER.md)
