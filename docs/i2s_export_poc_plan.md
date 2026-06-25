# I2_S Export PoC Plan

Document position: [Index](./index.md) -> Python export correctness milestone, before the bitnet.cpp/GGUF runtime track.

Related docs:

- [GGUF / bitnet.cpp Export Scoping Plan](./bitnet_cpp_export_scoping.md)
- [Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md)
- [Groupwise Alpha Hypothesis](./groupwise_alpha_hypothesis.md)

## Purpose

The per-tensor native gate decided the direction: train per-tensor b1.58 natively,
then export directly to bitnet.cpp/I2_S (no groupwise extension or custom kernel).

Before touching a C++ runtime, this PoC fixes the **export correctness** half in
pure Python, so the next stage has a precise reference to check against:

```text
per-tensor-native model  ->  I2_S-style artifact  ->  import  ->  identical logits/PPL?
```

This is the safe order: prove the artifact reconstructs the model exactly in
Python first; only then debug the bitnet.cpp loader/kernel against a known-good
reference. Status: **PASS (commit 5df98bf)**.

## Reference Artifact

`bitnet_llama/i2s_export.py`, modeling bitnet.cpp's direct b1.58 path
(per-tensor scale):

```text
gamma = mean(|W|)                       # one fp scalar per weight matrix
T     = clamp(round(W / gamma), -1, 1)  # ternary {-1,0,+1}
W_hat = gamma * T                        # dense reconstruction for reference matmul
```

`I2SWeight` per target linear:

```text
out_features : int
in_features  : int
gamma        : float                    # per-tensor scale
packed       : uint8[ceil(numel/4)]     # 2-bit ternary codes, reuses packing.pack_two_bit
```

- Bit packing is the same `two_bit` scheme as `PackedTernaryWeight` (single source),
  so I2_S is the per-tensor counterpart of the packed format.
- Source layers are `PerTensorBitLinear` (native per-tensor b1.58 STE). Embedding,
  `lm_head`, and norms stay fp16.
- `to_dense()` reproduces `PerTensorBitLinear.quantize_weight()` exactly, so the
  import target must be a plain `nn.Linear` model (a runtime dequantizes to
  `gamma*T` then does a normal matmul); importing back into `PerTensorBitLinear`
  would re-quantize and shift the scale.

## PTX Results (fixed)

`scripts/check_i2s_export.py`, tiny 2-layer Llama (vocab 256), byte eval batch
from `data/tiny_corpus.txt`:

| ID | Check | Result |
| --- | --- | --- |
| PTX-101 | layer `to_dense()` == `PerTensorBitLinear` forward weight | PASS, max_err `0.00e+00` |
| PTX-102 | imported dense model logits == per-tensor-native model | PASS, max_err `3.58e-07` |
| PTX-103 | `load(save(artifact))` import logits identical | PASS, max_err `3.58e-07` |
| PTX-104 | storage ratio vs fp16 | PASS, target `8.00x` (2.00 bits/elem), whole-model `3.68x` |
| PTX-105 | tiny-text PPL recorded; native == imported | PASS, native/imported PPL `254.64` (CE delta `0.0`), fp16 base `264.45` |

Run:

```bash
.venv/bin/python scripts/check_i2s_export.py --json-out reports/i2s_export_tc.json --strict
```

Report: [reports/i2s_export_tc.json](../reports/i2s_export_tc.json).

## What This Proves (and does not)

Proven:

- the I2_S artifact (per-tensor `gamma` + 2-bit codes) reconstructs the
  per-tensor-native model **exactly** (logit/PPL preserved through export/import)
- target linears reach the theoretical 2.0 bits/elem for 2-bit packing; whole
  model `3.68x` vs fp16 on the tiny config (diluted by fp16 embedding/lm_head)

NOT proven yet:

- **runtime speed**: this is a dense reconstruction reference, not a kernel.
  Latency/peak-memory wins require the real bitnet.cpp I2_S kernel.
- real GGUF on-disk layout compatibility (field order, tensor names, block
  structure) — that is the next stage's job.

## Runtime Gate Status

The C++/runtime track now has two separate conclusions:

```text
Python artifact correctness : PASS (this document, PTX-101..105)
bitnet.cpp I2_S runtime     : PASS on x86/Linux official model, broken on local Mac M5
```

RT-111 verified that official bitnet.cpp I2_S is faithful on x86 Colab:
`f32 PPL 1.8547` vs `i2_s PPL 1.8548`. The earlier Mac M5 collapse is a local
M5/macOS26/clang21 build/backend problem, not an I2_S design or algorithm issue.

RT-112 then verified this project's own tiny per-tensor-native model on x86
I2_S. The correct export path is **Path A'**: materialize `Wq=gamma*T`, convert
to F32/F16 GGUF, then let upstream `llama-quantize I2_S` repack it. This keeps
`max(|Wq|)=gamma` and gives I2_S/F16/F32 PPL parity. The direct latent-FP Path A
collapses, as expected, because upstream I2_S uses `sign(W)*absmax`.

## bitnet.cpp / GGUF Runtime Gate Matrix

| ID | Area | Check | Pass criterion |
| --- | --- | --- | --- |
| RT-101 | Format remap | reference `I2SWeight` -> actual GGUF I2_S block layout | DONE |
| RT-102 | Loader/build | official/tiny GGUF build and load smoke | DONE |
| RT-103 | Export plumbing | tiny HF -> F32 GGUF -> I2_S load smoke | DONE |
| RT-104 | Semantics | latent Path A vs Python `gamma*T` reference | DONE: latent Path A unfaithful |
| RT-105 | Layout/debug | source semantics + code/order investigation | DONE enough for strategy |
| RT-106 | Activation/protocol | int8 activation and F16 GGUF controls | DONE: not activation/protocol |
| RT-107 | Local runtime | zero-free and official model on Mac I2_S | DONE: Mac runtime broken |
| RT-108/109 | Mac TL1 | TL1 sanity/rebuild | DONE: local toolchain blocked |
| RT-111 | x86 sanity | official f32 vs i2_s PPL on x86 | DONE: parity |
| RT-112 | Our x86 runtime | our tiny Python/F16/F32/I2_S PPL | DONE: Path A' parity |
| RT-113 / EXPORT-006 | Storage | x86 GGUF artifact sizes | DONE: 16x target-linear vs f32 |
| RT-113 / EXPORT-007 | Latency | x86 llama-bench pp/tg metrics | DONE: ~2x token-gen |

Decision rule:

```text
RT-112 passed -> I2_S export is deployable on x86/Linux for this tiny model.
RT-113 passed -> I2_S is also efficient on the tiny x86 artifact.
Path B direct byte writer is not needed unless future larger artifacts drift.
Mac M5 failures do not block x86/Linux deployment; treat them as toolchain bugs.
```

## Implementation Order From Here

1. (this doc) fix the Python PoC as a milestone. DONE.
2. RT-101: re-inspect the bitnet.cpp/GGUF I2_S format from upstream source and
   produce the field mapping table. **DONE** -> [I2_S Layout Audit](./bitnet_cpp_i2s_layout_audit.md)
   (byte layout differs: 128-block interleave, MSB fields, trailing fp32 scale
   plus padding; I2_S is a separate quantize step, not the convert script).
3. RT-101..113 completed; see [GGUF / bitnet.cpp Export Scoping Plan](./bitnet_cpp_export_scoping.md).
4. Next: scale the same Path A' export/runtime measurement to a larger
   pretrained/small model where linears dominate the artifact.
