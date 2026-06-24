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

## Next: bitnet.cpp / GGUF Runtime Gate (TC draft)

The C++/runtime track checks the exported artifact against this Python reference.

| ID | Area | Check | Pass criterion |
| --- | --- | --- | --- |
| RT-101 | Format remap | reference `I2SWeight` -> actual GGUF I2_S block layout | field-by-field mapping table, no gaps |
| RT-102 | Loader | exported GGUF loads in bitnet.cpp | no loader/shape/name error |
| RT-103 | Logit parity | bitnet.cpp logits vs Python `gamma*T` reference | max logit delta below recorded threshold |
| RT-104 | PPL parity | tiny-text PPL bitnet.cpp vs Python reference | delta small and recorded |
| RT-105 | Storage | on-disk GGUF size vs fp16 GGUF | exact ratio recorded |
| RT-106 | Latency | per-token latency vs fp16 llama.cpp baseline | measured, no overclaim |
| RT-107 | Generation | short generation smoke in bitnet.cpp | finite, non-degenerate |

Decision rule:

```text
RT-103 / RT-104 pass with small delta -> I2_S export is the deployable path.
Logit/PPL drift large -> inspect tensor layout/transpose/scale dtype before kernel blame.
```

## Implementation Order From Here

1. (this doc) fix the Python PoC as a milestone. DONE.
2. Re-inspect the current bitnet.cpp/GGUF I2_S format from upstream source (do not
   assume), produce the RT-101 field mapping table.
3. Write a GGUF writer for the per-tensor-native model (or adapt
   `convert-hf-to-gguf-bitnet`), targeting I2_S.
4. Build bitnet.cpp, run RT-102..107 against the Python reference.
