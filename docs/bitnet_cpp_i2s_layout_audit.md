# bitnet.cpp I2_S Layout Audit (RT-101)

Document position: [Index](./index.md) -> upstream format audit before writing any GGUF exporter.

Related docs:

- [I2_S Export PoC Plan](./i2s_export_poc_plan.md)
- [GGUF / bitnet.cpp Export Scoping Plan](./bitnet_cpp_export_scoping.md)

## Purpose

Fix the actual bitnet.cpp I2_S on-disk format from upstream **before** building a
writer, per the "do not assume the format" rule. This is RT-101: the mapping
table from our Python reference `I2SWeight` to the real GGUF I2_S layout, plus an
explicit list of what still must be confirmed from source/runtime.

Confidence labels below: **[confirmed]** from upstream source/quotes,
**[community]** from a reverse-engineering issue (treat as strong hint, verify in
source before trusting a writer), **[TODO]** not yet pinned.

## Upstream anchors

- Repo: [microsoft/BitNet](https://github.com/microsoft/BitNet) — official 1-bit
  inference framework; README shows `setup_env.py -q i2_s` and quant types
  `{i2_s, tl1, tl2}`, with a 2026-01-15 CPU-optimization update (format/kernel
  expectations move — re-pin commit hash when implementing).
- Format note (reverse-engineered): [Issue #412 — I2_S format documentation](https://github.com/microsoft/BitNet/issues/412)
- Converter: `utils/convert-hf-to-gguf-bitnet.py`
- Papers: [Bitnet.cpp (2502.11880)](https://arxiv.org/abs/2502.11880),
  [1-bit AI Infra (2410.16144)](https://arxiv.org/abs/2410.16144)

**[TODO] pin the exact commit hash** of microsoft/BitNet + its vendored llama.cpp
fork at implementation time; the format is not in any spec, only in source.

## I2_S weight byte layout  [community, verify in ggml-bitnet source]

From Issue #412 (reverse-engineered from the llama.cpp fork):

- Block: each **32-byte block stores 128 ternary elements** in 4 groups of 32
  (128-element interleaving, NOT sequential packing).
- Within byte `gp` (gp in 0..31) of a block, the four 2-bit fields hold elements
  at offsets `gp, 32+gp, 64+gp, 96+gp`:
  - `bits[7:6]` = offset 0, `bits[5:4]` = offset 32, `bits[3:2]` = offset 64,
    `bits[1:0]` = offset 96  (extract: `shift = 6 - 2*group; (byte >> shift) & 0x3`)
- Ternary->code: **`0b00 = 0`, `0b01 = +1`, `0b10 = -1`** (`0b11` unused).
- Per-tensor scale: **trailing 32 bytes = one float32 replicated 8x**, appended
  after `ceil(numel/4)` code bytes. Total tensor bytes = `ceil(numel/4) + 32`.

## Scale semantics  [partly confirmed]

- I2_S scale is **per-tensor float32** [community].
- The runtime dequant is `code_value * scale` with `code_value in {-1,0,+1}`, so
  the stored scale is a **multiplicative gamma** (not 1/gamma). **[TODO confirm]**
  in the I2_S dot-product/dequant kernel (`ggml-bitnet-*.cpp`).
- BitNet b1.58 *training* uses `gamma = mean(|W|)`. But note: the converter's
  TL1/TL2 path computes **`scale = max(|W|)`** [confirmed in
  `convert-hf-to-gguf-bitnet.py`], not mean. Whichever scale we store, we must
  store codes `= round(W/scale)` consistently so `scale * code == our to_dense()`.

## Converter reality  [confirmed]

`convert-hf-to-gguf-bitnet.py` emits only **F32 / F16 / TL1 / TL2** — there is no
I2_S path in the convert script. I2_S is produced by a **separate quantization
step** (`setup_env.py -q i2_s`, i.e. a `llama-quantize`-style pass to
`GGML_TYPE_I2_S`). Consequence: two possible writer paths (see below).

## Tensor distribution  [community]

- `token_embd.weight` is **F16**, not I2_S (embeddings excluded; matches our
  fp16 embedding policy).
- **Tied embeddings**: lm_head reuses `token_embd`, so there is **no
  `output.weight`** tensor.
- norms stay F16/F32. I2_S applies to the linear projections.

## RT-101 mapping table: our `I2SWeight` -> bitnet.cpp I2_S

| Aspect | Our `I2SWeight` (i2s_export.py) | bitnet.cpp I2_S | Required transform |
| --- | --- | --- | --- |
| code packing order | sequential, 4 elems/byte, row-major flat | 128-elem block, byte holds offsets `[gp,32+gp,64+gp,96+gp]` | **re-pack with 128-block interleave** |
| 2-bit field order | LSB-first (`elem0` in bits[1:0]) | MSB-first (`offset0` in bits[7:6]) | **reorder fields** |
| code mapping | `0b00=-1, 0b01=0, 0b10=+1` (shifted `T+1`) | `0b00=0, 0b01=+1, 0b10=-1` | **remap codes** |
| scale storage | one fp32 scalar field | fp32 **replicated 8x in trailing 32 bytes** of tensor data | **append 32-byte trailing scale** |
| scale value | `gamma = mean(|W|)`, dequant `gamma*T` | per-tensor fp32, multiplicative | keep `gamma`; **[TODO] confirm multiply** |
| tail padding | pad to multiple of 4 | pad to multiple of **128** per block | pad to 128 |
| dtype/transpose | torch `[out, in]` | GGUF `[out, in]` (assumed) | **[TODO] confirm no transpose** |
| tensor names | `model.layers.N.self_attn.q_proj.weight` ... | GGUF `blk.N.attn_q/attn_k/attn_v/attn_output/ffn_gate/ffn_up/ffn_down.weight` (llama.cpp convention) | **name map; [TODO] confirm via tensor_map** |
| embeddings / lm_head | fp16 embed, separate lm_head | `token_embd.weight` F16, **tied** (no `output.weight`) | match tie; keep embed F16 |

Conclusion: our reference artifact is **semantically identical** (ternary + a
single per-tensor scale, dequant `scale*T`) but **byte-incompatible**. A writer
must re-pack (interleave + MSB fields + code remap + trailing scale).

## Two export paths

- **Path A — F32 GGUF -> upstream I2_S quantize.** Write a standard F32/F16 GGUF
  of the per-tensor-native model with `convert-hf-to-gguf-bitnet.py`, then run
  `setup_env.py -q i2_s`. Lowest format risk (upstream owns the byte layout), but
  the upstream quantizer chooses its own scale rule, so the exported codes/scale
  may differ from our trained `gamma` — RT-104 logit parity will measure that.
- **Path B — direct I2_S byte writer.** Emit the I2_S tensor bytes ourselves per
  the layout above, storing our `gamma` and `T` exactly. Maximum control and
  exact match to our Python reference, but we own every byte detail and must
  track upstream format changes.

Decision: **try Path A first** (cheaper, upstream-owned format), and use our
Python reference as the RT-104 ground truth. Fall back to Path B only if Path A's
scale rule degrades logit/PPL parity.

## Remaining confirmations before writing (RT-101 exit criteria)

1. **[TODO]** pin microsoft/BitNet + llama.cpp-fork commit hashes.
2. **[TODO]** confirm I2_S byte layout in `ggml-bitnet*` source (the #412 note is
   community-reverse-engineered).
3. **[TODO]** confirm scale is multiplicative gamma in the I2_S dequant kernel.
4. **[TODO]** confirm tensor element order / no transpose for I2_S tensors.
5. **[TODO]** confirm GGUF tensor-name map (`tensor_map.get_name`) for the target
   architecture, and tied-embedding handling.
6. **[TODO]** confirm whether `setup_env.py -q i2_s` re-derives codes/scale (Path
   A) or just repacks an already-ternary tensor.

## Next steps

- RT-102: build bitnet.cpp at the pinned commit; load an official I2_S GGUF
  (e.g. `microsoft/bitnet-b1.58-2B-4T-gguf`) as a smoke; inspect a dumped I2_S
  tensor to verify the #412 layout against real bytes.
- RT-103: export our tiny per-tensor-native model via Path A; if scale parity is
  poor, implement Path B writer (`scripts/i2s_to_gguf.py`).
- RT-104: logit/PPL parity vs the Python `i2s_export` reference.
- RT-105..107: storage, latency, generation smoke.
