# GGUF / bitnet.cpp Export Scoping Plan

Document position: [Index](./index.md) -> next track after the packed reference ladder completed.

Related docs:

- [Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md)
- [Memory-Traffic-First BitNet Plan](./memory_traffic_first_plan.md)
- [Research Signal Note](./research_signal_note.md)

## Purpose

Phase 1-4 proved the Python reference ladder:

```text
packed storage -> model export/import -> packed runtime module -> blocked dequant matmul
```

The remaining unsolved part is latency. The Python reference path proves that
dense weight materialization can be avoided, but it is slower than dense matmul.
The first export question was:

```text
Can this project's groupwise alpha*T ternary format be exported into an
existing optimized ternary runtime before writing a custom kernel?
```

That question has now been narrowed. Post-hoc export of a groupwise-trained
model to per-tensor I2_S is lossy, but a model trained natively with per-tensor
b1.58 STE matches groupwise quality. The active export path is therefore:

```text
train per-tensor-native b1.58 -> export directly to bitnet.cpp/GGUF I2_S
```

## Status

Date: 2026-06-24

```text
Step 0/1 complete.
Mapping decision for groupwise -> I2_S: lossy re-quantization.
Step 2 complete: native per-tensor b1.58 gate PASSED.
RT-101 layout audit + RT-102 build/verify complete.
RT-103A/B/C complete: Path A holds END TO END for a plain-LLaMA model.
Next: RT-104 logit/PPL parity (bitnet.cpp I2_S vs Python i2s_export reference).
```

**RT-103C (I2_S quantize + runtime smoke): PASS. Path A is real.**
`llama-quantize --token-embedding-type f16 ggml-model-f32.gguf ggml-model-i2_s.gguf I2_S 1 1`
succeeded (36MB -> 16MB); each target linear logged "converting to i2_s",
norms kept f32. Byte law holds (attn 256x256 -> 16416 B = 65536/4+32).
`llama-cli` LOADS and GENERATES from the I2_S model (rc=0, `arch = llama`, no
assert/unsupported) — so **a plain-LLaMA F32 GGUF can be I2_S-quantized and run by
bitnet.cpp with zero architecture surgery (no SubLN / BitnetForCausalLM needed)**.
Output is gibberish (random untrained tiny model; quality is RT-104). Note: the
`llama-gguf` example tool reports "failed to read tensor data" on I2_S (its size
calc doesn't know type 36) but the real `llama_model_loader` handles it fine.

The current bitnet.cpp/GGUF route is viable only through the per-tensor-native
training path. The bit-level ternary family is compatible, and the scale
granularity matches I2_S when the model is trained with one per-tensor
`gamma = mean(abs(W))` from the start. What remains is engineering validation:
artifact writing, loader compatibility, logit/PPL preservation, storage, and
runtime latency.

Post-hoc conversion of the groupwise scaled-STE model to I2_S remains a failed
path. It collapses groupwise `alpha` to one tensor scale after the model has
already adapted to local scales, and the Wikitext gate showed large PPL damage.

Source anchors used for this scoping:

- [microsoft/BitNet paper branch](https://github.com/microsoft/BitNet/tree/paper)
- [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
- [1-bit AI Infra: Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144)

## Why This Before A Custom Kernel

Direct CPU/Metal/CUDA kernels are possible, but they are a different engineering
track. Export scoping can answer several questions first:

- whether an existing runtime format can represent this project's ternary
  `T in {-1,0,+1}` plus groupwise `alpha`
- whether logit equality survives outside the Python module path
- whether real latency and memory improve in an existing inference stack
- whether a custom kernel is still necessary after export experiments

## Non-Goals

- Do not assume the current bitnet.cpp or GGUF format without checking upstream
  source/docs first.
- Do not claim latency improvement until an exported artifact runs in an
  optimized runtime.
- Do not change the conversion algorithm to fit an export format before
  measuring quality impact.

## Scoping Steps

### Step 0: Inspect Current Runtime Format

Verify the current bitnet.cpp/GGUF expectations:

- tensor names and layer mapping
- ternary/b1.58 encoding layout
- scale granularity and dtype
- metadata requirements
- supported model architectures
- available logit or generation test path

Output: a compatibility table against `PackedTernaryWeight`.

Result:

| Area | This project | bitnet.cpp I2_S / TL family | Compatibility |
| --- | --- | --- | --- |
| Value domain | ternary `{-1,0,+1}` | ternary `{-1,0,+1}` | Yes |
| Bit packing | trit `1.6 bpw` or 2-bit | I2_S/TL 2-bit-ish ternary families | Partial/usable family |
| Scale granularity | groupwise `alpha[out, in/group]` | per-tensor scale in the direct b1.58 path | No |
| Ternarization rule | `lambda * mean(abs(W_block))` threshold + masked mean alpha | absmean round/clamp style | No |
| Activation path | optional fake quant in this project | runtime-specific activation handling | Not a blocker for weight export scoping |

Conclusion: bit layout is not the blocker. Scale granularity is.

### Step 1: Mapping Decision

Compare this project format:

```text
T packed as trit bytes
alpha shape = [out_features, n_groups]
grouping along input dimension
reconstruction = alpha[:, group] * T[:, input]
```

against the target runtime format.

Decision:

- direct mapping
- transform with lossless re-layout
- transform with lossy re-quantization
- blocked: custom kernel/export format needed

Result:

```text
lossy re-quantization
```

Direct I2_S-style export would collapse groupwise `alpha` to a per-tensor
scale. That means the exported runtime would no longer be exactly equivalent to
`PackedTernaryWeight` or `PackedTernaryLinear`.

The local mapping check measured the gap on the tiny Llama-shaped fixture:

```text
groupwise output error     : 0.4339
per-tensor b1.58 error     : 0.5139
relative degradation       : +18.4%
affected target linears    : 14 / 14
```

Generated by:

```bash
.venv/bin/python scripts/check_export_mapping.py \
  --json-out reports/export_mapping_gap.json --strict
```

Important caveat: this is a random-init fixture, so the absolute error is not a
quality claim. The relative gap is enough to classify direct mapping as lossy.
The next gate must be real-text CE/PPL.

### Step 2: Lossy Export Quality Gate

Before writing an export artifact, add a per-tensor b1.58 candidate to the
arena and compare it against the current groupwise scaled-STE/S1 path on real
text.

Candidate:

```text
per_tensor_b158_i2s_candidate
```

Implementation status:

```text
implemented
```

Code:

- `bitnet_llama/conversion.py`: `per_tensor_b158_approx`
- `scripts/run_tiny_real_arena.py`: `s1_scaled_ste_export_pt_int8_kv`,
  `s1_scaled_ste_export_pt_int4_kv`

Local fixture smoke signal:

| Candidate | Acc | Loss |
| --- | ---: | ---: |
| `s1_scaled_ste_int4` groupwise | `0.311` | `2.400` |
| `s1_scaled_ste_export_pt_int4` per-tensor | `0.274` | `2.472` |

Interpretation: the tiny fixture points in the same direction as EXPORT-002 for
**post-hoc** groupwise -> per-tensor conversion, but it is not authoritative.
The fixture is only a few kilobytes. The decision must come from the Wikitext
real-text sweep and must include a native per-tensor candidate.

Pass criteria:

- CE/PPL degradation versus groupwise scaled-STE is small enough to justify
  borrowing bitnet.cpp runtime speed
- generation smoke remains finite and non-degenerate
- KL/logit drift is recorded, not hidden
- failure is accepted as a signal to avoid lossy export

Colab gate:

```text
data      : Wikitext tiny real-text sample
seeds     : 31, 32, 33
compare   : s1_scaled_ste_int4_kv
            vs s1_scaled_ste_export_pt_int4_kv
            vs per_tensor_ste_native_int4_kv
metrics   : CE loss, PPL, token accuracy, KL-to-fp16, generation smoke, Pareto
decision  : native per-tensor close to groupwise -> direct I2_S export
            native per-tensor also fails -> groupwise export/kernel fallback
```

### Step 2 Result (2026-06-24): post-hoc export is lossy, NATIVE per-tensor is not

The Colab Wikitext gate (seeds 31/32/33) split the question cleanly. Post-hoc
per-tensor export of the groupwise scaled-STE model is badly lossy (PPL +55/77/69%),
but a per-tensor b1.58 model trained natively with CE-only STE
(`per_tensor_ste_native`, added in commit `8d35350`) matches groupwise within +-1% PPL.

| seed | groupwise PPL | post-hoc export PPL | native per-tensor PPL |
| ---: | ---: | ---: | ---: |
| 31 | 6.34 | 9.85 | 6.28 |
| 32 | 5.95 | 10.55 | 6.01 |
| 33 | 6.71 | 11.31 | 6.71 |

Native per-tensor stayed on the Pareto frontier 2/3 (quality+resource winner on
seed 33), KL-to-fp16 ~= groupwise (0.149-0.175, not the 0.55+ of post-hoc), and
generation smoke was finite/non-degenerate on all three (decodes to English).

Decision:

```text
DIRECT I2_S EXPORT IS VIABLE -- via per-tensor-native training, not post-hoc conversion.
```

The mapping is "lossy" only if you convert a groupwise-trained model after the
fact. If the model is trained per-tensor (BitNet b1.58 native) from the start,
its scale granularity already matches I2_S, so export becomes lossless and quality
is preserved. No groupwise GGUF extension or custom kernel is needed for the
export track. See [Groupwise Alpha Hypothesis](./groupwise_alpha_hypothesis.md)
Gate Result (strong form refuted).

### Step 3: Minimal Export Artifact

Use a `per_tensor_ste_native`-trained model (I2_S-compatible scale) as the source,
not a groupwise model.

**RT-103A (HF export round-trip): PASS.** `scripts/export_hf_per_tensor.py` builds
a tiny standard `LlamaForCausalLM` (9.5M params, borrowing the LLaMA SPM tokenizer
+ vocab 32002 from `bitnet_b1_58-large`), saves `config.json` + `model.safetensors`
+ tokenizer files, and reloads via `AutoModelForCausalLM` with **logit max_err
0.0**. Path A note honored: the dir holds latent fp weights (upstream re-quantizes).

Open item carried to RT-103B: the borrowed `tokenizer_config.json` references a
custom `BitnetTokenizer` class, so `AutoTokenizer` fails without it. Fix in
RT-103B by normalizing the tokenizer to a standard LLaMA tokenizer (or shipping
the custom tokenizer file). Also unresolved: our tiny model is plain LLaMA (no
BitNet SubLN `attn_sub_norm`/`ffn_sub_norm`), so RT-103B must check whether the
converter/runtime accepts a plain-LLaMA-shaped model or expects `BitnetForCausalLM`.

**RT-103B (F32 GGUF convert smoke): PASS, zero converter edits.**
`convert-hf-to-gguf-bitnet.py models/tiny_pt_native --outtype f32` succeeded
(rc=0) -> `ggml-model-f32.gguf` (37M, 20 tensors). `llama-gguf` reads it
(version 3, n_tensors 20, tensor data accessible). Findings:

- Tensor names match RT-101 exactly: `token_embd.weight`,
  `blk.N.attn_q/attn_k/attn_v/attn_output.weight`,
  `blk.N.ffn_gate/ffn_up/ffn_down.weight`, `blk.N.attn_norm/ffn_norm.weight`,
  `output_norm.weight`. lm_head tied (no `output.weight`).
- The RT-103A tokenizer worry did NOT block the converter: it reads
  tokenizer.model/json directly, not via AutoTokenizer's custom class.
- **plain LLaMA (no SubLN) converts fine at F32** — the converter did not force
  `BitnetForCausalLM`/SubLN. Whether the I2_S quantize + bitnet.cpp I2_S kernel
  accept this llama-arch GGUF is the RT-103C question.

A Python reference of this artifact is now implemented and PASSED (see
[I2_S Export PoC Plan](./i2s_export_poc_plan.md), commit `5df98bf`):
`bitnet_llama/i2s_export.py` writes `gamma + 2-bit codes` and re-imports with
`gamma*T` reproducing the model logits/PPL exactly (PTX-101..105, target `8x` vs
fp16). The remaining Step 3-5 work is mapping this reference onto the actual GGUF
I2_S on-disk layout and validating in the bitnet.cpp runtime (RT-101..107 in the
PoC doc).

Start with a tiny Llama-shaped fixture model before pretrained models.

Pass criteria:

- exported artifact loads
- target linears are represented by the export path
- non-target tensors remain fp16 or expected runtime dtype
- metadata round-trip is inspectable

### Step 4: Correctness Gate

Compare:

```text
Python S1-converted model logits
Python PackedTernaryLinear logits
exported-runtime logits
```

Pass criteria:

- exact or numerically tiny logit delta on the tiny fixture
- no layer-name or transpose mismatch
- generation smoke finite and non-degenerate

### Step 5: Runtime Gate

Only after correctness:

- storage size
- load memory
- per-token latency
- long-context memory interaction with KV cache

## TC Draft

| ID | Area | Check | Pass criterion |
| --- | --- | --- | --- |
| EXPORT-001 | Format | current bitnet.cpp/GGUF format inspected | compatibility table exists |
| EXPORT-002 | Mapping | `PackedTernaryWeight` -> target layout | direct/lossless/lossy/blocked decision |
| EXPORT-003 | Quality gate | native per-tensor b1.58 arena candidate on real text | PASS: within +-1% PPL of groupwise |
| EXPORT-004 | Tiny artifact | fixture model exports and loads | no loader error |
| EXPORT-005 | Logits | exported runtime logits vs Python reference | max error threshold recorded |
| EXPORT-006 | Storage | artifact size vs fp16 | report exact ratio |
| EXPORT-007 | Latency | runtime latency vs Python reference/dense | report, no overclaim |

## Decision Rule

Proceed to export implementation when the source model is trained in the target
scale format.

The groupwise -> I2_S mapping is lossy and should not be used for deployment.
The per-tensor-native path has passed the real-text quality gate, so I2_S export
can now be claimed as viable in this narrower sense:

```text
I2_S-compatible model = trained per-tensor-native from the start.
Not I2_S-compatible model = groupwise-trained model converted post-hoc.
```

If mapping is blocked, split into:

1. custom GGUF extension/export path
2. custom CPU/Metal/CUDA fused kernel track
3. bitnet.cpp contribution or adapter track
