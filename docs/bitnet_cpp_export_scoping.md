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

### RT-104 result: Path A is mechanically open but semantically UNFAITHFUL

RT-104A built a trained tiny per-tensor-native model on Wikitext (SPM tokenizer,
100k tokens, train CE 10.37 -> 2.39) and recorded the Python reference
(`scripts/rt104_build_reference.py`, `reports/rt104_reference.json`):
`i2s_export(gamma*T)` PPL `1998.6` == per-tensor-STE PPL (i2s_export reproduces
the trained model exactly). RT-104B exported it via Path A (f32 -> I2_S) and it
loaded/ran. RT-104C then measured parity:

- `llama-perplexity` on the I2_S model: **PPL 62554** vs our reference **1998.6**
  (confounded by tokenizer decode/re-encode + PPL protocol + int8 activations, so
  not a clean number, but a >30x gap).
- **Decisive weight-level check** (`scripts/rt104c_scale_check.py`, parses the
  I2_S GGUF directly): bitnet's stored per-tensor scale **equals `max(|W|)`
  exactly** (ratio 1.0000 across tensors), confirmed against `ggml-quants.c`
  `quantize_i2_s` (`i2_scale = max(|src|)`). Our per-tensor-native / BitNet b1.58
  uses `mean(|W|)` (absmean). Byte layout largely confirmed (law `ceil(numel/4)+32`,
  128-block interleave, trailing fp32 scale, code map `0b00=-1/0b01=0/0b10=+1`).

Verdict:

```text
Upstream I2_S quantize uses absmax scale (+ its own ternarization), NOT our
absmean+round. So Path A re-quantizes with different semantics and does NOT carry
our trained per-tensor quantization to the runtime (PPL collapses on this model).
Path A is fine for models trained the bitnet.cpp way; it is unfaithful to ours.
```

Decision: to deploy OUR per-tensor b1.58 quantization faithfully, use **Path B** —
write the I2_S bytes ourselves (our codes + our `mean|W|` scale) into bitnet's
layout, which we now know: 32-byte/128-elem blocks, byte `gp` holds offsets
`[gp,32+gp,64+gp,96+gp]` at bits `[7:6,5:4,3:2,1:0]`, code `0b00=-1/0b01=0/0b10=+1`,
then a trailing fp32 scale (x8 = 32 bytes). Remaining for Path B: confirm the
ACTIVE quantizer's exact element order against a byte-diff of our writer vs a
golden file (our decode currently matches ~50-62%, so finalize the order/rule
before trusting the writer).

### RT-105A investigation (layout cert): semantics confirmed, byte-order open

Read the ACTIVE quantizer `src/ggml-bitnet-mad.cpp::quantize_i2_s` and the decode
`ggml-quants.c::dequantize_row_i2_s`. Confirmed from source:

```text
i2_scale = max(|W|)                      # ABSMAX, per tensor (one fp32 at offset n/4)
q8 = (|W|<1e-6) ? 0 : (W>0 ? +1 : -1)    # PURE SIGN (no zeros, no magnitude/round)
pack: 128-elem blocks; byte gp gets elem (i*128 + g*32 + gp) in field g
decode map2bit = {00:-1, 01:0, 10:+1, 11:0}; trailing = 1 fp32 scale + padding (NOT 8x)
```

Empirically confirmed on the golden file: scale == absmax exactly (ratio 1.0000);
code histogram is pure {00,10} (i.e. sign, no zeros); trailing 32B = 1 scale +
garbage. So bitnet.cpp I2_S = **sign(W) x absmax**, decisively cruder than our
absmean+round b1.58 — which is exactly why Path A collapses our model's PPL.

Open item (does NOT change the strategic verdict): a Python decode of the golden
codes matches `sign(F32-GGUF)` only ~62.5% (should be ~100% given pure-sign source).
Likely a row-chunk / element-read nuance in the verification harness (note the
routing comment "each quantize a row, will put a scale in next row first 4B,
will diminish by next quantize" — the per-row scale-overwrite scheme). This must
be resolved (e.g. first-N side-by-side dump of f32 vs decoded codes) before a
byte-exact Path B writer is trusted, but the quantizer SEMANTICS are settled.

For Path B the encoder is the inverse of `dequantize_row_i2_s` (element e=i*128+j,
byte i*32+(j%32), field j/32, code -1->00/0->01/+1->10, scale fp32 at n/4); we
write OUR `round(W/mean|W|)` codes + `mean|W|` scale so the runtime reconstructs
our gamma*T. Verify by loading in bitnet.cpp and dequantizing back, not by
byte-diff against the (sign x absmax) golden.

### RT-104D (Path A'): feed ALREADY-ternarized dense weights

Insight (math): upstream I2_S = `sign(W) x max|W|`. If we feed the materialized
per-tensor b1.58 weights `Wq = gamma*T` (gamma=mean|W|, T in {-1,0,+1}) instead
of latent FP, then `max|Wq|=gamma`, `sign(Wq)=T`, zeros stay zero, so
`Q_upstream(Wq) = gamma*T = Wq` — lossless repack, no Path B byte-writer.

Tested (`scripts/rt104d_quantized_dense.py`): materialized Wq into a dense HF
model, Path A export + I2_S quantize.

- **CONFIRMED (the scale half):** the I2_S stored scale equals our `gamma=mean|W|`
  exactly (attn_q 0.0496667, attn_k 0.0526054, attn_v 0.0443794) — i.e. upstream
  stored `max|Wq| = gamma`, NOT the latent absmax (0.389). The insight's core holds:
  feeding ternary-dense makes upstream preserve our mean-scale.
- bitnet PPL improved 62554 -> **21384** (3x) vs the latent path. Directionally right.

NOT yet closed (honest):
- bitnet PPL (21384) is still ~10x our Python reference (2012).
- Our GGUF code decoder matches `sign(Wq)` only ~50% (the scale path is bug-free,
  so this is most likely an element-order bug in the verification harness, but a
  real code reordering is not yet ruled out). So byte-faithfulness of the CODES is
  not yet proven.
- Therefore the residual (21384 vs 2012) is not yet split between (a) runtime int8
  activation quant and (b) a code issue. -> RT-106.

Net: the scale-repack insight is validated; weight-export via Path A' is promising
(scale exact), but a clean output-parity verdict needs the code-decode/order item
resolved and activation int8 isolated.

### RT-106 (activation/protocol isolation): I2_S runtime corrupts our ternary Wq

Disambiguated the bitnet PPL gap (i2_s 21384 vs Python 2012) with two controls:

- **activation int8 ruled out** (`scripts/rt106_activation_sweep.py`): adding
  pre-linear int8 activation fake-quant (per-tensor / per-token) to the Python
  reference barely moves PPL (2012 -> ~2010). Not the cause.
- **PPL protocol ruled out**: an F16 GGUF of the SAME Wq run through the SAME
  `llama-perplexity` gives **PPL 1863** ~= Python 2012. So the runtime/protocol
  is faithful for f16.

Therefore the I2_S path itself corrupts our weights: **f16 1863 -> i2_s 21384
(11x)** on the same model. The quantizer math is `sign(W) x absmax`, which is
provably lossless on `Wq=gamma*T` (signs preserve T, zeros<1e-6 -> code 0b01,
scale=gamma — confirmed stored exactly). Yet the I2_S *runtime* output collapses.

Leading hypothesis: **the zero code (0b01) is mishandled by the I2_S quantize/
kernel on this Apple-Silicon/Metal build.** Official BitNet models are pure-sign
(no zeros; the golden file histogram was {00,10} only), so the zero path is
likely undertested. Our round-based b1.58 produces many zeros -> corruption. This
also matches the parser's ~50% / break-at-17 (misalignment where zeros occur).

Consequence: Path A' (upstream I2_S quantize) is **not faithful on this platform**
for zero-bearing ternary weights, and Path B would hit the same runtime if it is
a kernel/zero issue. The WEIGHTS are validated (f16 parity 1863~2012); the gap is
I2_S-on-this-build fidelity, not our quantization.

Next decisive (cheap) test: quantize a ZERO-FREE variant (pure sign x scale) to
I2_S and check if PPL recovers toward its f16 value -> confirms the zero-code
hypothesis. Then options: keep zeros + use a sound type (TL1 on ARM / verify I2_S
on x86), or accept pure-sign for I2_S deployment (losing our sparsity).

### RT-107 verdict: I2_S is non-functional on this Apple-Silicon build (not our quantization)

The zero-code hypothesis was REFUTED and the cause fully localized:

| model | F16/F32 PPL | I2_S PPL | notes |
| --- | ---: | ---: | --- |
| ours, round b1.58 (Wq, has zeros) | 1863 (f16) | 21384 | RT-106 |
| ours, pure-sign (zero-free, codes {00,10}) | 1717 (f16) | 25722 | RT-107 zero-free probe |
| **official BitNet-b1.58-large** | **13.95** (f32) | **112791** | decisive control |

Even the OFFICIAL pretrained model (arch `llama`, same as ours) collapses ~8000x
through I2_S. And CPU-only (`-ngl 0`) I2_S **crashes** (exit 1, `NEON=0` build).
So I2_S on this machine is broken on BOTH backends — Metal produces garbage, CPU
crashes — independent of zeros, arch, activation, or PPL protocol (all ruled out;
f16 parity 1863~2012 proves the weights + runtime/protocol are otherwise fine).

```text
CONCLUSION: bitnet.cpp's I2_S type is non-functional on this Apple-Silicon/Metal
build. This is a platform/build limitation, NOT a flaw in our per-tensor b1.58
quantization (validated by f16 GGUF parity). The earlier "gibberish" generation
on the official I2_S model was this same corruption, not a weak model.
```

Path A / Path B via I2_S are both blocked here (it's a runtime kernel issue, not
encoding). Redirect:

- **TL1** is bitnet.cpp's ARM-recommended ternary type — the realistic on-device
  path on this Mac. Needs per-dim codegen (`utils/codegen_tl1.py`) and a check of
  how TL1 represents zeros.
- Or verify I2_S on x86 (different kernel) — not available locally.
- Our weights/quantization are export-ready (f16 parity holds); only the ternary
  *runtime* needs a working type/platform.

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
