# The Systems Promise and Quality Limits of Teacher-Free b1.58 Conversion for Dense LLaMA Models

Status: working draft (RT-111..129 consolidated). Numbers are from this project's runs;
see the Reproduction appendix. Companion: [Paper Skeleton](./paper_skeleton.md) (claim
table + figures), [Index](./index.md).

---

## Abstract

On-device LLMs for low-resource users are bottlenecked by per-token memory traffic, not
parameter count. We study whether an existing dense LLaMA checkpoint can be *converted*
toward BitNet-style ternary (b1.58) weights as a post-training procedure, and run it end
to end on a real ternary runtime. We show that per-tensor b1.58 weights
(`Wq = gamma * T`, `gamma = mean|W|`, `T in {-1,0,+1}`) export **faithfully** into the
existing bitnet.cpp I2_S 2-bit runtime — no custom byte-writer, no custom kernel — and
that the artifact is smaller and faster on commodity x86 CPUs, with the benefit *growing
with model size* (token-generation up to ~7.5x vs f32 at 1.1B). One-shot ternary PTQ
collapses, but a short, teacher-free, cross-entropy adaptation of only the target
linears recovers most of the loss; the I2_S runtime preserves that recovered behavior to
within ~0.01 nats; and, decoded with a standard repetition penalty or sampling, the
adapted ternary model produces non-degenerate, readable text matching the degeneration
profile of FP and Q2_K (greedy alone degenerates — a known small-model artifact). We then
run a full post-training-quantization toolbox sweep — scale granularity, scale/threshold
objective, AWQ/SmoothQuant scaling, GPTQ/Hessian assignment, and a 2-bit codebook — and
find the conversion bottleneck is **not the quantizer** but adaptation/data: no one-shot
quantizer trick rescues conversion, while a short CE pass beats all of them. We therefore
frame b1.58 conversion as a systems-strong, decoding-usable path whose remaining limit is
factual quality (it does not beat a one-shot Q2_K on perplexity, nor match FP on facts),
pointing to adaptation/data — not bit/codebook engineering — as the next lever.

## 1. Introduction

The dominant cost of autoregressive decoding on commodity hardware is moving weights from
memory for each generated token. For low-resource ("흙수저") deployment the practical goal
is therefore to reduce per-token weight traffic, not parameter count. BitNet-style ternary
weights (`{-1,0,+1}` with one scale) are attractive because a 2-bit ternary runtime moves
~8x fewer weight bytes than f16. The open question this project attacks is whether an
*existing* FP/BF16 model can be moved into that regime by a conversion procedure — the way
ordinary quantization (RTN/GPTQ/AWQ) converts a model — rather than by training a BitNet
model from scratch.

We separate two sub-problems that prior framing conflates:

1. **Runtime conversion** — can b1.58 weights be represented and executed in a real,
   optimized ternary runtime at the expected size/speed?
2. **Quality conversion** — does the converted model retain usable quality?

Our contribution is to answer (1) decisively yes on x86/Linux, to characterize (2)
honestly (recovery is real and scales; generation is usable under sane decoding; factual
quality remains a gap), and to rule out a tempting wrong turn: that a smarter one-shot
quantizer is the missing piece.

## 2. Problem Definition

**Weight codebook.** Per-tensor b1.58: `Wq = gamma * T`, `gamma = mean|W|`, `T_ij in
{-1,0,+1}` by `T = clamp(round(W/gamma), -1, 1)`. This is the BitNet-native rule.

**Runtime target.** bitnet.cpp I2_S (ggml type 36): 2-bit ternary codes plus one fp32
per-tensor scale, with int8-quantized activations in the matmul kernel. Embeddings,
lm_head and norms are kept f16. We pin a single bitnet.cpp commit and its vendored
llama.cpp for all runtime results.

**Export path (Path A').** Materialize `Wq = gamma*T` into a dense HF checkpoint, convert
to an f32 GGUF, then `llama-quantize ... I2_S`. Because `max|Wq| = gamma`, `sign(Wq) = T`
and zeros stay zero, the upstream sign×absmax I2_S quantizer reproduces `Wq` exactly — so
no custom byte-writer is needed (this is the key enabling observation; the naive "latent
FP → I2_S" path instead re-quantizes with different semantics and collapses).

**Metrics.** Cross-entropy / perplexity on a held-out WikiText-2 slice (one
`llama-perplexity` binary, one eval set); on-disk bytes and target-linear bytes;
`llama-bench` token-generation throughput; and generation degeneration tags
(ok / repetitive / loop / salad / empty) plus repeated-n-gram and unique-token ratios.
We deliberately do not judge quality by PPL alone.

## 3. Method

1. **Convert + export** the per-tensor b1.58 weights via Path A' (Section 2).
2. **Teacher-free CE adaptation.** Replace target linears (attn q/k/v/o, mlp
   gate/up/down) with per-tensor b1.58 STE modules (forward uses `gamma*T`, gradients flow
   to the latent FP weight through a straight-through estimator), freeze everything else,
   and train on next-token cross-entropy only — no teacher, no distillation. An ablation
   (RT-124B/QR-005) shows the cheapest recipe (target linears only; no norms, no lm_head)
   is also the best at fixed budget.
3. **Decode** with a standard repetition penalty (~1.2) or temperature/top-p sampling;
   greedy is not used for the usability verdict (Section 5.4).

## 4. Experimental Setup

- **Models.** A tiny 2-layer LLaMA (~10M) for plumbing; `JackFram/llama-160m` (160M,
  linear-dominated) for the main quality screens; `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  for scale-up.
- **Runtime.** bitnet.cpp at a pinned commit, built on x86 Colab (clang-14) with a
  required const-correctness patch (two `int8_t * y_col` occurrences). The local Apple-M5
  build is excluded: its I2_S/TL1 kernels are broken by a toolchain/backend issue, not the
  algorithm (Section 7).
- **Data.** WikiText-2-raw (`Salesforce/wikitext`); a fixed eval slice for all
  perplexity rows; a fixed prompt panel for generation.
- **Tools.** `llama-perplexity` (CE/PPL), `llama-bench` (tg t/s), `llama-cli` (generation).

## 5. Results

### 5.1 Faithful export on x86 (RT-111/112)

On the official `bitnet_b1_58-large`, x86 I2_S matches f32: PPL 1.8547 (f32) vs 1.8548
(i2_s). Our trained tiny per-tensor model: ternary Path A' i2_s 305.02 ≈ f16 306.48 ≈ f32
306.42; the latent "Path A" control collapses (i2_s 2071 vs f16 806), confirming the
sign×absmax-vs-absmean distinction. **No Path B byte-writer is required.**

### 5.2 Storage and speed are a scale law (RT-113/114/115) — Figures 1–2

| model | whole-file i2_s/f32 | target-linear i2_s/f32 | i2_s tg vs f32 |
| --- | ---: | ---: | ---: |
| tiny (~10M) | 0.450 | 0.0625 | ~2x |
| Llama-160M | 0.196 | 0.0625 | 5.69x |
| TinyLlama-1.1B | 0.1149 | 0.0625 | 7.51x |

The target-linear compression is scale-invariant (~16x = 2-bit vs f32); the whole-file
ratio converges toward it as the fixed embedding/lm_head cost shrinks. Token-generation
speedup *grows* with scale (memory-traffic-bound). (llama-bench, x86, t=2; f32/f16
absolutes are noisy on shared CPU, i2_s is stable, so we report ratios.)

### 5.3 Quality recovery and runtime preservation (RT-116/120) — Figures 3–4

| model | FP PPL | one-shot PTQ PPL | adapted PPL | recovered fraction |
| --- | ---: | ---: | ---: | ---: |
| Llama-160M | 23.3 | 115,808 | 52.0 | 0.905 |
| TinyLlama-1.1B (budget-scaled) | 10.1 | 101,549 | 162.5 | 0.698 |

`recovered_fraction = (CE_ptq - CE_adapted)/(CE_ptq - CE_fp)`. The 1.1B point rose from
0.480 to 0.698 by scaling the training-token budget ~16x (effective batch 24, 800 steps,
~4.9M tokens), confirming the low fixed-budget number was under-training, not a scale
failure. Runtime preservation (adapted i2_s vs adapted f16): +0.0020 nats (160M),
+0.0023 (1.1B PTQ-budget), −0.0148 (1.1B budget-scaled) — |delta| ≤ 0.015 nats at every
scale; I2_S faithfully runs the adapted model.

### 5.4 Decoding usability (RT-129) — Figure 7

RT-122 found the 1.1B adapted model degenerates ("= = =" loops) — but only under greedy.
With a standard repetition penalty or sampling the same model is non-degenerate:

| model | decode | ok/12 | loop | salad | rep-3gram |
| --- | --- | ---: | ---: | ---: | ---: |
| adapted i2_s | greedy | 1 | 0 | 9 | 0.656 |
| adapted i2_s | rep-penalty 1.2 | 12 | 0 | 0 | 0.003 |
| adapted i2_s | temp0.8/top_p0.95 | 12 | 0 | 0 | 0.074 |
| FP f16 | greedy | 11 | 0 | 0 | 0.049 |
| Q2_K | greedy | 12 | 0 | 0 | 0.115 |
| PTQ i2_s (no train) | greedy | 0 | 9 | 0 | 0.781 |

The adapted ternary model reaches the FP/Q2_K degeneration tier under sane decoding;
adapted i2_s == adapted f16 at every decode; PTQ (no adaptation) stays collapsed under
any decode. So usability needs adaptation **and** a repetition penalty, and "1.58-bit
is unusable" was a greedy-decoding artifact.

## 6. Negative Results (scoping)

### 6.1 We do not beat one-shot Q2_K on PPL (RT-121) — Figure 5

On Llama-160M (one tool, one eval, embd+lm_head f16 everywhere): ours (1.58-bit + CE)
PPL 114.1 vs Q2_K (~2.6-bit, no training) 97.9. Ours is smaller (121.5 vs 134.1 MB),
lower-bit, and runs the faster I2_S kernel, and it rescues the PTQ collapse (135,309 →
114) — but it is **not** a quality-per-bit win over a mature one-shot quantizer.

### 6.2 The quantizer is not the lever (RT-124..127)

A full one-shot PTQ toolbox on Llama-160M (CE_fp 3.15, ternary 11.66):

| lever | best one-shot effect |
| --- | --- |
| scale granularity (block/row) | +2.36 / +1.84 nats (partial; needs per-block runtime; row is foldable) |
| scale/threshold objective | absmean already optimal (MSE/threshold hurt) |
| AWQ/SmoothQuant diagonal scaling | +0.14 nats |
| GPTQ/Hessian assignment | +0.51 nats (6% of the nearest→FP gap) |
| signed-epsilon 2-bit codebook | does not beat ternary |

No lever makes one-shot conversion usable; the best stack stays at PPL in the thousands,
while a short teacher-free CE pass reaches ~114. The reconstruction residual is
codebook-dominated and roughly granularity-invariant. **Conclusion: adaptation/data, not
quantizer design, is the lever.**

### 6.3 gpt-oss / MoE is out of scope (RT-117/118)

gpt-oss-20b ships MXFP4 (experts already 4-bit). The official GGUF is 12.11 GB and even a
2-bit Q2_K is 11.47 GB — every quant bottoms at a ~11.5 GB floor, so ternary adds <1 GB.
The recipe's value is dense FP-weight models, not already-low-bit MoE.

## 7. Limitations

- **Factual parity (open).** "Usable-tier" means non-degenerate/diverse generation, not
  correct facts; the WikiText-CE model is weaker than FP/Q2_K on factual content. This is
  the main remaining gap and is a data/objective problem.
- **PPL-per-bit.** We do not beat Q2_K on PPL (Section 6.1).
- **Statistics.** Recovery is reported at a single seed (G6 open).
- **Cross-tool PPL.** PyTorch CE and llama.cpp perplexity differ in absolute value
  (tokenization/windowing); we keep comparisons within one tool and report relative deltas.
- **Architecture scope.** Results are LLaMA-family; the gpt-oss negative bounds the claim.
- **Runtime portability.** Ternary I2_S/TL1 is broken on the local Apple-M5 build
  (NEON-detect + clang LUT blowup); all runtime claims are x86/Linux.

## 8. Future Work

The locked conclusion points the next effort at adaptation/data, not quantizer/codebook:

1. **Factual recovery (G10):** run the
   [Factual Gap Experiment Plan](./factual_gap_experiment_plan.md): first measure the
   current FP/Q2_K/adapted-I2_S factual gap (FACT-001), then test instruction-style
   adaptation data, longer/better-data CE, and repetition-aware / free-run objectives to
   close the gap and reduce reliance on a decode-time repetition penalty.
2. **Cheap carry-forward:** per-output-channel (row) scale is a foldable one-shot init
   improvement (+1.84 nats) worth folding into the adaptation init.
3. **Scale-up:** confirm storage/speed/recovery on a stronger small base model.
4. **Optional diagnostic:** a pairwise/Hadamard phase-rotation probe to close any residual
   assignment loophole (not the main path — GPTQ already gained only 6%).

## Appendix A: Reproduction

Drivers (run on x86 with a built bitnet.cpp; PyTorch screens need only a GPU):

```text
rt112_x86_arena.py        faithful export parity (Path A vs A')
rt113_storage_latency.py  storage + llama-bench
rt114_scaleup.py          pretrained scale-up (storage/latency/parity)  [--model-id]
rt116_quality_recovery.py teacher-free CE recovery (QR-001/002/003)     [--grad-accum-steps, --train-norms/--train-lm-head]
rt121_baseline_panel.py   FP/RTN/Q2_K/Q3_K_M/Q4_0/OURS one-tool panel
rt122_prompt_panel_gguf.py 1.1B generation panel (greedy)
rt123_sensitivity_scan.py mixed-bit per-group sensitivity (demoted)
rt124a/b/c, rt125, rt127  quantization-aware toolbox sweep
rt129_decoding_probe.py   decoding stability (the usability rescue)
```

Result JSONs live in `reports/rt1xx_*.json`. The G1 budget-scaled run uses the
[G1 runbook](./g1_budget_scaling_runbook.md).

## Appendix B: Figure / Table checklist (A2)

| # | figure | data source | status |
| --- | --- | --- | --- |
| 1 | storage scale law | rt113/114/115 jsons | have |
| 2 | tg speedup scale law | rt113/114/115 jsons | have |
| 3 | PTQ collapse → CE recovery | rt116 + rt120 jsons | have |
| 4 | adapted i2_s vs f16 (nats) | rt116/120 QR-003 | have |
| 5 | baseline panel vs Q2_K | rt121 json | have |
| 6 | greedy generation panel | rt122 json | have |
| 7 | decoding rescue | rt129 json | have |
| A | gpt-oss MXFP4 floor | rt117 + Stage-0 sizes | have (size ladder table) |

## Appendix C: Missing-evidence list (A3)

- **Factual eval** (G10): no instruction/QA benchmark yet — only WikiText CE + degeneration
  tags. Needed before any "usable assistant" wording. Planned in
  [Factual Gap Experiment Plan](./factual_gap_experiment_plan.md), starting with
  FACT-001 current factual gap panel.
- **Seeds** (G6): single-seed recovery; add 2–3 seeds on 160M for variance bars.
- **1.1B baseline panel**: RT-121 is 160M-only; a 1.1B OURS-vs-Q2_K PPL point would
  strengthen Figure 5's generality.
- **B5 contrast**: no "no-scale ternary QAT" arm — would isolate the per-tensor-gamma
  contribution from "any QAT".
- **Latency error bars**: f32/f16 tg are noisy on shared CPU; a quiet host / more reps
  would tighten Figure 2's absolute numbers (ratios are stable).
- **Real hybrid GGUF**: mixed-bit (RT-123) used a PyTorch proxy; no `--tensor-type` in the
  pinned fork, so a real hybrid would need GGUF surgery (only if mixed-bit is revisited).
