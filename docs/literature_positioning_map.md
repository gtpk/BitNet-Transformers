# Literature Positioning Map

Document position: [Index](./index.md) -> external literature map after the project
split into multiple papers.

Last updated: 2026-06-27.

Related:

- [Paper Series Plan](./paper_series_plan.md)
- [Paper Evidence Matrix](./paper_evidence_matrix.md)
- [Literature Deep Dive 01: PTQTP](./literature_deep_dive_ptqtp.md)
- [Literature Deep Dive 02: CAT-Q](./literature_deep_dive_catq.md)
- [Literature Deep Dive 03: TWLA](./literature_deep_dive_twla.md)
- [Literature Deep Dive 04: PT2-LLM](./literature_deep_dive_pt2_llm.md)
- [Literature Deep Dive 05: KD / KL Objectives](./literature_deep_dive_kd_kl.md)
- [Literature Deep Dive 06: Precision Scaling Laws](./literature_deep_dive_precision_scaling.md)
- [Fair Comparison Framework](./fair_comparison_framework.md)
- [Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md)
- [Hybrid / Variable BitNet Conversion Plan](./hybrid_variable_bitnet_conversion_plan.md)
- [Factual Gap Experiment Plan](./factual_gap_experiment_plan.md)

## Executive Read

The field moved fast. Our project is no longer alone in asking whether existing
models can be moved toward ternary/b1.58 after training. Several 2025-2026 papers go
directly after post-training ternarization and are ahead of us on quality and scale.

Our strongest lane is therefore not:

```text
"we are the first to ternarize a pretrained LLM"
```

It is:

```text
systems-grounded b1.58 conversion:
  real I2_S runtime,
  memory-traffic scaling,
  honest failure boundaries,
  factual/objective recovery analysis,
  and a path toward low-resource usable artifacts.
```

## Main Literature Clusters

### 1. Native BitNet Family

| work | what it shows | relevance to us |
| --- | --- | --- |
| BitNet: Scaling 1-bit Transformers for LLMs (2023) | BitLinear as a replacement for `nn.Linear`, trained natively | establishes that b1.58/native 1-bit can work, but not as post-hoc conversion |
| The Era of 1-bit LLMs / BitNet b1.58 (2024) | every weight ternary `{-1,0,+1}`; native training can match FP at same model size/tokens | proves the target space is viable if trained into it |
| BitNet b1.58 2B4T Technical Report (2025) | open 2B native b1.58 model trained on 4T tokens | the quality bar for native training; far more pretraining cost than our conversion goal |
| BitNet a4.8 / BitNet v2 (2024-2025) | activation outlier handling, 4-bit activations, Hadamard transform | points to activation smoothing/rotation as a serious axis |
| bitnet.cpp (2025) | TL and I2_S runtimes for efficient ternary inference | our Paper 1 directly lives here; we validated Path A' export into I2_S |

Takeaway:

```text
Native BitNet success is real, but it is not ordinary quantization. It is an
architecture/training recipe. This supports our Paper 2 framing.
```

### 2. Direct Competitors: Post-Training Ternarization

These are the closest to our conversion goal.

| work | central idea | where it is ahead | idea we can borrow |
| --- | --- | --- | --- |
| TernaryLLM (2024) | Dual Learnable Ternarization + feature knowledge distillation | stronger quality recovery than our early CE-only path | asymmetric shift/scale, feature-level objective |
| PT2-LLM (2025) | asymmetric ternary quantizer, iterative ternary fitting, activation-aware grid alignment, structural reordering | directly targets PTQ ternarization | asymmetric `mu + alpha*T`, structural column reordering |
| PTQTP (2025/2026) | dual ternary trit-planes, progressive approximation, model-agnostic deployment | admits pure 1.58 may be under-capacity and uses `2 x 1.58` capacity | multi-strip / trit-plane idea for Paper 4 |
| TWLA (2026) | E2M asymmetric ternary quantizer, Kronecker orthogonal tri-modal shaping, inter-layer aware activation mixed precision | W1.58A4, activation-aware, rotation-aware, inter-layer-aware | our rotation idea is validated, but should be structured and layer-aware |
| CAT-Q (2026) | learnable modulation + softened ternarization; 512 calibration samples; 1.7B-8B and 14B-235B claims | far ahead on scale and practical PTQ claim | soft ternarization / learnable modulation as a stronger baseline |

Deep dives:

- [PTQTP deep dive](./literature_deep_dive_ptqtp.md)
- [CAT-Q deep dive](./literature_deep_dive_catq.md)
- [TWLA deep dive](./literature_deep_dive_twla.md)
- [PT2-LLM deep dive](./literature_deep_dive_pt2_llm.md)

Takeaway:

```text
The frontier already agrees with our diagnosis:
  pure nearest ternary is too hard;
  distribution shaping, asymmetric parameters, rotation, interaction-aware allocation,
  and extra representational planes are the promising levers.
```

This weakens any claim that our quantizer experiments are the final word. It strengthens
the claim that our negative result is correct and that Paper 4 should not be naive
mixed-bit; it should compare against these stronger mechanisms.

### 3. General Low-Bit PTQ and Rotation Methods

| work | central idea | relevance |
| --- | --- | --- |
| GPTQ | second-order one-shot weight quantization | our RT-125 used the same spirit; gain was real but too small for pure ternary |
| AWQ | activation-aware salient channel scaling | our AWQ-like diagonal probe was weak, but AWQ is still a required baseline |
| OmniQuant | learnable clipping and equivalent transformations | stronger version of scale/threshold/equivalent-transform tuning |
| QuIP | incoherence processing with random orthogonal transforms and adaptive rounding | supports the rotation/incoherence direction |
| QuaRot | output-preserving rotations to remove activation outliers | rotation can be exact in FP and beneficial for quantization |
| SpinQuant | learned rotations outperform random rotations | if we do rotation, learn it; do not just hand-wave `e^{i theta}` |
| TesseraQ | block reconstruction for ultra-low-bit PTQ | suggests block reconstruction could be stronger than our simple layer probes |

Takeaway:

```text
Our "rotation as complex phase" intuition points in the right family, but the
literature implements it as real orthogonal transformations/Hadamard/Kronecker/learned
rotations, not literal complex weights.
```

### 4. Precision / Quantization Scaling Laws

| work | key message | relevance |
| --- | --- | --- |
| Scaling Laws for Precision | lower precision reduces effective parameter count; PTQ degradation grows with training/token regime | supports the user's capacity concern: b1.58 may need more width/planes/capacity |
| Low-Bit Quantization Favors Undertrained LLMs | well-trained/overtrained models can be more sensitive to low-bit quantization | warns that big modern dense checkpoints may be harder to convert than small undertrained ones |
| Scaling Laws for Mixed Quantization | larger models can preserve performance with higher low-precision ratio; finer granularity helps | supports hybrid/variable capacity as a budgeted topology problem |

Takeaway:

```text
The user's "1/4 capacity is not enough; increase channels/strips/layers" intuition has
literature support. It should be framed as effective-parameter/capacity restoration,
not as a vague bigger-is-better move.
```

### 5. Factual Forgetting / Objective Literature

| work family | key message | relevance |
| --- | --- | --- |
| catastrophic forgetting in LLM tuning | fine-tuning can overwrite prior knowledge; forgetting relates to steps/params/loss landscape | matches FACT-001/002: CE recovered fluency but lost facts |
| KL/replay/anti-forgetting methods | anchors can help, but the choice of tokens/distribution matters | matches FACT-003B/003C: raw KL copied EOS; content-KL helped |

Takeaway:

```text
Our factual gap is not a quantization artifact. It is a fine-tuning/objective problem
inside a severely constrained parameterization.
```

### 6. Content-KL Nearest Neighbors

Search result: the exact recipe name `content-KL` and the exact combination

```text
KL(base || student) over content vocabulary only
exclude EOS/BOS/PAD/special stop tokens from the KL distribution
renormalize
use it to recover factual behavior in a b1.58/I2_S converted model
```

does not appear to be a standard named method in the papers checked so far. But the
components are known. Our method is best framed as a specific composition of known
distillation/masking ideas for a low-bit conversion failure mode.

| neighbor | what exists | how it differs from our content-KL |
| --- | --- | --- |
| completion/assistant-only SFT | TRL supports loss only on assistant/completion tokens | masks sequence positions, not EOS/special vocabulary mass inside KL |
| LLM KD with KL variants | MiniLLM uses reverse KL; DistiLLM uses skew KL; AKL mixes forward/reverse KL | changes KL direction/weighting, not specifically stop-token removal |
| selective/token-aware KL | SHRED builds modified KL targets and preserves/demotes different token positions | closest conceptual neighbor; selects sequence positions/top-K targets, while ours masks stop-token vocabulary entries |
| length/stop bias literature | label smoothing and local sequence objectives can induce output-length bias | explains why raw KL can copy short/empty-answer behavior |

Interpretation:

```text
content-KL is probably not a new divergence theory.
It is a practical objective design:
  copy content distribution,
  do not copy the teacher's stop decision.
```

Safe claim:

```text
In our b1.58 conversion setting, content-KL is the first objective we found that
improves factual score without empty collapse.
```

Unsafe claim:

```text
content-KL is a generally novel KL method.
```

Deep dive:

- [KD / KL objectives deep dive](./literature_deep_dive_kd_kl.md)

## Where Others Are Further Ahead

| area | stronger external work | implication |
| --- | --- | --- |
| direct PTQ ternarization quality | CAT-Q, TWLA, PTQTP, PT2-LLM, TernaryLLM | Paper 2 must cite these; our simple quantizer sweep is not SOTA |
| rotation/distribution shaping | TWLA, QuIP, QuaRot, SpinQuant, BitNet v2 | rotation should become a serious candidate, not just an appendix idea |
| representational capacity | PTQTP dual trit-planes, BitNet Reloaded hidden-size expansion | Paper 4 should test multi-strip/variable capacity |
| large-model scale | CAT-Q claims up to 235B; PTQTP claims 70B families | our current 160M/1.1B runs are exploratory, not scale leadership |
| factual/benchmark quality | Q2_K/FP still dominate our factual panel | Paper 3 is a method signal, not a product win yet |

## Where We Still Have A Distinct Lane

| lane | why it remains useful |
| --- | --- |
| Path A' I2_S export | many papers report quantization quality; fewer show the exact bitnet.cpp I2_S export path and runtime parity for converted artifacts |
| memory-traffic-first measurements | our storage/speed scale law directly matches the low-resource product goal |
| honest negative boundaries | we have a clean record showing which simple ideas fail: nearest, GPTQ-lite, signed-eps, raw KL, data-only |
| factual-objective diagnosis | content-KL vs raw-KL EOS failure is a concrete insight not covered by most PTQ papers |
| product framing | the target is not leaderboard PPL; it is cheap conversion into small/fast/usable local artifacts |

## Ideas To Borrow Next

Priority order if FACT-003C plateaus:

### A. Stronger PTQ Baseline Reproduction

Before claiming Paper 2 broadly, compare against at least one strong recent method:

```text
  CAT-Q if code is runnable
  PT2-LLM if code/formulas are enough for PT2-lite
  TWLA if code is runnable
  PTQTP if code is available enough
```

Minimum goal:

```text
  Run one method on Llama-160M or TinyLlama-1.1B with the same factual/PPL/runtime
scorecard.
```

### A2. Stronger Factual Objective Variants

Before inventing another factual loss from scratch, combine content masking with known
KD objective axes:

```text
content-RKL from MiniLLM
skew-content-KL from DistiLLM
content-AKL from AKL
sparse on/off-policy content-KL if exposure bias remains
```

### B. Structured Rotation, Not Free-Form Complex Numbers

Use the rotation literature's form:

```text
Hadamard / Kronecker / orthogonal rotation
foldable into adjacent linear layers
preserve FP function before quantization
then ternarize in the rotated basis
```

This directly connects our `e^{i theta}` intuition to QuIP/QuaRot/SpinQuant/TWLA.

### C. Multi-Strip / Trit-Plane Capacity

PTQTP strongly supports the idea:

```text
W ~= alpha_1*T_1 + alpha_2*T_2 + ...
```

This is exactly the user's "increase strips/channels/layers because 1.58 alone cannot
hold the function" intuition. It is more plausible than signed-epsilon because it adds
rank/capacity rather than merely changing the zero code.

Deep dive:

- [Precision scaling laws deep dive](./literature_deep_dive_precision_scaling.md)

### D. Asymmetric Ternary Parameterization

PT2-LLM and TernaryLLM emphasize shifts:

```text
Wq = mu + alpha*T
```

Our I2_S runtime does not directly store `mu`, so this is not free. But it can be tested
as:

```text
row bias fold if mathematically possible
or hybrid fallback where only selected layers use asymmetric reconstruction
```

### E. Inter-Layer-Aware Allocation

Our RT-123 found non-additivity. TWLA makes this explicit with adjacent-layer
interaction costs. If we do DP again, it should not be independent item knapsack; it
should be sequence/interaction-aware:

```text
cost(layer_i, choice_i, choice_{i+1})
```

That turns the selector into a chain-structured dynamic program rather than a naive
additive knapsack.

## Update To Our Paper Split

| paper | literature effect |
| --- | --- |
| Paper 1 Systems | still strong; bitnet.cpp is the anchor |
| Paper 2 Conversion Limits | must cite CAT-Q/TWLA/PTQTP as methods that go beyond our simple levers |
| Paper 3 Content-KL | remains distinct; connect to forgetting/KL replay literature |
| Paper 4 Hybrid Capacity | becomes more important, because PTQTP/precision scaling laws support capacity expansion |

## Immediate Recommendation

Do not rush into another random idea. The next disciplined step is:

```text
1. Finish FACT-003C lambda=0.5.
2. If content-KL plateaus below ~0.3, run HYBRID-001A.
3. In parallel, pick one strong external baseline to reproduce:
   CAT-Q or TWLA first, PTQTP if code is ready.
4. Reframe rotation as structured orthogonal rotation, not literal complex storage.
```

If one of CAT-Q/TWLA/PTQTP is reproducible and clearly beats our path, we should not
fight it. We should absorb it as the conversion engine and keep our contribution on
runtime/export/factual/objective/product scorecard.

## Source List

- BitNet: Scaling 1-bit Transformers for Large Language Models — https://arxiv.org/abs/2310.11453
- The Era of 1-bit LLMs — https://arxiv.org/abs/2402.17764
- BitNet b1.58 2B4T Technical Report — https://arxiv.org/abs/2504.12285
- Bitnet.cpp — https://arxiv.org/abs/2502.11880
- BitNet a4.8 — https://arxiv.org/abs/2411.04965
- BitNet v2 — https://arxiv.org/abs/2504.18415
- TernaryLLM — https://arxiv.org/abs/2406.07177
- PT2-LLM — https://arxiv.org/abs/2510.03267
- TWLA — https://arxiv.org/abs/2606.13054
- CAT-Q — https://arxiv.org/abs/2606.26650
- PTQTP — https://arxiv.org/abs/2509.16989
- PB-LLM — https://arxiv.org/abs/2310.00034
- ARB-LLM — https://arxiv.org/abs/2410.03129
- GPTQ — https://arxiv.org/abs/2210.17323
- AWQ — https://arxiv.org/abs/2306.00978
- OmniQuant — https://arxiv.org/abs/2308.13137
- QuIP — https://arxiv.org/abs/2307.13304
- QuaRot — https://arxiv.org/abs/2404.00456
- SpinQuant — https://arxiv.org/abs/2405.16406
- TesseraQ — https://arxiv.org/abs/2410.19103
- Scaling Laws for Precision — https://arxiv.org/abs/2411.04330
- Low-Bit Quantization Favors Undertrained LLMs — https://arxiv.org/abs/2411.17691
- Scaling Laws for Mixed Quantization — https://arxiv.org/abs/2410.06722
- MiniLLM / On-Policy Distillation — https://arxiv.org/abs/2306.08543
- DistiLLM — https://arxiv.org/abs/2402.03898
- Rethinking KL Divergence in KD for LLMs / AKL — https://arxiv.org/abs/2404.02657
- SHRED token-selective self-distillation — https://arxiv.org/abs/2605.07482
- TRL SFTTrainer assistant/completion-only loss — https://huggingface.co/docs/trl/en/sft_trainer
- Implicit Length Bias of Label Smoothing — https://arxiv.org/abs/2205.00659
