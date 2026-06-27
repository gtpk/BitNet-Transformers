# Literature Deep Dive 02: CAT-Q / BitTern

Document position: [Index](./index.md) -> [Literature Positioning Map](./literature_positioning_map.md) -> second deep dive.

Paper:

- CAT-Q: Cost-efficient and Accurate Ternary Quantization for LLMs
- arXiv: https://arxiv.org/abs/2606.26650
- Code/project link in paper: https://github.com/IntelChina-AI/BitTern
- Status checked: 2026-06-27

## One-Line Read

CAT-Q is a stronger threat than PTQTP in one specific way: it claims that **single
ternary-plane PTQ can work** if the weight distribution and ternarization path are made
learnable and smooth during calibration.

If reproducible, CAT-Q attacks the premise behind our current pivot:

```text
"pure one-plane b1.58 is too weak, so we need planes/hybrid capacity"
```

CAT-Q says:

```text
"maybe one-plane b1.58 is enough, but your optimization path was too crude."
```

## What They Claim

From the arXiv abstract and introduction:

| claim | meaning |
| --- | --- |
| PTQ, not QAT | quantizes pretrained LLMs without massive retraining |
| 512 calibration samples | about 1M calibration tokens if length 2048 |
| 1.7B-8B models | claims better than BitNet v1/v2 families trained on 100B tokens |
| 14B-235B models | claims 8-60 hours on 8 A100-80GB GPUs |
| diverse architectures | dense and MoE models are included |
| ternary weights, optional 8-bit activations | weight-only ternary and W1.58A8 variants |
| ICML 2026 oral | high visibility and likely strong review signal |

This is a direct external challenge to our current all-I2_S conversion story.

## Core Method

CAT-Q has two coupled pieces:

```text
LM = learnable modulation
ST = softened ternarization
```

### 1. Learnable Modulation

Instead of applying BitNet-style static absmean ternarization directly to `W`, CAT-Q
learns factors that reshape the pre-trained weight distribution and threshold.

The paper writes the transformed weight as:

```text
W_hat = (W - mu) / alpha
mu    = mu_0 + delta_mu * alpha_0
alpha = delta_alpha * alpha_0
Delta = delta_Delta * Delta_0
```

where:

```text
mu_0     = mean(W)
alpha_0  = mean(abs(W - mu_0))
delta_*  = learnable calibration factors
```

Important nuance:

```text
They use mu to choose better ternary codes,
but keep the deployed approximation hardware-friendly as W ~= alpha*T, not W ~= alpha*T + mu.
```

That is a major insight for us: a transformation can be used as a **code-selection
proxy** even if the runtime remains simple.

### 2. Softened Ternarization

CAT-Q does not jump immediately into hard `{-1,0,+1}`. It uses a differentiable
transition from identity-like mapping toward ternary mapping:

```text
early calibration: continuous / smooth ternarization
late calibration: hard ternarization
```

The point is not just STE. The point is a controlled optimization path:

```text
make the quantization target gradually harder
so the small number of learnable factors converges stably.
```

This connects to our failed/weak one-shot levers:

```text
RT-124 scale search was static.
RT-125 GPTQ was assignment-aware but not a smooth optimization path.
FACT/CE adaptation changed model behavior, not the quantizer path.
```

CAT-Q attacks the quantizer path itself.

### 3. Sliding-Layer Optimization

CAT-Q also uses a sliding-layer output reconstruction scheme rather than isolated
single-layer weight reconstruction. This matters because our RT-123 found layer effects
are non-additive:

```text
restoring one layer can worsen downstream ternary mismatch.
```

CAT-Q's sliding-layer design is one external answer to that exact issue.

## Why CAT-Q Matters To Us

| our current finding | CAT-Q's challenge |
| --- | --- |
| absmean ternary is strong among simple static rules | learn modulation, do not stay static |
| one-shot pure ternary collapses | use calibration optimization, not one-shot |
| GPTQ-lite assignment helps only a little | use smooth ternarization + modulation + sliding-layer reconstruction |
| content-KL is the first factual lever | maybe better quantized base reduces need for factual repair |
| hybrid/multi-plane may be needed | maybe one-plane is enough with better PTQ |

So CAT-Q is not merely another baseline. It tests whether our diagnosis

```text
"one-plane capacity is insufficient"
```

is actually true, or whether the real issue was:

```text
"our one-plane optimization was too primitive."
```

## Where CAT-Q Is Ahead Of Us

| axis | CAT-Q | us |
| --- | --- | --- |
| PTQ quality claim | strong, large-scale | weak all-I2_S factual quality |
| model scale | 1.7B-235B | 160M/1.1B |
| calibration | 512 samples | CE/content-KL adaptation, not comparable yet |
| optimization | learnable modulation + smooth path | static quantizers + STE adaptation |
| publication signal | ICML 2026 oral | internal project |
| architecture coverage | dense + MoE claim | dense LLaMA verified; gpt-oss out of scope |

If CAT-Q reproduces, it should become a required baseline for Paper 2 and a candidate
conversion engine for the product path.

## Where We May Still Differ

| axis | our possible edge |
| --- | --- |
| runtime export | CAT-Q paper says compression/acceleration, but we need to inspect actual runtime path and whether it maps to I2_S/TL/Q kernels |
| memory-traffic scorecard | we already measure whole-file, target-linear, tg speed, parity |
| factual score | CAT-Q likely reports benchmarks; our fixed factual panel can expose retention failure modes |
| objective recovery | content-KL could be applied on top of CAT-Q if factual gap remains |
| adaptive allocation | CAT-Q appears one-plane and layer-sliding, not necessarily budgeted hybrid |

The best future framing:

```text
CAT-Q may be a better ternary initializer/conversion engine.
Our contribution can wrap it with:
  runtime export,
  factual recovery,
  and budget-aware deployment decisions.
```

## Questions / Doubts To Verify

| question | why it matters | how to test |
| --- | --- | --- |
| Is the BitTern code public and runnable? | arXiv links GitHub, but current web fetch did not inspect files | clone and run minimal example |
| What exact ternary artifact is produced? | I2_S requires `alpha*T`; CAT-Q uses modulation proxy | inspect saved weights/scales |
| Does CAT-Q map to bitnet.cpp I2_S directly? | product path needs real runtime | try Path A' export or custom layout mapping |
| What is real storage? | learnable factors/scales may add overhead | measure whole-file and target-linear bytes |
| Does it preserve facts? | our current bottleneck is factual quality | run FACT panel |
| Does it beat Q2_K under same runtime? | Q2_K is practical baseline | compare size, tg, PPL, fact |
| Does content-KL help after CAT-Q? | separates quantizer vs objective | CAT-Q -> content-KL adaptation |
| Is 512-sample calibration enough on small models? | claims may rely on large-model robustness | test Llama-160M / TinyLlama-1.1B |

## Insight For Our Next Experiments

CAT-Q suggests a smaller experiment than full PTQTP reproduction:

```text
learn only a few per-layer/per-group modulation parameters
use a softened ternary transition
materialize Wq = alpha*T
export via our known I2_S Path A'
```

This is attractive because it keeps our existing runtime path:

```text
I2_S can still store alpha*T if final artifact is one-plane ternary.
```

That is a major advantage over PTQTP:

```text
PTQTP dual-plane may need new runtime/layout.
CAT-Q one-plane may reuse I2_S if final representation is alpha*T.
```

So the next local reproduction should not start with 235B or full CAT-Q. It should be:

```text
CAT-Q-lite:
  learn delta_mu, delta_alpha, delta_delta
  smooth ternarization schedule
  small calibration set
  final Wq = alpha*T
  I2_S export through Path A'
```

## Proposed Minimal Reproduction

First target: Llama-160M, because we already have all baselines.

| step | action | expected signal |
| --- | --- | --- |
| 1 | implement per-layer or per-row/group modulation factors | one-plane ternary codes change vs absmean |
| 2 | use continuous-to-hard ternary schedule | stable loss over calibration |
| 3 | materialize `alpha*T` and export I2_S | reuse RT-112/114 path |
| 4 | evaluate PPL, FACT panel, generation tags | compare to RT-121/FACT baselines |
| 5 | optionally apply content-KL | test quantizer+objective combination |

Pass signal:

```text
CAT-Q-lite one-plane improves over our absmean all-I2_S by a large margin
and remains I2_S-compatible.
```

Fail signal:

```text
CAT-Q-lite does not move FACT/PPL enough at 160M/1.1B.
Then PTQTP/multi-plane capacity becomes more likely necessary.
```

## How This Changes Our Roadmap

| old direction | update after CAT-Q |
| --- | --- |
| jump to hybrid/multi-plane | first check whether better one-plane optimization works |
| content-KL as main recovery | keep it, but use after a stronger PTQ initializer |
| PTQTP as next implementation | PTQTP remains important, but CAT-Q-lite may be lower-cost because it preserves I2_S |
| Paper 2 claim | must say "our simple quantizers fail", not "one-plane PTQ cannot work" |
| Paper 4 | should branch: CAT-Q-lite one-plane first, PTQTP/hybrid if plateau |

## Bottom Line

CAT-Q is the paper that most directly tests whether our current pivot to capacity
expansion is premature.

The disciplined next move is:

```text
Try CAT-Q-lite before committing to multi-plane runtime complexity.
If CAT-Q-lite works, we get a much cleaner product path:
  one-plane I2_S + better calibration + content-KL.
If it fails, PTQTP/hybrid capacity is justified with stronger evidence.
```

