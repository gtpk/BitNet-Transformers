# Literature Deep Dive 04: PT2-LLM

Document position: [Index](./index.md) -> [Literature Positioning Map](./literature_positioning_map.md) -> fourth deep dive.

Paper:

- PT^2-LLM: Post-Training Ternarization for Large Language Models
- arXiv: https://arxiv.org/abs/2510.03267
- Code page: https://github.com/XIANGLONGYAN/PT2-LLM
- Status checked: 2026-06-27

## One-Line Read

PT2-LLM is the cleanest hint for `mu + alpha*T` conversion. It says ordinary
symmetric ternary is too rigid for pretrained weights, so PTQ needs:

```text
asymmetric row-wise ternary grid
+ iterative assignment/grid fitting
+ activation-aware grid relocation
+ structural column reordering
```

For us, PT2-LLM is less about factual recovery and more about the missing
pre-adaptation quantizer:

```text
Can we initialize the b1.58 model much closer to FP before CE/content-KL adaptation?
```

## What They Claim

| component | meaning |
| --- | --- |
| ATQ | asymmetric ternary quantizer |
| ITF | iterative ternary fitting; alternates grid solve and ternary assignment |
| AGA | activation-aware grid alignment; adjusts grid under output error |
| SSR | structural similarity-based reordering; groups similar columns into quantization blocks |
| PTQ | no retraining/backprop; uses 128 Wikitext2 calibration samples in the paper |
| result claim | competitive with SOTA 2-bit PTQ at lower average bitwidth, with speedup |

This is a direct competitor to our one-shot quantizer study. The paper's framing is
very close to our diagnosis:

```text
nearest/threshold ternary fails because pretrained weights are biased, dispersed,
and outlier-heavy.
```

## Core Method

### 1. Asymmetric row-wise grid

Our I2_S-compatible form is:

```text
Wq = gamma * T
T in {-1, 0, +1}
```

PT2-LLM uses a shifted row-wise ternary grid:

```text
Wq = mu + alpha * T
T in {-1, 0, +1}
```

This matters because pretrained weight rows may have non-zero means. Native BitNet or
QAT can reshape distributions during training, but PTQ inherits the existing row bias.
So `mu` is not cosmetic; it is a way to avoid spending the three ternary symbols on a
miscentered distribution.

I2_S cannot store `mu` directly. That is the main compatibility tension.

### 2. ITF: iterative ternary fitting

Given current ternary assignment `T`, solve the best row-wise `mu, alpha`. Then, given
the grid `mu + alpha*T`, remap each weight to the closest of the three grid points.
Repeat until `T` stabilizes.

This is important for us because RT-124/125 showed that fixed threshold / simple GPTQ
assignment only recovers a small fraction of the all-ternary gap. PT2-LLM suggests the
right local solver is:

```text
grid solve -> flexible ternary assignment -> grid solve -> ...
```

rather than a one-pass threshold.

### 3. AGA: activation-aware grid alignment

ITF minimizes weight error. AGA then freezes the ternary assignment `T` and moves only
the grid parameters to reduce layer output error:

```text
min_{mu, alpha} || (W - (mu + alpha*T)) X ||_F^2
```

where `X` is calibration activation data. This matches a recurring lesson from our
experiments:

```text
weight reconstruction is not the same as output/CE/factual reconstruction.
```

The subtle part is that PT2-LLM freezes `T` during AGA. They report that greedily
changing `T` under the activation objective can overfit calibration. That gives us a
concrete design rule:

```text
change continuous grid under activation metric;
do not freely churn discrete ternary assignment unless validated.
```

### 4. SSR: structural similarity-based reordering

Blockwise ternarization is sensitive to which columns land in the same block. PT2-LLM
clusters or selects structurally similar columns so each block has a more compact
distribution and outliers are grouped with other outliers.

The key identity is simple:

```text
W X = (W P) (P^T X)
```

Column permutation can be made function-preserving if the corresponding activation
channels are permuted consistently. Unlike arbitrary rotation, a permutation is cheap.

This gives us a much cheaper alternative before full TWLA/QuaRot-style rotations:

```text
try reorder first; rotate later.
```

## Code / Reproduction Surface

The GitHub repository exists, but at the time checked it mostly exposes README/figures
and a TODO list saying post-training ternarization code, quantized models, and results
are to be released. So PT2-LLM is currently a paper-level baseline for us, not yet a
drop-in runnable comparator.

Implication:

```text
We can implement PT2-lite from the formulas before full reproduction is possible.
```

## Where PT2-LLM Is Ahead Of Us

| axis | PT2-LLM | us |
| --- | --- | --- |
| quantizer initialization | asymmetric `mu + alpha*T`, iterative fitting | mostly absmean `gamma*T` |
| activation-aware relocation | closed-form grid alignment under calibration output error | simple output-aware/GPTQ-lite probes only |
| structural reordering | explicit SSR block construction | not implemented |
| scale | LLaMA/LLaMA2/LLaMA3/Qwen3 up to large models | 160M/1.1B for our main data |
| benchmark breadth | WikiText2/C4 + seven QA tasks | local PPL/factual/prompt panels |

Hard truth:

```text
Our RT-124~127 negative result rules out simple quantizer tweaks, not PT2-style
asymmetric activation-aware fitting.
```

## Where We May Still Differ

| axis | our possible edge |
| --- | --- |
| I2_S runtime | PT2's `mu` is not I2_S-native; our Path A' is already bitnet.cpp-tested |
| factual objective | PT2 is PTQ/reconstruction; it does not address our FACT-003 raw-KL/content-KL issue |
| product scorecard | we measure storage, latency, degeneration, factual score, runtime parity |
| adaptation | PT2 is training-free; we can use PT2-lite as initialization before CE/content-KL |
| hybrid compatibility | `mu` may require selective fallback or folding; this naturally feeds Paper 4 |

## Compatibility Problem: Where Does `mu` Go?

PT2-LLM's best form is:

```text
y = (mu + alpha*T) x
  = alpha*T*x + mu*(1^T x)
```

I2_S can compute `alpha*T*x`, but not the extra dynamic term:

```text
mu*(sum_j x_j)
```

This term is input-dependent. It is not a normal static bias unless `sum_j x_j` is
constant, which is generally false.

So there are three realistic options:

| option | idea | cost |
| --- | --- | --- |
| PT2-init then project to I2_S | use ITF/AGA to choose better `T`, then drop/fold `mu` approximately | runtime stays I2_S, possible quality loss |
| hybrid selected layers | allow asymmetric/fp correction only in layers where `mu` matters | small runtime complexity |
| custom asymmetric ternary kernel | support `mu + alpha*T` exactly | bigger systems project |

This is why PT2-LLM is a hint, not an immediate replacement for our I2_S path.

## Minimal PT2-Lite Plan

### PT2-lite A: ITF-only, I2_S-compatible

```text
For each row:
  initialize mu = mean(W_row)
  initialize T from centered W_row
  repeat <= 10:
    solve mu, alpha for fixed T
    update T by nearest grid point
  project to I2_S by either:
    A1: keep T, set gamma = mean(abs(W_row[T != 0]))
    A2: keep T, set gamma = least-squares alpha without mu
```

Questions:

```text
Does better T alone improve I2_S PPL/fact after projection?
How much quality lives in T vs in mu?
```

### PT2-lite B: AGA with frozen T

```text
Collect calibration activations X.
Freeze T from ITF.
Solve output-aware mu, alpha.
Then test:
  exact asymmetric dense/f16 reference
  projected I2_S version
```

Questions:

```text
Does activation-aware relocation improve output/factual score?
Does dropping mu erase most of the gain?
```

### PT2-lite C: SSR-only before our current quantizer

```text
For each block:
  build column similarity / residual mean selector
  reorder columns into homogeneous blocks
  run existing gamma*T ternary
  undo/propagate permutation as needed
```

Questions:

```text
Can cheap permutation reduce the one-shot gap without non-I2_S terms?
Is SSR a better first step than arbitrary rotation?
```

### PT2-lite D: combine with content-KL

```text
PT2-lite init -> CE/content-KL adaptation -> I2_S export -> FACT panel
```

Questions:

```text
Does a better PTQ start reduce the amount of adaptation needed?
Does it preserve factual knowledge better than absmean init?
```

## How This Changes Our Roadmap

PT2-LLM inserts a new branch before TWLA-lite:

```text
CAT-Q-lite:
  keep one-plane I2_S, improve optimization path

PT2-lite:
  test asymmetric grid/reordering, then see whether gains survive I2_S projection

TWLA-lite:
  add rotation + activation-bit allocation if output/activation bottlenecks remain

PTQTP-lite:
  add extra trit-plane capacity if one-plane still underfits
```

The key decision is:

```text
If PT2 gains require mu, it becomes a hybrid/custom-kernel idea.
If PT2 improves T even after mu projection, it becomes a cheap I2_S initializer.
```

## Questions / Doubts To Verify

| question | why it matters |
| --- | --- |
| Is code actually released beyond README? | determines whether we reproduce or implement PT2-lite ourselves |
| How much of PT2's gain comes from `mu`? | I2_S cannot store dynamic `mu*(sum x)` cheaply |
| Does SSR survive transformer channel wiring? | permutations must be folded consistently across adjacent layers |
| Does AGA overfit our small calibration sets? | PT2 freezes T because assignment updates overfit |
| Does PT2 improve facts or only PPL/QA averages? | our open gap is factual retention |
| Does PT2's speed rely on a custom ternary runtime? | we need bitnet.cpp/I2_S compatibility or a clear replacement runtime |

## Bottom Line

PT2-LLM gives us a very concrete lesson:

```text
The ternary codebook should not be treated as {-gamma, 0, +gamma} only.
For pretrained models, the useful grid is often shifted and activation-aligned.
```

But it also exposes the runtime catch:

```text
mu + alpha*T is not I2_S-native.
```

Therefore the most useful next test is not full PT2 reproduction. It is:

```text
PT2-lite: learn better T/mu/alpha, then measure how much survives after projection
back into pure I2_S gamma*T.
```

If it survives, PT2-lite becomes our best cheap initialization before content-KL.
If it does not, PT2 supports the hybrid/custom-kernel direction rather than the pure
I2_S conversion direction.

## Source List

- PT2-LLM arXiv — https://arxiv.org/abs/2510.03267
- PT2-LLM GitHub — https://github.com/XIANGLONGYAN/PT2-LLM
