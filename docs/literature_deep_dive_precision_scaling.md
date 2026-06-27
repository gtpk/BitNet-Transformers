# Literature Deep Dive 06: Precision Scaling Laws

Document position: [Index](./index.md) -> [Literature Positioning Map](./literature_positioning_map.md) -> sixth deep dive.

Papers:

- Scaling Laws for Precision — https://arxiv.org/abs/2411.04330
- Low-Bit Quantization Favors Undertrained LLMs — https://arxiv.org/abs/2411.17691
- Scaling Laws for Mixed Quantization in Large Language Models — https://arxiv.org/abs/2410.06722
- Status checked: 2026-06-27

## One-Line Read

Precision scaling laws give the mathematical language for the user's core intuition:

```text
1.58 bits is not just "smaller storage."
It can reduce the model's effective capacity.
```

Therefore, if a pretrained FP model does not fit into one b1.58 ternary plane, the
right response is not endless threshold tuning. The right response may be:

```text
increase effective capacity:
  more trit planes,
  selective higher precision,
  wider/chunked adapters,
  or train/adapt the model into the low-precision manifold.
```

This is the theoretical backbone for Paper 4 / HYBRID capacity.

## Core Concepts

### 1. Precision acts like effective parameter count

The precision-scaling view says a model with `N` parameters at `P` bits behaves like it
has fewer useful parameters than the same `N` at high precision:

```text
N_effective = N * g(P)
0 < g(P) <= 1
g(P) decreases sharply at very low P
```

The exact fitted form depends on the paper and training setup, but the project-level
meaning is stable:

```text
lower precision is a capacity reduction, not only a storage compression.
```

For our setting:

```text
FP16 LLaMA -> one-plane b1.58
```

is not a harmless representation change. It asks the same architecture to perform
inside a smaller effective parameter budget.

### 2. The "1/4 capacity" intuition is legitimate

A rough storage ratio:

```text
1.58 / 16 ~= 0.099
```

or vs 8-bit:

```text
1.58 / 8 ~= 0.197
```

does not translate directly to parameter capacity, but it captures the pressure:

```text
one ternary plane is an extreme representational bottleneck.
```

So the user's intuition:

```text
"if b1.58 cannot contain the function, increase channels/strips/layers"
```

should be reframed as:

```text
restore N_effective while keeping memory traffic low.
```

This is a stronger, testable statement.

### 3. Undertrained models may quantize better

The "Low-Bit Quantization Favors Undertrained LLMs" result warns that models trained
for many tokens can become more sensitive to low-bit PTQ. Undertrained models often
have more slack/redundancy, so quantization damage is smaller.

For us, this means:

```text
pretrained checkpoint choice matters.
```

A very polished dense model may be harder to convert than a slightly undertrained,
redundant, or larger model. This also explains why one-shot b1.58 conversion can look
hopeless on a compact trained checkpoint even though native BitNet can work from
scratch:

```text
native training learns inside the low-precision manifold;
PTQ must project a high-precision solution into it.
```

### 4. Mixed quantization gives a budgeted capacity knob

Mixed-quantization scaling laws treat two quantities separately:

```text
quantization ratio: how much of the model is low precision
bitwidth: how low those parts go
```

This is exactly the structure we need:

```text
mostly I2_S,
small fraction Q2/Q3/F16,
chosen by impact per byte/token.
```

The lesson is not "use mixed precision forever." It is:

```text
capacity can be allocated non-uniformly.
```

That turns the problem from:

```text
Can all layers be 1.58 bits?
```

into:

```text
Where is one-plane b1.58 enough,
and where does the model need extra effective capacity?
```

## How This Explains Our Results

| our result | precision-scaling interpretation |
| --- | --- |
| one-shot ternary PTQ collapses | projection loses too much effective capacity at once |
| GPTQ/AWQ/signed-eps barely help | local quantizer tweaks cannot restore global capacity |
| CE/content-KL adaptation helps | training moves the function into the low-precision manifold |
| facts remain hard | factual behavior may require capacity/objective not restored by WikiText CE |
| Q2_K beats our pure b1.58 PPL | Q2_K has higher effective capacity and mature kernels |
| PTQTP dual trit-plane looks plausible | adding planes increases effective capacity while staying ternary-like |
| hybrid F16/Q2 pockets look plausible | selective precision restores capacity where one-plane fails |

This is the key reframing:

```text
If the failure is effective capacity,
then "better thresholds" are second-order.
```

## Mathematical Framing For Paper 4

Let each layer choice `c_l` have:

```text
memory cost: B_l(c_l)
runtime cost: T_l(c_l)
effective capacity: C_l(c_l)
quality contribution: Q_l(c_l, c_{l-1}, c_{l+1})
```

Pure all-I2_S fixes:

```text
c_l = one_plane_ternary
```

Hybrid/variable BitNet asks:

```text
maximize quality under memory/latency budget
```

with choices like:

```text
one_plane_I2_S
two_plane_ternary
row-scale I2_S
Q2_K pocket
Q3 pocket
F16 residual pocket
rotation + one_plane_I2_S
```

The important correction after RT-123/TWLA:

```text
Q_l is not independent.
```

Adjacent-layer interactions matter, so the selector should be:

```text
score(c_l, c_{l+1})
```

not just:

```text
score(c_l)
```

This leads to a chain-structured dynamic program or beam search, not a naive additive
knapsack.

## Practical Design Rules

### Rule 1: Do not worship pure 1.58

Pure b1.58 is valuable as a floor:

```text
smallest, fastest, most memory-traffic-friendly baseline
```

But product quality may require:

```text
mostly b1.58 + small capacity budget.
```

### Rule 2: Extra capacity should buy output quality, not just lower MSE

A higher-precision pocket is worth it only if it improves:

```text
FACT score
degeneration rate
PPL/CE
or downstream task score
```

per added byte and per lost token/sec.

### Rule 3: Capacity and objective are separate axes

Content-KL can improve objective alignment. Multi-plane/hybrid can improve capacity.

They should be tested as:

```text
objective only
capacity only
objective + capacity
```

Otherwise we cannot tell whether factual gains come from more bits or a better loss.

### Rule 4: Model selection matters

If undertrained models quantize better, then benchmark rows should include:

```text
compact well-trained model
larger undertrained/redundant model
native BitNet if available
Q2_K baseline
```

This is necessary for fair "poor-resource model" claims.

## Experiment Implications

### CAP-001: Capacity sweep at fixed runtime family

Use the same model/eval and compare:

```text
all I2_S
two-plane selected late layers
Q2 selected late layers
F16 selected late layers
```

Measure:

```text
file size
target-linear bytes
token-gen t/s
CE/PPL
FACT rate
degeneration
```

Goal:

```text
find the minimum capacity budget that moves facts.
```

### CAP-002: Plane scaling

For selected layers:

```text
W ~= alpha_1*T_1 + alpha_2*T_2 + ... + alpha_k*T_k
k in {1, 2, 3}
```

Question:

```text
Does a second ternary plane recover more quality per byte than Q2_K/F16 pockets?
```

This directly tests PTQTP-style capacity expansion against simpler hybrid precision.

### CAP-003: Width/strip adapter

Instead of changing the original linear:

```text
y = W_i2s x + A B x
```

where `A B` is a small low-rank or ternary/FP residual strip.

Question:

```text
Can a small dynamic capacity path recover facts while keeping most traffic I2_S?
```

This is closer to "add channels/strips" than ordinary mixed precision.

### CAP-004: Undertrained vs well-trained checkpoint

Pick two same-family models:

```text
smaller/well-trained
larger/undertrained or redundant
```

Apply the same all-I2_S + adaptation recipe.

Question:

```text
Does the undertrained/redundant model have lower b1.58 conversion tax?
```

## What This Does Not Prove

Precision scaling laws do **not** prove:

```text
every b1.58 failure is capacity failure
```

They only give a strong prior. Our own experiments show at least three separable
failure modes:

```text
runtime/export failure      -> solved on x86 I2_S
objective/factual failure   -> content-KL helps
capacity failure            -> suspected, not yet isolated
```

So capacity experiments must control for objective and runtime.

## How This Changes Our Paper Split

| paper | update |
| --- | --- |
| Paper 1 Systems | precision scaling explains why speed/storage is the easy half |
| Paper 2 Conversion Limits | one-shot failure can be framed as effective-capacity collapse |
| Paper 3 Content-KL | objective recovery is not enough if capacity remains insufficient |
| Paper 4 Hybrid Capacity | becomes the natural follow-up, with a scaling-law justification |

## Bottom Line

The precision-scaling literature supports the user's instinct:

```text
If one b1.58 plane cannot carry the function,
do not keep polishing thresholds forever.
Restore capacity under a byte/token budget.
```

The next disciplined project question is:

```text
What is the cheapest extra capacity that turns fluent-but-wrong b1.58 output into
factual-enough output?
```

That question is more promising than pure all-I2_S dogma, and it is much closer to the
real goal: a poor-resource LLM that is small, fast, and actually useful.

## Source List

- Scaling Laws for Precision — https://arxiv.org/abs/2411.04330
- Low-Bit Quantization Favors Undertrained LLMs — https://arxiv.org/abs/2411.17691
- Scaling Laws for Mixed Quantization in Large Language Models — https://arxiv.org/abs/2410.06722
