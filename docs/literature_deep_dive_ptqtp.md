# Literature Deep Dive 01: PTQTP

Document position: [Index](./index.md) -> [Literature Positioning Map](./literature_positioning_map.md) -> first deep dive.

Paper:

- PTQTP: Post-Training Quantization to Trit-Planes for Large Language Models
- arXiv: https://arxiv.org/abs/2509.16989
- Status checked: 2026-06-27

## One-Line Read

PTQTP is the closest external competitor to our "increase strips / planes / capacity"
idea. It says pure one-plane b1.58 is too small for PTQ, so approximate each weight
matrix with **two ternary trit-planes** and row/group scales:

```text
W ~= diag(alpha_1) * T_1 + diag(alpha_2) * T_2
T_k in {-1, 0, +1}
```

This is almost exactly the direction we were circling when we said "1.58 alone cannot
hold the function; maybe strip/width/layer capacity must grow."

## What They Claim

From the abstract and method:

| claim | meaning |
| --- | --- |
| post-training ternary, no retraining | convert pretrained LLMs directly |
| dual trit-planes | use `2 x 1.58-bit` instead of one ternary plane |
| multiplication-free additive inference | operations are ternary add/sub plus scales, not dense FP multiply |
| model-agnostic | no architecture modification |
| 0.6B-70B experiments | much larger scale than our current 160M/1.1B runs |
| faster than FP16 | paper reports up to 4.63x end-to-end inference speedup |
| strong reasoning retention | paper claims 82.4% math reasoning retention vs 0% for competitors |

Their high-level thesis:

```text
single-plane ternary lacks expressiveness;
dual trit-plane restores enough expressiveness while keeping uniform ternary operations.
```

## Method Understanding

### Representation

For a weight matrix `W in R^{n x d}`, PTQTP approximates:

```text
W_hat = sum_{k=1}^{2} diag(alpha^(k)) T^(k)
```

where:

```text
T^(k) in {-1, 0, +1}^{n x d}
alpha^(k) is row-wise or group-row-wise scale
```

Compared with our all-I2_S representation:

```text
ours:   W_hat = gamma * T
PTQTP:  W_hat = alpha_1*T_1 + alpha_2*T_2
```

So PTQTP is not "better one-plane ternary"; it is a **capacity expansion**.

### Optimization

The paper alternates between:

1. fixing ternary planes and solving scales with ridge regression;
2. fixing scales and choosing ternary codes that reduce squared reconstruction error;
3. doing this group-wise, typically group size 128;
4. adapting regularization based on local conditioning.

This is important because it turns the plane fitting into a stable local least-squares
problem rather than a naive residual sign pass.

### Relation To Our RT-124/125

Our experiments saw:

```text
row/block scale helps but remains insufficient;
GPTQ-style assignment helps but closes only a small gap;
signed-epsilon 2-bit does not help;
pure one-plane ternary is too small.
```

PTQTP's answer is different:

```text
do not change zero into epsilon;
do not only optimize one plane;
add another ternary plane.
```

That matches our evidence better than signed-epsilon did.

## Where PTQTP Is Ahead Of Us

| axis | PTQTP | us |
| --- | --- | --- |
| conversion quality | claims strong low-bit PTQ quality | all-I2_S factual still weak |
| model scale | 0.6B-70B | 160M/1.1B |
| conversion type | no retraining/fine-tuning | adaptation/objective needed |
| representational capacity | dual trit-plane | currently one I2_S plane |
| benchmark breadth | language/reasoning/math/coding claims | small factual panel + PPL/CE |

Hard truth:

```text
If PTQTP reproduces, it beats our current all-I2_S conversion engine.
```

## Where We May Still Differ

| axis | our possible differentiation |
| --- | --- |
| real runtime path | we deeply validated bitnet.cpp I2_S; PTQTP runtime/code path still needs direct inspection |
| product scorecard | we track storage, tg speed, post-training cost, factual score, degeneration, runtime parity |
| factual/objective recovery | PTQTP appears primarily reconstruction/PTQ-focused; our content-KL work targets facts |
| adaptive allocation | PTQTP seems uniform two-plane; we can test selective planes/pockets per layer |
| hybrid with existing runtimes | we can decide between I2_S, Q2/Q3, F16 pocket, residual under one budget |

Possible reframing:

```text
PTQTP is a strong conversion engine.
Our project can become the runtime/factual/budget-aware compiler around such engines.
```

## Questions / Doubts To Verify

These are not accusations; they are the exact things we must check before treating
PTQTP as solved product technology.

| question | why it matters | how to test |
| --- | --- | --- |
| Is code actually available and runnable? | arXiv says code will be available; search did not confirm stable repo | find repo, clone, run tiny model |
| What is the exact storage after two planes + scales? | `2 x 1.58-bit` is not I2_S 2-bit; effective bytes may approach Q2_K | compute whole-file and target-linear bytes |
| Does it have a bitnet.cpp-compatible runtime? | multiplication-free on paper may not equal existing local runtime | inspect kernels or export path |
| Does it preserve factual behavior? | paper emphasizes benchmarks/reasoning; our product gap is factual QA | run our 27-prompt factual panel |
| Is two-plane needed everywhere? | uniform two-plane may waste bytes in easy layers | run per-layer/late-layer plane ablation |
| Can content-KL improve it further? | PTQTP may solve capacity but not objective | adapt PTQTP weights with content-KL |
| Does it beat Q2_K under same bytes/speed? | Q2_K is practical baseline | compare size, tg, PPL, fact score |

## Insight For Our Next Experiments

PTQTP makes one thing clear:

```text
capacity expansion is not optional if one-plane b1.58 plateaus.
```

But it does not force us to copy uniform two-plane everywhere. A better product question:

```text
How many extra ternary planes are needed, and where?
```

This turns Paper 4 into:

```text
adaptive trit-plane allocation under a memory-traffic budget
```

instead of:

```text
try random hybrid pockets
```

## Proposed Minimal Reproduction

Before touching 70B-scale claims, implement or reproduce a tiny PTQTP-like baseline:

```text
Input: one trained FP or adapted model weight matrix W
For each row/group:
  initialize T1, T2
  solve alpha1, alpha2 by ridge/least squares
  update each pair (T1[j], T2[j]) over {-1,0,1}^2
  iterate until stable
Output: alpha1*T1 + alpha2*T2 materialized dense reference
```

First local tests:

| test | pass criterion |
| --- | --- |
| matrix reconstruction | dual-plane rel-L1/rel-MSE beats one-plane |
| 160M one-shot PPL | beats RT-124 group128 and RT-125 GPTQ |
| 1.1B factual panel | improves over all-I2_S content-KL baseline or at least PTQ |
| storage estimate | compare against I2_S, Q2_K, F16 |

Then decide whether to:

```text
use PTQTP as a baseline only
or build adaptive PTQTP+content-KL as our next method.
```

## How This Changes Our Roadmap

| old idea | after PTQTP |
| --- | --- |
| signed-epsilon 2-bit | demoted; PTQTP suggests extra plane is better than replacing zero |
| naive mixed-bit DP | demoted; interactions matter, and capacity is better represented as plane count |
| HYBRID-001 | should include trit-plane arms |
| content-KL | keep as objective layer, not as the whole conversion engine |
| Paper 4 | should become adaptive plane/capacity allocation |

## Bottom Line

PTQTP is not just related work. It is a direct competitor and a strong hint.

The right response is not to ignore it or fight it ideologically. The right response is:

```text
reproduce it,
measure it with our product scorecard,
then add what PTQTP does not address:
  factual objective,
  runtime export,
  adaptive allocation,
  low-resource usability.
```

