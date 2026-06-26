# Paper 2 Skeleton: Why Existing LLMs Resist b1.58 Conversion

Working title:

```text
Why Post-Training b1.58 Conversion Is Not Ordinary Quantization
```

Status: nearly ready. This is the negative-results / problem-definition paper.

Central table: [Paper Evidence Matrix](./paper_evidence_matrix.md).

## Draft Abstract

Native b1.58 LLMs exist, but that does not imply that a trained floating-point LLM
can be converted by ordinary post-training quantization. We study same-shape b1.58
conversion of LLaMA-family checkpoints and find that one-shot ternary rounding
collapses quality. We then apply a sequence of standard quantization levers:
scale granularity, scale/threshold objectives, activation-diagonal transforms,
GPTQ-style output-aware assignment, and a 2-bit signed-epsilon codebook. These
levers produce local improvements, but none closes the gap to either the FP model
or standard Q2_K. The strongest one-shot effects remain far above usable PPL, while
short adaptation dominates all quantizer tweaks. The result is a problem-definition
claim: existing-model b1.58 conversion is a function reconstruction and adaptation
problem, not a normal rounding problem.

## Result Table For The Paper

| method / lever | measured result on Llama-160M | interpretation |
| --- | --- | --- |
| FP f16 | PPL 43.2 in RT-121 baseline panel | practical FP reference for same-tool baseline |
| Q2_K one-shot | PPL 97.9, 134MB | stronger quality than ours, larger/slower than I2_S |
| one-shot ternary RTN | PPL 115,808-135,309 depending eval path | pure PTQ collapse |
| ours b1.58 + CE | PPL 114.1, 121.5MB | usable relative to PTQ, but does not beat Q2_K on PPL |
| row scale | +1.84 nats vs per-tensor | real but still PPL 18,422 |
| group128 scale | +2.36 nats vs per-tensor | strongest one-shot scale lever, still PPL 10,935 |
| AWQ/SmoothQuant diagonal | +0.14 nats | too small |
| GPTQ/Hessian assignment | +0.51 nats; 6% of gap | assignment matters, but not enough |
| signed-epsilon 2-bit | worse than ternary for all eps tested | zero is not the blocker |

## Blank Cells Before Submission

| blank | why it matters | next action |
| --- | --- | --- |
| 1.1B Q2_K vs ours baseline | RT-121 is 160M-only | repeat baseline table at 1.1B |
| seed variance | distinguishes stable limit from lucky run | 2-3 seeds on 160M adapted recipe |
| no-scale QAT contrast | proves gamma is not incidental | run B5 in [G5 Baseline Plan](./g5_baseline_plan.md) |

## Thesis

Existing FP LLMs cannot be made good b1.58 models by a same-shape one-shot
quantizer. Native BitNet works because the model is trained inside a ternary-friendly
function class. Post-training conversion must reconstruct a function, not merely
round weights.

## Do Claim

```text
one-shot b1.58 PTQ collapses
standard PTQ levers do not rescue it
Q2_K remains a stronger quality baseline
layer effects are non-additive
the quantizer is not the main lever
```

## Do Not Claim

```text
b1.58 is impossible
native BitNet is weak
hybrid capacity will work before HYBRID-001 data exists
```

## Core Results

| result | evidence |
| --- | --- |
| one-shot collapse | RT-116/120/121 |
| Q2_K beats ours on PPL | RT-121: ours 114 vs Q2_K 98 on 160M |
| additive DP weak | RT-123: many single restores worsen CE |
| scale granularity partial only | RT-124A |
| absmean already strong | RT-124B |
| activation diagonal weak | RT-124C |
| GPTQ/Hessian small gain | RT-125: ~6% of gap |
| signed-epsilon no gain | RT-127 |
| runtime exonerated | RT-111/112, FACT-001 |

## Key Hints We Saw

```text
Layer interactions are non-additive.
All-ternary can be self-consistent but weak.
Restoring one piece can mismatch downstream ternary blocks.
PPL recovery does not equal factual recovery.
```

## Native BitNet Context

Public BitNet materials show native BitNet is not merely a post-hoc LLaMA
quantization:

```text
BitLinear
SubLN
relu2
no bias
native training
SFT/DPO
BitNet-specific runtime
```

So the fair conclusion is:

```text
native b1.58 success != same-shape FP checkpoint conversion success
```

## Figures

1. Q2_K vs ours vs PTQ baseline table.
2. RT-124..127 quantizer lever bar chart.
3. RT-123 non-additivity histogram.
4. Native BitNet vs post-training conversion problem diagram.

## Missing Before Final

```text
Maybe add one larger-model Q2_K vs ours point.
Maybe add seed variance on the strongest conversion recipe.
```
