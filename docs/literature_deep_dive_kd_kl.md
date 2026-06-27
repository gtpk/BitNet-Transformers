# Literature Deep Dive 05: KD / KL Objectives

Document position: [Index](./index.md) -> [Literature Positioning Map](./literature_positioning_map.md) -> fifth deep dive.

Papers:

- MiniLLM: On-Policy Distillation of Large Language Models — https://arxiv.org/abs/2306.08543
- DistiLLM: Towards Streamlined Distillation for Large Language Models — https://arxiv.org/abs/2402.03898
- Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models / AKL — https://arxiv.org/abs/2404.02657
- Status checked: 2026-06-27

## One-Line Read

MiniLLM, DistiLLM, and AKL are not direct BitNet papers. They are the map for where
our `content-KL` objective sits:

```text
MiniLLM: choose reverse KL + on-policy training.
DistiLLM: stabilize KL with skew divergence + adaptive off-policy data.
AKL: adaptively mix forward/reverse KL by head/tail gaps.
ours: mask the KL vocabulary support so the student copies content tokens but not stop/EOS behavior.
```

So content-KL should not be claimed as a new general KL divergence. It is a targeted
objective design for a very specific failure mode:

```text
raw KL copied the base model's early-stop/empty-answer behavior;
content-KL removed stop/special tokens from the KL anchor.
```

## Why This Matters For Our Project

FACT-003B and FACT-003C showed the key failure:

```text
raw base-KL lambda=1.0:
  fact -> 0.00
  empty outputs -> 25/27

content-KL lambda=0.2:
  fact -> 0.185
  ok outputs -> 27/27
  CE recovery -> 0.845
```

This means:

```text
The question was not merely "use KL or not?"
It was "which part of the teacher distribution should be copied?"
```

The distillation literature helps us name this properly.

## MiniLLM: Reverse KL + On-Policy Student Trajectories

MiniLLM argues that standard forward KL can make the student cover too much of the
teacher's low-probability support. It replaces forward KL with reverse KL and optimizes
on student-generated sequences.

Its core intuition:

```text
For generation, do not force the smaller/weaker student to cover every teacher mode.
Focus it on high-quality modes and train on its own trajectories to reduce exposure bias.
```

How it relates to us:

| MiniLLM idea | our connection |
| --- | --- |
| reverse KL avoids overestimating low-probability regions | b1.58 student has severe capacity limits; dense teacher distribution may be too broad |
| on-policy training uses student-generated text | RT-129 showed decoding behavior matters; factual failures may depend on free-run trajectories |
| language modeling loss is mixed in for general ability | our CE/content-KL also needs a fluency/factual balance |

What it does **not** answer:

```text
MiniLLM changes KL direction and trajectory distribution.
It does not specifically prevent EOS/stop-token copying.
```

Useful next idea:

```text
FACT-004 on-policy content-KL:
  sample short student continuations with repetition penalty,
  score content-token KL against base,
  keep CE on mixed data.
```

This would test whether our factual gap is partly exposure-bias/free-run mismatch.

## DistiLLM: Skew KL + Adaptive Off-Policy Efficiency

DistiLLM says LLM distillation lacks a stable standard objective and that fully
on-policy student-generated output can be expensive. It proposes skew KL/SRKL for
stable gradients and an adaptive off-policy replay scheme.

Its core intuition:

```text
Pure KL variants can have unstable or suboptimal gradients.
Student-generated data helps but is expensive/noisy.
Use skewed divergence and replay to get most of the benefit cheaply.
```

How it relates to us:

| DistiLLM idea | our connection |
| --- | --- |
| skew KL stabilizes gradients | FACT-003 lambda sweep is sensitive; raw KL was unstable/empty |
| adaptive off-policy SGO | useful if we move beyond fixed factual/instruction data |
| replay buffer | compatible with low-resource goal if generation is sparse |
| expensive teacher/student generation is a bottleneck | our conversion should not become full distillation-scale training |

What it does **not** answer:

```text
DistiLLM changes divergence smoothing and data schedule.
It does not define a content-only vocabulary mask.
```

Useful next idea:

```text
FACT-004 skew-content-KL:
  replace raw content-KL with skewed content-KL,
  keep EOS/special excluded,
  sweep skew alpha and lambda.
```

This directly imports DistiLLM's stability idea without losing the content-mask lesson.

## AKL: Adaptive Forward/Reverse KL By Head/Tail Gaps

AKL challenges the simplistic story that forward KL is always mean-seeking and reverse
KL is always mode-seeking in LLM KD. It argues that in practical finite training,
forward KL focuses on the head of the teacher distribution early, while reverse KL
focuses more on the tail; AKL adaptively mixes them according to head/tail gaps.

Its core intuition:

```text
The useful KL direction can change across tokens and training stages.
Head/tail mismatch should determine how much FKL or RKL to use.
```

How it relates to us:

| AKL idea | our connection |
| --- | --- |
| finite-budget training matters | our b1.58 adaptation is intentionally short and resource-limited |
| head/tail behavior matters | factual answers may live in head tokens; calibration/format may live elsewhere |
| adaptive weights beat fixed KL choice | FACT-003 lambda=0.1/0.2/1.0 showed strong non-monotonicity |

What it does **not** answer:

```text
AKL changes FKL/RKL mixture weights.
It still assumes the full vocabulary support unless modified.
```

Useful next idea:

```text
FACT-004 content-AKL:
  compute AKL only over content vocabulary,
  exclude EOS/BOS/PAD/template stop tokens,
  adapt FKL/RKL weights by content head/tail gap.
```

This is the most principled extension if content-KL plateaus.

## Where Content-KL Fits

Content-KL is best described as:

```text
a vocabulary-support intervention for KL anchoring
```

not:

```text
a new divergence family
```

The distinction:

| method family | primary axis | content-KL relation |
| --- | --- | --- |
| MiniLLM | KL direction + on-policy trajectories | could be combined with content mask |
| DistiLLM | skewed KL + off-policy replay efficiency | could stabilize content-KL |
| AKL | adaptive FKL/RKL weighting by head/tail gap | could choose direction inside content tokens |
| content-KL | vocabulary support mask | orthogonal to direction/skew/replay |

Our current formula is conceptually:

```text
V_content = V \ {EOS, BOS, PAD, special stop/template tokens}

teacher_content = normalize(p_base[V_content])
student_content = normalize(p_student[V_content])

L_content_KL = KL(teacher_content || student_content)
```

Then combine with answer-only CE:

```text
L = L_answer_CE + lambda * L_content_KL
```

This explains FACT-003B/003C:

```text
raw KL copied stop decisions -> empty outputs
content-KL copied content preferences while leaving stopping behavior to CE/decoding
```

## Safe Claims

Safe:

```text
In our b1.58 conversion setting, masking stop/special tokens out of the KL anchor
prevented empty-answer collapse and improved factual score over CE-only and raw KL.
```

Safe:

```text
Content-KL is orthogonal to KL direction/skew/on-policy choices from KD literature.
```

Safe:

```text
MiniLLM/DistiLLM/AKL suggest stronger next variants:
content-RKL, skew-content-KL, content-AKL, and on-policy content-KL.
```

Unsafe:

```text
Content-KL is a generally novel distillation method.
```

Unsafe:

```text
Content-KL solves factual recovery.
```

It has found the first useful lever, not the end of the problem.

## Next Experiment Branches

If `lambda=0.5` does not cross a strong factual threshold, use this order:

### FACT-004A: content-KL direction ablation

```text
same data/checkpoint
same EOS/special mask
compare:
  content-FKL
  content-RKL
  symmetric content-KL
```

Question:

```text
Is our current success about vocabulary masking, KL direction, or both?
```

### FACT-004B: skew-content-KL

```text
content mask fixed
skew alpha in {0.05, 0.1, 0.2}
lambda near best FACT-003C
```

Question:

```text
Can DistiLLM-style skewing stabilize the objective and widen the lambda sweet spot?
```

### FACT-004C: content-AKL

```text
content mask fixed
split teacher content distribution into head/tail
adapt FKL/RKL weight by current student gap
```

Question:

```text
Can AKL recover more facts without over-anchoring format/stop behavior?
```

### FACT-004D: sparse on-policy content-KL

```text
occasionally sample student continuations
compute teacher content-KL on those prefixes
reuse a small replay buffer
```

Question:

```text
Is the remaining factual gap partly a free-run exposure-bias problem?
```

## Bottom Line

The distillation literature makes our position sharper:

```text
content-KL is not "new KL."
It is "KL support selection" for a low-bit conversion failure mode.
```

That is still useful. It gives us a composable axis:

```text
support mask: full vocab vs content vocab
direction: FKL vs RKL vs symmetric
stability: raw vs skew
schedule: offline vs sparse on-policy/off-policy
```

The next serious factual-recovery experiment should not just sweep lambda again. It
should combine content masking with one of MiniLLM/DistiLLM/AKL's stronger axes.

## Source List

- MiniLLM — https://arxiv.org/abs/2306.08543
- DistiLLM — https://arxiv.org/abs/2402.03898
- AKL / Rethinking KL Divergence in KD for LLMs — https://arxiv.org/abs/2404.02657
