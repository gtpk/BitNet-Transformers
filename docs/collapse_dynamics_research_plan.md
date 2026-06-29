# Generation Collapse Dynamics Research Plan

Status: active research reframing.

Document position: [Index](./index.md) -> [Current Theory](./current_theory_hypothesis_plan.md) -> DINO / FACT objective branch.

Related literature synthesis:

```text
docs/literature_deep_dive_collapse_dynamics.md
```

## One-Line Reframe

The next question is not:

```text
Is this objective good?
```

The next question is:

```text
When, why, and under which training dynamics does generation collapse begin?
```

This project should treat collapse as a dynamic phenomenon, not a final scalar
score.

## Why The Reframe Is Needed

The project already found the dangerous pattern:

```text
loss can improve while generation collapses.
```

Examples:

| track | loss / CE behavior | generation / factual behavior | lesson |
| --- | --- | --- | --- |
| FACT-003B raw KL | KL/CE can move | empty output collapse | raw teacher stop mass can dominate |
| FACT-004A lm_head unfreeze | recovered CE stays decent | facts collapse | extra freedom can forget faster |
| FACT-003D hard replay | train factual CE improves | held-out/eval facts fall | small hard replay creates shortcuts |
| FACT-003H v1/v2 | training can appear active | loop/salad collapse | final CE alone is insufficient |
| DINO-DIAG-001 | exact-match small | gold-token logprob/rank improves | final exact-match hides pre-decode signal |

Therefore:

```text
final PPL / final fact_rate is too late.
```

We need a time series.

## Research Position

Most adjacent methods focus less on "why it collapses" and more on "how to avoid
collapse":

```text
RLHF / DPO / diffusion RL / DINO-like self-supervision
```

Common tools:

```text
regularization
curriculum
adaptive weighting
optimizer changes
training schedules
teacher/student dynamics
```

The shared lesson is:

```text
objective terms matter,
but training dynamics decide whether they help or collapse.
```

## Collapse Taxonomy

Do not write only:

```text
salad
```

Write:

| collapse type | observable signature | likely cause | next action |
| --- | --- | --- | --- |
| empty collapse | empty or too-short outputs; EOS/stop mass high | raw KL copied stop behavior | content-KL, EOS mask, answer-length guard |
| loop collapse | repeated n-grams, same phrase loops | top-1 confidence spike, entropy drop | repetition-aware diagnostics, temperature/entropy regularizer |
| salad collapse | malformed tokens/words, incoherent text | representation instability or under-recovery | check gradient/update norms, hidden variance |
| fluent confabulation | fluent but wrong facts | CE learned style but not knowledge retention | content/logit distillation, factual eval, entity-targeted objective |
| train-only memorization | train factual score high, held-out flat/down | small replay shortcut | larger representative stream, no hard replay |
| rank-only recovery | gold rank/logprob improves but exact match flat | distribution moved, decoding/readout not enough | larger model gate, answer-token weighting |
| hidden overconstraint | hidden loss down but FACT flat/down | FP hidden geometry impossible in I2_S | remove/reduce hidden alignment |

## Step-Level Telemetry

Every nontrivial FACT/DINO run should emit a `metrics.jsonl` with one JSON object
per log step.

Minimum fields:

```json
{
  "step": 200,
  "total_loss": 0.0,
  "train_ce": 0.0,
  "content_kl": 0.0,
  "dino_loss": 0.0,
  "hidden_loss": 0.0,
  "grad_norm": 0.0,
  "update_norm": 0.0,
  "update_to_param": 0.0,
  "hidden_var_mid": 0.0,
  "hidden_var_last": 0.0,
  "logit_entropy": 0.0,
  "top1_prob": 0.0,
  "gold_logprob_mean": 0.0,
  "gold_rank_mean": 0.0,
  "gold_rank_median": 0.0,
  "salad_rate": 0.0,
  "empty_rate": 0.0,
  "loop_rate": 0.0,
  "ok_rate": 0.0
}
```

Recommended extra fields:

```json
{
  "category_simple_fact_gold_logprob": 0.0,
  "category_entity_attr_gold_logprob": 0.0,
  "category_reasoning_gold_logprob": 0.0,
  "teacher_student_topk_overlap": 0.0,
  "answer_token_entropy": 0.0,
  "function_token_entropy": 0.0,
  "entity_token_entropy": 0.0
}
```

## Collapse Onset Detection

Define a collapse onset step, not only a final label.

For generation probes at checkpoint `t`:

```text
collapse_score(t) =
  w_empty * empty_rate(t)
  + w_loop * loop_rate(t)
  + w_salad * salad_rate(t)
  - w_ok * ok_rate(t)
```

An onset occurs when:

```text
collapse_score(t) - collapse_score(t - k) > delta
```

and at least one of:

```text
logit_entropy drops sharply
top1_prob rises sharply
gold_rank worsens
loop_rate rises
```

For this project, start with a coarse rule:

```text
collapse_onset =
  first checkpoint where
    empty_rate + loop_rate + salad_rate >= 0.30
  AND previous checkpoint was < 0.15
```

Use exact thresholds as diagnostics, not paper claims.

## Step Windows To Inspect

The project repeatedly saw mid-training surprises. Therefore inspect:

```text
step 0
step 50
step 100
step 150
step 200
step 250
then every 100 steps
final
```

If a run is known to collapse around `180-220`, add dense probes:

```text
step 160, 180, 200, 220, 240
```

## Dynamics Questions

Each failed run should answer:

1. Did loss improve before collapse?
2. Did content KL improve before collapse?
3. Did gradient norm spike?
4. Did update norm spike?
5. Did hidden variance shrink or explode?
6. Did logit entropy collapse?
7. Did top-1 probability spike?
8. Did gold rank improve first, then degrade?
9. Did collapse affect all categories or only entity/instruction prompts?
10. Did f16 and i2_s collapse identically?

## Curriculum Adaptation

Current objectives often start all terms at step 0:

```text
CE + content-KL + DINO / replay / blend
```

New curriculum hypothesis:

```text
Stage 1: stabilize language / I2_S manifold
  CE + content-KL

Stage 2: add DINO-logit
  CE + content-KL + small DINO-logit

Stage 3: increase DINO weight
  CE + content-KL + scheduled DINO-logit
```

Do not add hidden alignment in the default curriculum. DINO-DIAG showed hidden
alignment can erase the logit benefit.

Example schedule:

```text
0-25% steps:     lambda_dino = 0
25-60% steps:    lambda_dino linear warmup to target
60-100% steps:   lambda_dino fixed or cosine decay
```

Alternative schedule if entropy collapse appears:

```text
increase DINO only while logit_entropy is above floor
freeze/reduce DINO when top1_prob spikes
```

## Diagnostic Matrix

| observation | interpretation | next action |
| --- | --- | --- |
| KL/logprob improves, exact-match flat | objective moves distribution but not enough for decode | larger model gate, answer-token weighting |
| simple facts improve, entity_attr flat | coverage problem | entity-rich unlabeled prompts, entity-token weighted KL |
| loss improves, generation collapses | dynamics problem | curriculum, entropy/top1 guard, optimizer schedule |
| hidden loss improves, fact worsens | hidden overconstraint | remove hidden loss |
| train facts improve, heldout flat | memorization | larger representative stream, no small hard replay |
| i2_s differs from f16 | runtime/export issue | debug I2_S parity first |
| 160M works, 1.1B fails | scale/optimizer schedule | 1.1B-specific LR/warmup/optimizer sweep |

## Immediate Experimental Change

Before the next 1.1B DINO/FACT run:

```text
instrument first,
then train.
```

Required:

```text
metrics.jsonl
checkpointed generation probes
gold-rank/logprob diagnostics
category-level FACT table
f16 vs i2_s parity at final
```

Recommended first implementation:

```text
scripts/collapse_dynamics_probe.py
```

Inputs:

```text
--checkpoints ckpt_step_0 ckpt_step_100 ...
--fact-panel data/fact_panel.jsonl
--popqa-tight data/popqa_heldout_tight.jsonl
--teacher-model optional
--json-out reports/collapse_dynamics_<run>.json
```

Output:

```text
time-series table
collapse onset step
category-level gold-rank deltas
generation tag trajectories
```

## Claim Discipline

Allowed claim:

```text
Generation collapse during I2_S adaptation is a dynamic training phenomenon that
can be localized in step time and decomposed into entropy, confidence, rank, and
generation-tag trajectories.
```

Not allowed yet:

```text
We have solved generation collapse.
```

The immediate goal is diagnosis:

```text
find the moment the model begins to fail,
then design the schedule/objective around that moment.
```
