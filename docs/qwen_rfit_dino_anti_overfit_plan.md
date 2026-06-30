# Qwen RFIT-D: DINO As Anti-Overfit Regularizer

Document position: [Index](./index.md) -> Qwen ladder -> RFIT-D plan.

Status: proposed after Qwen-1.5B rung showed undertraining at 800 and
over-training by 1600.

2026-07-01 update:

```text
RFIT-D is now a special case of AAMC:
docs/adaptive_anchor_manifold_controller_plan.md
```

That means DINO is not turned on because it is fashionable or because it helped
one small run. It is turned on only when telemetry says the model needs a
manifold/stability anchor. If the failure is ordinary train-stream overfit with
clean generation, the first knob is `lambda`, not DINO.

## Why This Exists

The earlier DINO interpretation was:

```text
DINO might directly improve factual recovery.
```

That version failed as a main factual lever on TinyLlama-1.1B. It improved
distributional signals and generation stability, but did not beat the content-KL
factual baseline.

The Qwen-1.5B result suggests a different use:

```text
DINO might prevent answer-CE over-training.
```

Qwen-1.5B with the minimal content-KL recipe showed:

| point | FACT | first_token_hit | eval CE | recovered | train CE | read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 800 | 0.111 | 0.000 | 4.85 | 0.814 | 1.58 | undertrained / readout weak |
| 1600 | 0.222 | 0.111 | 5.68 | 0.749 | 0.37 | FACT up, eval CE worse, train overfit |

The model got better at the FACT panel from 800 to 1600, but the held-out CE got
worse while train CE collapsed. That is exactly the regime where a weak
consistency regularizer may help: not to teach facts directly, but to keep the
student from memorizing the answer-CE stream and drifting away from the base
distribution.

The later RFIT interpretation refines this:

```text
if generation is clean but eval CE/FACT worsens -> increase lambda first.
if generation collapses or becomes unstable -> add weak late DINO.
```

So RFIT-D remains valid, but only behind the AAMC gate.

## Hypothesis

```text
Anti-Overfit DINO Hypothesis:

For Qwen-1.5B I2_S adaptation, weak late logit-DINO can reduce train-stream
overfitting and preserve teacher-like output distribution, allowing FACT/readout
to improve without the eval-CE regression seen at 1600.
```

This is not a named rule yet. It is a testable hypothesis.

## What We Must Not Repeat

Do not repeat the old DINO mistake:

```text
content-KL + large DINO from step 0
```

The old failure mode was:

```text
generation becomes fluent,
gold rank may improve,
but answers drift into base-LM rambling and FACT stays low.
```

RFIT-D therefore uses DINO only as a weak, late regularizer.

## Method

Base recipe:

```text
model: Qwen/Qwen2.5-1.5B-Instruct
precision: bf16 + adamw8bit + grad checkpointing
target: I2_S linears only
train_source: mixed
answer_loss_only: true
content-KL: lambda = 0.2 or 0.4
DINO hidden alignment: OFF
```

RFIT-D changes:

```text
dino_logit_weight: small, start with 0.05
dino_hidden_weight: 0
dino warmup: start after the model exits transient
preferred start: step 300 or step 400
```

Conceptual objective:

```text
L =
  L_answer_CE
  + lambda * KL_content(p_teacher || p_student)
  + alpha(t) * KL_content(p_student_clean || p_student_view)
```

where:

```text
alpha(t) = 0                      for t < warmup_step
alpha(t) = dino_logit_weight       after warmup_step
```

The DINO term should be content-only and logit-only. No hidden alignment.

## Primary Test Arms

Run only after RFIT-A peak is known.

| arm | lr | lambda | DINO | steps | score points | purpose |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| RFIT-A | 1e-4 | 0.2 | 0 | 800 | 400/600/800 | lower-LR baseline |
| RFIT-B | 1e-4 | 0.4 | 0 | 800 | 400/600/800 | stronger teacher anchor |
| RFIT-D1 | 1e-4 | 0.2 | 0.05 after 300 | 800 | 400/600/800 | anti-overfit with old lambda |
| RFIT-D2 | 1e-4 | 0.4 | 0.05 after 300 | 800 | 400/600/800 | anti-overfit with stronger anchor |

Do not launch D1/D2 before RFIT-A establishes the baseline trajectory.

## Required Metrics

Every checkpoint score must record:

```text
FACT exact
first_token_hit
gold_rank_mean
eval CE / recovered_fraction
train CE
degen_gap / salad / loop / empty
sample generations
```

The decisive pattern is not a single number. We need:

```text
FACT up,
first_token_hit up,
eval CE not worse,
train CE not collapsed,
samples less rambling.
```

## Pass / Fail

### Strong Pass

```text
FACT > 0.333
first_token_hit > 0.222
eval CE <= RFIT-A at same score point
samples emit short answers more often
```

Interpretation:

```text
DINO has a new valid role: anti-overfit consistency regularizer.
```

### Partial Pass

```text
FACT 0.25-0.333
but eval CE is better than the no-DINO run
and samples ramble less
```

Interpretation:

```text
DINO may help recipe fitting, but it does not restore scale-up superiority yet.
Tune weight/warmup or combine with lambda=0.4.
```

### Fail

```text
FACT <= no-DINO baseline
or samples become more rambling
or degen/salad returns
```

Interpretation:

```text
DINO remains a dynamics diagnostic, not a Qwen factual lever.
Do not use DINO in the main Qwen ladder.
```

## Why This Is Different From Old DINO

Old DINO asked:

```text
Can DINO raise factual quality directly?
```

RFIT-D asks:

```text
Can weak late DINO stop Qwen-1.5B from overfitting the answer-CE stream while
content-KL recovers the model?
```

That is a different mechanism and a different failure target.

## Recommended Next Action

1. Finish RFIT-A/B fixed-arm diagnosis and inspect `FACT@400/600/800`,
   `train_ce`, `eval_ce`, `gold_rank`, and generation tags.
2. If the pattern is clean-generation overfit, raise/shape `lambda` first.
3. If the pattern includes degen/salad/loop/empty or unstable hidden variance,
   run RFIT-D as weak late DINO (`alpha=0.05`, hidden OFF).
4. If neither fixed lambda nor AAMC-style control beats Qwen-0.5B `0.333`,
   stop RFIT on this recipe and revisit data/model choice rather than stacking
   more fixed auxiliary losses.
