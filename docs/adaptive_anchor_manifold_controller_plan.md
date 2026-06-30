# Adaptive Anchor-Manifold Controller (AAMC) Plan

Document position: [Index](./index.md) -> Qwen / DINO / collapse-dynamics branch.

Status: operating hypothesis and experiment controller, not a discovered law.

## Why This Exists

Fixed objective weights have repeatedly failed in different ways:

| run family | symptom | lesson |
| --- | --- | --- |
| FACT-003C content-KL | lambda `0.2` works, `0.1` too weak, `0.5` washes out facts | teacher anchor has a sweet spot |
| FACT-003D hard replay | small factual CE is memorized and hurts held-out facts | direct fact pressure can overfit |
| DINO/TL1B-1600 | generation recovers with longer budget, but factual exact stays weak | stability is not factuality |
| ANS-001 | answer-token weighting traps 1.1B at high CE/gold-rank | stronger readout pressure can block consolidation |
| Qwen-1.5B RFIT | 800 undertrained, 1600 train-overfit/eval-worse | schedule and anchor strength must adapt to dynamics |

So the next question is no longer:

```text
Which single fixed objective is best?
```

It is:

```text
Which control signal should change which objective weight, and when?
```

AAMC is the proposed controller:

```text
lambda_t  -> teacher/content anchor strength
alpha_t   -> manifold/stability consistency strength, usually DINO-logit
```

It stays inside the I2_S-rooted track. It does not change runtime format.

## Core Objective

Use the same I2_S target-linears and materialized `Wq = gamma*T` path. Only the
adaptation objective changes:

```text
L_t =
  L_answer_CE
  + lambda_t * KL_content(p_teacher || p_student)
  + alpha_t  * KL_content(p_student_clean || p_student_view)
```

where:

```text
KL_content excludes EOS/special/control tokens.
alpha_t uses logit-only DINO-style consistency.
hidden alignment stays OFF unless a later diagnostic specifically requires it.
```

Interpretation:

| knob | role | primary failure it treats |
| --- | --- | --- |
| `lambda_t` | teacher/content anchor | overfit, factual drift, eval CE regression |
| `alpha_t` | manifold/stability anchor | salad, loop, empty, unstable generation |

This is deliberately different from "turn on every auxiliary loss." Each knob
has a different diagnosis.

## Signals To Log

Every run that claims to use AAMC must log these over time, not just at the end:

| signal | meaning |
| --- | --- |
| `train_ce` | whether the adaptation stream is being memorized |
| `eval_ce` / recovered fraction | whether held-out language modeling is improving |
| `FACT` / first-token-hit | whether factual answers are emitted |
| `gold_rank_mean` | whether the right answer is becoming reachable |
| `degen_gap` | student degeneration relative to its FP teacher |
| `salad_rate`, `loop_rate`, `empty_rate` | generation-collapse subtype |
| entropy / top-1 probability | overconfidence vs uncertainty |
| hidden activation variance | transient instability / consolidation |
| gradient norm / update norm | optimizer shock or flatness |

The important ratios are:

```text
overfit_score =
  high(train_ce_drop) + eval_ce_worse + FACT_flat_or_down
  + entropy_down + top1_overconfident

collapse_score =
  degen_gap_up + salad/loop/empty_up
  + hidden_var_spike + entropy_instability
```

Do not overinterpret one scalar. The controller uses a pattern.

## Controller Policy V0

This is the first practical version. It is intentionally simple enough to test.

```text
initialize:
  lambda = 0.2
  alpha  = 0.0

every score interval, e.g. 100 or 200 steps:
  if collapse_score is high:
      alpha = min(alpha + 0.05, 0.10)

  elif overfit_score is high and collapse_score is low:
      lambda = min(lambda + 0.1, 0.5)

  elif train_ce stalls high and gold_rank does not improve:
      lambda = max(lambda - 0.1, 0.2)
      alpha  = 0.0

  else:
      keep current weights
```

Clamp ranges:

```text
0.2 <= lambda <= 0.5
0.0 <= alpha  <= 0.10
```

Why these bounds:

- `lambda=0.1` was too weak in FACT-003C.
- `lambda=0.5` could wash out facts in TinyLlama, but Qwen-1.5B RFIT suggests
  stronger anchor may be useful against overfit.
- full DINO from the start was too intrusive in TinyLlama; if used, it should be
  weak and late.

## State Table

| observed state | diagnosis | action |
| --- | --- | --- |
| train CE keeps falling, eval CE worsens, FACT flat/down, generation ok | overfit / factual drift | increase `lambda` |
| gold rank improves, but salad/loop/empty or high `degen_gap` persists | manifold instability | increase `alpha` weakly |
| train CE high and flat, gold rank flat, degen high | objective too stiff / trapped | reduce `lambda`, keep `alpha=0`, consider LR |
| FACT improves but eval CE worsens | useful but overfit-prone | score earlier checkpoints; increase `lambda` only if generation ok |
| FACT flat, gold rank improves, generation rambling | readout/format gap | do not add DINO blindly; consider format/data intervention |
| train panel rises but held-out/eval drops | memorization shortcut | stop small hard replay; use representative stream |

## How This Changes The Next Experiments

### Qwen RFIT

RFIT-A/B already test fixed lambda values. AAMC changes the next step:

```text
Do not run DINO because it sounds promising.
Run DINO only if collapse_score rises or generation becomes unstable.
```

For the currently observed Qwen-1.5B pattern:

```text
train CE overfits,
eval CE worsens,
generation stays ok,
FACT is below Qwen-0.5B.
```

So the first controller action is:

```text
increase lambda / find a stronger teacher anchor,
not add DINO as the primary knob.
```

DINO becomes RFIT-D only if a run shows:

```text
FACT or gold_rank improves while degen/salad/loop appears.
```

### TinyLlama / Collapse Dynamics

TinyLlama showed long transient collapse. For that regime:

```text
collapse_score was the primary signal,
so weak late DINO or longer schedule is plausible.
```

AAMC prevents the old mistake:

```text
do not call an endpoint failure "method failure" before watching dynamics.
```

### PC / 160M Smokes

160M can test code paths and sign of a knob, but it cannot decide Qwen-1.5B or
TinyLlama-1.1B behavior. Use it for:

```text
finite-loss smoke,
metric logging,
controller implementation,
cheap sign checks.
```

Do not use it as the final gate for:

```text
lambda optimum,
factual ceiling,
generation-collapse onset.
```

### PC / Qwen-0.5B Overfit Wind-Tunnel

The better PC test is Qwen-0.5B, not only 160M:

```text
Qwen-0.5B is small enough for the RTX 3080,
but close enough to the Qwen ladder to exercise the same tokenizer/model family.
```

Purpose:

```text
deliberately create a mild overfit regime,
then test whether AAMC raises lambda before eval/FACT deteriorates.
```

This is a controller test, not the final quality claim.

Why it is useful:

```text
small/medium models enter train/eval mismatch quickly,
so PC can tell whether the controller reacts,
while Colab/L4 is reserved for the expensive 1.5B/7B gates.
```

Recommended PC arms:

| arm | lambda policy | alpha policy | purpose |
| --- | --- | --- | --- |
| fixed-0.2 | `lambda=0.2` | `alpha=0` | under-anchored baseline |
| fixed-0.4 | `lambda=0.4` | `alpha=0` | stronger anchor baseline |
| dynamic-lambda | start `0.2`, raise to `0.4` on overfit signal | `alpha=0` | AAMC first test |
| dynamic-lambda+dino | same as above | `alpha=0.05` only if collapse appears | conditional DINO sanity |

Overfit signal:

```text
train_ce keeps falling,
eval_ce worsens or stalls,
FACT / first_token_hit does not improve,
generation remains ok.
```

Collapse signal:

```text
degen_gap rises,
salad/loop/empty rises,
hidden variance spikes,
entropy/top1 becomes unstable.
```

Pass:

```text
dynamic-lambda >= best fixed arm on FACT,
eval_ce no worse than fixed-0.2,
train_ce not collapsed,
generation tags ok.
```

Fail:

```text
dynamic-lambda only follows the best fixed arm,
or controller changes lambda after the damage is already done,
or conditional DINO reintroduces collapse.
```

Interpretation:

```text
PC pass  -> controller mechanics are worth moving to Qwen-1.5B.
PC fail  -> fix telemetry/control logic before spending Colab.
```

## Experiment Plan

### AAMC-000: Telemetry Readiness

Goal:

```text
confirm metrics.jsonl contains all controller signals.
```

Required fields:

```text
step, train_ce, eval_ce if scored, kl, dino_loss if enabled,
FACT when scored, first_token_hit, gold_rank_mean,
entropy, top1_prob, degen_gap, salad_rate, loop_rate, empty_rate,
hidden_var_mid, grad_norm, update_norm
```

Pass:

```text
all fields present for at least one 160M/Qwen-0.5B smoke.
```

### AAMC-001: Fixed-Arm Diagnosis

Use Qwen-1.5B RFIT-A/B/C style fixed arms:

```text
lambda in {0.2, 0.4, maybe 0.5}
alpha = 0
score at 400/600/800
```

Purpose:

```text
estimate whether the failure is overfit, underfit, or collapse before dynamic control.
```

### AAMC-002: Rule-Based Lambda Controller

First dynamic run:

```text
start lambda=0.2
alpha=0
score every 200 steps
raise lambda to 0.4 if overfit_score appears
```

No DINO in this first dynamic run unless collapse_score is nonzero.

Success:

```text
FACT >= fixed best,
eval CE no worse,
train CE not collapsed,
samples less rambling.
```

### AAMC-003: Conditional DINO

Only run if AAMC-001/002 shows generation instability:

```text
alpha=0.05 after the first unstable checkpoint
lambda follows AAMC-002
hidden alignment OFF
```

Success:

```text
degen/salad/loop decrease without hurting FACT or eval CE.
```

## Pass / Fail Criteria

Strong pass:

```text
FACT beats Qwen-0.5B content-KL result 0.333,
first_token_hit improves,
eval CE is not worse than fixed RFIT,
generation tags stay ok.
```

Partial pass:

```text
FACT beats the best Qwen-1.5B fixed run,
but remains below 0.333.
```

Fail:

```text
controller oscillates,
or all controlled runs stay below fixed content-KL/RFIT baselines,
or DINO reintroduces collapse.
```

## Claim Discipline

Do not call AAMC a discovered rule yet.

Allowed current claim:

```text
Our experiments show that fixed objective weights fail differently by model and
scale. AAMC is a proposed telemetry-driven controller that maps observed failure
modes to lambda/DINO adjustments.
```

Do not claim:

```text
AAMC solves factual recovery.
```

It becomes a named empirical rule only if controlled runs beat the fixed-arm
baselines across at least two model/rung settings.
