# DINO-DIAG-001 result (160M): the decisive evidence that DINO-logit works

Driver: scripts/dino_diag001_token_analysis.py (seeds 41/42/43 averaged, topk 10, FP teacher
Felladrin/Llama-160M-Chat-v1). Token-level dissection of the 001b sweep's baseline vs dino_logit
checkpoints on the 27 FACT prompts. This is the result that re-classified DINO from "ambiguous
+0.037" to "a real distribution-level positive masked by a 160M decoding ceiling".

## Headline

FACT exact-match moved a little; **gold-token logprob/rank moved a lot.**

| metric (lambda 0.2, seeds avg) | value |
| --- | --- |
| mean delta log P(gold) | **+0.372** (74% of prompts UP) |
| gold rank improved (dino < baseline) | **78% of prompts** |
| mean delta entropy | +0.049 (NOT sharpening) |
| mean teacher-student top-k overlap | 0.53 |

Rank moves (exact-match stayed 0, distribution moved hugely):
cap_italy 48->7, cap_france 60->11, largest_planet 50->8, cap_russia 37->8, closest_planet 87->29,
water_made 73->32. 160M pushes the gold answer toward the top but cannot pull it to rank 1 ->
greedy/rep-penalty decode misses it = a READOUT ceiling, not "nothing happened".

## Category split (lambda 0.2) -- the limit

per-category mean delta log P(gold): **simple_fact +0.574** >> reasoning +0.112 >>
**entity_attr -0.04 ~= instruction -0.004**.

- DINO-logit strongly helps common facts (capital/planet/element).
- It does NOT help, and sometimes regresses, rare entity-attributes: currency_japan(yen) -0.92,
  author_romeo(shakespeare) -0.71 -- note the teacher KNOWS shakespeare (teacher_rank 5) yet the
  student went 258->537. Broad unlabeled text rarely contains these specific entity associations,
  so the content-KL gradient never pushes them.

## lambda 0.1 vs 0.2 -- not a strength problem

lambda 0.1 mean delta gold_logp +0.342 ~= lambda 0.2 +0.372; top-k overlap 0.53 both. Raising
lambda does NOT increase the gold signal (lambda 0.4 breaks generation, sweep). The ceiling is not
lambda -- it is WHAT the unlabeled content-KL can teach (common facts yes, rare entities no).

## Why DINO works / why it is limited

Works: the teacher's soft distribution puts mass on common factual tokens; the student follows it
at the distribution level (gold rank up), NOT by memorising a train set (popqa_train ~0). Not pure
function-token KL either -- the factual gold tokens themselves rose.

Limited: the unlabeled stream under-covers entity-attribute links; soft KL alone is weak for rare
relations; and 160M can raise gold to rank ~7 but not convert it to a rank-1 generation.

## Decision -> DINO-I2S-002 1.1B gate (with predictions)

The new, sharp question ("does a bigger model convert the raised gold mass into exact-match?")
cannot be answered at 160M -> escalate. Predictions to carry in: simple_fact rises; entity_attr
lags; overall may exceed the content-KL 0.185 ceiling; >=0.4 unlikely yet.

Spec: TinyLlama-1.1B, dino_logit-only (lambda 0.2), hidden OFF, EOS/special KL mask ON. Metrics:
FACT exact-match + per-category, gold logprob/rank vs teacher, PopQA tight, tags, i2_s~=f16.
Branch table in docs/dino_i2s_self_distillation_plan.md / the session decision.
