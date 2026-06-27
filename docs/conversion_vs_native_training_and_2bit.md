# Conversion vs Native BitNet Training vs 2-bit Quantization

Document position: [Index](./index.md) -> hard comparison note after FACT-003D.

Last updated: 2026-06-27.

## The Uncomfortable Question

The project goal is not to win a purity contest. The goal is:

```text
turn existing public models into small, fast, useful low-resource artifacts.
```

This raises two hard questions:

```text
1. If b1.58 conversion needs too much adaptation, is it just native BitNet training again?
2. If 2-bit quantization already exists, how is this different from trying harder at Q2?
```

This document fixes the comparison axes so the project does not fool itself.

## Three Different Problems

### A. Native BitNet training

```text
random/base initialization -> train inside the b1.58 constraint from the beginning
```

Strength:

```text
the whole network learns a BitNet-native internal representation.
```

Cost:

```text
pretraining-scale tokens and compute.
```

### B. Standard 2-bit quantization

```text
existing FP model -> low-bit artifact with little/no training
```

Examples:

```text
Q2_K, GPTQ/AWQ/QuIP-like methods, AQLM, other 2-4 bit PTQ/QAT families
```

Strength:

```text
preserves quality much better today.
```

Cost:

```text
larger weight traffic than I2_S and often less specialized runtime speed.
```

### C. Our b1.58 conversion path

```text
existing FP model
-> per-tensor ternary target linears
-> I2_S runtime artifact
-> short adaptation/objective/replay
```

Strength:

```text
smallest/fastest runtime substrate we have tested.
```

Cost/risk:

```text
quality recovery may require enough adaptation that it approaches retraining.
```

## Why Native BitNet Can Work While Conversion Is Hard

Native BitNet trains the model while the constraint is part of its life:

```text
weights, activations, norms, residual paths, and layer-to-layer distributions
co-adapt to the ternary lattice.
```

Conversion starts from a dense solution:

```text
W_fp in a continuous high-precision basin
```

and projects it into:

```text
W_b1.58 = gamma * T
T in {-1, 0, +1}
```

That is not a small perturbation. It is a change of internal language.

Good shorthand:

```text
native BitNet learns inside the low-precision manifold.
conversion tries to translate an FP model into that manifold after the fact.
```

The risk:

```text
If the translation requires too many tokens, it becomes native training in disguise.
```

## Current Evidence

### Good news

| evidence | meaning |
| --- | --- |
| I2_S x86 parity | runtime/export is not the blocker |
| storage/speed scale law | b1.58 gives real memory-traffic benefit |
| CE adaptation recovery | one-shot collapse is not irreversible |
| content-KL | objective design can move facts without EOS collapse |
| protected replay at 160M | factual recovery may transfer beyond memorised items |

### Bad news

| evidence | meaning |
| --- | --- |
| Q2_K beats our 160M PPL | standard 2-bit quantization remains stronger on quality |
| FACT-003C fact 0.185 | same-topology objective tuning is not enough for product quality |
| FACT-004A lm_head unfreeze failed | more trainable freedom can increase forgetting |
| HYBRID-001A post-hoc restore failed | capacity cannot be patched after all-I2_S co-adaptation |
| one-shot PTQ tools insufficient | better thresholds/codebooks are not the main lever |

Current honest status:

```text
systems substrate: solved enough to be useful
quality parity: not solved
adaptation cost scaling: unknown and now central
```

## How This Differs From "Just 2-bit Quantization"

### Standard 2-bit quantization optimizes for preservation

Q2-like methods generally ask:

```text
How do we preserve the FP model with minimal or no training?
```

Success metric:

```text
PPL/benchmark close to FP, with acceptable storage/speed.
```

### Our path optimizes for a different point in the trade space

Our question is:

```text
Can we accept more adaptation cost to reach a smaller/faster runtime class than Q2?
```

Success metric:

```text
quality per byte/token/sec under a low-resource constraint,
not quality alone.
```

### Concrete difference

| axis | Q2_K / 2-bit PTQ | our b1.58/I2_S path |
| --- | --- | --- |
| training cost | near-zero or calibration-only | nonzero adaptation |
| quality today | stronger | weaker but improving |
| target-linear storage | higher than I2_S | 1.58/2-bit I2_S floor |
| runtime | mature quantized matmul | bitnet.cpp I2_S x86 validated |
| product bet | preserve FP well enough | trade training for memory-traffic speed |
| main risk | speed/storage not enough for poor-resource target | adaptation cost too high |

So yes:

```text
It is related to 2-bit quantization.
```

But no:

```text
It is not the same objective.
```

The fair comparison is not:

```text
Does b1.58 beat Q2_K on PPL today?
```

It is:

```text
How much adaptation cost is needed for b1.58/I2_S to reach a useful quality tier,
and does the speed/storage gain justify that cost?
```

## The Key Unknown: Adaptation Scaling

The central curve is:

```text
factual_score = F(adaptation_tokens, replay_size, objective, model_size)
```

We need to know whether it behaves like:

### Case 1: Cheap conversion

```text
0.185 -> 0.30 -> 0.45 -> 0.60
with millions to tens of millions of adaptation tokens
```

Interpretation:

```text
conversion is meaningfully cheaper than native BitNet training.
```

### Case 2: Slow plateau

```text
0.185 -> 0.25 -> 0.30 -> plateau
```

Interpretation:

```text
same-topology b1.58 conversion is quality-limited; need hybrid/capacity or stronger model.
```

### Case 3: Pretraining-scale requirement

```text
requires billions/trillions of tokens to approach Q2_K
```

Interpretation:

```text
the approach is essentially native BitNet retraining and loses the conversion advantage.
```

FACT-003D is important because it is the first serious test of this curve on facts.

## Target Tiers

Avoid saying "success" without naming the tier.

| factual score on fixed panel | tier | meaning |
| ---: | --- | --- |
| <= 0.20 | weak | fluent but factually poor |
| 0.25-0.35 | mechanism success | replay/objective moves unseen facts; not product-quality |
| 0.40-0.55 | usable-ish start | candidate low-resource artifact, still below Q2_K |
| 0.60-0.70 | strong | approaching Q2_K behavior |
| >= 0.74 | Q2_K parity | major result under our current panel |
| >= 0.81 | FP reference | dense reference level |

Our short-term product-direction target:

```text
0.40+
no degeneration
I2_S ~= F16
speed/storage advantage preserved
```

Our research/mechanism target:

```text
0.25-0.35 is enough to prove protected replay transfers,
but not enough to call the model good.
```

## Model Changes Require Revalidation

Every new base model requires at least:

```text
1. FP factual reference
2. Q2_K factual reference
3. one-shot I2_S collapse check
4. best adapted all-I2_S recipe
5. protected replay / objective branch
6. I2_S vs F16 parity
7. storage/speed
```

Reason:

```text
The conversion tax depends on tokenizer, architecture, pretraining state,
model size, and how much factual knowledge the base model already has.
```

The ladder is reusable; the numbers are not.

## Decision Rule: When To Keep Going

Continue b1.58 conversion if:

```text
FACT-003D or successors show monotonic gains with modest adaptation budget;
speed/storage advantage remains large;
quality reaches at least 0.40 on simple facts.
```

Pivot or narrow the claim if:

```text
factual score plateaus below 0.30 despite replay/objective scaling;
adaptation cost grows toward pretraining scale;
Q2_K remains much better at acceptable speed/storage.
```

Use hybrid/capacity if:

```text
objective/replay improves but stalls below product tier,
and trained-from-start hybrid shows gains without huge runtime cost.
```

## What To Compare In Future Tables

Use this table shape for any serious claim:

| method | from-scratch tokens | conversion/adaptation tokens | trainable params | target bits | size | tg speed | factual | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| native BitNet | huge | n/a | all | 1.58 | low | high | TBD/high | expensive pretraining |
| Q2_K | base sunk cost | near 0 | none | ~2.6 | medium | medium | 0.74 ref | best practical baseline |
| ours all-I2_S | base sunk cost | TBD | target linears | 1.58 | smallest | fastest | TBD | adaptation cost is the question |
| ours hybrid | base sunk cost | TBD | target + selected | mixed | small | fast-ish | TBD | if all-I2_S quality plateaus |

This prevents unfair comparisons.

## Bottom Line

The project is not currently at native BitNet quality. It is also not proven to require
native BitNet-scale training.

The central question is now:

```text
Can factual quality scale with modest post-training cost,
or does b1.58 conversion collapse into retraining?
```

And the practical comparison against 2-bit quantization is:

```text
Can I2_S buy enough speed/storage to justify the extra adaptation cost,
while reaching a usable factual tier?
```

That is the honest problem statement from here.
