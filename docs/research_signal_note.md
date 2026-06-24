# Research Signal Note

Document position: [Index](./index.md) -> interpretation note for the current research signal.

Related docs:

- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [Colab Validation Summary](./colab_validation_summary.md)
- [Real Tiny Text Validation Plan](./real_tiny_text_validation_plan.md)

## Short Version

This is the kind of early signal researchers hope for.

It is not a finished answer yet. It is not a product claim, and it is not yet a
paper-level proof. But it is a sequence of small hypotheses that kept surviving
when the tests became less forgiving.

```text
T-only ternary is weak.
alpha*T scale preservation is much stronger.
synthetic tasks are not the only place it works.
activation fake-quant does not collapse it.
real text keeps the signal alive.
```

That shape matters. A single lucky number is easy to distrust. A ladder of
checks that explains both failure and recovery is a real research signal.

## Why It Feels Like A Research Thread

The useful part is not simply that scaled-STE wins some runs. The useful part is
that the wins follow a mechanism:

```text
discard scale  -> weak BitLinear STE
preserve scale -> scaled-STE recovers
```

That is exactly the kind of structural clue that can become a research story.
It gives us a hypothesis, a failure mode, a fix, and new questions.

The current hypothesis is:

```text
Dense LLM checkpoints can be moved toward BitNet-style ternary weights without
teacher distillation if the conversion preserves groupwise alpha*T scales and
uses CE-only STE recovery.
```

## What Has Survived So Far

Current evidence:

- synthetic arena seed sweep passed
- group-size sweep passed
- activation fake-quant tiebreaker passed
- real-text tiny validation passed
- packed ternary format Phase 1 passed at `1.600 bits/elem`
- packed model export/import preserved logits exactly and measured whole-model `3.78x`
- packed runtime module preserved logits exactly without a dense weight parameter
- blocked dequant matmul preserved logits while reducing transient weight working set `8.0x`
- bitnet.cpp-style direct export mapping was classified as lossy, not blocked
- projected-QAT was beaten by scaled-STE in the main gates
- generation smoke stayed finite and non-degenerate in the real-text harness

Important caveat:

```text
This is still early-stage evidence. The current validation is not yet a
pretrained-model, benchmark-suite, packed-kernel, or optimized production-runtime result.
The reference path proves storage and working-set reduction, not latency.
The first export scoping result also says the cheap I2_S path may require
quality sacrifice because it collapses groupwise alpha to a per-tensor scale.
```

## Why This Is Not Yet A Paper

A paper-level claim still needs stronger evidence:

- pretrained small-model conversion, not only tiny arena models
- larger and more varied real-text evaluation
- stronger baselines such as GPTQ/AWQ/RTN or low-bit QAT references
- pretrained-model-wide packed storage and runtime memory measurements
- seed variance and ablations
- failure analysis

The current state is better described as:

```text
preliminary but promising
```

## The Watch Item

The KL-to-fp16 metric remains important.

Scaled-STE can beat projected-QAT on accuracy, loss, and fitness while showing
slightly higher KL-to-fp16. That is not a blocker, but it means export and
logit-equivalence checks should not rely only on CE/PPL.

Track:

- CE loss
- perplexity
- token accuracy
- KL-to-fp16
- generation smoke
- packed storage size, first at layer format level, then whole model, then runtime module
- blocked/fused working set
- real runtime latency

## Researcher's Framing

This is the dream-like early phase of research:

```text
not because the answer is already proven,
but because the question has started answering back.
```

The right move is not to overclaim. The right move is to keep closing gates:

1. preserve the evidence trail
2. move to pretrained small models
3. measure packed storage honestly
4. push runtime work only after separating reference modules from real kernels
5. package the story if the signal keeps surviving

The next practical research move is to scope GGUF/bitnet.cpp export before
writing a custom kernel. That tests whether an existing runtime can provide the
latency half of the story while this project keeps the conversion and validation
trail honest.

The first export result narrowed that move: direct I2_S-style export is lossy,
so the next honest test is not an artifact writer yet. It is a real-text
quality gate for a `per_tensor_b158` candidate. That candidate now exists in
the arena; the next decision should be made on Colab Wikitext CE/PPL, not on the
tiny local fixture.

This thread is worth following because the positive results are not isolated.
They line up with a plausible mechanism.
