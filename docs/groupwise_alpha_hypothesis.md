# Groupwise Alpha Hypothesis

Document position: [Index](./index.md) -> mechanism note for why this fork's
scaled-STE path may outperform direct BitNet b1.58-style per-tensor export.

Related docs:

- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [GGUF / bitnet.cpp Export Scoping Plan](./bitnet_cpp_export_scoping.md)
- [Research Signal Note](./research_signal_note.md)

## Gate Result (2026-06-24): strong form REFUTED

The native per-tensor gate below settled this. The strong hypothesis ("groupwise
local scale is the *source* of the quality gain, so per-tensor must lose") is
**refuted**. A per-tensor b1.58 model trained natively with CE-only STE matches
groupwise scaled-STE within +-1% PPL on Wikitext (seeds 31/32/33), stays on the
Pareto frontier 2/3 (and wins outright on seed 33), and keeps low KL-to-fp16.

```text
PPL (Wikitext), per_tensor_native vs groupwise scaled-STE:
  seed 31: 6.28 vs 6.34   (-0.9%)
  seed 32: 6.01 vs 5.95   (+1.0%)
  seed 33: 6.71 vs 6.71   ( 0.0%)
post-hoc per-tensor export of the groupwise model: 9.85 / 10.55 / 11.31 (broken)
```

So the earlier +18.4% (fixture) and +67% (real-text) per-tensor losses were a
**post-hoc conversion artifact (hypothesis B)**, not evidence that per-tensor is
inherently weak (hypothesis A). Trained per-tensor from the start, the model
recovers fully. Practical consequence: per-tensor-native training is the right
export source. Official bitnet.cpp I2_S has since been verified on x86, while the
local Mac M5 build is broken; our own x86 I2_S artifact also passed in RT-112 via
ternary-dense Path A'. Groupwise alpha remains a legitimate format with slightly
better reconstruction, but it is not necessary for the current default path.

## Short Version

The original working hypothesis (now refined by the gate above) was:

```text
The quality gain is mostly from preserving local weight scale.
```

Both BitNet b1.58/bitnet.cpp and this project use ternary weights:

```text
T in {-1, 0, +1}
```

The important difference is the scale structure.

```text
BitNet / bitnet.cpp direct path:
    W ~= gamma * T
    gamma is one per-tensor scale

This project:
    W ~= alpha_g * T_g
    alpha_g is groupwise, along input blocks
```

The groupwise path has more local scale freedom, so it can preserve the shape of
the original dense weight better before and during CE-only STE recovery.

## Algorithm Difference

BitNet b1.58-style per-tensor approximation:

```text
gamma = mean(abs(W))
T = clamp(round(W / gamma), -1, 1)
W_approx = gamma * T
```

This fork's S1 / scaled-STE approximation:

```text
for each input block g:
    threshold_g = lambda * mean(abs(W_g))
    T_g = sign(W_g) if abs(W_g) > threshold_g else 0
    alpha_g = mean(abs(W_g) over T_g != 0)
    W_approx_g = alpha_g * T_g
```

The first uses one scale for a whole tensor. The second uses many scales, one
per output row and input block.

## Why Groupwise Alpha Can Win

LLM linear weights are not uniform inside a matrix. Different rows, attention
projections, MLP projections, heads, and input-channel regions can have
different magnitude distributions.

With one global scale:

```text
large regions set gamma too high for small regions
small regions set gamma too low for large regions
```

Either small weights collapse toward zero, or large weights saturate. Groupwise
`alpha` reduces that conflict because each block gets its own local magnitude
reference.

In optimization terms:

```text
per-tensor:
    minimize roughly ||W - gamma * T||

groupwise:
    minimize roughly sum_g ||W_g - alpha_g * T_g||
```

The groupwise version has more scale degrees of freedom while keeping ternary
codes. That usually lowers reconstruction error, at the cost of scale metadata
and a harder runtime/export format.

## Why Thresholding May Help

The per-tensor BitNet path uses round-and-clamp:

```text
T = clamp(round(W / gamma), -1, 1)
```

This fork uses thresholded sign selection:

```text
T_g = sign(W_g) if abs(W_g) > lambda * mean(abs(W_g)) else 0
```

This behaves like local sparsity selection. Small local weights are treated as
less reliable and sent to zero; surviving weights keep their sign and receive a
masked-mean scale. That may preserve the function better than forcing a global
rounding rule on every projection.

This is still a hypothesis. It needs ablations to separate:

- scale granularity
- thresholding rule
- STE recovery
- activation fake quant

## Why CE-Only STE Matters

The winning candidate is not pure PTQ. It performs short CE-only recovery inside
the constrained `alpha_g * T_g` representation.

That matters because the model can adapt to the new ternary constraint instead
of only accepting a one-shot projection.

Observed pattern so far:

```text
T-only / scale-discarding path       -> weak
groupwise alpha*T path               -> much stronger
groupwise alpha*T + CE-only STE      -> survives larger gates
per-tensor export after scaled-STE   -> loses some recovered structure
```

## Evidence So Far

Current supporting signals:

- scale-less BitLinear STE was weak, while `ScaledBitLinear` recovered quality
- synthetic seed sweep passed
- group-size sweep passed
- activation fake-quant tiebreaker passed
- Wikitext real-text validation passed for groupwise scaled-STE
- packed storage/runtime references preserve groupwise `alpha*T` exactly
- direct bitnet.cpp-style mapping is lossy because it collapses groupwise
  `alpha` to per-tensor `gamma`
- local export mapping gap: per-tensor b1.58 output error was `+18.4%` worse
  than groupwise S1 on the tiny fixture

Current counterweight:

```text
Groupwise alpha is not free.
```

It adds scale metadata and makes direct bitnet.cpp/GGUF export harder. It has
not yet proven end-to-end latency improvement in an optimized runtime.

## Trade-Off

The research trade-off is now clear:

```text
BitNet / bitnet.cpp per-tensor scale:
    simpler format
    easier optimized runtime
    potentially worse post-training conversion quality

This fork's groupwise alpha:
    better local reconstruction
    better early quality signal
    more metadata
    harder export/kernel path
```

This is a useful tension, not a failure. It gives the project a concrete
research question:

```text
How much scale metadata is needed to preserve post-training conversion quality,
and can that metadata still be served efficiently on-device?
```

## Falsification / Next Tests

The strongest form of the hypothesis has already been falsified. Remaining tests
are useful for understanding *why* per-tensor-native recovers, not for deciding
whether direct I2_S export is allowed.

1. ~~`per_tensor_b158` real-text quality gate on Wikitext seeds `31/32/33`.~~
   **DONE (2026-06-24): refuted the strong hypothesis — native per-tensor matches
   groupwise within +-1% PPL. See Gate Result at the top.**
2. Scale granularity sweep: per-tensor, per-row, groupwise `G=32/64/128`.
3. Ternary rule ablation: BitNet round/clamp with groupwise scale vs thresholded
   sign selection with per-tensor scale.
4. Metadata accounting: scale bytes vs quality gained.
5. Pretrained small-model conversion, not only tiny arena models.
6. Runtime path comparison: direct I2_S export vs groupwise GGUF extension vs
   custom fused kernel.
   **DONE for the default path:** per-tensor-native -> ternary-dense Path A' ->
   upstream I2_S passed on x86 in RT-112. Groupwise GGUF and custom kernels are
   fallback/ablation tracks.

Resolved decision:

```text
per-tensor-native b1.58 kept PPL close enough.
Use bitnet.cpp-style I2_S export as the primary runtime path.
Keep groupwise export/kernel work as fallback or ablation, not as the main path.
```
