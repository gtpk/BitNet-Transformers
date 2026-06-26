# Complex / Phase Rotation Candidate Note (RT-126B / PHASE-001)

Document position: [Index](./index.md) -> diagnostic branch after
[Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md).

Related:

- [Why Existing Models Resist b1.58 Conversion](./why_b158_conversion_is_hard.md)
- [Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md)
- [Quality Recovery Plan](./quality_recovery_plan.md)

## Purpose

RT-124..127 closed most one-shot quantizer levers:

```text
scale granularity helps but does not rescue;
scale/threshold objective does not help;
activation diagonal scaling does not help;
GPTQ/Hessian assignment helps only modestly;
signed-epsilon 2-bit does not beat ternary.
```

The conclusion was: **the bottleneck is adaptation/objective, not static quantizer design.**

However, rotation/incoherence preprocessing was not directly tested. The user proposed
a more specific version:

```text
Represent rotations through complex phases:
e^{i theta} = cos(theta) + i sin(theta)
```

This document preserves a narrow follow-up candidate:

```text
Can cheap pairwise phase rotations make FP weights more ternary-friendly without
destroying the memory-traffic goal?
```

This is not a return to arbitrary dense rotations. It is a cheap, structured,
mostly-integer/swap/sign branch of the rotation family.

## Mathematical Form

Pair hidden dimensions:

```text
z_j = x_{2j} + i x_{2j+1}
z'_j = z_j * exp(i theta_j)
```

In real coordinates this is:

```text
[a']   [ cos(theta)  -sin(theta) ] [a]
[b'] = [ sin(theta)   cos(theta) ] [b]
```

For a linear layer:

```text
XW = (XR)(R^T W)
```

If `R` is orthogonal/unitary, the FP function is preserved before quantization.
Quantization is then applied to:

```text
W_rot = R^T W
Wq = Q(W_rot)
```

The hope is that `W_rot` is more incoherent / less axis-aligned / less outlier-heavy,
so ternary rounding loses less useful information.

## Why This Might Help

Ternary quantization is coordinate-dependent:

```text
Q(W) != R Q(R^T W)
```

If a few coordinates carry large or delicate values, nearest ternary rounding can be
bad. A rotation can spread that information across coordinates. This is the core
intuition behind QuIP/QuaRot/SpinQuant-style methods.

For our project, the interesting case is not arbitrary learned rotation. It is:

```text
Can very cheap rotations improve b1.58 enough to matter?
```

## Runtime Classes

| Class | theta choices | Operation | Runtime viability |
| --- | --- | --- | --- |
| sign/swap phase | `0, pi/2, pi, 3pi/2` | sign flip and pair swap | very cheap |
| Hadamard phase | `+/- pi/4` | `(a+b)/sqrt(2), (a-b)/sqrt(2)` | cheap-ish; scale handling needed |
| random block Hadamard | fixed +/- mixing | add/sub tree | plausible but not free |
| learned phase | arbitrary theta | sin/cos multiply | upper bound only; not low-resource path |
| dense orthogonal | arbitrary dense `R` | dense matmul | diagnostic only; violates goal |

Only the first two classes should be considered deployment-relevant.

## Hypotheses

### H1: Cheap phase rotation reduces ternary CE loss

Expected signal:

```text
phase-rotated ternary CE < nearest ternary CE by >= 0.5 nats
```

If true, phase rotation deserves combination with QAT/adaptation.

### H2: Rotation only helps if dense/learned

Expected signal:

```text
learned/dense upper bound helps,
cheap phase does not.
```

If true, the idea is scientifically interesting but not aligned with the
memory-traffic-first product goal.

### H3: Rotation barely helps

Expected signal:

```text
best cheap phase improvement < 0.2 nats
```

If true, RT-124..127 synthesis stands unchanged: one-shot quantizer tricks are not
the lever; adaptation/objective remains the path.

## Experiment Ladder

### PHASE-001A: Local Pairwise Phase Screen

Model:

```text
JackFram/llama-160m
```

Baseline:

```text
FP CE
nearest per-tensor b1.58 CE
RT-124A row/group results
RT-125 GPTQ result
```

Candidates:

```text
identity
pair swap/sign search: theta in {0, pi/2, pi, 3pi/2}
fixed pi/4 Hadamard pair rotation
random pi/4 sign pattern
```

Minimum metric:

```text
CE/PPL after PyTorch materialization
reconstruction error
outlier/incoherence statistics before/after
runtime class
```

Pass rule:

```text
cheap phase improves CE by >= 0.5 nats vs nearest ternary
or improves beyond RT-125 GPTQ.
```

### PHASE-001B: Combine With Row Scale

RT-124A showed row-wise scale is the best cheap deployable lever.

Test:

```text
row-scale ternary
phase rotation + row-scale ternary
phase rotation + row-scale + short CE adaptation
```

Pass rule:

```text
phase adds measurable gain on top of row-scale, not only on the weak per-tensor baseline.
```

### PHASE-001C: Learned Phase Upper Bound

Use learned `theta_j` only as an upper bound:

```text
min_theta CE(Q(R(theta)^T W))
```

or layer-local proxy:

```text
min_theta ||XW - X R(theta) Q(R(theta)^T W)||^2
```

Pass rule:

```text
If learned phase helps a lot but cheap phase does not, this is not a near-term runtime
path. It may guide future structured rotations.
```

### PHASE-001D: QAT Integration Gate

Only if PHASE-001A/B passes:

```text
initialize with phase-rotated ternary
run teacher-free CE adaptation
compare to old linears-only QAT
export only if runtime path remains plausible
```

Pass rule:

```text
better CE and lower loop-rate than old QAT at equal training budget.
```

## Important Folding Question

The function-preserving identity:

```text
XW = (XR)(R^T W)
```

requires the activation side to be rotated too. There are three possible handling
strategies:

1. **Explicit runtime rotation**

   Easy to reason about, but may cost too much.

2. **Fold across adjacent linears**

   For compatible blocks, push `R` into the previous/next projection so no standalone
   runtime rotation remains. This is architecture-sensitive.

3. **Use rotation only as initialization**

   Train the quantized model after rotation, then materialize back into a normal
   b1.58 form if possible. This is the safest near-term path.

PHASE-001A is allowed to be an upper-bound screen. Deployment claims require either
strategy 2 or 3.

## What Would Count As A Real Discovery

Strong:

```text
cheap phase + row scale + QAT beats old QAT at same budget,
and preserves I2_S-like runtime cost.
```

Moderate:

```text
cheap phase improves one-shot CE but disappears after QAT.
```

This means phase is only an initialization detail.

Weak / fail:

```text
only learned/dense phase helps, or improvement is <0.2 nats.
```

Then keep the current conclusion: adaptation/objective is the main lever.

## Decision

Do not run PHASE-001A as the immediate next experiment. Keep it as a later diagnostic
candidate after the factual gap work. Run it only if the goal is to falsify the remaining
rotation loophole. Do not promote arbitrary complex rotations to runtime work until cheap
phase rotations show a clear signal.

The current default roadmap remains:

```text
decoding stability -> better adaptation data/objective -> optional phase-rotation init
```

not:

```text
custom complex-valued runtime.
```
