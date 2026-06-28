# PC Negative Branch Map

Status: current 160M cheap-screen synthesis.

Purpose:

```text
Record what the RTX 3080 / 160M track already ruled out,
what remains open,
and when a ruled-out branch is allowed to reopen.
```

This matters because the project is now branch-heavy. A negative result is not
dead weight; it removes whole classes of future work.

## Scope

This document only summarizes cheap PC-side screens on 160M.

It does not override 1.1B Colab results. The 160M predictor has already been
wrong in magnitude for some factual runs, so the correct reading is:

```text
PC null result = no cheap green light
not an absolute theorem about 1.1B.
```

However, method failures such as:

```text
top-k <= random-k
```

are stronger than simple scale failures, because they say the proposed selector
itself did not add information.

## Summary Table

| branch | question | result | status |
| --- | --- | --- | --- |
| WSYNC scaling | can row/group/row-norm data-free scaling rescue b1.58 PTQ? | no; FACT 0.0 for all ternary arms | closed for data-free |
| H-I2S rotation | can fixed Hadamard projection rescue data-free ternary? | no; worse than per-tensor, FACT 0.0 | closed for data-free |
| SIDE-001 all-layer sidecar | does tiny low-rank auxiliary capacity help everywhere? | weak / non-monotonic; no >=0.05 FACT gain | no cheap green light |
| EGROW-001 locator | can we rank bottleneck layers reproducibly? | yes; top-8 overlap 7/8 | diagnostic survives |
| EGROW-002 targeted sidecar | does B_l top-k sidecar beat random-k / none? | no; top-k <= random-k <= none on eval | selector not actionable at 160M; EGROW-004/005 remain conditional, not executed |

## Branch 1: Data-Free Geometry

Sources:

```text
reports/rt_wsync_160m.md
reports/rt_wsync_160m_hi2s.md
```

### What Was Tested

Data-free b1.58 conversion of 160M without STE adaptation or representative
data:

```text
per-tensor I2_S
row scale
group scale
row-norm correction
block-Hadamard H-I2S
```

### Result

All behavior collapses:

```text
FACT = 0.0 for every ternary transform.
```

Group scaling improves CE relative to per-tensor, but only between broken
states:

```text
per-tensor CE 11.64, ppl 113k
group CE 9.62, ppl 15k
FP CE 3.09, ppl 21.9
```

Hadamard is worse:

```text
h_i2s CE 12.49, ppl 264k, FACT 0.0
```

### Interpretation

Data-free geometry does not solve the conversion problem.

The important project rule:

```text
Do not launch a Colab run just because weight MSE or CE moves inside a collapsed regime.
Behavior must move.
```

### Reopen Condition

Only reopen if the transform is no longer data-free:

```text
rotation + STE adaptation
rotation + activation-aware calibration
learned but cheap/foldable transform
```

That would be a new adaptation/objective branch, not WSYNC.

## Branch 2: Uniform Sidecar Capacity

Source:

```text
reports/side001_160m.md
```

### What Was Tested

I2_S trunk plus LoRA sidecar on all target linears:

```text
rank 0 / 2 / 4 / 8
content-KL 0.2 + PopQA blend 5%
300 steps
```

Byte overhead:

| rank | fp16 sidecar bytes | % of I2_S target bytes |
| ---: | ---: | ---: |
| 2 | 847,872 | 1.50% |
| 4 | 1,695,744 | 2.99% |
| 8 | 3,391,488 | 5.99% |

### Result

No clear FACT lever:

| rank | eval_panel | PopQA tight | CE | status |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.185 | 0.020 | 4.058 | baseline |
| 2 | 0.222 | 0.035 | 4.040 | small, sub-threshold bump |
| 4 | 0.185 | 0.015 | 4.024 | flat |
| 8 | 0.185 | 0.015 | 4.031 | flat + tags degrade |

### Interpretation

Sidecar helps CE slightly, but does not clearly improve factual behavior.

```text
CE improves, FACT flat
```

means the added capacity may be used for local language modeling rather than
recovering missing facts.

### Reopen Condition

Uniform sidecar can reopen only if a 1.1B run shows a capacity plateau after
representative data succeeds or nearly succeeds.

Do not reopen just because rank-2 had a noisy +0.037 on a 27-item panel.

## Branch 3: EGROW Locator

Source:

```text
reports/egrow_160m_layer_instability.md
```

### What Was Tested

Layer bottleneck score:

```text
B_l =
  instability_l
  * output_residual_l
  * task_saliency_l
```

Across two seeds, top-8 overlap was:

```text
7/8
```

Shared hotspots:

```text
layers.0.mlp.down_proj
layers.3.mlp.down_proj
layers.9.mlp.down_proj
layers.10.mlp.down_proj
layers.11.mlp.down_proj
layers.9.self_attn.o_proj
layers.11.self_attn.o_proj
```

### Honest Correction

The original user intuition was:

```text
STE keeps flipping codes => layer is confused => capacity shortage
```

But EGROW-001 found:

```text
flip_rate ~= 0
temporal_entropy ~= 0.002
```

The codes settle. The useful discriminator is:

```text
output_residual x task_saliency
```

So EGROW survives as a sensitivity locator, not an entropy/oscillation locator.

## Branch 4: Targeted Sidecar By EGROW

Source:

```text
reports/egrow002_160m.md
```

### What Was Tested

Rank-4 sidecar on:

```text
top-k layers by B_l
vs
type-matched random-k layers
vs
no sidecar
```

### Result

| arm | eval_panel | PopQA tight | PopQA train | CE | sidecar bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.222 | 0.015 | 0.025 | 4.061 | 0 |
| top-k | 0.148 | 0.025 | 0.050 | 4.042 | 135,168 |
| random-k | 0.185 | 0.020 | 0.037 | 4.063 | 135,168 |

The ordering is:

```text
top-k <= random-k <= none
```

on the noisy eval panel.

### Interpretation

The result is noisy because 27 questions means only a few hits separate arms.
The robust conclusion is not "top-k is bad." The robust conclusion is:

```text
there is no positive localization signal.
```

If B_l located useful growth sites, top-k should clearly beat random-k. It did
not.

Therefore:

```text
B_l ranking = diagnostic
B_l targeted sidecar = not actionable at 160M
```

### What About EGROW-003/004/005?

The EGROW ladder was:

```text
EGROW-001 logger
EGROW-002 top-k sidecar vs random-k on PC
EGROW-003 false-positive controls
EGROW-004 1.1B top-k confirmation
EGROW-005 runtime accounting
```

Only EGROW-001 and EGROW-002 have actually run.

EGROW-004/005 are not "failed." They are gated off because EGROW-002 did not
earn the 1.1B launch condition:

```text
launch EGROW-004 only if top-k > random-k and top-k > none on PC.
```

That condition was not met.

So the precise status is:

| stage | status | reason |
| --- | --- | --- |
| EGROW-001 | done / diagnostic survives | stable sensitivity ranking |
| EGROW-002 | done / negative | top-k did not beat random-k or none |
| EGROW-003 | skipped for now | no positive top-k signal to stress-test |
| EGROW-004 | inactive / conditional | no PC green light for 1.1B top-k sidecar |
| EGROW-005 | inactive / conditional | runtime accounting only matters after quality passes |

Reopen EGROW-004 only if one of these happens:

```text
FACT-003H lands as a plateau and we explicitly decide to test 1.1B capacity anyway;
or a new locator beats random-k on PC;
or a new growth action replaces rank-4 sidecar while keeping the same top-k test.
```

## What Is Now Closed

Closed for the current 160M cheap-screen regime:

```text
data-free scaling
data-free fixed Hadamard rotation
uniform all-layer sidecar
B_l-targeted post-hoc sidecar
post-hoc FP layer restore
small hard replay as generalizing factual data
```

Not closed, but currently inactive:

```text
EGROW-004 1.1B top-k confirmation
EGROW-005 runtime accounting
```

They need a new trigger because EGROW-002 did not supply one.

These should not consume more PC time unless a Colab 1.1B result specifically
reopens them.

## What Remains Open

### 1. Representative Data / Objective

Main active branch:

```text
FACT-003H PopQA blend 1.1B
```

This is the current mainline because:

```text
small hard replay failed by memorization,
but PopQA blend removes the memorization signature on 160M
and has enough diversity to be representative adaptation data.
```

### 2. Capacity, But Only After Data Branch Decides

Capacity is not dead globally. What is dead is cheap post-hoc 160M capacity.

Reopen capacity if FACT-003H lands as:

```text
good CE/tags
PopQA train and heldout move
but FACT eval stays near content-KL baseline
```

Then possible next forms are:

```text
train-from-start sidecar on 1.1B
PTQTP-lite / second ternary plane
selective Q2/Q3 pocket
larger base model
```

### 3. Learned / Adapted Geometry

Data-free geometry is closed. Geometry with adaptation is still possible, but it
must enter as:

```text
transform + representative data + STE
```

not as pure WSYNC.

## Current Decision Rule

Wait for FACT-003H.

If FACT-003H succeeds:

```text
main bottleneck = representative data / objective
capacity/geometry remains closed for now
scale data and model size
```

If FACT-003H plateaus:

```text
data alone insufficient
reopen 1.1B capacity, but not the exact 160M post-hoc sidecar recipe
```

If FACT-003H collapses:

```text
objective/data mixture is unstable
debug data distribution and loss balance before capacity
```

## PC Use While Waiting

Do not run more capacity/geometry screens by default.

Allowed PC work:

```text
eval parser validation
FACT-003H result table preparation
data de-leak / tight-panel improvements
documentation
```

New PC experiments need a fresh hypothesis that is not already killed by:

```text
WSYNC, SIDE-001, EGROW-002
```
