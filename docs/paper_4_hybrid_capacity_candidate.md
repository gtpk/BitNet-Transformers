# Paper 4 Candidate: Hybrid / Variable Capacity BitNet

Working title:

```text
Variable-Capacity b1.58 Conversion: Mostly-I2_S Models with Precision Pockets
```

Status: candidate only. This is not a paper yet; it becomes one if HYBRID-001 produces
a positive signal.

Central table: [Paper Evidence Matrix](./paper_evidence_matrix.md).

## Draft Abstract Placeholder

This is not yet a result paper. The hypothesis is that an all-I2_S, same-topology
converted model may be too capacity-constrained to recover factual behavior even
when the adaptation objective is improved. If content-KL plateaus below useful
factual quality, we will test whether small precision or capacity pockets can
recover facts while preserving most of the I2_S memory-traffic benefit. Candidate
interventions include late-layer F16 restores, attention-only or MLP-only restores,
Q2/Q3 replacement for helpful regions, multi-strip ternary representations, and
low-rank residuals. The paper becomes valid only if HYBRID-001 shows a positive
quality/storage tradeoff against the best all-I2_S content-KL baseline and a simple
Q2_K baseline.

## Planned Result Table

| arm | storage cost | runtime class | fact score | CE/PPL | decision |
| --- | ---: | --- | ---: | ---: | --- |
| all-I2_S + best content-KL | baseline | fastest | TBD | TBD | baseline after lambda sweep |
| last 1 block F16 | TBD | mostly I2_S | TBD | TBD | HYBRID-001A |
| last 2 blocks F16 | TBD | mostly I2_S | TBD | TBD | HYBRID-001A |
| last 4 blocks F16 | TBD | slower | TBD | TBD | HYBRID-001A |
| last 2 attention-only F16 | TBD | mostly I2_S | TBD | TBD | HYBRID-001A |
| last 2 MLP-only F16 | TBD | mostly I2_S | TBD | TBD | HYBRID-001A |
| helpful region Q2/Q3 | TBD | existing quant kernels | TBD | TBD | run only if F16 restore helps |
| multi-strip ternary R=2/R=4 | TBD | custom/experimental | TBD | TBD | run only if capacity signal is clear |
| low-rank residual | TBD | extra matmul | TBD | TBD | run only if storage budget allows |

## Blank Cells Before This Becomes A Paper

| blank | why it matters | next action |
| --- | --- | --- |
| content-KL plateau point | hybrid should answer a real plateau, not impatience | finish FACT-003C lambda sweep |
| helpful region | determines whether capacity is localized | HYBRID-001A restore scan |
| budgeted comparison to Q2_K | product path must beat a practical baseline in at least one axis | compare bytes, tg, fact, PPL |
| runtime feasibility | F16 pockets may reduce speed advantage | measure or estimate mixed runtime |

## Thesis Candidate

If same-topology all-I2_S plus content-KL cannot recover enough factual quality, the
model may need more capacity in selected regions:

```text
mostly I2_S
+ late-layer precision pockets
+ attention/MLP selective restore
+ multi-strip ternary
+ low-rank residual
```

## Motivation

Hints already collected:

```text
native BitNet is not just plain LLaMA + ternary weights
public BitNet uses BitLinear, SubLN, relu2, no bias, native training
BitNet Reloaded suggests capacity expansion can matter
same-shape one-shot conversion fails
FACT recovery is still far below Q2_K/FP
```

## First Experiment: HYBRID-001A

Run only if FACT-003C content-KL plateaus below useful factual quality.

Arms:

```text
all-I2_S baseline
last 1 block F16
last 2 blocks F16
last 4 blocks F16
last 2 attention-only F16
last 2 MLP-only F16
```

Metrics:

```text
FACT panel
PPL/CE
generation tags
storage proxy
runtime feasibility class
```

Pass signal:

```text
fact_rate +0.15 absolute over content-KL baseline
or fact_rate >=0.30 with no degeneration
```

## Follow-Up If Positive

```text
Q2/Q3 replacement for the helpful region
multi-strip ternary R=2/R=4
low-rank residual AB
whole-model re-adaptation under chosen topology
```

## Follow-Up If Negative

```text
Do not keep adding capacity blindly.
Return to objective/replay or larger base model.
```

## Not Yet A Claim

Do not cite hybrid as a result until:

```text
HYBRID-001A has a positive factual signal
and the cost/quality scorecard beats a simple Q2_K tradeoff in at least one dimension.
```

## Evidence Links

| evidence | link |
| --- | --- |
| central matrix | [Paper Evidence Matrix](./paper_evidence_matrix.md#paper-4-hybrid-capacity-candidate-evidence) |
| hybrid plan | [Hybrid / Variable BitNet Conversion Plan](./hybrid_variable_bitnet_conversion_plan.md) |
| sidecar plan | [I2_S + LoRA / Residual Sidecar Plan](./i2s_lora_sidecar_plan.md) |
| entropy-guided growth plan | [Entropy-Guided I2_S Growth Plan](./entropy_guided_i2s_growth_plan.md) |
| PC negative branch map | [PC Negative Branch Map](./pc_negative_branch_map.md) |
| measured ledger | [Evidence Ledger](../reports/EVIDENCE_LEDGER.md#f-capacity--geometry-track----comprehensively-negative-at-160m-cost-ledger-rdt-001) |
