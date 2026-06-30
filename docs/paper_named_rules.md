# Named Rules And Principles

Document position: [Index](./index.md) -> paper series -> named rules.

Status: working vocabulary for the paper series, 2026-06-30.

## Purpose

The project has produced many experimental facts. This document turns only the
important observed facts into named rules so that each paper can speak in a
stable language instead of retelling the whole history.

Naming rule:

```text
Name a rule only when our experiment actually observed the pattern.
Do not name strategy, taste, or future intent as a "law".
If it is not observed yet, mark it as Hypothesis or Decision Note.
```

The names are intentionally compact. A rule is not a theorem unless the evidence
matrix says it is solved; most are empirical observations with clear scope.

## Do-Not-Forget Summary

This is the one-page memory card for the current project state.

### What Is Solved

```text
If we can make ternary-compatible weights, I2_S can run them faithfully and fast.
```

- **Gamma Repack Law** is the key export trick: do not export latent FP weights
  directly. Materialize `Wq = gamma*T`; then upstream absmax I2_S stores our
  `gamma`.
- **Linear-Dominance Compression Law** explains why larger dense LLaMA-like
  models get closer to the 16x target-linear storage floor.
- **Memory-Traffic Amplification Rule** explains why token-generation speedups
  grew with scale.
- **Runtime Fidelity Rule** exonerates I2_S once the weights are already
  ternary-compatible: runtime parity is not the quality bottleneck.

### What Is Not Solved By Plain Quantization

```text
One-shot same-shape gamma*T conversion is not enough.
```

- **Same-Shape Ternary Gap** is real: native BitNet working from scratch does not
  mean a dense checkpoint can be converted by one pass.
- **Quantizer Lever Ceiling** says scale/threshold/GPTQ/AWQ-like tricks help only
  a little compared with the b1.58 gap.
- **Non-Additive Restore Principle** warns that layer-wise fixes can hurt because
  the all-ternary model is co-adapted.
- **Adaptation Dominance Rule** is the current practical lesson: data/objective
  adaptation matters more than another small codebook tweak.

### What Actually Moves Quality

```text
Quality failures split into objective, dynamics, and readout.
```

- **Content-Anchor Sweet Spot**: content-KL is the first real factual lever, but
  only at the right strength.
- **Stop-Mass Contamination Rule**: raw KL can copy EOS/empty behavior instead of
  content.
- **Transient Collapse Rule** and **Consolidation-Length Scaling Rule** corrected
  our earlier false negative: TinyLlama-1.1B did not "impossibly collapse"; 800
  steps stopped during a transient.
- **Reachable-But-Not-Emitted Rule** is the newest bottleneck: facts can move up
  in rank while generation still fails to emit concise answers.

### What To Avoid Repeating

```text
Do not re-run cheap branches already falsified unless the premise changes.
```

- Weight-only scaling/rotation screens did not rescue ternary collapse.
- Post-hoc FP restore did not help; it broke co-adaptation.
- Small hard factual replay overfit and did not generalize at 1.1B.
- Sidecar/top-k bottleneck localization did not become an actionable quality
  lever at 160M.
- Reconstruction-only PT2 projection improved weight metrics but worsened
  behavior.

### Current Best Mental Model

```text
The project is not "quantize weights and hope".
It is a low-bit conversion compiler:

export math
  + ternary-compatible adaptation
  + collapse dynamics
  + factual readout / answer format
  + fair final artifact comparison
```

### Current Open Questions

- Can answer-site weighting or short-answer formatting turn reachable factual
  rank into exact factual answers?
- Can a telemetry-driven AAMC controller beat fixed `lambda`/DINO arms by
  changing the objective weights only when overfit or collapse telemetry asks
  for it? This is an open hypothesis, not a named rule.
- Can PT2-style activation-aware initialization improve the starting point
  without leaving the I2_S-rooted artifact goal?
- If hybrid capacity returns, can it be trained jointly instead of patched in
  post-hoc?
- Which final artifact wins under the fair scorecard: native BitNet, Q2_K, PT2,
  pure I2_S+adaptation, or I2_S-rooted hybrid?

## Paper 1: I2_S Systems

### Gamma Repack Law

If a trained ternary weight is materialized as:

```text
Wq = gamma * T,  T in {-1, 0, +1}
```

then upstream absmax I2_S quantization stores `max(abs(Wq)) = gamma`. This is why
Path A' preserves our `mean(abs)` scale while direct latent-FP I2_S export does
not.

Evidence: RT-104D, RT-112.

### Linear-Dominance Compression Law

Target linears always approach the I2_S floor:

```text
I2_S / f32 target-linear storage ~= 1 / 16
```

Whole-model storage approaches that floor as linear layers dominate embeddings
and other fp16/f32 floor terms.

Evidence: RT-113/114/115.

### Memory-Traffic Amplification Rule

When generation is weight-bandwidth bound, reducing target-linear traffic gives
larger token-generation speedups at larger model scale.

Evidence: tiny -> Llama-160M -> TinyLlama-1.1B speedup trend.

### Runtime Fidelity Rule

For already ternary-compatible weights, I2_S runtime error is not the primary
quality bottleneck: `i2_s ~= f16/f32` under the same evaluation path.

Evidence: RT-111/112/114/115 and adapted F16/I2_S parity checks.

## Paper 2: Conversion Limits

### Same-Shape Ternary Gap

Native BitNet success does not imply that a pretrained dense checkpoint can be
mapped into the same topology with one `gamma*T` pass. Same-shape one-shot b1.58
conversion loses too much function.

Evidence: RT-121, RT-124..127.

### Quantizer Lever Ceiling

Scale granularity, threshold selection, diagonal smoothing, GPTQ-style
assignment, and signed-epsilon codebooks move the loss, but their gains are
small compared with the post-training ternary gap.

Evidence: RT-124A/B/C, RT-125, RT-127.

### Non-Additive Restore Principle

Layer-level interventions are not additive after a model has co-adapted to the
all-ternary state. Restoring a layer or group to FP can hurt because upstream and
downstream distributions no longer match.

Evidence: RT-123, HYBRID-001A.

### Adaptation Dominance Rule

At this bit budget, data/objective adaptation dominates pure quantizer design.
Quantizer improvements can initialize or assist, but do not replace adaptation.

Evidence: RT-116/120 vs RT-124..127.

## Paper 3: Content-KL Factual Recovery

### Runtime Exoneration Rule

When adapted F16 and adapted I2_S agree, factual failure belongs to the adapted
model/objective, not the runtime.

Evidence: FACT-001/002/003 parity.

### Stop-Mass Contamination Rule

Raw KL on chat-style answer tokens can copy the teacher's stop behavior instead
of preserving content, producing empty outputs.

Evidence: FACT-003B.

### Content-Anchor Sweet Spot

Content-KL helps only in a finite band. Too weak gives no anchor; too strong
over-anchors and washes out task-specific facts.

Evidence: FACT-003C lambda sweep.

### Fluent-But-Fact-Poor Rule

CE/PPL recovery can restore fluent generation without restoring factual recall.
Fluency and factual readout must be measured separately.

Evidence: FACT-001/002, RT-122/129.

## Paper 4: Hybrid Capacity Candidate

### Post-Hoc Capacity Mismatch Rule

Adding precision after training is not equivalent to training with that capacity
from the start. Post-hoc FP restore can break co-adaptation.

Evidence: HYBRID-001A.

### Sensitivity-Is-Not-Actionability Rule

A layer can be a stable sensitivity hotspot without being a useful location for
extra capacity. Ranking bottlenecks does not guarantee sidecar gains.

Evidence: EGROW-001 vs EGROW-002.

### Joint-Hybrid Hypothesis

If hybrid capacity helps, it likely needs to be present during adaptation so the
network can co-adapt around it.

Status: hypothesis inferred from negative post-hoc restore and sidecar screens,
not yet a discovered rule.

## Paper 5: Collapse Dynamics

### Transient Collapse Rule

An intermediate degenerate generation state is not necessarily final collapse.
Some models enter a degenerate transient and later consolidate into stable
generation.

Evidence: Pythia-410M/1B and TL1B-1600.

### Consolidation-Length Scaling Rule

The duration of the degenerate transient grows with model/checkpoint conditions.
An 800-step run can misclassify a slow-consolidating model as failed.

Evidence: Pythia ladder and TinyLlama 800 vs 1600.

### Teacher-Relative Collapse Rule

Collapse should be measured against the model's own FP teacher behavior, not by
absolute generation labels alone.

Evidence: Pythia teacher-baseline correction.

### Dynamics-Before-Endpoint Principle

Loss, entropy, top-1 probability, gold rank, hidden variance, and degeneration
rates must be logged over time. Final score alone hides the mechanism.

Evidence: collapse-dynamics audit and TL1B-1600 reinterpretation.

## Paper 6: Factual Readout / Answer Format

### Reachable-But-Not-Emitted Rule

A factual token can become much more likely internally yet still not appear in
the generated answer because the model emits a rambling or wrong answer format.

Evidence: TL1B-1600 gold-rank improvement with low exact FACT.

### Rank-Exact Decoupling Rule

Gold-rank improvement and exact-match improvement are separable. Exact-match
requires the correct token to win the decoding competition at the answer site.

Evidence: DINO diagnostics and TL1B-1600.

### Format Drift Rule

Base-LM-style continuation can be fluent but unusable for factual QA. Answer
format is an independent variable from fluency.

Evidence: TL1B-1600 generation samples.

## Paper 7: PT2-I2S Model Competition

### Reconstruction-Is-Not-Behavior Rule

Lower weight MSE or better reconstruction can still worsen behavior. Behavioral
scorecards are the final arbiter.

Evidence: PT2-I2S-001 and WSYNC/rotation failures.

### Projection-Survival Hypothesis

Ideas that improve an asymmetric PT2 artifact may not survive projection back
into pure I2_S. Exact, projected, and adapted variants must be measured
separately.

Status: hypothesis / comparison protocol. It should not be cited as a discovered
law until PT2-I2S-002 or later fills the scorecard.

## Cross-Paper Meta Rules

### Do Not Trust A Single Endpoint

Endpoint-only interpretation caused several false negatives. Log trajectories
whenever the model can enter a transient.

### Separate Runtime, Representation, Objective, And Decoding

Many failures looked similar at the surface but had different causes:

```text
runtime mismatch
representation/capacity gap
objective/data forgetting
generation/answer-format readout
```

The project advanced when these were separated.

### Negative Results Are Branch Pruners

WSYNC, sidecar, EGROW, hard replay, and one-shot PTQ negatives are useful because
they remove cheap but misleading branches and leave sharper questions.
