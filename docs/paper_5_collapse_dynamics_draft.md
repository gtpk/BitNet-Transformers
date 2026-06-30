# Paper 5 Draft: Generation Collapse Is A Transient

Status: short 2-3 page draft. Needs plots before submission.

Related skeleton: [Paper 5 Skeleton](./paper_5_collapse_dynamics.md)  
Central evidence: [Pythia Ladder Results](../reports/pythia_ladder/RESULTS.md)

## Abstract

Low-bit adaptation failures are often judged at a fixed final step. We show that this
can be misleading: generation collapse can be a transient that resolves later. Using
teacher-relative telemetry on Pythia-160M, Pythia-410M, Pythia-1B, and TinyLlama-1.1B,
we track degenerate generation, gold answer rank, entropy/top-1 behavior, hidden
variance, and CE over training. Pythia shows increasing transient duration with
scale but recovers by 800 steps through 1B. TinyLlama-1.1B appeared collapsed at 800
steps, yet a 1600-step run recovered generation stability and CE. The result
separates generation stability from factual correctness: the same TL1B-1600 model is
fluent and stable but still factually weak. We argue that low-bit adaptation should
be studied as a dynamical process, not only as a final-score comparison.

## 1. Method

For each run we compare the student to its own FP teacher. Collapse is measured not
as an absolute label but as a teacher-relative gap:

```text
degen_gap = degenerate_rate(student) - degenerate_rate(teacher)
```

We also track:

```text
gold_rank
train_ce
entropy / top1
hidden_var_mid / hidden_var_last
salad / loop / empty tags
```

This instrumentation prevents confusing a base-model style difference with real
student degeneration.

## 2. Results

| model | transient | final generation | final rank | interpretation |
| --- | --- | --- | ---: | --- |
| Pythia-160M | none | stable | 272 | small scale stable |
| Pythia-410M | step ~50-250 | recovers | 128 | slow consolidation |
| Pythia-1B | step ~0-250 | recovers | ~150 | no generic 1B wall |
| TL1B-1600 | unresolved at 800; recovers by ~850 | ok 27/27 | 375 | 800-step failure was premature |

The critical reversal is TinyLlama. Earlier DINO-002 looked like hard collapse at
800 steps. TL1B-1600 shows that the model was still in a transient. Generation
recovered, CE recovered, and degen_gap became clean.

## 3. What Did Not Recover

Factual exact answers did not recover. TL1B-1600 achieved FACT 0.111, below the
content-KL baseline 0.185. The samples are fluent but rambling, e.g. the model
continues in base-LM style rather than giving `A: Paris`. This means collapse
dynamics and factual readout must be separate papers.

## 4. Implication

The next low-bit adaptation experiments should report trajectories, not only final
tables. A failed step-800 run may be a failed schedule, not a failed method.

## 5. Named Rules

This paper uses four named rules from
[Named Rules And Principles](./paper_named_rules.md):

- **Transient Collapse Rule**: degenerate generation during training can be a
  temporary phase, not final failure.
- **Consolidation-Length Scaling Rule**: larger or different checkpoints may need
  longer budgets before generation stabilizes.
- **Teacher-Relative Collapse Rule**: collapse should be judged against the
  model's own FP teacher behavior.
- **Dynamics-Before-Endpoint Principle**: trajectory metrics are mandatory because
  endpoint scores can mislead.

## References

- Pythia model suite: <https://arxiv.org/abs/2304.01373>
- DINO self-distillation: <https://arxiv.org/abs/2104.14294>
- Neural text degeneration: <https://arxiv.org/abs/1904.09751>
- Internal evidence: [Pythia Ladder Results](../reports/pythia_ladder/RESULTS.md), [TL1B-1600 Summary](../reports/pythia_ladder/tl1b_long_metrics_summary.md)
