# Paper 5 Skeleton: Generation Collapse Dynamics

Working title:

```text
Generation Collapse Is A Transient: Dynamics Of Low-Bit LLM Adaptation
```

Status: new strong topic. Draftable as a dynamics report once plots are produced.

Central tables:

- [Paper Evidence Matrix](./paper_evidence_matrix.md)
- [Evidence Ledger](../reports/EVIDENCE_LEDGER.md)
- [Pythia Ladder Results](../reports/pythia_ladder/RESULTS.md)
- [TL1B-1600 Summary](../reports/pythia_ladder/tl1b_long_metrics_summary.md)

## Draft Abstract

Low-bit adaptation runs are often judged only by final loss, perplexity, or a small
generation panel. We show that this hides the central phenomenon: generation collapse
can be a time-local transient. In Pythia-160M/410M/1B, a DINO-style content objective
produces scale-dependent degenerate transients that later consolidate. TinyLlama-1.1B
appeared collapsed at 800 steps, but a 1600-step run recovered clean generation and
CE, proving the earlier failure was a budget-limited transient rather than a hard
impossibility. However, factual exact answer quality remained weak, separating
generation stability from factual readout. The result reframes low-bit adaptation
from "which final objective wins?" to "when does collapse start, how long does it
last, and what signals predict recovery?"

## Thesis

```text
Generation collapse in b1.58/I2_S adaptation is a dynamic transient/consolidation
phenomenon. Apparent failure at a fixed step can be a schedule artifact.
```

## Core Evidence

| run | evidence | link |
| --- | --- | --- |
| Pythia-160M | no transient; DINO positive | [p160m metrics](../reports/pythia_ladder/p160m_metrics.jsonl) |
| Pythia-410M | transient step ~50-250, recovery ~275 | [p410m metrics](../reports/pythia_ladder/p410m_metrics.jsonl), [p410m CUDA](../reports/pythia_ladder/p410m_cuda_metrics.jsonl) |
| Pythia-1B | transient step ~0-250, recovery ~225-300 | [p1b summary](../reports/pythia_ladder/p1b_metrics_summary.md) |
| TinyLlama-1.1B | unresolved at 800, recovered generation by 1600 | [TL1B summary](../reports/pythia_ladder/tl1b_long_metrics_summary.md) |
| precision/optimizer controls | fp32/bf16 and adamw/adamw8bit were not the cause | [fp32 control](../reports/dino_ctrl_fp32_result.md), [adamw control](../reports/dino_ctrl_adamw_result.md) |

## Result Table

| model | transient | degen recovery | final gold_rank | factual exact | verdict |
| --- | --- | --- | ---: | ---: | --- |
| Pythia-160M | none | clean throughout | 272 | n/a | stable |
| Pythia-410M | ~50-250 | recovers | 128 | n/a | stable, slow consolidation |
| Pythia-1B | ~0-250 | recovers | ~150 | n/a | stable, no generic 1B wall |
| TinyLlama-1.1B 800 | unresolved | looked collapsed | poor | 0.111-ish / salad run dependent | premature failure |
| TinyLlama-1.1B 1600 | ~0-850 | recovers | 375 | 0.111 | generation stable, factual readout weak |

## Figures

1. `degen_gap` vs step across Pythia rungs and TL1B.
2. `gold_rank` vs step across the same runs.
3. `train_ce`, entropy/top1, and hidden variance around transient onset/recovery.
4. Example generations: collapsed, recovered-but-wrong, stable.

## Key Claims

Do claim:

```text
The 800-step TinyLlama failure was not a hard generation impossibility.
Pythia shows scale-dependent transient duration through 1B.
Final loss alone is insufficient; step-level telemetry is required.
```

Do not claim:

```text
DINO solves factuality.
Longer training alone fixes answer correctness.
Every model will recover if trained long enough.
```

## Missing Before Submission

| missing | why | next |
| --- | --- | --- |
| clean plots | paper needs visual trajectory | plot JSONL telemetry |
| optional Pythia 1.4B/2.8B | checks if Pythia eventually collapses | run only if paper needs scale completeness |
| TinyLlama base-vs-chat | isolates model-family/chat-tuning factor | optional ablation |

## Related Docs

- [Collapse Dynamics Research Plan](./collapse_dynamics_research_plan.md)
- [Literature Deep Dive: Collapse Dynamics](./literature_deep_dive_collapse_dynamics.md)
- [Current Theory](./current_theory_hypothesis_plan.md)
