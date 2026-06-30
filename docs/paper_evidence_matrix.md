# Paper Evidence Matrix

Document position: [Index](./index.md) -> publication evidence matrix for the
multi-paper split.

Related:

- [Paper Series Plan](./paper_series_plan.md)
- [Paper 1: I2_S Systems](./paper_1_i2s_systems.md)
- [Paper 2: Conversion Limits](./paper_2_conversion_limits.md)
- [Paper 3: Content-KL Factual Recovery](./paper_3_content_kl_factual_recovery.md)
- [Paper 4: Hybrid Capacity Candidate](./paper_4_hybrid_capacity_candidate.md)
- [Fair Comparison Framework](./fair_comparison_framework.md)

## Purpose

This document is the single table-first view of the project. It separates:

```text
known data       results already measured and safe to cite
blank cells      data still unknown
interpretation   what each number means and what it does not mean
```

The rule is simple: a paper draft may make only the claims whose rows are already
filled here. Empty cells are not weaknesses; they are the next experiment list.

## Paper 1: I2_S Systems Evidence

Claim: if b1.58-compatible weights exist, they can be exported to bitnet.cpp I2_S
faithfully on x86, and the storage/speed gains improve with scale.

| run | model | storage result | speed result | parity result | status |
| --- | --- | --- | --- | --- | --- |
| RT-111 | official BitNet GGUF | n/a | n/a | f32 PPL 1.8547 vs i2_s 1.8548 | solved |
| RT-112 | tiny Path A' artifact | i2_s 15.94MB vs f32 36.26MB whole | not main metric | Wq path: i2_s 305.02 ~= f16 306.48 ~= f32 306.42 | solved |
| RT-113 | tiny ~10M | whole i2_s/f32 0.450; target 0.0626 | tg ~2x vs f32/f16 | faithful enough for systems check | solved |
| RT-114 | Llama-160M | whole 0.196; target 0.0625 | tg 5.69x vs f32, 3.00x vs f16 | i2_s vs f16 +0.0418 nats | solved |
| RT-115 | TinyLlama-1.1B | whole 0.1149; target 0.0625 | tg 7.51x vs f32, 4.23x vs f16 | i2_s vs f16 -0.0071 nats | solved |
| Mac M5 audit | official + ours | n/a | ternary runtime broken locally | x86 passes, M5 build/toolchain fails | appendix caveat |

Known blank cells:

| blank | why it matters | minimum fill |
| --- | --- | --- |
| quiet-host latency error bars | Colab 2-core timing is noisy | rerun RT-114/115 on a quieter x86/Linux host |
| ARM production runtime | Mac M5 build was broken, not proof ARM is impossible | known-good ARM or upstream fix |
| real application latency | llama-bench is synthetic | one end-to-end prompt workload |

Do not use Paper 1 to claim quality parity. It proves the runtime substrate.

## Paper 2: Conversion Limits Evidence

Claim: same-shape one-shot b1.58 conversion is not ordinary quantization; quantizer
tweaks are too small compared with adaptation.

| lever | best observed effect | remaining gap | interpretation |
| --- | ---: | ---: | --- |
| one-shot ternary RTN | PPL 115,808 on 160M | FP PPL 23.3 | collapse |
| row scale | +1.84 nats vs per-tensor | still PPL 18,422 | real but insufficient |
| block/group scale | +2.36 nats vs per-tensor | still PPL 10,935 | strongest one-shot scale lever, needs runtime support |
| scale/threshold objective | absmean wins | no rescue | BitNet absmean is already strong |
| AWQ/SmoothQuant-style diagonal | +0.14 nats | tiny | activation outliers are not the main blocker |
| GPTQ/Hessian assignment | +0.51 nats, 6% of gap | tiny relative to 8.5 nat gap | output-aware assignment helps but cannot rescue pure one-shot |
| signed-epsilon 2-bit | worse than ternary | no gain | removing zero is not the fix |
| Q2_K baseline | Q2_K PPL 97.9 vs ours 114.1 | Q2_K still better on PPL | ours is smaller/faster, not best PPL-per-bit |

Known blank cells:

| blank | why it matters | minimum fill |
| --- | --- | --- |
| one larger-model Q2_K vs ours | RT-121 is 160M-only | repeat baseline panel at 1.1B or a larger dense LLaMA |
| seed variance | many conversion results are single-seed | 2-3 seeds on the strongest 160M adapted recipe |
| no-scale QAT contrast | separates scale from generic QAT | B5 from [G5 Baseline Plan](./g5_baseline_plan.md) |

Do not conclude b1.58 is impossible. Conclude post-hoc same-shape one-shot b1.58
is not enough.

## Paper 3: Content-KL Factual Recovery Evidence

Claim: after runtime and quantizer are exonerated, factual recovery is controlled by
the adaptation objective. Content-KL is the first objective that moves facts without
copying EOS/empty behavior.

| arm | fact_i2s | fact_f16 | CE recovery | behavior | status |
| --- | ---: | ---: | ---: | --- | --- |
| FP f16 reference | 0.815 | n/a | n/a | ok | sanity reference |
| Q2_K reference | 0.741 | n/a | n/a | ok | sanity reference |
| PTQ i2_s | 0.000 | n/a | n/a | salad/collapse | collapse baseline |
| WikiText adapted | 0.037 | 0.000 | high but not fact-aligned | ok/fluent | facts not recovered |
| FACT-002 instruction | 0.000 | 0.000 | 0.403 | empty collapse | data alone fails |
| FACT-002 mixed | 0.074 | 0.074 | 0.814 | ok/fluent | fluency without facts |
| FACT-003A answer mask, instr | 0.037 | 0.037 | 0.555 | ok | mask helps format collapse |
| FACT-003A answer mask, mixed | 0.148-0.150 | 0.185-0.190 | 0.822 | ok | first pass signal |
| FACT-003B raw KL 1.0 | 0.000 | 0.000 | 0.474 | empty collapse | copied EOS/stop |
| FACT-003C content-KL 0.1 | 0.037 | 0.037 | 0.484 | salad | too weak |
| FACT-003C content-KL 0.2 | 0.185 | 0.185 | 0.845 | ok 27/27 | current best |
| FACT-003C content-KL 0.5 | TBD | TBD | TBD | TBD | pending |

Known blank cells:

| blank | why it matters | minimum fill |
| --- | --- | --- |
| content-KL lambda=0.5 | decides whether 0.2 is sweet spot or part of rising curve | score FACT panel + CE recovery |
| lambda around 0.3/0.4 | needed only if 0.5 improves or partially collapses | two more sweep points |
| seed check for best lambda | verifies objective stability | 2-3 seeds on best content-KL recipe |
| larger factual benchmark | current panel is 27 prompts, not a benchmark | small held-out factual QA subset |
| factual score after hybrid | tests whether capacity, not objective, is now limiting | HYBRID-001A if content-KL plateaus |

Do not claim factual parity with FP/Q2_K. The current claim is narrower: content-KL
is the first working factual lever and fixes the raw-KL EOS failure.

## Paper 5: Collapse Dynamics Evidence

Claim candidate: generation collapse during low-bit adaptation is a dynamic
transient/consolidation phenomenon. A run can look failed at step 800 but recover by
step 1600.

| run | transient / recovery | final generation | factual exact | interpretation |
| --- | --- | --- | ---: | --- |
| Pythia-160M | no transient | stable | n/a | small rung stable |
| Pythia-410M | transient step ~50-250, recovery ~275 | stable | n/a | slower consolidation |
| Pythia-1B | transient step ~0-250, recovery ~225-300 | stable | n/a | no generic 1B wall |
| TL1B-1600 | unresolved at 800, recovers by ~850-1600 | ok 27/27, CE 4.08 | 0.111 | 800-step DINO collapse was premature |

Known blank cells:

| blank | why it matters | minimum fill |
| --- | --- | --- |
| clean collapse plots | needed for a readable paper | plot degen_gap/gold_rank/CE/top1/hidden_var vs step |
| Pythia 1.4B/2.8B | tests whether Pythia eventually collapses | optional ladder completion |
| TinyLlama chat-vs-base | tests whether chat-tune / data style causes slower consolidation | base-model ablation |

Do not claim DINO solves factuality. The new claim is dynamics: collapse at an
intermediate step can be a transient.

## Paper 6: Readout / Answer-Format Evidence

Claim candidate: after low-bit adaptation, factual tokens can be reachable in the
distribution but fail to be emitted as concise answers.

| run | gold-rank signal | exact answer | behavior | interpretation |
| --- | ---: | ---: | --- | --- |
| TL1B-1600 | final gold_rank 375, improved from ~2006 near recovery onset | 0.111 | fluent but Q/A format drift | reachable but not emitted |
| ANS-001 160M beta=0 | gold_rank_mean 1, first_token_hit 0.778 | 0.000 | ok26/empty1 | 160M does not reproduce 1.1B readout bottleneck |
| ANS-001 160M beta=4 | gold_rank_mean 1, first_token_hit 0.778 | 0.074 | ok27 | answer-token weighting is safe and directionally positive |
| ANS-001 1.1B | running | TBD | TBD | decisive readout test |

Known blank cells:

| blank | why it matters | minimum fill |
| --- | --- | --- |
| ANS-001 1.1B final | decides whether answer-token weighting beats content-KL 0.185 | FACT/gold_rank/tags/CE table |
| ANS-002 short-answer format | if ANS-001 is flat, format data is next | one 1.1B or 160M-gated run |
| first-token metrics standardization | needed to compare readout experiments | report first-token hit, gold-rank, exact answer together |

## Paper 7: PT2-I2S Model Competition Evidence

Claim candidate: PT2-LLM is both a competitor and a donor. The final question is
which deployed model is better under a constraint, not whether a method is pure.

| row | status | interpretation |
| --- | --- | --- |
| PT2-I2S-001 ITF by weight-MSE | done; MSE improves but behavior/gold-rank worsens | reconstruction-only fitting is not enough |
| PT2-I2S-002 activation-aware grid | pending | next meaningful PT2 smoke |
| PT2 exact vs projected-I2_S vs ours adapt | pending | required final comparison |

Known blank cells:

| blank | why it matters | minimum fill |
| --- | --- | --- |
| PT2 exact quality row | tells how strong asymmetric ternary is | PPL/FACT/generation on same model |
| PT2 projected-I2_S row | tells whether gains survive pure I2_S | compare against old I2_S PTQ |
| PT2-init + adaptation row | tells whether PT2 helps us go farther | same adaptation schedule scorecard |

## Paper 4: Hybrid Capacity Candidate Evidence

Claim candidate: if all-I2_S plus content-KL plateaus below useful factual quality,
selective precision/capacity pockets may be the product path.

No positive evidence yet. This paper is intentionally empty until HYBRID-001.

| arm | storage cost | expected runtime class | fact score | PPL/CE | decision |
| --- | ---: | --- | ---: | ---: | --- |
| all-I2_S + best content-KL | baseline | fastest | TBD after lambda sweep | TBD | baseline |
| last 1 block F16 | TBD | mostly fast | TBD | TBD | HYBRID-001A |
| last 2 blocks F16 | TBD | mostly fast | TBD | TBD | HYBRID-001A |
| last 4 blocks F16 | TBD | slower | TBD | TBD | HYBRID-001A |
| last 2 attention F16 | TBD | mostly fast | TBD | TBD | HYBRID-001A |
| last 2 MLP F16 | TBD | mostly fast | TBD | TBD | HYBRID-001A |
| helpful region Q2/Q3 | TBD | quantized fallback | TBD | TBD | only if F16 restore helps |
| multi-strip ternary R=2/R=4 | TBD | custom/experimental | TBD | TBD | only if capacity signal exists |
| low-rank residual | TBD | extra matmul | TBD | TBD | only if storage budget allows |

HYBRID-001 should not start before the content-KL sweep closes unless the user
explicitly decides the product path needs capacity now.

## Current Publication Readiness

| paper | evidence status | current draft level | next blocker |
| --- | --- | --- | --- |
| Paper 1 systems | strong | draftable now | optional timing error bars |
| Paper 2 limits | strong | draftable now | optional seed/larger-model baseline |
| Paper 3 content-KL | promising, active | draftable as workshop/tech report | seed / larger factual eval |
| Paper 4 hybrid | hypothesis only | not draftable as result paper | HYBRID positive signal |
| Paper 5 collapse dynamics | strong new topic | draftable as dynamics report | plots + optional larger Pythia rung |
| Paper 6 readout/answer format | active | not draftable until ANS-001 final | ANS-001/002 |
| Paper 7 PT2-I2S | active competitor/donor track | not draftable until PT2-I2S-002 | PT2 comparison scorecard |

## Interpretation Guardrails

Use these sentences to prevent overclaiming:

```text
I2_S export is solved for faithful runtime of already-ternary-compatible weights.
It does not by itself solve quality.

One-shot b1.58 conversion is not ordinary quantization. Standard PTQ tools help
only marginally at this bit budget.

Content-KL improves factual retention, but the current model is still far below
FP/Q2_K factual parity.

Hybrid capacity is a candidate response to a plateau, not yet an established result.
```
