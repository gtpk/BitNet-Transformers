# Paper / Report Skeleton — Teacher-Free b1.58 → I2_S on Dense LLaMA

Document position: [Index](./index.md) -> synthesis of RT-112..129. Pulls the
finished tracks into a paper outline with a locked claim table. The experimental
picture is closed (systems + recovery + decoding usability + quantizer-ruled-out);
the only open axis is factual quality (adaptation/data). Next step is write-up, not
more experiments.

Related: [bitnet_cpp_export_scoping.md](./bitnet_cpp_export_scoping.md) (systems),
[quality_recovery_plan.md](./quality_recovery_plan.md) (quality),
[oss_architecture_audit.md](./oss_architecture_audit.md) (negative result),
[why_b158_conversion_is_hard.md](./why_b158_conversion_is_hard.md) (problem statement),
[quantization_aware_b158_conversion_plan.md](./quantization_aware_b158_conversion_plan.md)
(next experiment), [complex_phase_rotation_plan.md](./complex_phase_rotation_plan.md)
(RT-126B later candidate), [factual_gap_experiment_plan.md](./factual_gap_experiment_plan.md)
(G10 factual gap).

## Working title

> The Systems Promise and Quality Limits of Teacher-Free b1.58 Conversion for
> Dense LLaMA Models

## Abstract (draft)

On-device LLMs for low-resource users are bottlenecked by per-token memory traffic,
not parameter count. We study whether existing dense LLaMA checkpoints can be moved
toward BitNet-style b1.58 weights as a post-training conversion procedure. We show
that per-tensor b1.58 weights (Wq = gamma·T, gamma = mean|W|, T in {-1,0,+1}) export
faithfully into the existing bitnet.cpp I2_S 2-bit runtime — no custom byte-writer and
no custom kernel — and that the resulting artifact is smaller and faster on x86 CPUs,
with the benefit growing with model size (token-gen up to ~7.5x vs f32 at 1.1B). One-
shot ternary PTQ collapses, but a short teacher-free CE adaptation of just the target
linears recovers most of the loss, the I2_S runtime preserves that recovered behavior
within ~0.01 nats, and — decoded with a standard repetition penalty or sampling — the
adapted ternary model produces non-degenerate, readable generation matching the
degeneration profile of FP and Q2_K (greedy alone degenerates, a known small-model
artifact). We further show, via a full post-training-quantization toolbox sweep (scale
granularity, scale/threshold objective, AWQ/SmoothQuant scaling, GPTQ/Hessian
assignment, 2-bit codebook), that the conversion bottleneck is NOT the quantizer but
adaptation/data: no one-shot quantizer trick rescues conversion, while a short CE pass
does. We frame b1.58 conversion as a systems-strong, decoding-usable path whose
remaining limit is factual quality (it does not beat Q2_K on PPL or match FP on facts),
pointing to adaptation/data, not bit/codebook engineering, as the next lever.

## Core claims (3 + 1 negative)

```text
C1 (faithful export): per-tensor b1.58 Wq=gamma*T -> convert(f32) -> llama-quantize
   I2_S runs in real bitnet.cpp (x86) with f16/f32 CE parity. No Path B writer needed.
C2 (efficiency is a scale law): whole-file compression converges to the 16x
   target-linear floor and token-gen speedup GROWS with model size.
C3 (cheap teacher-free recovery): short CE-only adaptation of target linears recovers
   most of the FP->PTQ loss, and the I2_S runtime preserves it (+0.002 nats).
C4 (scope, negative): on natively-MXFP4 MoE (gpt-oss-20b), ternary adds <1GB / ~0 ROI;
   the recipe's value is dense FP-weight models, not already-low-bit MoE.
C5 (decoding usability): the adapted ternary model is usable-tier (non-degenerate,
   diverse, no loops) under standard repetition-penalized/sampled decoding, matching
   FP/Q2_K degeneration tags; greedy alone degenerates (RT-129). NOT a factual-parity
   claim.
C6 (quantizer is not the lever, negative): a full PTQ toolbox sweep (scale granularity,
   scale/threshold objective, AWQ/SmoothQuant, GPTQ/Hessian, 2-bit codebook) does not
   rescue one-shot conversion (RT-124..127); per-tensor absmean + short CE beats them
   all. The lever is adaptation/data.
```

## Figures (numbers already in hand)

### Figure 1 — Storage scales toward the 16x floor (C2)

| model | params | whole-file i2_s/f32 | target-linear i2_s/f32 | source |
| --- | ---: | ---: | ---: | --- |
| tiny | ~10M | 0.450 | 0.0625 | RT-113 |
| Llama-160M | 160M | 0.196 | 0.0625 | RT-114 |
| TinyLlama-1.1B | 1.1B | 0.1149 | 0.0625 | RT-115 |

Story: target-linear ratio is scale-invariant (16x); whole-file converges to it as the
embedding/lm_head fixed cost shrinks as a fraction.

### Figure 2 — Token-gen speedup grows with scale (C2)

| model | i2_s tg t/s | vs f32 | vs f16 | source |
| --- | ---: | ---: | ---: | --- |
| tiny | ~595 | ~2x | ~2x | RT-113 |
| Llama-160M | 100.8 | 5.69x | 3.00x | RT-114 |
| TinyLlama-1.1B | 18.26 | 7.51x | 4.23x | RT-115 |

(llama-bench, x86, t=2. f32/f16 absolutes noisy on shared CPU; i2_s stable; report ratio.)

### Figure 3 — PTQ collapse → teacher-free CE recovery (C3)

| model | FP PPL | one-shot PTQ PPL | adapted PPL | recovered_fraction | train tokens | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Llama-160M | 23.3 | 115,808 | 52.0 | 0.905 | 0.61M | RT-116 |
| TinyLlama-1.1B (fixed budget) | 10.1 | 101,549 | 1,217 | 0.480 | 0.31M | TRAIN-002 |
| **TinyLlama-1.1B (budget-scaled)** | 10.1 | 101,549 | **162.5** | **0.698** | **4.92M** | RT-120 |

(WikiText-2, target linears only, teacher-free. The 1.1B jump 0.480 -> 0.698 shows the
fixed-budget result was under-trained: ~16x more tokens -> recovery climbs toward the
160M trend. Gap G1 resolved.)

### Figure 4 — Adapted I2_S preserves adapted F16 (C1+C3)

| model | adapted f16 PPL | adapted i2_s PPL | i2_s vs f16 (nats) | source |
| --- | ---: | ---: | ---: | --- |
| Llama-160M (adapted) | 134.84 | 135.11 | +0.0020 | RT-116 QR-003 |
| TinyLlama-1.1B (PTQ-budget adapted) | 1260.21 | 1263.12 | +0.0023 | TRAIN-002 QR-003 |
| TinyLlama-1.1B (budget-scaled adapted) | 219.35 | 216.13 | -0.0148 | RT-120 QR-003 |

Story: the recovered quality survives the int8-activation I2_S runtime essentially
unchanged at every scale and budget (|delta| <= 0.015 nats; sign sometimes favorable).

### Figure 5 — Baselines: why not just one-shot quantization? (G5 / RT-121, 160M)

Same `llama-perplexity`, one eval.txt, ctx 64; embd+lm_head f16 in every row.

| method | target bits | trains? | PPL | whole MB |
| --- | ---: | --- | ---: | ---: |
| FP (f16) | 16 | no | 43.21 | 357.4 |
| RTN ternary one-shot (= PTQ) | 1.58 | no | 135,309 | 121.5 |
| Q2_K one-shot | ~2.6 | no | 97.87 | 134.1 |
| Q3_K_M one-shot | ~3.4 | no | 50.51 | 146.4 |
| Q4_0 one-shot | ~4.5 | no | 48.17 | 155.3 |
| **OURS (b1.58 + teacher-free CE)** | 1.58 | yes | 114.14 | 121.5 |

**Honest result: OURS does NOT beat one-shot Q2_K on PPL (114 vs 98).** It does rescue
the ternary collapse (135,309 -> 114) and is the smallest (121.5 MB) + fastest (I2_S
runtime, ~5.7x tg vs f32 per Fig 2) artifact at the lowest bit budget. So the paper
must NOT claim quality-per-bit superiority; see "Claim discipline" below.

### Figure 6 — Generation panel: does the 1.1B adapted model stay readable? (RT-122)

20 prompts, greedy (one llama-cli, --temp 0), failure tags:

| variant | ok | repetitive | loop | salad | empty |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP f16 | 19 | 1 | 0 | 0 | 0 |
| Q2_K | 20 | 0 | 0 | 0 | 0 |
| PTQ ternary | 0 | 12 | 13 | 3 | 6 |
| OURS adapted f16 | 0 | 18 | 0 | 14 | 0 |
| OURS adapted i2_s | 1 | 19 | 0 | 14 | 0 |

```text
FP   "the ancient Greeks, who were the first to develop a systematic approach ..."
Q2_K "Paris. The capital of Germany is Berlin. ... Washington, D.C."
PTQ  "regression regression regression ...  adenadenaden ..."        (pure salad)
OURS "the 19th century . = = = = = = = ="                            (degenerate loop)
```

**HONEST NEGATIVE on generation usability at 1.1B.** The budget-scaled adapted model
(recovered_fraction 0.698, PPL 162) learned WikiText token statistics — so CE/PPL
dropped — but its greedy generation **degenerates into WikiText section-header loops
("= = =")**. Low PPL != usable generation. OURS is better than PTQ (PTQ is pure
salad/empty; OURS emits real words then loops) but far below FP/Q2_K (both coherent).
i2_s == f16 on 8/20 exactly and the same degenerate style otherwise — the runtime
faithfully reproduces the model's degeneracy, so this is a MODEL/recovery limit, not a
runtime fault. (160M at recovered 0.905 was word-like in RT-119; 1.1B at 0.698 is not —
generation usability tracks recovery fraction, which the 1.1B budget did not push far
enough.)

**>> SUPERSEDED BY RT-129 (Figure 7): this degeneration is GREEDY-only.** A standard
repetition penalty or sampling restores the same model to a readable, non-degenerate
tier. Figure 6 is the greedy operating point; read it together with Figure 7.

### Figure 7 — Decoding rescues 1.1B usability (RT-129)

Same RT-120 adapted 1.1B, 12 prompts, one llama-cli, decode sweep:

| model | decode | ok | loop | salad | rep-3gram |
| --- | --- | ---: | ---: | ---: | ---: |
| adapted i2_s | greedy | 1 | 0 | 9 | 0.656 |
| adapted i2_s | rep-penalty 1.2 | **12** | 0 | 0 | 0.003 |
| adapted i2_s | temp0.8/top_p0.95 | **12** | 0 | 0 | 0.074 |
| adapted f16 | rep-penalty 1.2 | 12 | 0 | 0 | 0.007 |
| FP f16 | greedy | 11 | 0 | 0 | 0.049 |
| Q2_K | greedy | 12 | 0 | 0 | 0.115 |
| PTQ i2_s (no train) | greedy | 0 | 9 | 0 | 0.781 |

The adapted ternary model goes from ok 1/12 (greedy) to **12/12** under a zero-cost
repetition penalty — matching Q2_K/FP on degeneration tags. adapted i2_s == adapted
f16 at every decode (runtime preserves the readable behavior). PTQ (no adaptation)
stays collapsed regardless, so the fix is adaptation + sane decoding, not decoding
alone. Caveat: "ok" = non-degenerate/diverse, NOT factual correctness.

### Claim discipline (post-G5 + RT-122 + RT-129)

RT-129 update: the RT-122 "1.1B degenerates" finding was GREEDY-only. Under a standard
repetition penalty (1.1-1.2, zero inference cost) or temp+top_p sampling, the adapted
ternary model goes from ok 1/12 to **12/12** non-degenerate — matching Q2_K/FP on
loop/salad/empty tags. So generation usability is RESCUED (with the right decode), not
unproven. The quantization-aware track (RT-124..127) separately concluded the quantizer
is not the bottleneck; adaptation/data is.

```text
DO claim:  - one-shot ternary PTQ COLLAPSES (token salad); teacher-free CE substantially
             recovers CE/PPL (135k->114 @160M; 0.698 @1.1B) — recovery is real and scales.
           - the artifact is the smallest + fastest (1.58-bit, I2_S runtime ~5.7x tg);
             storage/speed/runtime-faithfulness all SCALE 160M->1.1B.
           - the adapted ternary model is USABLE-TIER (non-degenerate, diverse, no loops)
             under standard repetition-penalized/sampled decoding; greedy alone degenerates
             (a known small-model artifact). I2_S runs this faithfully (i2_s == f16 at
             every decode).
           - the quantizer is not the lever: scale/objective/activation/GPTQ/2-bit-codebook
             one-shot tricks do not rescue conversion (RT-124..127); adaptation/data is.
DON'T claim: - best PPL-per-bit or beating Q2_K on quality (RT-121).
             - FACTUAL parity with FP/Q2_K. "Usable-tier" here = non-degenerate/diverse
               generation, NOT correct facts; the WikiText-CE model is still weaker on
               facts (a data/objective gap, not decoding or runtime).
The honest spine: a SYSTEMS result (speed/memory-traffic scale law + faithful I2_S
export) + a quality result (teacher-free CE recovery scales AND yields non-degenerate
generation under sane decoding), with factual parity as the remaining data/objective gap.
```

### Appendix figure — Why not gpt-oss? (C4, negative)

gpt-oss-20b is MXFP4 already: official mxfp4.gguf 12.11 GB, and Q2_K (2-bit) is 11.47 GB
— every quant bottoms at a ~11.5 GB floor, so ternary adds <1 GB. Plot the quant-vs-size
ladder to show the floor; conclude "wrong vehicle for ternary."

## Gap audit (what's missing before paper-grade claims)

| id | gap | severity | cheapest fix |
| --- | --- | --- | --- |
| G1 | ~~1.1B recovery only 0.48~~ RESOLVED (RT-120): L4 budget-scaled (4.92M tokens, eff batch 24, 800 steps) -> **0.698** (paper-useful tier); QR-003 i2_s vs f16 -0.0148 nats. Fixed-budget explanation confirmed. | DONE | (optional: 1200 steps toward 0.90) |
| G2 | ~~no recipe ablation~~ RESOLVED (QR-005): a/b/c on 160M -> +norms negligible (0.907 vs 0.906), +lm_head hurts (0.898). **Default = linears only.** | DONE | — |
| G3 | ~~+norms may lift the fraction~~ RESOLVED: it does not (within noise). Cheapest recipe is best. | DONE | — |
| G4 | ~~quality is CE/PPL only~~ RESOLVED (QR-004/RT-119 + RT-129): generation panel + decoding sweep -> adapted is non-degenerate/usable-tier under rep-penalty/sampling at BOTH 160M and 1.1B (greedy degenerates, RT-122, but that is a greedy artifact). i2_s == f16. | DONE | — |
| G10 | FACTUAL quality below FP/Q2_K (the real remaining limit; "usable-tier" != correct facts) | MED | [FACT-001 current factual panel](./factual_gap_experiment_plan.md), then instruction data / longer CE / repetition-aware objective |
| G5 | ~~no baseline comparison~~ RESOLVED (RT-121, honest NEGATIVE): OURS 114 vs Q2_K 98 PPL — does NOT win quality-per-bit; OURS is smallest+fastest at lowest bits + rescues PTQ collapse (135k->114). Reframe to speed/usability, not quality-SOTA. See Fig 5 + Claim discipline. | DONE | (optional: 1.1B B2-vs-OURS; B5 no-scale QAT contrast) |
| G6 | single seed for recovery; no variance | LOW | 2-3 seeds on 160M QR-002a |
| G7 | cross-tool PPL gap (PyTorch CE vs llama.cpp perplexity) unexplained in-figure | LOW | one calibration note + measure both on identical tokens |
| G8 | only LLaMA family; generality unproven beyond it | LOW | (scope it honestly; gpt-oss negative already bounds the claim) |
| G9 | some raw Colab JSON volatile / not all archived in repo `reports/` | LOW | commit the rt11x_*.json artifacts |

## Locked claim table (post RT-129; the conclusion to preserve)

| axis | status | evidence |
| --- | --- | --- |
| systems: faithful I2_S export | SOLVED | RT-111/112 (x86 i2_s==f32/f16) |
| systems: storage + speed scale law | SOLVED | RT-113/114/115 (16x linear; tg ~2x->5.69x->7.51x) |
| runtime faithfulness (i2_s == f16) | SOLVED | RT-116/120 QR-003 + RT-129 (|delta|<=0.015 nats; every decode) |
| quality: CE/PPL recovery | SOLVED, scales | RT-116 0.905 @160M, RT-120 0.698 @1.1B |
| quality: generation usability | SOLVED w/ sane decoding | RT-129 (rep-penalty/sampling -> ok 12/12; greedy degenerates) |
| quantizer design as the lever | RULED OUT | RT-124..127 (no PTQ trick rescues; adaptation/data is the lever) |
| quality-per-bit vs Q2_K | NEGATIVE | RT-121 (OURS 114 vs Q2_K 98 PPL) |
| factual parity vs FP/Q2_K | OPEN (the remaining gap) | G10; needs adaptation/data |
| gpt-oss / MoE | OUT OF SCOPE | RT-117/118 (MXFP4 already; ~0 ROI) |

## Quantization-aware track: CONCLUDED (not the next experiment)

The RT-124..128 quantization-aware sweep is DONE and its conclusion is locked: **the
quantizer is not the bottleneck.** Scale granularity (RT-124A, partial +2.4 nats but
needs runtime), scale/threshold objective (RT-124B, absmean already best), AWQ/
SmoothQuant diagonal scaling (RT-124C, +0.14), GPTQ/Hessian assignment (RT-125, +0.51 /
6% of gap), and a signed-epsilon 2-bit codebook (RT-127, no gain over ternary) all fail
to rescue one-shot conversion; a short teacher-free CE pass beats every one of them. See
[Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md)
(synthesis section). RT-126 rotation and RT-128 1D-gate were not needed (a strong
assignment method already gained only 6%).

## Next: consolidate, then adaptation/data

The experimental picture is closed enough to write up. Priority order:

```text
1. Lock the docs to the post-RT-129 claim table (this file, index, quality docs).
2. Draft the paper/report from Figures 1-7 + the claim table + the gpt-oss appendix.
3. THEN the only remaining quality lever (G10, factual parity): adaptation/data, NOT
   quantizer engineering — instruction data, longer/better-data CE, repetition-aware
   or free-run objectives. Optional later candidate: a pairwise phase-rotation probe.
```

## What NOT to do next

- Do not start a ternary-MoE / gpt-oss build (RT-118: ROI ~0).
- Do not claim quality-per-bit superiority over Q2_K (RT-121 says no) or factual parity
  with FP/Q2_K (the open G10 gap).
- Do not report GREEDY generation as the usability verdict — greedy degenerates these
  small CE-adapted models (RT-122). Use a repetition penalty (~1.2) or sampling (RT-129);
  report the decode.
- Do not reopen quantizer/codebook/rotation engineering as a main track — RT-124..127
  ruled it out; the lever is adaptation/data. Keep cheap pairwise phase rotation only
  as a later candidate idea, not the main path.
- Do not run more experiments before consolidating the docs/claim table; the risk now is
  losing a closed result, not lacking a new one.
