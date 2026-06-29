# Minimal Content-KL Is The Stable TinyLlama v0, But Collapse Is Model-Specific

A research note wrapping up the base-anchored b1.58 / I2_S factual-recovery line. This document has
been updated after the Pythia ladder reached 1B.

Updated headline:

> **TinyLlama-1.1B has a same-topology I2_S collapse problem, but this is not a generic 1B scale
> wall.** Minimal content-KL remains the stable TinyLlama v0 recipe, while Pythia-160M/410M/1B show
> that the same DINO/content-KL objective can recover after a degenerate transient in another model
> family.

Document position: [Index](./index.md). Companion: [v0 recipe + closed branches](./i2s_v0_recipe_and_closed_branches.md),
[DINO plan](./dino_i2s_self_distillation_plan.md). Result files under `reports/` are cited inline.

---

## 1. Setup

- **Model / conversion.** TinyLlama-1.1B-Chat, converted to per-tensor **b1.58 / I2_S** (ternary
  weights, `Wq = gamma * T`, `T in {-1,0,+1}`, `gamma = mean|W|`; PerTensorBitLinear STE). All
  target linears ternary; lm_head and embeddings frozen. We call this **same-topology** adaptation:
  the architecture is unchanged, only the target linears are quantised and adapted.
- **Base-anchored recipe.** The b1.58 student is anchored to the *same* original FP model as a
  frozen self-teacher, not a new large teacher. The goal is a usable, cheaply-runnable b1.58 model
  that preserves factual/instruction behaviour.
- **Systems context (prior, solved).** Small + fast + faithful is done: the I2_S runtime matches
  f16 in nats (`i2_s ~= f16`), and fluency/anti-collapse decoding are handled. The *only* open gap
  was factual/answer-behaviour forgetting during b1.58 adaptation.
- **Evaluation.** A held-out factual panel (`data/factual_panel_v1.jsonl`, 27 prompts across
  simple_fact / entity_attr / reasoning / instruction; FP reference 0.81, Q2_K 0.74), never trained
  on. Public-QA transfer is probed with a de-leaked PopQA tight heldout. We report exact-match
  `fact_rate`, generation tags (ok/salad/empty/loop), WikiText CE `recovered_fraction`, and (for the
  diagnostic) gold-token logprob/rank vs the FP teacher.

---

## 2. The recovery ladder: content-KL is the lever, and it saturates at 0.185

| step | change | factual panel | note |
| --- | --- | ---: | --- |
| FACT-002 | data swap (instruction/mixed) | 0.00-0.07 | data is not the lever |
| FACT-003A | answer-only CE | 0.15 | the objective is the lever; fixes empty-collapse |
| FACT-003B | naive base-KL (lambda 1.0) | 0.00 | copies chat teacher's early-EOS -> empty |
| **FACT-003C** | **content-KL (drop EOS/special), lambda 0.2** | **0.185** | **best; recovered_fraction 0.845; ok 27/27; i2_s==f16** |

The content-KL strength shows an **inverted-U** (lambda 0.1 -> 0.037, **0.2 -> 0.185**, 0.5 -> 0.037):
too weak leaves the CE plateau, too strong over-anchors to generic base fluency and washes specific
facts out. Capacity probes (post-hoc FP restore of late layers; tiny I2_S+LoRA sidecars; entropy/
sensitivity-guided growth) and data-free weight-only geometry (scaling, Hadamard rotation) were all
negative. **Same-topology 1.1B adaptation + objective/strength tuning caps the factual panel at
~0.185.**

---

## 3. The objective-augmentation wall (the central negative)

To push past 0.185 we added, on top of content-KL, three *different kinds* of auxiliary objective.
All three collapsed at 1.1B:

| auxiliary objective on top of content-KL | 160M | 1.1B | failure mode at 1.1B |
| --- | :---: | :---: | --- |
| hard factual replay (FACT-003D, mu-CE on a curated set) | n/a | fail | memorises the train set, eval transfer drops (mu 1.0 -> 0.111, mu 0.25 -> 0.037) |
| representative-data PopQA blend (FACT-003H) | recovers | fail | v1 recovered 0.474 / fact 0.0 / loops; v2 recovered 0.706 / eval 0.037 **salad** / popqa_tight 0.013 |
| no-label self-distillation, DINO-logit (DINO-002) | **positive** | fail | recovered 0.620 but eval 0.111 **salad 26/27**; gold ranks catastrophic (cap_france rank 431 vs teacher 3) |
| DINO-logit, fully stabilised (DINO-003) | n/a | fail | centering + warmup + weight 0.1 + lr 1e-4 -> **COLLAPSE at step 200, panel salad 0.90** |

Three independent objectives, three collapses, in the same direction. The DINO-003 attempt is the
clincher: with DINO centering, a 150-step weight warmup, half the weight (0.1) and half the LR
(1e-4) and an early-collapse detector, the 1.1B student was still 90% degenerate by step 200. **At
1.1B, content-KL alone is fluent (0.185); content-KL + anything collapses.**

---

## 4. DINO decomposition: the objective is not wrong; the 1.1B model cannot absorb it

DINO-logit is the cleanest evidence because at **160M it genuinely works**, which rules out "the
objective is simply bad":

- **160M token-level diagnostic (DINO-DIAG-001):** vs the content-KL baseline, dino_logit raises
  the gold-answer probability broadly -- **mean delta log P(gold) +0.372, gold rank improved on 78%
  of prompts** (e.g. cap_italy rank 48->7, cap_france 60->11). Entropy ~flat (not mere sharpening),
  no train-set memorisation. The small exact-match gain (~+0.037-0.074) at 160M is a *decoding
  ceiling*: the model raises gold to rank ~7 but cannot pull it to rank 1.
- **Category structure:** the gain is real but bounded -- simple_fact +0.574 >> reasoning +0.112 >>
  entity_attr ~ -0.04 ~ instruction. Broad unlabeled text teaches common-fact content but not rare
  entity-attribute links.
- **At 1.1B the same objective inverts:** instead of raising gold ranks it drives the model to
  word-salad (gold ranks 400-4000 vs the FP teacher's 1-7). Centering/warmup/low-weight/low-LR do
  not prevent it.

So DINO is mechanistically valid; the failure is that the 1.1B same-topology I2_S student cannot be
adapted under the extra training pressure without losing coherent generation. (Note: 160M ran with
plain AdamW/fp32; 1.1B with bf16 + 8-bit AdamW + grad-checkpointing for memory. We treat the 1.1B
fragility as scale/regime-specific; isolating optimiser vs objective further was judged not worth
the compute given the consistent three-objective pattern.)

---

## 5. Central observation, revised after Pythia

```
For TinyLlama-1.1B, same-topology I2_S adaptation is stable only under a minimal
content-anchoring objective within the 800-step budget. Adding hard replay, PopQA
blend, or DINO-logit caused generation collapse rather than improvement.
```

However, Pythia changes the broader interpretation:

```text
Pythia-160M: stable
Pythia-410M: transient -> recovery
Pythia-1B: transient -> recovery
TinyLlama-1.1B: transient unresolved within 800 steps
```

So the stronger, current observation is:

```text
collapse is not a generic 1B scale wall.
It is a model-family / chat-tuning / schedule-specific adaptation dynamics issue.
```

The three same-direction TinyLlama failures (replay / blend / DINO, the last even stabilised) remain
important, but they should no longer be generalized to all 1B all-I2_S models. They flip the research
question:

- **Old question:** *will adding an objective raise TinyLlama's ceiling within 800 steps?* -> answered:
  not with the tested recipes.
- **New question:** *is TinyLlama a hard collapse or only a longer transient?* -> TinyLlama longer-budget
  gate.
- **Second question:** *which model families consolidate under I2_S adaptation?* -> Pythia/Qwen/Gemma
  ladder.

---

## 6. v0 deliverable (locked)

```text
base      : TinyLlama-1.1B-Chat (frozen FP self-teacher)
quant     : per-tensor b1.58, all target linears ternary; lm_head/embeds frozen
objective : answer-only CE + 0.2 * content-KL(base||student), EOS/special dropped from KL
optimiser : the minimal recipe ONLY -- no replay, no blend, no self-distillation
result    : factual panel 0.185, recovered_fraction 0.845, tags ok 27/27, i2_s == f16
```

This is a usable, cheaply-runnable b1.58 model that preserves the base's factual behaviour up to the
same-topology ceiling. It is **not** a "facts-you-can-trust assistant"; 0.185 is the documented
ceiling for this base under same-topology conversion.

---

## 7. Claim discipline

- **Claim:** for same-topology 1.1B I2_S conversion, minimal content-KL is the stable optimum, and
  auxiliary objectives (hard replay, representative-data blend, no-label self-distillation) cause
  catastrophic generation collapse at 1.1B while being benign/positive at 160M.
- **Claim:** no-label self-distillation (DINO-logit) measurably raises the teacher's factual content
  mass in the student at 160M (gold logprob/rank), i.e. it "forgets less" -- a distribution-level
  effect, not memorisation.
- **Do NOT claim:** that DINO (or any objective) raises the 1.1B factual ceiling -- it does not.
- **Do NOT claim:** new factual knowledge was created -- the objectives are retention/anchoring.
- **Data caveat:** PopQA is research/direction only (license unclear); results there are a
  blend-mechanism signal, not a product-data or real-factual-ability claim.

---

## 8. Future work: TinyLlama longer budget, then model-family ladder

The immediate next project is not another objective. It is a schedule/dynamics test on the actual
failing target:

```text
TinyLlama-1.1B longer-budget run:
  1600 steps first,
  extend to 2400 only if degen_gap/gold_rank trajectory suggests recovery.
```

If TinyLlama recovers:

```text
the original I2_S product path reopens as a schedule/curriculum problem.
```

If TinyLlama remains collapsed:

```text
TinyLlama is a hard model-specific collapse under this same-topology setup.
Then move to model-family/product ladders:
  Qwen/Gemma audit,
  Qwen 1.5B/3B/7B,
  optionally Pythia 1.4B/2.8B for academic completeness.
```

The one-line thesis carried forward:

> The bottleneck is not simply objective choice or 1B scale. It is whether a given base model can
> consolidate through the degenerate transient induced by same-topology I2_S adaptation.
