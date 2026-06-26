# Factual Gap Experiment Plan (G10 / FACT-001..004)

Document position: [Index](./index.md) -> after [Paper Draft](./paper_draft.md).

Related:

- [Paper Draft](./paper_draft.md)
- [Quality Recovery Plan](./quality_recovery_plan.md)
- [G5 Baseline Comparison Plan](./g5_baseline_plan.md)
- [Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md)

## Purpose

RT-129 rescued the **generation usability** claim:

```text
adapted b1.58/I2_S is non-degenerate and readable under repetition-penalized or
sampled decoding, and I2_S preserves adapted F16 behavior.
```

But RT-129 did **not** prove factual quality:

```text
"ok" = non-degenerate / diverse / not salad.
"ok" != correct, useful, or FP/Q2_K-level factual answers.
```

This document defines the next gap:

```text
G10: factual parity with FP/Q2_K is open.
```

The goal is to measure that gap first, then test whether better adaptation data and
objectives can reduce it without losing the b1.58/I2_S systems benefits.

## Locked Context

Already solved:

```text
faithful I2_S export
storage/speed scale law
I2_S runtime preservation
CE/PPL recovery
non-degenerate generation under sane decoding
quantizer-not-the-lever
```

Still open:

```text
Does the adapted I2_S model answer facts and simple instructions close to FP/Q2_K?
If not, which adaptation/data recipe closes the gap most cheaply?
```

Do **not** restart quantizer/codebook work for this gap. RT-124..127 ruled that out.

## Main Hypotheses

### H1: The factual gap is mostly data mismatch

Current recovery used WikiText-style CE. It restores token statistics and reduces
degeneration, but it does not teach assistant-style factual answering.

Expected signal:

```text
instruction/factual adaptation improves factual score more than longer WikiText CE
at similar token budget.
```

### H2: The factual gap is mostly under-adaptation

The 1.1B budget-scaled run recovered 0.698 of the PTQ->FP CE gap, not 0.9+. More tokens
or better optimizer settings may improve facts too.

Expected signal:

```text
longer CE improves factual score in line with CE/PPL recovery.
```

### H3: The factual gap is decoding/objective instability

RT-129 showed greedy was a bad operating point. Factual prompts may still be sensitive
to repetition or sampling settings.

Expected signal:

```text
same adapted checkpoint changes factual score materially across decode configs.
```

### H4: The factual gap is capacity/precision

Even with better data, b1.58 may not preserve enough fine-grained knowledge.

Expected signal:

```text
FP/Q2_K remain far ahead after instruction/longer-CE adaptation, while I2_S faithfully
preserves adapted F16. The problem is model quality, not runtime.
```

## Evaluation Philosophy

This plan is intentionally split into:

```text
FACT-001: measure current factual gap
FACT-002..004: improve adaptation/data only after the gap is measured
```

Do not train first. A new training run without a fixed factual panel can only produce
another ambiguous PPL number.

## Models And Variants

Use the TinyLlama-1.1B budget-scaled artifacts first because RT-129 already established
their decoding behavior.

Required variants:

| Variant | Role |
| --- | --- |
| FP f16 | upper reference |
| Q2_K | mature low-bit reference |
| PTQ I2_S | collapse control |
| adapted F16 `Wq` | model-quality reference before runtime |
| adapted I2_S | deployment artifact |

Optional variants:

| Variant | Role |
| --- | --- |
| 160M adapted | cheap smoke and seed variance |
| longer-CE adapted | under-adaptation check |
| instruction-adapted | data/objective check |
| repetition-aware adapted | objective check |

## Decode Policy

RT-129 changed the default decode.

Primary decode:

```text
repetition_penalty = 1.2
temperature = 0 or low temperature if supported
```

Secondary decode:

```text
temperature = 0.8
top_p = 0.95
repetition_penalty = 1.1 or 1.2
```

Greedy is still reported as a diagnostic, but not used as the usability verdict.

## FACT-001: Current Factual Gap Baseline

Question:

```text
How far is current adapted I2_S from FP/Q2_K on factual prompts under sane decoding?
```

Prompt panel:

Use a small, fixed, license-safe panel first. The first version should be manually
inspectable before adding automated benchmarks.

Categories:

| Category | Examples |
| --- | --- |
| simple factual QA | capitals, dates, authors, physical facts |
| entity attributes | "Who wrote ...?", "What is the capital of ...?" |
| short reasoning | one-hop arithmetic, comparison, temporal order |
| instruction following | "Answer in one sentence", "List three..." |
| abstention / uncertainty | answer unknown or trick prompts without fabricating |
| Korean/English mixed | if the target deployment needs Korean behavior |

Example prompt style:

```text
Q: What is the capital of France?
A:

Q: Who wrote Pride and Prejudice?
A:

Q: If a train leaves at 3pm and the trip takes 2 hours, when does it arrive?
A:

Q: Answer in Korean: What is water made of?
A:
```

Metrics:

```text
exact/contains score where applicable
manual tag: correct / partially_correct / wrong / off_topic / loop / empty
format-following score
repeated-3gram rate
unique-token ratio
adapted_i2s_vs_f16 same-tier flag
```

Minimum output:

```text
reports/fact001_current_gap.json
reports/fact001_current_gap.md
```

Pass/report rule:

```text
No pass required. This establishes the factual gap.
```

Decision:

```text
If adapted I2_S is close to Q2_K on simple factual prompts:
  paper can say "non-degenerate and basic-factual usable" with caveats.

If adapted I2_S is far below Q2_K but i2_s ~= f16:
  runtime is exonerated; proceed to FACT-002/003 data adaptation.

If adapted I2_S diverges from adapted F16:
  unexpected runtime/decode issue; return to I2_S parity diagnostics.
```

## FACT-002: Longer CE vs Better Data

Question:

```text
Is factual quality limited by adaptation budget or by WikiText-only data?
```

Arms:

| Arm | Data | Budget | Purpose |
| --- | --- | --- | --- |
| A baseline | existing RT-120 adapted | existing | reference |
| B longer WikiText CE | same WikiText-like data | +1x or +2x tokens | under-adaptation test |
| C factual/instruction CE | small instruction/factual corpus | matched tokens | data mismatch test |
| D mixed CE | WikiText + instruction/factual | matched tokens | preserve LM stats + facts |

Training recipe:

```text
target linears only
per-tensor b1.58 STE
same optimizer class as RT-120 unless deliberately changed
same decode policy as FACT-001
export adapted F16 Wq and adapted I2_S for winners only
```

Metrics:

```text
WikiText CE/PPL
FACT-001 factual score
loop/salad/empty tags
I2_S-vs-F16 CE delta
I2_S-vs-F16 generation same-tier
```

Decision:

```text
If longer WikiText improves facts:
  under-adaptation is a major cause; scale budget first.

If instruction/factual data improves facts more efficiently:
  data mismatch is the main cause; promote instruction adaptation.

If both improve but mixed is best:
  use mixed curriculum.

If none improves:
  move to objective or capacity diagnostics.
```

## FACT-003: Objective Branch (data was not the lever — FACT-002 closed to S3)

FACT-002 (RT-131) showed that swapping adaptation *data* (instruction/mixed) recovers
fluency but not facts (mixed: 0.81 CE recovered, fact 0.07). So the lever is the training
**objective**, not the data. FACT-003 is a cheap-to-expensive ladder; stop at the first
arm that lifts `fact_rate` materially without regressing I2_S-vs-F16 parity.

Order (cheapest/safest first):

```text
FACT-003A  answer-only loss mask     -- cheapest, no new loss term
FACT-003B  base-KL replay            -- keep base answer distribution on non-eval prompts
FACT-003C  protected factual replay  -- small fact set disjoint from FACT-001 + leakage check
```

### FACT-003A: answer-only loss mask (do this first)

Hypothesis: instruction-only collapsed to empty answers and mixed hallucinated because CE
on the full `Q: ..\nA: ..` stream overfits the *prompt formatting* rather than the answer
content. Computing CE on response tokens only should reduce formatting overfit and let more
of the adapted capacity carry answer behaviour.

Implemented in `scripts/rt116_quality_recovery.py` via `--answer-loss-only`:

```text
- instruction data is tokenized per example as prompt 'Q: ..\nA:' + answer ' <response>' + '\n\n'
- an aligned answer-mask is True only on response tokens
- training sets labels = -100 on prompt + separator tokens (CE counts answer tokens only)
- WikiText content tokens always count (no prompt/answer split), so 'mixed' trains fluency
  on WikiText and answer-content on instruction
- arm tag becomes QR-002a(linears)+ansmask; result JSON carries answer_loss_only=true
- when the flag is OFF the token stream is byte-identical to the FACT-002 runs (reproducible)
```

Arms (same RT-120 recipe/budget as FACT-002, factual panel still eval-only):

```bash
# instruction + answer-only mask
python scripts/rt116_quality_recovery.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source instruction --answer-loss-only --steps 800 --seq-len 256 --batch 4 \
  --grad-accum-steps 6 --lr 2e-4 --max-train-tokens 2000000 --dtype float32 \
  --optim adamw8bit --grad-checkpointing --bitnet /content/bitnet.cpp \
  --out-dir /content/bnt_runs/tinyllama_fact003a_instr \
  --json-out reports/rt132_fact003a_instr_train.json --log-every 25
# mixed + answer-only mask: --train-source mixed --out-dir ..._mixed
```

Score with the RT-130 panel (rep1.2) and compare `fact_rate` to FACT-002 (instr 0.00,
mixed 0.07), Q2_K 0.74, FP 0.81. Pass = `fact_rate` clearly up (e.g. >= 0.15) and no
empty-collapse; if still ~0 but fluent, go to FACT-003B (base-KL replay).

#### FACT-003A / RT-132 RESULT (2026-06-26): answer-only mask is the first lever to move facts

| arm | FACT-002 (no mask) | **FACT-003A (+ansmask)** | CE recovered | degeneration | i2s vs f16 |
| --- | ---: | ---: | ---: | --- | --- |
| instruction | 0.00 (empty 25/27) | 0.04 (1/27) | 0.40 -> **0.555** | ok 27 (collapse fixed) | -0.0047 nats |
| **mixed** | 0.07 | **0.15** (i2_s 4/27; f16 0.19) | 0.814 -> **0.822** | ok 26/27 | +0.0070 nats |

References reproduced: FP 0.815, Q2_K 0.741, PTQ 0.0. Reads:
- **The objective moved facts where data could not.** mixed `fact_rate` doubled 0.07 -> 0.15
  (meets the 003A pass bar); instruction 0.00 -> 0.04. Confirms S3: the lever is the
  training objective, not the data.
- **Empty-collapse fixed.** Instruction-only no longer degenerates to empty answers
  (empty 25 -> ok 27), validating the prompt-format-overfit hypothesis. CE recovery up on
  both arms.
- **Runtime exonerated again** (adapted i2_s ~ f16). Artifacts: `reports/rt132_fact003a_*`.

But absolute facts (0.15) are still far below Q2_K 0.74 — answer-only masking is necessary,
not sufficient. This motivated the strategy pivot below.

### Strategy pivot (2026-06-26): teacher-free -> base-anchored practical conversion

Goal is reframed from "teacher-free research result" to a **usable, cheaply-runnable b1.58
/ I2_S model**: keep the systems wins (small/fast, i2_s==f16, fluency recovery, decoding
anti-collapse) and stop the factual/instruction forgetting during adaptation. The cheapest
anchor is the SAME original FP model used as a self-teacher (no new large teacher): hold the
b1.58 model's answer distribution near the base model's on a small replay set. FACT-003B
implements this (answer-only CE + base-KL replay); a precomputed base-logits cache keeps it
runnable on an underdog GPU. Near-term target: fact_rate 0.15 -> >= 0.40 ("usable" tier),
not Q2_K parity.

### FACT-003B: answer-only CE + base-KL replay (base-anchored, IMPLEMENTED)

The chosen practical recipe (see the strategy pivot above). Loss:

```text
L = CE_answer(adapt data)  +  lambda * KL(base || student) on answer tokens of a replay set
```

The anchor teacher is the SAME original FP model (a self-teacher, not a new large teacher),
frozen in fp16; the replay set is a fixed slice of Dolly instructions (disjoint from the
FACT-001 panel — enforced by `scripts/check_fact_panel_overlap.py`). Forward-KD direction
KL(base||student) makes the b1.58 student keep the base model's answer distribution, which
is exactly the behaviour data-swaps lost.

Implemented in `scripts/rt116_quality_recovery.py`:

```text
--base-kl-replay     enable the anchor (loads a frozen base teacher + builds the replay pool)
--kl-weight  L       weight lambda on the KL term (default 1.0)
--kl-temp    T       distillation temperature, KD scaled by T^2 (default 1.0)
--replay-tokens N    size of the fixed instruction replay pool (default 200k)
--replay-batch B     replay windows per step for KL (default 2; keep small for memory)
--teacher-dtype      base teacher dtype (default float16, halves its memory)
- composes with --answer-loss-only; arm tag gains +basekl<lambda>; per-step log adds kl=..
- result JSON records base_kl_replay / kl_weight / kl_temp / replay_tokens / replay_batch
- teacher is freed before GGUF export; flag OFF -> behaviour identical to FACT-003A
```

Preflight (run once): leakage gate must PASS.

```bash
python scripts/check_fact_panel_overlap.py   # Dolly replay vs data/factual_panel_v1.jsonl
```

Arms (mixed is the FACT-003A winner; sweep lambda):

```bash
python scripts/rt116_quality_recovery.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source mixed --answer-loss-only --base-kl-replay --kl-weight 1.0 \
  --steps 800 --seq-len 256 --batch 4 --grad-accum-steps 6 --lr 2e-4 \
  --max-train-tokens 2000000 --dtype float32 --optim adamw8bit --grad-checkpointing \
  --bitnet /content/bitnet.cpp --out-dir /content/bnt_runs/tinyllama_fact003b_mixed_kl1 \
  --json-out reports/rt133_fact003b_mixed_kl1_train.json --log-every 25
# sweep: --kl-weight {0.5, 1.0, 2.0}; pick by fact_rate vs WikiText-CE/PPL tradeoff
```

Score with rt130 (rep1.2). Target: fact_rate 0.15 -> >= 0.40 ("usable" tier), WikiText
CE/PPL not collapsed, i2_s == f16 preserved. Memory note: adds a frozen fp16 teacher
(~2.2 GB for 1.1B) + a small replay forward; fits an L4, tight on a T4 (lower --replay-batch
or --teacher-dtype, or use the v2 logits cache below).

### FACT-003B v2 (logits cache) and FACT-003C (only if needed)

```text
v2 cache: precompute top-k base logits for the fixed replay set once, drop the in-memory
          teacher -> faster + lower memory (the underdog-GPU path). Implement after v1 works.
FACT-003C: protected factual replay (small fact set disjoint from FACT-001) + the leakage
           gate above; only if base-KL replay alone is insufficient.
extras:    CE + repetition/unlikelihood or entropy-floor regularizers, only if degeneration
           returns under sane decoding.
```

Pass rule:

```text
same or better factual score than FACT-002 winner,
lower repetition under greedy or lower repetition-penalty requirement,
no regression in I2_S-vs-F16 parity.
```

## FACT-004: Promotion To Paper Claim

A factual-quality result can enter the paper only if:

```text
adapted I2_S improves over current RT-120 adapted I2_S on FACT-001 panel
adapted I2_S remains same-tier as adapted F16
generation stays non-degenerate under the standard decode
storage/speed path remains I2_S
comparison against Q2_K is reported honestly
```

Possible claim levels:

| Level | Requirement | Paper wording |
| --- | --- | --- |
| L0 | non-degenerate only | "usable-tier text, factual gap open" |
| L1 | simple factual panel close to Q2_K | "basic factual prompts mostly preserved" |
| L2 | instruction/factual panel within narrow band of Q2_K | "approaches Q2_K on this small factual panel" |
| L3 | benchmark subset close to Q2_K/FP | "factual quality substantially recovered" |

Current state after RT-129:

```text
L0 achieved.
L1+ not yet measured.
```

## Suggested Driver

Create:

```text
scripts/rt130_factual_gap_panel.py
```

Responsibilities:

```text
load or receive GGUF paths for FP/Q2_K/PTQ/adapted-f16/adapted-i2s
run llama-cli with fixed decode configs
save raw outputs
score with simple exact/contains rules where possible
emit markdown side-by-side table
emit JSON with tags and aggregate rates
```

Minimum CLI shape:

```bash
python scripts/rt130_factual_gap_panel.py \
  --bitnet /content/bitnet.cpp \
  --models-dir /content/bnt/reports/tinyllama_rt120_ggufs \
  --prompt-file data/factual_panel_v1.jsonl \
  --decode rep1.2 \
  --json-out reports/fact001_current_gap.json \
  --markdown-out reports/fact001_current_gap.md
```

Prompt file schema:

```json
{"id":"capital_france","category":"simple_fact","prompt":"Q: What is the capital of France?\nA:","answers":["Paris"],"must_contain":["Paris"]}
```

## Data Rules

Start with a hand-checkable prompt panel before using larger benchmarks.

Rules:

```text
license-safe prompts only
no training on the FACT-001 eval panel
keep FACT-001 fixed across all arms
separate adaptation data from evaluation prompts
report if a prompt is ambiguous or model-size inappropriate
```

Potential adaptation data:

```text
small instruction-following corpus
simple QA/factual statements
mixed WikiText + instruction
project-local synthetic factual templates only for controlled ablation
```

Do not claim broad factual quality from synthetic templates alone.

## Expected Outcomes

### Outcome A: adapted I2_S is already close to Q2_K on simple facts

Then update the paper claim from L0 to L1 and run seed/size hygiene.

### Outcome B: adapted I2_S is readable but factually weak

This is the most likely outcome. Proceed to FACT-002 instruction/mixed-data adaptation.

### Outcome C: instruction data closes much of the gap

Then the final recipe becomes:

```text
FP checkpoint -> b1.58 QAT with mixed/instruction CE -> I2_S export -> rep-penalized decode
```

### Outcome D: factual gap persists despite better data

Then the remaining limit may be capacity/precision. Do not return to one-shot quantizer
search; instead evaluate adaptive capacity:

```text
row-scale carry-forward
small protected factual pockets
longer QAT
larger base model
```

## Decision

The next concrete experiment should be:

```text
FACT-001 / RT-130: current factual gap panel
```

Only after FACT-001 should we spend GPU on instruction or longer adaptation.

## FACT-001 / RT-130 RESULT (2026-06-25): the factual gap is large and is a DATA problem (Outcome B)

`scripts/rt130_factual_gap_panel.py` on `data/factual_panel_v1.jsonl` (27 prompts),
TinyLlama-1.1B variants, rep-penalty 1.2 primary.

| variant | fact_rate | tags |
| --- | ---: | --- |
| FP f16 | 0.81 (22/27) | all ok |
| Q2_K | 0.74 (20/27) | all ok |
| PTQ i2_s | 0.00 | salad 24 |
| adapted f16 | 0.00 | ok 24 / salad 3 |
| adapted i2_s | 0.04 (1/27) | all ok |
| adapted i2_s (greedy) | 0.00 | salad 12 (greedy degenerates) |
| adapted i2_s (t0.8/p0.95) | 0.07 | ok 26 |

adapted i2_s vs f16 hit-agreement: **26/27**.

Findings:
1. **The base model knows the facts** (FP 0.81, Q2_K 0.74). Ternary *quantization* does
   not erase factual knowledge — Q2_K of the same base keeps it.
2. **Our adapted model is fluent but factually ~empty** (0.00-0.04), despite all-ok
   degeneration tags under rep-penalty. The WikiText-CE adaptation produced a fluent
   WikiText continuation model and **overwrote the base model's factual/instruction
   knowledge (catastrophic forgetting).**
3. **Runtime is exonerated**: adapted i2_s == adapted f16 (26/27); I2_S faithfully
   preserves the (factually weak) adapted behavior. This is NOT a quantizer/runtime gap.
4. **PPL-recovery != knowledge-recovery**: the RT-116/120 "recovery" recovered WikiText
   modeling, not the model's knowledge.

```text
VERDICT (FACT-001): Outcome B — readable but factually weak vs Q2_K; runtime/quantizer
exonerated. Claim level stays L0 (non-degenerate), L1 NOT achieved (facts far below
Q2_K). The lever is adaptation DATA/objective, not bits/runtime: the next experiment is
FACT-002 (instruction/factual or MIXED WikiText+instruction adaptation), to recover
fluency WITHOUT forgetting facts. Critically, an honest paper must state that the
WikiText-CE recovery trades factual knowledge for fluency.
```

## FACT-002 runbook (data-only adaptation; same recipe/budget as RT-120)

For the complete single-flight execution path, including preflight, reference prep,
instruction/mixed arms, scoring, decision tree, failure branches, archive checklist,
and handoff prompt, use:

- [Factual Recovery Master Runbook](./factual_recovery_master_runbook.md)

The short commands below are kept as the minimal FACT-002 reminder.

rt116 now takes `--train-source {wikitext|instruction|mixed}` (instruction = Dolly-15k
formatted `Q:..\nA:..`; eval CE stays WikiText; factual eval is the fixed RT-130 panel,
never trained on). Arms, all at the RT-120 budget (800 steps, microbatch 4 x accum 6 =
eff 24, fp32 + AdamW8bit + grad-ckpt, --bitnet to export f16+i2_s):

```bash
# A baseline = existing RT-120 adapted (wikitext) -> already have it.
# C instruction-only:
python scripts/rt116_quality_recovery.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source instruction --steps 800 --seq-len 256 --batch 4 --grad-accum-steps 6 \
  --lr 2e-4 --max-train-tokens 2000000 --dtype float32 --optim adamw8bit --grad-checkpointing \
  --bitnet /content/bitnet.cpp --out-dir /content/bnt_runs/tinyllama_fact002_instr \
  --json-out reports/rt131_fact002_instr.json --log-every 25
# D mixed (instruction + wikitext): --train-source mixed --out-dir ..._mixed
```

Then score each adapted dir with the RT-130 panel (rep1.2) and compare fact_rate to
A (0.04), Q2_K (0.74), FP (0.81). Decision: fact_rate >= 0.4 and low degeneration =>
data-only recovery works; fluent-but-fact~0 => CE objective insufficient (FACT-003);
adapted i2_s != f16 => runtime issue (not expected).

## FACT-002 / RT-131 RESULT (2026-06-26): data does NOT close the gap -> S3 objective gap

Ran the master runbook end-to-end on Colab L4 (same recipe/budget as RT-120: 800 steps,
eff-batch 24, fp32 + AdamW8bit + grad-ckpt; factual panel eval-only, never trained on).

| arm | train_source | CE_adapted (FP 2.31 / PTQ 11.53) | recovered | adapted PPL (FP 10.1) | fact_i2s (rep1.2) | i2s vs f16 | degeneration |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| A wikitext | wikitext | — | (RT-120) | — | 0.037 (1/27) | 26/27 | canned WikiText boilerplate, every prompt |
| C instruction | Dolly-15k Q/A | 7.82 | 0.40 | 2480 | 0.000 (0/27) | 27/27 | **empty 25/27** (early EOS) |
| D mixed | wikitext+instr | 4.03 | **0.81** | 56.2 | 0.074 (2/27) | 27/27 | fluent, varied (ok 26/27) |

References this run: FP 0.815, Q2_K 0.741, PTQ 0.0 (reproduced RT-130 exactly).

Reads:
- **Mixed is the clear best arm and recovers fluency strongly** (CE 0.81 recovered, PPL
  2480->56, no empty/canned collapse, `ok` tags) — yet `fact_rate` stays at the floor
  (0.074 vs Q2_K 0.74). The model speaks well and hallucinates: "The French Navy is a
  sub-national... English naval officer", "The Grand Tires are a city in Milan, Paris",
  "Moscow, Moscow, Moscow..." loops. See `reports/rt131_fact002_mixed_fact.md`.
- **Instruction-only collapses** to empty answers (25/27) — Dolly Q/A formatting taught
  early EOS; worst arm despite 0.40 CE recovery.
- **WikiText-only** emits one canned passage for every prompt (degenerate), as in RT-130.
- **Runtime fully exonerated** on all arms: adapted i2_s == adapted f16 (26-27/27
  hit-agreement; |i2_s-f16| = 0.012 nats on mixed). The gap is not a bit/codebook problem.

Decision (runbook Stage 4 tree): refs sane (FP>=0.7, Q2_K>=0.6) -> i2_s~f16 -> best
fact_rate 0.074 < 0.15 -> outputs fluent/non-degenerate -> **S3: objective gap**.

Conclusion: the CE-on-demonstrations objective recovers *fluency* but not *factual
knowledge* at this scale/budget — the lever is the objective, not the data and not the
runtime. Next is FACT-003 (objective branch: replay/KL regularization, answer-only loss
mask, repetition penalty, or protected factual replay). FACT-003 requires code changes;
it is NOT already supported by rt116. Artifacts: `reports/rt131_fact002_*`.
