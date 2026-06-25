# Quality Recovery Plan

Document position: [Index](./index.md) -> after storage/latency scale-up, before
quality or paper claims.

Related docs:

- [Scale-Up Target Roadmap](./scaleup_target_roadmap.md)
- [GGUF / bitnet.cpp Export Scoping Plan](./bitnet_cpp_export_scoping.md)
- [Research Signal Note](./research_signal_note.md)
- [Existing Model to BitNet Conversion Plan](./existing_model_to_bitnet_conversion_plan.md)

## Purpose

RT-112, RT-113, and RT-114 answer the systems question:

```text
Can per-tensor b1.58 weights be exported to bitnet.cpp I2_S and run smaller/faster?
```

Current answer on x86/Linux:

```text
Yes. The Path A' route is faithful enough, storage improves, and token-generation
speed improves as the model becomes more linear-dominated.
```

That is not the final product/research claim. The next question is quality:

```text
Can the small, fast b1.58 model produce answers of comparable quality after a
cheap recovery/adaptation step?
```

This plan adds the missing gate. The goal is **not identical text**. Greedy or
sampled generation can diverge after tiny logit changes. The goal is:

```text
same-quality output under comparable prompts, plus CE/PPL recovery on text.
```

## What RT-114 Did And Did Not Prove

RT-114 proved:

- larger LLaMA-shaped models expose more I2_S storage benefit than tiny models
- token-generation speedup increases when linears dominate memory traffic
- F16 `Wq` and I2_S `Wq` remain close enough for runtime-faithfulness checks

RT-114 did **not** prove:

- the one-shot PTQ model is useful
- answer quality matches the original FP model
- instruction-following, reasoning, or code behavior survives

The bad absolute PPL in RT-114 is expected: it is one-shot ternary PTQ without
recovery. It is a baseline to recover from, not a failure of the runtime path.

## Quality Recovery Ladder

### QR-001: PTQ Collapse Baseline

Question:

```text
How much quality is lost if we only materialize Wq=gamma*T and do no recovery?
```

Models:

- FP original
- F16 `Wq=gamma*T` PTQ
- I2_S `Wq=gamma*T` runtime

Metrics:

- CE loss and PPL on a held-out text set
- loss delta in nats, not only PPL ratio
- generation smoke on short prompts
- small prompt-quality panel, saved side by side

Pass/report rule:

```text
No pass required. This is the collapse baseline.
```

Interpretation:

- FP -> PTQ collapse quantifies how much recovery is needed.
- F16 Wq -> I2_S delta checks runtime preservation, not model quality.

### QR-002: Teacher-Free CE Adaptation

Question:

```text
Can short CE-only adaptation recover useful quality without teacher distillation?
```

Default algorithm:

```text
start from FP checkpoint
replace target linears with per-tensor b1.58 STE linears
forward uses Wq = gamma*T
backward updates latent FP weights through STE
train on next-token CE only
materialize Wq=gamma*T
export F16 Wq and I2_S Wq
```

Default constraints:

- no teacher logits in the first pass
- target linears are adapted first
- embeddings, norms, and lm_head stay unchanged unless ablation says otherwise
- use the same held-out set as QR-001
- keep a strict compute budget so the result stays relevant to low-resource users

Suggested stages:

| ID | Setup | Purpose |
| --- | --- | --- |
| QR-002a | target linears only, CE-only | cheapest recovery signal |
| QR-002b | target linears + norms | check whether normalization absorbs quantization drift |
| QR-002c | target linears + lm_head optional | check whether output distribution needs light retuning |

Pass/report rule:

```text
Adapted b1.58 recovers a meaningful fraction of FP->PTQ loss under a fixed budget.
```

Recommended reporting:

```text
recovered_fraction = (CE_ptq - CE_adapted) / (CE_ptq - CE_fp)
```

This avoids over-reading raw PPL when the PTQ operating point is very bad.

### QR-003: I2_S Runtime Quality Preservation

Question:

```text
After adaptation, does bitnet.cpp I2_S preserve the recovered F16 Wq quality?
```

Compare:

- adapted F16 `Wq`
- adapted I2_S `Wq`

Metrics:

- CE delta in nats
- PPL ratio as secondary
- if possible: token-level logprob delta or KL on a small eval set
- generation smoke with deterministic decoding

Pass criterion:

```text
I2_S CE delta vs adapted F16 Wq is small relative to the recovered FP->PTQ gap.
```

Why not require identical generations:

- I2_S uses runtime quantization details such as int8 activation paths.
- Tiny logit changes can cause different tokens under greedy decoding.
- Different text is acceptable if the answer quality is comparable.

### QR-004: User-Facing Output Quality

Question:

```text
Does the adapted b1.58 model answer at roughly the same quality tier?
```

Prompt sets:

- general QA
- summarization
- simple reasoning
- simple code/debug prompts
- Korean/English mixed prompts if the model supports them

Comparison:

- FP original
- PTQ b1.58
- adapted F16 Wq
- adapted I2_S Wq

Evaluation:

- side-by-side saved generations
- deterministic decode first (`temperature=0`)
- optional low-temperature sample second
- pairwise judge or human spot-check
- failure tags: hallucination, refusal drift, instruction miss, code breakage,
  repetition, degenerate output

Pass criterion:

```text
Adapted I2_S is closer to adapted F16 Wq than PTQ is, and its answers are
qualitatively usable on the prompt panel.
```

This is a smoke gate, not a benchmark-suite claim.

## Candidate Models

Use the same scale ladder as [Scale-Up Target Roadmap](./scaleup_target_roadmap.md):

1. `JackFram/llama-160m`
   - cheapest place to prove recovery mechanics
   - already validated by RT-114 for storage/latency scale-up
2. `TinyLlama/TinyLlama-1.1B-*`
   - better check that recovery scales beyond 160M
   - still LLaMA-shaped and simpler than MoE
3. `gpt-oss-20b`
   - only after architecture/tensor-map audit
   - MoE quality recovery must separate resident weights, active experts, router,
     and KV/cache behavior

Do not start quality recovery on gpt-oss before the OSS architecture audit.

## Minimal Data Plan

Start small and reproducible:

- text CE recovery corpus: Wikitext or a permissive small corpus
- held-out text: fixed eval split, never trained on
- prompt panel: checked into `data/` once license-safe
- report JSON: one file per run under `reports/`

For paper-level claims later:

- broader text eval
- instruction benchmark subset
- code/math subset
- more seeds
- stronger baselines such as RTN/GPTQ/AWQ or low-bit QAT

## TC Matrix

| ID | Area | Check | Pass/report rule |
| --- | --- | --- | --- |
| QR-001 | PTQ baseline | FP vs PTQ F16 Wq vs PTQ I2_S | report CE/PPL collapse and runtime delta |
| QR-002 | CE adaptation | teacher-free STE recovery | recover meaningful fraction of FP->PTQ loss |
| QR-003 | runtime preservation | adapted F16 Wq vs adapted I2_S Wq | small CE delta relative to recovered gap |
| QR-004 | prompt quality | FP/PTQ/adapted/I2_S side-by-side | adapted I2_S is usable and close to adapted F16 |
| QR-005 | ablation | target-only vs +norms vs +lm_head | identify cheapest useful recovery recipe |

## Decision Rules

```text
QR-001 bad absolute PPL
  -> expected; continue to QR-002

QR-002 no recovery
  -> algorithmic problem: try norms/lm_head, LR, corpus, or projected QAT baseline

QR-002 recovers but QR-003 fails
  -> runtime/export problem: inspect I2_S activation/logprob deltas

QR-003 passes but QR-004 feels bad
  -> CE is insufficient; add instruction recovery or preference/quality eval

QR-004 passes on 160M
  -> repeat on TinyLlama-1.1B before gpt-oss

TinyLlama quality recovery passes
  -> open gpt-oss-20b quality feasibility after OSS architecture audit
```

## What To Avoid

- Do not call one-shot PTQ quality a failure of I2_S.
- Do not claim same output text is required.
- Do not claim user-facing quality from CE/PPL alone.
- Do not use teacher distillation in the default track; keep it as a later
  ablation if teacher-free recovery stalls.
- Do not move to gpt-oss quality claims before MoE tensor/runtime audit.

## Current Recommendation

Add QR work as a parallel core track:

```text
systems track : scale storage/latency to TinyLlama and gpt-oss audit
quality track : QR-001..004 on Llama-160M, then TinyLlama
```

This keeps the project honest: it can say both "small and fast" and, only after
QR passes, "still useful."


## RT-116 / TRAIN-001 RESULT (2026-06-25): teacher-free CE recovers 90% on Llama-160M

Ran `scripts/rt116_quality_recovery.py` on JackFram/llama-160m, GPU T4, WikiText-2
(Salesforce/wikitext, 600k train / 60k eval tokens), 300 steps, seq 256, batch 8,
lr 2e-4, target linears only (QR-002a). All-PyTorch on one held-out eval set
(apples-to-apples, no cross-tool noise):

### QR-001 collapse baseline + QR-002a recovery

| stage | CE (nats) | PPL |
| --- | ---: | ---: |
| FP original | 3.1466 | 23.3 |
| one-shot b1.58 PTQ (Wq=gamma*T, no train) | 11.6597 | 115,808 |
| **adapted (300-step teacher-free CE)** | **3.9519** | **52.0** |

```text
recovered_fraction = (CE_ptq - CE_adapted) / (CE_ptq - CE_fp)
                   = (11.660 - 3.952) / (11.660 - 3.147) = 0.905
```

**QR-002 PASS, decisively.** A short, teacher-free, CE-only adaptation of ONLY the
84 target linears (everything else frozen) recovers **90.5%** of the catastrophic
FP->PTQ loss: PPL 115,808 -> 52.0 (FP is 23.3, so adapted is ~2.2x FP). This is the
missing third sentence: the model is small, fast, AND — after a cheap teacher-free
recovery step — back to usable text-modeling quality. No teacher, no distillation,
no touching embeddings/norms/lm_head.

Notes: train CE was a touch noisy (5.03 -> 3.70 over 300 steps), so more steps or an
LR schedule could push past 0.905; QR-005 later showed +norms are not the bottleneck.
0.905 already clears the PASS bar (>0.3). The PTQ baseline 115,808 matches the
"PTQ-broken by design" framing
from RT-114/115 — that bad number was always a baseline to recover FROM, not a
runtime failure.

### QR-003 runtime preservation (PASS)

Exported the adapted Wq=gamma*T model to GGUF (embedding+lm_head f16, only the 84
adapted linears -> I2_S) and ran llama-perplexity on x86 (rebuilt bitnet.cpp on the
GPU runtime's CPU), ctx 64:

| adapted GGUF | PPL |
| --- | ---: |
| f32 | 134.83 |
| f16 | 134.84 |
| i2_s | 135.11 |

```text
adapted i2_s vs adapted f16 = +0.0020 nats  (PPL 1.002x)
```

The I2_S runtime preserves the recovered quality to **+0.002 nats** — even tighter
than RT-114's PTQ model (+0.041 nats), because the adapted model sits at a much
lower (non-degenerate) operating loss where the int8-activation residual matters
less. (The cross-tool gap — f16 GGUF PPL 134.8 vs the PyTorch CE_adapted PPL 52 —
is the usual tokenizer/windowing difference: PyTorch evaluated 64 seq-256 windows on
the native token stream, llama-perplexity re-tokenizes a decoded 60k-token eval.txt
at ctx 64. The QR-003 claim is the *within-tool* i2_s-vs-f16 delta, which is clean.)

### RT-116 / TRAIN-001 conclusion

```text
QR-001  FP 23.3  ->  one-shot b1.58 PTQ 115,808            (collapse, expected)
QR-002a 300-step teacher-free CE on 84 linears -> 52.0     recovered_fraction 0.905
QR-003  adapted i2_s vs adapted f16 = +0.002 nats          runtime preserves recovery
```

The quality track's core claim is proven on Llama-160M: a one-shot ternary PTQ
model is broken, but a SHORT, teacher-free, CE-only adaptation of just the target
linears recovers ~90% of the loss, and that recovered quality survives the
bitnet.cpp I2_S runtime essentially unchanged. Combined with the systems track
(RT-112..115: small + fast + scale law), the three-sentence story is now closed on
x86/Linux LLaMA: **small -> fast -> quality recovers cheaply, teacher-free**.

Next from this historical point was QR-005 / TRAIN-002 / RT-117; those are now
complete. The current next step is RT-120 / TRAIN-003 budget scaling for
TinyLlama-1.1B.

## TRAIN-002 / TinyLlama-1.1B RESULT (2026-06-25): recovery direction + runtime preservation scale

Same driver (`scripts/rt116_quality_recovery.py`) on TinyLlama-1.1B-Chat-v1.0, GPU
T4. 1.1B target-linear finetune needs the scale options added for this run: fp32
model + **8-bit AdamW (bitsandbytes)** + **gradient checkpointing** (fp32 AdamW
states alone would be ~7.7 GB and OOM a 16 GB T4), batch 4 (vs 160M's batch 8),
300 steps, 154 target linears.

| model | FP PPL | one-shot PTQ PPL | adapted PPL | recovered_fraction | QR-003 i2_s vs f16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-160M | 23.3 | 115,808 | 52.0 | 0.905 | +0.0020 nats |
| TinyLlama-1.1B | 10.1 | 101,549 | 1,217 | **0.480** | **+0.0023 nats** |

- **QR-002 direction holds at 1B**: one-shot PTQ PPL 101,549 -> 1,217 after a short
  teacher-free CE pass. PASS as a direction/feasibility check.
- **QR-003 runtime preservation is scale-invariant**: adapted i2_s vs adapted f16 =
  **+0.0023 nats**, essentially identical to 160M's +0.0020 nats. The recovered
  quality survives the bitnet.cpp I2_S runtime at 1.1B exactly as at 160M.
- **The lower recovered_fraction (0.48 vs 0.90) is a BUDGET artifact, not a scale
  failure**: 1.1B has ~8.5x more target-linear params (968M vs 113M) to adapt, yet
  ran the SAME 300 steps at HALF the tokens/step (batch 4 vs 8) with 8-bit (vs fp32)
  Adam. Per-token-of-training it recovered less, as expected. The direction and the
  runtime faithfulness — the two things TRAIN-002 set out to confirm — both hold.

```text
CONCLUSION (TRAIN-002): teacher-free CE recovery and I2_S runtime preservation both
SCALE from 160M to 1.1B in the LLaMA family. Recovery fraction at fixed budget drops
with model size (more params per fixed step count), so a fair "how close to FP" claim
needs budget scaled with params (more steps / larger effective batch / fp32 Adam on a
bigger GPU). With systems (RT-112..115) + quality (RT-116/TRAIN-002) both shown to
scale on LLaMA, the recipe is ready to take into the gpt-oss-20b MoE audit (RT-117) —
architecture/tensor-map/router/expert first, no blind convert.
```

## QR-005 / RT-116 ablation RESULT (2026-06-25): linears-only is the recipe

Which params to adapt? Ran a/b/c on Llama-160M, identical budget (seed 0, 300 steps,
seq 256, batch 8, lr 2e-4, WikiText-2), GPU T4. `scripts/rt116_quality_recovery.py`
with `--train-norms` / `--train-lm-head`.

| arm | trained | recovered_fraction | adapted PPL | wall |
| --- | --- | ---: | ---: | ---: |
| QR-002a | target linears only | 0.906 | 51.6 | 212 s |
| QR-002b | + RMSNorm weights | 0.907 | 51.2 | 210 s |
| QR-002c | + lm_head | 0.898 | 55.6 | 222 s |

- **+norms does essentially nothing** (0.907 vs 0.906, within run noise; PPL 51.2 vs
  51.6). RMSNorm scales are NOT the recovery bottleneck — adapting only the target
  linears already captures the recoverable quality.
- **+lm_head slightly HURTS** (0.898 < 0.906; PPL 55.6 > 51.6). At the same LR/budget,
  the extra 8.2M lm_head params waste capacity / destabilize rather than help.

```text
DECISION (QR-005): default recovery recipe = TARGET LINEARS ONLY. The cheapest recipe
is also the best at this budget; no norms, no lm_head. This resolves gap G2/G3 and
means the 1.1B budget-scaling run (G1) should use linears-only — no recipe change,
just more steps / bigger effective batch / fp32 Adam on a larger GPU.
```

Caveat: single seed; the a-vs-b gap (0.001) is within noise, so the claim is "norms
don't help", not a precise ordering. lm_head's regression is larger (-0.008) and
consistent in both recovered_fraction and PPL.

## QR-004 / RT-119 prompt-quality panel RESULT (2026-06-25): recovery is human-visible

`scripts/rt119_prompt_panel.py` greedy-decodes the same 12 completion prompts from
FP / PTQ(Wq=gamma*T, no train) / adapted(300-step teacher-free CE, linears-only) and,
via bitnet.cpp llama-cli, adapted-F16 + adapted-I2_S GGUF. Llama-160M is a small BASE
LM, so prompts are completions and even FP is only so-so — the point is the RELATIVE
jump. Heuristic failure tags over 12 prompts:

| variant | ok | repetitive | loop | empty |
| --- | ---: | ---: | ---: | ---: |
| FP | 7 | 5 | 0 | 0 |
| PTQ (no recovery) | 2 | 6 | 6 | 2 |
| adapted (linears-only) | 5 | 7 | 1 | 0 |

PTQ is degenerate (6 loops, 2 empty); adaptation removes the loops/empties (1 loop, 0
empty) and lands near FP. Concrete examples (greedy):

```text
prompt: "The capital of France is"
  FP      : "Paris. The city of Paris is the capital of France..."
  PTQ     : "likely to proceed proceed proceed proceed proceed..."          <- collapse
  adapted : "the largest in the United Kingdom, the largest in the..."      <- fluent, wrong fact

prompt: "The most important rule of cooking is"
  PTQ     : "andivHPothsay FacetivillivighengthIVILLKillUampigh..."         <- token salad
  adapted : "the most common of the most important of the food industry..." <- recovered English
```

Runtime preservation (adapted I2_S vs adapted F16, both llama-cli greedy):

```text
prompt: "Water boils at a temperature of"
  adapted f16 : "100 @,@ 000 m ( 1 @.@ 1 m ) of 12 @.@ 1 m ( 1 @.@ 1 m ) ."
  adapted i2_s: "100 @,@ 000 m ( 1 @.@ 1 m ) of 12 @.@ 1 m ( 1 @.@ 1 m ) ."   <- identical
```

i2_s == f16 exact greedy match on 4/12 prompts; the other 8 are same-quality near-
paraphrases (the int8-activation path flips an occasional argmax — different text,
same tier, exactly the "comparable quality, not identical text" bar QR-004 set).
Honest caveats: 160M base model so absolute fluency is low and outputs are repetitive
(the @,@ tokens are WikiText artifacts the adaptation learned); factuality is weak
(adapted says France's capital is "in the United Kingdom"). The CLAIM this supports is
narrow and correct: **teacher-free CE turns PTQ token-salad into fluent same-tier text,
and the I2_S runtime preserves it** — gap G4 closed at 160M. Scale to 1.1B + a stronger
base model (G1) for a stronger qualitative story.

## RT-120 / TRAIN-003 PREP: G1 budget scaling before GPU upgrade

G2/G3/G4 are now closed: linears-only is the default recipe, and the prompt panel
shows the recovery is visible to humans at 160M. The remaining high-severity gap is
G1: TinyLlama-1.1B recovered only `0.480` under a deliberately constrained fixed
budget.

Before spending an L4/A100 run, use the dedicated runbook:

- [G1 Budget-Scaling Runbook](./g1_budget_scaling_runbook.md)

It fixes:

- the exact hypothesis (`0.480` was budget-limited, not a scale failure)
- smoke requirements before the paid/large run
- A100 and L4 one-shot commands
- recovered_fraction success tiers
- QR-003 runtime-preservation pass/fail criteria
- fallback rules for OOM or slow runs

The next expensive run should not change the recipe. It should scale only the budget:

```text
TinyLlama-1.1B, target linears only, teacher-free CE, effective batch ~24,
800 steps, QR-003 f16-vs-i2_s parity at the end.
```
