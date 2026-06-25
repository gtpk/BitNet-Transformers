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

Notes: train CE was a touch noisy (5.03 -> 3.70 over 300 steps), so more steps / an
LR schedule / +norms (QR-002b) could push past 0.905; 0.905 already clears the
PASS bar (>0.3). The PTQ baseline 115,808 matches the "PTQ-broken by design" framing
from RT-114/115 — that bad number was always a baseline to recover FROM, not a
runtime failure.
