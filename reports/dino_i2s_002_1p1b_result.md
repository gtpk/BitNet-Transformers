# DINO-I2S-002 result (1.1B): generation COLLAPSED -- instability, not the predicted coverage story

Run: colab/run_dino002.sh (TinyLlama-1.1B, dino_logit-only lambda 0.2 + content-KL replay 0.2, EOS
mask on, hidden off, bf16 + adamw8bit + grad-ckpt, 800 steps, seed 1). Scored: scripts/score_dino002.py.

## Result -- FAIL (collapse)

| metric | value |
| --- | --- |
| recovered_fraction / ce_adapted | 0.620 / 5.79 |
| eval_panel fact_rate | 0.111 (3/27) -- **tags salad 26 / ok 1** (degenerate) |
| by_category | simple_fact 0.059, entity_attr 0.0, reasoning 0.5, instruction 0.0 |
| popqa_tight | 0.0 / 300 -- **salad 267** (degenerate) |
| gold_rank vs FP teacher | catastrophic: cap_france 431 (teacher 3), cap_japan 3640 (1), cap_italy 1175 (2), closest_planet 4089 (1), largest_planet 949 (2); dino_closed_gap_on 5/27 |

The eval 0.111 is NOT a factual signal -- 26/27 outputs are salad; the 3 "hits" are salad text that
happens to contain the answer substring. The model's output distribution is broken: gold tokens sit
at rank 400-4000 while the FP teacher has them at rank 1-7.

## Interpretation: instability, the 160M positive did NOT transfer

This is the OPPOSITE of DINO-DIAG-001 at 160M, where dino_logit RAISED gold ranks (cap_france
60->11) with clean tags. At 1.1B the same objective drove the model to word-salad. Per the session
branch table this is the "tags collapse -> instability (temperature/centering/optimizer)" outcome,
NOT the predicted "simple_fact up / entity_attr lag" coverage limit.

Same 1.1B fragility seen before: FACT-003H v2 (PopQA blend, 1.1B) also collapsed to salad
(recovered 0.706, eval salad), while plain content-KL FACT-003C recovered to a fluent 0.185.
Adding a second objective (blend, or now the DINO unlabeled content-KL) on top of content-KL at
1.1B + bf16 + adamw8bit destabilises generation; at 160M (adamw, fp32) the same addition is benign.

## The open question (what to test next, gated)

Is the destabiliser (a) the DINO term itself at 1.1B, or (b) the 1.1B recipe (adamw8bit / bf16 /
grad-ckpt / LR / no DINO-centering)? A clean separation:

1. **Recipe check**: rerun the EXACT 1.1B recipe with dino-logit-weight 0 (content-KL only). If it
   recovers to a fluent ~0.185 (like FACT-003C) -> the DINO term is the destabiliser at 1.1B. If it
   ALSO collapses -> the adamw8bit/bf16 1.1B recipe is the problem (and FACT-003H's collapse too).
2. If the DINO term is the culprit -> stabilise it: DINO centering (planned but not enabled), lower
   dino weight / LR, smaller view corruption, or adamw (not 8bit) + fp32-ish stability.

Do NOT escalate DINO further until this separation is done. The 160M DINO mechanism is real
(DINO-DIAG-001); the failure here is a 1.1B training-stability problem, not evidence DINO is wrong.
