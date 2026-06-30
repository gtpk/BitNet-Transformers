# TinyLlama-1.1B longer-budget (1600 steps) -- the decisive (b1) result

DINO recipe (content-KL 0.2 + DINO-logit 0.2, no centering/warmup) that COLLAPSED at 800 steps
(DINO-002), re-run to 1600 steps with teacher-relative telemetry. Question: hard collapse or a
transient that just exceeds the 800-step budget? Drive: bnt_results/tl1b_long.

## Trajectory (teacher baseline: degen 0.20, gold_rank 3)

| step | degen_gap | gold_rank | top1 | train_ce |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +0.10 | 8885 | 0.68 | 11.84 |
| 150 | +0.80 | 7009 | 0.04 | 7.51 |
| 300 | +0.80 | 5690 | 0.03 | 7.29 |
| 600 | +0.80 | 3001 | 0.11 | 6.32 |
| 800 | +0.40 | 2006 | 0.09 | 5.86 |
| 850 | -0.20 | 2384 | 0.11 | 6.28 |
| 1000 | -0.20 | 1772 | 0.11 | 5.65 |
| 1250 | -0.20 | 912 | - | - |
| 1400 | -0.20 | 577 | - | - |
| 1599 | -0.20 | 375 | 0.18 | 3.95 |

Final: recovered_fraction **0.806**, CE_adapted **4.08** (~ the content-KL FACT-003C baseline 0.845/4.10).

## Three-level verdict (kept separate, per the agreed framing)

1. **Generation stability: RECOVERED.** degen_gap sustained ~0 from step ~850 (10+ checks), tags
   **ok 27/27** (no salad/loop), CE recovered to 4.08. The DINO-002 "collapse" at 800 was a
   degenerate transient that resolves by ~step 850 -- **a BUDGET/SCHEDULE limit, NOT a hard collapse.**
   hidden_var_mid stayed low (~0.3-1.7) throughout, unlike pythia's spike (model-difference).
2. **Factual rank: improved but low.** gold_rank 2006 (recovery onset) -> 375 (final), still far from
   the FP teacher's 3. The factual token is reachable internally but not top.
3. **Exact factual answer: 0.111 (3/27) -- BELOW the content-KL 0.185 baseline.** The recovered model
   is FLUENT but NOT factual and drifts off the Q/A answer format. Sample generations:

```
Q: What is the capital of France?  A: -> "The capital city in Spain is a country that has been
   built around 10th century. It was founded by the Spanish Civilian Republic..."   [miss]
Q: What is the capital of Japan?   A: -> "The capital city in India (B) is a country that has been
   built on the bank of the capital city..."   [miss]
Q: Water is made of hydrogen and what other element? A: -> "The water in the ocean is a fluid,
   with its properties that make it hard to float..."   [miss]
```

Fluent, coherent, but factually wrong AND rambling (base-LM continuation style) instead of emitting
"A: <answer>". gold_rank 375 reachable internally but generation does not surface it -> a
**readout/format decouple**, not a generation collapse.

## Conclusion

- **Reframe CONFIRMED:** the 1.1B "collapse" under an auxiliary objective is a budget-limited
  transient, not a hard impossibility -- with 1600 steps generation fully recovers (ok 27/27,
  recovered_fraction 0.806). The earlier "content-KL + anything collapses at 1.1B" was an
  800-step-budget artifact for the COLLAPSE part.
- **But DINO at 1.1B is NOT a factual win:** even recovered, FACT 0.111 < content-KL 0.185, and the
  model rambles off-format. So DINO does not beat the v0 content-KL recipe on facts; it trades the
  terse Q/A answer behaviour for fluent base-LM-style continuation (the unlabeled-text DINO pressure
  drifts the format).

## Recommendation

- content-KL 0.185 remains the v0 FACTUAL best at 1.1B.
- The limiter is now FACTUAL READOUT / answer-format, not generation collapse -> the productive next
  levers are answer-format/answer-token-weighted objective or decoding, NOT just more budget.
- TL1B-2400 (gold_rank still falling at 1600) might lift gold_rank further but unlikely to fix the
  format-drift -> lower priority than addressing readout/format.
- PT2-lite stays a transient-shortener candidate, not a factual rescue.
