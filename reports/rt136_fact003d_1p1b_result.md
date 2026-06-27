# FACT-003D 1.1B mu=1.0 -- protected factual replay: transfer FAILED (memorisation overfit)

TinyLlama-1.1B, content-KL lambda=0.2 + mu=1.0 answer-CE on the 291 short atomic facts. PyTorch
ternary-forward scoring on the materialized adapted dir (GPU free post-train). Survived two VM
recycles via Drive ckpt + the resume fixes (RNG, CPU-load OOM).

| panel | fact_rate | tags | note |
| --- | ---: | --- | --- |
| eval_panel (real held-out eval) | **0.111** (3/27) | ok 25, repetitive 1, salad 1 | fluent but wrong |
| heldout_atomic (transfer) | 0.134 (13/97) | ok 95 | low |
| train_atomic (memorise control) | **1.00** (80/80) | ok 80 | fully memorised |
| recovered_fraction | 0.831 (ppl 47) | -- | fluency recovered fine |

Baselines: FACT-003C (no replay) eval 0.185 (rt130 i2_s); 160M mu=0 control 0.037; 160M mu=1 mean 0.259.

## Verdict: FAILED (<=0.185) -- memorisation without transfer

- train_atomic 1.0 (the 291 replay facts are perfectly learned) but eval_panel **0.185 -> 0.111
  (down)**. Outputs are FLUENT (ok 25/27), so this is not a fluency collapse (contrast FACT-004A
  lm_head, which was fluent-but-forgot). It is **fluent-but-overfit**: mu=1.0 replay narrowed the
  model onto the 291 replay facts and crowded out its broader factual recall.
- The train-1.0 / eval-0.111 signature is scorer-independent evidence of overfitting-not-transfer
  (the small PyTorch-vs-rt130 scorer offset doesn't explain a memorise-everything / eval-down split).

## The 160M predictor did NOT transfer to 1.1B (methodology lesson)

- 160M predicted 0.037 -> 0.259 (big lift); 1.1B gave 0.185 -> 0.111 (a drop). Opposite direction.
- Cause = different baseline REGIMES. 160M starts at a ~0 factual floor, so memorising 291 facts +
  replay ADDS behaviour it lacked -> rises. 1.1B already had 0.185 recall, so aggressive replay
  OVERFITS and displaces existing recall -> falls.
- Takeaway: the 160M predictor is reliable for "does X do anything" (it cleared the mu=0 control
  robustly) but NOT for "does X preserve/improve when a real baseline already exists." Branch-kill
  with it, but do not trust its magnitude/direction across baseline regimes.

## Next options (per the FACT-003C/D decision tree)

1. **Lower mu (0.25 / 0.5):** mu=1.0 over-memorised at 1.1B; a gentler factual nudge may preserve
   general recall while still anchoring facts. Cheapest next 1.1B run. (160M sweet-spot mu=1.0 does
   NOT carry over -- pick mu by 1.1B behaviour, not the 160M sweep.)
2. **Broader / public factual data:** 291 facts is small enough to memorise outright; a larger,
   more diverse factual set (TriviaQA/NQ short-answer, de-leaked) may force generalisation instead
   of memorisation.
3. **content-AKL:** anchor factual content via KL (not direct CE), so the model is pulled toward the
   base's factual distribution without hard-memorising specific answer tokens.

Note: FACT-003E (length-mix, 160M) showed length variety buys seed STABILITY not magnitude -- it does
not address this 1.1B overfitting (same memorise-the-set mechanism), so the lever is mu / data
breadth / objective, not surface form.
