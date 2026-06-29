# DINO-CONTROL-B (1.1B fp32 + adamw8bit): COLLAPSED -> confound resolved = SCALE, not precision/optimizer

Precision-isolation control. fp32 student + adamw8bit (vs DINO-002's bf16 + adamw8bit); everything
else identical to DINO-002 (dino-weight 0.2, lr 2e-4, no centering/warmup). Run:
colab/run_dino_ctrl_fp32.sh.

## Result -- COLLAPSED at step 200 (harder/earlier than bf16)

```
[collapse-check] step 200 salad/empty/loop frac=1.00   <- 100% degenerate, COLLAPSE DETECTED
recovered_fraction=0.493 (at stop)
```

fp32 collapsed AT step 200 with frac 1.00 -- even earlier/harder than Control-A (bf16 + plain
adamw, which was clean at 200 and collapsed by 300). So fp32 is not even marginally more stable.

## The confound is resolved: SCALE / base model, not precision or optimizer

| factor | control | result |
| --- | --- | --- |
| optimizer (adamw8bit -> plain adamw) | DINO-CONTROL-A (bf16) | collapsed (step 300) -> NOT the cause |
| precision (bf16 -> fp32) | DINO-CONTROL-B (this) | collapsed (step 200, frac 1.00) -> NOT the cause |
| **scale / base model** | (remaining) | **CONFIRMED cause** |

The same DINO-logit objective is a clear positive at 160M (DINO-DIAG-001: gold logprob/rank up) but
collapses generation at 1.1B regardless of precision (fp32/bf16) or optimizer (adamw/adamw8bit). The
1.1B collapse is therefore **not a numerical artifact** -- it is a property of the model scale/base
under same-topology I2_S adaptation. This validates the scale-dependent stability thesis and removes
the confound the user (correctly) flagged.

## Decision (now firm)

- **The content-KL 0.185 ceiling at 1.1B STANDS.** Auxiliary objectives (replay / blend / DINO)
  collapse at 1.1B, and that collapse is scale-driven, not fp32/bf16/optimizer.
- **Next = the scale ladder (Pythia), as a mechanism study:** keep the same minimal content-KL +
  DINO-logit recipe, vary scale on a controlled same-family ladder (Pythia 160M/410M/1B/1.4B/2.8B,
  same tokenizer/pretraining/curriculum) to find the **collapse-onset scale** and characterise the
  threshold. Thesis: *"Auxiliary objectives exhibit a scale-dependent stability threshold under
  same-topology I2_S adaptation."*
- Qwen remains the separate product track (higher factual base), not the mechanism vehicle.

The 160M DINO mechanism stays a documented positive; the question is now WHERE on the scale axis it
stops working, which Pythia answers cleanly.
