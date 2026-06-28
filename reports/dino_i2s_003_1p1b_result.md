# DINO-I2S-003 result (1.1B): stabilisation FAILED -> content-KL 0.185 is the 1.1B ceiling

The one stabilisation attempt for the DINO-002 collapse. Run: colab/run_dino003.sh (TinyLlama-1.1B,
dino_logit 0.1 + content-KL replay 0.2, **DINO centering ON**, **warmup 150**, **lr 1e-4**,
early-collapse detector). Goal was stability, not score.

## Result -- COLLAPSE DETECTED at step 200

```
step 200/800  train_ce=7.25  kl=5.77  dino=7.02
[collapse-check] step 200 salad/empty/loop frac=0.90
[COLLAPSE DETECTED] 0.90 > 0.3 -> stopping early (DINO-I2S-003 stabilisation FAILED)
recovered_fraction=0.474 (at stop)
```

Even with every stabilisation knob (centering, weight halved to 0.1, 150-step warmup, lr halved to
1e-4), the 1.1B student was **90% degenerate** (salad/empty/loop) on the panel by step 200. The
detector saved ~60 min by stopping early. This reproduces DINO-002 (salad collapse) -- stabilisation
did not change the outcome.

## Conclusion (decisive): the 1.1B objective-only axis is capped

The pattern is now conclusive across three independent auxiliary objectives at 1.1B:

| 1.1B recipe | outcome |
| --- | --- |
| content-KL alone (FACT-003C) | fluent, fact 0.185 |
| content-KL + hard replay (FACT-003D) | overfit / no transfer |
| content-KL + PopQA blend (FACT-003H v1/v2) | salad collapse |
| content-KL + DINO logit (DINO-002) | salad collapse |
| content-KL + DINO logit, fully stabilised (DINO-003) | salad collapse (step 200) |

**Same-topology 1.1B I2_S adaptation is stable only with the minimal content-KL objective; adding
any auxiliary training pressure collapses generation.** At 160M (adamw/fp32) the same objectives are
benign-to-helpful (DINO-DIAG-001), so this is a 1.1B-specific adaptation fragility, not a flaw in any
single objective. The objective-only axis at 1.1B same-topology is therefore capped at content-KL
**0.185**.

## Decision: accept the ceiling, move the goalpost

- **Accept content-KL 0.185 as the 1.1B same-topology I2_S factual ceiling.** Stop adding auxiliary
  objectives at 1.1B (DINO/blend/replay are closed there).
- **Next axis = the base model, not the objective: the Qwen ladder** (a stronger/larger base whose
  factual floor is higher, then the same minimal content-KL I2_S recipe).
- The 160M DINO mechanism stays a documented positive (DINO-DIAG-001); if a future base proves more
  stable under adaptation, DINO-logit (centering on) is worth re-testing there. Cheap DINO iteration
  now runs locally on the Mac/MPS ([[mac-dev-env]], docs/mac_dev_env.md).

DINO track at 1.1B: CLOSED. See [[dino-i2s-track]], [[base-anchored-pivot]].
