# DINO-CONTROL-A (1.1B bf16 + plain adamw): COLLAPSED too -> optimizer is NOT the cause

Control to isolate the optimizer for the 1.1B DINO collapse. bf16 + **plain adamw** (vs DINO-002's
bf16 + adamw8bit); everything else identical to DINO-002 (dino-weight 0.2, lr 2e-4, no
centering/warmup). Run: colab/run_dino_ctrl_adamw.sh.

## Result -- COLLAPSED at step 300 (just later than adamw8bit)

```
[collapse-check] step 200 salad/empty/loop frac=0.00   <- clean here (misleading)
[collapse-check] step 300 salad/empty/loop frac=1.00   <- COLLAPSE DETECTED, stopped
recovered_fraction=0.483 (at stop)
```

Final score: eval_panel 0.037 (tags **loop 15 / ok 12**), popqa_tight 0.0 (loop 273/300), gold ranks
catastrophic vs FP teacher (cap_france 2724 vs 3, cap_russia 12610 vs 2, closest_planet 9190 vs 1).
Collapse mode is **loop**-dominated here vs **salad** for adamw8bit (DINO-002), but both are fully
degenerate with catastrophic gold ranks.

## Reading

Plain adamw was clean at step 200 (0.00) but fully collapsed by step 300. So switching off the 8-bit
optimizer **delays but does not prevent** the 1.1B collapse. **The optimizer (adamw8bit -> adamw) is
NOT the destabiliser.** This rules out one confound.

Status of the confound search for "160M works / 1.1B collapses":

| factor | tested? | verdict |
| --- | --- | --- |
| optimizer (adamw8bit vs adamw) | YES (this run) | NOT the cause (plain adamw also collapses) |
| precision (bf16 vs fp32) | next | all collapses so far are **bf16** -> prime suspect now |
| scale / base model | pending | the remaining hypothesis if precision is ruled out |

## Next: isolate PRECISION (fp32 + adamw8bit)

bf16 is now the common factor across every 1.1B collapse (DINO-002, DINO-003, Control-A). Test
fp32 + adamw8bit (fits ~13GB on the L4: fp32 model 4.4 + fp16 teacher 2.2 + adamw8bit 1.9 + fp32
grads 3.9). colab/run_dino_ctrl_fp32.sh. If fp32 stays stable -> **bf16 training was the
destabiliser**; if fp32 ALSO collapses -> **SCALE/base is the cause** -> Pythia ladder.
