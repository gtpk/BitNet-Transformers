# DINO-I2S-001 result (160M): logit self-distillation is the first positive objective

Driver: scripts/dino_i2s_selfdistill_smoke.py  (model Felladrin/Llama-160M-Chat-v1, steps 400,
view dropout@0.1, dino_logit_w 0.2, hidden_w 0.01, seed 41). Plan: docs/dino_i2s_self_distillation_plan.md.
Context: docs/i2s_v0_recipe_and_closed_branches.md (DINO is the one remaining open mechanism after
the CE/replay objective family was confirmed exhausted at 1.1B by FACT-003H v1+v2).

## Result

| arm | eval_panel | popqa_tight | popqa_train (memorise) | CE | dEval | tags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline (content-KL 0.2 replay) | 0.074 | 0.01 | 0.0 | 3.849 | - | ok 26, loop 1 |
| **dino_logit** (+ unlabeled-view content-KL) | **0.222** | 0.0 | 0.0 | 3.773 | **+0.148** | ok 27 |
| dino_hidden (dino_logit + hidden align) | 0.074 | 0.0 | 0.0 | 3.791 | +0.0 | ok 27 |

VERDICT: PASS -- dino_logit lifts eval_panel +0.148 (3x, 2/27 -> 6/27) over baseline with clean
tags (ok 27/27) and NO train-only memorisation signature (train 0.0). First cheap 160M mechanism
to clear the +0.05 bar since the capacity/geometry track went comprehensively negative.

## The failure decomposition (the important part)

dino_hidden = dino_logit + normalized hidden alignment (mid+last). Adding the hidden term
**cancelled the entire logit gain (0.222 -> 0.074, back to baseline).** So:

- **Output-distribution preservation IS the lever.** Following the FP teacher's content output
  distribution on broad UNLABELED text recovers factual behaviour (+0.148).
- **Internal-representation preservation is NOT, and actively HURTS.** Forcing the I2_S student
  to match FP hidden geometry overconstrains it and wipes out the logit gain. Read: "replicate
  FP internals" is the wrong target; "preserve FP content output distribution" is the right one.

This also distinguishes DINO from the dead branches: train_replay 0.0 = no memorisation shortcut
(unlike FACT-003D hard replay train=1.0), and the gain is on the DISJOINT eval panel, not the
training set -- a retention signal, not an overfit. Structurally less prone to the overfit that
killed FACT-003D/H.

## Caveats

- **27-prompt panel**: 0.222 = 6/27, baseline = 2/27 (+4 items). Meaningful but small N -> needs
  a seed/lambda robustness pass before escalating (DINO-I2S-001b).
- **160M does not reliably predict 1.1B** (FACT-003D's 160M predictor mispredicted the 1.1B
  outcome). This 0.222 is a MECHANISM signal; the 1.1B gate is the real test.
- popqa_tight ~0 for all arms: 160M barely does PopQA; the signal lives in eval_panel here.

## Next (DINO-I2S-001b robustness, then 1.1B gate)

B. 160M dino_logit-ONLY (hidden discarded) robustness: seeds 41/42/43 x lambda 0.1/0.2/0.4.
C. If >=2/3 seeds clear baseline +0.05 -> DINO-I2S-002 1.1B Colab gate.
   1.1B PASS: eval_panel > 0.185 (ideally >=0.25), tags ok, i2_s ~= f16, no train-only memorise.
   1.1B FAIL: CE-only improvement / tags collapse / 160M effect vanishes at 1.1B.
