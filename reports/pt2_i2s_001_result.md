# PT2-I2S-001 (ITF row-grid, no-train init): RECONSTRUCTION-ONLY -> ITF alone not useful

No-train initialization smoke (160M Felladrin/Llama-160M-Chat-v1, per-row iterative ternary fit;
plan docs/pt2_i2s_initializer_plan.md). Tests whether a better ternary code T (PT2-style asymmetric
mu+alpha*T row fit) helps downstream behaviour and survives projection to pure one-plane I2_S.

| arm | weight_MSE (rel) | CE | FACT eval_panel | gold_rank | degenerate | tags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp (reference) | 0 | 3.12 | 0.296 | 26 | 0.26 | ok20/salad1/empty6 |
| E0 absmean (current I2_S) | 0.282 | 11.64 | 0.0 | 1996 | 0.04 | ok26 |
| E2 itf_pure (row-fit T -> per-tensor gamma_proj*T) | **0.215** | 12.10 | 0.0 | **5590** | 0.0 | ok27 |
| E1 itf_asym (mu_r + alpha_r*T, upper bound) | **0.191** | 11.64 | 0.0 | 3929 | 0.04 | ok26/loop1 |

## Verdict: RECONSTRUCTION-ONLY (discard ITF-by-weight-MSE)

ITF lowers weight reconstruction error (0.282 -> 0.215 pure, 0.191 asym) but does **NOT** improve
behaviour -- FACT stays 0.0 (one-shot PTQ, all arms collapsed), and **gold_rank gets WORSE** with the
lower-MSE code (absmean 1996 -> itf_pure 5590, itf_asym 3929). Per the plan's pass-signal table this
is "neither E1 nor E2 improves behaviour -> ITF alone not useful", and per the decision table "only
reconstruction improves -> WSYNC-style dead branch".

**The important lesson: lower weight-MSE is the WRONG objective -- it is anti-correlated with gold_rank
here.** That is exactly the gap PT2-LLM closes with ACTIVATION-AWARE grid alignment (minimise output
error ||(W-Wq)X||, not ||W-Wq||). So PT2-I2S-001 does not kill PT2; it motivates **PT2-I2S-002**
(activation-aware alpha-only, frozen T) as the decisive PC smoke.

Caveat: init-behaviour is measured on the un-adapted PTQ model (CE ~11-12, FACT 0 for all), so the
gold_rank deltas are on a collapsed model and are directional, not definitive. The plan's PT2-I2S-005
(does a better init shorten the adaptation transient?) remains the real downstream test -- but only
worth paying for if an activation-aware init (002) first moves behaviour on the PC.
