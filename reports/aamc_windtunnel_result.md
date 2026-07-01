# AAMC overfit wind-tunnel (Qwen-0.5B, RTX 3080) — controller VALIDATED

Purpose (docs/adaptive_anchor_manifold_controller_plan.md): NOT a final-quality claim. A deliberately
overfit-prone regime (lr 3e-4, 1000 steps) to test whether the AAMC controller REACTS to overfit —
raises lambda (teacher anchor) — and whether that dynamic schedule beats fixed lambda. 4 arms, sequential
on one GPU, same content-KL FACT-003C base (answer_token_weight=0), fact-eval + aamc-score every 200,
aamc-min-step 300 (observe-only during the early PTQ transient).

## Full-panel FACT trajectory (fact-eval @400/600/800/1000)

| arm | policy | @400 | @600 | @800 | **@1000** | lambda trajectory | alpha |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| fixed020 | lambda 0.2 fixed | 0.037 | 0.074 | **0.259** | 0.111 | 0.2 | 0 |
| fixed040 | lambda 0.4 fixed | 0.148 | 0.111 | **0.222** | 0.148 | 0.4 | 0 |
| dynlam | AAMC dynamic-lambda | 0.037 | 0.222 | 0.185 | **0.296** | 0.2→0.3→0.4→0.5 | 0 |
| dyndino | AAMC + conditional DINO | 0.037 | 0.222 | 0.185 | **0.296** | 0.2→0.3→0.4→0.5 | 0 |

Both fixed arms PEAK at step 800 then OVERFIT-DECAY by 1000 (0.259→0.111, 0.222→0.148). The dynamic-lambda
arms instead END at 0.296 — their highest point — because the controller kept raising the teacher anchor
as overfit built.

## Controller decisions (dynlam/dyndino aamc.jsonl)

| step | train_ce | eval_ce | FACT | entropy | top1 | overfit_score | collapse | action | lambda | alpha |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 200 | 4.47 | 5.22 | 0.037 | 5.47 | 0.23 | 0 | 0 | warmup-hold | 0.2 | 0 |
| 400 | 2.95 | 5.06 | 0.037 | 3.82 | 0.32 | 4 | 0 | overfit→raise lambda | 0.2→0.3 | 0 |
| 600 | 2.18 | 5.43 | 0.222 | 3.22 | 0.37 | 4 | 0 | overfit→raise lambda | 0.3→0.4 | 0 |
| 800 | 1.35 | 5.58 | 0.185 | 2.42 | 0.56 | 5 | 0 | overfit→raise lambda | 0.4→0.5 | 0 |
| 1000 | 1.00 | 5.80 | 0.296 | 2.12 | 0.64 | 4 | 0 | (lambda at max 0.5) | 0.5 | 0 |

## Verdict — AAMC controller VALIDATED (mechanics + quality)

1. **Warmup-hold**: at step 200 (< aamc-min-step 300) the controller correctly took NO action during the
   normal early PTQ transient (action=warmup-hold), so it does not react to the expected startup degen.
2. **Reacts to overfit**: from step 400 the overfit pattern (train_ce falling, eval_ce not improving, FACT
   flat, entropy down, top1 overconfident) scored overfit_score 4–5, and the controller raised lambda
   0.2→0.3→0.4→0.5 — a pattern of conditions, not one scalar, exactly per Policy V0.
3. **Dynamic beats fixed**: dynlam/dyndino FACT@1000 = **0.296**, DECISIVELY above both fixed arms' decayed
   endpoints (fixed020 0.111, fixed040 0.148). Raising the teacher anchor as overfit built prevented the
   fixed arms' 800→1000 collapse. (Note: @800 the dynamic arms dipped to 0.185 below fixed020's 0.259 —
   the lambda-raise is not instantaneous — but by 1000 the anchor paid off.)
4. **Conditional DINO correct**: collapse_score stayed 0 the whole run (degen_gap 0, no salad/loop), so the
   dyndino arm's alpha NEVER rose — DINO stayed off exactly as designed (fires only on collapse). dyndino
   was therefore identical to dynlam here, confirming the alpha path is collapse-gated, not overfit-gated.

Allowed claim (per plan's claim-discipline): *fixed objective weights fail differently, and a
telemetry-driven controller that raises lambda on an overfit pattern beats fixed lambda in an overfit
regime; DINO stays off unless collapse appears.* This is one model/rung; not yet "AAMC solves factual
recovery." Files: reports/aamc_wt/{fixed020,fixed040,dynlam,dyndino}_{train.json,*.facteval.jsonl,*.aamc.jsonl}.
