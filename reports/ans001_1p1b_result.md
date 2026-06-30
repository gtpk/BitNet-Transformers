# ANS-001 (answer-token-weighted CE) at TinyLlama-1.1B

Goal: TL1B-1600 showed facts are REACHABLE (gold_rank 375) but not EMITTED (fluent rambling, FACT
0.111 < content-KL 0.185). ANS-001 up-weights the first-k answer-span tokens (beta) on top of the
content-KL recipe (NO DINO) to pull the gold token out into a short answer. 160M smoke (beta 0 vs 4)
was harmless + slightly positive (FACT 0.000->0.074) but 160M first_token_hit was already saturated
(0.778, gold_rank=1) so it could not test the 1.1B readout regime -> went straight to 1.1B.

## beta=4 -- STUCK (the "Answer-Site Overweight Trap"), KILLED at step 450

| step | degen_gap | gold_rank | CE |
| ---: | ---: | ---: | ---: |
| 100 | +0.00 | 5784 | 7.51 |
| 200 | +0.80 | 6136 | 7.4 |
| 300 | +0.80 | 5956 | 7.4 |
| 400 | +0.80 | 5632 | 7.4 |
| 450 | +0.80 | 5095 | 7.3 |

beta=4 did NOT recover: degen_gap +0.80, gold_rank ~5000-6000, CE ~7.3-7.4 ALL FLAT for 250+ steps
(no progress). Unlike TL1B-1600/DINO (degenerate but gold_rank + CE kept MOVING -> worth waiting) and
unlike content-KL alone (CE -> ~4.1 by mid-run). **Killed at step 450** (negative evidence saved to
Drive metrics_beta4_stuck.jsonl). 160M (beta=4 harmless) did NOT predict this -- another
160M-mispredicts-1.1B case.

**Finding -- Answer-Site Overweight Trap:** a strong token-level loss on the answer site (beta=4)
breaks the 1.1B content-KL recovery/consolidation dynamics -- the model gets stuck in a degenerate
plateau instead of consolidating. answer-token weighting may still be a readout lever, but hitting
it hard directly destabilises 1.1B (consistent with the broader "1.1B adaptation is fragile to extra
training pressure" theme, though here it STALLS rather than collapses-into-salad).

## beta=1 -- gentler retry (RUNNING)

content-KL 0.2 + answer-token-weighted CE beta=1 k=3, NO DINO, 800 steps. OUT=bnt_results/ans001b1_1p1b.
Early-stop rule: if still stuck (CE>7, degen_gap +0.8, gold_rank flat) by step 250-300 -> kill ->
answer-token weighting is not the readout lever -> ANS-002 (short-answer Q/A format curriculum). If
CE<6 + gold_rank falling -> run to 800 + score FACT vs the 0.185 baseline. Verdict pending.
