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

## beta=1 -- gentler retry: ALSO STUCK, killed at step 300

| step | degen_gap | gold_rank | CE |
| ---: | ---: | ---: | ---: |
| 150 | +0.8 | 6115 | 7.4 |
| 200 | +0.8 | 6042 | 7.4 |
| 250 | +0.8 | 5758 | 7.2 |
| 300 | +0.8 | 6150 | 7.4 |

beta=1 trapped IDENTICALLY to beta=4: CE ~7.4 flat, gold_rank ~6000 flat, degen_gap +0.8 -- never
enters the content-KL recovery (which reaches CE ~4.1). Killed at step 300 per the early-stop rule
(metrics_beta1_stuck.jsonl preserved).

**Key caveat:** beta=1 (mild, weight 2x on first-3 answer tokens) traps just as hard as beta=4
(strong). If it were a weight-magnitude effect, beta=1 should trap less. Identical trapping suggests
the answer-token-weighted-CE PATH itself (any beta>0) -- not the weight value -- breaks the 1.1B
content-KL recovery dynamics. So the honest claim is "this weighted-CE-on-content-KL recipe traps
1.1B recovery", not necessarily "all answer-token weighting is dead".

## ANS-001 verdict + reframe

answer-token weighting (the ANS-001 weighted-CE) is NOT a working readout lever at 1.1B -- it traps
recovery (beta 1 and 4 both). Recommend NOT chasing the beta sweep further.

**Bigger-picture reframe:** the "fluent rambling / readout decouple" that motivated ANS-001 was a
DINO artifact (TL1B-1600 had it because DINO's unlabeled-content-KL drifts the format). **content-KL
ALONE (FACT-003C) already answers in Q/A format at 0.185 (ok 27/27) -- it does NOT ramble.** So the
readout "bottleneck" was specific to the DINO-recovered model, not content-KL. ANS-001 tried to push
content-KL's 0.185 higher via answer emphasis and failed (traps).

Net: content-KL 0.185 remains the robust 1.1B factual best; every lever tried to beat it (capacity,
hard replay, blend, DINO, answer-token weighting) has failed. ANS-002 (more Q/A format data) is low
value because content-KL already formats correctly. The more promising directions are a stronger
BASE model (Qwen ladder -- higher factual floor under the same minimal content-KL recipe) or
accepting 0.185 + the scientific findings (budget-limited transient, model-specific collapse).
