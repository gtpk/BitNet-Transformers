# pythia-1b telemetry trajectory (teacher-relative) -- STABLE, recovers

Colab L4, bf16+adamw, 800 steps. Full per-step jsonl on Drive: bnt_results/p1b/metrics.jsonl
(--ckpt-every-min 10). teacher baseline: degen 0.00, gold_rank 4.7, top1 0.239.

| step | degen_gap | gold_rank | gold_rank_ratio | top1 | logit_entropy | hidden_var_mid | train_ce |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +1.00 | 6069 | 1291 | 0.155 | 5.01 | 1.9 | 9.63 |
| 100 | +1.00 | 5472 | 1164 | 0.033 | 7.53 | 30.6 | 7.82 |
| 200 | +1.00 | 5480 | 1166 | 0.045 | 7.75 | 45.3 | 7.09 |
| 225 | **+0.00** | 4638 | 987 | 0.116 | 7.56 | 33.4 | 7.00 |
| 250 | +0.00 | 3251 | 692 | 0.079 | 7.27 | 18.9 | 6.51 |
| 275 | +0.10 | 844 | 180 | 0.070 | 7.59 | 7.7 | 6.10 |
| 300 | +0.00 | 634 | 135 | 0.186 | 6.29 | 4.0 | 5.61 |
| 400 | +0.00 | 400 | 85 | 0.320 | 4.54 | 1.7 | 3.79 |
| 500 | +0.00 | 195 | 42 | 0.413 | 3.64 | 1.5 | 3.30 |
| 600 | +0.00 | 203 | 43 | 0.505 | 3.12 | 1.8 | 2.95 |
| 700 | +0.00 | 121 | 26 | 0.366 | 3.60 | 1.8 | 2.90 |
| 750 | +0.00 | 195 | 42 | 0.297 | 4.14 | 1.8 | 2.75 |

Signature of the transient -> consolidation: degen_gap +1.00 through step ~200 then -> 0 by step 225;
gold_rank stuck ~5000-6000 then falls sharply (step 275 onward) to ~120-200 (ratio ~20-40x teacher =
the b1.58 quantized factual ceiling); **hidden_var_mid BLOWS UP during the transient (30-55) then
collapses to ~1.7 on consolidation** -- a clean mid-layer-variance signature of the unstable phase.
top1 rises 0.04 -> 0.3-0.5, entropy falls 7.7 -> 3-4, train_ce 9.6 -> 2.75. STABLE: the transient
resolves well within the 800-step budget. (Same shape as 160m/410m; TinyLlama-1.1B never resolved it.)
