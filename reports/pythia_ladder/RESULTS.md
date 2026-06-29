# PYTHIA-LADDER-001 results (running)

Same recipe (content-KL 0.2 + DINO-logit 0.2, no centering/warmup, all target linears b1.58 I2_S,
lm_head/embeds frozen), same data, 800 steps, telemetry every 25 steps. Per the runbook
([docs/pythia_ladder_runbook.md](../../docs/pythia_ladder_runbook.md)). Collapse onset = first step
with degenerate_rate (panel salad/empty/loop) > 0.3 sustained 3 checks. Primary signal = degeneracy
+ gold_rank vs the model's own FP teacher (Pythia is BASE, absolute fact_rate low by design).

NOTE on precision/optimizer: Controls A/B already proved precision (bf16/fp32) and optimizer
(adamw/adamw8bit) do NOT affect the 1.1B collapse, so rungs may use different host/precision without
reintroducing a confound. Recorded per rung below.

| scale | host / precision | collapse_onset_step | max degen | recovered | gold_rank (0 -> end) | verdict |
| --- | --- | --- | ---: | ---: | --- | --- |
| pythia-160m | Mac MPS / fp32 adamw | **NONE** | 0.00 | 0.962 | 8248 -> 272 | **STABLE, DINO positive** |
| pythia-410m | (running) | | | | | |
| pythia-1b | (pending, Colab) | | | | | |
| pythia-1.4b | (pending, Colab) | | | | | |
| pythia-2.8b | (pending, Colab/A100) | | | | | |

## pythia-160m (rung 1) -- STABLE

No collapse over 800 steps (degenerate_rate 0.00 throughout). The DINO objective works: gold_rank
8248 (step 0, untrained ternary) -> 166 (step 200) -> 272 (end); grad_norm settles ~4.4; entropy
~4.5, top1 ~0.29 (no spike to 1.0); hidden_var_last stable ~95. recovered_fraction 0.962. This is the
expected baseline rung: 160M is below any collapse threshold and the auxiliary objective is benign/
positive -- matching the TinyLlama-160M DINO-DIAG-001 positive. metrics: reports/pythia_ladder/p160m_metrics.jsonl.
