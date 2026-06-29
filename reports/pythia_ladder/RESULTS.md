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
| pythia-410m | 3080 bf16 (auth) + Mac fp32 (aux) | **NONE (transient 50-250, recovers)** | 1.00 (transient) | 0.894 | 19631 -> 128 | **STABLE (slow consolidation)** |
| pythia-1b | (pending, Colab) | | | | | |
| pythia-1.4b | (pending, Colab) | | | | | |
| pythia-2.8b | (pending, Colab/A100) | | | | | |

## pythia-160m (rung 1) -- STABLE

No collapse over 800 steps (degenerate_rate 0.00 throughout). The DINO objective works: gold_rank
8248 (step 0, untrained ternary) -> 166 (step 200) -> 272 (end); grad_norm settles ~4.4; entropy
~4.5, top1 ~0.29 (no spike to 1.0); hidden_var_last stable ~95. recovered_fraction 0.962. This is the
expected baseline rung: 160M is below any collapse threshold and the auxiliary objective is benign/
positive -- matching the TinyLlama-160M DINO-DIAG-001 positive. metrics: reports/pythia_ladder/p160m_metrics.jsonl.

## pythia-410m (rung 2) -- STABLE (recoverable transient) -- the key dynamics result

410M passes through a **degenerate transient (steps ~50-250: degen_gap +1.00 vs the clean FP
teacher; gold_rank stuck ~5000-8000) then RECOVERS** (degen_gap -> 0 from ~step 275; gold_rank_ratio
967 -> ~6-21; top1 rises 0.05->0.27, entropy falls 8->4.2, train_ce falls 9.9->3.7). Final (Mac fp32,
full 800): degen 0, gold_rank 128, recovered_fraction 0.894. Cross-validated across precision: 3080
bf16 (authoritative, teacher-relative) and Mac fp32 (aux) show the SAME trajectory -> the transient is
scale-driven, not a precision artifact. **3080 authoritative final: degen_gap +0.00 (fully recovered,
clean like teacher), gold_rank_ratio 967 -> 22 (big improvement, plateaus ~20x teacher = the b1.58
quantized-student factual ceiling, NOT collapse), recovered_fraction 0.898.**

160M had NO transient; 410M has one but resolves it within the 800-step budget; TinyLlama-1.1B never
resolved it (collapse). **Emerging scale-law: transient length grows with scale; collapse = the
degenerate transient is not resolved within the training budget** (a recoverable schedule problem,
not necessarily a hard capacity wall -> prescriptions: longer training, DINO warmup delay, lower LR,
entropy/top1 guard, stage-wise objective). Onset so far: > 410M. Next: pythia-1b -- key diagnostic
"is 1b's step 800 like 410m's step 250 (mid-transient, about to recover)?". metrics:
p410m_metrics.jsonl (Mac fp32, full 800) + p410m_cuda_metrics.jsonl (3080 bf16, teacher-relative).
