# Evidence Ledger -- base-anchored b1.58 / I2_S factual recovery

Single source of the **actual measured values** for every experiment in this line, with pointers to
the raw data. Narrative/interpretation lives in [docs/i2s_same_topology_findings.md](../docs/i2s_same_topology_findings.md)
and [docs/pythia_ladder_runbook.md](../docs/pythia_ladder_runbook.md); this file is the numbers +
provenance. Panel = `data/factual_panel_v1.jsonl` (27 held-out prompts; FP ref 0.81, Q2_K 0.74).

## How the evidence is preserved

- **Per-experiment reports** (`reports/*.md`): the committed record of each run's numbers + verdict.
- **Raw per-step telemetry** (`reports/pythia_ladder/*_metrics.jsonl`): in-repo for pythia-160m/410m
  (full 800-step jsonl, 33-34 records each); pythia-1b full jsonl on Drive `bnt_results/p1b/metrics.jsonl`
  (in-repo summary: `p1b_metrics_summary.md`).
- **Drive archive** (`MyDrive/bnt_results/<run>/`): metrics.jsonl, run.log, pyscore.json, train.json,
  adapted_model (large models were pruned during the 2026-06-28 Drive cleanup; the small json/log
  records were kept). **Drive ckpt** (`MyDrive/bnt_ckpt/<run>/ckpt.pt`) only for resumable runs.
- **Code provenance**: trainer `scripts/rt116_quality_recovery.py`; scorers `scripts/fact004a_160m_smoke.py`
  (score_dir), `scripts/score_dino002.py`, `scripts/dino_diag001_token_analysis.py`; conversion
  `bitnet_llama/conversion.py` (+ `module.py` PerTensorBitLinear). All results pushed to origin/main.

---

## A. TinyLlama-1.1B factual-recovery ladder (content-KL is the lever; ceiling 0.185)

| exp | recipe change | factual panel | recovered_fraction | tags | report |
| --- | --- | ---: | ---: | --- | --- |
| FACT-002 | data swap instruction / mixed | 0.00 / 0.07 | - | empty-collapse | rt131_fact002_summary.md |
| FACT-003A | + answer-only CE | 0.15 | - | fixes empty | rt132_fact003a_*_fact.md |
| FACT-003B | + naive base-KL lambda 1.0 | 0.00 | 0.47 | empty (chat-teacher early-EOS) | rt133_fact003b_mixed_kl1_fact.md |
| **FACT-003C** | **+ content-KL (drop EOS) lambda 0.2** | **0.185** | **0.845** | ok 27/27, i2_s==f16 | rt134_fact003c_mixed_ckl0.2_fact.md |
| FACT-003C lambda 0.1 | weaker | 0.037 | - | salad (CE stuck) | rt134_fact003c_mixed_ckl0.1_fact.md |
| FACT-003C lambda 0.5 | stronger | 0.037 | - | ok (over-anchored) | rt134_fact003c_mixed_ckl0.5_fact.md |
| FACT-004A | + unfreeze lm_head | 0.04 / 0.00 | 0.806 | fluent, facts washed | rt135_fact004a_*_fact.md |
| HYBRID-001A | post-hoc FP-restore late layers | 0.148 (A0) worse A1-A5 | - | co-adaptation breaks | hybrid001a_capacity_probe.md |

Inverted-U on lambda (0.037 / **0.185** / 0.037). content-KL 0.2 = the locked **v0** recipe.

## B. Objective-augmentation wall at TinyLlama-1.1B (every add-on collapses)

| exp | objective on top of content-KL | panel / popqa | recovered | failure mode | report |
| --- | --- | --- | ---: | --- | --- |
| FACT-003D mu=1.0 | hard atomic-fact replay | eval 0.111, train_atomic 1.0 | 0.831 | memorise (overfit) | rt136_fact003d_1p1b_result.md |
| FACT-003D mu=0.25 | gentler hard replay | eval 0.037 | 0.783 | still net-negative | rt137_fact003d_mu025_result.md |
| FACT-003H v1 | PopQA blend 5% | eval 0.0, all panels 0.0 | 0.474 | all-loops degenerate | rt138_fact003h_1p1b_result.md |
| FACT-003H v2 (seed1) | PopQA blend 5% | eval 0.037, popqa_tight 0.013 | 0.706 | salad (8/27) | (session; Drive fact003h_..._v2) |
| DINO-002 | DINO-logit 0.2 | eval 0.111 salad 26/27, popqa 0 | 0.620 | salad; gold ranks 431-4089 vs teacher 1-7 | dino_i2s_002_1p1b_result.md |
| DINO-003 | DINO-logit + centering+warmup+lr1e-4 | COLLAPSE step 200, salad 0.90 | 0.474 | stabilisation FAILED | dino_i2s_003_1p1b_result.md |

## C. DINO is mechanistically valid at 160M (rules out "bad objective")

| exp | measure | value | report |
| --- | --- | --- | --- |
| DINO-001 (160M, seed41) | baseline eval / dino_logit eval | 0.074 -> **0.222 (+0.148)** | dino_i2s_001_160m_result.md |
| DINO-001 dino_hidden | hidden-align cancels the gain | 0.222 -> 0.074 | (same) |
| DINO-DIAG-001 (160M, seeds avg) | mean delta log P(gold) | **+0.372**, gold rank up on 78% | dino_diag001_result.md |
| DINO-DIAG-001 category | simple_fact / reasoning / entity_attr | +0.574 / +0.112 / -0.04 | (same) |
| DINO-001b sweep | lambda 0.1 / 0.2 / 0.4 dEval (3 seeds) | [0,+.074,+.074] / [+.037,+.037,-.037] / breaks | (session; reports/dino_i2s_001b_sweep on box) |

Lever = preserve FP teacher CONTENT OUTPUT distribution; replicating FP HIDDEN geometry hurts.

## D. Confound controls -- the 1.1B collapse is NOT precision/optimizer (it is model/scale dynamics)

| control | change vs DINO-002 (bf16+adamw8bit) | result | report |
| --- | --- | --- | --- |
| CONTROL-A | bf16 + **plain adamw** | COLLAPSE step 300 (eval 0.037, loop 273/300) | dino_ctrl_adamw_result.md |
| CONTROL-B | **fp32** + adamw8bit | COLLAPSE step 200, frac 1.00 | dino_ctrl_fp32_result.md |

Optimizer AND precision both ruled out -> not a numerical artifact.

## E. Pythia ladder (same recipe, varying scale) -- NO generic 1B scale wall

Recipe: content-KL 0.2 + DINO-logit 0.2, no centering/warmup, 800 steps, teacher-relative telemetry.
Collapse judged RELATIVE to each model's own FP teacher (degen_gap, gold_rank_ratio).

| rung | teacher gold_rank | transient | recovers? | final gold_rank | recovered_fraction | verdict | raw data |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| pythia-160m | 390 | none | n/a | 272 | 0.962 | STABLE | p160m_metrics.jsonl (in-repo) |
| pythia-410m | 20 | step 50-250 | yes ~275 | 128 | 0.894 (fp32) / 0.898 (bf16) | STABLE | p410m_metrics.jsonl + p410m_cuda_metrics.jsonl (in-repo) |
| pythia-1b | 5 | step 0-250 | yes ~225-300 | ~150 | ~high (train_ce 9.6->2.75) | STABLE | p1b_metrics_summary.md (in-repo); full jsonl on Drive |
| TinyLlama-1.1B | (chat) | unresolved @800 | NO (at 800 steps) | catastrophic | 0.474-0.706 | COLLAPSE | DINO-002/003 reports |

Telemetry marker: `hidden_var_mid` blows up during the transient (pythia-1b: 30-55) then collapses
to ~1.7 on consolidation -- a clean signature of the unstable phase resolving.

## F. Capacity / geometry track -- comprehensively NEGATIVE at 160M (cost ledger RDT-001)

| branch | best vs baseline | verdict | report |
| --- | --- | --- | --- |
| WSYNC scaling / H-I2S (data-free) | fact 0.0 (ternary collapse) | FAIL | rt_wsync_160m*.md |
| SIDE-001 I2_S+LoRA sidecar | eval 0.185->0.222(r2, sub-threshold) | no clear lever | side001_160m.md |
| EGROW-002 top-k vs random-k sidecar | top-k -0.037 vs random | localization not a lever | egrow002_160m.md |
| SIGMA-001 residual feedback | fact 0.037 (collapsed regime) | FAIL | sigma001_residual_feedback.md |
| RHT-002 dithered Hadamard | fact 0.0 | FAIL | rht002_dithered_reference.md |
| HOME-001 activation homeostasis | eval flat 0.111 | no signal | home001_160m/summary.md |

Cost ledger verdict (rdt001_cost_ledger.md): NO low-cost branch buys >=0.05 eval.

## G. IN PROGRESS (2026-06-30)

- **TinyLlama-1.1B longer-budget (1600 steps)** -- the decisive test of whether the 1.1B collapse is a
  recoverable transient (budget-limited) or a hard model-specific collapse. Drive: bnt_results/tl1b_long
  (metrics.jsonl + run.log), ckpt bnt_ckpt/tl1b_long. telemetry-full, probe every 50, no early-stop,
  --ckpt-every-min 10 --resume. Verdict pending -> will be appended here + to pythia_ladder/RESULTS.md.

## Reproduction

Each report names its driver flags. Pythia ladder + TinyLlama longer-budget: `rt116_quality_recovery.py`
with `--telemetry-full` (teacher-relative telemetry). Pythia needs the GPT-NeoX target-linear matcher
(commit 421e333) + n_target>0 guard. Controls: `colab/run_dino_ctrl_{adamw,fp32}.sh`. Ladder launch
configs: `colab/run_dino002.sh` / session cells. Datasets de-leaked vs the panel (never trained on the
panel). i2_s==f16 parity established in the RT-112..115 systems work.
