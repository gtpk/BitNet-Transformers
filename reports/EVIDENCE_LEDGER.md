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

## G2. PT2-lite initializer smoke (no-train, 160M) -- ITF reconstruction is the wrong objective

| arm | weight_MSE | CE | FACT | gold_rank | report |
| --- | ---: | ---: | ---: | ---: | --- |
| E0 absmean (I2_S) | 0.282 | 11.64 | 0.0 | 1996 | pt2_i2s_001_result.md |
| E2 itf_pure (row-fit -> pure I2_S) | 0.215 | 12.10 | 0.0 | 5590 | (same) |
| E1 itf_asym (mu+alpha*T upper bound) | 0.191 | 11.64 | 0.0 | 3929 | (same) |

PT2-I2S-001: ITF lowers weight_MSE but gold_rank gets WORSE (anti-correlated) -> reconstruction is
the wrong objective; ITF-by-weight-MSE discarded. Motivates PT2-I2S-002 (activation-aware output-error).

## G. TinyLlama-1.1B longer-budget (1600 steps) -- reframe CONFIRMED, but fluent-not-factual

The DINO recipe that collapsed at 800 steps (DINO-002), re-run to 1600 with teacher-relative
telemetry. Report: reports/pythia_ladder/tl1b_long_metrics_summary.md (with sample generations).

| metric | value |
| --- | --- |
| generation stability | RECOVERED -- degen_gap ~0 sustained from step ~850, tags ok 27/27 (no salad/loop) |
| recovered_fraction / CE_adapted | 0.806 / 4.08 (~ content-KL baseline 0.845/4.10) |
| gold_rank (final) | 2006 (recovery onset) -> 375 (still far from FP teacher 3) |
| **FACT exact rate** | **0.111 (3/27) -- BELOW content-KL baseline 0.185** |
| sample gens | fluent + coherent but factually WRONG and off Q/A format (base-LM rambling) |

Two separate findings: (1) the 1.1B "collapse" under an auxiliary objective is a BUDGET-LIMITED
TRANSIENT, not a hard impossibility -- generation fully recovers by step ~850 (the 800-step DINO-002
failure was premature). (2) But DINO at 1.1B is NOT a factual win: FACT 0.111 < 0.185, and it trades
terse Q/A answering for fluent rambling -> a readout/answer-format decouple (gold_rank 375 reachable
internally, not emitted). content-KL 0.185 stays the v0 factual best. Next lever = answer-format /
answer-token-weighted objective or decoding, not just more budget; PT2-lite = transient-shortener.

## H. ANS readout track -- active

Motivation: TL1B-1600 recovered generation stability but stayed fact-poor. The final gold rank (375)
suggests the correct token is reachable internally but not emitted as a concise answer. ANS-001 tests
answer-token-weighted CE:

`L = CE_answer + 0.2 * content_KL + beta * CE_answer_token`, with `beta=4`, first `k=3` answer tokens.

| run | beta | FACT | gold_rank_mean | first_token_hit | CE | tags | interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ANS-001 160M baseline | 0 | 0.000 | 1 | 0.778 | 3.955 | ok26/empty1 | baseline |
| ANS-001 160M anstok | 4 | 0.074 | 1 | 0.778 | 3.972 | ok27 | directionally positive and harmless |
| ANS-001 1.1B | 4 | running | running | running | running | running | decisive readout test |

Caveat: 160M does **not** reproduce the 1.1B readout bottleneck because its first-token rank is already
saturated (`gold_rank_mean=1`). The 160M result is a safety/sanity signal, not a magnitude predictor.

## Reproduction

Each report names its driver flags. Pythia ladder + TinyLlama longer-budget: `rt116_quality_recovery.py`
with `--telemetry-full` (teacher-relative telemetry). Pythia needs the GPT-NeoX target-linear matcher
(commit 421e333) + n_target>0 guard. Controls: `colab/run_dino_ctrl_{adamw,fp32}.sh`. Ladder launch
configs: `colab/run_dino002.sh` / session cells. Datasets de-leaked vs the panel (never trained on the
panel). i2_s==f16 parity established in the RT-112..115 systems work.

## H. ANS-001 answer-token-weighted CE (1.1B) -- readout lever attempt

The TL1B-1600 limiter was readout (gold_rank 375 reachable but not emitted, fluent-rambling). ANS-001
up-weights the first-k answer tokens (beta) on content-KL (no DINO). reports/ans001_1p1b_result.md.

| arm | result | note |
| --- | --- | --- |
| 160M beta=0 vs 4 | FACT 0.000 -> 0.074, CE/tags clean | harmless + slight gain; but 160M first_token_hit saturated (0.778) so it can't test 1.1B readout |
| **1.1B beta=4** | **STUCK, killed @450** | CE 7.3 / gold_rank ~5000 / degen_gap +0.8 FLAT 250+ steps = "Answer-Site Overweight Trap" (strong token loss breaks 1.1B content-KL recovery); 160M did not predict |
| 1.1B beta=1 | STUCK, killed @300 | trapped IDENTICALLY to beta=4 (CE 7.4 / gold_rank ~6000 / degen +0.8 flat) -- mild beta traps as hard as strong => the weighted-CE recipe (any beta>0), not the magnitude, breaks 1.1B content-KL recovery |

ANS-001 verdict: answer-token weighting is NOT a working readout lever at 1.1B (beta 1 and 4 both
trap recovery). REFRAME: the fluent-rambling/readout-decouple was a DINO artifact -- content-KL ALONE
(FACT-003C) already answers in Q/A format at 0.185 (no rambling). So content-KL 0.185 stays the robust
1.1B factual best; EVERY lever tried (capacity, hard replay, blend, DINO, answer-token weighting) fails
to beat it. ANS-002 (more format data) is low value (content-KL already formats). More promising: a
stronger BASE (Qwen ladder, higher factual floor under the same recipe) or accept 0.185 + the
scientific findings (budget-limited transient, model-specific collapse).

## I. Qwen ladder (new axis: base-model quality) -- compatibility + baseline

TinyLlama-1.1B objective search closed (content-KL 0.185 ceiling; every lever -- capacity/replay/
blend/DINO/answer-token weighting -- failed to beat it). New question: does a stronger base lift the
ceiling under the SAME minimal content-KL I2_S recipe? Audit + baseline (Mac/MPS, raw Q:/A: panel,
same as TinyLlama):

| item | result |
| --- | --- |
| Qwen2.5-0.5B compatibility | PASS no code change -- Qwen is Llama-named (q/k/v/o/gate/up/down_proj), 168 target linears (24x7); q/k/v have BIAS (copied, weight-only ternary); tie_word_embeddings=True (lm_head frozen via embed exclude); forward finite |
| **Qwen2.5-0.5B-Instruct FP FACT** | **0.963 (26/27)** -- vs TinyLlama-1.1B FP ~0.81, at HALF the size (much higher factual floor) |
| Qwen-0.5B one-shot I2_S (no train) | FACT 0.000, CE 11.73, salad 27/27 (collapses like any one-shot ternary PTQ -- expected) |
| **Qwen-0.5B content-KL I2_S recovery** | **FACT 0.333 (9/27)** -- FACT-003C recipe (answer_token_weight=0.0, no DINO), 800 steps, 3080/bf16. CE 2.86(fp)->11.58(ptq)->4.08(adapted), **recovered_fraction 0.859**. degen_gap ~0 throughout (gold_rank 46997->~52, NO collapse -- unlike TinyLlama). gold_rank_mean 52, first_token_hit 0.222, tags 27/27 ok (no salad) |

### Verdict -- PARTIAL positive (does NOT yet name the rule)

Qwen-0.5B-I2_S content-KL FACT **0.333 > TinyLlama-1.1B's 0.185** (+0.148, +80% relative) at HALF the
params, with a CLEAN recovery (recovered_fraction 0.859, degen_gap ~0 the whole run -- the
objective-augmentation collapse that wrecked every TinyLlama-1.1B lever simply does not happen here).
This SUPPORTS the base-quality hypothesis: a stronger base does lift the I2_S factual ceiling under the
same minimal recipe.

BUT it is held BELOW the pre-registered 0.4 "confirm" bar, AND the sample gens still ramble: the hits
are substring matches embedded in hallucinated text ("The capital city of Rome, founded in 1840 by
Joseph Levy"), paris/moscow miss outright, first_token_hit only 0.222. So the readout (emit the short
fact crisply) is NOT solved -- same fluent-but-imprecise failure, just with a higher floor. Per the
pre-registered rule, "Base-Floor Transfer Rule" is NOT named yet (requires FACT clearly >0.185 with a
confirming trend). NEXT: Qwen-1.5B/1.7B same scorecard -- if FACT climbs with base quality (toward/past
0.4 with crisper gens), the rule is confirmed; if it plateaus near ~0.33, the base lifts the floor but
b1.58 still caps the readout. Files: reports/qwen_ladder/qwen05_ckl_{train,fact,metrics}.json(l).

### Qwen-1.5B rung (COLAB L4, bf16+adamw8bit+grad-ckpt, SAME minimal recipe) -- 'bigger Qwen = higher FACT' does NOT hold

Qwen2.5-1.5B-Instruct (FP teacher gold_rank 3.1, top1 0.898 -- even sharper than 0.5B's 0.769), same
content-KL FACT-003C recipe (lambda 0.2, answer_loss_only, answer_token_weight=0.0, no DINO, batch2/
accum12, mixed). 1.5B has a MUCH longer degenerate transient than 0.5B (degen +1.0 through step ~100,
cleared only by step 250 vs 0.5B's step 50) -> 800 steps UNDERTRAINS it. Scored at 800, then --resume to 1600.

| step | FACT | first_token_hit | gold_rank_mean | eval CE adapted | recovered | train_ce |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 800 | 0.111 (3/27) | 0.000 | 70.9 | 4.85 | 0.859 | 1.58 |
| 1600 | **0.222 (6/27)** | 0.111 | 68.2 | 5.68 | 0.749 | 0.37 |

**Verdict -- NEGATIVE for the clean base-scaling story; over-training confirmed.** Best 1.5B FACT
**0.222 (at 1600) < Qwen-0.5B's 0.333**, though both beat TinyLlama 0.185 (so Qwen base > TinyLlama
base still holds; bigger-Qwen-is-better does NOT). The 800->1600 resume LIFTED FACT (0.111->0.222,
fth 0.000->0.111) so 800 was genuinely undertrained -- BUT eval CE WORSENED with budget (4.85->5.68)
while train_ce crashed to 0.37 = the answer-CE-dominant minimal recipe OVER-TRAINS Qwen-1.5B (memorizes
the train stream, drifts off teacher: KL rose mid-run). Gens stay fluent-but-hallucinating ("capital of
Italy is Milan", "capital of France is France"), first_token_hit only 0.111. So "Base-Floor Transfer
Rule" is NOT named. Read: 0.5B-Instruct's 0.333 remains the best Qwen-I2_S point; 1.5B needs a RECIPE
RE-FIT (lower LR ~1e-4, higher lambda ~0.4-0.5 to anchor harder to teacher, fewer steps ~400-600, or
early-stop on held-out FACT not train_ce) before any base-scaling conclusion. Files:
reports/qwen_ladder/qwen15_ckl_{fact_800,fact_1600,train}.json.

### Qwen-1.5B RFIT sweep (recipe-fit A-D) -- late-DINO anti-overfit TIES 0.5B 0.333 (reports/qwen_ladder/qwen15_rfit_sweep.md)

Follow-up: is 0.222 a real 1.5B cap or a recipe mismatch? Sweep A->B->C->D (Colab L4, same content-KL
base, full-panel FACT at fact-eval steps for peak-hunting):

| arm | lr | λ | steps | extra | peak FACT |
| --- | ---: | ---: | ---: | --- | ---: |
| A | 1e-4 | 0.2 | 800 | -- | 0.111 |
| B | 1e-4 | 0.4 | 800 | -- | 0.148 |
| C | 5e-5 | 0.4 | 1000 | -- | 0.222 (STABLE all ckpts, no overfit decay) |
| **D** | 1e-4 | 0.4 | 800 | **late DINO** (logit w0.1 start300 warmup100) | **0.333** (traj 0.0->0.296->0.333) |

**Gentle lr alone (A/B) does NOT help; lower lr (C) only STABILIZES at 0.222; the late anti-overfit DINO
(D) breaks 0.222 -> 0.333, TYING Qwen-0.5B.** Validates DINO as an **Anti-Overfit Consistency Regularizer**
(weak/late/logit-only) -- distinct from the failed DINO-as-factual-lever. D disrupts on turn-on (@400=0.0)
then slows train drift so content-KL holds facts to 0.333. D gens genuinely better ("capital of Italy is
Rome"/"Germany is Berlin" crisp, vs earlier "Italy is Milan" hallucination); first_token_hit still low
(0.074). REFINES rung-2: bigger Qwen needs an anti-overfit regularizer to MATCH the smaller base (0.333),
not yet to EXCEED it. Open: does D's recipe push Qwen-1.7B/3B PAST 0.333, or is readout the cross-scale
wall? Files: reports/qwen_ladder/qwen15_rfit{A,B,C,D}_* + qwen15_rfit_sweep.md.
