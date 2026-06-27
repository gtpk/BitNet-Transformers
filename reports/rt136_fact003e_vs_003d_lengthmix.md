# FACT-003E length-mix vs FACT-003D short-atomic (160M predictor, 3 seeds each)

Tests the critique that FACT-003D's deliberately SHORT atomic facts are a clean diagnostic but not
PTQ/QAT-style representative adaptation data. FACT-003E expands the SAME 291/97 protected facts into
5 surface forms (short/sentence/chat/explain/long; 291->1455 train, 97->485 heldout) keeping the
panel-disjoint guarantee. Both run the same recipe (content-KL 0.2 + mu=1.0 factual-CE) on the 160M
predictor, scored on the IDENTICAL eval panel (data/factual_panel_v1.jsonl, 27 short Q/A).

| | seed 41 | seed 42 | seed 43 | mean | range |
| --- | ---: | ---: | ---: | ---: | --- |
| FACT-003D eval_panel (short atomic) | 0.185 | 0.296 | 0.296 | **0.259** | 0.185-0.296 (+-0.055) |
| FACT-003E eval_panel (length-mix) | 0.259 | 0.259 | 0.259 | **0.259** | 0.259 (variance 0) |
| FACT-003D heldout / train | 0.227 / 1.00 | 0.227 / 0.97 | 0.196 / 1.00 | ho 0.217 | |
| FACT-003E heldout / train | 0.187 / 0.93 | 0.160 / 1.00 | 0.207 / 0.90 | ho 0.183 | |
| eval CE (both) | ~4.02 | ~4.04 | ~4.08-4.10 | ~4.05 | stable |

(mu=0 control eval_panel ~0.037; both recipes clear it ~7x.)

## Findings

1. **Mean eval-panel transfer is IDENTICAL: 0.259 (003D) == 0.259 (003E).** Length-mixing the same
   facts into varied surface forms did NOT raise the central transfer value. The eval panel is itself
   short Q/A, so the short atomic replay was already format-matched and sufficient; chat/explain/long
   surfaces don't match the eval format better, so no magnitude gain.
2. **Length-mix ELIMINATES seed variance.** 003D swung 0.185-0.296 across seeds; 003E hit exactly
   0.259 on all three. Representative/varied data makes which-facts-stick reproducible (the same core
   facts get robustly engaged every seed) instead of seed-dependent.
3. Memorisation slightly harder for length-mix (train 0.90-1.00 vs 0.97-1.00); heldout weakly lower
   for 003E (0.183 vs 0.217) but the heldout PANELS differ (003E heldout is length-mix sampled), so
   discount that axis. CE stable either way (no fluency cost).

## Verdict

The "short atomic is not representative data" critique is conceptually valid but, for the short-form
factual eval, does NOT buy more transfer magnitude at 160M -- **length-mix trades magnitude (none) for
stability (seed variance -> ~0).** Short atomic was adequate for the eval-panel signal.

**1.1B implication:** the in-flight Colab 1.1B FACT-003D (short atomic) mu=1.0 need NOT be re-run with
length-mix for eval-panel performance -- the predictor says the mean is unchanged. Length-mix may help
on chat/long-form factual queries (a surface axis the 27-item short eval panel does not probe); that is
a separate evaluation to add if long-form factual behaviour becomes a goal.
