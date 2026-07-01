# Qwen-1.5B RFIT sweep (recipe-fit) — late-DINO anti-overfit ties Qwen-0.5B 0.333

Question (user-directed): the minimal 2e-4 recipe OVER-TRAINS Qwen-1.5B (800 undertrain 0.111,
1600 overfit 0.222). Does a gentler/regularized recipe let 1.5B beat Qwen-0.5B's FACT 0.333, or is
0.222 a real cap? Sweep A->B->C->D on Colab L4 (bf16+adamw8bit+grad-ckpt), same content-KL FACT-003C
base (answer_token_weight=0, mixed, batch2/accum12), full-panel FACT at fact-eval steps (peak-hunting).

| arm | lr | λ | steps | extra | **peak FACT** | fth | recovered | ce_adapted |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| A | 1e-4 | 0.2 | 800 | — | 0.111 @800 | 0.0 | 0.820 | 4.78 |
| B | 1e-4 | 0.4 | 800 | — | 0.148 @600 | 0.185@400 | 0.829 | 4.67 |
| C | 5e-5 | 0.4 | 1000 | — | 0.222 (STABLE @400/600/800/1000) | ~0.1 | 0.851 | 4.39 |
| **D** | 1e-4 | 0.4 | 800 | **late DINO** (logit w0.1, start300, warmup100) | **0.333 @800** | 0.074 | 0.830 | 4.66 |

FACT trajectories (fact-eval @400/600/800[/1000]):
- A: 0.037 / 0.037 / 0.111
- B: 0.111 / 0.148 / 0.148
- C: 0.222 / 0.222 / 0.222 / 0.222  (lowest lr = perfectly stable, no overfit decay)
- **D: 0.0 / 0.296 / 0.333**  (DINO turns on at step 300, disrupts @400, then anti-overfit lift 0.296->0.333)

## Verdict — late-DINO anti-overfit is a WIN at 1.5B; 1.5B now TIES 0.5B 0.333

The arc is the finding:
- **gentle lr alone (A/B) does NOT help** — lr1e-4 just slower, caps 0.111-0.148, below prior 2e-4@1600's 0.222.
- **lower lr (C, 5e-5) STABILIZES at 0.222** — kills the overfit decay (all four checkpoints 0.222) but does not raise the ceiling; ties prior 1.5B best in fewer steps.
- **late DINO (D) is the lever that breaks 0.222 -> 0.333** — the weak logit-DINO applied only after step 300 (anti-overfit consistency regularizer, NOT a factual booster) lifts FACT past every fixed-λ arm to 0.333, TYING Qwen-0.5B.

This VALIDATES the user's reframed hypothesis: DINO as an **Anti-Overfit Consistency Regularizer** (weak,
late, logit-only) — distinct from the failed "DINO as factual lever" (which caused rambling). Here DINO
disrupts briefly on turn-on (@400 FACT 0.0) then slows the train-stream drift, letting content-KL hold
the factual distribution to 0.333.

D gens are genuinely better than the earlier 1.5B (which hallucinated "capital of Italy is Milan"):
- rome: "The capital city in Italy is Rome." (correct, crisp)
- berlin: "The capital city in Germany is Berlin." (correct, crisp)
- tokyo: "Tokyo, Japan ... capital is Osaka" (Tokyo right, Osaka wrong)
- moscow: "capital city in Ukraine is Moscow" (Moscow right, country wrong)
first_token_hit stays low (0.074) — the answer follows a "The capital city in X is Y" preamble rather
than being the literal first token; gold_rank_mean 290 reflects that, not pure rambling.

## Implication
Qwen-1.5B is NOT capped at 0.222 — the minimal recipe over-trained it, and a late anti-overfit DINO
recovers the 0.333 that 0.5B reaches. So "bigger-Qwen-is-better does not hold" (rung-2) is REFINED:
bigger Qwen needs an anti-overfit regularizer to MATCH the smaller base; it does not yet EXCEED it.
Next open question: does D's late-DINO recipe push a still-stronger base (Qwen-1.7B/3B) PAST 0.333, or
does readout (first_token_hit) remain the wall across scales? Files:
reports/qwen_ladder/qwen15_rfit{A,B,C,D}_* (facteval.jsonl + train.json).
