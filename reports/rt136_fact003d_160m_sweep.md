# FACT-003D 160M mu-sweep + control (protected factual replay direction predictor)

Run on the 3080 box (Felladrin/Llama-160M-Chat-v1, content-KL lambda=0.2 + mu*answer-CE on the
curated atomic-facts train set, 400 steps). PyTorch-scored on the ternary-materialised dir
(rep-penalty 1.2, contains-match), three panels per arm:
- **eval_panel** = data/factual_panel_v1.jsonl (27, the real held-out factual eval; disjoint from training)
- **heldout_atomic** = data/atomic_facts_heldout.jsonl (97; TRANSFER -- entities never trained)
- **train_atomic** = sample of data/atomic_facts_train.jsonl (60; MEMORISATION control)

| mu | eval_panel | heldout_atomic (transfer) | train_atomic (memorise) | eval CE |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 (control, no replay) | 0.037 | 0.093 | 0.083 | 3.899 |
| 0.5 | 0.222 | 0.216 | 0.983 | 3.977 |
| 1.0 | 0.259 | 0.227 | 0.983 | 4.029 |
| 2.0 | 0.259 | 0.227 | 1.000 | 4.130 |

## Reading (vs the mu=0.0 control)

- **Protected factual replay TRANSFERS, it does not merely memorise.** eval_panel (prompts entirely
  different from the atomic-facts training items) jumps **0.037 -> 0.259 (~7x)** when replay is added;
  heldout_atomic (held-out entities) **0.093 -> 0.227 (~2.4x)**. train_atomic saturates (~0.98-1.0) as
  expected, but the lift on the *disjoint* eval panel proves the gain is general factual-retrieval
  recovery, not item memorisation.
- **mu = 1.0 is the sweet spot.** 0.5 is weaker; 2.0 adds no transfer over 1.0 and costs eval CE
  (3.977 -> 4.130). Confirms mu=1.0 as the 1.1B setting.
- **No fluency collapse.** eval CE moves only 3.90 -> 4.03 (contrast FACT-004A lm_head unfreeze, which
  kept CE but DESTROYED facts -- here facts rise AND CE holds).

## Caveat for the 1.1B read-across

160M starts from a ~0 factual floor (control eval_panel 0.037), so it has large headroom. At 1.1B the
no-replay FACT-003C baseline is already 0.185, so the ABSOLUTE lift will be smaller. The robust
predictions are the DIRECTION (transfer, not memorisation) and the mu choice (1.0). The 1.1B mu=1.0
run (RT-136, Colab) is the decisive test.
