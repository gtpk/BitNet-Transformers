# FACT-003H PopQA blend -- 160M mechanism smoke (NOT a 1.1B predictor)

PopQA (12,706 train / 1,412 heldout, de-leaked) blended into the mixed stream at 5%
(--factual-blend-frac 0.05, FACT-003G mechanism; NOT a separate mu*loss), 160M, 400 steps.

| panel | fact_rate | note |
| --- | ---: | --- |
| eval_panel | 0.111 | unchanged vs base |
| heldout_replay (PopQA) | 0.107 | |
| train_replay (PopQA) | 0.075 | NOT saturated |
| eval CE | 3.966 | fluent |

blend composition (logged): `40,000 factual (5%) + 380,000 instruction + 380,000 wikitext` -- 40k of
PopQA's ~174k tokens => NO repetition (vs FACT-003D atomic: 291 facts repeated ~18x).

## What this validates (and what it does NOT)

VALIDATED:
- Pipeline: PopQA -> de-leak -> entity-split -> blend -> train -> score all work end to end.
- **Memorisation signature is GONE.** FACT-003D mu=1.0 had train_atomic 1.00 vs eval 0.111 (a
  memorise-the-cards gap). Blend has train_replay 0.075 ~= heldout 0.107 ~= eval_panel 0.111 -- the
  three panels move together, the hallmark of distribution-learning not card-memorisation. (The
  auto-"memorisation" READING is the degenerate single-arm logic; train is LOW, not high.)

DOES NOT show / cannot show:
- No eval lift at 160M (0.111). A tiny model sees each of 12k diverse facts very few times at 5%, so
  it learns few (train only 0.075). This is the opposite failure mode to 160M's atomic-bolt-on run
  (which over-fit simple facts and rose to 0.259).
- **160M does NOT predict 1.1B** (established: FACT-003D 160M predicted 0.259, 1.1B gave 0.111). So
  this smoke is mechanism-validation ONLY -- the GO/NO-GO to 1.1B must not lean on 160M magnitude.

## Status / next

FACT-003H is 1.1B-ready (data + blend mechanism). Gated on the FACT-003D mu=0.25 1.1B result: if
mu=0.25 fails/ambiguous, launch FACT-003H PopQA blend at 1.1B (the model with the capacity to learn
from diverse blend without the small-set overfit). LICENSE: PopQA = research/direction only.
