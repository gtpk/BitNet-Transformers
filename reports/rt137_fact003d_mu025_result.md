# FACT-003D mu=0.25 (gentler protected replay) -- FAILED, lower mu did NOT preserve

TinyLlama-1.1B, content-KL 0.2 + mu=0.25 bolt-on factual-CE on the 291 atomic facts. Survived
3 VM recycles via Drive ckpt + resume fixes + the keep-alive ping; all artifacts auto-saved to
Drive (run.log/metrics.jsonl/tb/pyscore.json/adapted_model). PyTorch ternary scoring.

| arm | eval_panel | heldout_atomic | train_atomic | note |
| --- | ---: | ---: | ---: | --- |
| FACT-003C (no replay) | 0.185 | -- | -- | best baseline |
| FACT-003D mu=1.0 | 0.111 | 0.134 | 1.00 | full memorise, eval down |
| **FACT-003D mu=0.25** | **0.037** | 0.103 | 0.588 | partial memorise, eval down MORE |

recovered_fraction 0.783 (fluent, tags ok 27/27). 

## Verdict: lowering mu does NOT rescue small hard replay

The "mu=1.0 over-memorised -> lower mu preserves recall" hypothesis is REFUTED. mu=0.25 memorised
LESS (train_atomic 1.00 -> 0.588) but eval_panel got WORSE (0.111 -> 0.037), still well below the
0.185 no-replay baseline. (eval 0.037 vs 0.111 is ~1 vs 3 of 27 = within panel noise; both are a
collapse vs 0.185.) So a small (291) hard-CE protected-replay set is net-negative on the held-out
eval at 1.1B at ANY mu -- the lever is not mu, it is that the set is too small and hard-CE is a
memorisation objective. Small hard replay is DEMOTED.

## Next: FACT-003H PopQA blend 1.1B (launched)

Bigger (12.7k), de-leaked PopQA blended into the mixed stream at 5% (no separate mu*loss) -- the
structural fix for the memorisation trap (no repetition; facts are a low-ratio part of the Q/A
distribution). Primary transfer metric = data/popqa_heldout_tight.jsonl (916, alias<=3). The 160M
smoke already validated the mechanism (no memorise signature). LICENSE: PopQA research-only.
