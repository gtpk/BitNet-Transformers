# FACT-003H PopQA blend 1.1B -- ANOMALOUS UNDER-RECOVERY (not a clean transfer verdict)

TinyLlama-1.1B, content-KL 0.2 + PopQA 12.7k blended at 5% (no bolt-on mu). Ran step 0->800 clean
(no recycle this time). pyscore + all artifacts on Drive (bnt_results/fact003h_popqa_blend0.05/).

| panel | fact_rate | CE | tags |
| --- | ---: | ---: | --- |
| eval_panel (27) | 0.0 | 7.149 | loop 27 |
| popqa_tight (300) | 0.0 | 7.149 | loop 300 |
| popqa_loose (150) | 0.0 | 7.149 | loop 150 |
| popqa_train (80) | 0.0 | 7.149 | loop 80 |

Training: ce_fp 2.295, ce_ptq 11.49, **ce_adapted 7.128, recovered_fraction 0.474**. train_ce
dropped 11.6->7.7 by step 50 then PLATEAUED at ~7.4 for the remaining 750 steps (early stall).

## This is an ANOMALY, not a blend verdict

The model is degenerate (eval CE 7.13, every panel fact 0.0 / 100% loop tags), so the A-E transfer
decision table CANNOT be applied -- there is nothing to read about transfer from a model that did
not recover. The recovered_fraction 0.474 is far below FACT-003D's 0.806 under the SAME base recipe
(content-KL + answer-CE, same optim/lr/steps). Critically, **the 160M PopQA blend recovers FINE**
(HOME-001 eta=0, same blend, recovered_fraction 0.885), so the blend mechanism is not broken -- this
1.1B run specifically STALLED early (train loss frozen at ~7.4 from step 50). Likely an optimizer/
adamw8bit one-off (FACT-003D recovered 0.806 with the same optimizer), not a property of PopQA blend.

## Next: RE-RUN required before any FACT-003H conclusion

Cannot conclude "blend transfers / fails" from a stalled model. Re-run FACT-003H (fresh, same recipe)
to distinguish a one-off stall from a real 1.1B+blend interaction. If the re-run recovers normally
(recovered ~0.8) and is fluent, THEN apply the A-E table to its tight/eval/train scores. If it stalls
again, investigate the 1.1B blend stream / adamw8bit interaction.
