# RT-131 / FACT-002 Summary

| arm | fact_i2s | fact_f16 | agreement | adapted PPL | recovered | qr003 delta | tags |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| wikitext | 0.037 | 0.0 | 26/27 | None | None | None | {'ok': 27} |
| instr | 0.0 | 0.0 | 27/27 | 2480.3099320685615 | 0.40276354941552744 | 0.05776073840795615 | {'empty': 25, 'ok': 2} |
| mixed | 0.074 | 0.074 | 27/27 | 56.24223758781454 | 0.8135895767832033 | 0.012341633944817687 | {'ok': 26, 'repetitive': 1} |

Best arm: mixed
Decision: S3 objective gap
Next procedure: write S3 verdict; implement I2 objective branch

Review checklist:
- what improved:
- what did not:
- F16/I2_S agreement:
- no eval leakage: data/factual_panel_v1.jsonl was eval-only
- claim level:
