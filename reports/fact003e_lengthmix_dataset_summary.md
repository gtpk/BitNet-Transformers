# FACT-003E Length-Mixed Factual Replay Dataset

styles: `short,sentence,chat,explain,long`

| split | source facts | expanded rows | prompt words min/mean/max | answer words min/mean/max |
| --- | ---: | ---: | ---: | ---: |
| train | 291 | 1455 | 5/16.8/43 | 1/6.1/15 |
| heldout | 97 | 485 | 5/16.5/39 | 1/1.1/3 |

This dataset preserves the protected train/held-out entity split from FACT-003D,
but replaces the single short QA surface form with a mixed-length adaptation set.

- train: `/Users/puka/repository/BitNet-Transformers/data/atomic_facts_lengthmix_train.jsonl`
- heldout: `/Users/puka/repository/BitNet-Transformers/data/atomic_facts_lengthmix_heldout.jsonl`
