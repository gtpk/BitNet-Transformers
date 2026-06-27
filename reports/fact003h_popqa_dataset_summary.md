# FACT-003H PopQA factual blend dataset

source: akariasai/PopQA:test (14267 rows)  LICENSE: HF card unclear -> research only

- kept 14118 (dropped 149 panel-leaks), entity-split -> train 12706 / heldout 1412
- post-write leak check: 0 (must be 0)
- approx tokens/row ~14; train ~174k tokens (5% blend of 2M = 100k -> ~0.57x of the set per blend budget; NO repetition)

prompt = 'Q: {question}\nA:'  answer = obj  must_contain = obj + possible_answers (alias-tolerant)
Blended via FACT-003G --factual-blend-file/-frac (NOT a separate mu*loss).

- train: `C:\Users\gtpk\BitNet-Transformers\data\popqa_blend_train.jsonl`
- heldout: `C:\Users\gtpk\BitNet-Transformers\data\popqa_blend_heldout.jsonl`
