# FACT-003E: Length-Balanced Factual Replay

Document position: [Index](./index.md) -> FACT recovery branch after FACT-003D.

Status: implementation-ready. Do **not** treat this as a result document until the
RTX 3080 run lands.

## Why This Exists

The user pointed out a real weakness in FACT-003D:

```text
PTQ/QAT calibration normally uses representative data.
Our protected factual replay was deliberately short atomic QA.
That is clean for diagnosis, but not representative of real prompt/answer lengths.
```

FACT-003D answered:

```text
Can a small protected fact set move factual behavior at all?
```

FACT-003E asks the next, more practical question:

```text
If the same protected facts are presented with mixed prompt and answer lengths,
does the model recover facts more robustly or with less degeneration?
```

This is still not a product dataset. It is a controlled ablation for **length and
surface-form representativeness**.

## Hypothesis

Short atomic replay may be too narrow:

```text
Q: What is the capital of Spain?
A: Madrid
```

The model later gets evaluated or used with prompts closer to:

```text
User: Please answer this factual question naturally.
What is the capital of Spain?

Assistant: The capital of Spain is Madrid.
```

If the replay distribution is too short, the model may learn an isolated fact lookup
surface without learning to route that fact through ordinary answer formats.

## Dataset Construction

Source files:

```text
data/atomic_facts_train.jsonl
data/atomic_facts_heldout.jsonl
```

Generated files:

```text
data/atomic_facts_lengthmix_train.jsonl
data/atomic_facts_lengthmix_heldout.jsonl
reports/fact003e_lengthmix_dataset_summary.{json,md}
```

Generation command:

```bash
python -X utf8 scripts/make_length_balanced_factual_replay.py
```

For every protected fact, generate five surfaces:

| style | prompt length | answer length | purpose |
| --- | --- | --- | --- |
| `short` | short | short | preserve FACT-003D baseline surface |
| `sentence` | short | sentence | teach complete-answer form |
| `chat` | medium | sentence | match chat-style inference |
| `explain` | short | sentence + generic explanation | test slightly longer answers |
| `long` | long/contextual | sentence | stress longer prompt routing |

The canonical answer stays in `must_contain`, so scoring still checks the same fact.
The protected train/held-out entity split is inherited from FACT-003D.

## 3080 Run

Run after syncing the box to the current commit:

```cmd
cd C:\Users\gtpk\BitNet-Transformers
C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 scripts\make_length_balanced_factual_replay.py
C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 scripts\fact003d_160m_sweep.py ^
  --label FACT-003E-lengthmix ^
  --mus 1.0 ^
  --steps 400 ^
  --seed 41 ^
  --work reports\fact003e_160m_lengthmix_seed41 ^
  --factual-replay data\atomic_facts_lengthmix_train.jsonl ^
  --heldout-file data\atomic_facts_lengthmix_heldout.jsonl ^
  --train-score-file data\atomic_facts_lengthmix_train.jsonl ^
  --heldout-sample 150 ^
  --train-sample 80
```

Optional fair comparison on the same seed:

```cmd
C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 scripts\fact003d_160m_sweep.py ^
  --label FACT-003D-atomic-seed41 ^
  --mus 1.0 ^
  --steps 400 ^
  --seed 41 ^
  --work reports\fact003d_160m_atomic_seed41 ^
  --factual-replay data\atomic_facts_train.jsonl ^
  --heldout-file data\atomic_facts_heldout.jsonl ^
  --train-score-file data\atomic_facts_train.jsonl
```

## Metrics

Use the existing PyTorch scorer from `scripts/fact004a_160m_smoke.py` through
`scripts/fact003d_160m_sweep.py`.

Primary:

| metric | meaning |
| --- | --- |
| `eval_panel fact_rate` | fixed FACT-001 panel; main transfer signal |
| `heldout_replay fact_rate` | protected split transfer, now with length-mixed prompts |
| `train_replay fact_rate` | memorisation control |
| `eval CE` | whether length-mix harms language-model recovery |
| tags/sample text | catches salad/loop/empty collapse |

Compare against FACT-003D atomic seed-41 and the existing seed-variance band:

```text
FACT-003D mu=1.0 160M seeds: eval_panel roughly 0.185-0.296
mu=0 control: roughly 0.037
```

## Decision Tree

### E1: length-mix improves eval/heldout and keeps CE stable

```text
lengthmix eval_panel >= atomic seed-matched by >= 0.05
heldout_replay rises
CE/tags stable
```

Interpretation:

```text
representative length/surface distribution matters.
Use length-mix replay for the next 1.1B or Qwen-stage protected replay.
```

Next:

```text
run 1.1B FACT-003E with the same mu=1.0 or a small mu sweep around 0.75/1.0/1.25.
```

### E2: same factual score, better tags/CE

Interpretation:

```text
length-mix helps fluency/format robustness but not factual retrieval.
Keep length-mix as a safer replay surface, but factual ceiling remains objective/data.
```

Next:

```text
combine length-mix with content-KL and protected replay; expand factual data breadth.
```

### E3: train_replay rises, heldout/eval flat

Interpretation:

```text
the replay is memorised, not transferred.
Length diversity alone is not enough; need broader factual coverage or benchmark-like data.
```

Next:

```text
public factual/instruction data, protected factual replay at larger scale, or teacher/replay objective.
```

### E4: length-mix is worse than atomic

Interpretation:

```text
short atomic facts are cleaner for this constrained 160M predictor.
Longer prompt/answer forms dilute the factual signal under fixed budget.
```

Next:

```text
keep atomic replay for mechanism tests; use length-mix only after increasing factual weight,
steps, or replay batch.
```

## Claim Discipline

This experiment cannot prove broad factual ability. It can only answer whether the
current protected-replay failure mode is partly caused by unrepresentative short-form
training data.

Do not claim:

```text
FACT-003E solves factual quality.
```

Allowed claim if positive:

```text
Representative prompt/answer lengths improve protected factual replay transfer in the
160M predictor and should be used for larger-model recovery runs.
```

