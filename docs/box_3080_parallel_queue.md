# RTX 3080 Parallel Queue

Document position: [Index](./index.md) -> Windows GPU box companion runbook.

Purpose:

```text
Use the local RTX 3080 box while Colab/L4 is busy with long 1.1B training runs.
Do not waste the 3080 on jobs it cannot fit. Use it as a fast predictor, evaluator,
and pipeline-prep machine.
```

Last updated: 2026-06-27.

## Machine Role

| machine | best use | avoid |
| --- | --- | --- |
| Colab L4/A100 | 1.1B adaptation, long FACT runs, GGUF export if already mounted | idle GPU sessions |
| RTX 3080 10GB | 160M predictor runs, seed variance, eval/generation scoring, script smoke, post-run analysis | full 1.1B FACT training with AdamW states |

The current rule:

```text
Colab = main training factory.
3080 = fast branch-killer / evaluator.
```

## Current Experiment State

Known:

```text
FACT-004A lm_head unfreeze:
  DISCARDED. 1.1B fact dropped 0.185 -> 0.04/0.00.

FACT-003D protected factual replay:
  IMPLEMENTED.
  160M predictor says mu=1.0 transfers:
    eval panel 0.037 -> 0.259
    heldout atomic 0.093 -> 0.227
    train atomic saturates, so memorisation control is visible.

FACT-003E length-balanced replay:
  IMPLEMENTATION READY.
  Tests the user's critique that FACT-003D short atomic replay is not representative
  PTQ/QAT adaptation data. Same protected facts, mixed short/sentence/chat/explain/long
  prompt and answer lengths.

Colab 1.1B FACT-003D mu=1.0:
  decisive run in progress / pending result.
```

Therefore the 3080 should **not** start another speculative 1.1B-like branch until the
Colab result lands. It should prepare the next decision.

## Priority Queue While Colab Is Busy

### Q1. FACT-003D 160M Seed-Variance Check

Why:

```text
The 160M predictor was useful once.
Before trusting it repeatedly, test whether mu=1.0 transfer survives seeds.
```

What to run:

```text
mu = 1.0 only
steps = 400
seeds = 41, 42, 43
score eval_panel, heldout_atomic, train_atomic
```

Expected reading:

```text
eval_panel consistently above mu=0.0 control (~0.037):
  160M predictor is reliable enough for future branch-killing.

only one seed moves:
  predictor is noisy; do not over-read a single 160M run.

train_atomic rises but heldout/eval do not:
  replay is memorisation at that seed/data draw.
```

Current script does not expose `--seed` directly. Two options:

```text
cheap now:
  rerun fact003d_160m_sweep.py with a different work dir after adding a --seed flag.

better:
  add --seed to scripts/fact003d_160m_sweep.py and pass it through rt116.
```

### Q2. FACT-003D Eval-Only Dry Run

Why:

```text
When Colab 1.1B finishes, the 3080 should be able to score/check artifacts without
debugging Windows issues again.
```

Use existing 160M output dirs:

```cmd
cd C:\Users\gtpk\BitNet-Transformers
C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 scripts\fact003d_160m_sweep.py --mus 1.0 --skip-train
```

Pass condition:

```text
summary.md regenerates without pyarrow/cp949/transformers errors.
```

### Q2b. FACT-003E Length-Balanced Replay Probe

Why:

```text
PTQ/QAT calibration normally uses representative data. FACT-003D used deliberately short
atomic facts. This probe checks whether mixing prompt/answer lengths improves transfer
or only dilutes the factual signal.
```

Build the mixed-length replay set:

```cmd
cd C:\Users\gtpk\BitNet-Transformers
C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 scripts\make_length_balanced_factual_replay.py
```

Run the 160M predictor:

```cmd
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

Optional seed-matched atomic control:

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

Reading:

| result | meaning | next |
| --- | --- | --- |
| lengthmix eval/heldout > atomic by >=0.05, CE stable | representative length helps transfer | use lengthmix in the next 1.1B run |
| same facts, better tags/CE | length helps fluency but not retrieval | keep lengthmix as safer surface, widen facts |
| train high, heldout/eval flat | memorisation | broader/public factual data |
| worse than atomic | long surfaces dilute signal under fixed budget | keep atomic for mechanism tests |

### Q3. Result-Parser Template For 1.1B

Why:

```text
When the 1.1B Colab result arrives, we need a fixed table and verdict, not ad-hoc prose.
```

Template:

| metric | baseline FACT-003C | FACT-003D mu=1.0 | verdict |
| --- | ---: | ---: | --- |
| FACT panel | 0.185 | TBD | success if >=0.25, strong if >=0.35 |
| heldout atomic | TBD | TBD | transfer if rises with train atomic |
| train atomic | TBD | TBD | memorisation control |
| CE/PPL | ~43 PPL class | TBD | fail if collapse |
| tags | ok | TBD | fail if empty/loop/salad |
| i2_s vs f16 | parity expected | TBD | runtime should remain exonerated |

### Q4. Next-Branch Prep Only After 1.1B Result

Do not run these yet, but keep them ready:

| if 1.1B result | next branch |
| --- | --- |
| FACT >=0.35 | data/replay scaling; maybe public clean QA subset |
| FACT 0.25-0.35 | widen protected replay or tune mu around 0.75/1.25 |
| FACT 0.20-0.25 | partial success; add broader facts or content-AKL |
| FACT <=0.185 but train atomic high | memorisation; need broader/public factual data |
| CE/tags collapse | lower mu or lower replay ratio |

## SSH Commands

From the Mac:

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "cd C:\Users\gtpk\BitNet-Transformers & git fetch origin & git reset --hard origin/main"
```

Check GPU/env:

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "cd C:\Users\gtpk\BitNet-Transformers & C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
```

Run eval-only dry run:

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "cd C:\Users\gtpk\BitNet-Transformers & set PYTHONUTF8=1 & C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 scripts\fact003d_160m_sweep.py --mus 1.0 --skip-train"
```

Run full 160M mu sweep if needed:

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "cd C:\Users\gtpk\BitNet-Transformers & set PYTHONUTF8=1 & C:\Users\gtpk\anaconda3\envs\bnt\python.exe -X utf8 scripts\fact003d_160m_sweep.py --mus 0.5,1.0,2.0 --steps 400"
```

GPU poll:

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader"
```

## What Not To Do On The 3080

Avoid:

```text
full 1.1B AdamW FACT training
long lambda sweeps without a branch decision
bitnet.cpp large-model rebuild/benchmark
speculative hybrid-from-start training before FACT-003D 1.1B result
```

The 3080 is valuable because it is persistent and cheap, not because it replaces Colab
for memory-heavy training.

## Immediate Recommendation

While the Colab 1.1B FACT-003D run is active:

```text
1. run Q2 eval-only dry run to keep the Windows scoring path warm.
2. if there is still idle time, add/run Q1 seed variance for mu=1.0.
3. wait for 1.1B before launching a new branch.
```

This gives us useful information without spending the 3080 on a speculative path.
