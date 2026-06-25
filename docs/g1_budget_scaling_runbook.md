# G1 Budget-Scaling Runbook (RT-120 / TRAIN-003)

Document position: [Index](./index.md) -> after [Quality Recovery Plan](./quality_recovery_plan.md)
and [Paper Skeleton](./paper_skeleton.md). This is the pre-upgrade runbook for
the expensive 1.1B quality-recovery run.

## Purpose

G1 is the remaining high-severity paper gap:

```text
TinyLlama-1.1B recovery is only 0.480 under the old fixed budget.
```

That number is useful but under-budgeted. TRAIN-002 adapted ~968M target-linear
parameters with the same `300` optimizer steps as Llama-160M, half the per-step
batch, and 8-bit AdamW. The next run should test a narrower hypothesis:

```text
If the training-token budget is scaled toward the 1.1B target-param count,
teacher-free linears-only CE recovery should move materially above 0.480 while
I2_S runtime preservation remains near +0.002 nats.
```

This is not a recipe search. QR-005 already chose the default recipe:

```text
target linears only; no norms; no lm_head
```

## Baseline To Beat

TRAIN-002, TinyLlama-1.1B, fixed budget:

| metric | value |
| --- | ---: |
| target linears | 154 |
| trainable target params | ~968M |
| optimizer | AdamW8bit |
| microbatch | 4 |
| grad accumulation | 1 |
| steps | 300 |
| effective train tokens | 307k |
| FP PPL | 10.1 |
| one-shot PTQ PPL | 101,549 |
| adapted PPL | 1,217 |
| recovered_fraction | 0.480 |
| adapted i2_s vs f16 | +0.0023 nats |

For comparison, Llama-160M used 300 steps x batch 8 x seq 256 = 614k tokens for
~113M target params, about 17x more training tokens per trainable target param than
the old 1.1B run. G1 should reduce that mismatch.

## Success Tiers

Use recovered fraction, not raw PPL, as the primary QR-002 metric.

| tier | recovered_fraction | approximate adapted PPL target | interpretation |
| --- | ---: | ---: | --- |
| minimum pass | `> 0.480` by a clear margin | `< 1,217` | budget scaling helps |
| paper-useful | `>= 0.70` | `~160` or lower | strong Figure 3 improvement |
| strong | `>= 0.80` | `~64` or lower | close to 160M-quality recovery trend |
| stretch | `>= 0.90` | `~25` or lower | 1.1B matches the 160M recovery story |

QR-003 runtime preservation remains a separate gate:

```text
PASS: adapted i2_s vs adapted f16 <= +0.010 nats
OK/watch: <= +0.020 nats
FAIL: > +0.020 nats or no GGUF/I2_S artifact
```

The expected QR-003 result is near the previous scale-invariant value
`+0.002` nats. If QR-002 improves but QR-003 fails, that is a runtime/export issue,
not a recovery issue.

## Hardware Choice

Preferred:

- A100 40GB or larger
- `float32` model compute
- full AdamW if memory permits
- gradient checkpointing on
- effective batch at least `24`

Fallback:

- L4 24GB
- `float32` model compute
- AdamW8bit
- gradient checkpointing on
- use gradient accumulation to reach effective batch `16-24`

Avoid spending the upgrade on:

- `+norms` or `+lm_head` arms (QR-005 already closed this)
- gpt-oss / MoE ternary work (RT-118 says ROI is near zero)
- prompt-panel-only work (G4 is already closed at 160M)

## Preflight Checklist

Before the paid/large run:

1. Confirm the repo is at or after the commit that includes `--grad-accum-steps`:

   ```bash
   git pull --ff-only
   python scripts/rt116_quality_recovery.py --help | grep grad-accum
   ```

2. Confirm GPU and memory:

   ```bash
   nvidia-smi
   python - <<'PY'
   import torch
   print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
   print(torch.cuda.mem_get_info() if torch.cuda.is_available() else "")
   PY
   ```

3. Confirm bitnet.cpp x86 binaries exist for QR-003:

   ```bash
   test -x /content/bitnet.cpp/build/bin/llama-quantize
   test -x /content/bitnet.cpp/build/bin/llama-perplexity
   test -f /content/bitnet.cpp/utils/convert-hf-to-gguf-bitnet.py
   ```

4. Run a tiny end-to-end smoke on the same GPU before the big command:

   ```bash
   python scripts/rt116_quality_recovery.py \
     --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
     --steps 2 \
     --seq-len 256 \
     --batch 1 \
     --grad-accum-steps 1 \
     --lr 2e-4 \
     --max-train-tokens 8192 \
     --max-eval-tokens 4096 \
     --ppl-eval-tokens 256 \
     --dtype float32 \
     --optim adamw8bit \
     --grad-checkpointing \
     --bitnet /content/bitnet.cpp \
     --out-dir /content/bnt_smoke/tinyllama_g1_smoke \
     --json-out reports/rt120_tinyllama_g1_smoke.json
   ```

Smoke pass means:

- model loads on GPU
- target linears are replaced
- backward/optimizer step works
- adapted HF dir saves
- f16 and i2_s GGUF conversion runs
- llama-perplexity returns numbers
- JSON is written

Do not interpret the smoke quality numbers.

## One-Shot Run: A100 Preferred

This is the preferred paper-strength G1 command. It keeps the recipe fixed and
raises the effective training-token budget to about `4.9M` tokens:

```bash
python scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --steps 800 \
  --seq-len 256 \
  --batch 8 \
  --grad-accum-steps 3 \
  --lr 2e-4 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 60000 \
  --ppl-eval-tokens 3000 \
  --dtype float32 \
  --optim adamw \
  --grad-checkpointing \
  --bitnet /content/bitnet.cpp \
  --out-dir /content/bnt_runs/tinyllama_g1_a100_s800_b8x3 \
  --json-out reports/rt120_tinyllama_g1_a100_s800_b8x3.json \
  --log-every 25
```

If A100 memory is lower than expected, keep the same effective batch by reducing
microbatch and increasing accumulation:

```text
batch 8, accum 3  -> effective batch 24
batch 6, accum 4  -> effective batch 24
batch 4, accum 6  -> effective batch 24
```

Prefer changing batch/accumulation before changing steps or LR.

## One-Shot Run: L4 Fallback

Use this if A100 is unavailable. It answers the same budget-scaling question, but
with 8-bit optimizer states:

```bash
python scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --steps 800 \
  --seq-len 256 \
  --batch 4 \
  --grad-accum-steps 6 \
  --lr 2e-4 \
  --max-train-tokens 2000000 \
  --max-eval-tokens 60000 \
  --ppl-eval-tokens 3000 \
  --dtype float32 \
  --optim adamw8bit \
  --grad-checkpointing \
  --bitnet /content/bitnet.cpp \
  --out-dir /content/bnt_runs/tinyllama_g1_l4_s800_b4x6 \
  --json-out reports/rt120_tinyllama_g1_l4_s800_b4x6.json \
  --log-every 25
```

This is still much stronger than TRAIN-002:

```text
old TRAIN-002: 300 * 4 * 256  = 0.31M effective tokens
G1 fallback : 800 * 24 * 256 = 4.92M effective tokens
```

## Result Interpretation

After the run, read the JSON and report:

```bash
python - <<'PY'
import json, math
p = "reports/rt120_tinyllama_g1_a100_s800_b8x3.json"
d = json.load(open(p))
print("CE fp/ptq/adapted:", d["ce_fp"], d["ce_ptq"], d["ce_adapted"])
print("PPL fp/ptq/adapted:", d["ppl_fp"], d["ppl_ptq"], d["ppl_adapted"])
print("recovered_fraction:", d["recovered_fraction"])
print("effective_batch:", d["effective_batch"])
print("effective_tokens_per_step:", d["effective_tokens_per_step"])
print("QR-003:", d.get("qr003"))
if d.get("qr003", {}).get("i2s_vs_f16_nats") is not None:
    print("qr003_delta_nats:", d["qr003"]["i2s_vs_f16_nats"])
PY
```

Decision:

```text
recovered_fraction >= 0.70 and QR-003 <= +0.010 nats
  -> G1 materially improved; update Figure 3/4 and paper skeleton.

0.48 < recovered_fraction < 0.70 and QR-003 passes
  -> budget helps but not enough; consider 1200 steps or LR schedule.

recovered_fraction <= 0.48
  -> fixed-budget explanation is weakened; inspect LR, train loss curve, corpus, and optimizer.

QR-003 fails while recovery improves
  -> runtime/export regression; compare f16/i2_s only, do not discard recovery.
```

## If The Run Fails

OOM before training:

- reduce microbatch first (`8 -> 6 -> 4 -> 2`)
- raise `--grad-accum-steps` to preserve effective batch
- switch `--optim adamw` to `--optim adamw8bit`
- keep `--grad-checkpointing`

OOM during QR-003 conversion/perplexity:

- keep the adapted HF directory
- rerun only the export/perplexity path later by using the same `--out-dir` if needed
- do not rerun training unless the adapted directory was lost

Training runs but does not improve:

- keep the JSON; it is still a valid negative result
- inspect train CE logs first
- only then try LR schedule or longer steps

Session disconnect:

- the current driver saves only at the end, so use a stable runtime
- set `--out-dir` under persistent storage if available
- keep console logs or redirect them to a file in persistent storage

## What This Run Should Update

If successful:

- [Quality Recovery Plan](./quality_recovery_plan.md): add `RT-120 / TRAIN-003 RESULT`
- [Paper Skeleton](./paper_skeleton.md): replace the 1.1B row in Figure 3
- [Colab Validation Summary](./colab_validation_summary.md): add G1 result
- `reports/rt120_tinyllama_g1_*.json`: commit the raw result

Do not update the gpt-oss decision. RT-118 already closed that path as low ROI for
this recipe.
