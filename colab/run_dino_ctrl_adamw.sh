#!/bin/bash
# DINO-CONTROL-A (optimizer isolation): bf16 + PLAIN adamw (vs DINO-002's bf16 + adamw8bit).
# fp32+adamw OOMs on a 24GB L4 (fixed cost ~21.5GB), so we separate the two confounds instead of
# conflating them. This run changes ONLY the optimizer (adamw8bit -> adamw), precision held at bf16.
# adamw8bit (8-bit optimizer states) is the prime suspect for the 1.1B collapse.
#   stable -> adamw8bit was the destabiliser (precision/scale fine); DINO works at 1.1B with adamw.
#   collapse -> not the optimizer; next isolate precision (fp32+adamw8bit), else SCALE is the cause.
# Everything else EXACTLY DINO-002 (dino-weight 0.2, lr 2e-4, no centering/warmup).
set -eo pipefail
cd /content/bnt

CKPT=/content/drive/MyDrive/bnt_ckpt/dino_ctrl_adamw
OUT=/content/drive/MyDrive/bnt_results/dino_ctrl_adamw
mkdir -p "$OUT"
RESUME=""
if [ -f "$CKPT/ckpt.pt" ]; then RESUME="--resume"; echo "[resume] found $CKPT/ckpt.pt"; fi

python -X utf8 scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source mixed --answer-loss-only \
  --base-kl-replay --kl-content-only --kl-weight 0.2 \
  --dino-logit-weight 0.2 --dino-view-mode dropout --dino-view-p 0.1 --dino-batch 2 \
  --dino-collapse-check-every 100 --dino-collapse-min-step 200 --dino-collapse-salad-thresh 0.3 \
  --exclude-panel \
  --steps 800 --seq-len 256 --batch 4 --grad-accum-steps 6 --lr 2e-4 --seed 1 \
  --dtype bfloat16 --optim adamw --grad-checkpointing \
  --max-train-tokens 2000000 \
  --out-dir "$OUT/adapted_model" \
  --json-out "$OUT/rt142_dino_ctrl_adamw_train.json" \
  --ckpt-dir "$CKPT" --ckpt-every-min 25 $RESUME \
  --metrics-out "$OUT/metrics.jsonl" --tb-logdir "$OUT/tb" \
  --log-every 25 2>&1 | tee -a "$OUT/run.log"

echo "==================== DINO-CTRL-A TRAIN DONE ====================" | tee -a "$OUT/run.log"

python -X utf8 scripts/score_dino002.py \
  --dino-dir "$OUT/adapted_model" \
  --teacher-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --panel data/factual_panel_v1.jsonl --tight data/popqa_heldout_tight.jsonl \
  --out "$OUT/pyscore.json" 2>&1 | tee -a "$OUT/pyscore.log"

echo "==================== DINO-CTRL-A SCORED + SAVED ====================" | tee -a "$OUT/run.log"
