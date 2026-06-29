#!/bin/bash
# DINO-CONTROL-B (precision isolation): fp32 student + adamw8bit (vs DINO-002's bf16 + adamw8bit).
# Control-A ruled out the optimizer (plain adamw also collapsed, step 300). bf16 is now the common
# factor across every 1.1B collapse (DINO-002/003, Control-A). This run changes ONLY precision
# (bf16 -> fp32), optimizer held at adamw8bit. Fits ~13GB on the 24GB L4 (fp32 model 4.4 + fp16
# teacher 2.2 + adamw8bit 1.9 + fp32 grads 3.9). Needs bitsandbytes (pip install before launch).
#   stable   -> bf16 training was the destabiliser; DINO works at 1.1B in fp32.
#   collapse -> precision is not it either -> SCALE/base is the cause -> Pythia ladder.
# Everything else EXACTLY DINO-002 (dino-weight 0.2, lr 2e-4, no centering/warmup).
set -eo pipefail
cd /content/bnt

CKPT=/content/drive/MyDrive/bnt_ckpt/dino_ctrl_fp32
OUT=/content/drive/MyDrive/bnt_results/dino_ctrl_fp32
mkdir -p "$OUT"
RESUME=""
if [ -f "$CKPT/ckpt.pt" ]; then RESUME="--resume"; echo "[resume] found $CKPT/ckpt.pt"; fi

python -X utf8 scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source mixed --answer-loss-only \
  --base-kl-replay --kl-content-only --kl-weight 0.2 --teacher-dtype float16 \
  --dino-logit-weight 0.2 --dino-view-mode dropout --dino-view-p 0.1 --dino-batch 2 \
  --dino-collapse-check-every 100 --dino-collapse-min-step 200 --dino-collapse-salad-thresh 0.3 \
  --exclude-panel \
  --steps 800 --seq-len 256 --batch 4 --grad-accum-steps 6 --lr 2e-4 --seed 1 \
  --dtype float32 --optim adamw8bit --grad-checkpointing \
  --max-train-tokens 2000000 \
  --out-dir "$OUT/adapted_model" \
  --json-out "$OUT/rt143_dino_ctrl_fp32_train.json" \
  --ckpt-dir "$CKPT" --ckpt-every-min 25 $RESUME \
  --metrics-out "$OUT/metrics.jsonl" --tb-logdir "$OUT/tb" \
  --log-every 25 2>&1 | tee -a "$OUT/run.log"

echo "==================== DINO-CTRL-B(fp32) TRAIN DONE ====================" | tee -a "$OUT/run.log"

python -X utf8 scripts/score_dino002.py \
  --dino-dir "$OUT/adapted_model" \
  --teacher-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --panel data/factual_panel_v1.jsonl --tight data/popqa_heldout_tight.jsonl \
  --out "$OUT/pyscore.json" 2>&1 | tee -a "$OUT/pyscore.log"

echo "==================== DINO-CTRL-B(fp32) SCORED + SAVED ====================" | tee -a "$OUT/run.log"
