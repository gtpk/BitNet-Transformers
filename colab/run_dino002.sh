#!/bin/bash
# DINO-I2S-002: 1.1B no-label self-distillation gate (dino_logit-only, hidden OFF, EOS mask ON).
# Run on Colab L4 after Drive is mounted. Resumable via Drive ckpt (survives VM recycle).
#
# Objective = v0 backbone (answer-CE + content-KL replay 0.2) + dino_logit-only self-distill 0.2.
# Predictions (DINO-DIAG-001): simple_fact rises, entity_attr lags, overall maybe > 0.185.
set -e
cd /content/bnt

CKPT=/content/drive/MyDrive/bnt_ckpt/dino002_1p1b
OUT=/content/drive/MyDrive/bnt_results/dino002_1p1b
mkdir -p "$OUT"

RESUME=""
if [ -f "$CKPT/ckpt.pt" ]; then RESUME="--resume"; echo "[resume] found $CKPT/ckpt.pt"; fi

python -X utf8 scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source mixed --answer-loss-only \
  --base-kl-replay --kl-content-only --kl-weight 0.2 \
  --dino-logit-weight 0.2 --dino-view-mode dropout --dino-view-p 0.1 --dino-batch 2 \
  --exclude-panel \
  --steps 800 --seq-len 256 --batch 4 --grad-accum-steps 6 --lr 2e-4 --seed 1 \
  --dtype bfloat16 --optim adamw8bit --grad-checkpointing \
  --max-train-tokens 2000000 \
  --out-dir "$OUT/adapted_model" \
  --json-out "$OUT/rt139_dino002_train.json" \
  --ckpt-dir "$CKPT" --ckpt-every-min 25 $RESUME \
  --metrics-out "$OUT/metrics.jsonl" --tb-logdir "$OUT/tb" \
  --log-every 25 2>&1 | tee -a "$OUT/run.log"

echo "==================== DINO-I2S-002 TRAIN COMPLETE ====================" | tee -a "$OUT/run.log"

python -X utf8 scripts/score_dino002.py \
  --dino-dir "$OUT/adapted_model" \
  --teacher-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --panel data/factual_panel_v1.jsonl --tight data/popqa_heldout_tight.jsonl \
  --out "$OUT/pyscore.json" 2>&1 | tee -a "$OUT/pyscore.log"

echo "==================== DINO-I2S-002 SCORED + SAVED ====================" | tee -a "$OUT/run.log"
