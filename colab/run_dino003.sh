#!/bin/bash
# DINO-I2S-003: ONE stabilisation attempt for the 1.1B DINO collapse (DINO-002 went salad).
# Goal is NOT higher score -- it is to test whether DINO can be made STABLE at 1.1B.
# Changes vs 002: DINO centering ON, dino weight 0.2->0.1, dino warmup 150 steps, lr 2e-4->1e-4,
# early-collapse detector (stop if >30% of panel generations are salad/empty/loop from step 200).
#
# Verdict (user): success = salad suppressed + gold_rank improving + eval near FACT-003C 0.185.
# failure = salad recurs / detector trips -> 1.1B same-topology I2_S cannot take an auxiliary
# objective -> accept content-KL 0.185 as the 1.1B ceiling and move to the Qwen ladder.
set -e
cd /content/bnt

CKPT=/content/drive/MyDrive/bnt_ckpt/dino003_1p1b
OUT=/content/drive/MyDrive/bnt_results/dino003_1p1b
mkdir -p "$OUT"
RESUME=""
if [ -f "$CKPT/ckpt.pt" ]; then RESUME="--resume"; echo "[resume] found $CKPT/ckpt.pt"; fi

python -X utf8 scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source mixed --answer-loss-only \
  --base-kl-replay --kl-content-only --kl-weight 0.2 \
  --dino-logit-weight 0.1 --dino-view-mode dropout --dino-view-p 0.1 --dino-batch 2 \
  --dino-center --dino-center-m 0.9 --dino-warmup-steps 150 \
  --dino-collapse-check-every 100 --dino-collapse-min-step 200 --dino-collapse-salad-thresh 0.3 \
  --exclude-panel \
  --steps 800 --seq-len 256 --batch 4 --grad-accum-steps 6 --lr 1e-4 --seed 1 \
  --dtype bfloat16 --optim adamw8bit --grad-checkpointing \
  --max-train-tokens 2000000 \
  --out-dir "$OUT/adapted_model" \
  --json-out "$OUT/rt140_dino003_train.json" \
  --ckpt-dir "$CKPT" --ckpt-every-min 25 $RESUME \
  --metrics-out "$OUT/metrics.jsonl" --tb-logdir "$OUT/tb" \
  --log-every 25 2>&1 | tee -a "$OUT/run.log"

echo "==================== DINO-I2S-003 TRAIN DONE ====================" | tee -a "$OUT/run.log"

python -X utf8 scripts/score_dino002.py \
  --dino-dir "$OUT/adapted_model" \
  --teacher-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --panel data/factual_panel_v1.jsonl --tight data/popqa_heldout_tight.jsonl \
  --out "$OUT/pyscore.json" 2>&1 | tee -a "$OUT/pyscore.log"

echo "==================== DINO-I2S-003 SCORED + SAVED ====================" | tee -a "$OUT/run.log"
