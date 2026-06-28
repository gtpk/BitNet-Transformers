#!/bin/bash
# DINO-CONTROL: disentangle SCALE vs PRECISION/OPTIMIZER for the 1.1B DINO collapse.
# DINO-002 collapsed at 1.1B with bf16 + adamw8bit. The 160M DINO positive ran with fp32 + adamw.
# So "160M works / 1.1B collapses" is confounded by precision + optimizer (and base model).
# This run = EXACTLY DINO-002's hyperparams (dino-weight 0.2, lr 2e-4, NO centering/warmup) but with
# fp32 + plain adamw + fp32 teacher (the 160M setup). It changes ONLY precision/optimizer.
#
#   collapse again -> SCALE/base is the cause -> the scale-stability thesis stands (-> Pythia ladder).
#   stays stable   -> the 1.1B collapse was a bf16/adamw8bit artifact -> earlier 1.1B conclusions
#                     (blend + DINO collapse) must be revisited.
# Early-collapse detector stops ~step 200 if it goes degenerate (saves compute), read-only so it does
# not perturb the numerics being tested.
set -e
cd /content/bnt

CKPT=/content/drive/MyDrive/bnt_ckpt/dino_control_fp32adamw
OUT=/content/drive/MyDrive/bnt_results/dino_control_fp32adamw
mkdir -p "$OUT"
RESUME=""
if [ -f "$CKPT/ckpt.pt" ]; then RESUME="--resume"; echo "[resume] found $CKPT/ckpt.pt"; fi

python -X utf8 scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source mixed --answer-loss-only \
  --base-kl-replay --kl-content-only --kl-weight 0.2 --teacher-dtype float32 \
  --dino-logit-weight 0.2 --dino-view-mode dropout --dino-view-p 0.1 --dino-batch 2 \
  --dino-collapse-check-every 100 --dino-collapse-min-step 200 --dino-collapse-salad-thresh 0.3 \
  --exclude-panel \
  --steps 800 --seq-len 256 --batch 4 --grad-accum-steps 6 --lr 2e-4 --seed 1 \
  --dtype float32 --optim adamw --grad-checkpointing \
  --max-train-tokens 2000000 \
  --out-dir "$OUT/adapted_model" \
  --json-out "$OUT/rt141_dino_control_train.json" \
  --ckpt-dir "$CKPT" --ckpt-every-min 25 $RESUME \
  --metrics-out "$OUT/metrics.jsonl" --tb-logdir "$OUT/tb" \
  --log-every 25 2>&1 | tee -a "$OUT/run.log"

echo "==================== DINO-CONTROL TRAIN DONE ====================" | tee -a "$OUT/run.log"

python -X utf8 scripts/score_dino002.py \
  --dino-dir "$OUT/adapted_model" \
  --teacher-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --panel data/factual_panel_v1.jsonl --tight data/popqa_heldout_tight.jsonl \
  --out "$OUT/pyscore.json" 2>&1 | tee -a "$OUT/pyscore.log"

echo "==================== DINO-CONTROL SCORED + SAVED ====================" | tee -a "$OUT/run.log"
