#!/usr/bin/env bash
set -euo pipefail

# Colab-oriented runner for the teacher-free BitNet conversion arena.
# Set SKIP_INSTALL=1 when dependencies are already installed.

PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m pip install -q "torch" "transformers" "accelerate"
fi

mkdir -p reports

"${PYTHON_BIN}" scripts/check_scaled_bitlinear.py \
  --json-out "${TC_JSON_OUT:-reports/scaled_bitlinear_tc_colab.json}"

arena_data_args=(--data-mode "${DATA_MODE:-synthetic}")
if [[ -n "${TEXT_PATH:-}" ]]; then
  arena_data_args+=(--text-path "$TEXT_PATH")
fi
if [[ -n "${TOKENIZER:-}" ]]; then
  arena_data_args+=(--tokenizer "$TOKENIZER")
fi

"${PYTHON_BIN}" scripts/run_tiny_real_arena.py \
  --seed "${SEED:-31}" \
  "${arena_data_args[@]}" \
  --train-steps "${TRAIN_STEPS:-800}" \
  --qat-steps "${QAT_STEPS:-128}" \
  --ste-qat-steps "${STE_QAT_STEPS:-128}" \
  --scaled-ste-steps "${SCALED_STE_STEPS:-128}" \
  --scaled-ste-learning-rate "${SCALED_STE_LEARNING_RATE:-0.002}" \
  --scaled-ste-group-size "${SCALED_STE_GROUP_SIZE:-64}" \
  --scaled-ste-lambda "${SCALED_STE_LAMBDA:-0.7}" \
  --scaled-ste-activation-bits "${SCALED_STE_ACTIVATION_BITS:-0}" \
  --hidden-size "${HIDDEN_SIZE:-128}" \
  --intermediate-size "${INTERMEDIATE_SIZE:-256}" \
  --num-layers "${NUM_LAYERS:-2}" \
  --num-heads "${NUM_HEADS:-4}" \
  --num-kv-heads "${NUM_KV_HEADS:-4}" \
  --seq-len "${SEQ_LEN:-64}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}" \
  --threads "${THREADS:-2}" \
  --json-out "${ARENA_JSON_OUT:-reports/tiny_real_arena_scaled_ste_colab.json}" \
  --strict
