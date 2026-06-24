# Real Tiny Text Validation Plan

Document position: [Index](./index.md) -> completed validation phase after Colab synthetic gates.

Related docs:

- [Colab Validation Summary](./colab_validation_summary.md)
- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [Evolutionary LLM Arena Plan](./evolutionary_llm_arena_plan.md)

## Purpose

The synthetic patterned arena has done its job: it found a teacher-free native
BitLinear-style candidate worth scaling. The next risk is different:

```text
Does scaled-STE still beat projected-QAT on real token distributions?
```

This phase should answer that before packed ternary kernels, GGUF export, or
TurboQuant integration work.

## Status

Date: 2026-06-24

```text
PASS
```

Colab Wikitext real-text validation passed after the synthetic seed,
group-size, and activation fake-quant gates. On a 200 KB Wikitext-2 sample
(`180177` train / `20020` eval byte tokens), seeds `31`, `32`, and `33` all
favored scaled-STE over projected-QAT on accuracy, loss, PPL, and fitness.
Both act0 and act8 scaled-STE stayed non-degenerate in generation smoke.

The full fixed result table is in
[Colab Validation Summary](./colab_validation_summary.md#real-text-validation-result).
The raw Colab JSON files were not yet committed locally, so paper-style
quantitative claims should either archive those files or rerun the sweep.

## Why Real Text Comes Before Kernels

Packed kernels and export paths turn an algorithm into a runtime artifact. They
are expensive to build and easy to overfit to the wrong target. The current
candidate has passed seed, group-size, and activation fake-quant gates, but all
of those gates used synthetic patterned tokens.

Real text validation removes the largest remaining quality risk first.

## Candidate Set

Minimum candidates:

- `fp16_dense`
- `s1_projected_qat_int4_kv`
- `s1_scaled_ste_int4_kv`
- `s1_scaled_ste_act8_int4_kv`

Optional candidates:

- `s1_groupwise_ptq_int4_kv`
- `s1_bitlinear_ste_int4_kv`

## Data

Use a tiny, cheap, reproducible text subset:

- Wikitext-style text if available in Colab
- small local text fixture if network or dataset access is inconvenient
- fixed train/eval split
- fixed tokenizer/config seed

The first version should be small enough to run repeatedly:

```text
train tokens: 50k-200k
eval tokens : 10k-50k
sequence len: 64 or 128
```

## Metrics

Primary:

- eval CE loss
- perplexity
- next-token accuracy
- resource-aware fitness
- Pareto frontier membership

Watch metrics:

- KL-to-fp16 logits
- short generation smoke
- NaN/Inf checks
- repeated-seed variance

KL-to-fp16 is a watch item because act8 Colab runs showed scaled-STE can have
better accuracy/loss/fitness than projected-QAT while still having slightly
higher KL-to-fp16.

## Pass Criteria

Proceed if:

- scaled-STE does not collapse on eval CE/PPL
- scaled-STE beats or ties projected-QAT on loss or accuracy
- scaled-STE remains on the Pareto frontier in at least most seeds
- generation smoke produces finite logits and non-degenerate text
- KL-to-fp16 does not diverge enough to contradict better CE/PPL

Pause if:

- scaled-STE loses clearly to projected-QAT on real-text CE/PPL across seeds
- act8 consistently damages real-text quality
- KL-to-fp16 grows while CE/PPL also worsens
- generation smoke degenerates despite low synthetic loss

## Next Implementation Task

Status: implemented in `scripts/run_tiny_real_arena.py`.

The arena now supports a real-text mode instead of requiring a separate one-off
notebook:

```text
scripts/run_tiny_real_arena.py
  --data-mode synthetic|text
  --text-path <path>
  --tokenizer <hf-tokenizer-or-local>
```

Default tokenization is byte-level with vocab size `256`, so local smoke tests
do not need downloads. A small committed fixture exists at `data/tiny_corpus.txt`.

Local harness smoke:

```bash
.venv/bin/python scripts/run_tiny_real_arena.py \
  --data-mode text \
  --text-path data/tiny_corpus.txt \
  --train-steps 40 \
  --qat-steps 12 \
  --ste-qat-steps 12 \
  --scaled-ste-steps 12 \
  --seq-len 64 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --json-out reports/tiny_real_text_fixture_smoke.json
```

This fixture smoke is a harness check, not a scientific result. The fixture is
only a few kilobytes, so small accuracy differences are noise. Its purpose is
to verify:

- text mode loads and splits real text
- byte tokenizer path works without network
- synthetic mode remains unchanged
- generation smoke reports finite, non-degenerate outputs

## Colab Real-Text Result

The actual real-text validation used a larger text file in Colab, then ran the
arena with `--data-mode text`.

Observed setup:

```text
train tokens: 180177
eval tokens : 20020
seeds       : 31, 32, 33
dataset     : Wikitext-2 sample, 200 KB text file
```

Summary:

```text
scaled-STE act0: PASS, 3/3 seeds over projected-QAT
scaled-STE act8: PASS, 3/3 seeds over projected-QAT
generation     : finite and non-degenerate
watch item     : KL-to-fp16 remains higher than projected-QAT, but CE/PPL are better
```

Reproduction command shape:

```bash
python scripts/run_tiny_real_arena.py \
  --data-mode text \
  --text-path data/wikitext_tiny.txt \
  --train-steps 800 \
  --qat-steps 128 \
  --ste-qat-steps 128 \
  --scaled-ste-steps 128 \
  --hidden-size 128 \
  --intermediate-size 256 \
  --num-layers 2 \
  --seq-len 64 \
  --batch-size 32 \
  --eval-batch-size 128 \
  --json-out reports/tiny_real_text_scaled_ste_seed31.json \
  --strict
```

Or use the Colab runner wrapper:

```bash
DATA_MODE=text \
TEXT_PATH=data/wikitext_tiny.txt \
SEED=31 \
TRAIN_STEPS=800 \
QAT_STEPS=128 \
STE_QAT_STEPS=128 \
SCALED_STE_STEPS=128 \
HIDDEN_SIZE=128 \
INTERMEDIATE_SIZE=256 \
NUM_LAYERS=2 \
SEQ_LEN=64 \
BATCH_SIZE=32 \
EVAL_BATCH_SIZE=128 \
ARENA_JSON_OUT=reports/tiny_real_text_scaled_ste_seed31.json \
bash scripts/run_colab_scaled_ste_arena.sh
```

## Decision After This Phase

Real text passed, so the quality gate no longer blocks storage/export work.

Next:

1. archive or regenerate the Colab JSON reports
2. proceed to [Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md)
3. implement model-level pack/unpack export with dense logit-equivalence TC
4. then scope packed reference matmul, GGUF/bitnet.cpp export, and TurboQuant KV cache

If a future rerun fails, inspect projection-wise error, tune scaled-STE group
size/lambda/activation quant, and consider stabilization before any kernel work.
