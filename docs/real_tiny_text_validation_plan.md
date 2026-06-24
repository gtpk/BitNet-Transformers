# Real Tiny Text Validation Plan

Document position: [Index](./index.md) -> next validation phase after Colab synthetic gates.

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

Add a real-text mode to the arena instead of creating a separate one-off
notebook. The preferred shape is:

```text
scripts/run_tiny_real_arena.py
  --data-mode synthetic|text
  --text-path <path>
  --tokenizer <hf-tokenizer-or-local>
```

If tokenizer/dataset setup becomes too heavy, use a small committed text
fixture first and keep the interface compatible with a later tokenizer-backed
path.

## Decision After This Phase

If real text passes:

1. archive Colab JSON reports
2. scope packed ternary kernel storage and matmul path
3. scope GGUF/bitnet.cpp export path
4. revisit TurboQuant KV cache for long-context runtime memory

If real text fails:

1. inspect projection-wise error
2. tune scaled-STE group size, lambda, and activation quant
3. consider Extra RMSNorm/stabilization before kernels
