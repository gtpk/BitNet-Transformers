# Colab Arena Runbook

Document position: [Index](./index.md) -> next executable scale-up step.

Related docs:

- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [Evolutionary LLM Arena Plan](./evolutionary_llm_arena_plan.md)
- [Existing Model to BitNet Conversion Plan](./existing_model_to_bitnet_conversion_plan.md)

## Goal

Run the teacher-free BitNet conversion arena at a size that is still cheap on
Colab but large enough to be more informative than the local CPU smoke.

This run tests:

```text
dense fp16 baseline
S0/S1 ternary PTQ
S1 projected-QAT
current BitLinear STE
scaled-STE BitLinear
runtime policies with packed b1.58 weights plus fp16/int8/int4 KV estimates
```

## Colab Setup

Use a GPU runtime, then run:

```python
!git clone https://github.com/gtpk/BitNet-Transformers.git
%cd BitNet-Transformers
!bash scripts/run_colab_scaled_ste_arena.sh
```

If this commit is not pushed yet, use a local upload or push the branch first.

## Faster Smoke

For a quick sanity check:

```python
%env TRAIN_STEPS=200
%env QAT_STEPS=48
%env STE_QAT_STEPS=48
%env SCALED_STE_STEPS=48
%env HIDDEN_SIZE=64
%env INTERMEDIATE_SIZE=128
%env NUM_LAYERS=1
%env SEQ_LEN=32
!bash scripts/run_colab_scaled_ste_arena.sh
```

Expected local-like signal:

```text
SSTE checks pass
scaled_ste_loss decreases
s1_scaled_ste_int4_kv is competitive with or better than projected-QAT
```

## Moderate Colab Run

Default script settings:

```text
TRAIN_STEPS=800
QAT_STEPS=128
STE_QAT_STEPS=128
SCALED_STE_STEPS=128
HIDDEN_SIZE=128
INTERMEDIATE_SIZE=256
NUM_LAYERS=2
SEQ_LEN=64
BATCH_SIZE=32
EVAL_BATCH_SIZE=128
```

Command:

```python
!bash scripts/run_colab_scaled_ste_arena.sh
```

Outputs:

```text
reports/scaled_bitlinear_tc_colab.json
reports/tiny_real_arena_scaled_ste_colab.json
```

## Sweep Commands

Seed sweep:

```python
for seed in [31, 32, 33]:
    !SEED={seed} ARENA_JSON_OUT=reports/tiny_real_arena_scaled_ste_colab_seed_{seed}.json bash scripts/run_colab_scaled_ste_arena.sh
```

Group-size sweep:

```python
for group_size in [32, 64, 128]:
    !SCALED_STE_GROUP_SIZE={group_size} ARENA_JSON_OUT=reports/tiny_real_arena_scaled_ste_colab_g{group_size}.json bash scripts/run_colab_scaled_ste_arena.sh
```

Activation fake-quant sweep:

```python
for bits in [0, 8]:
    !SCALED_STE_ACTIVATION_BITS={bits} ARENA_JSON_OUT=reports/tiny_real_arena_scaled_ste_colab_act{bits}.json bash scripts/run_colab_scaled_ste_arena.sh
```

## Reading The Result

The result JSON contains:

```text
quality_winner
resource_winner
projected_qat_loss_start/end
bitlinear_ste_loss_start/end
scaled_ste_loss_start/end
results[]
```

The important comparison is:

```text
s1_projected_qat_int4_kv vs s1_scaled_ste_int4_kv
```

Proceed if scaled-STE wins or remains on the Pareto frontier across at least
two seeds. Pause if only one seed wins or if activation fake quant collapses.

## Current Local Baseline

Local CPU 200-step result:

```text
quality winner        : s1_scaled_ste_int8_kv
resource-aware winner : s1_scaled_ste_int4_kv
scaled_ste_loss       : 3.1535 -> 2.7202
projected_qat_loss    : 2.9195 -> 2.6178
```

This is enough to justify a Colab run. It is not enough to justify packed kernel
work yet.
