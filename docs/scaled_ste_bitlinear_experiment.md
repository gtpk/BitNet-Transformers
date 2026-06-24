# Scaled-STE BitLinear Experiment

Document position: [Index](./index.md) -> conversion ladder S3 candidate -> Colab scale-up gate.

Related docs:

- [Existing Model to BitNet Conversion Plan](./existing_model_to_bitnet_conversion_plan.md)
- [Evolutionary LLM Arena Plan](./evolutionary_llm_arena_plan.md)
- [Colab Arena Runbook](./colab_arena_runbook.md)
- [Colab Validation Summary](./colab_validation_summary.md)

## Purpose

The first BitLinear STE arena candidate was useful because it failed clearly:
it trained, but it discarded S1 groupwise scales and underperformed projected
QAT. This experiment adds the missing scale term while keeping the same
teacher-free constraint.

The candidate is still a conversion experiment, not native BitNet pretraining:

```text
start from dense checkpoint
replace target LLaMA projection linears with fake-quant BitLinear
train with ordinary next-token CE only
do not use teacher logits, hidden states, or attention maps
```

## Formula

For a linear weight block `W_g` with shape `[out_features, group_size]`:

```text
tau_g   = lambda * mean(abs(W_g), dim=input)
T_g     = sign(W_g) if abs(W_g) > tau_g else 0
alpha_g = sum(abs(W_g) * [T_g != 0]) / max(count(T_g != 0), 1)
W_q,g   = alpha_g * T_g
```

The layer uses straight-through estimation:

```text
forward_weight = stop_gradient(W_q - W) + W
```

So the forward path sees `alpha*T`, while the backward path updates the latent
full-precision weight `W`.

## Why This Matters

The earlier `BitLinear` reference path forwards only `T` and applies activation
quantization immediately. That is not equivalent to the S1 conversion policy,
where the approximation is `alpha*T`.

This experiment tests the narrower claim:

```text
If S1 projected-QAT works, then a native fake-quant layer that preserves the
same S1 alpha*T form should be at least competitive.
```

## TC Matrix

| ID | Check | Method | Pass |
| --- | --- | --- | --- |
| SSTE-001 | S1 equivalence | Compare `ScaledBitLinear.quantize_weight_groupwise()` with `conversion.S1` | max abs diff <= `1e-6` |
| SSTE-002 | STE gradient | Run MSE backward through `ScaledBitLinear` | finite non-zero `weight.grad` |
| SSTE-003 | Activation fake quant smoke | Enable 8-bit activation fake quant | finite output |
| ARENA-001 | Replacement | Replace target LLaMA q/k/v/o/gate/up/down projections | replacement count > 0 |
| ARENA-002 | Recovery | Scaled-STE CE loss decreases | final loss < initial loss |
| ARENA-003 | Selection | Resource-aware winner is not fp16 dense | winner differs from `fp16_dense` |

Run:

```bash
.venv/bin/python scripts/check_scaled_bitlinear.py --json-out reports/scaled_bitlinear_tc.json
.venv/bin/python scripts/run_tiny_real_arena.py --train-steps 200 --json-out reports/tiny_real_arena_scaled_ste_smoke.json --strict
```

## Local Result

Reports:

- `reports/scaled_bitlinear_tc.json`
- `reports/tiny_real_arena_scaled_ste_smoke.json`

Key numbers from the 200-step local CPU smoke:

| Candidate | Accuracy | Loss | Fitness |
| --- | ---: | ---: | ---: |
| `fp16_dense` | `0.1623` | `2.9265` | `0.1623` |
| `s1_projected_qat_int4_kv` | `0.1678` | `2.8013` | `0.4604` |
| `s1_bitlinear_ste_int4_kv` | `0.0605` | `3.8172` | `0.3530` |
| `s1_scaled_ste_int4_kv` | `0.1870` | `2.7100` | `0.4781` |

Recovery losses:

```text
projected_qat_loss: 2.9195 -> 2.6178
bitlinear_ste_loss: 4.2146 -> 3.5910
scaled_ste_loss   : 3.1535 -> 2.7202
```

Interpretation:

```text
The scaled-STE candidate is the first native BitLinear-style candidate in this
arena that beats projected-QAT on quality and resource-aware fitness.
The useful next scale-up is a Colab run with a larger tiny model and more
recovery steps, not a packed kernel yet.
```

## Colab Validation Result

See [Colab Validation Summary](./colab_validation_summary.md).

Summary:

- faster smoke arena passed with `strict`
- SSTE TC passed `3/3`
- moderate arena with `800` train steps passed
- scaled-STE and projected-QAT were tied/competitive on the Pareto frontier
- seed sweep over `31`, `32`, `33` passed `3/3`
- scaled-STE was quality winner `3/3`
- group-size sweep over `32`, `64`, `128` passed `3/3`
- scaled-STE stayed quality winner `3/3` and frontier `3/3` with loss in
  `0.2875-0.2996`
- activation fake-quant seed `31` did not collapse quality but missed the
  frontier by a tiny accuracy/RAM tie-break
- activation fake-quant seeds `32` and `33` passed; scaled-STE was quality
  winner, resource winner, and Pareto member on both
- watch item: scaled-STE KL-to-fp16 is slightly higher than projected-QAT even
  when scaled-STE has better accuracy/loss/fitness

Updated interpretation:

```text
The Colab result satisfies the scale-up gate. ScaledBitLinear is now stable
enough to justify sensitivity sweeps and the first runtime/export scoping work.
Packed kernels should still wait until group-size and activation fake-quant
sweeps confirm that the conversion recipe is not brittle.
Group-size is now checked; activation fake-quant is the remaining deployment
adjacent sensitivity gate.
Seed 31 act8 is a borderline miss, not a collapse, so seeds 32/33 should decide
whether the frontier miss is noise or a repeatable activation-quant issue.
Seeds 32/33 clear the tiebreaker. Move to real tiny text before runtime/export
work because KL-to-fp16 should be checked on real token distributions.
```

## Kill Criteria Before Bigger Runs

Do not spend Colab budget if any of these fail locally:

- `SSTE-001` fails: the layer no longer implements the S1 formula.
- `SSTE-002` fails: STE training is not meaningful.
- `ARENA-002` fails repeatedly across seeds.
- scaled-STE loses to projected-QAT by both quality and resource-aware fitness.

## Next Experiments

1. Archive Colab JSON reports into `reports/`.
2. Move from synthetic patterned data to a tiny real text subset.
3. Track CE/PPL/token accuracy/generation smoke/KL-to-fp16.
4. Then scope packed ternary kernels, GGUF/bitnet.cpp export, or TurboQuant KV.
