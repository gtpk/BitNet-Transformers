# Evolutionary LLM Arena Plan

Document position: [Index](./index.md) -> candidate selection and resource-aware evaluation.

Related docs:

- [Memory-Traffic-First BitNet Plan](./memory_traffic_first_plan.md)
- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [Colab Arena Runbook](./colab_arena_runbook.md)
- [Colab Validation Summary](./colab_validation_summary.md)

## Purpose

This document defines a small, testable arena for evolving low-resource LLM candidates. The goal is not to claim that models magically self-improve. The goal is to build a measurement loop where architecture, quantization, runtime policy, and task data compete under resource pressure.

The arena is useful only if it can prove the following:

```text
quality signal > evaluation noise
resource-aware selection changes the winner
Pareto frontier exists
adversarial tasks stay valid and useful
holdout quality does not collapse
generation cost stays inside budget
```

## Core Objects

```text
m in M: model/runtime candidate
t in T: task/environment sample
q(m, t): task quality score in [0, 1]
r(m): resource cost, e.g. bytes/token, latency, RAM
```

Model fitness:

```text
F(m; D) = E_{t~D}[q(m,t)]
        - lambda_b * normalized_bytes(m)
        - lambda_l * normalized_latency(m)
        - lambda_r * normalized_ram(m)
```

Task/adversary fitness:

```text
A(t; P) = E_{m~P}[1 - q(m,t)]
        + beta * novelty(t)
        - gamma * invalidity(t)
        - penalty_if_too_easy_or_too_hard(t)
```

## Relationship To GANs

The arena is GAN-like only at the high level:

```text
model candidates try to survive tasks
adversaries try to expose model failures
```

It is not a standard differentiable GAN. Most mutations are non-differentiable:

- BitNet weight format
- KV cache policy
- tokenizer
- data mixture
- CPU/GPU/Metal runtime path
- packed kernel layout
- task generator rules

This is closer to adversarial co-evolution, population-based training, AutoML, red teaming, and self-play.

## Low-Resource Fitness

For this project, quality alone is not enough. A model that is slightly weaker but much cheaper may be better.

Primary axes:

- task quality
- bytes moved per generated token
- latency
- peak RAM
- energy proxy
- holdout robustness

Selection should use a Pareto frontier plus a scalar score for quick smoke tests.

## Feasibility Gates

### Gate 1: Fitness Signal-To-Noise

For two candidates `a` and `b`:

```text
delta = mean(F_a - F_b)
SE    = std(F_a - F_b) / sqrt(n)
```

Selection is meaningful only if:

```text
abs(delta) > k * SE
```

The smoke runner uses `k = 2` by default.

### Gate 2: Pareto Frontier

Candidate `a` dominates candidate `b` if:

```text
quality(a) >= quality(b)
bytes(a)   <= bytes(b)
latency(a) <= latency(b)
ram(a)     <= ram(b)
```

and at least one inequality is strict.

The arena is interesting if the frontier has multiple candidates or resource-aware selection differs from quality-only selection.

### Gate 3: Adversarial Task Validity

Good adversarial tasks should not be impossible or trivial.

Target pass-rate range:

```text
0.2 <= pass_rate(task) <= 0.8
```

Invalid task generation must stay low.

### Gate 4: Holdout Safety

Arena-generated tasks can overfit the population. A holdout set must remain stable:

```text
delta_holdout >= -epsilon
```

### Gate 5: Budget

One generation cost must be bounded:

```text
C_gen = N_models * N_tasks * C_eval + N_train * C_train
```

If this exceeds local budget, the arena is not a low-resource method.

## MVP

The first runner does not train real LLMs. It simulates candidates and tasks to verify that the selection machinery is not nonsense.

Candidates:

- fp16 tiny baseline
- int8 weight baseline
- current PyTorch BitLinear reference
- packed b1.58 weight
- packed b1.58 + int8 KV
- packed b1.58 + int4 KV
- packed b1.58 + QAT recovery

Tasks:

- arithmetic
- retrieval
- code
- summarization
- long-context stress

Metrics:

- mean quality
- resource-adjusted fitness
- pairwise separability
- Pareto frontier
- adversarial task difficulty
- holdout stability proxy

## Next Step After Smoke

If the synthetic smoke passes, replace simulated candidate scores with measured values:

1. Use `estimate_memory_traffic.py` for bytes/token.
2. Use tiny local models for actual loss/logit metrics.
3. Use CPU wall-clock timing for latency.
4. Add Colab GPU jobs only when model training or real batch evaluation becomes necessary.

## Current Local Smoke Results

The first local runners are in place:

```bash
.venv/bin/python scripts/run_arena_feasibility.py --strict
.venv/bin/python scripts/run_tiny_real_arena.py --train-steps 200 --json-out reports/tiny_real_arena_smoke_200.json --strict
```

Synthetic arena smoke:

- quality winner: `fp16_tiny_baseline`
- resource-aware winner: `packed_b1_58_qat_recovery`
- signal-to-noise, Pareto frontier, adversary validity, and holdout checks passed
- report: `reports/arena_feasibility_smoke.json`

Tiny real-model arena smoke:

- trains a tiny LLaMA on patterned sequences locally on CPU
- quality winner: `fp16_dense`
- resource-aware winner: `s1_groupwise_ptq_int4_kv`
- Pareto frontier: `fp16_dense`, `s1_groupwise_ptq_int4_kv`
- report: `reports/tiny_real_arena_smoke_200.json`

Projected QAT recovery smoke:

```bash
.venv/bin/python scripts/run_tiny_real_arena.py --train-steps 200 --json-out reports/tiny_real_arena_qat_smoke.json --strict
.venv/bin/python scripts/run_tiny_real_arena.py --train-steps 200 --json-out reports/tiny_real_arena_ste_qat_smoke.json --strict
```

- starts from S1 groupwise ternary projection
- uses CE loss only, no teacher logits/hidden states
- projects target linear weights back to S1 after each recovery step
- quality winner: `s1_projected_qat_int8_kv`
- resource-aware winner: `s1_projected_qat_int4_kv`
- projected QAT loss: `2.9195 -> 2.6178`
- report: `reports/tiny_real_arena_qat_smoke.json`

BitLinear STE reference smoke:

- replaces LLaMA attention and MLP projection linears with the current `BitLinear`
  reference layer
- initializes those layers from the S1 ternary code `T`, not from `alpha*T`,
  because the current `BitLinear` layer does not preserve S1 scale factors
- replaced layers: `7`
- BitLinear STE recovery loss: `4.2146 -> 3.5910`
- evaluated loss/accuracy: `3.8172 / 0.0605`
- quality winner remains `s1_projected_qat_int8_kv`
- resource-aware winner remains `s1_projected_qat_int4_kv`
- report: `reports/tiny_real_arena_ste_qat_smoke.json`

Scaled-STE BitLinear smoke:

```bash
.venv/bin/python scripts/check_scaled_bitlinear.py --json-out reports/scaled_bitlinear_tc.json
.venv/bin/python scripts/run_tiny_real_arena.py --train-steps 200 --json-out reports/tiny_real_arena_scaled_ste_smoke.json --strict
```

- adds `ScaledBitLinear`, which forwards S1-style `alpha*T` and uses STE for
  the latent full-precision weight
- TC checks confirm S1 `alpha*T` equivalence, finite STE gradients, and finite
  activation fake-quant output
- replaced layers: `7`
- scaled-STE recovery loss: `3.1535 -> 2.7202`
- quality winner: `s1_scaled_ste_int8_kv`
- resource-aware winner: `s1_scaled_ste_int4_kv`
- Pareto frontier: `s1_projected_qat_int4_kv`, `s1_scaled_ste_int4_kv`
- reports: `reports/scaled_bitlinear_tc.json`,
  `reports/tiny_real_arena_scaled_ste_smoke.json`

Colab validation:

- faster smoke arena passed with `strict`
- moderate arena with `800` train steps passed
- scaled-STE and projected-QAT remained tied/competitive on the Pareto frontier
- seed sweep `31/32/33` passed with `rc=0` for all runs
- scaled-STE was quality winner `3/3`
- group-size sweep `32/64/128` passed with scaled-STE quality winner `3/3`
  and frontier `3/3`
- activation fake-quant seed `31` showed no quality collapse, but act8 missed
  the frontier by a tiny accuracy/RAM tie-break
- activation fake-quant seeds `32/33` passed with scaled-STE quality winner,
  resource winner, and Pareto membership on both seeds
- milestone: [Colab Validation Summary](./colab_validation_summary.md)

Interpretation:

```text
quality-only selection and low-resource selection already diverge.
Projected QAT shows that CE-only post-training recovery is a meaningful next
candidate before moving larger jobs to Colab.
The current BitLinear STE reference path can train, but it underperforms the
scaled S1 projected-QAT path. This points to a specific implementation gap:
native BitLinear should preserve groupwise scales, or the arena should add a
separate scaled-STE BitLinear candidate before spending Colab budget.
Scaled-STE closes that gap in the tiny local arena and is now the first native
BitLinear-style candidate worth scaling in Colab.
The first Colab seed sweep keeps scaled-STE on the frontier and makes it the
quality winner in all three seeds, so the arena can proceed to sensitivity
sweeps before runtime kernel work.
The group-size sweep narrows risk around S1 scale granularity; activation
fake-quant is the remaining near-runtime sensitivity check.
Seed 31 act8 is inconclusive rather than failing: the quality metrics remain
healthy, while Pareto status flips because projected-QAT keeps a slightly lower
RAM proxy. Use seed 32/33 as the tiebreaker.
Seed 32/33 clear the tiebreaker, so activation fake-quant is not blocking.
The remaining risk is whether the synthetic task signal transfers to real text.
```

Next step:

```text
Move to real tiny text validation, archive raw JSON reports, and keep KL-to-fp16
as a watch metric before packed-kernel or export work.
```
