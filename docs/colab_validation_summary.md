# Colab Validation Summary

Document position: [Index](./index.md) -> completed Colab validation milestone.

Related docs:

- [Colab Arena Runbook](./colab_arena_runbook.md)
- [Scaled-STE BitLinear Experiment](./scaled_ste_bitlinear_experiment.md)
- [Evolutionary LLM Arena Plan](./evolutionary_llm_arena_plan.md)

## Status

Date: 2026-06-24

Conclusion:

```text
PROCEED
```

`ScaledBitLinear` passed the Colab scale-up gate. It remained on the Pareto
frontier in the moderate run, was the quality winner across all three seed
sweep runs, stayed robust across the first group-size sweep, and passed the
activation fake-quant tiebreaker.

## Validation Checklist

| Step | Result | Notes |
| --- | --- | --- |
| `colab-mcp` connection | Pass | Cell add, execution, and polling worked |
| Environment cleanup | Pass | Removed editable `transformers 4.35.2`; used standard `transformers 4.57.6` with a clean clone |
| Faster smoke arena | Pass | `strict` passed; SSTE TC passed `3/3` |
| Moderate arena | Pass | `800` train steps; scaled-STE and projected-QAT were tied/competitive on the Pareto frontier |
| Seed sweep | Pass | Seeds `31`, `32`, `33` all returned `rc=0`; scaled-STE was quality winner `3/3` |
| Group-size sweep | Pass | Group sizes `32`, `64`, `128` all kept scaled-STE quality winner `3/3` and frontier `3/3`; loss stayed in a narrow `0.2875-0.2996` band |
| Activation fake-quant seed 31 | Borderline | No quality collapse. `bits=8` improved loss/KL but lost frontier by a tiny accuracy/RAM tie-break against projected-QAT |
| Activation fake-quant seeds 32/33 | Pass | `bits=8` made scaled-STE quality winner, resource winner, and Pareto member on both seeds |

## Research Interpretation

The Colab runs support the local result:

```text
ScaledBitLinear = S1 groupwise scale preservation + CE-only STE
```

This candidate is no longer just a local tiny smoke artifact. It is stable
enough across Colab seed, group-size, and activation fake-quant sweeps to
justify the next optimization stage.

The previous hold conditions are now satisfied:

- packed ternary kernel can move from "blocked" to "candidate next phase"
- GGUF or bitnet.cpp export can be scoped after sweep stability
- TurboQuant KV-cache work can be revisited as the runtime/memory side branch

## Activation Fake-Quant Borderline Result

Seed `31` is the known worst-case seed from the earlier sweep. On that seed,
activation fake-quant did not collapse quality, but it did nudge scaled-STE off
the Pareto frontier.

| Candidate | Bits | Accuracy | Loss | KL to fp16 | Fitness | Pareto |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `s1_scaled_ste_int4` | `0` | `0.908` | `0.291` | `0.092` | `1.210` | Yes |
| `s1_projected_qat_int4` | `0` | `0.907` | `0.294` | `0.063` | `1.212` | Yes |
| `s1_scaled_ste_int4` | `8` | `0.906` | `0.286` | `0.083` | `1.209` | No |
| `s1_projected_qat_int4` | `8` | `0.907` | `0.294` | `0.063` | `1.212` | Yes |

Interpretation:

```text
This is not activation collapse. Accuracy moved by only -0.002, loss improved,
and KL improved. The frontier miss is caused by a very small accuracy margin:
projected-QAT keeps slightly lower RAM proxy, so once scaled-STE loses its
accuracy edge by ~0.001, projected-QAT dominates it on the Pareto check.
```

Decision:

```text
Resolved by act8 seeds 32 and 33. This was seed-specific frontier noise, not
activation-quant collapse.
```

## Activation Fake-Quant Tiebreaker Result

| Seed | Candidate | Accuracy | Loss | KL to fp16 | Fitness | Pareto |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| `32` | `s1_scaled_ste_int4` | `0.918` | `0.261` | `0.084` | `1.221` | Yes |
| `32` | `s1_projected_qat_int4` | `0.909` | `0.285` | `0.076` | `1.214` | Yes |
| `33` | `s1_scaled_ste_int4` | `0.908` | `0.287` | `0.138` | `1.211` | Yes |
| `33` | `s1_projected_qat_int4` | `0.905` | `0.306` | `0.104` | `1.209` | Yes |

Final act8 decision:

```text
PASS
```

Activation fake-quant is robust enough for the next validation phase. The seed
31 frontier miss is now classified as a tiny Pareto-margin artifact.

Watch item:

```text
scaled-STE has slightly higher KL-to-fp16 than projected-QAT on the act8 runs,
even when scaled-STE has better accuracy, loss, and fitness.
```

This does not block progress, but it should be tracked in real-text validation
and any export/logit-equivalence checks.

## Artifact Note

The seed sweep JSON files were generated inside the Colab session:

```text
reports/tiny_real_arena_scaled_ste_colab_seed_31.json
reports/tiny_real_arena_scaled_ste_colab_seed_32.json
reports/tiny_real_arena_scaled_ste_colab_seed_33.json
```

The group-size sweep JSON files were also generated inside the Colab session:

```text
reports/tiny_real_arena_scaled_ste_colab_g32.json
reports/tiny_real_arena_scaled_ste_colab_g64.json
reports/tiny_real_arena_scaled_ste_colab_g128.json
```

The activation fake-quant JSON files were also generated inside the Colab
session:

```text
reports/tiny_real_arena_scaled_ste_colab_act0.json
reports/tiny_real_arena_scaled_ste_colab_act8.json
reports/tiny_real_arena_scaled_ste_colab_act8_seed32.json
reports/tiny_real_arena_scaled_ste_colab_act8_seed33.json
```

They were not committed from the local workspace and may be ephemeral. Treat
this document as the milestone record, not as a replacement for raw result
archival. Re-run the sweep before making a paper-style quantitative claim.

## Next Actions

Recommended order:

1. Archive sweep JSON reports from Colab back into `reports/`.
2. Move the arena from synthetic patterned tokens to a tiny real text subset.
3. Track KL-to-fp16 alongside CE loss, perplexity, token accuracy, and generation smoke.
4. Only after real-text validation, scope packed ternary kernels or export paths.

The immediate next experiment should be real tiny text validation. Runtime and
export work are now allowed, but real text removes more risk first.
