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
sweep runs, and stayed robust across the first group-size sweep.

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

## Research Interpretation

The Colab runs support the local result:

```text
ScaledBitLinear = S1 groupwise scale preservation + CE-only STE
```

This candidate is no longer just a local tiny smoke artifact. It is stable
enough across the first Colab seed and group-size sweeps to justify the next
optimization stage.

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
Do not pause yet. Run act8 on seeds 32 and 33 as the tiebreaker.
```

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

They were not committed from the local workspace and may be ephemeral. Treat
this document as the milestone record, not as a replacement for raw result
archival. Re-run the sweep before making a paper-style quantitative claim.

## Next Actions

Recommended order:

1. Activation fake-quant tiebreaker: `SCALED_STE_ACTIVATION_BITS=8` on seeds `32`, `33`.
2. Archive sweep JSON reports from Colab back into `reports/`.
3. Move the arena from synthetic patterned tokens to a tiny real text subset if act8 does not collapse.
4. Only after the above, scope packed ternary kernels or export paths.

The immediate next experiment should be the act8 seed `32/33` tiebreaker. If
both also fall off frontier despite stable loss/accuracy, activation quant
needs tuning before runtime/export work. If they remain frontier or near-frontier
without quality collapse, proceed to the real tiny text subset.
