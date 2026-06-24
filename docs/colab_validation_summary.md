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

It then **passed real-text validation** (Wikitext-2, seeds 31/32/33, act0 and
act8): scaled-STE beat projected-QAT on accuracy, loss, PPL, and fitness in all
three seeds and stayed on the Pareto frontier. The first storage artifact
(packed ternary format, Phase 1) also landed and proved the b1.58 byte
reduction. See the Real-Text Validation Result and Packed Ternary Format
Milestone sections below.

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

## Real-Text Validation Result

Date: 2026-06-24

Real text removes the largest remaining quality risk before kernel/export work.
The arena was run in `--data-mode text` on a 200 KB Wikitext-2 sample
(byte-level tokenizer, `180177` train / `20020` eval tokens) at the moderate
config across seeds `31`, `32`, `33`, at both activation settings.

| Seed | Candidate | Acc | Loss | PPL | KL to fp16 | Fitness | Pareto |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `31` | `fp16_dense` | `0.435` | `1.911` | `6.76` | `0.000` | `0.435` | No |
| `31` | `s1_projected_qat_int4` | `0.453` | `1.879` | `6.55` | `0.113` | `0.758` | Yes |
| `31` | `s1_scaled_ste_int4` (act0) | `0.466` | `1.847` | `6.34` | `0.180` | `0.768` | Yes |
| `31` | `s1_scaled_ste_int4` (act8) | `0.469` | `1.828` | `6.22` | `0.168` | `0.772` | Yes |
| `32` | `fp16_dense` | `0.453` | `1.826` | `6.21` | `0.000` | `0.453` | No |
| `32` | `s1_projected_qat_int4` | `0.458` | `1.805` | `6.08` | `0.112` | `0.763` | Yes |
| `32` | `s1_scaled_ste_int4` (act0) | `0.477` | `1.784` | `5.95` | `0.152` | `0.780` | Yes |
| `32` | `s1_scaled_ste_int4` (act8) | `0.476` | `1.786` | `5.96` | `0.142` | `0.779` | Yes |
| `33` | `fp16_dense` | `0.424` | `1.956` | `7.07` | `0.000` | `0.424` | No |
| `33` | `s1_projected_qat_int4` | `0.431` | `1.926` | `6.86` | `0.101` | `0.735` | Yes |
| `33` | `s1_scaled_ste_int4` (act0) | `0.434` | `1.904` | `6.71` | `0.145` | `0.737` | Yes |
| `33` | `s1_scaled_ste_int4` (act8) | `0.437` | `1.897` | `6.67` | `0.148` | `0.740` | Yes |

Real-text decision:

```text
PASS
```

- scaled-STE (act0 and act8) beat projected-QAT on accuracy, loss, PPL, and
  fitness in all three seeds, and stayed on the Pareto frontier `3/3`.
- Generation smoke produced finite, non-degenerate decodes for every candidate.
- On real text, act8 did **not** hurt scaled-STE (seeds 31/33 slightly better
  than act0, seed 32 tied). This confirms the synthetic seed-31 act8 frontier
  miss was synthetic-specific noise.
- KL-to-fp16 watch item persists (scaled-STE higher than projected-QAT) but
  CE/PPL are better, so it is not a contradiction. Carry it into export/logit
  checks rather than treating it as a pause.

## Packed Ternary Format Milestone

Date: 2026-06-24

With real text passed, the first storage artifact landed (Phase 1 of
[Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md)):
`bitnet_llama/packing.py` + `scripts/check_packed_ternary.py`. This proves the
theoretical b1.58 maps to a real byte reduction.

```text
trit packing      : 1.600 bits/elem  (b1.58 bound = log2(3) = 1.585)
512x2048 layer    : trit 8.65x, two_bit 7.11x vs fp16
to_dense()        : == conversion.S1 alpha*T exactly (max_err 0.0)
ScaledBitLinear export round-trip : max_err 7e-9
TC PACK-001..006  : all pass
```

## Packed Model Export/Import Milestone

Date: 2026-06-24

Phase 2 was verified locally because it is pure packing plus logit comparison
and does not require GPU execution. This moves the claim from "each layer can
round-trip" to "the whole model preserves output after export/import."

```text
PACK-101 logit equality      : max_logit_err=0.00e+00
PACK-102 save/load artifact  : max_logit_err=0.00e+00
PACK-103 whole-model storage : 14 layers packed, 3.78x vs fp16
                               203.3 KB packed+fp16-others vs 769.2 KB fp16
```

The whole-model compression ratio is lower than the layer-only `8.65x` number
because embedding, `lm_head`, norms, and biases remain fp16. That is the right
number to track for end-to-end artifacts.

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

The real-text JSON files (with the `data/wikitext_tiny.txt` 200 KB sample) were
also generated inside the Colab session:

```text
reports/tiny_real_arena_text_wikitext_seed31.json
reports/tiny_real_arena_text_wikitext_seed32.json
reports/tiny_real_arena_text_wikitext_seed33.json
reports/tiny_real_arena_text_wikitext_act8_seed31.json
reports/tiny_real_arena_text_wikitext_act8_seed32.json
reports/tiny_real_arena_text_wikitext_act8_seed33.json
```

They were not committed from the local workspace and may be ephemeral. Treat
this document as the milestone record, not as a replacement for raw result
archival. Re-run the sweep before making a paper-style quantitative claim.

## Next Actions

Synthetic gates, real-text validation, and packed-format Phase 1/2 are all done.
Recommended order from here:

1. Archive the real-text JSON reports from Colab back into `reports/` or rerun
   the sweep before paper-style quantitative claims.
2. Packed format Phase 3: implement `PackedTernaryLinear` as a reference
   runtime module that holds packed weights and unpacks on-the-fly for
   `F.linear`.
3. Check packed-module model logits against the S1-unpacked model.
4. Use the export/runtime logit checks to watch the KL-to-fp16 item; only then
   scope a real kernel target (CPU first, then Metal/CUDA/bitnet.cpp).

See [Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md) for the
format spec and TC matrix.
