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

## Packed Runtime Module Milestone

Date: 2026-06-24

Phase 3 verified a runtime-facing module, not just a saved artifact.
`PackedTernaryLinear` holds uint8 packed codes and scale buffers with no dense
`[out,in]` float weight parameter, then reconstructs `alpha*T` on-the-fly for a
reference `F.linear`.

```text
PACK-201 layer forward      : max_err=0.00e+00
PACK-202 model logits       : 14 modules swapped, max_err=0.00e+00
PACK-203 no dense weight    : uint8 codes, no float weight parameter
PACK-204 state round-trip   : max_err=0.00e+00
target linear storage       : 74.0 KB packed vs 640.0 KB fp16, 8.65x
```

Important limitation: this is still a reference runtime. Forward currently
materializes dense `alpha*T` before calling `F.linear`, so storage/load memory is
reduced but compute-time peak memory and latency are not solved yet. That is the
Phase 4 job.

## Blocked Dequant Matmul Reference Milestone

Date: 2026-06-24

Phase 4 answered the memory part of the runtime question: packed weights can be
used for matmul without ever materializing the full dense `[out,in]` weight.
The reference path walks output-row chunks, unpacks only that chunk, applies
groupwise scales, and accumulates with `F.linear`-equivalent math.

```text
PACK-301 correctness     : max_err=0.00e+00
PACK-302 working set     : 8192 vs 65536 weight elements, 8.0x smaller
PACK-303 fused module    : max_logit_err=0.00e+00
PACK-304 latency honesty : dense 0.339 ms, blocked 0.399 ms, 1.2x slower
```

Interpretation:

```text
Memory win: real, at the reference working-set level.
Speed win : not yet. Python-loop blocked matmul is slower, as expected.
```

The Python/PyTorch reference ladder is now complete. Further latency work needs
either a real fused kernel or an export path into an optimized ternary runtime.

## Export Mapping Scoping Milestone

Date: 2026-06-24

GGUF/bitnet.cpp Step 0/1 was completed. The ternary value domain and 2-bit-style
packing family are compatible enough to investigate, but the scale model is not
lossless for this project.

```text
This project      : groupwise alpha[out, in/group], lambda-threshold ternary
I2_S-style export : per-tensor scale, absmean round/clamp ternary
Mapping decision  : lossy re-quantization
```

The local mapping check compared groupwise S1 against per-tensor b1.58 on the
same tiny Llama-shaped fixture:

```text
groupwise output error : 0.4339
per-tensor output err  : 0.5139
relative degradation   : +18.4%
affected layers        : 14 / 14
```

This first mapping check only ruled out **post-hoc** groupwise -> per-tensor
export. It did not prove that per-tensor b1.58 itself is weak. The follow-up
gate therefore added a native per-tensor candidate and measured real-text CE/PPL.

The arena candidate now exists:

```text
s1_scaled_ste_export_pt_int8_kv
s1_scaled_ste_export_pt_int4_kv
```

Local fixture smoke is directionally negative but not decisive:

```text
groupwise scaled-STE int4 : acc 0.311, loss 2.400
per-tensor export int4    : acc 0.274, loss 2.472
```

The authoritative native per-tensor gate below is the decision record.

## Per-Tensor Native Gate Result (decisive)

Date: 2026-06-24

The authoritative Colab Wikitext gate (seeds 31/32/33) compared the groupwise
baseline, a post-hoc per-tensor export of the groupwise model, and a per-tensor
b1.58 model trained natively with CE-only STE (`per_tensor_ste_native`).

| Seed | groupwise PPL | post-hoc export PPL | native per-tensor PPL | native acc | native KL | native Pareto |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `31` | `6.34` | `9.85` | `6.28` | `0.451` | `0.175` | No* |
| `32` | `5.95` | `10.55` | `6.01` | `0.468` | `0.165` | Yes |
| `33` | `6.71` | `11.31` | `6.71` | `0.439` | `0.149` | Yes (q+r winner) |

*seed 31 native per-tensor has lower PPL than groupwise but falls off the
frontier on a tiny fitness/RAM tie-break.

Decision:

```text
PASS -> direct I2_S export is viable via per-tensor-native training.
```

- Native per-tensor PPL is within +-1% of groupwise on all three seeds
  (-0.9% / +1.0% / 0.0%), far inside the +5-10% pass threshold.
- It stays on the Pareto frontier 2/3 (outright quality+resource winner on seed 33).
- KL-to-fp16 (~0.15-0.18) matches groupwise; nothing like the 0.55+ of post-hoc export.
- Generation smoke finite + non-degenerate on all three (decodes to English).
- Native per-tensor (PPL ~6) crushes post-hoc export (PPL ~10), which proves the
  earlier export loss was a post-hoc conversion artifact, not a per-tensor weakness.

Consequence: the export track does not need a groupwise GGUF extension or a custom
kernel. Train per-tensor b1.58 native, then export to bitnet.cpp I2_S.

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

Synthetic gates, real-text validation, and packed-format Phase 1/2/3/4 reference are all done.
Recommended order from here:

1. Archive the real-text JSON reports from Colab back into `reports/` or rerun
   the sweep before paper-style quantitative claims.
2. Define I2_S-style export TCs for the per-tensor-native path: artifact
   loadability, logit equality against the Python reference, storage ratio, PPL
   on tiny real text, and latency/memory against the Python reference path.
3. Implement the exporter from a `per_tensor_ste_native` state_dict to the target
   bitnet.cpp/GGUF I2_S layout.
4. Keep groupwise GGUF extension and custom fused kernels as fallback tracks only
   if the direct I2_S artifact/runtime path fails.

See [Packed Ternary Weight Format Plan](./packed_ternary_format_plan.md) for the
format spec and TC matrix.
