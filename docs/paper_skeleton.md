# Paper / Report Skeleton — Teacher-Free b1.58 → I2_S on Dense LLaMA

Document position: [Index](./index.md) -> synthesis of RT-112..118. Pulls the
finished tracks into a paper outline, audits the gaps, and scopes the next
high-value reinforcing experiment (G1 budget scaling).

Related: [bitnet_cpp_export_scoping.md](./bitnet_cpp_export_scoping.md) (systems),
[quality_recovery_plan.md](./quality_recovery_plan.md) (quality),
[oss_architecture_audit.md](./oss_architecture_audit.md) (negative result).

## Working title

> Teacher-Free Ternary Recovery: small, fast, and still useful b1.58 LLaMA on
> commodity CPUs via per-tensor I2_S export

## Abstract (draft)

On-device LLMs for low-resource users are bottlenecked by per-token memory traffic,
not parameter count. We show that a dense LLaMA model materialized to per-tensor
b1.58 weights (Wq = gamma·T, gamma = mean|W|, T in {-1,0,+1}) exports **losslessly**
into the existing bitnet.cpp I2_S 2-bit runtime — no custom byte-writer and no custom
kernel — and that the resulting artifact is both smaller and faster on x86 CPUs, with
the benefit **growing with model size**. A one-shot ternary PTQ collapses quality, but
a **short, teacher-free, CE-only** adaptation of just the target linears recovers most
of the lost quality (90% at 160M), and that recovery is preserved through the I2_S
runtime to within 0.002 nats. We confirm the storage, speed, runtime-faithfulness, and
recovery results scale from ~10M to 1.1B parameters in the LLaMA family. Finally, we
report a negative result: for natively-low-bit MoE models (gpt-oss-20b, MXFP4) the
recipe adds essentially no storage benefit, scoping its value to dense FP-weight models.

## Core claims (3 + 1 negative)

```text
C1 (faithful export): per-tensor b1.58 Wq=gamma*T -> convert(f32) -> llama-quantize
   I2_S runs in real bitnet.cpp (x86) with f16/f32 CE parity. No Path B writer needed.
C2 (efficiency is a scale law): whole-file compression converges to the 16x
   target-linear floor and token-gen speedup GROWS with model size.
C3 (cheap teacher-free recovery): short CE-only adaptation of target linears recovers
   most of the FP->PTQ loss, and the I2_S runtime preserves it (+0.002 nats).
C4 (scope, negative): on natively-MXFP4 MoE (gpt-oss-20b), ternary adds <1GB / ~0 ROI;
   the recipe's value is dense FP-weight models, not already-low-bit MoE.
```

## Figures (numbers already in hand)

### Figure 1 — Storage scales toward the 16x floor (C2)

| model | params | whole-file i2_s/f32 | target-linear i2_s/f32 | source |
| --- | ---: | ---: | ---: | --- |
| tiny | ~10M | 0.450 | 0.0625 | RT-113 |
| Llama-160M | 160M | 0.196 | 0.0625 | RT-114 |
| TinyLlama-1.1B | 1.1B | 0.1149 | 0.0625 | RT-115 |

Story: target-linear ratio is scale-invariant (16x); whole-file converges to it as the
embedding/lm_head fixed cost shrinks as a fraction.

### Figure 2 — Token-gen speedup grows with scale (C2)

| model | i2_s tg t/s | vs f32 | vs f16 | source |
| --- | ---: | ---: | ---: | --- |
| tiny | ~595 | ~2x | ~2x | RT-113 |
| Llama-160M | 100.8 | 5.69x | 3.00x | RT-114 |
| TinyLlama-1.1B | 18.26 | 7.51x | 4.23x | RT-115 |

(llama-bench, x86, t=2. f32/f16 absolutes noisy on shared CPU; i2_s stable; report ratio.)

### Figure 3 — PTQ collapse → teacher-free CE recovery (C3)

| model | FP PPL | one-shot PTQ PPL | adapted PPL | recovered_fraction | train tokens | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Llama-160M | 23.3 | 115,808 | 52.0 | 0.905 | 0.61M | RT-116 |
| TinyLlama-1.1B (fixed budget) | 10.1 | 101,549 | 1,217 | 0.480 | 0.31M | TRAIN-002 |
| **TinyLlama-1.1B (budget-scaled)** | 10.1 | 101,549 | **162.5** | **0.698** | **4.92M** | RT-120 |

(WikiText-2, target linears only, teacher-free. The 1.1B jump 0.480 -> 0.698 shows the
fixed-budget result was under-trained: ~16x more tokens -> recovery climbs toward the
160M trend. Gap G1 resolved.)

### Figure 4 — Adapted I2_S preserves adapted F16 (C1+C3)

| model | adapted f16 PPL | adapted i2_s PPL | i2_s vs f16 (nats) | source |
| --- | ---: | ---: | ---: | --- |
| Llama-160M (adapted) | 134.84 | 135.11 | +0.0020 | RT-116 QR-003 |
| TinyLlama-1.1B (PTQ-budget adapted) | 1260.21 | 1263.12 | +0.0023 | TRAIN-002 QR-003 |
| TinyLlama-1.1B (budget-scaled adapted) | 219.35 | 216.13 | -0.0148 | RT-120 QR-003 |

Story: the recovered quality survives the int8-activation I2_S runtime essentially
unchanged at every scale and budget (|delta| <= 0.015 nats; sign sometimes favorable).

### Appendix figure — Why not gpt-oss? (C4, negative)

gpt-oss-20b is MXFP4 already: official mxfp4.gguf 12.11 GB, and Q2_K (2-bit) is 11.47 GB
— every quant bottoms at a ~11.5 GB floor, so ternary adds <1 GB. Plot the quant-vs-size
ladder to show the floor; conclude "wrong vehicle for ternary."

## Gap audit (what's missing before paper-grade claims)

| id | gap | severity | cheapest fix |
| --- | --- | --- | --- |
| G1 | ~~1.1B recovery only 0.48~~ RESOLVED (RT-120): L4 budget-scaled (4.92M tokens, eff batch 24, 800 steps) -> **0.698** (paper-useful tier); QR-003 i2_s vs f16 -0.0148 nats. Fixed-budget explanation confirmed. | DONE | (optional: 1200 steps toward 0.90) |
| G2 | ~~no recipe ablation~~ RESOLVED (QR-005): a/b/c on 160M -> +norms negligible (0.907 vs 0.906), +lm_head hurts (0.898). **Default = linears only.** | DONE | — |
| G3 | ~~+norms may lift the fraction~~ RESOLVED: it does not (within noise). Cheapest recipe is best. | DONE | — |
| G4 | ~~quality is CE/PPL only~~ RESOLVED (QR-004/RT-119): greedy panel shows PTQ token-salad -> adapted fluent English -> i2_s same tier as f16. Closed at 160M (base model: weak fluency/factuality — strengthen with G1 + better base) | DONE | — |
| G5 | no baseline comparison (RTN / GPTQ / AWQ / QAT) | MED | add at least RTN + one QAT point on 160M |
| G6 | single seed for recovery; no variance | LOW | 2-3 seeds on 160M QR-002a |
| G7 | cross-tool PPL gap (PyTorch CE vs llama.cpp perplexity) unexplained in-figure | LOW | one calibration note + measure both on identical tokens |
| G8 | only LLaMA family; generality unproven beyond it | LOW | (scope it honestly; gpt-oss negative already bounds the claim) |
| G9 | some raw Colab JSON volatile / not all archived in repo `reports/` | LOW | commit the rt11x_*.json artifacts |

## Next reinforcing experiment: G1 budget scaling (RT-120 / TRAIN-003)

Rationale: the cheap gaps are closed. QR-005 showed that the simplest recipe
(`target linears only`) is the default, and QR-004 showed human-visible recovery at
160M. The only high-severity gap left in the main figure is whether the 1.1B
recovery fraction `0.480` was merely under-budgeted.

Design:

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- Recipe: target linears only; no norms; no lm_head.
- Budget: raise effective training tokens from TRAIN-002's `0.31M` to about `4.9M`
  using gradient accumulation (`800` steps, effective batch `24`, seq `256`).
- Preferred hardware: A100 with fp32 AdamW; fallback: L4 with AdamW8bit.
- Primary metric: recovered_fraction.
- Runtime gate: adapted I2_S vs adapted F16 should stay near `+0.002` nats and pass
  if `<= +0.010` nats.

See [G1 Budget-Scaling Runbook](./g1_budget_scaling_runbook.md) for smoke commands,
one-shot A100/L4 commands, fallback rules, and result interpretation.

Decision order now: **G1 budget scaling -> G5 baseline -> G6 seed variance**.

## What NOT to do next

- Do not start a ternary-MoE / gpt-oss build (RT-118: ROI ~0).
- Do not spend L4/A100 without the G1 smoke and one-shot runbook.
- Do not claim user-facing quality from CE/PPL alone; QR-004 closes this only at
  160M, so 1.1B qualitative examples should follow a successful G1 run.
