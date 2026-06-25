# Paper / Report Skeleton — Teacher-Free b1.58 → I2_S on Dense LLaMA

Document position: [Index](./index.md) -> synthesis of RT-112..118. Pulls the
finished tracks into a paper outline, audits the gaps, and scopes the one cheap
reinforcing experiment (QR-002b).

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

| model | FP PPL | one-shot PTQ PPL | adapted PPL | recovered_fraction | source |
| --- | ---: | ---: | ---: | ---: | --- |
| Llama-160M | 23.3 | 115,808 | 52.0 | 0.905 | RT-116 |
| TinyLlama-1.1B | 10.1 | 101,549 | 1,217 | 0.480 | TRAIN-002 |

(WikiText-2, 300 steps, target linears only, teacher-free. 1.1B lower = fixed-budget,
see Gap G1.)

### Figure 4 — Adapted I2_S preserves adapted F16 (C1+C3)

| model | adapted f16 PPL | adapted i2_s PPL | i2_s vs f16 (nats) | source |
| --- | ---: | ---: | ---: | --- |
| Llama-160M | 134.84 | 135.11 | +0.0020 | RT-116 QR-003 |
| TinyLlama-1.1B | 1260.21 | 1263.12 | +0.0023 | TRAIN-002 QR-003 |

Story: the recovered quality survives the int8-activation I2_S runtime essentially
unchanged, at both scales.

### Appendix figure — Why not gpt-oss? (C4, negative)

gpt-oss-20b is MXFP4 already: official mxfp4.gguf 12.11 GB, and Q2_K (2-bit) is 11.47 GB
— every quant bottoms at a ~11.5 GB floor, so ternary adds <1 GB. Plot the quant-vs-size
ladder to show the floor; conclude "wrong vehicle for ternary."

## Gap audit (what's missing before paper-grade claims)

| id | gap | severity | cheapest fix |
| --- | --- | --- | --- |
| G1 | 1.1B recovery is only 0.48 (fixed budget, batch 4, 8-bit Adam) | HIGH | budget-scaled 1.1B run on L4/A100 (fp32 Adam, batch↑, 500-800 steps) — AFTER G3 |
| G2 | ~~no recipe ablation~~ RESOLVED (QR-005): a/b/c on 160M -> +norms negligible (0.907 vs 0.906), +lm_head hurts (0.898). **Default = linears only.** | DONE | — |
| G3 | ~~+norms may lift the fraction~~ RESOLVED: it does not (within noise). Cheapest recipe is best. | DONE | — |
| G4 | ~~quality is CE/PPL only~~ RESOLVED (QR-004/RT-119): greedy panel shows PTQ token-salad -> adapted fluent English -> i2_s same tier as f16. Closed at 160M (base model: weak fluency/factuality — strengthen with G1 + better base) | DONE | — |
| G5 | no baseline comparison (RTN / GPTQ / AWQ / QAT) | MED | add at least RTN + one QAT point on 160M |
| G6 | single seed for recovery; no variance | LOW | 2-3 seeds on 160M QR-002a |
| G7 | cross-tool PPL gap (PyTorch CE vs llama.cpp perplexity) unexplained in-figure | LOW | one calibration note + measure both on identical tokens |
| G8 | only LLaMA family; generality unproven beyond it | LOW | (scope it honestly; gpt-oss negative already bounds the claim) |
| G9 | some raw Colab JSON volatile / not all archived in repo `reports/` | LOW | commit the rt11x_*.json artifacts |

## Next reinforcing experiment: QR-002b (+norms) — design

Rationale: cheapest, highest-signal fix for G2/G3. Adapting only target linears leaves
LayerNorm scales fixed at their FP values while the linear distributions shift under
ternarization; letting norms move may absorb quantization drift and lift
recovered_fraction at near-zero extra cost (norms are ~0.1M params).

Design:
- Model: Llama-160M (fast iteration; promote to 1.1B only if it helps).
- Arms (same 300-step / seq256 / batch8 / lr2e-4 budget as RT-116):
  - QR-002a: target linears only (baseline, recovered 0.905) — already have it.
  - QR-002b: target linears + all RMSNorm weights (`*layernorm*`, `model.norm`).
  - QR-002c: target linears + norms + lm_head (lm_head is 8.2M params — bigger).
- Metric: recovered_fraction = (CE_ptq - CE_adapted)/(CE_ptq - CE_fp), same eval set.
- Pass: QR-002b > QR-002a recovered_fraction by a clear margin -> adopt +norms as the
  default recipe and re-run the 1.1B point (G1) with +norms before spending L4/A100.
- Driver change: `scripts/rt116_quality_recovery.py` needs `--train-norms` /
  `--train-lm-head` flags that add those parameters to the trainable set (currently it
  freezes everything but PerTensorBitLinear weights). Small, additive change.

Decision order: G2/G3 (QR-002b, cheap) -> if it helps, G1 (1.1B budget-scaled with the
improved recipe) -> G4/G5 (prompt panel + baselines) for paper-grade quality claims.

UPDATE (QR-005 done): G2/G3 resolved — +norms negligible, +lm_head hurts, so the
**default recipe is target-linears-only**. Next is therefore G1 directly (1.1B
budget-scaled, linears-only, fp32 Adam + bigger batch on an L4/A100), then G4/G5.

## What NOT to do next

- Do not start a ternary-MoE / gpt-oss build (RT-118: ROI ~0).
- Do not spend L4/A100 on 1.1B budget scaling BEFORE QR-002b tells us the best recipe.
- Do not claim user-facing quality from CE/PPL alone (needs G4).
