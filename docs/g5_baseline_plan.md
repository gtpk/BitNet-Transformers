# G5 Baseline Comparison Plan (RT-121 / BASE-001)

Document position: [Index](./index.md) -> after [G1 runbook](./g1_budget_scaling_runbook.md)
and [Paper Skeleton](./paper_skeleton.md). Closes paper gap G5.

## The reviewer question this answers

Our internal story is closed (PTQ collapse -> teacher-free CE recovery -> I2_S runtime
preservation, scaling 160M->1.1B). The first external question is:

```text
"Why not just use an existing quantization or QAT method?"
```

G5 builds a fair baseline panel so the paper can answer that directly, instead of only
comparing ours-to-ours.

## Design principle: one tool, one eval, one table

All perplexities are measured with the SAME `llama-perplexity` binary on the SAME
`eval.txt` (the held-out WikiText slice), so the numbers are directly comparable and
free of the PyTorch-vs-llama.cpp cross-tool gap that muddied earlier comparisons. We
also report **bits/weight on the target linears** and **whether the method trains**, so
the comparison is honest about cost, not just PPL.

Start on **Llama-160M** (cheap, easy to interpret, blocks reviewer questions early);
promote the winners to TinyLlama-1.1B only if needed.

## Baseline panel (160M, WikiText eval, llama-perplexity)

| id | method | target-linear bits | trains? | how to produce |
| --- | --- | ---: | --- | --- |
| B0 | FP reference | 16 (f16) | no | convert FP -> `ggml-model-f16.gguf` |
| B1 | RTN ternary one-shot (= our PTQ, no recovery) | ~1.58 | no | materialize Wq=gamma*T -> I2_S gguf |
| B2 | llama.cpp `Q2_K` one-shot | ~2.6 | no | `llama-quantize FP.f32 ... Q2_K` |
| B3 | llama.cpp `Q3_K_M` one-shot | ~3.4 | no | `llama-quantize ... Q3_K_M` |
| B4 | llama.cpp `Q4_0` one-shot | ~4.5 | no | `llama-quantize ... Q4_0` |
| **OURS** | per-tensor b1.58 + teacher-free CE (linears-only) | ~1.58 | yes (cheap) | adapted -> I2_S gguf (RT-116 recipe) |
| B5 | ternary QAT WITHOUT our scale (contrast) | ~1.58 | yes | same CE loop, but sign/absmax STE instead of absmean gamma |
| A1 (appendix) | GPTQ or AWQ 4-bit | 4 | calib | auto-gptq / autoawq, different bit budget |

Notes:
- B0/B1/OURS already exist from RT-116 (FP f16, PTQ i2_s, adapted i2_s). B2/B3/B4 are
  one `llama-quantize` call each on the same FP f32 GGUF — essentially free.
- B2 (`Q2_K`) is the key "why not existing 2-bit" point: a widely-used one-shot ~2.6-bit
  quant. If OURS (1.58-bit + a short CE pass) beats B2, the training cost is justified.
- B5 isolates our *method's* contribution from "any ternary QAT": same teacher-free CE,
  but replace the per-tensor absmean-gamma STE with a sign x absmax (or no-scale) STE.
  This answers "is the per-tensor b1.58 scale doing the work, or just QAT?".
- A1 (GPTQ/AWQ) is a *different bit budget* (4-bit) — appendix only, frames where 1.58-bit
  OURS sits relative to mature 4-bit one-shot methods. Needs extra libs (auto-gptq/awq);
  if integration is fiddly, use `Q4_0`/`Q4_K_M` as the 4-bit one-shot proxy (B4).

## Metrics & table to produce

```text
method | bits/wt (target lin) | trains? | PPL (llama-perplexity, eval.txt) | whole-file MB
```

Plus a one-line "PPL vs bits" reading: at ~2-bit, OURS vs B1/B2; at 4-bit, B4/A1 as the
more-bits reference.

## Pass / interpretation

```text
OURS PPL << B1 (RTN ternary no-train)
  -> the teacher-free CE pass is what makes ternary viable (expected; quantifies it).

OURS PPL <= B2 (Q2_K one-shot, MORE bits)
  -> strong: 1.58-bit + cheap CE beats a standard one-shot 2-bit. Headline baseline result.

OURS PPL ~ B3/B4 (3-4 bit one-shot)
  -> ours reaches higher-bit one-shot quality at ~1.58 bits + a short train. Best case.

OURS PPL >> B2
  -> honest negative: at this budget existing one-shot 2-bit is better; reframe the claim
     around runtime speed / memory-traffic, not PPL-per-bit.

OURS ~ B5 (no-scale ternary QAT)
  -> the win is "QAT", not our scale; soften the per-tensor-b1.58 emphasis.
OURS << B5
  -> the per-tensor absmean gamma is a real contributor; keep it central.
```

## Execution order

1. **Cheap first (no new training):** B0/B1/OURS already have GGUFs; add B2/B3/B4 by
   `llama-quantize` on the FP f32 GGUF; run `llama-perplexity` on all on one `eval.txt`.
   This alone answers the main reviewer question.
2. **B5 contrast (one training run):** add a `--quantizer {absmean,absmax,sign}` option to
   the recovery driver (or a sibling), train the no-scale ternary arm on 160M, same budget.
3. **A1 appendix (optional):** GPTQ/AWQ 4-bit if a 4-bit external point is wanted.
4. **Then G6** seed variance (2-3 seeds on 160M OURS) for paper hygiene.

## What NOT to do

- Do not chase higher 1.1B recovery (1200 steps) before G5/G6 — comparison and hygiene
  beat a few more recovery points for paper strength right now.
- Do not compare PyTorch CE to llama.cpp PPL across methods — keep everything in
  llama-perplexity on the same eval.txt.
- Do not claim a GPTQ/AWQ win/loss without matching the eval and bit budget honestly.

## Driver work needed

- B2/B3/B4: no code; just `llama-quantize` type args + `llama-perplexity`. A small
  `scripts/rt121_baseline_panel.py` can orchestrate: build FP f32/f16, quantize to the
  K-quants, reuse the adapted/PTQ I2_S GGUFs, run perplexity, emit the table.
- B5: add a quantizer-policy switch to `PerTensorBitLinear` / the recovery driver.
