# I2_S + LoRA / Residual Sidecar Plan

Document position: [Index](./index.md) -> after
[Hybrid / Variable BitNet Conversion Plan](./hybrid_variable_bitnet_conversion_plan.md).

Status: proposal / next experiment track. No result yet.

## Why This Track Exists

The project has learned three hard facts:

```text
pure one-shot I2_S collapses,
small hard factual replay overfits,
data-free scaling / H-I2S rotation does not rescue behavior.
```

But the systems substrate is strong:

```text
I2_S export/runtime works,
storage is small,
token-generation speed is high.
```

So the next capacity question is:

```text
Can we keep the I2_S base and add a very small trainable residual
instead of moving the whole model to Q2/Q3/Q4?
```

This is a compromise between:

```text
all-I2_S purity        -> fastest/smallest, but may lack capacity
Q2_K/Q3/Q4 everywhere  -> better quality, but weaker memory-traffic win
```

The proposed artifact is:

```text
mostly I2_S + tiny sidecar
```

not:

```text
pure I2_S.
```

Therefore this is Track B by representation, but it can become a product track if
the sidecar is tiny and the speed/storage win remains large.

## Core Form

For one target linear:

```text
y = W x
```

current I2_S:

```text
y_i2s = gamma*T*x
T in {-1,0,+1}
```

sidecar version:

```text
y_side = gamma*T*x + s * B(Ax)
```

where:

```text
A in R^{r x in}
B in R^{out x r}
r << min(in, out)
s = alpha / r
```

Equivalently:

```text
W_side = gamma*T + s * B A
```

The sidecar is a low-rank residual that tries to capture the part of the FP function
that one ternary plane cannot express.

## Why This Might Work

The residual error after ternary projection is:

```text
E = W - gamma*T
```

The output-weighted error is:

```text
Tr(E Sigma_x E^T)
```

A rank-`r` sidecar can target the largest task-relevant modes:

```text
min_{A,B} E_x || W x - (gamma*T + B A)x ||^2
```

If the harmful error is low-rank or concentrated in a few directions, then a small
`r` may buy a disproportionate quality gain.

This is the same broad idea as:

```text
quantized base + low-rank correction,
```

but specialized for:

```text
I2_S base + memory-traffic-first deployment.
```

## What This Is Testing

SIDE tests a different hypothesis from PopQA blend.

| question | PopQA blend | I2_S+sidecar |
| --- | --- | --- |
| Is factual failure data/objective mismatch? | yes | indirectly |
| Is one-plane I2_S capacity insufficient? | no direct answer | yes |
| Can tiny extra params recover behavior? | no | yes |
| Does runtime remain mostly I2_S? | yes | maybe, depends on sidecar cost |

The key diagnostic:

```text
If PopQA blend fails but sidecar helps, capacity is likely the bottleneck.
If PopQA blend helps and sidecar helps further, both data and capacity matter.
If neither helps, model size/objective/base choice may be the bottleneck.
```

## Track Classification

### Track A-Compatible Variant

This variant keeps the final main matrix as I2_S and adds a tiny optional module:

```text
I2_S matmul + low-rank sidecar matmul
```

Allowed Track A claim only if:

```text
sidecar bytes and ops are reported separately,
I2_S parity is preserved for the base,
speed/storage remain meaningfully better than Q2_K/Q4,
```

and the result is called:

```text
mostly-I2_S + sidecar
```

not pure I2_S.

### Track B Upper-Bound Variant

Use larger ranks or more target modules:

```text
r = 16/32,
all q/k/v/o/gate/up/down,
possibly lm_head,
```

only to estimate how much capacity is missing. Do not productize unless the overhead
is still acceptable.

## Experimental Ladder

### SIDE-000: Parameter And Byte Accounting

Before training, compute overhead.

For one linear `out x in`, LoRA params are:

```text
r * (in + out)
```

For all target linears:

```text
sidecar_params = sum_l r_l * (in_l + out_l)
sidecar_bytes_fp16 = 2 * sidecar_params
```

Report:

```text
sidecar bytes / I2_S target bytes
sidecar bytes / whole GGUF bytes
sidecar ops / I2_S ops proxy
```

Pass to continue:

```text
r=4 or r=8 overhead is small enough to remain "mostly I2_S".
```

### SIDE-001: 160M Frozen-Base Sidecar Smoke On PC

Purpose:

```text
Does a small residual move FACT at all?
```

Model:

```text
JackFram/llama-160m
I2_S / b1.58 target linears frozen
LoRA sidecars trainable
lm_head frozen
norms frozen
```

Arms:

| arm | base | rank | trainable |
| --- | --- | ---: | --- |
| S0 | current I2_S adapted baseline | 0 | none |
| S1 | I2_S + LoRA | 2 | sidecar only |
| S2 | I2_S + LoRA | 4 | sidecar only |
| S3 | I2_S + LoRA | 8 | sidecar only |
| S4 | I2_S + LoRA top-saliency layers | 8 | sidecar only |

Training data:

```text
same content-KL + representative blend recipe as current best,
or a short PopQA blend subset if FACT-003H is still running.
```

Metrics:

```text
FACT panel
PopQA tight held-out
CE/PPL
degeneration tags
sidecar bytes
train time
```

Pass:

```text
rank 4/8 improves FACT by >= 0.05 absolute
without worse tags,
and overhead is materially smaller than Q2_K/Q3 everywhere.
```

Fail:

```text
FACT flat,
or only rank 16+ helps,
or CE improves but facts do not.
```

### SIDE-002: 160M Co-Adapted Sidecar Smoke

Only if SIDE-001 shows a signal.

Purpose:

```text
Test whether the sidecar must be present during ternary adaptation.
```

Arms:

```text
I2_S STE target linears + LoRA sidecar from step 0
vs
frozen I2_S base + sidecar
```

Why:

Post-hoc FP restore failed because of co-adaptation mismatch. A sidecar may also
need to be trained together with the I2_S base.

Pass:

```text
co-adapted sidecar beats frozen-base sidecar under the same bytes.
```

### SIDE-003: 1.1B Colab Confirmation

Only if SIDE-001 or SIDE-002 passes on 160M.

Model:

```text
TinyLlama-1.1B
rank chosen from 160M
same PopQA blend / content-KL recipe
```

Primary question:

```text
Does the 1.1B FACT score move beyond the content-KL baseline 0.185?
```

Pass:

```text
FACT improves by >= 0.05 absolute over the best same-topology baseline,
PopQA tight held-out improves,
i2_s/base behavior remains stable,
tags stay ok.
```

Strong pass:

```text
FACT >= 0.30
with small sidecar overhead.
```

### SIDE-004: Runtime Accounting

Only after SIDE-003 passes.

Measure:

```text
I2_S-only token-gen speed
I2_S + sidecar PyTorch proxy ops
estimated fused runtime
whole artifact bytes
target-linear bytes
```

Decision:

```text
If speed/storage still beat Q2_K/Q4 meaningfully:
  keep sidecar as product candidate.

If quality improves but speed collapses:
  sidecar is only an upper-bound diagnostic.
```

### SIDE-005: Compression Of The Sidecar

Only if SIDE-003/004 pass.

Options:

```text
fp16 sidecar
int8 sidecar
ternary sidecar
merged second ternary plane
top-layer-only sidecar
```

Goal:

```text
move the sidecar back toward Track A.
```

## Implementation Plan

### New module

Proposed file:

```text
bitnet_llama/sidecar.py
```

Core class:

```python
class I2SLoRALinear(nn.Module):
    def __init__(self, packed_or_dense_i2s_linear, rank, alpha):
        ...

    def forward(self, x):
        return base_i2s(x) + scale * lora_B(lora_A(x))
```

For the first PyTorch smoke, the base can be dense materialized `gamma*T` rather than
packed I2_S. Runtime claims come later.

### Conversion helper

Proposed function:

```text
replace_target_linears_with_i2s_lora(model, rank, alpha, target_policy)
```

Requirements:

```text
base I2_S weights frozen by default
LoRA A/B trainable
target module names recorded
rank and alpha recorded in JSON
```

### Training integration

Preferred:

```text
extend scripts/rt116_quality_recovery.py
```

Flags:

```text
--sidecar-rank INT
--sidecar-alpha FLOAT
--sidecar-target {all,attn,mlp,top_saliency}
--sidecar-train-base {false,true}
--sidecar-init {zero,svd_residual,random}
```

Defaults:

```text
rank=0 means disabled;
existing behavior unchanged.
```

### Result JSON fields

Every run must record:

```text
sidecar_enabled
sidecar_rank
sidecar_alpha
sidecar_target
sidecar_train_base
sidecar_params
sidecar_bytes_fp16
sidecar_bytes_ratio_vs_target_i2s
trainable_params
```

## PC vs Colab Split

### PC / RTX 3080

Run:

```text
SIDE-000 parameter accounting
SIDE-001 160M frozen-base smoke
SIDE-002 160M co-adapted smoke if SIDE-001 passes
small eval/scoring checks
```

Do not run:

```text
1.1B full sidecar adaptation
7B experiments
runtime speed claims
```

Target PC command after implementation:

```bash
python scripts/rt116_quality_recovery.py \
  --model-id JackFram/llama-160m \
  --train-source mixed \
  --answer-loss-only \
  --base-kl-replay \
  --kl-content-only \
  --kl-weight 0.2 \
  --factual-blend-file data/popqa_blend_train.jsonl \
  --factual-blend-frac 0.05 \
  --sidecar-rank 4 \
  --sidecar-alpha 8 \
  --sidecar-target all \
  --sidecar-train-base false \
  --steps 300 \
  --seq-len 256 \
  --batch 8 \
  --lr 2e-4 \
  --out-dir runs/side001_160m_rank4 \
  --json-out reports/side001_160m_rank4.json \
  --log-every 25
```

Rank sweep:

```bash
for R in 0 2 4 8; do
  python scripts/rt116_quality_recovery.py \
    --model-id JackFram/llama-160m \
    --train-source mixed \
    --answer-loss-only \
    --base-kl-replay \
    --kl-content-only \
    --kl-weight 0.2 \
    --factual-blend-file data/popqa_blend_train.jsonl \
    --factual-blend-frac 0.05 \
    --sidecar-rank $R \
    --sidecar-alpha 8 \
    --sidecar-target all \
    --sidecar-train-base false \
    --steps 300 \
    --seq-len 256 \
    --batch 8 \
    --lr 2e-4 \
    --out-dir runs/side001_160m_rank${R} \
    --json-out reports/side001_160m_rank${R}.json \
    --log-every 25
done
```

Scoring target:

```bash
python scripts/rt130_factual_gap_panel.py \
  --variants side001_rank0=... side001_rank4=... side001_rank8=... \
  --panel data/factual_panel_v1.jsonl \
  --popqa-tight data/popqa_heldout_tight.jsonl \
  --json-out reports/side001_160m_score.json \
  --md-out reports/side001_160m_score.md
```

The exact scoring command may need adjustment to the current `rt130` CLI; the
required output fields are fixed:

```text
FACT panel,
PopQA tight held-out,
PopQA train,
CE/PPL,
tags,
sidecar bytes.
```

### Colab / L4 / A100

Run only after PC signal:

```text
SIDE-003 TinyLlama-1.1B confirmation
SIDE-004 runtime accounting
larger model audit if 1.1B works
```

Target Colab command after PC pass:

```bash
python scripts/rt116_quality_recovery.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-source mixed \
  --answer-loss-only \
  --base-kl-replay \
  --kl-content-only \
  --kl-weight 0.2 \
  --factual-blend-file data/popqa_blend_train.jsonl \
  --factual-blend-frac 0.05 \
  --sidecar-rank 4 \
  --sidecar-alpha 8 \
  --sidecar-target all \
  --sidecar-train-base false \
  --steps 800 \
  --seq-len 256 \
  --batch 4 \
  --grad-accum-steps 6 \
  --lr 2e-4 \
  --dtype float32 \
  --optim adamw8bit \
  --grad-checkpointing \
  --bitnet /content/bitnet.cpp \
  --out-dir /content/drive/MyDrive/bnt_runs/side003_1p1b_rank4 \
  --json-out reports/side003_1p1b_rank4.json \
  --log-every 25
```

## Decision Table

| result | interpretation | next |
| --- | --- | --- |
| rank 4/8 improves 160M FACT | small residual capacity matters | run 1.1B SIDE-003 |
| only high rank helps | capacity gap too large for tiny sidecar | compare PTQTP-lite / Q2 pocket |
| CE improves but FACT flat | sidecar helps language modeling, not facts | wait for PopQA/objective result |
| sidecar overfits PopQA train | data/objective issue remains | broader blend / regularization |
| 1.1B improves but runtime cost high | diagnostic upper bound | compress sidecar or selective targets |
| 1.1B improves with low cost | product candidate | integrate into fair comparison |

## Stop Rules

Stop SIDE if:

```text
rank 8 all-target sidecar does not improve FACT on 160M,
or sidecar overhead approaches Q2_K/Q3 bytes without comparable quality,
or training becomes as expensive as normal fine-tuning without preserving speed.
```

## Immediate Next Actions

1. Wait for FACT-003H PopQA blend result if it is about to finish.
2. In parallel on PC, implement SIDE-000 accounting and SIDE-001 scaffold.
3. Run rank `2/4/8` 160M smoke only after the scaffold reproduces the rank-0 baseline.
4. If any rank moves FACT by `>=0.05`, schedule a Colab 1.1B SIDE-003.
5. If no rank moves FACT, keep sidecar as a documented negative and move to PTQTP-lite
   or representative data/objective scaling.
