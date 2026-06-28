# Nature-Inspired I2_S Smoke POC Plan

Status: proposal / smoke queue.

Purpose:

```text
Turn the external analogies into cheap, falsifiable POCs.
```

The ideas came from:

```text
physics / signal processing / statistics / biology / accounting
```

but every POC must stay inside the project philosophy:

```text
I2_S is the trunk.
Any auxiliary mechanism must be small, measured, and removable.
```

## Global Rules

### Rule 1: No New Branch Without A Smoke Gate

Each idea gets one cheap smoke before becoming a real track.

```text
PC 160M smoke first
Colab 1.1B only if PC gives a real signal
```

Exception:

```text
FACT-003H remains the active 1.1B mainline.
```

### Rule 2: Behavior Beats Reconstruction

No POC passes on weight MSE alone.

Pass requires movement in at least one behavior metric:

```text
FACT eval_panel
PopQA tight heldout
CE without tag collapse
```

### Rule 3: Compare Against The Correct Baseline

For current factual recovery:

```text
baseline = content-KL lambda=0.2 + PopQA blend 5% if available
```

For pure conversion:

```text
baseline = per-tensor I2_S / existing RT-124/WSYNC tables
```

### Rule 4: Never Hide The Cost

Every POC must report:

```text
extra train cost
extra bytes
extra ops proxy
whether the final artifact is still I2_S-rooted
```

## Priority Order

| rank | POC | borrowed idea | why first / later |
| ---: | --- | --- | --- |
| 1 | HOME-001 activation homeostasis | biological homeostatic plasticity / synaptic scaling | most aligned with current failure: fluent but factual drift |
| 2 | RDT-001 rate-distortion ledger | rate-distortion / activity-based costing | cheap analysis; stops us from chasing low-ROI branches |
| 3 | SIGMA-001 residual feedback init | delta-sigma / noise shaping | plausible one-shot/init improvement, but weight-only history is weak |
| 4 | RHT-002 dithered / multi-RHT reference | randomized projection / hypersphere quantization | fixed H-I2S failed; only worth tiny reference tests |
| 5 | ECC-001 residual syndrome sidecar | error-correcting codes | interesting, but sidecar/EGROW just went negative; lowest priority |

## POC 1: HOME-001 Activation Homeostasis

### Hypothesis

FACT failures are not only missing capacity. During b1.58 adaptation, hidden
states may drift away from the base model's factual manifold.

Biology analogy:

```text
learn, but keep activity set-points stable.
```

### Loss

Add a small activation-stat regularizer:

```text
L =
  L_answer_CE
  + lambda_KL * L_content_KL
  + eta_home * L_home
```

where:

```text
L_home =
  sum_l || mean(h_l^student) - mean(h_l^base) ||_2^2
  + rho * || rms(h_l^student) - rms(h_l^base) ||_2^2
```

Use a frozen base model only for smoke on 160M. Do not start with 1.1B because
the teacher doubles memory.

### Smoke Design

Model:

```text
Felladrin/Llama-160M-Chat-v1
```

Arms:

| arm | eta_home | layers |
| --- | ---: | --- |
| base | 0 | none |
| home-last | 0.01 | last hidden state only |
| home-mid-last | 0.01 | middle + last hidden states |
| home-strong | 0.05 | last hidden state only |

Recipe:

```text
content-KL lambda=0.2
PopQA blend 5%
answer-loss-only
300 steps
```

Metrics:

```text
eval_panel
popqa_tight
popqa_train
CE
tags
home_loss curve
activation drift before/after
```

Pass:

```text
popqa_tight or eval_panel improves by >=0.05 over base,
CE does not worsen by >0.10 nats,
tags stay mostly ok.
```

Fail:

```text
FACT flat,
or CE worsens,
or generation becomes empty/loop/salad.
```

Next if pass:

```text
HOME-002: 1.1B with lightweight layer subset or teacher-free stats cache.
```

Next if fail:

```text
homeostasis is not the missing lever; keep as analysis only.
```

### Implementation Hint

Add to `scripts/rt116_quality_recovery.py`:

```text
--homeostasis-weight FLOAT
--homeostasis-layers {last,mid_last}
--homeostasis-stat {mean_rms}
```

For 160M:

```text
load frozen base teacher
forward same train batch with output_hidden_states=True
compute mean/rms stats no-grad on teacher
compute mean/rms stats with grad on student
add eta_home * L_home
```

Do not export runtime until behavior passes.

Implemented smoke entrypoint:

```text
scripts/home001_activation_homeostasis_smoke.py
```

Default command:

```bash
python3 scripts/home001_activation_homeostasis_smoke.py --etas 0,0.01,0.05
```

This trains each arm with the same content-KL + PopQA blend recipe, then scores:

```text
FACT eval_panel
PopQA tight heldout
PopQA train sample
CE
generation tags
```

## POC 2: RDT-001 Rate-Distortion / Cost Ledger

### Hypothesis

We need a ledger that says:

```text
how much behavior each extra byte buys.
```

This is not a model-quality POC. It is a decision POC.

### Formula

For each branch:

```text
utility =
  Delta FACT
  - alpha * Delta CE
  - beta  * extra_bytes_ratio
  - gamma * extra_ops_proxy
```

Also report:

```text
FACT gain per MB
CE gain per MB
token-gen cost proxy
```

### Smoke Design

Inputs:

```text
reports/side001_160m.md
reports/egrow002_160m.md
reports/rt_wsync_160m.md
reports/rt_wsync_160m_hi2s.md
future FACT-003H summary
```

Output:

```text
reports/rdt001_cost_ledger.md
reports/rdt001_cost_ledger.json
```

Pass:

```text
ledger cleanly ranks branches and flags low-ROI ideas before Colab.
```

Fail:

```text
metrics too noisy to guide decisions.
```

### Implementation Hint

Create:

```text
scripts/rdt001_cost_ledger.py
```

Start with markdown/JSON parsing from existing report files. This can run on PC
or Mac; no GPU needed.

Implemented:

```text
scripts/rdt001_cost_ledger.py
reports/rdt001_cost_ledger.md
reports/rdt001_cost_ledger.json
```

Current smoke verdict:

```text
NO LOW-COST POSITIVE:
current PC branches do not buy >=0.05 eval;
wait for FACT-003H or a new mechanism.
```

## POC 3: SIGMA-001 Residual Feedback / Noise-Shaped Ternary Init

### Hypothesis

One-shot ternary fails partly because quantization error is discarded locally.
Delta-sigma suggests pushing residual into later coordinates so low-saliency
directions absorb error.

### Idea

For a row or block:

```text
r_0 = 0
for group g:
  z_g = W_g + alpha * r_{g-1}
  Q_g = I2_S(z_g)
  r_g = z_g - Q_g
```

Final artifact:

```text
still gamma*T
```

The residual feedback is only an initialization/projection rule.

### Smoke Design

Model:

```text
Llama-160M
```

Arms:

| arm | residual feedback |
| --- | --- |
| pt | none |
| sigma-row | row-wise left-to-right |
| sigma-block | block-wise by input groups |
| sigma-saliency | push residual toward low-saliency columns |

Metrics:

```text
CE
FACT
output residual
row_norm_ratio
```

Pass:

```text
FACT moves off 0.0 or CE improves >0.5 nats without staying in collapsed regime.
```

Fail:

```text
only weight MSE changes or FACT stays 0.0.
```

Interpretation:

```text
If SIGMA fails, weight-only projection remains dead.
If SIGMA helps CE but not FACT, it is only an init candidate before adaptation.
```

Implemented reference:

```text
scripts/sigma001_residual_feedback.py
```

Default command:

```bash
python3 scripts/sigma001_residual_feedback.py \
  --model-id Felladrin/Llama-160M-Chat-v1
```

Default arms:

```text
fp
pt
sigma_row_a0.5
sigma_g128_a0.5
sigma_g128_a1.0
```

## POC 4: RHT-002 Dithered / Multi-RHT Reference

### Hypothesis

Fixed one-shot block-Hadamard failed, but randomized/dithered/multi-RHT might
reduce outlier sensitivity better.

### Scope

Tiny reference only. No kernel.

### Arms

| arm | transform |
| --- | --- |
| pt | none |
| h1 | fixed block-Hadamard once |
| rht1 | random signs + Hadamard once |
| rht2 | two independent random-sign Hadamards |
| rht1+dither | RHT with small stochastic dither before ternary |

### Pass

```text
must beat per-tensor on CE and move FACT off 0.0.
```

Fail:

```text
FACT 0.0 or CE only moves inside collapsed regime.
```

Decision:

```text
If fail, no more RHT for weights.
KV-cache TurboQuant-style projection remains separate.
```

Implemented reference:

```text
scripts/rht002_dithered_reference.py
```

Default command:

```bash
python3 scripts/rht002_dithered_reference.py \
  --model-id Felladrin/Llama-160M-Chat-v1
```

Default arms:

```text
fp
pt
h1
rht1
rht2
rht1_dither
```

## POC 5: ECC-001 Residual Syndrome Sidecar

### Hypothesis

The sidecar should not learn generic residuals. It should store only a small
"syndrome" for task-relevant errors.

### Idea

Instead of training sidecar on all CE:

```text
L_side =
  L_FACT_surrogate
  + lambda * || sidecar_output ||^2
```

or:

```text
train sidecar only on high-saliency / high-error tokens
```

### Why Last

SIDE-001 and EGROW-002 are already weak. ECC-001 is only worth trying if:

```text
FACT-003H plateaus,
and HOME-001 fails,
and we still need a tiny auxiliary branch.
```

### Smoke

No standalone script yet. This should be implemented only after HOME/RDT/SIGMA/RHT
are read, because SIDE-001 and EGROW-002 already gave negative sidecar evidence.

Implementation target:

```text
scripts/ecc001_syndrome_sidecar_smoke.py
```

Use existing sidecar code, but change the training mask:

```text
sidecar gradients only on factual answer tokens or high-loss windows.
```

Pass:

```text
FACT gain >=0.05 with sidecar bytes << Q2_K overhead.
```

Fail:

```text
same as SIDE-001: CE moves, FACT flat.
```

## Recommended Execution Order

### Now / PC

```text
1. RDT-001 ledger (no GPU, immediate)
2. HOME-001 160M smoke (GPU, most promising)
3. SIGMA-001 weight-only reference (GPU/CPU, quick)
```

### Only If Needed

```text
4. RHT-002 tiny reference
5. ECC-001 sidecar syndrome
```

### Do Not Launch On Colab Yet

No new Colab branch until:

```text
FACT-003H result is known,
or HOME-001 gives a strong PC signal.
```

## Current Best Bet

The best new smoke is HOME-001.

Reason:

```text
current failures look like factual-manifold drift,
not raw capacity shortage.
```

The best cheap support tool is RDT-001.

Reason:

```text
it prevents low-ROI branches from consuming Colab time.
```

So the next practical sequence is:

```text
RDT-001 scaffold
HOME-001 implementation in rt116
HOME-001 160M arms
then decide whether Colab deserves a HOME-002 run.
```
