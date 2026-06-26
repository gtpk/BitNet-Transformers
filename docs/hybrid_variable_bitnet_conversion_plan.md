# Hybrid / Variable BitNet Conversion Plan

Document position: [Index](./index.md) -> after
[Native BitNet Architecture Audit](./native_bitnet_architecture_audit.md).

Related:

- [Why Existing Models Resist b1.58 Conversion](./why_b158_conversion_is_hard.md)
- [Native BitNet Architecture Audit](./native_bitnet_architecture_audit.md)
- [Factual Gap Experiment Plan](./factual_gap_experiment_plan.md)
- [Mixed-Bit DP Plan](./mixed_bit_dp_plan.md)
- [Complex / Phase Rotation Probe Plan](./complex_phase_rotation_plan.md)

## Product Goal

The project is not mainly trying to publish a paper. The practical goal is:

```text
make a low-resource, memory-traffic-light LLM that is actually useful.
```

Therefore teacher-free conversion is useful only if it helps that goal. If a small
teacher, replay model, or factual objective is needed, it is allowed. The hard
constraint is not "no teacher"; the hard constraint is:

```text
bytes/token must stay much lower than FP16/Q4-style baselines,
and the result must produce useful answers.
```

## Why This Plan Exists

The project tried the obvious path:

```text
same architecture
same number of matrices
same hidden width
every target linear -> one per-tensor b1.58 I2_S matrix
short adaptation
```

What worked:

```text
I2_S export is faithful on x86.
storage and token-generation speed scale strongly.
CE/PPL recovery is real.
non-degenerate generation is recoverable with sane decoding.
```

What did not work:

```text
Q2_K still beats our 1.58-bit model on PPL.
factual quality stays far below FP/Q2_K.
data-only adaptation recovers fluency, not facts.
raw base-KL can copy EOS/empty behavior and backfire.
one-shot quantizer tricks do not solve the gap.
```

What changed after FACT-003C:

```text
content-KL lambda=0.2 is the current best factual lever.
lambda=0.1 was too weak and failed as salad.
lambda=0.5 is pending.
```

Therefore this hybrid plan is not the immediate next run while the content-KL
sweep is still open. It becomes the next branch if content-KL plateaus below the
useful factual tier.

So the next question is not:

```text
Can we find a smarter scalar quantizer?
```

It is:

```text
Does the converted model need more or differently placed capacity than a 1:1
all-b1.58 mapping gives it?
```

## Core Hypothesis

Native BitNet succeeds because the model is born in a BitNet-compatible function
class. Existing FP checkpoints are not. A same-shape projection tries to preserve
too much continuous information in too few discrete states.

For one weight matrix:

```text
FP:        W in R^{out x in}
I2_S:      Wq = gamma * T, T in {-1,0,+1}^{out x in}
multi-R:   Wq = sum_{r=1..R} gamma_r * T_r
residual:  Wq = gamma*T + A B
hybrid:    Wq_l in {I2_S, Q2_K, Q3_K, F16, multi-R, residual}
```

The optimization target should be:

```text
minimize task_loss(model_topology)
subject to bytes_per_token(model_topology) <= budget
```

This is no longer ordinary quantization. It is budgeted architecture conversion.

## Capacity Knobs

| Knob | Meaning | Cost | Runtime path | Why test it |
| --- | --- | --- | --- | --- |
| late-layer restore | keep last K blocks Q2/Q3/F16 | small if K is small | GGUF hybrid or PyTorch proxy | tests "last layers need more precision" |
| attention-only restore | keep q/k/v/o higher precision | medium | hybrid runtime | attention may carry factual routing |
| MLP-only restore | keep gate/up/down higher precision | medium | hybrid runtime | facts often live in FFN memory |
| multi-strip ternary | `sum gamma_r*T_r`, R=2/4 | R times selected I2_S cost | multiple I2_S matmuls or custom later | cheap way to add states without FP |
| low-rank residual | `gamma*T + A B` | small if rank low | extra small matmul | captures projection residual |
| row-scale I2_S init | per-output gamma, folded/proxied | small | fold/proxy first | RT-124A row scale was the only useful quantizer lever |
| BitNet-native block | SubLN + relu2 transplant | high | PyTorch first | tests if architecture mismatch matters |
| factual head/refiner | tiny answer adapter near output | small | adapter runtime | targets factual gap directly |

## What Not To Do

Do not return to:

```text
another global threshold search
another signed-epsilon codebook
another one-shot GPTQ-only assignment
full dense rotation as the main path
ternary-MoE for gpt-oss
```

Those either already failed in RT-124..127, violate the memory-traffic goal, or
target the wrong model family.

Rotation remains a later diagnostic only if it is cheap and structured:

```text
sign/swap phase
Hadamard-like pair/block transform
```

See [Complex / Phase Rotation Probe Plan](./complex_phase_rotation_plan.md).

## HYBRID-001 Experiment Ladder

### HYBRID-001A: late-layer capacity upper bound

Question:

```text
Are facts/free-run stability bottlenecked in the last blocks?
```

Arms:

```text
all-I2_S adapted baseline
last 1 block restored to F16
last 2 blocks restored to F16
last 4 blocks restored to F16
last 2 attention-only restored
last 2 MLP-only restored
```

First run this as a PyTorch or GGUF-proxy evaluation. No new training.

Metrics:

```text
FACT-001 factual panel hit rate
degeneration tags under rep-penalty 1.2
WikiText CE/PPL
storage proxy: total target bytes and whole-file bytes
speed proxy: fraction of target-linear bytes still I2_S
```

Branch:

```text
If last-K restore raises facts meaningfully:
  late capacity matters -> continue HYBRID-001B/C with cheap versions.

If attention-only helps:
  factual routing/attention precision matters -> attention hybrid.

If MLP-only helps:
  FFN memory precision matters -> MLP hybrid or residual.

If no restore helps:
  quality gap is not local late-layer precision -> go back to objective/replay or BitNet-native block changes.
```

Pass bar:

```text
fact_rate improves by >= 0.15 absolute over current best adapted I2_S
without worse degeneration,
or reaches >= 0.30 as a credible product-direction signal.
```

This is deliberately a low bar. The goal is not final quality; the goal is to
detect whether selective capacity can move facts at all.

### HYBRID-001B: cheap hybrid precision

Only if HYBRID-001A finds a sensitive region.

Replace the winning F16 restore with cheaper options:

```text
Q2_K
Q3_K_M
Q4_0
row-scale ternary proxy
```

Decision:

```text
If Q2/Q3 gets most of the F16 gain:
  product path = selective hybrid GGUF.

If only F16 gets the gain:
  capacity is real but too expensive -> try multi-strip/residual.
```

### HYBRID-001C: multi-strip ternary

Represent one sensitive matrix as multiple ternary strips:

```text
Wq_R = gamma_1*T_1 + ... + gamma_R*T_R
```

Interpretation:

```text
R=1 is ordinary b1.58.
R=2 is roughly "two cheap ternary experts" for the same linear.
R=4 tests whether the problem is state count/capacity, not precision format.
```

Implementation options:

```text
PyTorch proxy first: sum of R ternary matmuls.
Runtime later: either R I2_S tensors and R matmuls for selected layers,
or a custom fused kernel only if the signal is strong.
```

Branch:

```text
If R=2 late-layer strip recovers most F16-restore gain:
  dynamic strip allocation is promising.

If R must be large:
  speed benefit erodes; prefer small residual or Q2/Q3 hybrid.

If R does not help:
  ternary state count alone is not the bottleneck.
```

### HYBRID-001D: low-rank residual

Use b1.58 as the fast base and add a tiny residual:

```text
W ~= gamma*T + A B
rank(A B) in {4, 8, 16, 32}
```

This is the clean compromise if:

```text
F16 restore helps,
multi-strip is too expensive,
and facts need a small continuous correction.
```

Report:

```text
residual rank
extra MB
extra matmul cost
fact_rate
PPL
generation tags
```

### HYBRID-001E: BitNet-native block transplant

Only after the cheaper knobs.

Test whether native BitNet architecture details matter:

```text
SubLN insertion
relu2 FFN variant
late BitNet-native replacement block
```

This is no longer simple conversion. It is architecture surgery and should be
treated as a future track.

## Dynamic Allocation View

The user's "strip/width/layer can grow or shrink" idea becomes:

```text
start each layer at R=1 I2_S
give extra strips or residual rank only where validation loss/factual score says so
remove extra capacity where it does not move quality
```

This is similar to knapsack, but with a warning from RT-123:

```text
layer effects are non-additive.
```

So dynamic allocation should not be pure DP over independent layer gains. Use DP
only as an index/proposal generator:

```text
1. measure local proposals
2. choose a small candidate set
3. re-adapt the whole model under the chosen topology
4. evaluate globally
```

## Implementation Packet

Expected new files:

```text
scripts/rt134_hybrid_capacity_probe.py
scripts/rt135_multistrip_probe.py
bitnet_llama/hybrid.py
reports/rt134_hybrid_capacity_probe.json
reports/rt135_multistrip_probe.json
```

Minimum API:

```text
--model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0
--adapted-checkpoint PATH
--restore-last-k 0,1,2,4
--restore-kind all,attn,mlp
--precision f16,q2_k,q3_k,row_ternary,multistrip,residual
--fact-panel data/factual_panel_v1.jsonl
--decode rep_penalty_1p2
--json-out reports/...
```

Suggested first command:

```bash
python scripts/rt134_hybrid_capacity_probe.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapted-checkpoint /content/drive/MyDrive/bnt/fact003a_mixed \
  --restore-last-k 0 1 2 4 \
  --restore-kind all attn mlp \
  --fact-panel data/factual_panel_v1.jsonl \
  --decode rep_penalty_1p2 \
  --json-out reports/rt134_hybrid_capacity_probe.json
```

## Colab Handoff Prompt

Use this if another AI will run the experiment:

```text
You are continuing the BitNet-Transformers project.

Do not use Colab MCP. Work in a normal Colab notebook or shell.

Goal:
Run HYBRID-001A from docs/hybrid_variable_bitnet_conversion_plan.md.

Context:
- all-I2_S b1.58 export/runtime is solved on x86
- one-shot quantizer tricks are ruled out
- FACT-003A answer-only mask moved facts to ~0.15 but still far below Q2_K
- FACT-003B raw base-KL backfired by copying EOS/empty behavior
- current hypothesis: same-shape all-I2_S lacks capacity; test selective late-layer
  higher precision after content-KL plateaus

Tasks:
1. git fetch && git reset --hard origin/main
2. inspect docs/hybrid_variable_bitnet_conversion_plan.md
3. implement scripts/rt134_hybrid_capacity_probe.py if missing
4. run all arms:
   - all-I2_S baseline
   - last 1/2/4 blocks restored to F16
   - last 2 attention-only F16
   - last 2 MLP-only F16
5. evaluate the fixed factual panel, generation degeneration, CE/PPL, and storage proxy
6. write reports/rt134_hybrid_capacity_probe.json
7. update docs/hybrid_variable_bitnet_conversion_plan.md with RESULT
8. update docs/index.md and docs/paper_draft.md only if the conclusion changes claims

Decision:
- If fact_rate improves by >=0.15 absolute, continue HYBRID-001B/C.
- If no restore helps, stop hybrid-late-layer work and return to objective/replay.
- If only F16 helps, test residual/multi-strip rather than claiming deployment success.
```

## Decision Tree

```text
HYBRID-001A late restore
|
|-- facts improve
|   |
|   |-- Q2/Q3 keeps most gain -> selective hybrid GGUF product path
|   |-- only F16 helps -> try multi-strip/residual
|   |-- attention-only helps -> attention capacity track
|   |-- MLP-only helps -> FFN memory/residual track
|
|-- facts do not improve
    |
    |-- PPL improves but facts do not -> objective/factual replay, not capacity
    |-- generation improves only -> decoding/free-run stability track
    |-- nothing improves -> same-shape conversion limit; consider native BitNet training or larger model
```

## Current Recommendation

Run HYBRID-001A after the FACT-003C content-KL sweep if the best lambda still
lands below a useful factual tier (roughly `fact_rate < 0.3~0.4`).

Reason:

```text
It is the cheapest test of the user's capacity hypothesis once the best current
objective has been exhausted:
"1.58-bit works, but same-shape all-layer conversion lacks enough representational
capacity; give selected parts more room."
```

If HYBRID-001A gives no signal, capacity expansion is probably not the next lever.
If it gives a signal, the project has a clear product path:

```text
mostly-I2_S model + small precision/capacity pockets
```
