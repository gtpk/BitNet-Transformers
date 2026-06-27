# Literature Deep Dive 03: TWLA

Document position: [Index](./index.md) -> [Literature Positioning Map](./literature_positioning_map.md) -> third deep dive.

Paper:

- TWLA: Achieving Ternary Weights and Low-Bit Activations for LLMs via Post-Training Quantization
- arXiv: https://arxiv.org/abs/2606.13054
- Code: https://github.com/Kishon-zzx/TWLA
- Status checked: 2026-06-27

## One-Line Read

TWLA is the paper that most directly validates our "rotation + interaction-aware
allocation" intuition. It argues that W1.58 alone is not enough for real acceleration:
the hard part is **W1.58 plus low-bit activations**, and that requires distribution
shaping plus inter-layer-aware bit allocation.

If CAT-Q asks:

```text
Can one-plane ternary PTQ work with a better optimization path?
```

TWLA asks:

```text
Can ternary PTQ become a real end-to-end inference system with 4-bit activations?
```

This is closer to the systems/product goal than weight-only PPL.

## What They Claim

| claim | meaning |
| --- | --- |
| W1.58A4 | ternary weights and 4-bit activations, not just weight-only |
| PTQ, no QAT | post-training framework rather than native BitNet training |
| E2M-ATQ | asymmetric ternary quantizer with output-aware metric relocation |
| KOTMS | Kronecker orthogonal rotation reshapes weights and suppresses activation outliers |
| ILA-AMP | inter-layer-aware activation mixed precision using adjacent-layer interaction costs |
| code released | GitHub repo exists with scripts for KOTMS, ILA-AMP, and `run_twla.py` |
| ICML 2026 | high visibility and likely strong baseline |

The paper's main target is not only storage:

```text
end-to-end acceleration requires activations to become low-bit too.
```

That is a useful warning for us, because our current I2_S path mostly proves weight
traffic and token-generation speed on CPU, while not yet proving a full W1.58A4 stack.

## Core Method

### 1. E2M-ATQ: Euclidean-To-Manifold Asymmetric Ternary Quantizer

TWLA uses an asymmetric row-wise ternary form:

```text
W_bar = mu * 1^T + diag(alpha) * T
T in {-1, 0, +1}
```

This is different from our current I2_S-compatible form:

```text
Wq = gamma * T
```

The important part is not only the extra `mu`; it is the two-stage optimization:

```text
Stage 1: find stable T, mu, alpha in Euclidean/Frobenius weight space.
Stage 2: freeze T, then relocate mu and alpha under a calibration-induced metric.
```

The second stage minimizes layer output error:

```text
|| (W - mu*1^T - diag(alpha)*T) X ||_F^2
```

by using the activation second moment:

```text
S = X^T X
```

This directly addresses one of our recurring issues:

```text
weight reconstruction is not the same as output/CE reconstruction.
```

### 2. KOTMS: Kronecker Orthogonal Tri-Modal Shaping

TWLA says pretrained LLM weights are often unimodal-ish, while ternary wants three
attraction regions:

```text
-alpha + mu, mu, +alpha + mu
```

KOTMS applies an orthogonal rotation so the coordinate system becomes more
ternary-friendly. Because the rotation is orthogonal, it can preserve the full-precision
function if the inverse rotation is folded appropriately. The same rotation also mixes
activations and statistically suppresses outliers.

This is the cleanest literature answer to our earlier "complex phase / e^{i theta}"
idea:

```text
do not store complex weights;
use foldable real orthogonal rotations.
```

TWLA specifically uses Kronecker structure to make this practical.

### 3. ILA-AMP: Inter-Layer Aware Activation Mixed Precision

TWLA explicitly says independent layer-wise sensitivity is insufficient because
activation quantization shifts distributions and couples errors across adjacent layers.
It introduces adjacent-layer second-order interaction costs and optimizes activation
bits under a mixed-precision budget.

This is exactly what our RT-123 was hinting at:

```text
single-layer changes were non-additive;
one layer restore could worsen downstream ternary mismatch.
```

TWLA's answer:

```text
allocation should include interaction costs, not just item-wise sensitivity.
```

## Code / Reproduction Surface

The GitHub repo exists and is small enough to inspect later:

```text
https://github.com/Kishon-zzx/TWLA
```

README exposes a three-step pipeline for Qwen3-8B:

```text
1. scripts/KOTMS.py
2. scripts/ILA_AMP.py
3. run_twla.py
```

Example commands in README:

```text
python scripts/KOTMS.py --model Qwen/Qwen3-8B --export_rotated ... --ngpus 4
python scripts/ILA_AMP.py --model Qwen/Qwen3-8B --import_rotated ... --dp_cache ...
python run_twla.py --model Qwen/Qwen3-8B --import_rotated ... --eval_qa --abits 16
python run_twla.py --model Qwen/Qwen3-8B --import_rotated ... --eval_qa --dp_avg_abits 4
```

This is more actionable than papers whose code link is not yet inspectable.

## Where TWLA Is Ahead Of Us

| axis | TWLA | us |
| --- | --- | --- |
| activation quantization | W1.58A4 is central | mostly weight/I2_S-focused so far |
| rotation | Kronecker orthogonal shaping | only conceptual rotation plan |
| inter-layer allocation | explicit adjacent-layer second-order costs | RT-123 showed non-additivity but did not solve it |
| scale | LLaMA/Qwen 7B-70B+ families in paper | 160M/1.1B |
| code availability | repo released | our code is local but not comparable baseline |
| benchmark breadth | WikiText2 + seven zero-shot tasks | small PPL/factual/generation panels |

Hard truth:

```text
TWLA is closer to an end-to-end low-bit inference paper than our current weight-only
conversion track.
```

## Where We May Still Differ

| axis | our possible edge |
| --- | --- |
| bitnet.cpp/I2_S export | TWLA may use its own quantized model path; our Path A' is tested on bitnet.cpp |
| factual objective | TWLA is PTQ/reconstruction/QA-focused; our content-KL directly targets factual retention |
| product scorecard | we include post-training cost, storage, tg speed, factual, degeneration, runtime parity |
| adaptive topology | TWLA allocates activation bits, not necessarily weight-plane/F16 pockets |
| low-resource CPU story | TWLA examples use multi-GPU calibration; our runtime story includes commodity x86 |

TWLA should be treated as:

```text
a stronger external engine for rotation + activation-aware quantization,
not a replacement for our runtime/factual/product framing.
```

## Questions / Doubts To Verify

| question | why it matters | how to test |
| --- | --- | --- |
| Can TWLA run on smaller dense LLaMA? | Qwen3-8B examples may be heavy | try 160M or TinyLlama if code supports it |
| What is actual runtime artifact? | W1.58A4 accuracy does not guarantee bitnet.cpp compatibility | inspect saved model format and kernels |
| Is `mu` runtime-friendly? | asymmetric `mu + alpha*T` is not I2_S-native | see if `mu` can fold into bias/norm or needs custom kernel |
| Does KOTMS break I2_S Path A'? | rotations may require extra matrices at runtime or fold into adjacent layers | inspect fold strategy |
| Does activation 4-bit matter on CPU token-gen? | our bottleneck is weight traffic; activation cost may dominate elsewhere | compare llama-bench style metrics if exportable |
| Does TWLA preserve facts? | our open gap is factual quality | run FACT panel |
| Does content-KL help after TWLA? | separates PTQ quality from objective quality | TWLA -> content-KL adaptation |

## Insight For Our Roadmap

TWLA changes the meaning of "adaptive allocation." It should not only mean:

```text
which layers get extra weight precision?
```

It should include:

```text
which layers get activation bits?
which adjacent-layer pairs have coupled failure modes?
which rotations make both weights and activations easier?
```

This suggests a better future selector:

```text
choice_l = {
  weight representation: I2_S / CAT-Q-lite / two-plane / Q2 / F16 pocket
  activation bits: 4 / 6 / 8 / 16
  rotation: none / shared orthogonal / block-Kronecker
}

cost(choice_l, choice_{l+1}) includes adjacent interaction.
```

This is much closer to a real "BitNet conversion compiler" than our original additive
knapsack idea.

## Minimal Local Reproduction Idea

Do not try full TWLA first. Start with the pieces that test our hypotheses:

### TWLA-lite A: E2M-ATQ without rotation

```text
For each linear:
  build ternary T by row-wise asymmetric warm-start
  collect calibration X
  solve row-wise 2x2 system for alpha, mu under S = X^T X
  materialize W_bar = mu + alpha*T
```

Questions:

```text
Does asymmetric output-aware relocation beat our absmean and CAT-Q-lite?
Can mu be folded or approximated into I2_S-compatible form?
```

### TWLA-lite B: rotation-only probe

```text
Apply Hadamard/Kronecker-like foldable rotation.
Ternarize after rotation.
Compare PPL/fact to no-rotation.
```

Questions:

```text
Does rotation reduce one-plane factual/CE loss?
Does it help activation quantization enough to matter?
```

### TWLA-lite C: interaction-aware allocation

Use RT-123 style sensitivity but add adjacent interaction:

```text
score(layer_i choice_i, layer_{i+1} choice_{i+1})
```

Then solve a chain DP instead of independent knapsack.

## How This Changes Our Paper Split

| paper | update from TWLA |
| --- | --- |
| Paper 1 Systems | mention TWLA as end-to-end low-bit activation/runtime competitor |
| Paper 2 Conversion Limits | our "DP failed" should be refined: independent DP failed; interaction-aware allocation is known useful |
| Paper 3 Content-KL | content-KL can sit on top of TWLA as objective recovery |
| Paper 4 Hybrid | should include rotation + activation-bit allocation, not only F16/plane pockets |

## Bottom Line

TWLA gives us three sharp lessons:

```text
1. Our rotation idea is real, but should be real orthogonal/Kronecker, not literal complex storage.
2. Our DP idea was not wrong; the independent additive assumption was wrong.
3. Weight-only I2_S is not the final product story if activation quantization matters for target hardware.
```

The next disciplined path is:

```text
CAT-Q-lite first if we want to preserve one-plane I2_S.
TWLA-lite next if CAT-Q-lite is insufficient or if activation quantization becomes the bottleneck.
PTQTP-lite if one-plane remains under-capacity.
```

