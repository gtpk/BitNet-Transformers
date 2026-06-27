# Literature Deep Dive 07: Rotation, AWQ, SmoothQuant, and the Conversion Stack

Document position: [Index](./index.md) -> [Literature Positioning Map](./literature_positioning_map.md) -> seventh deep dive.

Status checked: 2026-06-28.

Related:

- [Weight-Only Sync Plan](./weight_only_sync_plan.md)
- [Complex / Phase Rotation Probe Plan](./complex_phase_rotation_plan.md)
- [PTQTP Deep Dive](./literature_deep_dive_ptqtp.md)
- [TWLA Deep Dive](./literature_deep_dive_twla.md)
- [Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md)

## One-Line Read

The user's instinct is not off-track. The recent low-bit LLM literature is converging on:

```text
do not quantize the original coordinate system directly;
first reshape the representation so low-bit quantization sees an easier problem.
```

The methods differ in how they reshape:

| work | reshape mechanism | data needed? | closest project hook |
| --- | --- | --- | --- |
| SmoothQuant | move activation outliers into weights by equivalent scaling | small calibration | activation/weight scale migration |
| AWQ | protect salient channels using activation-aware scaling | small calibration | channel sensitivity / protected paths |
| QuaRot | output-preserving rotations remove outliers | mostly data-free for rotations | WSYNC rotation init |
| SpinQuant | learned rotations beat random rotations | optimization/calibration | learned rotation upper bound |
| PTQTP | add a second ternary trit-plane | no/few data depending implementation | capacity/plane expansion |
| TWLA | asymmetric ternary + Kronecker rotation + activation mixed precision | calibration | full conversion compiler hint |

For our project, the best synthesis is:

```text
I2_S runtime is our systems substrate.
But all-I2_S one-plane conversion is too brittle.
Borrow the literature's reshape/protect/capacity ideas before or alongside PopQA blend.
```

## The Shared Mathematical Pattern

Most methods are variations of:

```text
y = W x
```

Introduce an invertible transform `R`:

```text
y = (W R^{-1}) (R x)
```

or a diagonal scaling `S`:

```text
y = (W S^{-1}) (S x)
```

In full precision the function can be unchanged. After quantization:

```text
Q(W) x  may be bad
Q(W R^{-1}) (R x) may be easier
Q(W S^{-1}) (S x) may be easier
```

This is the precise version of:

```text
rotate / scale / reorder into a quantization-friendly basis.
```

Important correction:

```text
transpose alone is not a legal model transform.
```

Valid transforms must preserve the FP function by pairing the transform and inverse
across a hidden state, layer boundary, or equivalent foldable path.

## QuaRot

Source:

- arXiv: <https://arxiv.org/abs/2404.00456>
- Code: <https://github.com/spcl/QuaRot>

### What It Does

QuaRot applies rotations to hidden states, feed-forward activations, attention-related
activations, and KV cache so that outliers are spread out before quantization.

The key idea:

```text
orthogonal rotations preserve full-precision outputs
but change coordinate-wise quantization difficulty.
```

The paper targets end-to-end 4-bit inference:

```text
weights + activations + KV cache
```

It reports strong 4-bit LLaMA results and also notes lossless 6/8-bit variants without
calibration data.

### What We Borrow

QuaRot is the cleanest justification for WSYNC:

```text
try output-preserving rotations before any adaptation data.
```

Start small:

```text
signed permutation
Hadamard / block-Hadamard diagnostic
rotation-aware per-tensor b1.58
rotation-aware row-scale b1.58
```

### What Does Not Fit Directly

QuaRot is mostly a 4-bit W/A/KV paper. It does not prove:

```text
one-plane I2_S b1.58 becomes good after rotation.
```

Also, arbitrary rotations may need runtime support. Our product goal prefers transforms
that can be folded or cheaply executed.

## SpinQuant

Source:

- arXiv: <https://arxiv.org/abs/2405.16406>

### What It Does

SpinQuant agrees that rotations help, then shows a crucial warning:

```text
random rotations vary a lot;
learned rotations are safer.
```

The paper reports that different random rotations can lead to large downstream
zero-shot reasoning differences, so it learns rotation matrices to improve quantized
accuracy.

### What We Borrow

This tells us not to over-read one random Hadamard/rotation experiment.

For our roadmap:

```text
RT-WSYNC-003 random rotations
  -> if noisy positive, add learned rotation upper bound
```

SpinQuant is also a good explanation for why the user's "rotation" idea needs a
selection/optimization rule:

```text
rotation is a family, not a single trick.
```

### What Does Not Fit Directly

Learned rotations require some optimization signal. That may reintroduce calibration
data or expensive post-training. Therefore learned rotation is:

```text
upper bound / research candidate
not immediate low-resource product path
```

unless a cheap objective is found.

## AWQ

Source:

- arXiv: <https://arxiv.org/abs/2306.00978>
- Project: <https://github.com/mit-han-lab/llm-awq>

### What It Does

AWQ finds that not all weights/channels matter equally. Salient channels are identified
from activation statistics, then protected through an equivalent scaling transform rather
than expensive mixed precision.

In our language:

```text
some channels are fragile;
do not ternarize them blindly in the same coordinate system.
```

### What We Borrow

AWQ is the right mental model for:

```text
sensitivity-aware channel protection
selective scale/equalization
layer/channel importance before plane allocation
```

It is directly relevant to:

```text
HYBRID / variable capacity
WSYNC diagonal scaling
PopQA or calibration-driven channel scoring
```

### What Does Not Fit Directly

AWQ uses activation statistics. It is not pure weight-only.

For our project this creates two modes:

```text
AWQ-lite weight-only:
  use weight norms as a weak proxy

AWQ-real:
  use a small representative PopQA/instruction calibration stream
```

The second is more faithful to the paper and more likely to work.

## SmoothQuant

Source:

- arXiv: <https://arxiv.org/abs/2211.10438>
- Code: <https://github.com/mit-han-lab/smoothquant>

### What It Does

SmoothQuant observes:

```text
weights are easier to quantize than activations;
activation outliers are the main problem for W8A8.
```

It migrates activation difficulty into weights with an equivalent diagonal transform.

For a channel scale `s`:

```text
W x = (W S^{-1}) (S x)
```

The model is unchanged in FP, but quantization sees smoother activations.

### What We Borrow

SmoothQuant gives a template for legal equivalent transformations:

```text
do not just scale a layer;
scale one side and inverse-scale the connected side.
```

For b1.58, this is useful even if our immediate runtime is weight-dominated:

```text
activation outliers may decide which weight coordinates must be protected.
```

### What Does Not Fit Directly

SmoothQuant is W8A8, not W1.58. It also needs calibration activations to pick good
scales.

Therefore:

```text
SmoothQuant is not the answer;
its equivalent-transform discipline is the answer.
```

## PTQTP

Source:

- arXiv: <https://arxiv.org/abs/2509.16989>
- Existing deep dive: [PTQTP](./literature_deep_dive_ptqtp.md)

### What It Adds

PTQTP says the quiet part out loud:

```text
one ternary plane may not have enough expressiveness for PTQ.
```

It uses:

```text
W ~= alpha_1 T_1 + alpha_2 T_2
```

This maps directly to the user's "increase strips / planes / channels" intuition.

### What We Borrow

If WSYNC/PopQA plateau, the next serious capacity test is not random F16 restore. It is:

```text
adaptive trit-plane allocation
```

Not every layer may need two planes. A product-aware version is:

```text
I2_S one plane for easy layers
two planes for fragile layers
Q2/Q3/F16 pockets only if needed
```

## TWLA

Source:

- arXiv: <https://arxiv.org/abs/2606.13054>
- Code: <https://github.com/Kishon-zzx/TWLA>
- Existing deep dive: [TWLA](./literature_deep_dive_twla.md)

### What It Adds

TWLA combines three things:

```text
asymmetric ternary quantizer
Kronecker orthogonal rotation
inter-layer-aware activation mixed precision
```

This is the closest paper to the full version of our emerging idea:

```text
sensitivity-aware conversion compiler for ternary LLMs
```

### What We Borrow

TWLA validates three project lessons:

| our observation | TWLA-style answer |
| --- | --- |
| one-layer changes are non-additive | inter-layer-aware allocation |
| rotation may help | Kronecker orthogonal shaping |
| W-only speed story is incomplete | activation low-bit matters too |

### What Does Not Fit Directly

TWLA's asymmetric `mu + alpha*T` and activation mixed precision may not map cleanly to
bitnet.cpp I2_S.

So for us:

```text
TWLA is a method oracle / baseline,
not automatically our deployment artifact.
```

## Ranking For Our Project

### Immediate Experiments

| rank | candidate | why now |
| ---: | --- | --- |
| 1 | WSYNC cheap rotation/equalization | avoids small-data trap; cheap 160M screen |
| 2 | PopQA blend 1.1B | representative data path already ready |
| 3 | AWQ/SmoothQuant-lite with PopQA calibration | if WSYNC pure weight-only is too weak |
| 4 | PTQTP-lite two-plane | if one-plane I2_S plateau remains |
| 5 | TWLA-lite or external TWLA reproduction | strongest but heavier; needs code/runtime audit |

### What To Not Do First

Do not start with:

```text
arbitrary dense learned rotation on 1.1B
full TWLA reproduction on 7B
manual layer-by-layer F16 restore
more tiny hard replay
```

These are either too expensive or already contradicted by our failure modes.

## Proposed Combined Roadmap

```text
Step 1: finish mu=0.25 hard replay result.

Step 2: if hard replay fails/ambiguous, run PopQA blend 1.1B.

Step 3: in parallel on 160M, run WSYNC-001/002:
  per-tensor, row-scale, row-norm, diagonal equalization.

Step 4: add WSYNC-003 cheap rotations:
  signed permutation, norm interleave, block Hadamard diagnostic.

Step 5: if any WSYNC improves the no-data start, combine:
  WSYNC init + PopQA blend.

Step 6: if one-plane still plateaus, run PTQTP-lite:
  two ternary planes, adaptive allocation.

Step 7: use TWLA as external north-star:
  compare whether its rotation/asymmetric quantizer moves our FACT panel.
```

## Final Synthesis

The literature says our intuition should be sharpened from:

```text
maybe rotate or transpose weights
```

to:

```text
find an output-preserving transform that makes the low-bit coordinate system easier,
then allocate capacity only where sensitivity demands it.
```

For this project the best near-term method candidate is:

```text
WSYNC init
  + PopQA / representative blend
  + optional adaptive trit-plane capacity
  + I2_S runtime where possible
```

That is the bridge between:

```text
data-free geometry methods
and
low-resource useful b1.58 artifacts.
```
