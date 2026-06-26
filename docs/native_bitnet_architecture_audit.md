# Native BitNet Architecture Audit

Document position: [Index](./index.md) -> after
[Why Existing Models Resist b1.58 Conversion](./why_b158_conversion_is_hard.md).

Related:

- [Hybrid / Variable BitNet Conversion Plan](./hybrid_variable_bitnet_conversion_plan.md)
- [Quantization-Aware b1.58 Conversion Plan](./quantization_aware_b158_conversion_plan.md)
- [Factual Gap Experiment Plan](./factual_gap_experiment_plan.md)
- [Paper Draft](./paper_draft.md)

## Purpose

The project has learned that:

```text
same-shape FP LLaMA -> all-I2_S b1.58 conversion is systems-feasible,
but quality/factual recovery is not solved by one-shot quantizer tricks.
```

The user hypothesis is now sharper:

```text
If native BitNet b1.58 really works, perhaps the working model is not just a
standard LLaMA with each Linear crushed to {-1,0,+1}. It may need BitNet-native
layers, capacity allocation, or architecture choices that make the ternary space
large enough.
```

This document records what public BitNet materials actually say, what they do not
say, and what that implies for this fork.

## Public Sources Checked

Primary/public sources:

- BitNet b1.58 paper: [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- BitNet 2B4T report: [BitNet b1.58 2B4T Technical Report](https://arxiv.org/html/2504.12285)
- Microsoft model card/config: [microsoft/bitnet-b1.58-2B-4T-bf16](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16/tree/main)
- Microsoft runtime: [microsoft/BitNet](https://github.com/microsoft/BitNet)
- BitNet v2: [Native 4-bit Activations with Hadamard Transformation](https://arxiv.org/abs/2504.18415)
- Small-model follow-up: [BitNet b1.58 Reloaded](https://arxiv.org/abs/2407.09527)

## What Is Publicly Confirmed

### 1. Native BitNet is trained inside the ternary family

The original b1.58 paper defines every weight as ternary `{-1,0,+1}` and reports
that, at sufficient scale/training, this can match full precision with lower
latency, memory, throughput and energy cost.

The 2B4T report is even more explicit: the model is trained from scratch as a
native 1-bit model, not converted after FP training.

Implication:

```text
native success != proof that post-hoc same-shape conversion should work.
```

This matches our evidence. I2_S export works, but quality conversion is the hard
part.

### 2. The core layer is BitLinear, not ordinary Linear

The 2B4T report says standard full-precision `torch.nn.Linear` is replaced by
custom `BitLinear` layers. Inside them:

```text
weights: absmean ternary, 1.58-bit forward path
activations: absmax int8, per-token
normalization: SubLN for quantized-training stability
```

This is close to the mathematical object we validated:

```text
Wq = gamma * T
gamma = mean(abs(W))
T in {-1,0,+1}
```

But it is not the same as our current post-hoc recipe, because native BitNet
learns the hidden states and layer interactions while this constraint is active
from the beginning.

### 3. The FFN and normalization choices differ from plain LLaMA

The 2B4T report lists several architecture choices beyond BitLinear:

```text
SubLN normalization
ReLU^2 / squared ReLU FFN activation instead of SwiGLU
RoPE
no bias terms
LLaMA 3 tokenizer
```

The Hugging Face config also exposes a custom `BitNetForCausalLM` class through
`trust_remote_code`, `hidden_act = relu2`, tied embeddings, and an online BitNet
quantization configuration.

Implication:

```text
plain LLaMA architecture + ternary weights is a useful substrate test,
but it is not a faithful native BitNet architecture.
```

### 4. BitNet public runtime is inference-first

The Microsoft `BitNet` repository is an official inference framework with I2_S
and TL1/TL2 quantization paths. It gives us runtime evidence and deployment
formats, not a complete recipe for converting arbitrary FP checkpoints into
high-quality BitNet checkpoints.

Implication:

```text
bitnet.cpp can run a good ternary model.
It does not solve the optimization problem of creating one from a dense FP model.
```

### 5. Activation outliers are a known BitNet-specific issue

BitNet v2 introduces H-BitLinear with an online Hadamard transform before
activation quantization. That work targets native 4-bit activations for 1-bit
LLMs, but the lesson matters here:

```text
BitNet quality is not only about the weight codebook.
Activation distribution and representation geometry are first-class design variables.
```

This supports keeping rotation/Hadamard as a later candidate, but not as the main
answer to our current factual gap.

### 6. Capacity expansion is not a strange idea

BitNet b1.58 Reloaded reports strong small-model results and explicitly notes
that doubling hidden layer sizes helped small language models.

Implication:

```text
The user's capacity hypothesis is plausible:
if each parameter carries fewer states, the model may need more width,
more strips, more layers, or selective higher precision to carry the same function.
```

## What Is Not Publicly Confirmed

### No evidence that the final layer is specially exempt

The public materials do not currently prove:

```text
the final transformer block is full precision
the lm_head is special beyond tying / runtime format choices
late layers are intentionally higher-bit
```

The "last layers may need more capacity" idea is our hypothesis, not an observed
BitNet design fact.

### No public same-shape FP-to-BitNet conversion recipe

The public story is native training:

```text
train inside BitNet constraints -> export/run with bitnet.cpp
```

It is not:

```text
take arbitrary FP LLaMA -> one-shot ternary -> same quality
```

This is why our project remains useful even after bitnet.cpp exists.

### No proof that pure 1.58-bit same-shape is enough for conversion

Native BitNet success proves the ternary family can be good when trained
appropriately. It does not prove that the same architecture, same width, and same
checkpoint can be projected into that family after FP training.

## Comparison With Our Current Model

| Axis | Native BitNet public evidence | This fork so far | Gap |
| --- | --- | --- | --- |
| core linear | BitLinear | per-tensor b1.58 STE / I2_S | close |
| training | from scratch + SFT/DPO | post-training adaptation | major |
| normalization | SubLN | LLaMA RMSNorm mostly unchanged | possible |
| FFN activation | relu2 | SwiGLU in LLaMA targets | possible |
| activation quant | per-token int8 in BitLinear | runtime I2_S int8; PyTorch approximations | partial |
| capacity | native architecture chosen for ternary | same-shape dense checkpoint | major |
| runtime | bitnet.cpp | x86 I2_S validated | solved |
| factual quality | trained model has benchmark ability | adapted model fluent but facts weak | open |

## Strategic Conclusion

The evidence points away from another one-shot quantizer and toward a topology
question:

```text
Can we convert an existing model into a BitNet-friendly topology, not merely a
BitNet-coded version of the same topology?
```

That means the next serious branch should allow one or more of:

```text
selective higher precision
multi-strip ternary linears
small residual/adapters
width or intermediate expansion
late-layer capacity pockets
BitNet-native block choices such as SubLN/relu2
```

The target is still memory traffic. We should not add dense capacity everywhere.
The right experiment is selective and budgeted:

```text
maximize quality under a bytes/token budget
```

The concrete plan is [Hybrid / Variable BitNet Conversion Plan](./hybrid_variable_bitnet_conversion_plan.md).

## Audit To Do If We Need More Precision

If this becomes the main implementation track, run a direct code audit of the
public HF model repository:

```text
configuration_bitnet.py
modeling_bitnet.py
quantization modules / AutoBitLinear
state_dict tensor names
lm_head / tied embedding handling
SubLN placement
FFN relu2 implementation
```

Exit criteria:

```text
BITNET-AUDIT-001: table of every BitNetForCausalLM module vs LlamaForCausalLM
BITNET-AUDIT-002: exact last-block/lm_head precision and tying audit
BITNET-AUDIT-003: list of architecture changes that can be ported to conversion
```

