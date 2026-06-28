# DINO-I2S Self-Distillation Plan

Status: proposal / next objective candidate.

Document position: [Index](./index.md) -> [Current Theory](./current_theory_hypothesis_plan.md) -> factual-quality objective branch.

## One-Line Idea

```text
Use no-label self-distillation to make an I2_S student preserve the FP/base
model's content distribution and representation geometry, instead of trying to
memorize a small factual replay set.
```

This is not a new runtime format.

```text
I2_S stays the root.
DINO-style self-distillation is only an adaptation objective.
```

## Why This Exists

The FACT track found a clear split:

| attempt | result | lesson |
| --- | --- | --- |
| FACT-003C content-KL | fact `0.185`, recovered `0.845`, ok outputs | content anchoring helps |
| FACT-003D small hard replay | train facts memorized, eval facts down | small factual CE creates shortcut memorization |
| FACT-003H PopQA blend | active / decisive | representative data may be the structural fix |
| WSYNC / H-I2S data-free | FACT `0.0` | weight-only sync is not enough |

The recurring problem is not that the I2_S runtime cannot execute the model. It
can. The problem is that adaptation can move the student away from the base
model's factual/instruction manifold.

DINO-like self-distillation suggests a different pressure:

```text
Do not teach a few facts harder.
Keep the converted model inside the teacher's semantic field on broad unlabeled text.
```

## Literature Mapping

The exact method is ours, but the pattern is established:

| family | relevant idea | what we borrow |
| --- | --- | --- |
| DINO | self-distillation without labels; teacher/student views; centering/sharpening to avoid collapse | no-label teacher-student content targets |
| BYOL / Mean Teacher | EMA or frozen teacher consistency | optional EMA teacher after a frozen-teacher smoke |
| LLM-QAT / SQAKD-style QAT+KD | pretrained model distribution can supervise low-bit student | quantized student follows high-precision/base distribution |
| MiniLLM / DistiLLM / AKL | KL direction and token selection matter | avoid raw EOS KL; use content-only KL |

References:

- DINO: <https://arxiv.org/abs/2104.14294>
- BYOL: <https://arxiv.org/abs/2006.07733>
- Mean Teacher: <https://arxiv.org/abs/1703.01780>
- LLM-QAT: <https://arxiv.org/abs/2305.17888>
- SQAKD: <https://arxiv.org/abs/2403.11106>
- MiniLLM: <https://arxiv.org/abs/2306.08543>
- DistiLLM: <https://arxiv.org/abs/2402.03898>

## Core Math

Let:

```text
T_theta = FP/base teacher
S_phi   = I2_S-rooted student
x       = unlabeled text
v1, v2  = two views of x
```

The student still uses I2_S target linears:

```text
Wq_l = gamma_l T_l
T_l in {-1, 0, +1}
```

The objective is:

```text
L =
  L_answer_CE
  + lambda_c * KL_content(p_T(. | v1) || p_S(. | v2))
  + beta_h   * L_hidden
  + beta_a   * L_attention
  + beta_b   * L_balance
```

For the first smoke, use only:

```text
L_smoke =
  L_answer_CE
  + lambda_c * KL_content
  + beta_h   * L_hidden
```

### Content KL

Raw KL failed in FACT-003B because the student copied the chat teacher's stop
mass and produced empty answers. Therefore:

```text
V_content = V \ {EOS, BOS, PAD, special/control tokens}
```

```text
KL_content =
  sum_{t in answer positions}
  KL(
    softmax(z_T[t, V_content] / tau_T)
    ||
    softmax(z_S[t, V_content] / tau_S)
  )
```

Important:

```text
EOS decisions are not distilled.
Content distributions are distilled.
```

### Hidden Alignment

Use normalized layer states so the student does not have to match raw scale:

```text
L_hidden =
  sum_{l in L_probe}
  || normalize(h_S^l(v2)) - stopgrad(normalize(h_T^l(v1))) ||_2^2
```

Start with:

```text
L_probe = {middle layer, final layer before lm_head}
```

Do not align every layer in the first smoke. Full-layer hidden matching may
overconstrain the I2_S student and hide the signal.

### Optional DINO Centering

DINO avoids collapse by centering teacher outputs. For LLM logits, a conservative
variant is:

```text
c <- m*c + (1-m)*mean_batch(z_T_content)
z_T_centered = z_T_content - c
```

Use this only if the smoke shows degenerate high-confidence copying or empty
outputs. It is not first-run default.

## Why This Is Different From FACT-003D Replay

FACT-003D said:

```text
Here are 291 facts. Reproduce these answer tokens.
```

The model found the cheap shortcut:

```text
memorize train facts, fail held-out factual behavior.
```

DINO-I2S says:

```text
For broad unlabeled contexts, preserve the base model's content distribution and
hidden geometry.
```

That target is much harder to satisfy by memorizing a tiny table.

## Why This Is Different From PopQA Blend

PopQA blend is still supervised factual text:

```text
question -> answer
```

DINO-I2S can use unlabeled text:

```text
raw text / prompts / contexts without gold answer labels
```

The teacher supplies the soft semantic target. This matters if:

```text
representative factual labels are scarce,
noisy,
or expensive to curate.
```

## Data Plan

### PC Smoke Data

Use small, cheap, broad text:

```text
WikiText sample
Dolly prompts without answer supervision
PopQA questions as prompts, but no answer CE
synthetic factual prompts only as evaluation, not training
```

### Colab Data

Only if PC smoke moves behavior:

```text
50k-200k unlabeled prompt/context examples
optional PopQA blend at <=5%
FACT panel remains held out
PopQA tight heldout remains held out
```

## Experiment Ladder

### DINO-I2S-000: Code Path Smoke

Purpose:

```text
verify teacher/student forward, content-token mask, hidden alignment, no NaNs.
```

Model:

```text
160M on PC
steps: 20-50
```

Pass:

```text
loss finite
student train_ce not exploding
tags not all empty/salad
```

### DINO-I2S-001: 160M Objective Smoke

Arms:

| arm | loss |
| --- | --- |
| baseline | current best content-KL `lambda=0.2` + available blend |
| dino_logit | content-KL on unlabeled views |
| dino_hidden | content-KL + hidden alignment |
| dino_hidden_centered | only if collapse appears |

Metrics:

```text
FACT eval_panel
PopQA tight heldout
train_replay, if any
CE/PPL
generation tags
I2_S vs f16 parity
extra forward cost
```

Pass:

```text
FACT or PopQA tight improves by >= 0.05 absolute over baseline
AND tags remain ok/non-degenerate
AND train-only memorization signature is absent
```

Fail:

```text
FACT flat,
PopQA flat,
or hidden alignment improves CE while behavior stays flat.
```

### DINO-I2S-002: 1.1B Colab Gate

Run only if DINO-I2S-001 passes.

Use:

```text
TinyLlama-1.1B
same I2_S target linears
content-KL + hidden alignment
FACT panel held out
PopQA tight heldout held out
```

Pass:

```text
fact_rate > current content-KL baseline 0.185
preferably >= 0.25
tags ok
i2_s ~= f16
```

If DINO-I2S-002 passes:

```text
promote DINO-style no-label self-distillation to the main factual-retention objective.
```

If it fails:

```text
same-topology objective space is likely near exhausted;
return to I2_S-rooted capacity/topology or larger base-model ladder.
```

## Implementation Sketch

Add a new driver rather than overloading every path in `rt116` immediately:

```text
scripts/dino_i2s_selfdistill_smoke.py
```

Expected args:

```text
--model-id Felladrin/Llama-160M-Chat-v1
--steps 300
--seq-len 256
--batch-size 4
--lambda-content 0.2
--hidden-weight 0.01
--hidden-layers mid,last
--teacher-mode frozen_base
--view-mode dropout,span
--json-out reports/dino_i2s_160m_smoke.json
```

Possible reuse from existing code:

| existing file | reuse |
| --- | --- |
| `scripts/rt116_quality_recovery.py` | I2_S adaptation recipe, scoring/export hooks |
| `scripts/home001_activation_homeostasis_smoke.py` | teacher/student hidden-stat pattern |
| `scripts/fact003d_160m_sweep.py` | FACT/heldout scoring flow |

Do not start with EMA teacher. Use frozen FP teacher first. EMA is a second-stage
variant if frozen teacher helps.

## PC / Colab Split

### PC / RTX 3080

Use PC for:

```text
DINO-I2S-000 code smoke
DINO-I2S-001 160M objective smoke
ablation of hidden_weight in {0, 0.003, 0.01, 0.03}
```

Do not use PC for:

```text
1.1B full hidden-alignment training
```

### Colab

Use Colab for:

```text
DINO-I2S-002 1.1B gate, only after PC pass
longer unlabeled data runs
I2_S export/runtime scoring
```

## Decision Tree

```text
if FACT-003H PopQA blend succeeds:
    DINO-I2S becomes optional refinement / label-efficiency study

elif FACT-003H is flat but non-degenerate:
    run DINO-I2S-001 on PC
    if PC passes:
        run DINO-I2S-002 on 1.1B
    else:
        objective-only branch weak; reconsider capacity/topology

elif FACT-003H collapses:
    first fix stability/data path
    then revisit DINO-I2S
```

## Risks

| risk | why it matters | mitigation |
| --- | --- | --- |
| teacher copies wrong style | TinyLlama-Chat can prefer terse answers / EOS | exclude EOS, score tags, use content-only KL |
| hidden matching overconstrains I2_S | student cannot represent every FP hidden detail | normalize hidden states, probe only mid/last layers first |
| no new factual knowledge | self-distillation preserves, not creates | claim retention, not knowledge acquisition |
| extra compute | teacher forward doubles training cost | PC smoke first, freeze teacher, small probe layers |
| collapse to high-frequency content | DINO-style collapse possible | centering/sharpening only if needed |

## Claim Discipline

Allowed claim if successful:

```text
No-label self-distillation helps preserve FP/base factual behavior during I2_S
conversion better than small hard replay.
```

Do not claim:

```text
the student learned new facts without data.
```

The honest claim is:

```text
It forgot less.
```

