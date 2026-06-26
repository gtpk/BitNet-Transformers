# Paper 3 Skeleton: Content-KL Factual Recovery

Working title:

```text
Content-KL Anchoring for Factual Recovery in b1.58 LLM Conversion
```

Status: active. This is the newest method paper. Do not freeze until the lambda sweep
finishes.

Central table: [Paper Evidence Matrix](./paper_evidence_matrix.md).

## Draft Abstract

After b1.58 export and runtime faithfulness are solved, the remaining product
question is not whether the low-bit runtime changes the model, but whether the
adaptation objective preserves useful knowledge. We show a factual gap in
teacher-free b1.58 conversion: FP and Q2_K TinyLlama answer a small factual panel,
while WikiText-adapted I2_S models become fluent but fact-poor. Data-only
instruction or mixed adaptation recovers fluency and CE but not facts. Answer-only
loss masking is the first objective change that moves the factual score. Raw
base-KL replay fails by copying the base chat model's stop/EOS behavior, causing
empty outputs. We introduce content-KL, which excludes EOS and special tokens from
the KL anchor and copies only content distribution mass. At lambda=0.2, content-KL
currently gives the best observed tradeoff: fact_i2s 0.185, CE recovery 0.845, and
non-degenerate outputs, while I2_S remains matched to F16. The result is not yet
factual parity with Q2_K, but it identifies the first working objective lever.

## Result Table For The Paper

| arm | fact_i2s | fact_f16 | CE recovery | behavior | interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| FP f16 reference | 0.815 | n/a | n/a | ok | base knows facts |
| Q2_K reference | 0.741 | n/a | n/a | ok | normal quantization preserves facts |
| PTQ i2_s | 0.000 | n/a | n/a | salad/collapse | no adaptation fails |
| WikiText adapted | 0.037 | 0.000 | high | ok/fluent | PPL recovery is not knowledge recovery |
| FACT-002 instruction | 0.000 | 0.000 | 0.403 | empty collapse | data-only instruction failed |
| FACT-002 mixed | 0.074 | 0.074 | 0.814 | ok/fluent | data recovers fluency, not facts |
| FACT-003A answer mask, mixed | 0.148-0.150 | 0.185-0.190 | 0.822 | ok | first objective signal |
| FACT-003B raw KL 1.0 | 0.000 | 0.000 | 0.474 | empty collapse | copied stop/EOS behavior |
| FACT-003C content-KL 0.1 | 0.037 | 0.037 | 0.484 | salad | anchor too weak |
| FACT-003C content-KL 0.2 | 0.185 | 0.185 | 0.845 | ok 27/27 | current best |
| FACT-003C content-KL 0.5 | TBD | TBD | TBD | TBD | pending |

## Blank Cells Before Submission

| blank | why it matters | next action |
| --- | --- | --- |
| content-KL lambda=0.5 | determines sweep shape | score and document the pending run |
| lambda=0.3/0.4 | needed if 0.5 beats 0.2 or partially collapses | run only if informative |
| seed check | lambda=0.2 may be stochastic | 2-3 seeds on best recipe |
| larger factual benchmark | 27 prompts are diagnostic, not a benchmark | create held-out factual subset |
| stronger quality target | current best 0.185 is below Q2_K 0.741 | decide whether objective tuning or HYBRID-001 is next |

## Thesis

The converted b1.58 model can be fluent and runtime-faithful while still losing
facts. The first objective that moves factual quality is content-KL: anchor the
student to the base model's content distribution while excluding EOS/special stop
mass from the KL.

## Do Claim

```text
FP/Q2_K know the facts; runtime is not the culprit.
Data-only adaptation recovers fluency, not facts.
Answer-only masking is necessary but not sufficient.
Raw KL fails because it copies the chat model's stop/EOS behavior.
Content-KL lambda=0.2 is the current best factual lever.
```

## Do Not Claim

```text
Q2_K/FP factual parity.
Final product quality.
That lambda=0.2 is universal before sweep/seed checks.
```

## Core Results

| arm | fact_i2s | CE recovered | behavior | conclusion |
| --- | ---: | ---: | --- | --- |
| FP f16 | 0.81 | n/a | ok | base knows facts |
| Q2_K | 0.74 | n/a | ok | quantization can preserve facts |
| WikiText adapted | 0.04 | high | fluent/canned | PPL recovery != knowledge |
| FACT-002 mixed | 0.07 | 0.81 | fluent hallucination | data alone insufficient |
| FACT-003A answer mask | 0.15 | 0.82 | ok | objective moves facts |
| FACT-003B raw KL 1.0 | 0.00 | 0.47 | empty collapse | copied EOS/stop |
| FACT-003C content-KL 0.1 | 0.037 | 0.484 | salad | too weak |
| FACT-003C content-KL 0.2 | 0.185 | 0.845 | ok 27/27 | current best |
| FACT-003C content-KL 0.5 | pending | pending | pending | decides sweep shape |

## Mechanistic Hint

Raw KL was not simply "too strong"; it copied the wrong part of the distribution:

```text
base chat model on Q/A prompt has high stop/EOS mass
student copies stop behavior
answers become empty
```

Content-KL changes the copied object:

```text
remove EOS/BOS/PAD/special ids from teacher and student distributions
renormalize
compute KL on content mass only
```

This keeps the useful anchor while not teaching the model to stop immediately.

## Figures

1. Factual panel: FP/Q2_K/PTQ/adapted.
2. FACT-002 data swap table.
3. FACT-003A/B/C comparison.
4. Lambda sweep curve: fact, CE recovery, degeneration.
5. Example outputs showing empty -> content answer.

## Missing Before Final

```text
lambda=0.5 result
possibly lambda=0.3/0.4 if 0.5 improves
seed check for best lambda
benchmark subset beyond 27-prompt panel
fair scorecard update
```

## Branch After Sweep

```text
If best fact_rate reaches >=0.3:
  continue objective tuning and benchmark.
If best fact_rate stays ~0.18:
  document plateau and start HYBRID-001A.
If lambda=0.5 collapses:
  freeze lambda=0.2 as default.
```
