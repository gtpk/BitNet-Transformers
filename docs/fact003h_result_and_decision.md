# FACT-003H PopQA Blend 1.1B -- Result Template + Decision Table

Document position: [Index](./index.md) -> the throne-decider after small hard replay (FACT-003D) died.

Status: TEMPLATE. FACT-003H is running on Colab (1.1B, PopQA 12.7k blended at 5%, content-KL 0.2,
no bolt-on mu). Fill the table from `MyDrive/bnt_results/fact003h_popqa_blend0.05/pyscore.json` when
the keep-alive loop reports completion. The whole point: does a representative-distribution blend
move 1.1B b1.58 factual behaviour where a tiny hard replay could not?

## Baselines (same eval panel, PyTorch ternary scoring)

```text
FP 160M reference (different model)         eval 0.296   -- not comparable, context only
FACT-003C content-KL (no replay)            eval 0.185   -- best 1.1B so far
FACT-003D mu=1.0 (bolt-on hard replay)      eval 0.111   train_atomic 1.00  (memorise, eval down)
FACT-003D mu=0.25 (gentler)                 eval 0.037   train_atomic 0.588 (worse, lever != mu)
```

## Result Template (fill on completion)

| metric | value | baseline / note |
| --- | ---: | --- |
| eval_panel (FACT-001, 27) | TBD | vs FACT-003C 0.185 (the bar to beat / hold) |
| popqa_tight_PRIMARY (916 sample) | TBD | PRIMARY transfer signal (alias<=3, clean) |
| popqa_loose (1412 sample) | TBD | permissive secondary read only |
| popqa_train_memorise (80) | TBD | memorisation control (high alone = shortcut) |
| CE / PPL (WikiText) | TBD | fluency; recovered_fraction in train.json |
| tags (eval_panel) | TBD | empty/loop/salad = degeneration |
| i2_s vs f16 agreement | TBD* | *scored in PyTorch ternary; for the formal I2_S/f16
GGUF parity export the saved adapted_model to bitnet.cpp + rt130 (prior runs: +0.0015 nats, 26-27/27) |

## Decision Table (next experiment by outcome)

| case | signal | conclusion | next |
| --- | --- | --- | --- |
| **A** | eval_panel > 0.185 AND popqa_tight rises | representative blend WORKS -- the structural fix landed | strengthen Track A mainline: blend ratio sweep / data scale (5k->20k) / 1.1B seed variance |
| **B** | popqa_tight rises but eval_panel flat (<=~0.185) | PopQA distribution transfers, but FACT panel domain-mismatches | widen the eval panel / add factual-task diversity; the model CAN do factual QA, the 27-item panel just doesn't catch it |
| **C** | popqa_train rises, popqa_tight flat | PopQA also became a memorise shortcut (even at 5% blend) | bigger/more-diverse data OR objective redesign (content-AKL); blend ratio down |
| **D** | everything flat (eval ~0.185, tight low) | same-topology I2_S adaptation has plateaued | add I2_S-rooted auxiliary capacity (SIDE I2_S+LoRA sidecar) / PTQTP-lite / from-start hybrid -- the sidecar is an auxiliary organ of I2_S, not a non-I2_S replacement |
| **E** | CE improves but facts drop | CE objective still misaligned with factual behaviour | strengthen content-KL/AKL / objective term, not data |

## Claim discipline

- popqa_tight is the PRIMARY transfer metric (runbook rule 7); popqa_loose is permissive secondary.
- A good PopQA result is a "representative-blend transfers" signal, NOT a benchmarked factual-ability
  claim -- PopQA license is unclear (research/direction only); a product claim needs a license-clean
  fixed benchmark held-out.
- WSYNC (data-free weight-only sync) is demoted (H8 false data-free): not a contributor here.
