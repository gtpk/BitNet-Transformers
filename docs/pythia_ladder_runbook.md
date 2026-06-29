# PYTHIA-LADDER-001 Runbook: scale-dependent auxiliary-objective stability threshold

Document position: [Index](./index.md). Follows the confound resolution
([reports/dino_ctrl_fp32_result.md](../reports/dino_ctrl_fp32_result.md)): the 1.1B DINO collapse is
SCALE-driven, not precision/optimizer. This runbook fixes the protocol BEFORE running, so the ladder
is a controlled measurement, not a model swap.

## Central question

> Same minimal content-KL + DINO-logit recipe, same tokenizer family, same data, varying ONLY scale.
> **At which scale does the auxiliary objective start to collapse generation?**

Core claim being tested:

> **b1.58 same-topology adaptation has a scale-dependent auxiliary-objective stability threshold.**

This is sharper than "DINO works/fails": we measure a *collapse-onset scale* and its telemetry
signature. TinyLlama 160M-vs-1.1B could not answer it (different family/tokenizer/pretraining/chat
tuning all confounded). Pythia is a controlled scale series (same architecture, tokenizer, Pile data,
training curriculum) -> scale is the only variable.

## 1. Model list (fixed)

EleutherAI Pythia (standard, non-deduped, for one consistent series):

| model | params | primary host |
| --- | --- | --- |
| EleutherAI/pythia-160m | 160M | Mac/MPS or 3080 |
| EleutherAI/pythia-410m | 410M | Mac/MPS or 3080 |
| EleutherAI/pythia-1b | 1B | Colab L4 / 3080 |
| EleutherAI/pythia-1.4b | 1.4B | Colab L4 |
| EleutherAI/pythia-2.8b | 2.8B | Colab L4 (bf16) or A100 |

All are BASE LM (not chat-tuned) on the Pile, sharing the GPT-NeoX-20B tokenizer. Self-anchored: the
frozen FP Pythia of the same size is the teacher (as in our TinyLlama setup).

## 2. Common data (fixed, tokenizer-agnostic text -> re-tokenized per model)

- Train stream: `--train-source mixed --answer-loss-only` (WikiText + Dolly instruction, de-leaked
  vs the factual panel), `--max-train-tokens 2_000_000`.
- Content-KL replay pool: de-leaked instruction (`--base-kl-replay --kl-content-only`).
- DINO unlabeled views: sampled from the same train stream (`--dino-view-mode dropout --dino-view-p 0.1`).
- Eval (held out, never trained): `data/factual_panel_v1.jsonl` (27, with categories) +
  `data/popqa_heldout_tight.jsonl`. NOTE: Pythia is base (not instruction-tuned), so absolute
  fact_rate may be low at every scale -- that is fine; the collapse signal is degeneracy + gold-rank
  vs the model's OWN FP teacher, not absolute fact_rate (see Section 6).

## 3. Common objective (fixed, identical across scales)

The exact recipe that is positive at 160M (TinyLlama) and collapses at 1.1B:

```text
answer-only CE  +  0.2 * content-KL(base||student, EOS/special dropped)
                +  0.2 * DINO-logit(teacher_clean || student_dropout_view, all positions, EOS dropped)
all target linears ternary (b1.58 I2_S); lm_head + embeds FROZEN; NO centering/warmup (we WANT to
observe the raw collapse, not suppress it).
```

## 4. Step / checkpoint / precision schedule (fixed)

```text
--steps 800  --seq-len 256  --batch 4 --grad-accum-steps 6 (effective 24)  --lr 2e-4  --seed 1
--dtype bfloat16 --optim adamw8bit --grad-checkpointing   (precision/optimizer ruled out as cause,
                                                           so use the memory-efficient pair)
--ckpt-dir <Drive>  --ckpt-every-min 25   (resumable)
--dino-collapse-check-every 50 --dino-collapse-min-step 0 --dino-collapse-salad-thresh 0.3
   (check EVERY 50 steps from the start to pin collapse_onset_step; do NOT stop at first crossing --
    log onset and CONTINUE to confirm it is sustained, see Section 7)
telemetry logged every 25 steps (Section 5).
```

Seeds: primary seed 1; if an onset looks borderline, repeat that one scale at seeds 2/3.

## 5. Telemetry schema (per logged step -> metrics.jsonl on Drive)

Required fields (this needs instrumentation -- see Prerequisite P2):

```json
{
  "scale": "pythia-1b", "step": 200, "pct": 25.0,
  "train_ce": 7.29, "content_kl": 5.81, "dino_loss": 5.48,
  "grad_norm": 1.42, "update_norm": 0.013,
  "logit_entropy": 3.10, "top1_prob": 0.22,
  "gold_rank_mean": 47.3, "gold_logp_mean": -6.1,
  "degenerate_rate": 0.00, "salad_rate": 0.0, "loop_rate": 0.0, "empty_rate": 0.0,
  "hidden_var_mid": 0.91, "hidden_var_last": 0.74
}
```

- `logit_entropy`, `top1_prob`: mean over a fixed held-out probe batch (early-warning of sharpening/
  spiking before visible salad).
- `gold_rank_mean`, `gold_logp_mean`: panel gold-token rank/logp (the DINO-DIAG-001 metric) -- the
  earliest quantitative sign of the student diverging from its FP teacher.
- `degenerate_rate` = salad+loop+empty fraction on the panel probe (the collapse detector value).
- `hidden_var_mid/last`: variance of mid/last hidden states (collapse often shows as a variance
  blow-up or collapse).
- `grad_norm` (pre-clip), `update_norm` (||optimizer step||): instability shows here first.

## 6. Collapse criterion (fixed)

```text
collapse_onset_step := first step S where degenerate_rate(S) > 0.30 AND degenerate_rate stays > 0.30
                       for the next 2 consecutive checks (sustained, not a transient blip).
If no such S by step 800 -> "stable" for that scale.
Secondary corroboration (must agree): gold_rank_mean diverges sharply upward from the FP teacher's,
top1_prob spikes toward 1.0 (or entropy collapses), grad_norm/update_norm spike at onset.
```

The primary metric is **degeneracy relative to the model's own FP teacher**, NOT absolute fact_rate
(Pythia base may sit near zero fact_rate at all scales). A scale is "stable" if the I2_S+DINO student
keeps the teacher's generation quality (low degenerate_rate, gold_rank tracking teacher).

## 7. Stop conditions

- Per scale: run to 800 steps if cheap (<=1B). For 1.4B/2.8B, once collapse is CONFIRMED sustained
  (criterion in S6 met for 3 checks), may stop early to save compute -- but ALWAYS record the onset
  step + telemetry around it; never stop on the first single crossing (could be transient).
- Global: if a smaller scale is stable and the next is collapse, the threshold is bracketed there;
  optionally bisect (e.g. add pythia-700m) only if the onset bracket is wide and worth narrowing.
- Hard stop on OutOfMemory (report + drop batch / move host), or NaN loss (report).

## 8. Verdict table (user)

| pattern | meaning | follow-up |
| --- | --- | --- |
| 160M ok, 410M ok, 1B collapse | onset ~1B | bisect 700M; confirm thesis |
| 160M ok, 410M collapse | earlier than expected scaling instability | re-check recipe; bisect 256M |
| all ok | TinyLlama-1.1B structure/data idiosyncrasy, not generic scale | revisit the TinyLlama-specific cause |
| all collapse | objective mismatched to Pythia base | reconsider objective/teacher for base LMs |
| onset step EARLIER as scale grows | strong scale-driven collapse thesis | headline result |

## 9. Cost estimate + host split

| scale | recipe fit | host | ~wall-clock (800 steps) | ~Colab CU |
| --- | --- | --- | --- | --- |
| 160M | trivial | Mac/MPS or 3080 (free) | ~10-20 min | 0 |
| 410M | trivial | Mac/MPS or 3080 (free) | ~20-40 min | 0 |
| 1B | bf16 ~11GB | 3080 (10GB? tight -> L4) / L4 | ~80 min | ~3-5 |
| 1.4B | bf16 ~14GB | Colab L4 | ~110 min | ~4-7 |
| 2.8B | bf16 ~22GB | Colab L4 (tight) or A100 | ~3 h L4 / ~1 h A100 | L4 ~10 / A100 ~15 |

Host split: **Mac/MPS = 160M/410M (free, local)**; **Colab L4 = 1B/1.4B**; **2.8B = L4 if it fits
(bf16+adamw8bit) else A100**. 3080 (10GB) can do 160M/410M/maybe-1B. Most of the ladder is cheap; only
2.8B is potentially A100.

## Prerequisites (MUST do before PYTHIA-LADDER-001 runs)

- **P1 -- GPT-NeoX target-linear support.** `bitnet_llama/conversion.is_target_weight_key` +
  `replace_targets` are Llama-named (q_proj/k_proj/v_proj/o_proj/gate/up/down_proj). Pythia/GPT-NeoX
  linears are `attention.query_key_value`, `attention.dense`, `mlp.dense_h_to_4h`, `mlp.dense_4h_to_h`.
  Extend the target matcher to cover these (and assert n_target_linears > 0 at load, else the model is
  NOT quantised and the run is meaningless). Verify on pythia-160m: replace_targets returns >0.
- **P2 -- telemetry instrumentation (Section 5).** rt116 currently logs train_ce/kl/dino/fce/home.
  Add the probe-based metrics (logit_entropy, top1_prob, gold_rank_mean/logp on a fixed panel probe,
  degenerate_rate, hidden_var_mid/last) and the optimiser metrics (grad_norm pre-clip, update_norm) to
  metrics.jsonl every log step. Reuse the collapse detector's generation + DINO-DIAG-001 gold-rank code.
- **P3 -- smallest-first smoke.** Before the full ladder, run pythia-160m as a code+telemetry smoke
  (few steps) to validate P1+P2 end-to-end (target linears > 0, telemetry fields populate, no NaN).
  Run it on the Mac/MPS (free).

## Execution order

1. Implement P1 + P2; verify with the pythia-160m smoke (P3) on Mac/MPS.
2. Run the ladder smallest -> largest: 160M, 410M (Mac), 1B, 1.4B (L4), 2.8B (L4/A100). Same recipe,
   same schedule, full telemetry, Drive-checkpointed.
3. Plot collapse_onset_step vs scale + the telemetry signatures at onset; fill the Section 8 table.
4. Write up: the scale-dependent stability threshold result.
