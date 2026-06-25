# Colab Quantization-Aware Conversion Prompt

Use this prompt when handing the next experiment to an AI agent that can operate a
Colab runtime. It assumes the agent can run shell commands and Python in Colab, but
should not rely on local Mac bitnet.cpp runtime results.

```text
You are helping with the BitNet-Transformers research repo.

Goal:
Determine whether existing FP LLaMA checkpoints can be converted into BitNet-like
b1.58 models using real quantization techniques, instead of the naive absmean
ternary projection used so far.

Repository:
https://github.com/gtpk/BitNet-Transformers

Branch:
origin/main

Core docs to read first:
1. docs/index.md
2. docs/why_b158_conversion_is_hard.md
3. docs/quantization_aware_b158_conversion_plan.md
4. docs/g5_baseline_plan.md
5. docs/quality_recovery_plan.md

Important prior conclusions:
- x86/Linux bitnet.cpp I2_S is valid. Mac M5 I2_S/TL1 is not trusted.
- Systems/export are solved: materialized Wq=gamma*T can be exported to I2_S.
- Quality is not solved: one-shot ternary PTQ collapses; CE recovery helps but does
  not beat Q2_K on PPL; 1.1B all-I2_S generation loops.
- RT-123 showed naive additive mixed-bit DP is weak because layer interactions are
  strong.
- Therefore the next question is: did we fail because we used a primitive quantizer?

Hard rules:
- Start with JackFram/llama-160m.
- Do not start with TinyLlama-1.1B until 160M gives a strong signal.
- Do not use PPL alone; collect at least a small prompt/loop signal for promoted
  candidates.
- Do not claim Q2_K is beaten unless the same eval/tool table proves it.
- Keep all raw JSON under reports/.
- If a command silently keeps an old git HEAD, run:
    git fetch origin
    git reset --hard origin/main

Step 0: setup
Run:
    git clone https://github.com/gtpk/BitNet-Transformers /content/bnt || true
    cd /content/bnt
    git fetch origin
    git reset --hard origin/main
    git status --short

Then inspect:
    sed -n '1,220p' docs/quantization_aware_b158_conversion_plan.md
    ls scripts

Step 1: implement or locate an RT-124 probe script
If scripts/rt124_quantization_aware_probe.py does not exist, create it.
Keep the first version PyTorch-only. Do not build GGUF yet.

The script should:
- load JackFram/llama-160m
- create a fixed calibration/eval text split, preferably the same WikiText path used
  by earlier RT-116/121 runs if available
- locate target linears: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- evaluate FP CE/PPL on a short held-out split
- materialize several ternary variants and evaluate CE/PPL:
    A. per-tensor absmean gamma, threshold default
    B. row-wise gamma
    C. groupwise gamma with group sizes 64 and 128
    D. MSE-optimal per-tensor gamma / threshold grid
    E. activation-MSE scale if calibration activations are available
- write reports/rt124_quantization_aware_probe.json

Minimum JSON schema:
{
  "model_id": "JackFram/llama-160m",
  "eval": {"tokens": ..., "ctx": ...},
  "fp": {"ce": ..., "ppl": ...},
  "baselines": {
    "ptq_absmean_per_tensor": {"ce": ..., "ppl": ...},
    "q2k_from_rt121": {"ce": ..., "ppl": ..., "source": "rt121 if available"}
  },
  "candidates": [
    {
      "id": "rowwise_absmean",
      "family": "scale_granularity",
      "runtime_class": "upper_bound_not_i2s",
      "ce": ...,
      "ppl": ...,
      "delta_vs_ptq_ce": ...,
      "delta_vs_fp_ce": ...
    }
  ],
  "decision": {
    "best_candidate": "...",
    "branch": "scale_bottleneck | objective_bottleneck | assignment_needed | codebook_limit",
    "notes": "..."
  }
}

Step 2: run the cheap probe
Use a small enough token budget for a first pass, then rerun the top candidates with
more tokens if the result is noisy.

Expected first command shape:
    python scripts/rt124_quantization_aware_probe.py \
      --model-id JackFram/llama-160m \
      --eval-tokens 4096 \
      --calib-tokens 4096 \
      --ctx 128 \
      --json-out reports/rt124_quantization_aware_probe.json

Step 3: branch using this rule

If row-wise/groupwise/blockwise scales greatly improve CE:
    Branch = scale granularity bottleneck.
    Next action = design foldable column scaling or block-scale runtime path.

If MSE/threshold/activation-MSE scale beats absmean:
    Branch = native BitNet quantizer is not the right conversion quantizer.
    Next action = integrate best objective into teacher-free CE adaptation.

If all simple scale/threshold variants barely help:
    Branch = assignment/codebook/layer interaction.
    Next action = implement RT-125 GPTQ-style activation-MSE ternary projection.

If signed-epsilon 2-bit is requested or pure ternary remains weak:
    Branch = codebook limit.
    Next action = RT-127 signed-epsilon {-1,-eps,+eps,+1} PyTorch probe.

Step 4: report back
Return:
1. exact git commit hash
2. commands run
3. the result table
4. JSON path(s)
5. pass/fail/branch decision
6. anything suspicious about data size, tokenizer, or old repo state

Do not overclaim. The acceptable final answer should look like:
"RT-124A/B shows [scale/objective/assignment/codebook] is the likely bottleneck.
Recommended next step is [specific RT number]."
```

## Short Result Template

```text
RT-124 first pass result

commit:
model:
eval tokens:
calib tokens:

| candidate | runtime class | CE | PPL | delta CE vs PTQ | interpretation |
| --- | --- | ---: | ---: | ---: | --- |

Decision:

Next:

Caveats:
```
