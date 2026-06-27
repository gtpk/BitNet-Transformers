# Windows Dev Environment + Session Handoff

Document position: [Index](./index.md). **Read this first** when continuing the BitNet b1.58
project on the Windows GPU server (gtpk@192.168.0.9). It records how to connect, the machine
spec, the dev-env setup, the current experiment state, and exactly where to continue — so a
fresh session can resume without re-discovery.

Last updated: 2026-06-27 (env build in progress).

---

## 1. SSH access

```text
host : 192.168.0.9   (LAN; reachable from the Mac dev box)
user : gtpk
auth : public-key (passwordless). NO password auth from the agent (non-interactive shell).
shell: Windows cmd.exe by default (Windows 11, build 22621). PowerShell available.
```

The agent's key is `~/.ssh/id_ed25519` (Mac), public key installed on the server:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFOPVhUslnm/O8UZAxvxhRaeT0K+VPgRnnPR7r0L0nNE claude-code-bitnet
```

Installed at `C:\Users\gtpk\.ssh\authorized_keys` (and/or
`C:\ProgramData\ssh\administrators_authorized_keys` if gtpk is an admin). Test:

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "hostname & ver"
```

Run remote commands non-interactively from the Mac with `ssh -o BatchMode=yes gtpk@192.168.0.9 "<cmd>"`.
cmd.exe quoting: separate statements with `&`, use `%VAR%` for env vars, `where`/`dir`/`findstr`.

## 2. Machine spec (verified 2026-06-27)

```text
GPU    : NVIDIA GeForce RTX 3080, 10240 MiB (10 GB) VRAM, driver 552.22
CUDA   : v12.1 (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1)
OS     : Windows 11 (10.0.22621), hostname GTPK
git    : 2.50.0.windows.1
python : C:\Python314\python.exe (3.14.2, default `python`), C:\Python313, and
         C:\Users\gtpk\anaconda3\python.exe (anaconda base). torch NOT installed anywhere yet.
home   : C:\Users\gtpk
```

**VRAM constraint (important):** 10 GB. The FACT-003 *training* recipe used ~17 GB on a Colab
L4 (1.1B fp32 student + grads + AdamW8bit + fp16 teacher). It will NOT fit at that config on
the 3080. So:
- **Inference / capacity probes (HYBRID-001A): fit fine** (1.1B fp16 ~2.2 GB, fp32 ~4.4 GB).
- **Training on the 3080** needs a reduced config (bf16, batch 1–2, grad-checkpointing,
  teacher in 8-bit or CPU-offloaded) — or keep training on Colab Pro L4 and use the 3080 for
  probes/inference/dev. Decide per task.

This box's win over Colab: **persistent (no VM recycle), private, always-on**. Colab Pro L4
(23 GB) is still the better box for the heavy FACT *training* runs.

## 3. Dev environment setup (build steps)

Status: **IN PROGRESS.** Fill in / check off as completed.

```text
[ ] clone repo:        git clone https://github.com/gtpk/BitNet-Transformers C:\Users\gtpk\BitNet-Transformers
[ ] conda env (py3.11): conda create -n bnt python=3.11 -y   (torch has no 3.14 wheels yet)
[ ] torch (CUDA 12.1):  pip install torch --index-url https://download.pytorch.org/whl/cu121
[ ] deps:               pip install transformers datasets safetensors sentencepiece accelerate huggingface_hub
[ ] (training only)     bitsandbytes is Linux-first; on Windows use bitsandbytes-windows or skip --optim adamw8bit
[ ] verify:             python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))"
```

Use the `bnt` conda env's python for all project work once built:
`C:\Users\gtpk\anaconda3\envs\bnt\python.exe`.

## 4. Project state — where we are (2026-06-27)

The teacher-free pivot is done; the project goal is a **usable, cheaply-runnable b1.58 / I2_S
model** ("흙수저용"), keeping the systems wins (small/fast, i2_s==f16, fluency, decoding
anti-collapse) and closing the factual gap. Full narrative:
[factual_gap_experiment_plan.md](./factual_gap_experiment_plan.md); strategy memory in the
agent's `base-anchored-pivot` memory.

FACT program result ladder (factual_panel_v1.jsonl, 27 prompts, eval-only; FP 0.81 / Q2_K 0.74):

```text
FACT-002 (data swap)        : 0.00–0.07   data is NOT the lever
FACT-003A (answer-only mask): 0.15        the OBJECTIVE is the lever (fixed empty-collapse)
FACT-003B (raw base-KL 1.0) : 0.00        anchor DESIGN bug: copied chat-teacher early-EOS
FACT-003C (content-KL)      : 0.185 @ λ=0.2   exclude EOS from KL = fix; INVERTED-U sweet spot
   λ sweep: 0.1 -> 0.037 (salad, too weak) | 0.2 -> 0.185 (best) | 0.5 -> 0.037 (over-anchored)
```

**Conclusion:** same-topology (target-linears-only, frozen lm_head/embeds) adaptation+objective
tuning at 1.1B **plateaus at fact ~0.185**. Reaching the 0.4 "usable" tier needs added
**capacity** (the variable/hybrid direction), not more λ. All committed; artifacts under
`reports/rt13{1,2,3,4}_*` (md5-verified).

## 5. NEXT TASK — HYBRID-001A late-layer capacity probe

Driver committed: `scripts/hybrid001a_capacity_probe.py`. Asks "is the 0.185 plateau capacity
or objective?" by restoring late layers to FP (from the trained STE latents) while the rest
stay ternary, and measuring fact_rate (PyTorch, rep-pen 1.2, contains-match) + CE + degeneration.
Arms A0(none)/A1(last1)/A2(last2)/A3(last4)/A4(last4 attn)/A5(last4 mlp). Verdict: some arm
>= 0.30 => capacity bottleneck -> HYBRID-001B cost-cut (Q4->Q3->Q2->2-plane on helpful layers);
barely moves => objective/data (lm_head unfreeze, protected factual replay).

```bash
python scripts/hybrid001a_capacity_probe.py \
  --ckpt <path>/fact003c_mixed_ckl0.2/ckpt.pt \
  --json-out reports/hybrid001a_capacity_probe.json --md-out reports/hybrid001a_capacity_probe.md
```

**BLOCKER — the latent checkpoint.** The probe needs the FACT-003C λ=0.2 STE latents:
`ckpt.pt` (6.4 GB), currently only on the **Colab Google Drive** at
`MyDrive/bnt_ckpt/fact003c_mixed_ckl0.2/ckpt.pt`. To run on Windows, get that file onto the box:
- easiest: download it from Google Drive (web or Drive-for-Desktop, same account) to the server, OR
- the Colab run that was launching HYBRID-001A may have finished — its result would be on the
  Colab VM disk at `/content/bnt/reports/hybrid001a_capacity_probe.{json,md}` (grab if the VM
  is still alive), OR
- regenerate the latents by re-running FACT-003C λ=0.2 (needs ~17 GB → Colab L4, not the 3080).

## 6. Transferring results back to origin

The Colab→origin direct `git push` was the standard; on Windows, `git push` works normally
(git 2.50 installed, repo cloned from gtpk's GitHub). Commit + push results from the Windows
box directly — no base64/Drive relay needed here. End commit messages with the project's
Co-Authored-By line.

## 7. Pointers

- Experiment plan / FACT narrative: [factual_gap_experiment_plan.md](./factual_gap_experiment_plan.md)
- Single-flight runbook: [factual_recovery_master_runbook.md](./factual_recovery_master_runbook.md)
- Hybrid/variable-capacity direction: [hybrid_variable_bitnet_conversion_plan.md](./hybrid_variable_bitnet_conversion_plan.md)
- Paper draft / claim table: [paper_draft.md](./paper_draft.md), [paper_skeleton.md](./paper_skeleton.md)
- bitnet.cpp runtime facts (pinned commit, const patch, I2_S=type36): agent memory `bitnet-cpp-runtime-env`
