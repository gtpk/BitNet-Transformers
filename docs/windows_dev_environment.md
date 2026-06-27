# Windows Dev Environment + Session Handoff

Document position: [Index](./index.md). **Read this first** when continuing the BitNet b1.58
project on the Windows GPU server (gtpk@192.168.0.9). It records how to connect, the machine
spec, the dev-env setup, the current experiment state, and exactly where to continue — so a
fresh session can resume without re-discovery.

Last updated: 2026-06-27 (RTX 3080 is a validated fast-predictor/eval box).

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

**Full SSH guide** (key setup, sshd/firewall, cmd quoting, long-job patterns, troubleshooting):
[SSH Access to the Windows GPU Server](./ssh_windows_access.md).

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

## 3. Dev environment setup — BUILT (2026-06-27)

```text
[x] repo:        C:\Users\gtpk\BitNet-Transformers   (git clone of gtpk/BitNet-Transformers)
[x] conda env:   bnt  (python 3.11) -> C:\Users\gtpk\anaconda3\envs\bnt\python.exe
[x] torch:       2.5.1+cu121   (pip install torch --index-url https://download.pytorch.org/whl/cu121)
[x] deps:        transformers 5.12.1, datasets, numpy 2.4.6, safetensors, sentencepiece, accelerate, huggingface_hub
[x] verified:    torch.cuda.is_available()=True, RTX 3080 (9 GB usable); bitnet_llama.module imports; GPU matmul OK
[ ] bitsandbytes: NOT installed (Linux-first; only needed for --optim adamw8bit training -> use plain adamw on Win,
                  or train on Colab L4). 
```

**Use this for all project work on the box:** `C:\Users\gtpk\anaconda3\envs\bnt\python.exe`
(run from `C:\Users\gtpk\BitNet-Transformers`). Example:

```bash
ssh -o BatchMode=yes gtpk@192.168.0.9 "cd C:\Users\gtpk\BitNet-Transformers & C:\Users\gtpk\anaconda3\envs\bnt\python.exe scripts\<driver>.py <args>"
```

Note: transformers is 5.x (newer than the Colab runs used) — watch for minor `generate()` /
`from_pretrained(dtype=...)` API differences if a script errors. Long jobs: launch detached
(e.g. `start /b ...` or a background python) and poll a logfile, since the SSH call is one-shot.

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

**Updated conclusion:** same-topology content-KL found a real lever but plateaus at
`fact ~0.185`. The first cheap capacity patch, post-hoc late-layer FP restore, failed.
The next successful signal came from **protected factual replay**:

```text
FACT-003D 160M predictor on this 3080:
  mu=0.0 control   eval_panel 0.037, heldout_atomic 0.093
  mu=1.0 replay    eval_panel 0.259, heldout_atomic 0.227
```

This is the first strong evidence that factual recovery can transfer to disjoint eval
items rather than only memorising the replay set. Colab 1.1B `mu=1.0` is the decisive
run.

## 5. HYBRID-001A — DONE (result: NOT a capacity bottleneck)

`scripts/hybrid001a_capacity_probe.py` was run on the FACT-003C λ=0.2 STE latents (committed
`977f21c`; `reports/hybrid001a_capacity_probe.{json,md}`). Restoring late target-linears to FP
(STE latent) while the rest stay ternary:

```text
arm  restore           FP    fact   CE     ppl
A0   none (all I2_S)    0     0.148  3.84   46.5   <- BEST on every axis
A1   last 1 block       44M   0.074  4.94   139
A2   last 2 blocks      88M   0.037  4.73   114
A3   last 4 blocks      176M  0.111  4.56   95
A4   last 4 attn        38M   0.111  5.40   222
A5   last 4 MLP         138M  0.037  4.97   144
```

**Verdict: not (post-hoc) capacity.** Every restore arm is <= A0 and CE gets WORSE. The model
was STE-trained with ALL target linears ternary, so early layers co-adapted to feed *ternary*
late layers; post-hoc un-quantizing late layers breaks that coherence (distribution mismatch).
This rules out the cheap post-hoc capacity patch — it does NOT rule out a **train-from-start
hybrid** (early layers co-adapt to FP late layers from the beginning).

## 5b. FACT-004A — DONE (result: lm_head unfreeze is harmful)

FACT-004A tested whether frozen `lm_head` was the bottleneck:

```text
FACT-003C frozen lm_head: fact 0.185
FACT-004A train lm_head:  fact 0.04 / 0.00
```

Verdict:

```text
lm_head unfreeze is DISCARDED.
More output freedom caused fluent forgetting, not factual recovery.
```

## 5c. Current 3080 Role — keep it busy but not speculative

The 3080 is now a validated fast-predictor:

```text
FACT-004A 160M smoke predicted the lm_head direction before/around the 1.1B result.
FACT-003D 160M mu-sweep predicted mu=1.0 and showed replay transfer.
```

While Colab runs the long 1.1B `FACT-003D mu=1.0` job, use the box for:

```text
1. eval-only dry runs to keep Windows scoring path warm
2. 160M seed variance for FACT-003D mu=1.0
3. post-result scoring/report parsing when Colab artifacts arrive
```

Do **not** launch new speculative branches on the 3080 before the 1.1B result lands.
See [RTX 3080 Parallel Queue](./box_3080_parallel_queue.md).

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
