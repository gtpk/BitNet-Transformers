#!/usr/bin/env python3
"""RT-116 / TRAIN-001 (QR-001..003): teacher-free b1.58 quality recovery on Llama-160M.

RT-112..115 closed the systems story (small + fast + faithful at scale). The open
question is quality: a one-shot PTQ ternary model has terrible absolute PPL by
design — can a SHORT, teacher-free CE-only adaptation recover a meaningful fraction
of the FP->PTQ loss? See docs/quality_recovery_plan.md for the full ladder.

This implements, all in PyTorch on one held-out text set (apples-to-apples, no
cross-tool noise):

  QR-001  collapse baseline : CE_fp (FP original) vs CE_ptq (Wq=gamma*T, no train)
  QR-002a CE adaptation     : replace target linears with per-tensor b1.58 STE,
                              freeze everything else, train next-token CE only,
                              report recovered_fraction = (CE_ptq-CE_adapted)
                                                          /(CE_ptq-CE_fp)
  QR-003  runtime preserve  : (with --bitnet) export adapted Wq=gamma*T -> f16 & i2_s
                              GGUF, llama-perplexity, check i2_s ~= f16 in nats.

Forward uses Wq=gamma*T (PerTensorBitLinear STE); backward updates the latent FP
weight through the STE. Teacher-free = plain next-token CE, no distillation.

USAGE (GPU strongly recommended for the training step):
  python scripts/rt116_quality_recovery.py --model-id JackFram/llama-160m \
    --steps 300 --seq-len 256 --batch 8 --lr 2e-4 \
    --bitnet /content/bitnet.cpp     # add --bitnet to also run QR-003 export

For the 1.1B budget-scaling run, keep `--batch` as the per-microbatch size that
fits in GPU memory and use `--grad-accum-steps` to raise the effective batch.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama.module import PerTensorBitLinear  # noqa: E402


def set_module(root, name, mod):
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, mod)


def replace_targets(model):
    """Swap target nn.Linear -> PerTensorBitLinear (latent = FP weight). Returns count."""
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            repl = PerTensorBitLinear(m.in_features, m.out_features, bias=m.bias is not None)
            with torch.no_grad():
                repl.weight.copy_(m.weight)
                if m.bias is not None:
                    repl.bias.copy_(m.bias)
            set_module(model, name, repl)
            n += 1
    return n


def materialize_and_save(model, out_dir, tokenizer):
    """Replace PerTensorBitLinear -> dense Linear holding Wq=gamma*T, save HF dir."""
    import copy
    m2 = copy.deepcopy(model).to("cpu").float()
    for name, mod in list(m2.named_modules()):
        if isinstance(mod, PerTensorBitLinear):
            dense = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            with torch.no_grad():
                dense.weight.copy_(C.per_tensor_b158_approx(mod.weight))
                if mod.bias is not None:
                    dense.bias.copy_(mod.bias)
            set_module(m2, name, dense)
    out_dir.mkdir(parents=True, exist_ok=True)
    m2.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)


def load_wikitext(tokenizer, max_train_tokens, max_eval_tokens):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def join_tok(split, cap):
        text = "\n\n".join(t for t in ds[split]["text"] if t.strip())
        ids = tokenizer(text, return_tensors=None)["input_ids"]
        return torch.tensor(ids[:cap], dtype=torch.long)

    return join_tok("train", max_train_tokens), join_tok("validation", max_eval_tokens)


def _wikitext_train_eval(tokenizer, max_train_tokens, max_eval_tokens):
    return load_wikitext(tokenizer, max_train_tokens, max_eval_tokens)


def _instruction_text(max_examples=20000):
    """Dolly-15k formatted as 'Q: instruction [context]\\nA: response' (license: CC BY-SA)."""
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    parts = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ctx = (ex.get("context") or "").strip()
        q = ex["instruction"].strip() + (("\n" + ctx) if ctx else "")
        parts.append(f"Q: {q}\nA: {ex['response'].strip()}")
    return "\n\n".join(parts)


def _instruction_ids_mask(tokenizer, max_examples=20000):
    """FACT-003A: Dolly Q/A as a token stream + answer mask (True on response tokens).

    Each example is tokenized as prompt 'Q: ..\\nA:' + answer ' <response>' + '\\n\\n' sep,
    with the answer-mask True only on the response tokens. Under --answer-loss-only the CE
    is computed on these tokens only, so the model is not trained to reproduce the Q/A prompt
    formatting (the thing instruction-only adaptation overfit into empty-answer collapse, FACT-002).
    """
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    sep = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    ids, mask = [], []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ctx = (ex.get("context") or "").strip()
        q = ex["instruction"].strip() + (("\n" + ctx) if ctx else "")
        p = tokenizer(f"Q: {q}\nA:", add_special_tokens=False)["input_ids"]
        a = tokenizer(" " + ex["response"].strip(), add_special_tokens=False)["input_ids"]
        ids += p + a + sep
        mask += [0] * len(p) + [1] * len(a) + [0] * len(sep)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def load_corpus(source, tokenizer, max_train_tokens, max_eval_tokens, answer_mask=False):
    """Returns (train_ids, eval_ids, train_answer_mask). eval is ALWAYS WikiText validation.

    train_answer_mask[i] is True for tokens whose CE counts under --answer-loss-only: response
    tokens for instruction data, and all content tokens for WikiText (no prompt/answer split).
    When answer_mask=False the mask is all-True and the token stream is byte-identical to the
    FACT-002 runs (the masked-stream tokenization is only built when actually needed)."""
    wt_train, wt_eval = _wikitext_train_eval(tokenizer, max_train_tokens, max_eval_tokens)
    if source == "wikitext":
        return wt_train, wt_eval, torch.ones_like(wt_train, dtype=torch.bool)
    if answer_mask:
        instr, instr_msk = _instruction_ids_mask(tokenizer)
    else:
        instr = torch.tensor(tokenizer(_instruction_text())["input_ids"], dtype=torch.long)
        instr_msk = torch.ones_like(instr, dtype=torch.bool)
    if source == "instruction":
        return instr[:max_train_tokens], wt_eval, instr_msk[:max_train_tokens]
    if source == "mixed":
        half = max_train_tokens // 2
        wt_part = wt_train[:half]
        train = torch.cat([instr[:half], wt_part])
        msk = torch.cat([instr_msk[:half], torch.ones_like(wt_part, dtype=torch.bool)])
        return train, wt_eval, msk
    raise ValueError(source)


@torch.no_grad()
def eval_ce(model, eval_ids, seq_len, device, max_windows=64):
    model.eval()
    n = min(eval_ids.numel() // seq_len, max_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    tot, cnt = 0.0, 0
    for i in range(0, n, 8):
        b = ids[i : i + 8]
        loss = model(input_ids=b, labels=b).loss
        tot += float(loss) * b.shape[0]
        cnt += b.shape[0]
    return tot / max(cnt, 1)


def base_kl_replay_term(model, teacher, replay_ids, replay_mask, seq_len, replay_batch, temp, gen, device):
    """FACT-003B: answer-masked forward-KL(base||student) on a sampled replay window.

    Returns a scalar loss tensor with grad through the student (teacher is frozen / no-grad),
    averaged over answer tokens only, scaled by temp^2 (standard KD). Returns None if the
    sampled window has no answer tokens."""
    import torch.nn.functional as F
    usable = replay_ids.numel() - 1
    starts = torch.randint(0, max(1, usable - seq_len), (replay_batch,), generator=gen).tolist()
    rx = torch.stack([replay_ids[s : s + seq_len] for s in starts]).to(device)
    rm = torch.stack([replay_mask[s : s + seq_len] for s in starts]).to(device)
    if not rm.any():
        return None
    with torch.no_grad():
        t = teacher(input_ids=rx).logits.float() / temp
        p_t = F.softmax(t, dim=-1)
        logp_t = F.log_softmax(t, dim=-1)
    logp_s = F.log_softmax(model(input_ids=rx).logits.float() / temp, dim=-1)
    kl_tok = (p_t * (logp_t - logp_s)).sum(-1)        # [B,T] forward KL per token
    return kl_tok[rm].mean() * (temp * temp)          # answer tokens only, KD temp^2 scale


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--out-dir", type=Path, default=None, help="adapted HF dir (default <work>/<slug>_adapted)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="number of microbatches per optimizer step; effective batch = batch * grad_accum_steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-tokens", type=int, default=600_000)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--ppl-eval-tokens", type=int, default=3_000,
                    help="tokens written to eval.txt for the GGUF llama-perplexity (QR-003); "
                         "keep small, CPU perplexity is ~per-token")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                    help="model compute dtype (bfloat16 saves memory for 1B+ on a 16GB GPU)")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                    help="adamw8bit (bitsandbytes) cuts optimizer-state memory ~4x for 1B+")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade compute for activation memory (needed for 1B+ on a 16GB GPU)")
    ap.add_argument("--train-norms", action="store_true",
                    help="QR-002b: also adapt RMSNorm weights (may absorb quantization drift)")
    ap.add_argument("--train-lm-head", action="store_true",
                    help="QR-002c: also adapt lm_head (output distribution retune)")
    ap.add_argument("--bitnet", type=Path, default=None, help="if set, run QR-003 export+perplexity")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--train-source", choices=["wikitext", "instruction", "mixed"], default="wikitext",
                    help="FACT-002: adaptation data (eval is always WikiText validation; factual eval is rt130)")
    ap.add_argument("--answer-loss-only", action="store_true",
                    help="FACT-003A: compute CE only on response tokens of instruction data "
                         "(prompt 'Q:..\\nA:' + separators masked to -100); WikiText content "
                         "tokens always count. Avoids overfitting the Q/A prompt formatting.")
    ap.add_argument("--base-kl-replay", action="store_true",
                    help="FACT-003B: base-anchored adaptation. Add KL(base||student) on the answer "
                         "tokens of a fixed instruction replay set, with the SAME original FP model "
                         "as a frozen self-teacher, so b1.58 keeps the base model's answer behaviour.")
    ap.add_argument("--kl-weight", type=float, default=1.0, help="FACT-003B: weight lambda on the KL replay term")
    ap.add_argument("--kl-temp", type=float, default=1.0, help="FACT-003B: distillation temperature for the KL term")
    ap.add_argument("--replay-tokens", type=int, default=200_000,
                    help="FACT-003B: size of the fixed instruction replay pool the KL anchor samples from")
    ap.add_argument("--replay-batch", type=int, default=2,
                    help="FACT-003B: microbatch of replay windows per step for the KL term (keep small)")
    ap.add_argument("--teacher-dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                    help="FACT-003B: dtype of the frozen base teacher (float16 halves its memory)")
    args = ap.parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    slug = args.model_id.split("/")[-1]
    out_dir = args.out_dir or args.work / f"{slug}_adapted"
    print(f"device={device}  model={args.model_id}")

    tdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=tdtype).to(device)
    if args.grad_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # else ckpt warns: no input needs grad (embeds frozen)

    train_ids, eval_ids, train_mask = load_corpus(
        args.train_source, tok, args.max_train_tokens, args.max_eval_tokens,
        answer_mask=args.answer_loss_only)
    print(f"train-source={args.train_source}")
    print(f"train tokens={train_ids.numel():,}  eval tokens={eval_ids.numel():,}")
    if args.answer_loss_only:
        print(f"FACT-003A answer-loss-only: {100*float(train_mask.float().mean()):.1f}% of train "
              f"tokens are answer/content (CE masked to these; prompt+sep -> -100)")

    # FACT-003B: frozen base teacher (self-anchor) + fixed instruction replay pool
    teacher = None
    replay_ids = replay_mask = None
    if args.base_kl_replay:
        ttd = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.teacher_dtype]
        teacher = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=ttd).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.config.use_cache = False
        r_ids, r_mask = _instruction_ids_mask(tok)
        replay_ids, replay_mask = r_ids[: args.replay_tokens], r_mask[: args.replay_tokens]
        print(f"FACT-003B base-KL replay: teacher={args.model_id} ({args.teacher_dtype}, frozen); "
              f"replay pool {replay_ids.numel():,} instruction tokens "
              f"({100*float(replay_mask.float().mean()):.1f}% answer); "
              f"lambda={args.kl_weight} temp={args.kl_temp} replay_batch={args.replay_batch}")

    # QR-001: FP baseline, then PTQ collapse
    ce_fp = eval_ce(model, eval_ids, args.seq_len, device)
    n_lin = replace_targets(model)
    model.to(device=device, dtype=tdtype)  # new PerTensorBitLinear modules -> model dtype
    ce_ptq = eval_ce(model, eval_ids, args.seq_len, device)
    print(f"QR-001  CE_fp={ce_fp:.4f} (ppl {math.exp(ce_fp):.2f})  "
          f"CE_ptq={ce_ptq:.4f} (ppl {math.exp(ce_ptq):.2f})  [{n_lin} target linears]")

    # QR-002a/b/c: freeze all, then unfreeze target linears (+ optional norms / lm_head)
    for p in model.parameters():
        p.requires_grad_(False)
    tparams = [m.weight for m in model.modules() if isinstance(m, PerTensorBitLinear)]
    if args.train_norms:
        tparams += [p for n, p in model.named_parameters() if "norm" in n.lower() and p.dim() == 1]
    if args.train_lm_head:
        tparams += [p for n, p in model.named_parameters() if n.endswith("lm_head.weight")]
    seen, uniq = set(), []
    for p in tparams:
        if id(p) not in seen:
            seen.add(id(p)); p.requires_grad_(True); uniq.append(p)
    tparams = uniq
    arm = ("QR-002a(linears)" + ("+norms" if args.train_norms else "")
           + ("+lmhead" if args.train_lm_head else "") + ("+ansmask" if args.answer_loss_only else "")
           + (f"+basekl{args.kl_weight}" if args.base_kl_replay else ""))
    print(f"adapting {sum(p.numel() for p in tparams)/1e6:.1f}M params across {len(tparams)} tensors [{arm}]")
    print(f"microbatch={args.batch}  grad_accum={args.grad_accum_steps}  "
          f"effective_batch={args.batch * args.grad_accum_steps}")
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(tparams, lr=args.lr)
    else:
        opt = torch.optim.AdamW(tparams, lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    usable = train_ids.numel() - 1
    model.train()
    import time
    t_start = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        train_ce_sum = 0.0
        train_kl_sum = 0.0
        for _ in range(args.grad_accum_steps):
            starts = torch.randint(0, max(1, usable - args.seq_len), (args.batch,), generator=g)
            sl = starts.tolist()
            x = torch.stack([train_ids[s : s + args.seq_len] for s in sl]).to(device)
            if args.answer_loss_only:
                m = torch.stack([train_mask[s : s + args.seq_len] for s in sl]).to(device)
                labels = x.clone()
                labels[~m] = -100
                if not (labels != -100).any():
                    continue  # window(s) landed entirely on prompt/sep tokens; skip (rare)
                loss = model(input_ids=x, labels=labels).loss
            else:
                loss = model(input_ids=x, labels=x).loss
            train_ce_sum += float(loss)
            if teacher is not None:  # FACT-003B: add base-KL replay anchor
                kl = base_kl_replay_term(model, teacher, replay_ids, replay_mask, args.seq_len,
                                         args.replay_batch, args.kl_temp, g, device)
                if kl is not None:
                    train_kl_sum += float(kl)
                    loss = loss + args.kl_weight * kl
            (loss / args.grad_accum_steps).backward()
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            done = step + 1
            elapsed = time.time() - t_start
            rate = elapsed / done                       # sec per optimizer step
            eta = rate * (args.steps - done)
            pct = 100.0 * done / args.steps
            kl_str = f"  kl={train_kl_sum / args.grad_accum_steps:.4f}" if teacher is not None else ""
            print(f"  step {step:4d}/{args.steps} ({pct:5.1f}%)  train_ce={train_ce_sum / args.grad_accum_steps:.4f}{kl_str}  "
                  f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m  ({rate:.1f}s/step)", flush=True)
    if teacher is not None:  # free the frozen base teacher before export
        del teacher
        if device == "cuda":
            torch.cuda.empty_cache()
    ce_adapted = eval_ce(model, eval_ids, args.seq_len, device)
    rec = (ce_ptq - ce_adapted) / max(ce_ptq - ce_fp, 1e-9)
    print(f"QR-002a CE_adapted={ce_adapted:.4f} (ppl {math.exp(ce_adapted):.2f})  "
          f"recovered_fraction={rec:.3f}")

    # save adapted materialized model for QR-003. NOTE: keep eval.txt SHORT — the
    # GGUF llama-perplexity processes every token and 60k tokens is ~13 min/format
    # on a 2-core CPU; a few thousand tokens is plenty for the f16-vs-i2_s parity.
    materialize_and_save(model, out_dir, tok)
    (out_dir / "eval.txt").write_text(tok.decode(eval_ids[: args.ppl_eval_tokens].tolist()),
                                      encoding="utf-8")
    print(f"saved adapted ternary HF dir -> {out_dir}")

    result = {"model": args.model_id, "device": device, "n_target_linears": n_lin,
              "steps": args.steps, "seq_len": args.seq_len, "batch": args.batch, "lr": args.lr,
              "grad_accum_steps": args.grad_accum_steps,
              "effective_batch": args.batch * args.grad_accum_steps,
              "effective_tokens_per_step": args.batch * args.grad_accum_steps * args.seq_len,
              "arm": arm, "train_source": args.train_source, "train_norms": args.train_norms, "train_lm_head": args.train_lm_head,
              "answer_loss_only": args.answer_loss_only,
              "base_kl_replay": args.base_kl_replay, "kl_weight": args.kl_weight, "kl_temp": args.kl_temp,
              "replay_tokens": args.replay_tokens, "replay_batch": args.replay_batch,
              "ce_fp": ce_fp, "ce_ptq": ce_ptq, "ce_adapted": ce_adapted,
              "ppl_fp": math.exp(ce_fp), "ppl_ptq": math.exp(ce_ptq), "ppl_adapted": math.exp(ce_adapted),
              "recovered_fraction": rec}

    # QR-003: export adapted -> f16 & i2_s, perplexity parity (needs bitnet.cpp)
    if args.bitnet:
        bn = args.bitnet.resolve()
        conv = bn / "utils/convert-hf-to-gguf-bitnet.py"
        q = bn / "build/bin/llama-quantize"
        ppx = bn / "build/bin/llama-perplexity"
        f32, f16 = out_dir / "ggml-model-f32.gguf", out_dir / "ggml-model-f16.gguf"
        i2s = out_dir / "ggml-model-i2_s.gguf"

        import time

        def sh(c, label=""):
            print(f">>> [QR-003] {label} START ...", flush=True)
            t0 = time.time()
            r = subprocess.run(c, shell=True, capture_output=True, text=True)
            print(f">>> [QR-003] {label} done rc{r.returncode} in {time.time()-t0:.0f}s", flush=True)
            print((r.stdout + r.stderr)[-300:], flush=True)
            return r

        sh(f'python "{conv}" "{out_dir}" --outtype f32', "convert f32 (writes a multi-GB GGUF)")
        sh(f'python "{conv}" "{out_dir}" --outtype f16', "convert f16")
        sh(f'"{q}" --token-embedding-type f16 --output-tensor-type f16 "{f32}" "{i2s}" I2_S 1 1', "quantize i2_s")
        import re
        pr = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")
        qr3 = {}
        for tag, g_ in [("f16", f16), ("i2_s", i2s)]:
            print(f">>> [QR-003] perplexity {tag} START (CPU, ~per-token) ...", flush=True)
            t0 = time.time()
            out = subprocess.run(f'"{ppx}" -m "{g_}" -f "{out_dir}/eval.txt" -c 64 -t 2',
                                 shell=True, capture_output=True, text=True)
            m = pr.search(out.stdout + out.stderr)
            qr3[tag] = float(m.group(1)) if m else None
            print(f"QR-003 adapted {tag} PPL = {qr3[tag]}  ({time.time()-t0:.0f}s)", flush=True)
        if qr3.get("f16") and qr3.get("i2_s"):
            d = math.log(qr3["i2_s"]) - math.log(qr3["f16"])
            print(f"QR-003 adapted i2_s vs f16 = {d:+.4f} nats")
            qr3["i2s_vs_f16_nats"] = d
        result["qr003"] = qr3

    print("\n" + "=" * 56)
    print(f"RT-116 / TRAIN-001  {args.model_id}")
    print("=" * 56)
    print(f"CE  fp={ce_fp:.4f}  ptq={ce_ptq:.4f}  adapted={ce_adapted:.4f}")
    print(f"PPL fp={math.exp(ce_fp):.1f}  ptq={math.exp(ce_ptq):.1f}  adapted={math.exp(ce_adapted):.1f}")
    print(f"recovered_fraction = (ptq-adapted)/(ptq-fp) = {rec:.3f}")
    if rec > 0.3:
        print("QR-002 VERDICT: PASS — short teacher-free CE recovers a meaningful fraction.")
    elif rec > 0.05:
        print("QR-002 VERDICT: WEAK — some recovery; try +norms/+lm_head, more steps, or LR.")
    else:
        print("QR-002 VERDICT: FAIL — no recovery; revisit recipe (norms/lm_head/LR/corpus).")

    jo = args.json_out or args.work / f"rt116_{slug}_recovery.json"
    jo.parent.mkdir(parents=True, exist_ok=True)
    jo.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {jo}")


if __name__ == "__main__":
    main()
