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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--out-dir", type=Path, default=None, help="adapted HF dir (default <work>/<slug>_adapted)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-tokens", type=int, default=600_000)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--ppl-eval-tokens", type=int, default=3_000,
                    help="tokens written to eval.txt for the GGUF llama-perplexity (QR-003); "
                         "keep small, CPU perplexity is ~per-token")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bitnet", type=Path, default=None, help="if set, run QR-003 export+perplexity")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    slug = args.model_id.split("/")[-1]
    out_dir = args.out_dir or args.work / f"{slug}_adapted"
    print(f"device={device}  model={args.model_id}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device)

    train_ids, eval_ids = load_wikitext(tok, args.max_train_tokens, args.max_eval_tokens)
    print(f"train tokens={train_ids.numel():,}  eval tokens={eval_ids.numel():,}")

    # QR-001: FP baseline, then PTQ collapse
    ce_fp = eval_ce(model, eval_ids, args.seq_len, device)
    n_lin = replace_targets(model)
    model.to(device)
    ce_ptq = eval_ce(model, eval_ids, args.seq_len, device)
    print(f"QR-001  CE_fp={ce_fp:.4f} (ppl {math.exp(ce_fp):.2f})  "
          f"CE_ptq={ce_ptq:.4f} (ppl {math.exp(ce_ptq):.2f})  [{n_lin} target linears]")

    # QR-002a: freeze all but target linears, CE-only adaptation
    for p in model.parameters():
        p.requires_grad_(False)
    tparams = [m.weight for m in model.modules() if isinstance(m, PerTensorBitLinear)]
    for p in tparams:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(tparams, lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    usable = train_ids.numel() - 1
    model.train()
    for step in range(args.steps):
        starts = torch.randint(0, max(1, usable - args.seq_len), (args.batch,), generator=g)
        x = torch.stack([train_ids[s : s + args.seq_len] for s in starts.tolist()]).to(device)
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"  step {step:4d}  train_ce={float(loss):.4f}")
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

        def sh(c):
            print("$", c)
            r = subprocess.run(c, shell=True, capture_output=True, text=True)
            print((r.stdout + r.stderr)[-300:])
            return r

        sh(f'python "{conv}" "{out_dir}" --outtype f32')
        sh(f'python "{conv}" "{out_dir}" --outtype f16')
        sh(f'"{q}" --token-embedding-type f16 --output-tensor-type f16 "{f32}" "{i2s}" I2_S 1 1')
        import re
        pr = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")
        qr3 = {}
        for tag, g_ in [("f16", f16), ("i2_s", i2s)]:
            out = subprocess.run(f'"{ppx}" -m "{g_}" -f "{out_dir}/eval.txt" -c 64 -t 2',
                                 shell=True, capture_output=True, text=True)
            m = pr.search(out.stdout + out.stderr)
            qr3[tag] = float(m.group(1)) if m else None
            print(f"QR-003 adapted {tag} PPL = {qr3[tag]}")
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
