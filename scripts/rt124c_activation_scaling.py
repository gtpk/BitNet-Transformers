#!/usr/bin/env python3
"""RT-124C: AWQ/SmoothQuant-style activation-aware diagonal scaling for b1.58.

Question: can an equivalent diagonal scaling make W easier to ternarize WITHOUT adding
inference matmuls? Identity: XW = (XD)(D^-1 W). Choose per-input-channel D from
calibration activation stats; ternarize D^-1 W; in deployment D folds upstream
(SmoothQuant) so the stored weight stays ternary and XD is free.

Screen (PyTorch, Llama-160M, one-shot, per-tensor gamma on the scaled weight):
  D_j = act_rms_j ** alpha , alpha in {0, 0.25, 0.5, 0.75, 1.0}   (alpha=0 -> baseline)
  Ws = W / D (per input channel j) ; gamma = mean|Ws| ; Ts = sign(Ws)*(|Ws|>0.5*gamma)
  W_eff = (gamma*Ts) * D   (equivalent weight; CE identical to the folded deployment)

alpha=0 reproduces RT-124A per-tensor absmean. Refs: AWQ (2306.00978),
SmoothQuant (2211.10438).

USAGE:
  python scripts/rt124c_activation_scaling.py --model-id JackFram/llama-160m \
    --json-out reports/rt124c_activation_scaling_160m.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402


def quant_awq(W, d):
    """d: per-input-channel scale (len in_features). Wq_eff = (gamma*T(W/d)) * d."""
    Ws = W / d.unsqueeze(0).clamp_min(1e-8)
    g = Ws.abs().mean().clamp_min(1e-8)
    T = torch.sign(Ws) * (Ws.abs() > 0.5 * g).to(Ws.dtype)
    return (g * T) * d.unsqueeze(0)


@torch.no_grad()
def eval_ce(model, eval_ids, seq_len, device, max_windows=64):
    model.eval()
    n = min(eval_ids.numel() // seq_len, max_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    tot, cnt = 0.0, 0
    for i in range(0, n, 8):
        b = ids[i:i + 8]
        tot += float(model(input_ids=b, labels=b).loss) * b.shape[0]
        cnt += b.shape[0]
    return tot / max(cnt, 1)


@torch.no_grad()
def calib_act_rms(model, targets, eval_ids, seq_len, device, n_windows=8):
    store = {}
    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            store[name] = store.get(name, 0) + (x * x).mean(dim=0)
        return hook
    hooks = [m.register_forward_hook(mk(n)) for n, m in targets.items()]
    n = min(eval_ids.numel() // seq_len, n_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    for i in range(0, n, 4):
        model(input_ids=ids[i:i + 4])
    for h in hooks:
        h.remove()
    return {k: v.sqrt() for k, v in store.items()}  # rms per input channel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt124c_activation_scaling.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()

    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    eval_ids = torch.tensor(tok(text)["input_ids"][: args.max_eval_tokens], dtype=torch.long)

    targets = {n: m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{n}.weight")}
    fp = {n: m.weight.detach().clone() for n, m in targets.items()}
    print(f"{len(targets)} target linears")
    ce_fp = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
    print(f"CE_fp={ce_fp:.4f} (ppl {math.exp(ce_fp):.2f})")
    rms = calib_act_rms(model, targets, eval_ids, args.seq_len, device)

    rows = []
    for alpha in args.alphas:
        with torch.no_grad():
            for n, m in targets.items():
                d = rms[n].to(fp[n].device).clamp_min(1e-6) ** alpha if alpha != 0 else torch.ones(fp[n].shape[1], device=fp[n].device)
                m.weight.copy_(quant_awq(fp[n], d))
        ce = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
        with torch.no_grad():
            for n, m in targets.items():
                m.weight.copy_(fp[n])
        rows.append({"alpha": alpha, "ce": round(ce, 5), "ppl": round(math.exp(ce), 3)})
        print(f"  alpha={alpha:<4} CE={ce:.4f} ppl={math.exp(ce):.2f}")

    base = next(r for r in rows if r["alpha"] == 0.0)["ce"]
    for r in rows:
        r["delta_ce_vs_alpha0"] = round(base - r["ce"], 5)
    best = min(rows, key=lambda r: r["ce"])
    out = {"model": args.model_id, "ce_fp": ce_fp, "ppl_fp": math.exp(ce_fp),
           "rows": rows, "best_alpha": best["alpha"],
           "note": "alpha=0 == per-tensor absmean baseline. D folds upstream (SmoothQuant) so deployable."}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nbest alpha={best['alpha']} CE={best['ce']} (vs alpha0 {base - best['ce']:+.4f} nats)")
    big = base - best["ce"] > 0.2 and best["alpha"] != 0.0
    print("VERDICT:", "ACTIVATION-AWARE SCALING HELPS — activation distribution mismatch is a cause; "
          "combine with RT-124B best + CE adaptation (foldable via SmoothQuant)" if big else
          "activation-aware diagonal scaling does NOT rescue much -> assignment/Hessian (RT-125)")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
