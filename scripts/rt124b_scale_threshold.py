#!/usr/bin/env python3
"""RT-124B: scale & threshold objective sweep for b1.58 conversion.

Question: is BitNet's native absmean rule the wrong rule for *conversion*? RT-124A
showed the residual is codebook-dominated (ternary deletes ~45% of |W| regardless of
scale granularity). RT-124B attacks the scale objective and the zero-threshold:

  T_ij = sign(W_ij) if |W_ij| > tau else 0 ;  Wq = gamma * T

Sweep (one-shot, PyTorch screen on Llama-160M, per-tensor scale unless noted):
  absmean       : gamma=mean|W|, tau=0.5*gamma  (BitNet-native rule)
  mse_scale     : tau=0.5*gamma, gamma=argmin ||W-gamma*T||^2 given T (closed form)
  thresh_search : grid over tau/gamma in {0.3..0.8}, pick min ||W-gamma*T||^2 (+ mse gamma)
  act_mse       : pick (gamma,tau) minimizing ||XW - X(gamma*T)||^2 on calibration X (diag approx)

For a fixed assignment T, the MSE-optimal scale is gamma* = <W,T>/<T,T> (least squares).
The threshold tau controls how many weights are zeroed (the 45% the codebook deletes).

USAGE:
  python scripts/rt124b_scale_threshold.py --model-id JackFram/llama-160m \
    --json-out reports/rt124b_scale_threshold_160m.json
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


def ternary_T(W, tau):
    """T in {-1,0,+1}: sign(W) where |W|>tau else 0."""
    return torch.sign(W) * (W.abs() > tau).to(W.dtype)


def mse_gamma(W, T):
    """Least-squares scale for a fixed assignment: gamma* = <W,T>/<T,T>."""
    tt = (T * T).sum()
    return (W * T).sum() / tt.clamp_min(1.0) if tt > 0 else W.abs().mean()


def quant_absmean(W):
    g = W.abs().mean().clamp_min(1e-8)
    T = ternary_T(W, 0.5 * g)
    return g * T


def quant_mse_scale(W):
    g0 = W.abs().mean().clamp_min(1e-8)
    T = ternary_T(W, 0.5 * g0)          # BitNet assignment
    g = mse_gamma(W, T)                  # MSE-optimal scale for that T
    return g * T


def quant_thresh_search(W, taus):
    g0 = W.abs().mean().clamp_min(1e-8)
    best, best_err = None, None
    for r in taus:
        T = ternary_T(W, r * g0)
        if T.abs().sum() == 0:
            continue
        g = mse_gamma(W, T)
        err = ((W - g * T) ** 2).sum()
        if best_err is None or err < best_err:
            best_err, best = err, g * T
    return best if best is not None else quant_absmean(W)


def quant_act_mse(W, xvar, taus):
    """(gamma,tau) minimizing sum_j xvar_j * ||W[:,j]-gamma*T[:,j]||^2 (diagonal X^TX approx).
    xvar: per-input-channel activation second moment (len = in_features)."""
    g0 = W.abs().mean().clamp_min(1e-8)
    w = xvar.clamp_min(1e-8).sqrt().unsqueeze(0)  # weight columns by sqrt(activation energy)
    best, best_err = None, None
    for r in taus:
        T = ternary_T(W, r * g0)
        if T.abs().sum() == 0:
            continue
        g = mse_gamma(W * w, T * w) if (T * w).abs().sum() > 0 else mse_gamma(W, T)
        err = ((w * (W - g * T)) ** 2).sum()
        if best_err is None or err < best_err:
            best_err, best = err, g * T
    return best if best is not None else quant_absmean(W)


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
def calib_input_var(model, targets, eval_ids, seq_len, device, n_windows=8):
    """Per-target-linear input second moment E[x^2] per input channel (for act_mse)."""
    store = {}
    hooks = []
    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1])
            s = (x * x).mean(dim=0)
            store[name] = store.get(name, 0) + s
        return hook
    name_by_mod = {m: n for n, m in targets.items()}
    for n, m in targets.items():
        hooks.append(m.register_forward_hook(mk(n)))
    n = min(eval_ids.numel() // seq_len, n_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    for i in range(0, n, 4):
        model(input_ids=ids[i:i + 4])
    for h in hooks:
        h.remove()
    return store


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt124b_scale_threshold.json")
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
    xvar = calib_input_var(model, targets, eval_ids, args.seq_len, device)

    taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    methods = {
        "absmean": lambda W, n: quant_absmean(W),
        "mse_scale": lambda W, n: quant_mse_scale(W),
        "thresh_search": lambda W, n: quant_thresh_search(W, taus),
        "act_mse": lambda W, n: quant_act_mse(W, xvar[n].to(W.device), taus),
    }

    rows = []
    for mname, fn in methods.items():
        nz, tot = 0, 0
        with torch.no_grad():
            for n, m in targets.items():
                Wq = fn(fp[n], n)
                nz += (Wq != 0).sum().item(); tot += Wq.numel()
                m.weight.copy_(Wq)
        ce = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
        with torch.no_grad():
            for n, m in targets.items():
                m.weight.copy_(fp[n])
        rows.append({"method": mname, "ce": round(ce, 5), "ppl": round(math.exp(ce), 3),
                     "nonzero_frac": round(nz / tot, 4)})
        print(f"  {mname:<14} CE={ce:.4f} ppl={math.exp(ce):.2f} nonzero={nz/tot:.3f}")

    base = next(r for r in rows if r["method"] == "absmean")["ce"]
    for r in rows:
        r["delta_ce_vs_absmean"] = round(base - r["ce"], 5)
    best = min(rows, key=lambda r: r["ce"])
    out = {"model": args.model_id, "ce_fp": ce_fp, "ppl_fp": math.exp(ce_fp),
           "taus": taus, "rows": rows, "best": best["method"]}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nbest: {best['method']} CE={best['ce']} (vs absmean {base - best['ce']:+.4f} nats)")
    big = base - best["ce"] > 0.2 and best["method"] != "absmean"
    print("VERDICT:", "OBJECTIVE MATTERS — a non-absmean scale/threshold beats absmean materially -> "
          "the conversion quantizer differs from the native BitNet rule (promote into adaptation)"
          if big else "objective/threshold does NOT rescue much -> assignment/codebook/interaction (RT-125)")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
