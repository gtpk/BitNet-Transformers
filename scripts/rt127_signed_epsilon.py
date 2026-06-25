#!/usr/bin/env python3
"""RT-127: signed-epsilon 2-bit codebook for existing-model conversion.

RT-124/125 showed the one-shot PTQ toolbox cannot rescue PURE {-1,0,+1} ternary (the
codebook is the wall). RT-127 tests whether the ZERO-HEAVY ternary codebook deletes too
many small signed connections, by replacing 0 with a small signed level:

  S_eps in {-1, -eps, +eps, +1}    (4 levels = 2 bits, NO exact zero)
  Wq = gamma * S_eps               eps in {1/8, 1/4, 1/3, 1/2}

This is no longer pure b1.58 — it is a 2-bit compromise (cf. least-squares binary
quantization, 2001.02786). For each eps we MSE-search per-tensor gamma to give the
codebook its fair shot, then measure CE. Compared against the ternary one-shot
baseline ({-1,0,+1} absmean) and FP.

USAGE:
  python scripts/rt127_signed_epsilon.py --model-id JackFram/llama-160m \
    --json-out reports/rt127_signed_epsilon_160m.json
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


def quant_ternary(W):
    g = W.abs().mean().clamp_min(1e-8)
    return g * torch.clamp(torch.round(W / g), -1, 1)


def quant_signed_eps(W, eps, gamma):
    """Nearest to gamma*{-1,-eps,+eps,+1} (no zero)."""
    m = W.abs()
    bnd = (eps + 1.0) / 2.0 * gamma          # magnitude boundary between eps and 1
    mag = torch.where(m > bnd, gamma, eps * gamma)
    return torch.sign(W) * mag               # sign(0)=0 -> maps the (rare) exact-0 to 0


def best_signed_eps(W, eps, cs):
    """MSE-search gamma = c*absmean over cs; return best Wq and gamma."""
    am = W.abs().mean().clamp_min(1e-8)
    best, best_err, best_g = None, None, None
    for c in cs:
        g = c * am
        Wq = quant_signed_eps(W, eps, g)
        err = ((W - Wq) ** 2).sum()
        if best_err is None or err < best_err:
            best_err, best, best_g = err, Wq, g
    return best, float(best_g)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--epsilons", nargs="+", type=float, default=[0.125, 0.25, 0.3333, 0.5])
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt127_signed_epsilon.json")
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

    with torch.no_grad():
        for n, m in targets.items():
            m.weight.copy_(quant_ternary(fp[n]))
    ce_tern = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
    with torch.no_grad():
        for n, m in targets.items():
            m.weight.copy_(fp[n])
    print(f"CE_fp={ce_fp:.4f} (ppl {math.exp(ce_fp):.2f})  CE_ternary={ce_tern:.4f} (ppl {math.exp(ce_tern):.2f})")

    cs = [c / 2 for c in range(1, 17)]  # gamma = (0.5 .. 8.0) * absmean
    rows = [{"codebook": "ternary {-1,0,1}", "eps": None, "ce": round(ce_tern, 5), "ppl": round(math.exp(ce_tern), 3), "bits": 1.58}]
    for eps in args.epsilons:
        with torch.no_grad():
            for n, m in targets.items():
                Wq, _ = best_signed_eps(fp[n], eps, cs)
                m.weight.copy_(Wq)
        ce = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
        with torch.no_grad():
            for n, m in targets.items():
                m.weight.copy_(fp[n])
        rows.append({"codebook": "signed-eps {-1,-e,e,1}", "eps": eps, "ce": round(ce, 5),
                     "ppl": round(math.exp(ce), 3), "bits": 2.0})
        print(f"  signed-eps eps={eps:<6} CE={ce:.4f} ppl={math.exp(ce):.2f}")

    best = min(rows, key=lambda r: r["ce"])
    out = {"model": args.model_id, "ce_fp": ce_fp, "ppl_fp": math.exp(ce_fp),
           "ce_ternary": ce_tern, "rows": rows, "best": best,
           "delta_best_vs_ternary": round(ce_tern - best["ce"], 5)}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    gain = ce_tern - best["ce"]
    print(f"\nbest: {best['codebook']} eps={best.get('eps')} CE={best['ce']} (vs ternary {gain:+.4f} nats)")
    if best["eps"] is not None and gain > 1.0:
        print("VERDICT: SIGNED-EPSILON HELPS A LOT — zero-heavy ternary deletes too many small signed "
              "connections; the 2-bit compromise is warranted (next: runtime path / 2-bit quant type)")
    elif best["eps"] is not None and gain > 0.3:
        print("VERDICT: signed-epsilon helps modestly -> 2-bit codebook gives some headroom; weigh vs runtime cost")
    else:
        print("VERDICT: signed-epsilon does NOT clearly beat ternary -> codebook size is not the main blocker; "
              "data/objective/adaptation dominates")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
