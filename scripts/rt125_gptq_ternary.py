#!/usr/bin/env python3
"""RT-125: GPTQ/Hessian-style ternary projection for b1.58 conversion.

RT-124 showed scale/objective/activation-diagonal don't rescue one-shot ternary; the
remaining lever is the ASSIGNMENT. Nearest rounding optimizes weight stats:
  min ||W - gamma*T||^2
GPTQ optimizes the layer OUTPUT with second-order error compensation:
  min ||XW - X(gamma*T)||^2 ~ (W-gamma*T)^T H (W-gamma*T),  H = X^T X
quantizing input-columns in order and propagating each column's error to the
remaining columns via the inverse Cholesky of H. Ref: GPTQ (2210.17323).

Screening version (per the plan): accumulate H per target linear from ONE FP pass
(hooks), GPTQ each layer independently (non-sequential), per-tensor gamma=absmean,
ternary levels {-gamma,0,+gamma}. Compare nearest (= RT-124A per-tensor) vs GPTQ.

USAGE:
  python scripts/rt125_gptq_ternary.py --model-id JackFram/llama-160m \
    --json-out reports/rt125_gptq_ternary_160m.json
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


def quant_ternary(w, gamma):
    return gamma * torch.clamp(torch.round(w / gamma), -1, 1)


def gptq_ternary(W, H, damp=0.01):
    """GPTQ ternary projection. W: [out,in] (modified copy returned), H: [in,in]."""
    W = W.clone().float()
    H = H.clone().float()
    out, inn = W.shape
    gamma = W.abs().mean().clamp_min(1e-8)
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0
    d = damp * torch.mean(torch.diag(H)).clamp_min(1e-8)
    H[range(inn), range(inn)] += d
    # inverse Cholesky (upper) for error propagation
    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        U = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        # fallback: diagonal-only (no cross-column propagation)
        U = torch.diag(1.0 / torch.sqrt(torch.diag(H)))
    Q = torch.zeros_like(W)
    for j in range(inn):
        w = W[:, j]
        djj = U[j, j].clamp_min(1e-12)
        q = quant_ternary(w, gamma)
        Q[:, j] = q
        err = (w - q) / djj
        if j + 1 < inn:
            W[:, j + 1:] -= err.unsqueeze(1) * U[j, j + 1:].unsqueeze(0)
    return Q


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
def accumulate_H(model, targets, eval_ids, seq_len, device, n_windows=16):
    H = {n: torch.zeros(m.in_features, m.in_features, device=device) for n, m in targets.items()}
    cnt = {n: 0 for n in targets}
    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            H[name] += x.t() @ x
            cnt[name] += x.shape[0]
        return hook
    hooks = [m.register_forward_hook(mk(n)) for n, m in targets.items()]
    n = min(eval_ids.numel() // seq_len, n_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    for i in range(0, n, 4):
        model(input_ids=ids[i:i + 4])
    for h in hooks:
        h.remove()
    for nme in H:
        H[nme] /= max(cnt[nme], 1)
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--calib-windows", type=int, default=16)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt125_gptq_ternary.json")
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

    # nearest baseline (per-tensor absmean) for in-run contrast
    with torch.no_grad():
        for n, m in targets.items():
            g = fp[n].abs().mean().clamp_min(1e-8)
            m.weight.copy_(quant_ternary(fp[n], g))
    ce_nearest = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
    with torch.no_grad():
        for n, m in targets.items():
            m.weight.copy_(fp[n])
    print(f"CE_nearest(per-tensor absmean)={ce_nearest:.4f} (ppl {math.exp(ce_nearest):.2f})")

    print("accumulating H (calibration)...")
    H = accumulate_H(model, targets, eval_ids, args.seq_len, device, args.calib_windows)

    print("GPTQ-ternary per layer...")
    with torch.no_grad():
        for n, m in targets.items():
            Q = gptq_ternary(fp[n], H[n], args.damp)
            m.weight.copy_(Q.to(m.weight.dtype))
    ce_gptq = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
    print(f"CE_GPTQ={ce_gptq:.4f} (ppl {math.exp(ce_gptq):.2f})")

    gap_total = ce_nearest - ce_fp
    closed = (ce_nearest - ce_gptq) / gap_total if gap_total > 0 else 0.0
    out = {"model": args.model_id, "ce_fp": ce_fp, "ppl_fp": math.exp(ce_fp),
           "ce_nearest": ce_nearest, "ppl_nearest": math.exp(ce_nearest),
           "ce_gptq": ce_gptq, "ppl_gptq": math.exp(ce_gptq),
           "delta_ce_gptq_vs_nearest": round(ce_nearest - ce_gptq, 5),
           "frac_of_nearest_to_fp_gap_closed": round(closed, 4),
           "note": "non-sequential screening GPTQ (H from one FP pass), per-tensor gamma, ternary."}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nGPTQ vs nearest: {ce_nearest - ce_gptq:+.4f} nats; closed {closed*100:.1f}% of the nearest->FP gap")
    if ce_nearest - ce_gptq > 0.5:
        print("VERDICT: ASSIGNMENT MATTERS — GPTQ output-aware ternary beats nearest materially -> "
              "promote (combine with CE adaptation + export)")
    elif ce_nearest - ce_gptq > 0.1:
        print("VERDICT: GPTQ helps modestly -> worth combining with adaptation, but pure ternary still hard")
    else:
        print("VERDICT: GPTQ barely helps -> pure ternary codebook likely too small (RT-127 signed-epsilon)")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
