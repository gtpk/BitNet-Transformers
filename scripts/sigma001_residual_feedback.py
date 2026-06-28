#!/usr/bin/env python3
"""SIGMA-001: residual-feedback / noise-shaped ternary projection smoke.

Borrowed idea: delta-sigma / noise shaping. If quantization error cannot be
removed, feed it into the next coordinate/block instead of discarding it locally.

This is a DATA-FREE reference. Final weights are still gamma*T with T in
{-1,0,+1}. If this only improves MSE/CE inside a collapsed regime and FACT stays
0.0, it is not a product lever.

Usage:
  python -X utf8 scripts/sigma001_residual_feedback.py --model-id Felladrin/Llama-160M-Chat-v1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bitnet_llama.conversion import is_target_weight_key  # noqa: E402
from fact004a_160m_smoke import generate, hit, tag  # noqa: E402
from rt116_quality_recovery import eval_ce, load_wikitext  # noqa: E402


def q_i2s(W: torch.Tensor) -> torch.Tensor:
    g = W.abs().mean().clamp_min(1e-8)
    return g * torch.clamp(torch.round(W / g), -1, 1)


def q_sigma_rows(W: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Scan output rows. Residual from row i is nudged into row i+1."""
    g = W.abs().mean().clamp_min(1e-8)
    out = torch.empty_like(W)
    residual = torch.zeros_like(W[0])
    for i in range(W.shape[0]):
        z = W[i] + alpha * residual
        q = g * torch.clamp(torch.round(z / g), -1, 1)
        out[i] = q
        residual = z - q
    return out


def q_sigma_blocks(W: torch.Tensor, group: int = 128, alpha: float = 0.5) -> torch.Tensor:
    """Scan input blocks within each row group. Residual block is passed forward."""
    g = W.abs().mean().clamp_min(1e-8)
    out = torch.empty_like(W)
    out_features, in_features = W.shape
    if in_features % group:
        group = math.gcd(in_features, group) or in_features
    residual = torch.zeros(out_features, group, device=W.device, dtype=W.dtype)
    for start in range(0, in_features, group):
        end = start + group
        z = W[:, start:end] + alpha * residual[:, : end - start]
        q = g * torch.clamp(torch.round(z / g), -1, 1)
        out[:, start:end] = q
        residual[:, : end - start] = z - q
    return out


def quant_for_arm(W: torch.Tensor, arm: str) -> torch.Tensor:
    if arm == "pt":
        return q_i2s(W)
    m = re.match(r"sigma_row_a([0-9.]+)", arm)
    if m:
        return q_sigma_rows(W, float(m.group(1)))
    m = re.match(r"sigma_g(\d+)_a([0-9.]+)", arm)
    if m:
        return q_sigma_blocks(W, int(m.group(1)), float(m.group(2)))
    raise ValueError(arm)


@torch.no_grad()
def apply_arm(model, arm: str):
    mses, ratios = [], []
    for name, p in model.named_parameters():
        if not is_target_weight_key(name):
            continue
        W = p.data
        Wq = quant_for_arm(W, arm)
        mses.append(((W - Wq) ** 2).mean().item())
        ratios.append((Wq.norm(dim=1) / W.norm(dim=1).clamp(min=1e-8)).mean().item())
        p.data = Wq
    return sum(mses) / max(len(mses), 1), sum(ratios) / max(len(ratios), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--arms", default="fp,pt,sigma_row_a0.5,sigma_g128_a0.5,sigma_g128_a1.0")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/sigma001_residual_feedback.json")
    ap.add_argument("--md-out", type=Path, default=REPO_ROOT / "reports/sigma001_residual_feedback.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    _, wt_eval = load_wikitext(tok, 1000, args.eval_tokens)
    panel = [json.loads(l) for l in open(args.panel, encoding="utf-8") if l.strip()]

    rows = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()
        wmse = rnr = 0.0
        if arm != "fp":
            wmse, rnr = apply_arm(model, arm)
        model.config.use_cache = True
        ce = eval_ce(model, wt_eval, 256, device, max_windows=args.ce_windows)
        hits, tags = 0, {}
        for p in panel:
            txt = generate(model, tok, p["prompt"], args.max_new, device)
            hits += int(hit(txt, p["must_contain"]))
            tg = tag(txt)
            tags[tg] = tags.get(tg, 0) + 1
        row = {"arm": arm, "ce": round(ce, 4), "ppl": round(math.exp(ce), 2),
               "fact_rate": round(hits / len(panel), 3), "fact_hits": hits,
               "weight_mse": round(wmse, 6), "row_norm_ratio": round(rnr, 4), "tags": tags}
        rows.append(row)
        print(row, flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    pt = next((r for r in rows if r["arm"] == "pt"), None)
    best_fact = max((r for r in rows if r["arm"] != "fp"), key=lambda r: (r["fact_rate"], -r["ce"]))
    best_ce = min((r for r in rows if r["arm"] != "fp"), key=lambda r: r["ce"])
    verdict = "FAIL"
    if pt and best_fact["fact_rate"] >= pt["fact_rate"] + 0.05:
        verdict = "PASS_FACT"
    elif pt and pt["ce"] - best_ce["ce"] >= 0.5:
        verdict = "PARTIAL_CE_ONLY"

    out = {"model": args.model_id, "rows": rows, "best_fact": best_fact,
           "best_ce": best_ce, "verdict": verdict}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = ["# SIGMA-001 residual-feedback ternary projection", "",
             f"model={args.model_id}  eval=WikiText {args.eval_tokens} tok  FACT panel={len(panel)}", "",
             "| arm | CE | ppl | fact_rate | weight_MSE | row_norm_ratio | tags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['ce']} | {r['ppl']} | {r['fact_rate']} | "
                     f"{r['weight_mse']} | {r['row_norm_ratio']} | {r['tags']} |")
    lines += ["", f"best_fact={best_fact['arm']}  best_ce={best_ce['arm']}",
              "", f"VERDICT: {verdict}",
              "",
              "PASS requires FACT movement, not only CE movement inside a collapsed regime."]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
