#!/usr/bin/env python3
"""RT-WSYNC-001: 160M weight-only b1.58 baseline table (data-free, no training).

Per docs/weight_only_sync_plan.md. Question: how much does b1.58 conversion damage CE/FACT, and do
weight-only transforms (row-scale, group-scale, row-norm correction) close the gap WITHOUT any
calibration/adaptation data? This is a possible INITIALIZATION/preprocessing stage before the PopQA
blend (FACT-003H), not a replacement for it. The honest limitation (plan Core Math): with no data we
assume Sigma_x ~= I, so we can only make weights geometrically closer, not activation-aware.

Arms (each = replace EVERY target linear's weight with the arm's quantized weight, then eval):
  fp        full precision (ceiling)
  pt        per-tensor b1.58            Wq = g*T, g=mean|W|, T=clamp(round(W/g),-1,1)   [the I2_S baseline]
  row       per-output-row b1.58        g_i=mean|W_i| per row (NOT pure I2_S -> diagnostic/hybrid)
  group     groupwise-input b1.58       g per (row, input-group of size G)
  row_norm  per-tensor + row-norm fix   Wq_pt then c_i=||W_i||/||Wq_i|| rescale per row (WSYNC-005)

Metrics: CE/PPL on WikiText, FACT panel fact_rate (PyTorch, rep-penalty 1.2), mean weight MSE over
target linears, mean row-norm ratio ||Wq_i||/||W_i||.

Pass (plan RT-WSYNC-001): any weight-only transform improves CE by >=0.5 nats OR FACT by >=0.05 vs
per-tensor. Claim discipline: a positive result = "weight-only preproc improves the b1.58 STARTING
POINT for later representative-data adaptation", NOT "weight-only solves factual recovery".

USAGE (3080 box, no training): python -X utf8 scripts/rt_wsync_weight_only.py --arms fp,pt,row,group,row_norm
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bitnet_llama.conversion import is_target_weight_key  # noqa: E402
from rt116_quality_recovery import load_wikitext, eval_ce  # noqa: E402
from fact004a_160m_smoke import generate, hit, tag  # noqa: E402


def q_pt(W):
    g = W.abs().mean().clamp(min=1e-8)
    return g * torch.clamp(torch.round(W / g), -1, 1)


def q_row(W):
    g = W.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return g * torch.clamp(torch.round(W / g), -1, 1)


def q_group(W, G=128):
    out, inp = W.shape
    if inp % G:
        G = math.gcd(inp, G) or inp  # fall back to a divisor so the reshape is exact
    Wg = W.reshape(out, inp // G, G)
    g = Wg.abs().mean(dim=2, keepdim=True).clamp(min=1e-8)
    return (g * torch.clamp(torch.round(Wg / g), -1, 1)).reshape(out, inp)


def q_row_norm(W):
    Wq = q_pt(W)
    c = W.norm(dim=1, keepdim=True) / Wq.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return c * Wq


QUANT = {"pt": q_pt, "row": q_row, "group": q_group, "row_norm": q_row_norm}


@torch.no_grad()
def apply_arm(model, arm):
    """Replace every target-linear weight in place with the arm's quantized weight.
    Returns (mean weight MSE, mean row-norm ratio) over the target linears."""
    mses, ratios = [], []
    for name, p in model.named_parameters():
        if not is_target_weight_key(name):
            continue
        W = p.data
        Wq = QUANT[arm](W)
        mses.append(((W - Wq) ** 2).mean().item())
        ratios.append((Wq.norm(dim=1) / W.norm(dim=1).clamp(min=1e-8)).mean().item())
        p.data = Wq
    return sum(mses) / max(len(mses), 1), sum(ratios) / max(len(ratios), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--arms", default="fp,pt,row,group,row_norm")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--eval-tokens", type=int, default=60000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt_wsync_160m.json")
    ap.add_argument("--md-out", type=Path, default=REPO_ROOT / "reports/rt_wsync_160m.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    _, wt = load_wikitext(tok, 1000, args.eval_tokens)
    panel = [json.loads(l) for l in open(args.panel, encoding="utf-8") if l.strip()]
    print(f"device={device} model={args.model_id} arms={args.arms} panel={len(panel)}", flush=True)

    rows = {}
    for arm in args.arms.split(","):
        m = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()
        wmse = rnr = 0.0
        if arm != "fp":
            wmse, rnr = apply_arm(m, arm)
        m.config.use_cache = True
        ce = eval_ce(m, wt, 256, device, max_windows=args.ce_windows)
        hits, tags = 0, {}
        for p in panel:
            txt = generate(m, tok, p["prompt"], args.max_new, device)
            hits += int(hit(txt, p["must_contain"]))
            t = tag(txt); tags[t] = tags.get(t, 0) + 1
        fr = round(hits / len(panel), 3)
        rows[arm] = {"ce": round(ce, 4), "ppl": round(math.exp(ce), 2), "fact_rate": fr,
                     "fact_hits": hits, "n": len(panel), "weight_mse": round(wmse, 6),
                     "row_norm_ratio": round(rnr, 4), "tags": tags}
        print(f"{arm}: CE {ce:.4f} (ppl {math.exp(ce):.1f}) | fact {hits}/{len(panel)} ({fr}) | "
              f"wMSE {wmse:.5f} rnr {rnr:.3f} | {tags}", flush=True)
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    lines = ["# RT-WSYNC-001 160M weight-only b1.58 table (data-free, no training)", "",
             f"model={args.model_id}  eval=WikiText {args.eval_tokens} tok  FACT panel {len(panel)} (rep-penalty 1.2)",
             "Sigma_x~=I (no calibration data): weight-geometry only.", "",
             "| arm | CE | ppl | fact_rate | weight_MSE | row_norm_ratio | tags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm in args.arms.split(","):
        r = rows[arm]
        lines.append(f"| {arm} | {r['ce']} | {r['ppl']} | {r['fact_rate']} | {r['weight_mse']} | "
                     f"{r['row_norm_ratio']} | {r['tags']} |")
    base = rows.get("pt")
    if base:
        cands = [a for a in rows if a not in ("fp", "pt")]
        if cands:
            best = min(cands, key=lambda a: rows[a]["ce"])
            d_ce = base["ce"] - rows[best]["ce"]
            d_fact = rows[best]["fact_rate"] - base["fact_rate"]
            passed = d_ce >= 0.5 or d_fact >= 0.05
            lines += ["", f"best non-fp/pt by CE: {best} (CE {rows[best]['ce']} vs pt {base['ce']}, "
                      f"dCE {d_ce:+.3f}; FACT {rows[best]['fact_rate']} vs pt {base['fact_rate']}, dFACT {d_fact:+.3f})",
                      "", "VERDICT: " + (
                          "PASS -- a weight-only transform clears the bar (>=0.5 nats CE or >=0.05 FACT vs "
                          "per-tensor); WSYNC may be a useful b1.58 INIT before PopQA blend. Claim only a "
                          "better STARTING POINT, NOT 'solves factual recovery'."
                          if passed else
                          "FAIL -- no weight-only transform beats per-tensor by >=0.5 nats CE or >=0.05 FACT. "
                          "Sigma_x~=I is too weak; the lever is representative data (PopQA blend) / capacity, "
                          "not data-free weight sync (plan decision S2/S4).")]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-3:]))
    print(f"wrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
