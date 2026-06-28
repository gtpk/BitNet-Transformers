#!/usr/bin/env python3
"""RHT-002: dithered / randomized Hadamard reference smoke for I2_S weights.

Fixed H-I2S failed. This checks whether the failure was merely the fixed
Hadamard choice by trying random signs, repeated RHT, and small dither before
I2_S ternary projection.

This is a quality/reference script only. No kernel, no speed claim.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bitnet_llama.conversion import is_target_weight_key  # noqa: E402
from fact004a_160m_smoke import generate, hit, tag  # noqa: E402
from rt116_quality_recovery import eval_ce, load_wikitext  # noqa: E402


def q_i2s(W):
    g = W.abs().mean().clamp_min(1e-8)
    return g * torch.clamp(torch.round(W / g), -1, 1)


def sylvester(k: int):
    H = torch.ones(1, 1)
    while H.shape[0] < k:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H[:k, :k]


def block_hadamard(n: int, tile: int, device, dtype):
    if n % tile:
        tile = math.gcd(n, tile) or n
    Ht = sylvester(tile) / math.sqrt(tile)
    H = torch.zeros(n, n)
    for i in range(0, n, tile):
        H[i:i + tile, i:i + tile] = Ht
    return H.to(device=device, dtype=dtype)


class RHTLinear(nn.Module):
    def __init__(self, lin: nn.Linear, arm: str, tile: int, seed: int, dither: float):
        super().__init__()
        W = lin.weight.detach()
        out, inp = W.shape
        gen = torch.Generator(device=W.device).manual_seed(seed + out + inp)
        H = block_hadamard(inp, tile, W.device, W.dtype)

        def signed_h():
            signs = torch.randint(0, 2, (inp,), generator=gen, device=W.device, dtype=torch.int64)
            signs = signs.to(W.dtype).mul_(2).sub_(1)
            return signs[:, None] * H

        transforms = []
        if arm == "h1":
            transforms = [H]
        elif arm == "rht1":
            transforms = [signed_h()]
        elif arm == "rht2":
            transforms = [signed_h(), signed_h()]
        elif arm == "rht1_dither":
            transforms = [signed_h()]
        else:
            raise ValueError(arm)

        R = torch.eye(inp, device=W.device, dtype=W.dtype)
        for T in transforms:
            R = T @ R
        Wrot = W @ R.t()
        if arm == "rht1_dither" and dither > 0:
            gamma = Wrot.abs().mean().clamp_min(1e-8)
            Wrot = Wrot + (torch.rand(Wrot.shape, generator=gen, device=W.device, dtype=W.dtype) - 0.5) * dither * gamma
        self.register_buffer("R", R)
        self.register_buffer("Wq", q_i2s(Wrot))
        self.bias = lin.bias

    def forward(self, x):
        import torch.nn.functional as F
        return F.linear(x @ self.R.t(), self.Wq, self.bias)


def set_submodule(root, dotted, mod):
    parent, _, child = dotted.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, mod)


@torch.no_grad()
def apply_arm(model, arm: str, tile: int, seed: int, dither: float):
    mses, ratios = [], []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not is_target_weight_key(name + ".weight"):
            continue
        if arm == "pt":
            W = mod.weight.detach()
            Wq = q_i2s(W)
            mses.append(((W - Wq) ** 2).mean().item())
            ratios.append((Wq.norm(dim=1) / W.norm(dim=1).clamp(min=1e-8)).mean().item())
            mod.weight.data = Wq
        else:
            wrapped = RHTLinear(mod, arm, tile, seed, dither)
            W = mod.weight.detach()
            Weff = wrapped.Wq @ wrapped.R
            mses.append(((W - Weff) ** 2).mean().item())
            ratios.append((Weff.norm(dim=1) / W.norm(dim=1).clamp(min=1e-8)).mean().item())
            set_submodule(model, name, wrapped)
    return sum(mses) / max(len(mses), 1), sum(ratios) / max(len(ratios), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--arms", default="fp,pt,h1,rht1,rht2,rht1_dither")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--dither", type=float, default=0.25)
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rht002_dithered_reference.json")
    ap.add_argument("--md-out", type=Path, default=REPO_ROOT / "reports/rht002_dithered_reference.md")
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
            wmse, rnr = apply_arm(model, arm, args.tile, args.seed, args.dither)
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

    lines = ["# RHT-002 dithered/randomized Hadamard reference", "",
             f"model={args.model_id}  tile={args.tile} seed={args.seed} dither={args.dither}", "",
             "| arm | CE | ppl | fact_rate | weight_MSE | row_norm_ratio | tags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['ce']} | {r['ppl']} | {r['fact_rate']} | "
                     f"{r['weight_mse']} | {r['row_norm_ratio']} | {r['tags']} |")
    lines += ["", f"best_fact={best_fact['arm']}  best_ce={best_ce['arm']}",
              "", f"VERDICT: {verdict}",
              "",
              "PASS requires FACT movement. CE-only movement inside a collapsed regime is partial at best."]
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()
