#!/usr/bin/env python3
"""PT2-I2S-001: improved ternary factorization (ITF) -- a no-train INITIALIZATION smoke.

PT2-LLM-style asymmetric ternary (W ~= mu + alpha*T) chooses a better ternary code T than absmean.
Question: does a better T help downstream BEHAVIOUR (FACT / gold_rank), or only weight reconstruction?
And does the gain SURVIVE projecting back to a pure one-plane I2_S (gamma*T, no mu)?

No training. For each per-tensor arm we overwrite the target linears with the quantised weight and
score the model as-is:
  fp        : original FP weights (reference ceiling)
  absmean   : current I2_S -- gamma=mean|W|, T=clamp(round(W/gamma)), Wq=gamma*T
  itf_pure  : TWN-style optimal ternary with a mu-aware T, RE-PROJECTED to pure I2_S (gamma_c*T, NO mu)
  itf_asym  : asymmetric PT2-lite UPPER BOUND -- Wq=mu + alpha*T (a dense offset, NOT pure I2_S)

VERDICT GATE (user): reconstruction-only wins are discarded. itf_pure must beat absmean on FACT
eval_panel and/or gold_rank (not just weight_MSE) to be worth applying to TinyLlama. itf_asym shows
the headroom a small mu correction would buy.

USAGE (local PC: Mac/MPS or 3080):
  python -X utf8 scripts/pt2_i2s_001_itf.py --model-id Felladrin/Llama-160M-Chat-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bitnet_llama import conversion as C  # noqa: E402
from rt116_quality_recovery import load_wikitext, eval_ce, telemetry_probe  # noqa: E402
from fact004a_160m_smoke import generate, hit, tag  # noqa: E402


def q_absmean(W):
    """E0 -- current I2_S: per-tensor gamma=mean|W|, T=clamp(round(W/gamma),-1,1)."""
    g = W.abs().mean().clamp_min(1e-8)
    T = torch.clamp(torch.round(W / g), -1, 1)
    return g * T, T, {"gamma": float(g)}


def _itf_row(W, iters=10):
    """PT2-I2S-001 ITF ROW GRID: per-row (per output channel) iterative ternary fit.
    For each row: init mu=mean, alpha=mean|W-mu|; repeat -- T=clamp(round((W-mu)/alpha)); then solve
    (alpha,mu) by least squares (linear regression of W on T with intercept). Returns (T, mu_r, alpha_r)
    with mu_r, alpha_r shaped [out,1]."""
    Wm = W.mean(dim=1, keepdim=True)
    alpha = (W - Wm).abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    mu = Wm.clone()
    T = torch.clamp(torch.round((W - mu) / alpha), -1, 1)
    for _ in range(iters):
        Tm = T.mean(dim=1, keepdim=True)
        denom = ((T - Tm) ** 2).sum(dim=1, keepdim=True).clamp_min(1e-8)
        alpha = ((W - W.mean(dim=1, keepdim=True)) * (T - Tm)).sum(dim=1, keepdim=True) / denom
        alpha = torch.where(alpha.abs() < 1e-6, torch.full_like(alpha, 1e-6), alpha)
        mu = W.mean(dim=1, keepdim=True) - alpha * Tm
        T = torch.clamp(torch.round((W - mu) / alpha), -1, 1)
    return T, mu, alpha


def q_itf_asym(W):
    """E1 -- exact PT2 row grid (UPPER BOUND, class A+/B): Wq = mu_r + alpha_r*T (per-row, NOT pure I2_S)."""
    T, mu, alpha = _itf_row(W)
    return mu + alpha * T, T, {"per_row": True}


def q_itf_pure(W):
    """E2 -- the row-fit T re-projected to PURE per-tensor I2_S (deployable): gamma_proj=<W,T>/<T,T>
    (single scalar), Wq=gamma_proj*T. Tests whether the better T survives one-plane I2_S projection."""
    T, _mu, _alpha = _itf_row(W)
    gamma_proj = ((W * T).sum() / (T * T).sum().clamp_min(1e-8))
    return gamma_proj * T, T, {"gamma_proj": float(gamma_proj)}


ARMS = {"absmean": q_absmean, "itf_pure": q_itf_pure, "itf_asym": q_itf_asym}


def quantize_inplace(model, arm_fn):
    """Overwrite target linears with the arm's quantised weight; return mean relative weight MSE."""
    mses, n = 0.0, 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            W = m.weight.data.float()
            Wq, _T, _meta = arm_fn(W)
            mse = ((W - Wq).pow(2).sum() / W.pow(2).sum().clamp_min(1e-8)).item()
            mses += mse
            n += 1
            m.weight.data.copy_(Wq.to(m.weight.dtype))
    return mses / max(n, 1), n


def load_jsonl(p, n=0):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows


def score_model(model, tok, panel, eval_ids, device, max_new, ce_windows):
    model.eval()
    model.config.use_cache = True
    hits, tags = 0, {}
    for p in panel:
        t = tag_txt = generate(model, tok, p["prompt"], max_new, device)
        h = hit(tag_txt, p["must_contain"]); hits += int(h)
        tg = tag(tag_txt); tags[tg] = tags.get(tg, 0) + 1
    probe = telemetry_probe(model, tok, panel, device, max_new=max_new)
    ce = eval_ce(model, eval_ids, 256, device, max_windows=ce_windows)
    return {"fact_rate": round(hits / len(panel), 3), "ce": round(ce, 3), "tags": tags,
            "gold_rank_mean": probe["gold_rank_mean"], "degenerate_rate": probe["degenerate_rate"],
            "logit_entropy": probe["logit_entropy"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "pt2_i2s_001")
    ap.add_argument("--arms", default="fp,absmean,itf_pure,itf_asym")
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  model={args.model_id}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    panel = load_jsonl(args.panel)
    _, eval_ids = load_wikitext(tok, 1000, args.eval_tokens)

    rows = []
    for arm in [a for a in args.arms.split(",") if a]:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device)
        wmse, ntgt = (0.0, 0) if arm == "fp" else quantize_inplace(model, ARMS[arm])
        s = score_model(model, tok, panel, eval_ids, device, args.max_new, args.ce_windows)
        row = {"arm": arm, "weight_mse": round(wmse, 5), "n_targets": ntgt, **s}
        rows.append(row)
        print(f"  {arm:9s}: wMSE {row['weight_mse']:.4f} | CE {s['ce']:.3f} | FACT {s['fact_rate']} | "
              f"gold_rank {s['gold_rank_mean']} | degen {s['degenerate_rate']} | {s['tags']}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    d = {r["arm"]: r for r in rows}
    lines = ["# PT2-I2S-001: improved ternary factorization (no-train init smoke)", "",
             f"model={args.model_id}  device={device}", "",
             "| arm | weight_MSE | CE | FACT eval_panel | gold_rank | degenerate | tags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['weight_mse']} | {r['ce']} | {r['fact_rate']} | "
                     f"{r['gold_rank_mean']} | {r['degenerate_rate']} | {r['tags']} |")
    # verdict: itf_pure must beat absmean on BEHAVIOUR (FACT or gold_rank), not just weight_MSE
    verdict = "INCONCLUSIVE"
    if "absmean" in d and "itf_pure" in d:
        a, c = d["absmean"], d["itf_pure"]
        wmse_better = c["weight_mse"] < a["weight_mse"]
        fact_better = c["fact_rate"] > a["fact_rate"] + 0.03
        rank_better = c["gold_rank_mean"] < a["gold_rank_mean"] * 0.8  # >=20% lower rank
        asym_gain = ("itf_asym" in d and
                     (d["itf_asym"]["fact_rate"] > a["fact_rate"] + 0.05 or
                      d["itf_asym"]["gold_rank_mean"] < a["gold_rank_mean"] * 0.8))
        if fact_better or rank_better:
            verdict = (f"itf_pure BEATS absmean on behaviour (FACT {a['fact_rate']}->{c['fact_rate']}, "
                       f"gold_rank {a['gold_rank_mean']}->{c['gold_rank_mean']}; wMSE {a['weight_mse']}->{c['weight_mse']}). "
                       f"Better T survives projection to pure I2_S -> worth applying to TinyLlama (PT2-I2S init).")
        elif wmse_better:
            verdict = (f"RECONSTRUCTION-ONLY: itf_pure lowers weight_MSE ({a['weight_mse']}->{c['weight_mse']}) "
                       f"but NOT FACT/gold_rank -> DISCARD per gate. "
                       + ("itf_asym (mu) DOES move behaviour -> PT2-I2S-003 mu-correction worth a look."
                          if asym_gain else "itf_asym mu also flat -> ternary-init is not the lever here."))
        else:
            verdict = "itf_pure does not improve weight_MSE or behaviour vs absmean -> ITF not useful at this scale."
    lines += ["", "VERDICT: " + verdict]
    (args.work / "summary.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-2:]))
    print(f"wrote {args.work / 'summary.md'}")


if __name__ == "__main__":
    main()
