#!/usr/bin/env python3
"""DINO-I2S-001b: robustness of the dino_logit positive (160M) before the 1.1B gate.

DINO-I2S-001 (seed 41, lambda 0.2) found dino_logit eval_panel 0.074->0.222 (+0.148) while
dino_hidden cancelled the gain. 0.222 = 6/27 on a small panel, so before paying for the 1.1B
Colab gate we confirm the logit signal is not seed/lambda luck.

Design (hidden DISCARDED -- DINO-I2S-001 showed hidden alignment hurts):
  for seed in 41,42,43:
    baseline  (content-KL 0.2 replay only)                 -> eval_panel b_s
    for lambda in 0.1,0.2,0.4:
      dino_logit (baseline + lambda * unlabeled-view content-KL) -> eval_panel d_{s,lambda}
  dEval_{s,lambda} = d_{s,lambda} - b_s

Verdict (user gate): if some lambda clears baseline +0.05 on >=2/3 seeds -> escalate to
DINO-I2S-002 1.1B Colab gate. Else the 160M positive was noise -> reconsider.

Scores eval_panel (the signal at 160M) + popqa_train (memorise check); skips popqa_tight (~0 at
160M) to keep the sweep short. Data prepped ONCE; training is in-process (reuses the smoke trainer).

USAGE (3080 box, via a committed .bat + schtasks so it survives disconnect):
  python -X utf8 scripts/dino_i2s_robustness_sweep.py --steps 400
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from rt116_quality_recovery import (  # noqa: E402
    load_corpus, _panel_exclude_set, _instruction_ids_mask, _wikitext_train_eval)
from fact004a_160m_smoke import score_dir  # noqa: E402
from dino_i2s_selfdistill_smoke import train_arm, load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "dino_i2s_001b_sweep")
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--lambdas", default="0.1,0.2,0.4")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dino-batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl-weight", type=float, default=0.2)
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--replay-batch", type=int, default=2)
    ap.add_argument("--replay-tokens", type=int, default=200_000)
    ap.add_argument("--view-mode", default="dropout")
    ap.add_argument("--view-p", type=float, default=0.1)
    ap.add_argument("--hidden-weight", type=float, default=0.0)  # discarded
    ap.add_argument("--center-m", type=float, default=0.9)
    ap.add_argument("--max-train-tokens", type=int, default=800_000)
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--train-sample", type=int, default=80)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--gate", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.work.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    lambdas = [float(l) for l in args.lambdas.split(",") if l]
    print(f"device={device}  model={args.model_id}  seeds={seeds}  lambdas={lambdas}  steps={args.steps}", flush=True)

    # ---- prep data ONCE (seed-independent corpus; only the per-step sampling RNG uses the seed) ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    cfg = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).config
    ids = sorted(set(int(i) for i in (tok.all_special_ids or []) if i is not None))
    special_vocab = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
    special_vocab[ids] = True
    panel_exclude = _panel_exclude_set(REPO_ROOT / "data/factual_panel_v1.jsonl")
    train_ids, _, train_mask = load_corpus("mixed", tok, args.max_train_tokens, args.eval_tokens,
                                           answer_mask=True, exclude_texts=panel_exclude)
    r_ids, r_mask = _instruction_ids_mask(tok, exclude_texts=panel_exclude)
    replay_ids, replay_mask = r_ids[: args.replay_tokens], r_mask[: args.replay_tokens]
    unlab_ids = train_ids
    ev = load_jsonl(REPO_ROOT / "data/factual_panel_v1.jsonl")
    tr = [{"id": r["id"], "prompt": r["prompt"], "must_contain": [r["answer"].strip().lower()]}
          for r in load_jsonl(REPO_ROOT / "data/popqa_blend_train.jsonl", args.train_sample)]
    _, wt = _wikitext_train_eval(tok, 1000, args.eval_tokens)
    print(f"train tokens={train_ids.numel():,}  replay={replay_ids.numel():,}", flush=True)

    def run(arm, seed, lam, out_dir):
        a = copy(args)
        a.seed = seed
        a.dino_logit_weight = lam
        train_arm(arm, a, tok, train_ids, train_mask, unlab_ids, replay_ids, replay_mask,
                  special_vocab, device, out_dir)
        e = score_dir(out_dir, tok, ev, device, args.max_new, wt, args.ce_windows)
        t = score_dir(out_dir, tok, tr, device, args.max_new, wt, args.ce_windows)
        return {"eval": e["fact_rate"], "ce": round(e["ce"], 3), "tags": e["tags"], "train": t["fact_rate"]}

    rows = []
    base_by_seed = {}
    for s in seeds:
        b = run("baseline", s, 0.0, args.work / f"s{s}_baseline")
        base_by_seed[s] = b["eval"]
        rows.append({"seed": s, "arm": "baseline", "lambda": None, **b, "dEval": 0.0})
        print(f"  [seed {s}] baseline: eval {b['eval']} | CE {b['ce']} | {b['tags']}", flush=True)
        for lam in lambdas:
            d = run("dino_logit", s, lam, args.work / f"s{s}_l{lam}")
            de = round(d["eval"] - base_by_seed[s], 3)
            rows.append({"seed": s, "arm": "dino_logit", "lambda": lam, **d, "dEval": de})
            print(f"  [seed {s}] dino_logit l={lam}: eval {d['eval']} (dEval {de:+.3f}) | "
                  f"train {d['train']} | CE {d['ce']} | {d['tags']}", flush=True)

    # ---- aggregate: per-lambda, how many seeds clear baseline + gate ----
    per_lambda = {}
    for lam in lambdas:
        cleared = [r for r in rows if r["arm"] == "dino_logit" and r["lambda"] == lam and r["dEval"] >= args.gate]
        deltas = [r["dEval"] for r in rows if r["arm"] == "dino_logit" and r["lambda"] == lam]
        per_lambda[lam] = {"seeds_cleared": len(cleared), "n_seeds": len(seeds),
                           "mean_dEval": round(sum(deltas) / max(len(deltas), 1), 3), "deltas": deltas}
    best_lambda = max(per_lambda, key=lambda L: (per_lambda[L]["seeds_cleared"], per_lambda[L]["mean_dEval"]))
    bl = per_lambda[best_lambda]
    passed = bl["seeds_cleared"] >= 2  # >=2/3 seeds

    lines = ["# DINO-I2S-001b robustness sweep (160M, dino_logit-only, hidden discarded)", "",
             f"model={args.model_id}  steps={args.steps}  view={args.view_mode}@{args.view_p}  "
             f"seeds={seeds}  lambdas={lambdas}  gate=+{args.gate}", "",
             "| seed | arm | lambda | eval_panel | dEval | popqa_train | CE | tags |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['lambda'] if r['lambda'] is not None else '-'} | "
                     f"{r['eval']} | {r['dEval']:+.3f} | {r['train']} | {r['ce']} | {r['tags']} |")
    lines += ["", "## Per-lambda (seeds clearing baseline +gate)"]
    for lam in lambdas:
        p = per_lambda[lam]
        lines.append(f"- lambda {lam}: {p['seeds_cleared']}/{p['n_seeds']} seeds cleared  "
                     f"(mean dEval {p['mean_dEval']:+.3f}, deltas {p['deltas']})")
    verdict = (
        f"PASS (DINO-I2S-001b): lambda {best_lambda} clears baseline +{args.gate} on "
        f"{bl['seeds_cleared']}/{len(seeds)} seeds (mean dEval {bl['mean_dEval']:+.3f}). The dino_logit "
        f"positive is seed-robust -> escalate to DINO-I2S-002 1.1B Colab gate (lambda {best_lambda}, "
        f"dino_logit-only; PASS eval>0.185 ideally >=0.25, tags ok, i2_s~=f16, no train memorise)."
        if passed else
        f"FAIL (DINO-I2S-001b): no lambda clears baseline +{args.gate} on >=2/3 seeds "
        f"(best lambda {best_lambda}: {bl['seeds_cleared']}/{len(seeds)}, mean {bl['mean_dEval']:+.3f}). "
        f"The DINO-I2S-001 0.222 was seed/lambda luck -> do NOT pay for 1.1B; reconsider DINO or close it.")
    lines += ["", "VERDICT: " + verdict]
    (args.work / "summary.json").write_text(
        json.dumps({"rows": rows, "per_lambda": {str(k): v for k, v in per_lambda.items()},
                    "best_lambda": best_lambda, "passed": passed, "verdict": verdict}, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {args.work / 'summary.md'}")


if __name__ == "__main__":
    main()
