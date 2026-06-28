#!/usr/bin/env python3
"""HOME-001: activation homeostasis smoke on 160M.

The biological analogy is homeostatic plasticity: adapt the ternary model, but
keep selected hidden-state mean/RMS statistics close to the base model. This
wrapper trains a small eta sweep with rt116's --homeostasis-* flags and scores
FACT + PopQA panels in PyTorch.

Usage (RTX 3080 / Windows-safe):
  python -X utf8 scripts/home001_activation_homeostasis_smoke.py --etas 0,0.01,0.05
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fact004a_160m_smoke import score_dir  # noqa: E402


def load_jsonl(path: Path, n: int = 0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows


def train_arm(args, eta: float, out_dir: Path):
    arm = f"home_eta{eta:g}".replace(".", "p")
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "rt116_quality_recovery.py"),
           "--model-id", args.model_id,
           "--train-source", "mixed", "--answer-loss-only",
           "--base-kl-replay", "--kl-content-only", "--kl-weight", "0.2",
           "--exclude-panel",
           "--factual-blend-file", str(REPO_ROOT / "data/popqa_blend_train.jsonl"),
           "--factual-blend-frac", "0.05",
           "--steps", str(args.steps), "--seq-len", str(args.seq_len),
           "--batch", str(args.batch), "--lr", str(args.lr),
           "--max-train-tokens", str(args.max_train_tokens),
           "--dtype", "float32", "--optim", "adamw",
           "--out-dir", str(out_dir),
           "--json-out", str(args.work / f"{arm}_train.json"),
           "--metrics-out", str(args.work / f"{arm}_metrics.jsonl"),
           "--log-every", str(args.log_every)]
    if eta > 0:
        cmd += ["--homeostasis-weight", str(eta),
                "--homeostasis-layers", args.homeostasis_layers,
                "--homeostasis-rho", str(args.homeostasis_rho)]
    print(f"\n===== HOME arm eta={eta:g} =====\n{' '.join(cmd)}", flush=True)
    import os
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"  trained eta={eta:g} in {(time.time()-t0)/60:.1f}m -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "home001_160m")
    ap.add_argument("--etas", default="0,0.01,0.05")
    ap.add_argument("--homeostasis-layers", choices=["last", "mid_last"], default="last")
    ap.add_argument("--homeostasis-rho", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-tokens", type=int, default=800_000)
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--tight-sample", type=int, default=200)
    ap.add_argument("--train-sample", type=int, default=80)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    etas = [float(x) for x in args.etas.split(",") if x.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    from rt116_quality_recovery import load_wikitext
    _, wt_eval = load_wikitext(tok, 1000, args.eval_tokens)

    eval_panel = load_jsonl(REPO_ROOT / "data/factual_panel_v1.jsonl")
    tight = load_jsonl(REPO_ROOT / "data/popqa_heldout_tight.jsonl", args.tight_sample)
    train = [
        {"id": r["id"], "prompt": r["prompt"], "must_contain": [r["answer"].strip().lower()]}
        for r in load_jsonl(REPO_ROOT / "data/popqa_blend_train.jsonl", args.train_sample)
    ]

    rows = []
    for eta in etas:
        arm = f"home_eta{eta:g}".replace(".", "p")
        out_dir = args.work / arm
        if not args.skip_train:
            train_arm(args, eta, out_dir)
        ev = score_dir(out_dir, tok, eval_panel, device, args.max_new, wt_eval, args.ce_windows)
        ti = score_dir(out_dir, tok, tight, device, args.max_new, wt_eval, args.ce_windows)
        tr = score_dir(out_dir, tok, train, device, args.max_new, wt_eval, args.ce_windows)
        train_json = args.work / f"{arm}_train.json"
        meta = json.load(open(train_json, encoding="utf-8")) if train_json.exists() else {}
        rows.append({
            "arm": arm, "eta": eta,
            "eval_panel": ev["fact_rate"],
            "popqa_tight": ti["fact_rate"],
            "popqa_train": tr["fact_rate"],
            "ce": round(ev["ce"], 3),
            "tags": ev["tags"],
            "recovered_fraction": meta.get("recovered_fraction"),
        })
        print(f"  {arm}: eval {ev['fact_rate']} | tight {ti['fact_rate']} | train {tr['fact_rate']} "
              f"| CE {ev['ce']:.3f} | {ev['tags']}", flush=True)

    base = rows[0]
    best = max(rows, key=lambda r: (r["eval_panel"], r["popqa_tight"], -r["ce"]))
    delta = best["eval_panel"] - base["eval_panel"]
    passed = best["eta"] > 0 and delta >= 0.05 and best["ce"] <= base["ce"] + 0.10

    lines = ["# HOME-001 activation homeostasis smoke (160M)", "",
             f"model={args.model_id}  recipe=content-KL 0.2 + PopQA blend 5%  steps={args.steps}",
             f"layers={args.homeostasis_layers} rho={args.homeostasis_rho}", "",
             "| arm | eta | eval_panel | popqa_tight | popqa_train | CE | recovered | tags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['eta']} | {r['eval_panel']} | {r['popqa_tight']} | "
                     f"{r['popqa_train']} | {r['ce']} | {r['recovered_fraction']} | {r['tags']} |")
    lines += ["", f"best={best['arm']} delta_eval_vs_eta0={delta:+.3f}", "",
              "VERDICT: " + (
                  "PASS -- activation homeostasis improves FACT without CE/tag collapse; consider HOME-002 1.1B."
                  if passed else
                  "NO CLEAR SIGNAL -- homeostasis does not buy >=0.05 eval over eta=0 under this smoke.")]

    (args.work / "summary.json").write_text(json.dumps({"rows": rows, "best": best}, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-3:]))
    print(f"wrote {args.work/'summary.md'}")


if __name__ == "__main__":
    main()
