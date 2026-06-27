#!/usr/bin/env python3
"""SIDE-000 + SIDE-001: I2_S + LoRA sidecar rank sweep on 160M (docs/i2s_lora_sidecar_plan.md).

I2_S is the ROOT; the LoRA sidecar is an auxiliary organ that adds a tiny low-rank correction on
top of the ternary base. This asks the capacity question that PopQA blend (a data/objective test)
cannot: is one-plane I2_S simply short of capacity, and does a small residual move FACT?

For each rank in {0,2,4,8} run the current best recipe (content-KL 0.2 + PopQA blend 5%) on 160M,
co-adapting the ternary base AND the sidecar (rank 0 = ternary-only baseline; rank R = + LoRA, same
training, only capacity differs), then PyTorch-score the materialised (folded) model on:
  eval_panel (FACT-001, 27) | popqa_tight (PRIMARY transfer) | popqa_train (memorise) | CE
and read the sidecar byte overhead from the run JSON (SIDE-000).

Pass (plan): rank 4/8 improves FACT by >=0.05 without worse tags, sidecar bytes << Q2_K/Q3.

USAGE (3080 box): python -X utf8 scripts/side001_sidecar_smoke.py --ranks 0,2,4,8 --steps 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fact004a_160m_smoke import score_dir  # PyTorch panel scorer


def load(p, n=0):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows


def train_rank(args, rank, out_dir, json_out):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "rt116_quality_recovery.py"),
           "--model-id", args.model_id,
           "--train-source", "mixed", "--answer-loss-only",
           "--base-kl-replay", "--kl-content-only", "--kl-weight", "0.2", "--exclude-panel",
           "--factual-blend-file", str(REPO_ROOT / "data/popqa_blend_train.jsonl"), "--factual-blend-frac", "0.05",
           "--steps", str(args.steps), "--seq-len", str(args.seq_len), "--batch", str(args.batch),
           "--lr", str(args.lr), "--seed", str(args.seed), "--max-train-tokens", str(args.max_train_tokens),
           "--dtype", "float32", "--optim", "adamw",
           "--out-dir", str(out_dir), "--json-out", str(json_out), "--log-every", str(args.log_every)]
    if rank > 0:
        cmd += ["--sidecar-rank", str(rank), "--sidecar-alpha", str(args.alpha),
                "--sidecar-target", args.target]
        if args.train_base:
            cmd += ["--sidecar-train-base"]
    print(f"\n===== rank={rank} =====\n{' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"  rank={rank} trained in {(time.time()-t0)/60:.1f}m", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--ranks", default="0,2,4,8")
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--target", default="all")
    ap.add_argument("--train-base", action="store_true", default=True,
                    help="co-adapt base+sidecar (default, clean capacity test); --no-train-base for frozen base")
    ap.add_argument("--no-train-base", dest="train_base", action="store_false")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "side001_160m")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max-train-tokens", type=int, default=800_000)
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--tight-sample", type=int, default=200)
    ap.add_argument("--train-sample", type=int, default=80)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.work.mkdir(parents=True, exist_ok=True)
    ranks = [int(r) for r in args.ranks.split(",")]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    from rt116_quality_recovery import load_wikitext
    _, wt = load_wikitext(tok, 1000, args.eval_tokens)
    ev = load(REPO_ROOT / "data/factual_panel_v1.jsonl")
    tight = load(REPO_ROOT / "data/popqa_heldout_tight.jsonl", args.tight_sample)
    tr = [{"id": r["id"], "prompt": r["prompt"], "must_contain": [r["answer"].strip().lower()]}
          for r in load(REPO_ROOT / "data/popqa_blend_train.jsonl", args.train_sample)]
    print(f"device={device} model={args.model_id} ranks={ranks} train_base={args.train_base} "
          f"| eval={len(ev)} tight={len(tight)} train={len(tr)}", flush=True)

    rows = []
    for rank in ranks:
        out_dir = args.work / f"rank{rank}"
        jout = args.work / f"rank{rank}_train.json"
        if not args.skip_train:
            train_rank(args, rank, out_dir, jout)
        meta = json.load(open(jout)) if jout.exists() else {}
        ev_r = score_dir(out_dir, tok, ev, device, 40, wt, args.ce_windows)
        ti_r = score_dir(out_dir, tok, tight, device, 40, wt, args.ce_windows)
        tr_r = score_dir(out_dir, tok, tr, device, 40, wt, args.ce_windows)
        rows.append({"rank": rank, "eval": ev_r["fact_rate"], "tight": ti_r["fact_rate"],
                     "train": tr_r["fact_rate"], "ce": round(ev_r["ce"], 3), "tags": ev_r["tags"],
                     "sidecar_bytes": meta.get("sidecar_sidecar_bytes_fp16", 0),
                     "bytes_ratio": meta.get("sidecar_sidecar_bytes_ratio_vs_target_i2s", 0.0),
                     "recovered": meta.get("recovered_fraction")})
        print(f"  rank={rank}: eval {ev_r['fact_rate']} | tight {ti_r['fact_rate']} | train {tr_r['fact_rate']} "
              f"| CE {ev_r['ce']:.3f} | sidecar {rows[-1]['bytes_ratio']*100:.2f}% I2_S bytes | {ev_r['tags']}", flush=True)

    base = next((r for r in rows if r["rank"] == 0), rows[0])
    best = max((r for r in rows if r["rank"] > 0), key=lambda r: r["eval"], default=None)
    lines = ["# SIDE-001 I2_S + LoRA sidecar rank sweep (160M, co-adapted)" if args.train_base
             else "# SIDE-001 I2_S + LoRA sidecar rank sweep (160M, frozen base)", "",
             f"model={args.model_id}  recipe=content-KL 0.2 + PopQA blend 5%  steps={args.steps}  "
             f"target={args.target} alpha={args.alpha}  (I2_S is the root; sidecar = auxiliary low-rank organ)",
             "",
             "| rank | eval_panel | popqa_tight (PRIMARY) | popqa_train | CE | sidecar bytes | % of I2_S bytes | tags |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['rank']} | {r['eval']} | {r['tight']} | {r['train']} | {r['ce']} | "
                     f"{r['sidecar_bytes']:,} | {r['bytes_ratio']*100:.2f}% | {r['tags']} |")
    if best:
        d_eval = best["eval"] - base["eval"]
        d_tight = best["tight"] - base["tight"]
        passed = d_eval >= 0.05 and best["bytes_ratio"] < 0.5
        lines += ["", f"best rank by eval: {best['rank']} (eval {best['eval']} vs rank0 {base['eval']}, "
                  f"d {d_eval:+.3f}; tight {best['tight']} vs {base['tight']}, d {d_tight:+.3f}; "
                  f"sidecar {best['bytes_ratio']*100:.2f}% of I2_S bytes)", "",
                  "VERDICT: " + (
                      "PASS -- a small sidecar lifts FACT >=0.05 over the ternary-only baseline at small "
                      "byte cost => one-plane I2_S capacity IS a (partial) bottleneck; schedule 1.1B SIDE-003. "
                      "Claim: mostly-I2_S + tiny auxiliary sidecar, NOT pure I2_S."
                      if passed else
                      "FAIL/INCONCLUSIVE at 160M -- no rank lifts eval >=0.05 over ternary-only (or only large "
                      "rank/bytes helps). Capacity via tiny sidecar is not the lever here; lean on representative "
                      "data/objective (PopQA blend) or larger capacity. Keep as documented negative.")]
    (args.work / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-3:]))
    print(f"wrote {args.work/'summary.md'}")


if __name__ == "__main__":
    main()
