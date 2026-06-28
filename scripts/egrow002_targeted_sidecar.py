#!/usr/bin/env python3
"""EGROW-002: targeted I2_S sidecar -- top-k bottleneck layers vs random-k (docs/entropy_guided...).

EGROW-001 ranked target linears by B_l (sensitivity = output_residual x task_saliency; flip/entropy
were ~0). SIDE-001 (sidecar on ALL layers) showed no clear FACT lever. EGROW-002 asks the sharper
question: does a sidecar on the top-k layers by B_l beat a sidecar on a RANDOM, type-matched
(byte-fair) k? If top-k > random-k AND FACT moves, the bottleneck localization is real and
I2_S-rooted growth has a target. If top-k == random-k, the ranking buys nothing -> the lever stays
data/objective.

Arms (160M, content-KL 0.2 + PopQA blend 5%, base co-adapted):
  none   rank 0 (ternary-only baseline)
  topk   rank R sidecar on the top-k layers by EGROW B_l
  randk  rank R sidecar on k random target linears, MATCHED to the top-k module types (byte-fair)

Scored on eval_panel + popqa_tight (PRIMARY) + popqa_train. Verdict = topk eval - randk eval.

USAGE (3080 box): python -X utf8 scripts/egrow002_targeted_sidecar.py --egrow reports/egrow_160m_s41.json --k 5 --rank 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fact004a_160m_smoke import score_dir


def load(p, n=0):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows


def mtype(name):
    return name.split(".")[-1]


def train_arm(args, layers, out_dir, jout):
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "rt116_quality_recovery.py"),
           "--model-id", args.model_id, "--train-source", "mixed", "--answer-loss-only",
           "--base-kl-replay", "--kl-content-only", "--kl-weight", "0.2", "--exclude-panel",
           "--factual-blend-file", str(REPO_ROOT / "data/popqa_blend_train.jsonl"), "--factual-blend-frac", "0.05",
           "--steps", str(args.steps), "--seq-len", str(args.seq_len), "--batch", str(args.batch),
           "--lr", str(args.lr), "--seed", str(args.seed), "--max-train-tokens", str(args.max_train_tokens),
           "--dtype", "float32", "--optim", "adamw",
           "--out-dir", str(out_dir), "--json-out", str(jout), "--log-every", str(args.log_every)]
    if layers:
        cmd += ["--sidecar-rank", str(args.rank), "--sidecar-alpha", str(args.alpha),
                "--sidecar-train-base", "--sidecar-layers", ",".join(layers)]
    print(f"\n===== {'rank0' if not layers else str(len(layers))+' layers'} =====\n{' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"  trained in {(time.time()-t0)/60:.1f}m", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--egrow", type=Path, default=REPO_ROOT / "reports/egrow_160m_s41.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--rand-seed", type=int, default=7)
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "egrow002_160m")
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
    egrow = json.load(open(args.egrow))["layers"]  # sorted by bottleneck_score desc
    all_names = [r["layer_name"] for r in egrow]
    topk = all_names[: args.k]
    # type-matched random-k from the non-top layers (byte-fair: same module types as top-k)
    rng = random.Random(args.rand_seed)
    non_top = [n for n in all_names if n not in set(topk)]
    randk = []
    for t in topk:
        pool = [n for n in non_top if mtype(n) == mtype(t) and n not in randk]
        randk.append(rng.choice(pool) if pool else rng.choice([n for n in non_top if n not in randk]))
    print(f"k={args.k} rank={args.rank}\n  TOP-k  : {topk}\n  RAND-k : {randk}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    from rt116_quality_recovery import load_wikitext
    _, wt = load_wikitext(tok, 1000, args.eval_tokens)
    ev = load(REPO_ROOT / "data/factual_panel_v1.jsonl")
    tight = load(REPO_ROOT / "data/popqa_heldout_tight.jsonl", args.tight_sample)
    tr = [{"id": r["id"], "prompt": r["prompt"], "must_contain": [r["answer"].strip().lower()]}
          for r in load(REPO_ROOT / "data/popqa_blend_train.jsonl", args.train_sample)]

    arms = [("none", []), ("topk", topk), ("randk", randk)]
    rows = []
    for nm, layers in arms:
        out_dir = args.work / nm
        jout = args.work / f"{nm}_train.json"
        if not args.skip_train:
            train_arm(args, layers, out_dir, jout)
        meta = json.load(open(jout)) if jout.exists() else {}
        e = score_dir(out_dir, tok, ev, device, 40, wt, args.ce_windows)
        ti = score_dir(out_dir, tok, tight, device, 40, wt, args.ce_windows)
        trs = score_dir(out_dir, tok, tr, device, 40, wt, args.ce_windows)
        rows.append({"arm": nm, "n_layers": len(layers), "eval": e["fact_rate"], "tight": ti["fact_rate"],
                     "train": trs["fact_rate"], "ce": round(e["ce"], 3), "tags": e["tags"],
                     "sidecar_bytes": meta.get("sidecar_sidecar_bytes_fp16", 0)})
        print(f"  {nm}: eval {e['fact_rate']} | tight {ti['fact_rate']} | train {trs['fact_rate']} "
              f"| CE {e['ce']:.3f} | sidecar {rows[-1]['sidecar_bytes']:,}B | {e['tags']}", flush=True)

    d = {r["arm"]: r for r in rows}
    d_eval = d["topk"]["eval"] - d["randk"]["eval"]
    d_vs_none = d["topk"]["eval"] - d["none"]["eval"]
    lines = ["# EGROW-002 targeted sidecar: top-k (B_l) vs random-k (160M)", "",
             f"model={args.model_id}  k={args.k} rank={args.rank}  recipe=content-KL 0.2 + PopQA blend 5%  steps={args.steps}",
             f"TOP-k: {topk}", f"RAND-k: {randk}", "",
             "| arm | layers | eval_panel | popqa_tight (PRIMARY) | popqa_train | CE | sidecar bytes | tags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['n_layers']} | {r['eval']} | {r['tight']} | {r['train']} | {r['ce']} | "
                     f"{r['sidecar_bytes']:,} | {r['tags']} |")
    passed = d_eval >= 0.03 and d_vs_none >= 0.03
    lines += ["", f"top-k vs random-k eval delta: {d_eval:+.3f}; top-k vs none: {d_vs_none:+.3f}", "",
              "VERDICT: " + (
                  "LOCALIZATION REAL -- top-k sidecar beats both random-k and none on eval; B_l locates "
                  "I2_S growth sites. Schedule 1.1B EGROW-004 (sidecar on 1.1B top-k by B_l)."
                  if passed else
                  "LOCALIZATION INCONCLUSIVE at 160M -- top-k does not beat random-k/none by >=0.03 eval. "
                  "The B_l ranking does not buy targeted-capacity gains here; the lever stays data/"
                  "objective. Keep EGROW ranking as a documented diagnostic.")]
    (args.work / "summary.json").write_text(json.dumps({"topk": topk, "randk": randk, "rows": rows}, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-3:]))
    print(f"wrote {args.work/'summary.md'}")


if __name__ == "__main__":
    main()
