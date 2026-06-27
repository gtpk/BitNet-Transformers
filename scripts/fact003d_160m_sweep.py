#!/usr/bin/env python3
"""FACT-003D 160M mu-sweep: predict the protected-factual-replay mu direction before 1.1B Colab.

Runs the FACT-003D/003E recipe (content-KL lambda=0.2 + mu*answer-CE on a protected
factual replay set) on a 160M LLaMA, and PyTorch-scores THREE panels per arm:
  - eval panel       (data/factual_panel_v1.jsonl) -- the real held-out factual eval
  - heldout replay   (default atomic heldout, or a caller-provided heldout file)
  - train replay*    (sample of the train replay file) -- MEMORISATION control

Reading (relative, at 160M; absolute fact_rate is a floor):
  heldout rises with train as mu grows  => protected replay TRANSFERS; pick the mu that lifts
      heldout without crashing eval-panel CE -> run that mu at 1.1B.
  only train rises, heldout flat         => item memorisation; need broader factual data (not just mu).
  eval-panel CE/tags crash at high mu    => mu too strong; favour the lower mu.

PyTorch-scored on the ternary-materialised dir (no bitnet.cpp), so it runs on the 10GB 3080.

USAGE (3080 box, conda bnt):
  python -X utf8 scripts/fact003d_160m_sweep.py --mus 0.5,1.0,2.0 --steps 400

FACT-003E length-mixed replay:
  python -X utf8 scripts/make_length_balanced_factual_replay.py
  python -X utf8 scripts/fact003d_160m_sweep.py --label FACT-003E-lengthmix \
    --mus 1.0 --steps 400 --seed 41 --work reports/fact003e_160m_lengthmix_seed41 \
    --factual-replay data/atomic_facts_lengthmix_train.jsonl \
    --heldout-file data/atomic_facts_lengthmix_heldout.jsonl \
    --train-score-file data/atomic_facts_lengthmix_train.jsonl --heldout-sample 150
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

from fact004a_160m_smoke import score_dir  # reuse the exact PyTorch panel scorer


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def panelize(rows, limit=0):
    if limit and limit > 0:
        rows = rows[:limit]
    panel = []
    for idx, row in enumerate(rows):
        must = row.get("must_contain")
        if must is None:
            must = [str(row["answer"]).strip().lower()]
        elif isinstance(must, str):
            must = [must.strip().lower()]
        else:
            must = [str(x).strip().lower() for x in must if str(x).strip()]
        panel.append({
            "id": row.get("id", f"row:{idx}"),
            "prompt": row["prompt"],
            "must_contain": must,
        })
    return panel


def train_arm(args, mu, seed, out_dir):
    replay_path = repo_path(args.factual_replay)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "rt116_quality_recovery.py"),
           "--model-id", args.model_id,
           "--train-source", "mixed", "--answer-loss-only",
           "--base-kl-replay", "--kl-content-only", "--kl-weight", "0.2", "--exclude-panel",
           "--factual-replay", str(replay_path),
           "--factual-weight", str(mu), "--factual-batch", "4",
           "--steps", str(args.steps), "--seq-len", str(args.seq_len),
           "--batch", str(args.batch), "--lr", str(args.lr), "--seed", str(seed),
           "--max-train-tokens", str(args.max_train_tokens),
           "--dtype", "float32", "--optim", "adamw",
           "--out-dir", str(out_dir),
           "--json-out", str(out_dir.parent / f"{out_dir.name}_train.json"),
           "--log-every", str(args.log_every)]
    print(f"\n===== mu={mu} seed={seed} =====\n{' '.join(cmd)}", flush=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    subprocess.run(cmd, check=True, env=env)
    print(f"  mu={mu} seed={seed} trained in {(time.time()-t0)/60:.1f}m -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--label", default="FACT-003D")
    ap.add_argument("--mus", default="0.5,1.0,2.0")
    ap.add_argument("--seeds", default="0", help="comma list; >1 seed at a single mu = Q1 seed-variance check")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "fact003d_160m_sweep")
    ap.add_argument("--seed", type=int, default=None, help="single-seed shorthand; overrides --seeds when set")
    ap.add_argument("--factual-replay", type=Path, default=REPO_ROOT / "data/atomic_facts_train.jsonl")
    ap.add_argument("--heldout-file", type=Path, default=REPO_ROOT / "data/atomic_facts_heldout.jsonl")
    ap.add_argument("--train-score-file", type=Path, default=None,
                    help="defaults to --factual-replay; sampled as memorisation/control panel")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-tokens", type=int, default=800_000)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--train-sample", type=int, default=60, help="how many train-atomic facts to score (memorisation control)")
    ap.add_argument("--heldout-sample", type=int, default=0, help="0 = score the whole heldout replay file")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.work.mkdir(parents=True, exist_ok=True)
    mus = [float(x) for x in args.mus.split(",")]
    seeds = [args.seed] if args.seed is not None else [int(s) for s in args.seeds.split(",")]
    multiseed = len(seeds) > 1
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    from rt116_quality_recovery import load_wikitext
    _, wt_eval = load_wikitext(tok, 1000, args.eval_tokens)

    eval_panel = load_jsonl(REPO_ROOT / "data/factual_panel_v1.jsonl")
    heldout_path = repo_path(args.heldout_file)
    replay_path = repo_path(args.factual_replay)
    train_score_path = repo_path(args.train_score_file or args.factual_replay)
    heldout = panelize(load_jsonl(heldout_path), args.heldout_sample)
    train_replay = panelize(load_jsonl(train_score_path), args.train_sample)
    print(f"device={device} label={args.label} model={args.model_id} mus={mus} | eval={len(eval_panel)} "
          f"heldout={len(heldout)} train_sample={len(train_replay)}", flush=True)
    print(f"replay={replay_path} heldout={heldout_path} train_score={train_score_path}", flush=True)

    rows = []
    for mu in mus:
        for seed in seeds:
            out_dir = args.work / (f"mu{mu}_s{seed}" if multiseed else f"mu{mu}")
            if not args.skip_train:
                train_arm(args, mu, seed, out_dir)
            ev = score_dir(out_dir, tok, eval_panel, device, args.max_new, wt_eval, args.ce_windows)
            ho = score_dir(out_dir, tok, heldout, device, args.max_new, wt_eval, args.ce_windows)
            tr = score_dir(out_dir, tok, train_replay, device, args.max_new, wt_eval, args.ce_windows)
            rows.append({"mu": mu, "seed": seed, "eval": ev, "heldout": ho, "train": tr})
            print(f"  mu={mu} seed={seed}: eval_panel {ev['fact_rate']} | heldout {ho['fact_rate']} | "
                  f"train_replay {tr['fact_rate']} | CE {ev['ce']:.3f}", flush=True)

    title = (f"{args.label} 160M seed-variance (mu={mus[0]})" if multiseed
             else f"{args.label} 160M mu-sweep")
    lines = [f"# {title} (protected factual replay)", "",
             f"model={args.model_id} recipe=content-KL 0.2 + mu*factual-CE steps={args.steps} "
             f"PyTorch-scored (ternary-materialised, rep-penalty 1.2)", "",
             f"- replay: `{replay_path}`",
             f"- heldout: `{heldout_path}` (sample={args.heldout_sample or 'all'})",
             f"- train-score: `{train_score_path}` (sample={args.train_sample})",
             "",
             "| mu | seed | eval_panel | heldout_replay (transfer) | train_replay (memorise) | eval CE |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['mu']} | {r['seed']} | {r['eval']['fact_rate']} | {r['heldout']['fact_rate']} | "
                     f"{r['train']['fact_rate']} | {r['eval']['ce']:.3f} |")
    if multiseed and rows:
        evs = [r["eval"]["fact_rate"] for r in rows]
        hos = [r["heldout"]["fact_rate"] for r in rows]
        mean = lambda xs: sum(xs) / len(xs)
        lo, hi = min(evs), max(evs)
        # control eval_panel ~0.037 (mu=0 baseline, reports/rt136_fact003d_160m_sweep.md)
        consistent = lo > 0.10  # every seed clearly above the mu=0 floor
        lines += ["", f"eval_panel across seeds: mean {mean(evs):.3f}  range [{lo}, {hi}]  "
                  f"(mu=0 control ~0.037); heldout mean {mean(hos):.3f}", "",
                  "READING: " + (
                      "eval_panel ABOVE the mu=0 control on EVERY seed -> 160M predictor is "
                      "seed-robust; trust it for future branch-killing."
                      if consistent else
                      "eval_panel NOT consistently above the mu=0 control -> predictor is noisy; "
                      "do not over-read a single 160M run.")]
    else:
        best = max(rows, key=lambda r: r["heldout"]["fact_rate"]) if rows else None
        if best:
            transfers = best["heldout"]["fact_rate"] > rows[0]["heldout"]["fact_rate"] + 1e-9
            lines += ["", f"best heldout: mu={best['mu']} @ {best['heldout']['fact_rate']} "
                      f"(train_atomic {best['train']['fact_rate']})", "",
                      "READING: " + (
                          f"heldout TRANSFER signal at mu={best['mu']} -> run that mu at 1.1B."
                          if transfers else
                          "heldout flat while train_atomic rises -> memorisation, need broader data.")]
    (args.work / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {args.work/'summary.md'}")


if __name__ == "__main__":
    main()
