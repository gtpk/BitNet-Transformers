#!/usr/bin/env python3
"""DINO-DIAG-001: WHY is dino_logit only a SMALL positive? Token-level dissection (no training).

DINO-I2S-001b found dino_logit is a real but small positive (~+0.037 eval at lambda 0.2, the
+0.148 single-seed result did not reproduce). FACT exact-match on 27 prompts is too coarse to know
WHY. This script compares the sweep's baseline vs dino_logit checkpoints (+ the FP teacher) on the
SAME 27 FACT prompts at the token level, averaging over seeds for stability.

For each prompt it teacher-forces "prompt + ' ' + Gold" and measures, per model, on the gold
answer tokens:
  - mean log P(gold tokens)           -> did DINO raise the gold answer probability?
  - rank of the first gold token       -> is gold climbing toward the top even if not #1?
  - entropy at the first answer pos     -> is DINO just sharpening?
  - top-k token set at first answer pos -> teacher/student top-k overlap (is DINO following teacher?)
plus generation correctness (rep-penalty 1.2) for baseline_correct / dino_correct.

Answers the user's 4 questions:
  1. Per-prompt + per-category: where did the gain land (capital vs author vs ...).
  2. delta log P(gold): is DINO moving the gold token even when exact-match stays flat.
  3. Where did KL drop: if CE improved but gold_logp flat -> the gain is on non-gold (function)
     tokens -> need an entity/answer-token-weighted KL.
  4. teacher-student top-k overlap + entropy: is lambda weak, or is the teacher signal spread too
     broadly across tokens to push factual ones.

USAGE (3080 box, after the 001b sweep checkpoints exist):
  python -X utf8 scripts/dino_diag001_token_analysis.py --sweep-dir reports/dino_i2s_001b_sweep \
    --lambda 0.2 --seeds 41,42,43
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fact004a_160m_smoke import generate, hit  # noqa: E402


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def gold_surface(gold):
    g = gold.strip()
    return " " + (g[0].upper() + g[1:] if g else g)


@torch.no_grad()
def token_metrics(model, tok, prompt, gold, device, topk):
    """Teacher-forced gold-token logprobs + first-answer-position rank/entropy/top-k set."""
    pids = tok(prompt, return_tensors="pt").input_ids.to(device)
    gids = tok(gold_surface(gold), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full = torch.cat([pids, gids], dim=1)
    logits = model(full).logits[0].float()
    logp = F.log_softmax(logits, dim=-1)
    Lp = pids.shape[1]
    gold_logps = [logp[Lp - 1 + i, g].item() for i, g in enumerate(gids[0].tolist())]
    pos0 = Lp - 1
    dist = logp[pos0]
    first_g = int(gids[0, 0].item())
    rank = int((dist > dist[first_g]).sum().item()) + 1
    ent = float(-(dist.exp() * dist).sum().item())
    tk = set(torch.topk(dist, topk).indices.tolist())
    return {"gold_logp": sum(gold_logps) / len(gold_logps), "rank": rank, "entropy": ent, "topk": tk}


def score_model_on_panel(model_dir, tok, panel, device, topk, max_new):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    model.config.use_cache = True
    out = {}
    for p in panel:
        gold = p["must_contain"][0]
        tm = token_metrics(model, tok, p["prompt"], gold, device, topk)
        txt = generate(model, tok, p["prompt"], max_new, device)
        tm["correct"] = int(hit(txt, p["must_contain"]))
        out[p["id"]] = tm
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1", help="FP teacher reference")
    ap.add_argument("--sweep-dir", type=Path, default=REPO_ROOT / "reports" / "dino_i2s_001b_sweep")
    ap.add_argument("--lambda", dest="lam", default="0.2")
    ap.add_argument("--seeds", default="41,42,43")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "dino_diag001")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    args.work.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    panel = load_jsonl(args.panel)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    print(f"device={device}  sweep={args.sweep_dir}  lambda={args.lam}  seeds={seeds}  n_panel={len(panel)}", flush=True)

    # teacher (FP) reference -- seed-independent, scored once from the HF id
    print("scoring FP teacher reference...", flush=True)
    teacher = score_model_on_panel(args.model_id, tok, panel, device, args.topk, args.max_new)

    # per seed: baseline + dino_logit(lambda) materialised dirs from the sweep
    base_runs, dino_runs = [], []
    for s in seeds:
        bdir = args.sweep_dir / f"s{s}_baseline"
        ddir = args.sweep_dir / f"s{s}_l{args.lam}"
        if not bdir.exists() or not ddir.exists():
            print(f"  [seed {s}] MISSING {bdir if not bdir.exists() else ddir} -- skipping", flush=True)
            continue
        print(f"  scoring seed {s} baseline + dino l={args.lam}...", flush=True)
        base_runs.append(score_model_on_panel(bdir, tok, panel, device, args.topk, args.max_new))
        dino_runs.append(score_model_on_panel(ddir, tok, panel, device, args.topk, args.max_new))
    if not base_runs:
        print("no seed checkpoints found -- run the 001b sweep first.", flush=True)
        return

    def avg(runs, pid, key):
        return sum(r[pid][key] for r in runs) / len(runs)

    rows = []
    for p in panel:
        pid = p["id"]
        b_logp, d_logp = avg(base_runs, pid, "gold_logp"), avg(dino_runs, pid, "gold_logp")
        b_rank, d_rank = avg(base_runs, pid, "rank"), avg(dino_runs, pid, "rank")
        b_ent, d_ent = avg(base_runs, pid, "entropy"), avg(dino_runs, pid, "entropy")
        b_corr, d_corr = avg(base_runs, pid, "correct"), avg(dino_runs, pid, "correct")
        # teacher/student top-k overlap (teacher vs dino), averaged over seeds
        ts_overlap = sum(len(teacher[pid]["topk"] & r[pid]["topk"]) for r in dino_runs) / (len(dino_runs) * args.topk)
        rows.append({
            "id": pid, "category": p.get("category", "?"), "gold": p["must_contain"][0],
            "base_correct": round(b_corr, 2), "dino_correct": round(d_corr, 2),
            "d_gold_logp": round(d_logp - b_logp, 3),
            "base_gold_rank": round(b_rank, 1), "dino_gold_rank": round(d_rank, 1),
            "teacher_gold_rank": teacher[pid]["rank"],
            "ts_topk_overlap": round(ts_overlap, 2),
            "d_entropy": round(d_ent - b_ent, 3),
        })

    # ---- aggregate diagnostics ----
    n = len(rows)
    mean_d_logp = sum(r["d_gold_logp"] for r in rows) / n
    frac_logp_up = sum(r["d_gold_logp"] > 0 for r in rows) / n
    mean_d_ent = sum(r["d_entropy"] for r in rows) / n
    mean_ts_overlap = sum(r["ts_topk_overlap"] for r in rows) / n
    rank_improved = sum(r["dino_gold_rank"] < r["base_gold_rank"] for r in rows) / n
    cats = {}
    for r in rows:
        c = r["category"]
        cats.setdefault(c, []).append(r["d_gold_logp"])
    cat_means = {c: round(sum(v) / len(v), 3) for c, v in cats.items()}
    # flips
    flips_gain = [r["id"] for r in rows if r["dino_correct"] > r["base_correct"]]
    flips_loss = [r["id"] for r in rows if r["dino_correct"] < r["base_correct"]]

    lines = ["# DINO-DIAG-001: token-level dissection of the small dino_logit positive (160M)", "",
             f"teacher={args.model_id}  sweep={args.sweep_dir.name}  lambda={args.lam}  seeds={seeds} "
             f"(averaged)  topk={args.topk}", "",
             "| id | category | gold | base_corr | dino_corr | d_gold_logp | base_rank | dino_rank | "
             "teacher_rank | ts_topk_ov | d_entropy |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda x: -x["d_gold_logp"]):
        lines.append(f"| {r['id']} | {r['category']} | {r['gold']} | {r['base_correct']} | {r['dino_correct']} "
                     f"| {r['d_gold_logp']:+.3f} | {r['base_gold_rank']} | {r['dino_gold_rank']} | "
                     f"{r['teacher_gold_rank']} | {r['ts_topk_overlap']} | {r['d_entropy']:+.3f} |")
    lines += ["", "## Aggregate",
              f"- mean delta log P(gold): **{mean_d_logp:+.3f}**  (fraction of prompts with gold_logp UP: {frac_logp_up:.0%})",
              f"- gold rank improved (dino < baseline) on {rank_improved:.0%} of prompts",
              f"- mean delta entropy: {mean_d_ent:+.3f}  (negative = sharpening)",
              f"- mean teacher-student top-k overlap: {mean_ts_overlap:.2f}",
              f"- per-category mean delta gold_logp: {cat_means}",
              f"- FACT flips: gained {flips_gain}; lost {flips_loss}", ""]
    # ---- verdict mapping (user's decision table) ----
    if mean_d_logp >= 0.15 and frac_logp_up >= 0.6:
        v = ("gold log P broadly UP -> DINO is genuinely working at the distribution level; the small "
             "exact-match gain is partly a decoding/extraction ceiling. 1.1B gate is WORTH it (more "
             "capacity -> the raised gold mass should convert to exact-match).")
    elif mean_d_logp <= 0.03 and mean_d_ent < -0.05:
        v = ("gold log P ~flat but entropy DOWN -> DINO mostly SHARPENS / improves function-token KL, "
             "not the factual gold token. Need ENTITY/ANSWER-TOKEN-WEIGHTED KL before 1.1B.")
    elif max(cat_means.values()) - min(cat_means.values()) >= 0.15:
        v = ("gain is CATEGORY-SKEWED (see per-category means) -> a category/data-targeted objective is "
             "the lever; broaden/reweight the unlabeled views toward the weak categories.")
    elif mean_ts_overlap >= 0.5 and mean_d_logp < 0.05:
        v = ("teacher-student top-k overlap high but gold_logp flat -> the teacher signal is spread too "
             "broadly across tokens to push factual ones (question 4B). Answer-token-weighted KL, not just bigger lambda.")
    else:
        v = ("weak/mixed token-level signal -> DINO is marginal at 160M; do NOT pay for 1.1B yet. "
             "Try answer-token-weighted KL or larger lambda and re-diagnose.")
    lines += ["VERDICT: " + v]
    (args.work / "summary.json").write_text(json.dumps({"rows": rows, "cat_means": cat_means,
        "mean_d_logp": mean_d_logp, "frac_logp_up": frac_logp_up, "mean_d_entropy": mean_d_ent,
        "mean_ts_overlap": mean_ts_overlap, "verdict": v}, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {args.work / 'summary.md'}")


if __name__ == "__main__":
    main()
