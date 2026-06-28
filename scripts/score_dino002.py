#!/usr/bin/env python3
"""DINO-I2S-002 1.1B gate scorer: FACT (overall + per category) + gold-rank vs FP teacher +
PopQA tight + tags + WikiText CE. One pass over a materialized (i2_s-folded) HF dir.

The DINO-DIAG-001 prediction to test at 1.1B: simple_fact rises, entity_attr lags; and the 160M
gold-rank improvement should convert to exact-match with more capacity. So besides exact-match this
reports, per prompt, the DINO student's gold-token rank vs the FP teacher's -- "did the bigger model
close the gap to the teacher it is distilling from".

USAGE (Colab, after the 1.1B DINO run materialises adapted_model):
  python score_dino002.py --dino-dir <adapted_model> --teacher-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --panel data/factual_panel_v1.jsonl --tight data/popqa_heldout_tight.jsonl --out pyscore.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fact004a_160m_smoke import generate, hit, tag  # noqa: E402


def load_jsonl(p, n=0):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows


def gold_surface(g):
    g = g.strip()
    return " " + (g[0].upper() + g[1:] if g else g)


@torch.no_grad()
def gold_rank_logp(model, tok, prompt, gold, device):
    pids = tok(prompt, return_tensors="pt").input_ids.to(device)
    gids = tok(gold_surface(gold), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full = torch.cat([pids, gids], dim=1)
    logp = F.log_softmax(model(full).logits[0].float(), dim=-1)
    Lp = pids.shape[1]
    gl = sum(logp[Lp - 1 + i, g].item() for i, g in enumerate(gids[0].tolist())) / gids.shape[1]
    dist = logp[Lp - 1]
    first = int(gids[0, 0].item())
    rank = int((dist > dist[first]).sum().item()) + 1
    return gl, rank


def panel_scores(model, tok, panel, device, max_new, want_rank):
    hits, tags = 0, defaultdict(int)
    by_cat = defaultdict(lambda: [0, 0])  # cat -> [hits, n]
    ranks = {}
    for p in panel:
        txt = generate(model, tok, p["prompt"], max_new, device)
        h = hit(txt, p["must_contain"]); hits += int(h)
        tags[tag(txt)] += 1
        c = p.get("category", "?"); by_cat[c][0] += int(h); by_cat[c][1] += 1
        if want_rank:
            gl, rk = gold_rank_logp(model, tok, p["prompt"], p["must_contain"][0], device)
            ranks[p["id"]] = {"gold_logp": round(gl, 3), "rank": rk}
    cat_rate = {c: round(v[0] / v[1], 3) for c, v in by_cat.items()}
    return {"fact_rate": round(hits / len(panel), 3), "hits": hits, "n": len(panel),
            "by_category": cat_rate, "tags": dict(tags), "ranks": ranks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dino-dir", type=Path, required=True)
    ap.add_argument("--teacher-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--tight", type=Path, default=REPO_ROOT / "data/popqa_heldout_tight.jsonl")
    ap.add_argument("--tight-sample", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.teacher_id)
    panel = load_jsonl(args.panel)
    tight = load_jsonl(args.tight, args.tight_sample)
    tight = [{"id": r.get("id", f"t{i}"), "prompt": r["prompt"],
              "must_contain": r.get("must_contain", [r.get("answer", "")]), "category": "popqa"}
             for i, r in enumerate(tight)]

    print(f"scoring DINO student {args.dino_dir} ...", flush=True)
    dm = AutoModelForCausalLM.from_pretrained(args.dino_dir, dtype=torch.float32).to(device).eval()
    dm.config.use_cache = True
    eval_d = panel_scores(dm, tok, panel, device, args.max_new, want_rank=True)
    tight_d = panel_scores(dm, tok, tight, device, args.max_new, want_rank=False)
    del dm
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"scoring FP teacher {args.teacher_id} (gold-rank reference) ...", flush=True)
    tm = AutoModelForCausalLM.from_pretrained(args.teacher_id, dtype=torch.float32).to(device).eval()
    tm.config.use_cache = True
    teacher_ranks = {p["id"]: gold_rank_logp(tm, tok, p["prompt"], p["must_contain"][0], device)[1] for p in panel}
    del tm
    if device == "cuda":
        torch.cuda.empty_cache()

    # gap-to-teacher: per prompt, dino rank vs teacher rank
    closed = sum(1 for p in panel if eval_d["ranks"][p["id"]]["rank"] <= teacher_ranks[p["id"]])
    gap = {p["id"]: {"dino_rank": eval_d["ranks"][p["id"]]["rank"], "teacher_rank": teacher_ranks[p["id"]],
                     "gold_logp": eval_d["ranks"][p["id"]]["gold_logp"], "category": p.get("category", "?")}
           for p in panel}

    out = {"eval_panel": {"fact_rate": eval_d["fact_rate"], "hits": eval_d["hits"], "n": eval_d["n"],
                          "by_category": eval_d["by_category"], "tags": eval_d["tags"]},
           "popqa_tight": {"fact_rate": tight_d["fact_rate"], "hits": tight_d["hits"], "n": tight_d["n"],
                           "tags": tight_d["tags"]},
           "gold_rank_vs_teacher": {"dino_closed_gap_on": closed, "n": len(panel), "per_prompt": gap}}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSUMMARY: eval {eval_d['fact_rate']} | by_cat {eval_d['by_category']} | "
          f"popqa_tight {tight_d['fact_rate']} | dino<=teacher rank on {closed}/{len(panel)} | tags {eval_d['tags']}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
