#!/usr/bin/env python3
"""Score a materialized HF dir on the factual panel (raw Q:/A:) -- FACT exact rate + gold_rank +
first_token_hit + sample generations. Reusable across the Qwen ladder etc. (apples-to-apples with the
TinyLlama 0.185 baseline: same panel, same generate/hit/tag, rep-penalty 1.2)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT)); sys.path.insert(0, str(REPO_ROOT / "scripts"))
from fact004a_160m_smoke import generate, hit, tag  # noqa: E402


def gold_surface(g):
    g = g.strip(); return " " + (g[0].upper() + g[1:] if g else g)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="materialized HF model dir (or HF id) to score")
    ap.add_argument("--teacher-id", default=None, help="FP reference for gold_rank ratio (optional)")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--n-samples", type=int, default=6)
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.dir)
    panel = [json.loads(l) for l in open(args.panel, encoding="utf-8") if l.strip()]
    m = AutoModelForCausalLM.from_pretrained(args.dir, dtype=torch.float32).to(dev).eval(); m.config.use_cache = True
    hits, tags, ranks, samples = 0, {}, [], []
    for i, p in enumerate(panel):
        txt = generate(m, tok, p["prompt"], args.max_new, dev)
        h = hit(txt, p["must_contain"]); hits += int(h)
        tg = tag(txt); tags[tg] = tags.get(tg, 0) + 1
        with torch.no_grad():
            pids = tok(p["prompt"], return_tensors="pt").input_ids.to(dev)
            gids = tok(gold_surface(p["must_contain"][0]), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            m.config.use_cache = False
            lp = F.log_softmax(m(torch.cat([pids, gids], 1)).logits[0].float(), -1)[pids.shape[1] - 1]
            m.config.use_cache = True
            first = int(gids[0, 0]); ranks.append(int((lp > lp[first]).sum()) + 1)
        if i < args.n_samples:
            samples.append({"gold": p["must_contain"][0], "hit": h, "out": txt[:110]})
    n = len(panel)
    res = {"dir": str(args.dir), "fact_rate": round(hits / n, 3), "hits": hits, "n": n,
           "gold_rank_mean": round(sum(ranks) / n, 1), "first_token_hit": round(sum(1 for r in ranks if r == 1) / n, 3),
           "tags": tags, "samples": samples}
    print("FACT %d/%d=%.3f | gold_rank_mean %.0f | first_token_hit %.3f | tags %s"
          % (hits, n, res["fact_rate"], res["gold_rank_mean"], res["first_token_hit"], tags))
    for s in samples:
        print(("  HIT  " if s["hit"] else "  miss ") + repr(s["gold"]) + " -> " + repr(s["out"]))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
