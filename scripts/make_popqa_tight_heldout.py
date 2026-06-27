#!/usr/bin/env python3
"""PC task 1b: sharpen the FACT-003H heldout scoring panel BEFORE the PopQA-blend 1.1B run.

The raw PopQA heldout (data/popqa_blend_heldout.jsonl) carries scoring noise: ~4% of items have
>10 aliases in must_contain (e.g. 19-69 alias strings) and ~half have multi-word answers. With a
contains-match scorer that noise makes "did heldout PopQA go up?" ambiguous -- a permissive
many-alias item can match spuriously. This builds a TIGHT heldout panel so the FACT-003H transfer
signal is unambiguous, leaving the loose panel available for a secondary (permissive) read.

Tight conditions (per user):
  - alias_count (len must_contain) <= --max-aliases (default 3)
  - has a short clean answer: at least one must_contain entry <= --max-answer-words (default 2)
  - prompt length 6..20 words (kept)
  - de-leaked vs the eval factual panel (re-verified)

Outputs:
  data/popqa_heldout_tight.jsonl
  reports/popqa_hygiene_tight_summary.json

Usage:  python -X utf8 scripts/make_popqa_tight_heldout.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL = REPO_ROOT / "data/factual_panel_v1.jsonl"


def panel_exclude_set(path):
    out = set()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        p = json.loads(line)
        out.add(re.sub(r"\s+", " ", p["prompt"]).strip().lower())
        for a in p.get("must_contain", []):
            a = re.sub(r"\s+", " ", a).strip().lower()
            if len(a) >= 5:
                out.add(a)
    return out


def leaks(prompt, must, excl):
    norm = re.sub(r"\s+", " ", prompt + " " + " ".join(must)).strip().lower()
    return any(e in norm for e in excl)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", type=Path, default=REPO_ROOT / "data/popqa_blend_heldout.jsonl")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data/popqa_heldout_tight.jsonl")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reports/popqa_hygiene_tight_summary.json")
    ap.add_argument("--max-aliases", type=int, default=3)
    ap.add_argument("--max-answer-words", type=int, default=2)
    ap.add_argument("--prompt-min", type=int, default=6)
    ap.add_argument("--prompt-max", type=int, default=20)
    args = ap.parse_args()

    excl = panel_exclude_set(PANEL)
    rows = [json.loads(l) for l in open(args.heldout, encoding="utf-8") if l.strip()]

    kept, reasons = [], {"alias": 0, "answer_len": 0, "prompt_len": 0, "leak": 0}
    for r in rows:
        must = [str(a).strip() for a in r.get("must_contain", []) if str(a).strip()]
        pw = len(r["prompt"].split())
        if len(must) > args.max_aliases:
            reasons["alias"] += 1; continue
        if not any(len(a.split()) <= args.max_answer_words for a in must):
            reasons["answer_len"] += 1; continue
        if not (args.prompt_min <= pw <= args.prompt_max):
            reasons["prompt_len"] += 1; continue
        if leaks(r["prompt"], must, excl):
            reasons["leak"] += 1; continue
        # keep only short alias forms so the contains-match is specific
        tight_must = [a.lower() for a in must if len(a.split()) <= args.max_answer_words] or [must[0].lower()]
        kept.append({"id": r.get("id"), "category": "popqa_tight", "prompt": r["prompt"],
                     "must_contain": sorted(set(tight_must))})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # post-write leak re-check
    leak = sum(1 for r in kept if leaks(r["prompt"], r["must_contain"], excl))
    import statistics
    al = [len(r["must_contain"]) for r in kept]
    summary = {
        "source_heldout": str(args.heldout), "source_rows": len(rows),
        "kept": len(kept), "dropped_reasons": reasons,
        "filters": {"max_aliases": args.max_aliases, "max_answer_words": args.max_answer_words,
                    "prompt_words": [args.prompt_min, args.prompt_max]},
        "tight_alias_count": {"min": min(al) if al else 0, "mean": round(statistics.mean(al), 2) if al else 0,
                              "max": max(al) if al else 0},
        "postcheck_leak": leak,
        "out": str(args.out),
        "scoring_caveat": ("PopQA heldout contains-match is alias-permissive: the RAW panel "
                           "(popqa_blend_heldout.jsonl) keeps up to 69 aliases / multi-word answers, "
                           "which can inflate fact_rate via spurious matches. Use this TIGHT panel "
                           "(<=3 short aliases, <=2-word answers) as the PRIMARY FACT-003H transfer "
                           "metric; report the loose panel only as a permissive secondary read."),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PopQA heldout tight: {len(rows)} -> kept {len(kept)} "
          f"(dropped alias={reasons['alias']} answer_len={reasons['answer_len']} "
          f"prompt_len={reasons['prompt_len']} leak={reasons['leak']}) | postcheck_leak={leak}")
    print(f"tight alias/q: min {summary['tight_alias_count']['min']} "
          f"mean {summary['tight_alias_count']['mean']} max {summary['tight_alias_count']['max']}")
    print(f"wrote {args.out}\n      {args.report}")
    assert leak == 0, "LEAK in tight panel"


if __name__ == "__main__":
    main()
