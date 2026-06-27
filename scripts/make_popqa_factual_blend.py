#!/usr/bin/env python3
"""FACT-003H: convert PopQA into a de-leaked public-QA blend set for the mixed-stream recipe.

PopQA (akariasai/PopQA, ~14.3k entity-centric short QA) maps almost 1:1 onto our FACT format:
  prompt       = "Q: {question}\nA:"
  answer       = canonical obj          (for the blended answer-only CE)
  must_contain = obj + possible_answers (for held-out scoring, alias-tolerant)

This replaces the 291 atomic facts that were too small for FACT-003G: at 5% of 2M train tokens
(=100k factual tokens) the atomic set repeated ~18x -> memorisation. PopQA at ~14k rows fills the
blend budget with essentially no repetition, so the factual data becomes a genuine low-ratio part
of the Q/A distribution instead of a flashcard deck.

Mechanism (FACT-003G, already in rt116): NOT a separate mu*loss -- the factual rows are BLENDED
into the one mixed train stream under the same answer-only CE via --factual-blend-file/-frac.

LICENSE NOTE: the PopQA HF card does not state a clear license. Use for RESEARCH / direction
validation only; do NOT claim it as product training data until the license is confirmed. Also
guardrail: a good result here is a BLEND-MECHANISM signal, not a "real factual ability" claim --
that needs a fixed, license-clean benchmark-like held-out.

USAGE (box/Colab with datasets):
  python -X utf8 scripts/make_popqa_factual_blend.py
  python -X utf8 scripts/make_popqa_factual_blend.py --max-rows 2000   # smaller smoke
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL = REPO_ROOT / "data/factual_panel_v1.jsonl"


def panel_exclude_set(path):
    """Normalized panel prompts + distinctive (>=5 char) answers, for de-leaking."""
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


def parse_answers(val):
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return [str(val)]
    for fn in (ast.literal_eval, json.loads):
        try:
            r = fn(val)
            return r if isinstance(r, list) else [r]
        except Exception:
            pass
    return [val]


def leaks(prompt, must, excl):
    norm = re.sub(r"\s+", " ", prompt + " " + " ".join(must)).strip().lower()
    return any(e in norm for e in excl)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="akariasai/PopQA")
    ap.add_argument("--split", default="test")
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = use all rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-train", type=Path, default=REPO_ROOT / "data/popqa_blend_train.jsonl")
    ap.add_argument("--out-heldout", type=Path, default=REPO_ROOT / "data/popqa_blend_heldout.jsonl")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "reports/fact003h_popqa_dataset_summary.md")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset(args.dataset, split=args.split)
    excl = panel_exclude_set(PANEL)

    rows, dropped = [], 0
    for ex in ds:
        q = (ex.get("question") or "").strip()
        obj = (ex.get("obj") or "").strip()
        if not q or not obj:
            continue
        answers = parse_answers(ex.get("possible_answers") or [])
        must = sorted({str(a).strip().lower() for a in ([obj] + answers) if str(a).strip()})
        prompt = f"Q: {q}\nA:"
        if leaks(prompt, must, excl):
            dropped += 1
            continue
        subj = ex.get("subj") or ex.get("subj_id") or q
        rows.append({"id": str(ex.get("id", len(rows))), "entity": f"popqa:{subj}",
                     "category": "popqa", "prompt": prompt, "answer": " " + obj, "must_contain": must})

    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    # split by ENTITY (stable seed-salted hash) so held-out subjects are never trained
    def order_key(r):
        h = 0
        for c in f"{args.seed}:{r['entity']}":
            h = (h * 131 + ord(c)) % (2 ** 32)
        return h
    rows.sort(key=order_key)
    n_held = max(1, int(round(len(rows) * args.heldout_frac)))
    held, train = rows[:n_held], rows[n_held:]

    def write(path, rs, as_panel):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rs:
                if as_panel:
                    rec = {"id": r["id"], "category": "popqa", "prompt": r["prompt"], "must_contain": r["must_contain"]}
                else:
                    rec = {"id": r["id"], "category": "popqa", "prompt": r["prompt"], "answer": r["answer"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write(args.out_train, train, as_panel=False)
    write(args.out_heldout, held, as_panel=True)

    # post-write leak re-check (must be 0)
    leak = sum(1 for r in (train + held) if leaks(r["prompt"], r["must_contain"], excl))

    # token-budget note: how much of a 5% blend of 2M tokens does this set cover without repeat?
    import statistics
    ans_words = [len(r["answer"].split()) for r in train]
    p_words = [len(r["prompt"].split()) for r in train]
    avg_tok = (statistics.mean(p_words) + statistics.mean(ans_words)) * 1.3 if train else 0  # ~1.3 tok/word

    lines = ["# FACT-003H PopQA factual blend dataset", "",
             f"source: {args.dataset}:{args.split} ({len(ds)} rows)  LICENSE: HF card unclear -> research only",
             "",
             f"- kept {len(rows)} (dropped {dropped} panel-leaks), entity-split -> train {len(train)} / heldout {len(held)}",
             f"- post-write leak check: {leak} (must be 0)",
             f"- approx tokens/row ~{avg_tok:.0f}; train ~{avg_tok*len(train)/1000:.0f}k tokens "
             f"(5% blend of 2M = 100k -> ~{100000/max(avg_tok*len(train),1):.2f}x of the set per blend budget; "
             f"{'NO repetition' if avg_tok*len(train) >= 100000 else 'REPEATS -> set still too small'})",
             "",
             "prompt = 'Q: {question}\\nA:'  answer = obj  must_contain = obj + possible_answers (alias-tolerant)",
             "Blended via FACT-003G --factual-blend-file/-frac (NOT a separate mu*loss).",
             "", f"- train: `{args.out_train}`", f"- heldout: `{args.out_heldout}`"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ASCII-only prints (Windows cp949 console safe -- PopQA questions contain non-ASCII names)
    print(f"PopQA rows={len(ds)} kept={len(rows)} dropped_leak={dropped} -> "
          f"train={len(train)} heldout={len(held)} | postcheck_leak={leak}")
    print(f"approx train tokens ~{avg_tok*len(train)/1000:.0f}k (5% blend budget 100k -> "
          f"{'no-repeat OK' if avg_tok*len(train) >= 100000 else 'TOO SMALL, repeats'})")
    print(f"wrote {args.out_train}\n      {args.out_heldout}\n      {args.report}")
    assert leak == 0, "LEAK DETECTED vs panel -- do not use"


if __name__ == "__main__":
    main()
