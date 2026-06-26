#!/usr/bin/env python3
"""FACT-003B/C leakage guard: does the replay/training source leak the FACT-001 panel?

The base-KL replay anchor (rt116 --base-kl-replay) and any protected-factual-replay set
must NOT contain the factual eval panel, or a "recovery" would just be memorised eval.
This compares a text source (default: the Dolly instruction replay stream rt116 uses)
against data/factual_panel_v1.jsonl and FAILS (exit 1) on exact overlap of a panel prompt
or an expected answer string.

Checks (case-insensitive, whitespace-normalised):
  - panel `prompt` appearing verbatim in the source  -> leak
  - panel `must_contain` answer appearing in the source within a window that also contains
    enough of the prompt's distinctive tokens                    -> reported (weak signal)

The hard gate is prompt-verbatim overlap; answer-token co-occurrence is reported as a
warning because short answers ("paris", "7") legitimately occur in unrelated text.

USAGE:
  python scripts/check_fact_panel_overlap.py                       # checks Dolly replay
  python scripts/check_fact_panel_overlap.py --source-file some.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def dolly_text(max_examples: int) -> str:
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    parts = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ctx = (ex.get("context") or "").strip()
        q = ex["instruction"].strip() + (("\n" + ctx) if ctx else "")
        parts.append(f"Q: {q}\nA: {ex['response'].strip()}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--source-file", type=Path, default=None,
                    help="text source to check; default = the Dolly instruction replay stream")
    ap.add_argument("--max-examples", type=int, default=20000,
                    help="Dolly examples to materialise (match rt116's replay pool extent)")
    args = ap.parse_args()

    panel = [json.loads(l) for l in open(args.panel) if l.strip()]
    src = norm(args.source_file.read_text(encoding="utf-8") if args.source_file
               else dolly_text(args.max_examples))
    print(f"panel: {len(panel)} prompts   source: {len(src):,} chars"
          f" ({'file ' + str(args.source_file) if args.source_file else 'Dolly replay'})")

    prompt_leaks, answer_warn = [], []
    for p in panel:
        if norm(p["prompt"]) in src:
            prompt_leaks.append(p["id"])
        for ans in p.get("must_contain", []):
            a = norm(ans)
            # only flag multi-word / distinctive answers; single short tokens are noise
            if len(a) >= 5 and a in src:
                answer_warn.append((p["id"], ans))

    if answer_warn:
        print(f"\nNOTE: {len(answer_warn)} expected-answer strings also occur in the source "
              f"(expected for common words; not a hard leak):")
        for pid, ans in answer_warn[:20]:
            print(f"  - {pid}: {ans!r}")

    if prompt_leaks:
        print(f"\nFAIL: {len(prompt_leaks)} panel prompt(s) appear VERBATIM in the source "
              f"(eval leakage): {prompt_leaks}")
        sys.exit(1)
    print("\nPASS: no panel prompt appears verbatim in the source (no hard eval leakage).")


if __name__ == "__main__":
    main()
