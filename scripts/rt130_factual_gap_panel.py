#!/usr/bin/env python3
"""RT-130 / FACT-001: factual gap panel (measure, do not train).

RT-129 rescued generation usability (non-degenerate under rep-penalty/sampling) but NOT
factual quality. This measures how far the adapted b1.58/I2_S model is from FP/Q2_K on a
fixed, hand-checkable factual panel, under SANE decoding (greedy is diagnostic only).

Variants (GGUF, llama-cli): FP f16 | Q2_K | PTQ i2_s (collapse control) |
adapted f16 | adapted i2_s. Score = contains-match of any `must_contain` answer string
in the answer slot (first ~15 words after the prompt). Also tags degeneration
(ok/loop/salad/empty) + repeated-3gram + unique-token, and checks adapted i2_s vs f16.

USAGE:
  python scripts/rt130_factual_gap_panel.py --bitnet /content/bitnet.cpp \
    --adapted-dir /content/bnt_runs/tinyllama_g1_l4_s800_b4x6 \
    --refs-dir /content/bitnet.cpp/models/rt122_panel/fp \
    --ptq-gguf /content/bitnet.cpp/models/rt122_panel/ptq/ggml-model-i2_s.gguf \
    --prompt-file data/factual_panel_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

DECODES = {
    "rep1.2": ["--temp", "0", "--repeat-penalty", "1.2"],          # primary (RT-129)
    "greedy": ["--temp", "0"],                                      # diagnostic
    "t0.8p0.95": ["--temp", "0.8", "--top-p", "0.95", "--repeat-penalty", "1.1", "--seed", "0"],
}


def gen(bitnet, gguf, prompt, decode_flags, n, threads):
    cmd = [f"{bitnet}/build/bin/llama-cli", "-m", str(gguf), "-p", prompt,
           "-n", str(n), "-t", str(threads), "--simple-io"] + decode_flags
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    cont = out.split(prompt, 1)[-1] if prompt in out else out
    return " ".join(cont.split())[:300]


def answer_slot(text, n_words=15):
    # the answer is right after "A:"; take the first line / first n words
    first = text.split("\nQ:")[0].split("Q:")[0]  # stop if it starts a new Q
    return " ".join(first.split()[:n_words]).lower()


def hit(text, must_contain):
    slot = answer_slot(text)
    return any(m.lower() in slot for m in must_contain)


def tag(text):
    toks = text.split()
    if len(toks) < 2:
        return "empty"
    if re.search(r"\b(\w+)( \1){2,}\b", text):
        return "loop"
    alpha = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha < 0.6:
        return "salad"
    if len(set(toks)) / len(toks) < 0.4:
        return "repetitive"
    return "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitnet", type=Path, required=True)
    ap.add_argument("--adapted-dir", type=Path, required=True)
    ap.add_argument("--refs-dir", type=Path, required=True)
    ap.add_argument("--ptq-gguf", type=Path, required=True)
    ap.add_argument("--prompt-file", type=Path, default=Path("data/factual_panel_v1.jsonl"))
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json-out", type=Path, default=Path("reports/fact001_current_gap.json"))
    ap.add_argument("--markdown-out", type=Path, default=Path("reports/fact001_current_gap.md"))
    args = ap.parse_args()

    bn = args.bitnet.resolve()
    panel = [json.loads(l) for l in open(args.prompt_file) if l.strip()]
    print(f"{len(panel)} factual prompts")

    # (name, gguf, [decodes]) — primary rep1.2 for all; adapted i2_s also greedy + sampling
    variants = [
        ("FP f16", args.refs_dir / "ggml-model-f16.gguf", ["rep1.2"]),
        ("Q2_K", args.refs_dir / "ggml-model-q2_k.gguf", ["rep1.2"]),
        ("PTQ i2_s", args.ptq_gguf, ["rep1.2"]),
        ("adapted f16", args.adapted_dir / "ggml-model-f16.gguf", ["rep1.2"]),
        ("adapted i2_s", args.adapted_dir / "ggml-model-i2_s.gguf", ["rep1.2", "greedy", "t0.8p0.95"]),
    ]

    results = {}   # (name,decode) -> {per-prompt}
    agg = {}
    raw = {}
    for name, gguf, decs in variants:
        if not Path(gguf).exists():
            print(f"!! missing {name}: {gguf}"); continue
        for dec in decs:
            hits = 0; tags = {}
            for p in panel:
                t = gen(bn, gguf, p["prompt"], DECODES[dec], args.max_new, args.threads)
                h = hit(t, p["must_contain"]); hits += int(h)
                tg = tag(t); tags[tg] = tags.get(tg, 0) + 1
                raw[f"{name}|{dec}|{p['id']}"] = {"hit": h, "tag": tg, "text": t}
            key = f"{name}|{dec}"
            agg[key] = {"fact_hit": hits, "n": len(panel), "fact_rate": round(hits / len(panel), 3), "tags": tags}
            print(f"{name:<14} {dec:<10} fact {hits}/{len(panel)} ({hits/len(panel):.2f})  tags {tags}")

    # adapted i2_s vs f16 agreement on rep1.2
    f16 = {p["id"]: raw.get(f"adapted f16|rep1.2|{p['id']}", {}).get("hit") for p in panel}
    i2s = {p["id"]: raw.get(f"adapted i2_s|rep1.2|{p['id']}", {}).get("hit") for p in panel}
    agree = sum(1 for p in panel if f16[p["id"]] == i2s[p["id"]])
    print(f"\nadapted i2_s vs f16 (rep1.2) hit-agreement: {agree}/{len(panel)}")

    lines = [f"# RT-130 / FACT-001 factual gap panel — TinyLlama-1.1B", "",
             f"{len(panel)} prompts, rep-penalty 1.2 primary (greedy/sampling diagnostic for adapted i2_s).",
             "fact_rate = contains-match of expected answer in the answer slot. NOT a benchmark.", "",
             "| variant | decode | fact_rate | hits | tags |", "| --- | --- | ---: | ---: | --- |"]
    for k, v in agg.items():
        nm, dc = k.split("|")
        lines.append(f"| {nm} | {dc} | {v['fact_rate']} | {v['fact_hit']}/{v['n']} | {v['tags']} |")
    lines += ["", f"adapted i2_s vs f16 (rep1.2) hit-agreement: {agree}/{len(panel)}", "",
              "## sample answers (rep1.2)"]
    for p in panel[:10]:
        lines.append(f"**{p['id']}** (expect {p['must_contain']})")
        for nm in ["FP f16", "Q2_K", "adapted i2_s"]:
            r = raw.get(f"{nm}|rep1.2|{p['id']}")
            if r:
                lines.append(f"- {nm} [{'HIT' if r['hit'] else 'miss'}]: {r['text'][:120]!r}")
        lines.append("")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"agg": agg, "agreement": f"{agree}/{len(panel)}", "raw": raw}, indent=2), encoding="utf-8")
    args.markdown_out.write_text("\n".join(lines), encoding="utf-8")

    # verdict
    def rate(nm, dc="rep1.2"):
        return agg.get(f"{nm}|{dc}", {}).get("fact_rate", 0.0)
    fp, q2k, ai, ptq = rate("FP f16"), rate("Q2_K"), rate("adapted i2_s"), rate("PTQ i2_s")
    print(f"\nfact_rate  FP {fp}  Q2_K {q2k}  adapted_i2s {ai}  PTQ {ptq}")
    print("\nVERDICT:")
    if ai >= 0.8 * q2k and q2k > 0:
        print("  L1: adapted i2_s ~ Q2_K on simple facts -> 'basic factual mostly preserved'.")
    elif ai > ptq and ai < 0.8 * q2k:
        print("  Outcome B (expected): readable but factually weak vs Q2_K; runtime exonerated if i2_s~f16.")
        print("  -> proceed to FACT-002 (instruction/mixed-data adaptation).")
    else:
        print("  adapted i2_s ~ PTQ on facts -> adaptation barely added knowledge; inspect.")
    if agree >= 0.8 * len(panel):
        print(f"  i2_s vs f16 agreement {agree}/{len(panel)} -> runtime preserves factual behavior.")
    print(f"Wrote {args.json_out} and {args.markdown_out}")


if __name__ == "__main__":
    main()
