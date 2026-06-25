#!/usr/bin/env python3
"""RT-129: decoding stability probe — is the 1.1B adapted degeneration a MODEL or a
GREEDY-DECODING problem?

RT-122 showed all-I2_S 1.1B adapted output degenerates ("= = =" loops) under GREEDY.
Two explanations need different fixes:
  A. the model wrecked its distribution        -> data/objective problem
  B. greedy exaggerated a weak model's repeat attractor -> decoding fixes some usability

This probe re-decodes the SAME prompts under several decoding configs and measures
loop/repeat/diversity, all via one llama-cli. No training.

Subjects (full decode sweep): adapted f16, adapted i2_s.
References (greedy only): FP f16, Q2_K (upper), PTQ ternary i2_s (collapse).

Decodes: greedy | rep1.1 | rep1.2 | temp0.7/top_p0.9 | temp0.8/top_p0.95.

USAGE:
  python scripts/rt129_decoding_probe.py --bitnet /content/bitnet.cpp \
    --adapted-dir /content/bnt_runs/tinyllama_g1_l4_s800_b4x6 \
    --refs-dir /content/bitnet.cpp/models/rt122_panel/fp \
    --ptq-gguf /content/bitnet.cpp/models/rt122_panel/ptq/ggml-model-i2_s.gguf
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

PROMPTS = [
    "The history of science begins with", "Water boils at a temperature of",
    "The capital of France is", "Once upon a time, there was a small",
    "The most important rule of cooking is", "Artificial intelligence is a field that",
    "The economy of a country depends on", "A computer program is a set of",
    "In 1969, the first humans landed on the", "Photosynthesis is the process by which plants",
    "The largest planet in our solar system is", "Climate change is caused mainly by",
]

DECODES = {
    "greedy": ["--temp", "0"],
    "rep1.1": ["--temp", "0", "--repeat-penalty", "1.1"],
    "rep1.2": ["--temp", "0", "--repeat-penalty", "1.2"],
    "t0.7p0.9": ["--temp", "0.7", "--top-p", "0.9", "--seed", "0"],
    "t0.8p0.95": ["--temp", "0.8", "--top-p", "0.95", "--seed", "0"],
}


def gen(bitnet, gguf, prompt, decode_flags, n, threads):
    cmd = [f"{bitnet}/build/bin/llama-cli", "-m", str(gguf), "-p", prompt,
           "-n", str(n), "-t", str(threads), "--simple-io"] + decode_flags
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    cont = out.split(prompt, 1)[-1] if prompt in out else out
    return " ".join(cont.split())[:400]


def metrics(text):
    toks = text.split()
    if len(toks) < 3:
        return {"tag": "empty", "uniq": 0.0, "rep3": 1.0}
    uniq = len(set(toks)) / len(toks)
    tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
    rep3 = 1.0 - (len(set(tri)) / max(len(tri), 1)) if tri else 0.0
    alpha = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if re.search(r"\b(\w+)( \1){2,}\b", text):
        tag = "loop"
    elif alpha < 0.65:
        tag = "salad"
    elif uniq < 0.4:
        tag = "repetitive"
    else:
        tag = "ok"
    return {"tag": tag, "uniq": round(uniq, 3), "rep3": round(rep3, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitnet", type=Path, required=True)
    ap.add_argument("--adapted-dir", type=Path, required=True)
    ap.add_argument("--refs-dir", type=Path, required=True, help="dir with FP ggml-model-f16.gguf + ggml-model-q2_k.gguf")
    ap.add_argument("--ptq-gguf", type=Path, required=True)
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("reports/rt129_decoding_probe.md"))
    args = ap.parse_args()

    bn = args.bitnet.resolve()
    prompts = PROMPTS[: args.n_prompts]
    subjects = [("adapted f16", args.adapted_dir / "ggml-model-f16.gguf", list(DECODES)),
                ("adapted i2_s", args.adapted_dir / "ggml-model-i2_s.gguf", list(DECODES))]
    refs = [("FP f16", args.refs_dir / "ggml-model-f16.gguf", ["greedy"]),
            ("Q2_K", args.refs_dir / "ggml-model-q2_k.gguf", ["greedy"]),
            ("PTQ i2_s", args.ptq_gguf, ["greedy"])]
    rows = []
    gens = {}
    for name, gguf, decs in subjects + refs:
        if not Path(gguf).exists():
            print(f"!! missing {name}: {gguf}"); continue
        for dec in decs:
            tags = Counter(); uniqs = []; reps = []
            for p in prompts:
                t = gen(bn, gguf, p, DECODES[dec], args.max_new, args.threads)
                m = metrics(t)
                tags[m["tag"]] += 1; uniqs.append(m["uniq"]); reps.append(m["rep3"])
                gens[f"{name}|{dec}|{p}"] = t
            row = {"model": name, "decode": dec, "ok": tags["ok"], "repetitive": tags["repetitive"],
                   "loop": tags["loop"], "salad": tags["salad"], "empty": tags["empty"],
                   "uniq": round(sum(uniqs) / len(uniqs), 3), "rep3": round(sum(reps) / len(reps), 3)}
            rows.append(row)
            print(f"{name:<14} {dec:<10} ok{tags['ok']} loop{tags['loop']} rep{tags['repetitive']} "
                  f"salad{tags['salad']} | uniq {row['uniq']} rep3 {row['rep3']}")

    lines = [f"# RT-129 decoding stability probe (TinyLlama-1.1B adapted)", "",
             f"{len(prompts)} prompts, {args.max_new} new tokens, llama-cli. loop/salad/empty = degenerate.", "",
             "| model | decode | ok | rep | loop | salad | empty | uniq | rep3gram |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['decode']} | {r['ok']} | {r['repetitive']} | {r['loop']} | "
                     f"{r['salad']} | {r['empty']} | {r['uniq']} | {r['rep3']} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({"rows": rows, "generations": gens}, indent=2), encoding="utf-8")

    # verdict: does any non-greedy decode move adapted i2_s toward readable?
    def get(model, dec):
        return next((r for r in rows if r["model"] == model and r["decode"] == dec), None)
    ai_greedy = get("adapted i2_s", "greedy")
    best_ai = max([r for r in rows if r["model"] == "adapted i2_s"], key=lambda r: r["ok"])
    print(f"\nadapted i2_s greedy: ok{ai_greedy['ok']} loop{ai_greedy['loop']}  ->  best decode "
          f"'{best_ai['decode']}': ok{best_ai['ok']} loop{best_ai['loop']}")
    if best_ai["ok"] >= max(3, ai_greedy["ok"] + 3):
        print("VERDICT: (B) DECODING — better decoding materially reduces degeneration; adapted is "
              "'weak under greedy, usable-ish under sampling/penalty'. Adjust the claim + report decode.")
    elif best_ai["ok"] > ai_greedy["ok"]:
        print("VERDICT: PARTIAL — decoding helps somewhat but does not reach a readable tier -> mostly (A) model/data.")
    else:
        print("VERDICT: (A) MODEL/DATA — no decoding config rescues it; the distribution is damaged -> "
              "longer adaptation / better data / repetition-aware objective.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
