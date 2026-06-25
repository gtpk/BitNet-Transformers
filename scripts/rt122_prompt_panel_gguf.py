#!/usr/bin/env python3
"""RT-122: GGUF prompt-quality panel — does the 1.1B adapted ternary model stay readable?

G5/RT-121 showed OURS does not win PPL-per-bit vs Q2_K, so the surviving claim is
"smallest + fastest, still usable". RT-122 stress-tests "still usable" on the
budget-scaled 1.1B (RT-120) by GREEDY-decoding the same prompts from five GGUF
variants through ONE tool (llama-cli), with heuristic failure tags:

  FP (f16)        | Q2_K (one-shot) | PTQ ternary (no train) |
  OURS adapted f16 | OURS adapted i2_s

Key reads: OURS vs PTQ (recovery survives), OURS vs Q2_K (lower quality but not
broken), OURS i2_s vs OURS f16 (runtime preserves it).

The OURS f16/i2_s GGUFs are reused from the RT-120 adapted out-dir if present, else
built from its HF checkpoint. FP/Q2_K/PTQ are built here. No training.

USAGE (x86 with bitnet.cpp built):
  python scripts/rt122_prompt_panel_gguf.py --bitnet /content/bitnet.cpp \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --adapted-dir /content/bnt_runs/tinyllama_g1_l4_s800_b4x6 \
    --out reports/rt122_panel_1p1b.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402

PROMPTS = [
    "The history of science begins with",
    "Water boils at a temperature of",
    "The capital of France is",
    "Once upon a time, there was a small",
    "The most important rule of cooking is",
    "Artificial intelligence is a field that",
    "The economy of a country depends on",
    "The sun rises in the east and sets in the",
    "A computer program is a set of",
    "In 1969, the first humans landed on the",
    "Photosynthesis is the process by which plants",
    "The three primary colors are",
    "She opened the door and saw",
    "The internet is a global network that",
    "To bake bread you need flour, water, and",
    "The largest planet in our solar system is",
    "A good leader should always",
    "The French Revolution began in the year",
    "Music is often described as the language of",
    "Climate change is caused mainly by",
]


def set_module(root, name, mod):
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, mod)


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-400:])
    return r.returncode


def materialize_ptq(fp_dir, ptq_dir):
    if (ptq_dir / "config.json").exists():
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained(fp_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        for name, mod in m.named_modules():
            if isinstance(mod, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
                mod.weight.copy_(C.per_tensor_b158_approx(mod.weight))
    ptq_dir.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(ptq_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(fp_dir).save_pretrained(ptq_dir)


def llama_gen(bitnet, gguf, prompt, n, threads):
    cmd = [f"{bitnet}/build/bin/llama-cli", "-m", str(gguf), "-p", prompt,
           "-n", str(n), "-t", str(threads), "--temp", "0", "--simple-io"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    cont = out.split(prompt, 1)[-1] if prompt in out else out
    return " ".join(cont.split())[:400]


def tags(text):
    toks = text.split()
    if len(toks) < 3:
        return ["empty"]
    fl = []
    if len(set(toks)) / len(toks) < 0.4:
        fl.append("repetitive")
    if re.search(r"\b(\w+)( \1){2,}\b", text):
        fl.append("loop")
    alpha = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha < 0.65:
        fl.append("salad")
    return fl or ["ok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitnet", type=Path, required=True)
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--adapted-dir", type=Path, required=True,
                    help="RT-120 adapted out-dir (HF checkpoint + maybe f16/i2_s GGUFs)")
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports/rt122_panel_1p1b.md")
    args = ap.parse_args()

    bn = args.bitnet.resolve()
    conv = bn / "utils/convert-hf-to-gguf-bitnet.py"
    quant = bn / "build/bin/llama-quantize"
    work = (args.work or bn / "models/rt122_panel").resolve()
    work.mkdir(parents=True, exist_ok=True)
    ad = args.adapted_dir.resolve()

    # FP
    fp_dir = work / "fp"
    if not (fp_dir / "config.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(args.model_id, local_dir=str(fp_dir))
    fp_f16 = fp_dir / "ggml-model-f16.gguf"
    fp_f32 = fp_dir / "ggml-model-f32.gguf"
    if not fp_f16.exists():
        run(f'python "{conv}" "{fp_dir}" --outtype f16')
    if not fp_f32.exists():
        run(f'python "{conv}" "{fp_dir}" --outtype f32')
    # Q2_K
    q2k = fp_dir / "ggml-model-q2_k.gguf"
    if not q2k.exists():
        run(f'"{quant}" --token-embedding-type f16 --output-tensor-type f16 "{fp_f32}" "{q2k}" Q2_K')
    # PTQ ternary
    ptq_dir = work / "ptq"
    materialize_ptq(fp_dir, ptq_dir)
    ptq_f32 = ptq_dir / "ggml-model-f32.gguf"
    if not ptq_f32.exists():
        run(f'python "{conv}" "{ptq_dir}" --outtype f32')
    ptq_i2s = ptq_dir / "ggml-model-i2_s.gguf"
    if not ptq_i2s.exists():
        run(f'"{quant}" --token-embedding-type f16 --output-tensor-type f16 "{ptq_f32}" "{ptq_i2s}" I2_S 1 1')
    # OURS (reuse RT-120 GGUFs, else build from adapted HF)
    ours_f16 = ad / "ggml-model-f16.gguf"
    ours_i2s = ad / "ggml-model-i2_s.gguf"
    if not ours_f16.exists():
        run(f'python "{conv}" "{ad}" --outtype f16')
    if not ours_i2s.exists():
        ad_f32 = ad / "ggml-model-f32.gguf"
        if not ad_f32.exists():
            run(f'python "{conv}" "{ad}" --outtype f32')
        run(f'"{quant}" --token-embedding-type f16 --output-tensor-type f16 "{ad_f32}" "{ours_i2s}" I2_S 1 1')

    variants = [("FP f16", fp_f16), ("Q2_K", q2k), ("PTQ ternary", ptq_i2s),
                ("OURS adapted f16", ours_f16), ("OURS adapted i2_s", ours_i2s)]
    print("variants:", [(n, g.exists()) for n, g in variants])

    gens = {n: {} for n, _ in variants}
    tagsum = {n: {} for n, _ in variants}
    for name, gguf in variants:
        for p in PROMPTS:
            t = llama_gen(bn, gguf, p, args.max_new, args.threads)
            gens[name][p] = t
            for tg in tags(t):
                tagsum[name][tg] = tagsum[name].get(tg, 0) + 1
        print(f"{name}: {tagsum[name]}")

    lines = [f"# RT-122 GGUF prompt panel — {args.model_id} (RT-120 adapted)", "",
             f"Greedy ({args.max_new} new tokens), one llama-cli, --temp 0. NOTE: OURS was "
             f"CE-adapted on WikiText, so its style drifts toward WikiText vs the FP chat model; "
             f"the test is readability/non-collapse, not instruction-following.", ""]
    for p in PROMPTS:
        lines.append(f"### `{p}`")
        for name, _ in variants:
            lines.append(f"- **{name}** {tags(gens[name][p])}: {gens[name][p][:280]!r}")
        lines.append("")
    lines.append("## failure-tag summary")
    for name, _ in variants:
        lines.append(f"- {name}: {tagsum[name]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"model": args.model_id, "generations": gens, "tag_summary": tagsum}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print("tag summary:", tagsum)


if __name__ == "__main__":
    main()
