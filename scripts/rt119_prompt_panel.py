#!/usr/bin/env python3
"""RT-119 / QR-004: prompt-quality panel — does the adapted b1.58 model SOUND recovered?

CE/PPL (RT-116) say quality recovers; this shows it to a human. Greedy-decode the same
prompts from four variants and save them side by side:

  FP            : original full-precision model
  PTQ           : Wq=gamma*T, NO adaptation (the collapse)
  adapted       : Wq=gamma*T after short teacher-free CE (linears-only, the QR-005 recipe)
  adapted i2_s  : the adapted model through the real bitnet.cpp I2_S runtime (with --bitnet)
  adapted f16   : the adapted model as F16 GGUF (runtime control, with --bitnet)

Key comparisons: adapted vs PTQ (did it recover), adapted vs FP (how close), and
adapted i2_s vs adapted f16 (runtime preserves it). Heuristic failure tags flag
degenerate/repetitive output. Llama-160m is a small BASE LM, so prompts are
COMPLETIONS, not instructions.

USAGE:
  python scripts/rt119_prompt_panel.py --model-id JackFram/llama-160m --steps 300 \
    --bitnet /content/bitnet.cpp --out reports/rt119_panel.md
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
from bitnet_llama.module import PerTensorBitLinear  # noqa: E402

PROMPTS = [
    "The history of science begins with",
    "In the early morning, the city was",
    "Water boils at a temperature of",
    "The capital of France is",
    "Once upon a time, there was a small",
    "The most important rule of cooking is",
    "Artificial intelligence is a field that",
    "She opened the door and saw",
    "The economy of a country depends on",
    "To solve this problem, we first need to",
    "The sun rises in the east and sets in the",
    "A computer program is a set of",
]


def set_module(root, name, mod):
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, mod)


def replace_targets(model):
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            repl = PerTensorBitLinear(m.in_features, m.out_features, bias=m.bias is not None)
            with torch.no_grad():
                repl.weight.copy_(m.weight)
                if m.bias is not None:
                    repl.bias.copy_(m.bias)
            set_module(model, name, repl)
            n += 1
    return n


@torch.no_grad()
def gen(model, tok, prompt, device, max_new=40):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def tags(text):
    """Heuristic failure flags for a completion."""
    toks = text.split()
    flags = []
    if len(toks) < 3:
        flags.append("empty")
        return flags
    uniq = len(set(toks)) / len(toks)
    if uniq < 0.4:
        flags.append("repetitive")
    # immediate token loops e.g. "the the the"
    if re.search(r"\b(\w+)( \1){2,}\b", text):
        flags.append("loop")
    if not re.search(r"[a-zA-Z]", text):
        flags.append("non-text")
    return flags or ["ok"]


def llama_gen(bitnet, gguf, prompt, n=40):
    cmd = [f"{bitnet}/build/bin/llama-cli", "-m", str(gguf), "-p", prompt,
           "-n", str(n), "-t", "2", "--temp", "0", "-no-cnv", "-st"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout
    # llama-cli echoes the prompt then the continuation; strip the prompt prefix
    return out.split(prompt, 1)[-1].strip()[:400] if prompt in out else out.strip()[-400:]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--bitnet", type=Path, default=None, help="also generate via adapted f16/i2_s GGUF")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports/rt119_panel.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)

    fp = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()
    print("generating FP ...")
    gen_fp = {p: gen(fp, tok, p, device, args.max_new) for p in PROMPTS}

    # PTQ (no training)
    import copy
    ptq = copy.deepcopy(fp)
    replace_targets(ptq)
    ptq.to(device).eval()
    print("generating PTQ ...")
    gen_ptq = {p: gen(ptq, tok, p, device, args.max_new) for p in PROMPTS}

    # adapted (linears-only short CE, the QR-005 default recipe)
    adapted = copy.deepcopy(fp)
    replace_targets(adapted)
    adapted.to(device)
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_ids = torch.tensor(tok("\n\n".join(t for t in ds["train"]["text"] if t.strip()))["input_ids"][:600_000])
    for p in adapted.parameters():
        p.requires_grad_(False)
    tparams = [m.weight for m in adapted.modules() if isinstance(m, PerTensorBitLinear)]
    for p in tparams:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(tparams, lr=args.lr)
    g = torch.Generator().manual_seed(0)
    usable = train_ids.numel() - 1
    adapted.train()
    print(f"adapting {sum(p.numel() for p in tparams)/1e6:.1f}M params, {args.steps} steps ...")
    for step in range(args.steps):
        starts = torch.randint(0, max(1, usable - args.seq_len), (args.batch,), generator=g)
        x = torch.stack([train_ids[s:s + args.seq_len] for s in starts.tolist()]).to(device)
        loss = adapted(input_ids=x, labels=x).loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    adapted.eval()
    print("generating adapted ...")
    gen_ad = {p: gen(adapted, tok, p, device, args.max_new) for p in PROMPTS}

    cols = [("FP", gen_fp), ("PTQ", gen_ptq), ("adapted", gen_ad)]

    # optional GGUF generation (adapted f16 / i2_s) via llama-cli
    if args.bitnet:
        from huggingface_hub import hf_hub_download  # noqa
        dd = (REPO_ROOT / "reports/rt119_adapted_gguf"); dd.mkdir(parents=True, exist_ok=True)
        # materialize gamma*T dense and export
        m2 = copy.deepcopy(adapted).to("cpu").float()
        for name, mod in list(m2.named_modules()):
            if isinstance(mod, PerTensorBitLinear):
                dense = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
                with torch.no_grad():
                    dense.weight.copy_(C.per_tensor_b158_approx(mod.weight))
                    if mod.bias is not None:
                        dense.bias.copy_(mod.bias)
                set_module(m2, name, dense)
        m2.save_pretrained(dd, safe_serialization=True); tok.save_pretrained(dd)
        bn = args.bitnet.resolve(); conv = bn / "utils/convert-hf-to-gguf-bitnet.py"
        subprocess.run(f'python "{conv}" "{dd}" --outtype f16', shell=True, capture_output=True)
        subprocess.run(f'python "{conv}" "{dd}" --outtype f32', shell=True, capture_output=True)
        subprocess.run(f'"{bn}/build/bin/llama-quantize" --token-embedding-type f16 --output-tensor-type f16 '
                       f'"{dd}/ggml-model-f32.gguf" "{dd}/ggml-model-i2_s.gguf" I2_S 1 1', shell=True, capture_output=True)
        print("generating adapted f16 / i2_s via llama-cli ...")
        gen_f16 = {p: llama_gen(bn, dd / "ggml-model-f16.gguf", p, args.max_new) for p in PROMPTS}
        gen_i2s = {p: llama_gen(bn, dd / "ggml-model-i2_s.gguf", p, args.max_new) for p in PROMPTS}
        cols += [("adapted f16", gen_f16), ("adapted i2_s", gen_i2s)]

    # write markdown panel
    lines = [f"# RT-119 / QR-004 prompt-quality panel — {args.model_id}", "",
             f"Greedy decode, {args.max_new} new tokens. PTQ = Wq=gamma*T no train; "
             f"adapted = +{args.steps}-step teacher-free CE (linears-only).", ""]
    tagsum = {name: {} for name, _ in cols}
    for p in PROMPTS:
        lines.append(f"### `{p}`")
        for name, d in cols:
            t = tags(d[p])
            for tg in t:
                tagsum[name][tg] = tagsum[name].get(tg, 0) + 1
            lines.append(f"- **{name}** {t}: {d[p].strip()[:300]!r}")
        lines.append("")
    lines.append("## failure-tag summary (count over prompts)")
    for name, _ in cols:
        lines.append(f"- {name}: {tagsum[name]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    (args.out.with_suffix(".json")).write_text(json.dumps(
        {"model": args.model_id, "steps": args.steps,
         "generations": {name: d for name, d in cols}, "tag_summary": tagsum}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print("tag summary:", {n: tagsum[n] for n, _ in cols})


if __name__ == "__main__":
    main()
