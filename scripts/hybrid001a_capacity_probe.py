#!/usr/bin/env python3
"""HYBRID-001A: late-layer capacity probe on the FACT-003C lambda=0.2 checkpoint.

FACT-003C found the objective lever (content-KL) but plateaued at fact_rate ~0.185 with
same-topology (target-linears-only, frozen lm_head/embeds) adaptation. The open question:
is the remaining gap a CAPACITY bottleneck (ternary is too few bits in certain layers) or
still an objective/data problem?

This is the cheap, decisive diagnostic: take the trained STE LATENT weights (from the rt116
Drive checkpoint, where each target linear is a PerTensorBitLinear whose .weight is the FP
latent), and selectively "restore" late layers to full precision (forward uses the FP latent
instead of gamma*T) while the rest stay ternary. We measure fact_rate / CE / degeneration in
PyTorch (no GGUF needed: adapted f16==i2_s was 27/27 in RT-134, so PyTorch ternary-forward
reproduces the I2_S fact_rate; A0 validates this). GGUF size/speed/agreement and cost
reduction (Q4->Q3->Q2->2-plane on the layers that helped) are the HYBRID-001B follow-up.

Arms (restore = run these target linears in FP instead of ternary):
  A0  all I2_S baseline            (restore none)            -> should reproduce ~0.185
  A1  last 1 block                 (layer L-1)
  A2  last 2 blocks                (layers L-2..L-1)
  A3  last 4 blocks                (layers L-4..L-1)
  A4  late attention only          (last 4 blocks, q/k/v/o)
  A5  late MLP only                (last 4 blocks, gate/up/down)

Verdict:
  fact 0.185 -> >= 0.30-0.40 on some arm  => CAPACITY bottleneck; go to HYBRID-001B cost cut.
  barely moves                            => not capacity; objective/data (lm_head unfreeze,
                                             protected factual replay, content-AKL).

USAGE (Colab x86, after FACT-003C):
  python scripts/hybrid001a_capacity_probe.py \
    --ckpt /content/drive/MyDrive/bnt_ckpt/fact003c_mixed_ckl0.2/ckpt.pt \
    --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --panel data/factual_panel_v1.jsonl \
    --json-out reports/hybrid001a_capacity_probe.json \
    --md-out reports/hybrid001a_capacity_probe.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import bitnet_llama.module as M  # noqa: E402
from bitnet_llama.module import PerTensorBitLinear  # noqa: E402
from rt116_quality_recovery import replace_targets, load_wikitext, eval_ce  # noqa: E402

# --- monkeypatch: a PerTensorBitLinear with .restore_fp=True runs in FP (latent), not ternary
_orig_forward = PerTensorBitLinear.forward


def _probe_forward(self, input):
    if getattr(self, "restore_fp", False):
        return F.linear(input, self.weight, self.bias)  # full-precision latent
    return _orig_forward(self, input)


M.PerTensorBitLinear.forward = _probe_forward


def answer_slot(text, n_words=15):
    first = text.split("\nQ:")[0].split("Q:")[0]
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


@torch.no_grad()
def generate(model, tok, prompt, max_new, device):
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    return " ".join(text.split())[:300]


def layer_block(name):
    """Return the transformer block index of a target-linear module name, or None."""
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def arm_restore_set(arm, target_names, n_layers, late=4):
    """Module names to run in FP for an arm."""
    def in_blocks(lo):
        return {n for n in target_names if (b := layer_block(n)) is not None and b >= lo}
    if arm == "A0":
        return set()
    if arm == "A1":
        return in_blocks(n_layers - 1)
    if arm == "A2":
        return in_blocks(n_layers - 2)
    if arm == "A3":
        return in_blocks(n_layers - 4)
    if arm == "A4":  # last `late` blocks, attention only
        return {n for n in in_blocks(n_layers - late) if any(p in n for p in ("q_proj", "k_proj", "v_proj", "o_proj"))}
    if arm == "A5":  # last `late` blocks, MLP only
        return {n for n in in_blocks(n_layers - late) if any(p in n for p in ("gate_proj", "up_proj", "down_proj"))}
    raise ValueError(arm)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True, help="rt116 Drive checkpoint with STE latents")
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--eval-tokens", type=int, default=60000, help="WikiText CE eval tokens")
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--arms", default="A0,A1,A2,A3,A4,A5")
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/hybrid001a_capacity_probe.json")
    ap.add_argument("--md-out", type=Path, default=REPO_ROOT / "reports/hybrid001a_capacity_probe.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    print(f"loading base {args.model_id} + replacing targets with PerTensorBitLinear ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32)
    n_lin = replace_targets(model)
    print(f"  {n_lin} target linears; loading STE latents from {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded latents (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    model.to(device).eval()
    model.config.use_cache = True

    n_layers = model.config.num_hidden_layers
    target_names = [n for n, m in model.named_modules() if isinstance(m, PerTensorBitLinear)]
    pmods = {n: m for n, m in model.named_modules() if isinstance(m, PerTensorBitLinear)}
    print(f"  {n_layers} blocks, {len(target_names)} PerTensorBitLinear modules", flush=True)

    panel = [json.loads(l) for l in open(args.panel) if l.strip()]
    print(f"  factual panel: {len(panel)} prompts", flush=True)
    wt_train, wt_eval = load_wikitext(tok, 1000, args.eval_tokens)

    fp_params = 0  # FP target-linear params in the restore set (size proxy)
    results = {}
    for arm in args.arms.split(","):
        restore = arm_restore_set(arm, target_names, n_layers)
        for n, m in pmods.items():
            m.restore_fp = (n in restore)
        rp = sum(pmods[n].weight.numel() for n in restore)
        t0 = time.time()
        hits, tags, raw = 0, {}, {}
        for i, p in enumerate(panel):
            txt = generate(model, tok, p["prompt"], args.max_new, device)
            h = hit(txt, p["must_contain"]); hits += int(h)
            tg = tag(txt); tags[tg] = tags.get(tg, 0) + 1
            raw[p["id"]] = {"hit": h, "tag": tg, "text": txt}
            if (i + 1) % 9 == 0 or i + 1 == len(panel):
                el = time.time() - t0
                print(f"    [{arm}] {i+1}/{len(panel)} prompts  elapsed {el/60:.1f}m", flush=True)
        ce = eval_ce(model, wt_eval, 256, device, max_windows=args.ce_windows)
        model.eval()  # eval_ce sets train? keep eval
        fr = round(hits / len(panel), 3)
        results[arm] = {"restore_blocks_from": None, "n_restored_linears": len(restore),
                        "restored_fp_params_M": round(rp / 1e6, 1),
                        "fact_hit": hits, "n": len(panel), "fact_rate": fr,
                        "ce": ce, "ppl": math.exp(ce), "tags": tags, "raw": raw}
        print(f"{arm}: fact {hits}/{len(panel)} ({fr})  CE {ce:.3f} (ppl {math.exp(ce):.1f})  "
              f"restored {len(restore)} linears / {rp/1e6:.0f}M FP params  tags {tags}", flush=True)

    base = results.get("A0", {}).get("fact_rate", 0.0)
    lines = ["# HYBRID-001A late-layer capacity probe (FACT-003C lambda=0.2 latents)", "",
             f"model={args.model_id}  ckpt={args.ckpt}  panel={len(panel)} prompts",
             "Restore = run those target linears in FP (latent) instead of ternary; rest stay b1.58.",
             "PyTorch fact_rate (rep-penalty 1.2, contains-match); A0 reproduces the I2_S baseline.", "",
             "| arm | restore | FP params | fact_rate | CE | ppl | tags |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    arm_desc = {"A0": "none (all I2_S)", "A1": "last 1 block", "A2": "last 2 blocks",
                "A3": "last 4 blocks", "A4": "last 4 attn only", "A5": "last 4 MLP only"}
    for arm in args.arms.split(","):
        r = results[arm]
        lines.append(f"| {arm} {arm_desc.get(arm,'')} | {r['n_restored_linears']} lin | "
                     f"{r['restored_fp_params_M']}M | {r['fact_rate']} | {r['ce']:.3f} | {r['ppl']:.1f} | {r['tags']} |")
    best = max(results.items(), key=lambda kv: kv[1]["fact_rate"])
    lines += ["", f"A0 baseline fact_rate = {base}; best arm = {best[0]} @ {best[1]['fact_rate']}.",
              "", "VERDICT: " + ("CAPACITY bottleneck (some arm >= 0.30) -> HYBRID-001B cost-cut "
              "(Q4->Q3->Q2->2-plane on the helpful layers)." if best[1]["fact_rate"] >= 0.30 else
              "facts barely move -> NOT mainly capacity; go to objective/data (lm_head unfreeze, "
              "protected factual replay, content-AKL).")]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines[-4:]))
    print(f"\nWrote {args.json_out} and {args.md_out}")


if __name__ == "__main__":
    main()
