#!/usr/bin/env python3
"""FACT-004A 160M smoke: does lm_head unfreeze have DIRECTIONALITY before spending 1.1B Colab time?

The 1.1B FACT-004A run (content-KL lambda=0.2 + --train-lm-head) tests the hypothesis that the
FROZEN lm_head is the bottleneck keeping fact_rate at ~0.185. This is the cheap branch-killer on
the 3080 (10 GB): run the SAME recipe on a 160M LLaMA, two arms --

  B0  target linears only          (frozen lm_head)   -- the FACT-003C topology, at 160M
  B1  target linears + lm_head      (--train-lm-head)  -- the FACT-004A change, at 160M

and score the factual panel + WikiText CE in PyTorch (the materialized dir holds the ternary
weights gamma*T, so a plain HF forward reproduces the I2_S behaviour; no bitnet.cpp/GGUF needed,
which dodges the Windows build saga). 160M has little factual knowledge so ABSOLUTE fact_rate is a
floor -- we read the DELTA B1-B0 and the generation tags/CE, not the absolute number.

Verdict (relative, at 160M):
  B1 moves fact up vs B0 (or clearly improves tags/CE on answers)  => lm_head has directionality;
      the 1.1B FACT-004A run is worth its Colab time.
  B1 == B0 (both at floor, no tag/CE improvement)                  => weak signal; lower the 1.1B
      lm_head priority, lean to protected factual replay / content-AKL. (Caveat: 160M may simply
      lack the knowledge to express -- a floor null is softer evidence than a 1.1B null.)

USAGE (3080 Windows box, conda env bnt; PyTorch-only, no --bitnet):
  python scripts/fact004a_160m_smoke.py --model-id Felladrin/Llama-160M-Chat-v1 --steps 400
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# --- PyTorch panel scorer (same logic as hybrid001a_capacity_probe.py, kept identical) ---
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


def score_dir(model_dir, tok, panel, device, max_new, wt_eval, ce_windows):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
    model.config.use_cache = True
    hits, tags, raw = 0, {}, {}
    for p in panel:
        txt = generate(model, tok, p["prompt"], max_new, device)
        h = hit(txt, p["must_contain"]); hits += int(h)
        tg = tag(txt); tags[tg] = tags.get(tg, 0) + 1
        raw[p["id"]] = {"hit": h, "tag": tg, "text": txt}
    from rt116_quality_recovery import eval_ce
    ce = eval_ce(model, wt_eval, 256, device, max_windows=ce_windows)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    fr = round(hits / len(panel), 3)
    return {"fact_hit": hits, "n": len(panel), "fact_rate": fr,
            "ce": ce, "ppl": math.exp(ce), "tags": tags, "raw": raw}


def train_arm(args, arm, train_lm_head, out_dir):
    """Run rt116 for one arm (no --bitnet: train + materialize only, PyTorch scoring after)."""
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "rt116_quality_recovery.py"),
           "--model-id", args.model_id,
           "--train-source", "mixed", "--answer-loss-only",
           "--base-kl-replay", "--kl-content-only", "--kl-weight", str(args.kl_weight),
           "--exclude-panel",
           "--steps", str(args.steps), "--seq-len", str(args.seq_len),
           "--batch", str(args.batch), "--lr", str(args.lr),
           "--max-train-tokens", str(args.max_train_tokens),
           "--dtype", "float32", "--optim", "adamw",
           "--out-dir", str(out_dir),
           "--json-out", str(out_dir.parent / f"{arm}_train.json"),
           "--log-every", str(args.log_every)]
    if train_lm_head:
        cmd.append("--train-lm-head")
    print(f"\n===== arm {arm} (train_lm_head={train_lm_head}) =====\n{' '.join(cmd)}", flush=True)
    t0 = time.time()
    import os
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    subprocess.run(cmd, check=True, env=env)
    print(f"  arm {arm} trained in {(time.time()-t0)/60:.1f}m -> {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "fact004a_160m_smoke")
    ap.add_argument("--panel", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl-weight", type=float, default=0.2)
    ap.add_argument("--max-train-tokens", type=int, default=800_000)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--skip-train", action="store_true", help="re-score existing arm dirs only")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    args.work.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    panel = [json.loads(l) for l in open(args.panel) if l.strip()]
    from rt116_quality_recovery import load_wikitext
    _, wt_eval = load_wikitext(tok, 1000, args.eval_tokens)
    print(f"device={device}  model={args.model_id}  panel={len(panel)} prompts  steps={args.steps}", flush=True)

    arms = [("B0_linears", False), ("B1_linears_lmhead", True)]
    dirs = {arm: args.work / arm for arm, _ in arms}
    if not args.skip_train:
        for arm, tlh in arms:
            train_arm(args, arm, tlh, dirs[arm])

    results = {}
    # base reference (the un-adapted 160M, fp) to anchor what "knowledge floor" looks like
    print("\n===== scoring base (un-adapted reference) =====", flush=True)
    results["base"] = score_dir(args.model_id, tok, panel, device, args.max_new, wt_eval, args.ce_windows)
    print(f"  base: fact {results['base']['fact_hit']}/{results['base']['n']} "
          f"({results['base']['fact_rate']})  CE {results['base']['ce']:.3f}  tags {results['base']['tags']}", flush=True)
    for arm, _ in arms:
        print(f"\n===== scoring {arm} =====", flush=True)
        r = score_dir(dirs[arm], tok, panel, device, args.max_new, wt_eval, args.ce_windows)
        results[arm] = r
        print(f"  {arm}: fact {r['fact_hit']}/{r['n']} ({r['fact_rate']})  CE {r['ce']:.3f} "
              f"(ppl {r['ppl']:.1f})  tags {r['tags']}", flush=True)

    b0, b1 = results["B0_linears"], results["B1_linears_lmhead"]
    d_fact = round(b1["fact_rate"] - b0["fact_rate"], 3)
    d_ce = round(b1["ce"] - b0["ce"], 3)
    if d_fact > 0.0 or d_ce < -0.05:
        verdict = (f"DIRECTIONAL: lm_head unfreeze moved fact {b0['fact_rate']}->{b1['fact_rate']} "
                   f"(d={d_fact:+}) / CE {b0['ce']:.3f}->{b1['ce']:.3f} (d={d_ce:+}). "
                   f"=> the 1.1B FACT-004A run is worth its Colab time.")
    else:
        verdict = (f"FLAT: lm_head unfreeze did NOT help at 160M (fact d={d_fact:+}, CE d={d_ce:+}). "
                   f"Weak signal -> lower 1.1B lm_head priority, lean protected factual replay / "
                   f"content-AKL. CAVEAT: 160M may lack the knowledge to express (floor null is "
                   f"softer than a 1.1B null).")

    lines = ["# FACT-004A 160M smoke (lm_head unfreeze directionality)", "",
             f"model={args.model_id}  recipe=content-KL lambda={args.kl_weight}  steps={args.steps}  "
             f"panel={len(panel)} prompts  PyTorch-scored (ternary-materialized, rep-penalty 1.2)", "",
             "| arm | fact_hit | fact_rate | CE | ppl | tags |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for key in ["base", "B0_linears", "B1_linears_lmhead"]:
        r = results[key]
        lines.append(f"| {key} | {r['fact_hit']}/{r['n']} | {r['fact_rate']} | {r['ce']:.3f} | "
                     f"{r['ppl']:.1f} | {r['tags']} |")
    lines += ["", f"delta(B1-B0): fact {d_fact:+}, CE {d_ce:+}", "", "VERDICT: " + verdict]

    (args.work / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {args.work/'summary.json'} and {args.work/'summary.md'}")


if __name__ == "__main__":
    main()
