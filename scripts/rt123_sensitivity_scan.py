#!/usr/bin/env python3
"""RT-123: per-group ternarization sensitivity scan (mixed-bit DP, step 1).

The pinned bitnet.cpp llama-quantize has NO --tensor-type override, so real per-group
hybrid GGUFs need byte surgery (deferred to RT-125 for the top policies). Per the
mixed-bit plan's sanctioned fallback, RT-123 ranks groups in PyTorch instead:

  baseline = all target linears ternarized (Wq = gamma*T)  [= one-shot b1.58 PTQ]
  for each group g: restore ONLY g to FP, measure CE.
  sensitivity(g) = CE(all-ternary) - CE(g restored to FP)
                 = the MAX CE recoverable by spending more bits on group g (FP upper bound).

A real Q2_K/Q3_K upgrade recovers a FRACTION of this; the ranking is what the DP
(RT-124) needs, and RT-125 validates the chosen hybrids with real artifacts.

Groups: per layer, attn = {q,k,v,o}_proj, mlp = {gate,up,down}_proj (24 groups @ 160M).
Cost: analytical bytes(group at choice) - bytes(group at I2_S) for Q2_K/Q3_K_M/Q4_0.

USAGE:
  python scripts/rt123_sensitivity_scan.py --model-id JackFram/llama-160m \
    --json-out reports/rt123_sensitivity_160m.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402

# approximate bits/weight for cost accounting (target linears)
BITS = {"I2_S": 2.0, "Q2_K": 2.625, "Q3_K_M": 3.4375, "Q4_0": 4.5, "f16": 16.0}


def group_of(name: str):
    m = re.search(r"layers\.(\d+)\.", name)
    if not m:
        return None
    li = int(m.group(1))
    if "self_attn" in name:
        return f"blk.{li:02d}.attn"
    if "mlp" in name:
        return f"blk.{li:02d}.mlp"
    return None


@torch.no_grad()
def eval_ce(model, eval_ids, seq_len, device, max_windows=64):
    model.eval()
    n = min(eval_ids.numel() // seq_len, max_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    tot = 0.0
    cnt = 0
    for i in range(0, n, 8):
        b = ids[i : i + 8]
        tot += float(model(input_ids=b, labels=b).loss) * b.shape[0]
        cnt += b.shape[0]
    return tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--choices", nargs="+", default=["Q2_K", "Q3_K_M"])
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt123_sensitivity.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()

    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    eval_ids = torch.tensor(tok(text)["input_ids"][: args.max_eval_tokens], dtype=torch.long)

    # collect target linears, their FP weight, their ternary (gamma*T) weight, group, elems
    targets = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            g = group_of(name)
            if g is None:
                continue
            fp = mod.weight.detach().clone()
            targets[name] = {"mod": mod, "fp": fp, "tern": C.per_tensor_b158_approx(fp),
                             "group": g, "elems": fp.numel()}
    groups = defaultdict(list)
    for name, t in targets.items():
        groups[t["group"]].append(name)
    print(f"{len(targets)} target linears in {len(groups)} groups")

    def set_state(fp_groups: set):
        with torch.no_grad():
            for name, t in targets.items():
                t["mod"].weight.copy_(t["fp"] if t["group"] in fp_groups else t["tern"])

    # baselines
    set_state(set(groups))  # all FP
    ce_fp = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
    set_state(set())  # all ternary (= PTQ)
    ce_base = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
    print(f"CE_fp={ce_fp:.4f} (ppl {math.exp(ce_fp):.2f})  CE_allT={ce_base:.4f} (ppl {math.exp(ce_base):.2f})")

    # per-group sensitivity
    group_elems = {g: sum(targets[n]["elems"] for n in names) for g, names in groups.items()}
    rows = []
    for g in sorted(groups):
        set_state({g})
        ce_g = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
        sens = ce_base - ce_g  # >=0 ; CE recovered by restoring g to FP (upper bound)
        for choice in args.choices:
            cost_mb = group_elems[g] * (BITS[choice] - BITS["I2_S"]) / 8 / 1e6
            rows.append({"group": g, "choice": choice, "elems": group_elems[g],
                         "cost_mb": round(cost_mb, 4), "delta_ce_fp_upper": round(sens, 5),
                         "delta_ce_per_mb": round(sens / cost_mb, 5) if cost_mb > 0 else None})
        print(f"  {g:<12} sens(FP-restore)={sens:+.5f}")

    # rank by sensitivity (group-level) and by per-mb for the cheapest choice
    by_sens = sorted(groups, key=lambda g: -(eval_sens := next(r["delta_ce_fp_upper"] for r in rows if r["group"] == g)))
    print("\ntop groups by FP-restore sensitivity:")
    for g in by_sens[:8]:
        s = next(r["delta_ce_fp_upper"] for r in rows if r["group"] == g)
        print(f"  {g:<12} {s:+.5f}")

    out = {"model": args.model_id, "ce_fp": ce_fp, "ce_allT": ce_base,
           "ppl_fp": math.exp(ce_fp), "ppl_allT": math.exp(ce_base),
           "note": "delta_ce_fp_upper = CE recovered by restoring group to FP (UPPER bound; "
                   "real Q2_K/Q3_K recovers a fraction). Ranking input for RT-124 DP; RT-125 validates.",
           "bits_per_weight": BITS, "rows": rows}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")
    pos = sum(1 for g in groups if next(r["delta_ce_fp_upper"] for r in rows if r["group"] == g) > 0.01)
    print(f"groups with FP-restore sensitivity > 0.01 nats: {pos}/{len(groups)}")
    print("VERDICT:", "mixed-bit promising (several sensitive groups)" if pos >= 3
          else "few sensitive groups -> mixed-bit allocation may not help much")


if __name__ == "__main__":
    main()
