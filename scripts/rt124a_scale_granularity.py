#!/usr/bin/env python3
"""RT-124A: scale-granularity sweep for b1.58 conversion (quantization-aware track).

Question: did per-tensor gamma make b1.58 look worse than it really is? Screen, in
PyTorch (fast reference per the plan), how much CE the one-shot ternary PTQ recovers
when the gamma scale is per-tensor vs row / column / groupwise / blockwise. All keep
T in {-1,0,+1}; only the scale granularity changes. No training.

Materialization per granularity (W: [out,in]):
  per_tensor : one gamma = mean|W|
  row        : gamma_i = mean|W[i,:]|         (per output channel) — I2_S-incompatible
  col        : gamma_j = mean|W[:,j]|         (per input channel) — maybe foldable
  group-G    : gamma per (row, input-group of G)  (groupwise / blockwise)
For each: T = clamp(round(W/gamma), -1, 1); Wq = gamma*T. Measure CE on WikiText.

Branch: if row/group greatly improves CE -> scale-granularity bottleneck (look for
foldable/block-scale runtime). If not -> codebook/assignment/interaction (RT-124B/125).

USAGE:
  python scripts/rt124a_scale_granularity.py --model-id JackFram/llama-160m \
    --json-out reports/rt124a_scale_granularity_160m.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402


def ternarize(W: torch.Tensor, gran: str, group: int = 128) -> torch.Tensor:
    """Return Wq = gamma*T with gamma at the requested granularity, T in {-1,0,+1}."""
    if gran == "per_tensor":
        g = W.abs().mean().clamp_min(1e-8)
        return g * torch.clamp(torch.round(W / g), -1, 1)
    if gran == "row":
        g = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        return g * torch.clamp(torch.round(W / g), -1, 1)
    if gran == "col":
        g = W.abs().mean(dim=0, keepdim=True).clamp_min(1e-8)
        return g * torch.clamp(torch.round(W / g), -1, 1)
    if gran.startswith("group"):
        out, inn = W.shape
        if inn % group != 0:
            # pad to multiple of group along input dim
            pad = group - (inn % group)
            Wp = torch.cat([W, torch.zeros(out, pad, device=W.device, dtype=W.dtype)], dim=1)
        else:
            Wp = W
        o, i2 = Wp.shape
        Wr = Wp.reshape(o, i2 // group, group)
        g = Wr.abs().mean(dim=2, keepdim=True).clamp_min(1e-8)
        Wq = g * torch.clamp(torch.round(Wr / g), -1, 1)
        return Wq.reshape(o, i2)[:, :inn]
    raise ValueError(gran)


@torch.no_grad()
def eval_ce(model, eval_ids, seq_len, device, max_windows=64):
    model.eval()
    n = min(eval_ids.numel() // seq_len, max_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    tot, cnt = 0.0, 0
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
    ap.add_argument("--grans", nargs="+",
                    default=["per_tensor", "row", "col", "group128", "group64"])
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt124a_scale_granularity.json")
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

    targets = {n: m for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{n}.weight")}
    fp = {n: m.weight.detach().clone() for n, m in targets.items()}
    print(f"{len(targets)} target linears")

    ce_fp = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
    print(f"CE_fp={ce_fp:.4f} (ppl {math.exp(ce_fp):.2f})")

    rows = []
    for gran in args.grans:
        rec_num, rec_den = 0.0, 0.0
        with torch.no_grad():
            for n, m in targets.items():
                Wq = ternarize(fp[n], gran)
                rec_num += (fp[n] - Wq).abs().sum().item()
                rec_den += fp[n].abs().sum().item()
                m.weight.copy_(Wq)
        ce = eval_ce(model, eval_ids, args.seq_len, device, args.max_windows)
        # restore FP for next granularity
        with torch.no_grad():
            for n, m in targets.items():
                m.weight.copy_(fp[n])
        # scale granularity: per-tensor & blockwise are I2_S-friendly; row/col/group need runtime work
        rt = {"per_tensor": "I2_S-native", "group128": "blockwise (I2_S-block aligned, custom runtime)",
              "group64": "groupwise (custom runtime)", "row": "row-scale (quality upper bound)",
              "col": "col-scale (maybe foldable)"}.get(gran, "?")
        rows.append({"granularity": gran, "ce": round(ce, 5), "ppl": round(math.exp(ce), 3),
                     "recon_rel_l1": round(rec_num / rec_den, 5), "runtime_class": rt})
        print(f"  {gran:<11} CE={ce:.4f} ppl={math.exp(ce):.2f} recon_relL1={rec_num/rec_den:.4f} [{rt}]")

    base = next(r for r in rows if r["granularity"] == "per_tensor")["ce"]
    for r in rows:
        r["delta_ce_vs_per_tensor"] = round(base - r["ce"], 5)  # >0 = better than per-tensor
    best = min(rows, key=lambda r: r["ce"])
    out = {"model": args.model_id, "ce_fp": ce_fp, "ppl_fp": math.exp(ce_fp),
           "rows": rows, "best": best["granularity"],
           "note": "All ternary T in {-1,0,+1}; only gamma granularity varies. PyTorch screen."}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nbest CE: {best['granularity']} (CE {best['ce']}, vs per_tensor delta "
          f"{base - best['ce']:+.4f} nats)")
    big = base - best["ce"] > 0.2 and best["granularity"] != "per_tensor"
    print("VERDICT:", "SCALE-GRANULARITY BOTTLENECK — finer scale helps materially -> seek foldable/block-scale runtime"
          if big else "scale granularity does NOT rescue b1.58 much -> codebook/assignment/interaction (RT-124B/125)")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
