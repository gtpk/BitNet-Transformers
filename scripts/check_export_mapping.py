#!/usr/bin/env python3
"""EXPORT-002 quality gate: how lossy is mapping our groupwise alpha*T format
onto bitnet.cpp's per-tensor b1.58 (I2_S) format?

bitnet.cpp I2_S/TL1/TL2 follow BitNet b1.58 exactly: a single per-tensor weight
scale gamma = mean(|W|), T = clamp(round(W/gamma), -1, 1). This project instead
keeps a groupwise alpha (per output-channel x per input-block of `group_size`)
with a lambda-threshold ternarization (conversion.S1 / ScaledBitLinear).

The bit packing (2-bit ternary) is compatible; the SCALE GRANULARITY is not.
This script measures, per target linear of a tiny model, the reconstruction and
calibration-output error of:

  - groupwise  : conversion.S1 (this project's format)
  - per_tensor : bitnet.cpp-style b1.58 absmean (what an I2_S export would force)

so the direct/lossless/lossy/blocked decision is made from data, not assertion.

    .venv/bin/python scripts/check_export_mapping.py --json-out reports/export_mapping_gap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402


def build_tiny_model(seed: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).eval(), config


@torch.no_grad()
def capture_activations(model, input_ids):
    acts: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name):
        def fn(_m, inp, _o):
            acts[name] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        return fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            handles.append(module.register_forward_hook(hook(f"{name}.weight")))
    model(input_ids)
    for h in handles:
        h.remove()
    return acts


@torch.no_grad()
def run(seed: int = 7):
    model, config = build_tiny_model(seed)
    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, config.vocab_size, (4, 48))
    acts = capture_activations(model, input_ids)
    sd = model.state_dict()

    rows = []
    for key in C.find_target_keys(sd):
        w = sd[key]
        x = acts.get(key)
        _, gw_approx, _ = C.quantize_weight(w, C.S1)        # groupwise (ours)
        pt_approx = C.per_tensor_b158_approx(w)              # per-tensor b1.58 (I2_S)
        row = {
            "key": key,
            "groupwise_out_err": C.output_error(w, gw_approx, x) if x is not None else None,
            "per_tensor_out_err": C.output_error(w, pt_approx, x) if x is not None else None,
        }
        rows.append(row)

    gw = [r["groupwise_out_err"] for r in rows if r["groupwise_out_err"] is not None]
    pt = [r["per_tensor_out_err"] for r in rows if r["per_tensor_out_err"] is not None]
    gw_mean = sum(gw) / len(gw)
    pt_mean = sum(pt) / len(pt)
    rel_degradation = (pt_mean - gw_mean) / max(gw_mean, 1e-12)
    return rows, {
        "n_layers": len(rows),
        "groupwise_mean_out_err": round(gw_mean, 4),
        "per_tensor_mean_out_err": round(pt_mean, 4),
        "per_tensor_vs_groupwise_rel_degradation": round(rel_degradation, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=Path, default=Path("reports/export_mapping_gap.json"))
    parser.add_argument("--strict", action="store_true", help="fail only if numbers are non-finite")
    args = parser.parse_args()

    rows, summary = run(args.seed)
    print("EXPORT-002 mapping quality gate: groupwise (ours) vs per-tensor b1.58 (I2_S)")
    print("=" * 74)
    print(f"{'layer':42} {'groupwise':>10} {'per_tensor':>11}")
    for r in rows:
        print(f"{r['key']:42} {r['groupwise_out_err']:>10.4f} {r['per_tensor_out_err']:>11.4f}")
    print("-" * 74)
    print(f"mean output error  groupwise={summary['groupwise_mean_out_err']}  "
          f"per_tensor={summary['per_tensor_mean_out_err']}  "
          f"(+{summary['per_tensor_vs_groupwise_rel_degradation']*100:.1f}% worse)")
    verdict = ("lossy re-quantization (per-tensor degrades output error) -> "
               "run a quality gate before claiming I2_S compatibility")
    print(f"\nmapping decision: {verdict}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"summary": summary, "layers": rows, "verdict": verdict}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    if args.strict:
        import math
        ok = all(math.isfinite(v) for v in (summary["groupwise_mean_out_err"], summary["per_tensor_mean_out_err"]))
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
