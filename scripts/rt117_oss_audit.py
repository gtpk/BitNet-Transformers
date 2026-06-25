#!/usr/bin/env python3
"""RT-117A / OSS-001: gpt-oss-20b metadata-only architecture audit (NO weight download).

The LLaMA-family story is closed (systems RT-112..115 + quality RT-116/TRAIN-002).
gpt-oss is the new question: does the per-tensor b1.58 -> I2_S recipe apply to a real
public MoE model? Step one is a CHEAP, SAFE audit — fetch only config.json, the
safetensors index, and per-tensor metadata (dtype/shape via HTTP range reads); do
NOT download the multi-GB weights. Then classify tensors, project storage/traffic,
and risk-rate (direct / adapt / blocked) before any conversion.

USAGE:
  python scripts/rt117_oss_audit.py --model-id openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

DTYPE_BITS = {"F64": 64, "F32": 32, "F16": 16, "BF16": 16, "F8_E4M3": 8, "F8_E5M2": 8,
              "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8, "BOOL": 8,
              "F4": 4, "MXFP4": 4, "U4": 4}

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "out_proj", "wqkv", "wo")
EXPERT = ("experts", "expert", "w1", "w2", "w3", "gate_proj", "up_proj", "down_proj", "mlp1", "mlp2")
ROUTER = ("router", "gate.weight", "gate_up", "routed", "gate.bias")
KEEP = ("embed", "wte", "lm_head", "norm", "ln", "bias", "rotary", "pos")


def template(name):
    """Collapse layer/expert indices so distinct tensor *kinds* are visible."""
    t = re.sub(r"\.\d+\.", ".N.", name)
    t = re.sub(r"\.\d+\b", ".N", t)
    return t


def classify(name):
    n = name.lower()
    if any(k in n for k in ("embed", "wte", "lm_head")):
        return "embed/lm_head"
    if "norm" in n or re.search(r"\bln", n):
        return "norm"
    if any(k in n for k in ROUTER) and "experts" not in n:
        return "router"
    if any(k in n for k in EXPERT):
        return "expert-ffn"
    if any(k in n for k in ATTN):
        return "attn"
    if n.endswith(".bias"):
        return "bias"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="openai/gpt-oss-20b")
    ap.add_argument("--json-out", default="reports/rt117_oss_audit.json")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    cfg = json.loads(open(hf_hub_download(args.model_id, "config.json")).read())
    print("=" * 64)
    print(f"RT-117A audit: {args.model_id}")
    print("=" * 64)
    print("architectures:", cfg.get("architectures"))
    keys = ["model_type", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "intermediate_size",
            "num_local_experts", "num_experts", "num_experts_per_tok", "n_routed_experts",
            "expert_dim", "moe_intermediate_size", "sliding_window", "tie_word_embeddings"]
    for k in keys:
        if k in cfg:
            print(f"  {k} = {cfg[k]}")
    if "quantization_config" in cfg:
        print("  quantization_config =", json.dumps(cfg["quantization_config"])[:300])
    print("  (full config keys:", sorted(cfg.keys()), ")")

    # per-tensor metadata (dtype/shape) via range reads — NO weight download
    try:
        from huggingface_hub import get_safetensors_metadata
        meta = get_safetensors_metadata(args.model_id)
        tensors = {}
        for fm in meta.files_metadata.values():
            for name, ti in fm.tensors.items():
                tensors[name] = (ti.dtype, tuple(ti.shape))
        src = "get_safetensors_metadata"
    except Exception as e:
        print(f"\n(get_safetensors_metadata unavailable: {e}; falling back to index names only)")
        idx = json.loads(open(hf_hub_download(args.model_id, "model.safetensors.index.json")).read())
        tensors = {name: (None, None) for name in idx["weight_map"]}
        src = "index.json (names only)"

    print(f"\ntensor metadata source: {src}  |  {len(tensors)} tensors")

    # distinct templates with dtype + shape + count
    tmpl = defaultdict(lambda: {"count": 0, "dtype": set(), "shape": None})
    for name, (dt, sh) in tensors.items():
        t = tmpl[template(name)]
        t["count"] += 1
        if dt:
            t["dtype"].add(dt)
        if sh and t["shape"] is None:
            t["shape"] = sh
    print("\n--- distinct tensor templates ---")
    for t in sorted(tmpl):
        info = tmpl[t]
        print(f"  [{classify(t):<13}] {t:<48} x{info['count']:<4} "
              f"dtype={sorted(info['dtype']) or '?'} shape={info['shape']}")

    # byte accounting by class (only if shapes known)
    have_shapes = any(sh for _, sh in tensors.values())
    cls_bytes = defaultdict(int)
    cls_elems = defaultdict(int)
    if have_shapes:
        for name, (dt, sh) in tensors.items():
            if not sh:
                continue
            numel = 1
            for d in sh:
                numel *= d
            bits = DTYPE_BITS.get(dt, 16)
            cls = classify(name)
            cls_bytes[cls] += numel * bits // 8
            cls_elems[cls] += numel
        tot = sum(cls_bytes.values())
        print("\n--- storage by class (current dtypes) ---")
        for c in sorted(cls_bytes, key=lambda x: -cls_bytes[x]):
            print(f"  {c:<14} {cls_elems[c]/1e6:>9.1f}M elems  {cls_bytes[c]/1e6:>9.1f} MB")
        print(f"  {'TOTAL':<14} {sum(cls_elems.values())/1e6:>9.1f}M elems  {tot/1e6:>9.1f} MB")

        # I2_S projection: target = attn + expert-ffn linears -> 2 bits/elem
        tgt = cls_elems["attn"] + cls_elems["expert-ffn"]
        i2s_tgt = tgt // 4  # 2-bit
        floor = cls_bytes["embed/lm_head"] + cls_bytes["norm"] + cls_bytes["router"] + cls_bytes["bias"]
        proj_whole = floor + i2s_tgt
        print("\n--- I2_S storage projection (attn+expert linears -> 2-bit, rest f16 floor) ---")
        print(f"  current total      : {tot/1e6:.1f} MB")
        print(f"  target-linear now  : {cls_bytes['attn']+cls_bytes['expert-ffn']:.0f} B "
              f"({(cls_bytes['attn']+cls_bytes['expert-ffn'])/1e6:.1f} MB)")
        print(f"  target-linear i2_s : {i2s_tgt/1e6:.1f} MB")
        print(f"  non-compressible floor (embed+norm+router+bias f16): {floor/1e6:.1f} MB")
        print(f"  projected whole    : {proj_whole/1e6:.1f} MB  (ratio {proj_whole/max(tot,1):.3f})")

        # active-expert per-token weight traffic (memory-traffic-first metric)
        n_exp = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts")
        n_act = cfg.get("num_experts_per_tok")
        if n_exp and n_act and cls_elems["expert-ffn"]:
            exp_per = cls_elems["expert-ffn"] / n_exp
            for label, per_elem_bytes in [("current", None), ("i2_s", 0.25)]:
                if per_elem_bytes is None:
                    attn_b = cls_bytes["attn"]; exp_b = exp_per * n_act * (cls_bytes["expert-ffn"]/max(cls_elems["expert-ffn"],1))
                else:
                    attn_b = cls_elems["attn"] * per_elem_bytes; exp_b = exp_per * n_act * per_elem_bytes
                print(f"  per-token weight traffic ({label}): attn {attn_b/1e6:.1f}MB + "
                      f"{n_act}/{n_exp} experts {exp_b/1e6:.1f}MB = {(attn_b+exp_b)/1e6:.1f} MB/token")

    out = {"model_id": args.model_id, "architectures": cfg.get("architectures"),
           "config_subset": {k: cfg[k] for k in keys if k in cfg},
           "has_quant_config": "quantization_config" in cfg,
           "n_tensors": len(tensors), "metadata_source": src,
           "templates": {t: {"count": tmpl[t]["count"], "dtype": sorted(tmpl[t]["dtype"]),
                             "shape": tmpl[t]["shape"], "class": classify(t)} for t in tmpl},
           "class_bytes": dict(cls_bytes), "class_elems": dict(cls_elems)}
    import os
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(out, open(args.json_out, "w"), indent=2, default=str)
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
