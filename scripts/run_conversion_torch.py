#!/usr/bin/env python3
"""Convert a tiny real LLaMA to S0/S1 ternary and measure layer + output error.

This is the torch-based counterpart to ``run_conversion_feasibility.py``. It
builds a small ``transformers`` LLaMA, captures calibration activations at each
target linear via forward hooks, applies S0 (naive per-tensor) and S1
(groupwise-input) ternary PTQ, and reports:

  - per-projection weight MSE ratio (vs zero baseline)
  - per-projection layer output error (relative L2 of X @ W^T)  [PTQ-002]
  - end-to-end logit error and CE-loss delta on the same batch
  - theoretical ternary storage / compression                   [MEM-001]

Teacher-free: the only reference is the original (un-quantized) model's own
activations and weights, never a separate teacher's logits/hidden states.

    .venv/bin/python scripts/run_conversion_torch.py --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path
from bitnet_llama import conversion as C


def build_tiny_llama(seed: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config).eval()
    return model, config


def capture_activations(model, input_ids):
    """Run a forward pass, capturing the input activation to each target linear."""
    acts: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs, _output):
            acts[name] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        return hook

    name_by_module = {}
    for name, module in model.named_modules():
        weight_key = f"{name}.weight"
        if isinstance(module, torch.nn.Linear) and C.is_target_weight_key(weight_key):
            name_by_module[name] = weight_key
            handles.append(module.register_forward_hook(make_hook(weight_key)))

    with torch.no_grad():
        model(input_ids)
    for h in handles:
        h.remove()
    return acts


def evaluate_config(model, config, input_ids, conv_cfg):
    sd = model.state_dict()
    acts = capture_activations(model, input_ids)

    new_sd, stats = C.convert_state_dict(sd, conv_cfg)

    # per-layer output error using captured calibration activations
    per_layer = []
    output_errors = []
    for s in stats:
        x = acts.get(s.key)
        if x is None:
            continue
        oerr = C.output_error(sd[s.key], new_sd[s.key], x)
        output_errors.append(oerr)
        per_layer.append({
            "key": s.key,
            "shape": list(s.shape),
            "sparsity": round(s.sparsity, 4),
            "weight_mse_ratio": round(s.weight_mse_ratio, 4),
            "output_error": round(oerr, 4),
            "compression_vs_fp16": round(s.compression_vs_fp16, 3),
        })

    # end-to-end: load converted weights, compare logits + CE on same batch
    with torch.no_grad():
        labels = input_ids.clone()
        ref = model(input_ids, labels=labels)
        ref_logits, ref_loss = ref.logits, float(ref.loss)

        import copy
        conv_model = copy.deepcopy(model)
        conv_model.load_state_dict(new_sd)
        conv_model.eval()
        out = conv_model(input_ids, labels=labels)
        conv_logits, conv_loss = out.logits, float(out.loss)

    logit_rel = float(
        torch.linalg.vector_norm(conv_logits - ref_logits)
        / torch.linalg.vector_norm(ref_logits).clamp(min=1e-12)
    )

    def _ppl(loss):
        return float(torch.exp(torch.tensor(loss)))

    n = len(output_errors)
    return {
        "config": conv_cfg.name,
        "policy": conv_cfg.policy,
        "lambda": conv_cfg.lambda_value,
        "group_size": conv_cfg.group_size,
        "num_layers": len(stats),
        "mean_weight_mse_ratio": round(sum(s.weight_mse_ratio for s in stats) / len(stats), 4),
        "mean_output_error": round(sum(output_errors) / n, 4) if n else None,
        "max_output_error": round(max(output_errors), 4) if n else None,
        "mean_sparsity": round(sum(s.sparsity for s in stats) / len(stats), 4),
        "mean_compression_vs_fp16": round(sum(s.compression_vs_fp16 for s in stats) / len(stats), 3),
        "all_domain_ok": all(s.domain_ok for s in stats),
        "all_scale_finite": all(s.scale_finite for s in stats),
        "ref_loss": round(ref_loss, 4),
        "conv_loss": round(conv_loss, 4),
        "ref_ppl": round(_ppl(ref_loss), 3),
        "conv_ppl": round(_ppl(conv_loss), 3),
        "logit_rel_error": round(logit_rel, 4),
        "per_layer": per_layer,
    }


def worst_projection(result):
    """Identify the projection family with the largest mean output error."""
    by_family = defaultdict(list)
    for layer in result["per_layer"]:
        family = layer["key"].split(".")[-2]
        by_family[family].append(layer["output_error"])
    return {fam: round(sum(v) / len(v), 4) for fam, v in sorted(by_family.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--json-out", type=Path, default=Path("reports/conversion_tiny_llama.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    model, config = build_tiny_llama(args.seed)
    torch.manual_seed(args.seed + 1)
    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.seq_len))

    results = [evaluate_config(model, config, input_ids, cfg) for cfg in (C.S0, C.S1)]
    by_family = {r["config"]: worst_projection(r) for r in results}

    print("BitNet tiny-LLaMA ternary conversion (S0 vs S1)")
    print("=" * 52)
    for r in results:
        print(f"\n[{r['config']}] policy={r['policy']} lambda={r['lambda']} group={r['group_size']}")
        print(f"  weight_mse_ratio (mean) : {r['mean_weight_mse_ratio']}")
        print(f"  output_error     (mean) : {r['mean_output_error']}  (max {r['max_output_error']})")
        print(f"  sparsity         (mean) : {r['mean_sparsity']}")
        print(f"  compression vs fp16     : {r['mean_compression_vs_fp16']}x")
        print(f"  PPL  ref={r['ref_ppl']} -> conv={r['conv_ppl']}   logit_rel_err={r['logit_rel_error']}")
        print(f"  per-projection output error: {by_family[r['config']]}")

    s0, s1 = results[0], results[1]
    s1_not_worse = s1["mean_output_error"] <= s0["mean_output_error"] + 1e-6
    print("\nchecks:")
    print(f"  domain_ok (S0,S1)            : {s0['all_domain_ok']}, {s1['all_domain_ok']}")
    print(f"  scale_finite (S0,S1)         : {s0['all_scale_finite']}, {s1['all_scale_finite']}")
    print(f"  S1 output_error <= S0 (PTQ-003): {s1_not_worse}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "batch": args.batch,
        "seq_len": args.seq_len,
        "results": results,
        "per_projection_output_error": by_family,
        "checks": {
            "s0_domain_ok": s0["all_domain_ok"],
            "s1_domain_ok": s1["all_domain_ok"],
            "s0_scale_finite": s0["all_scale_finite"],
            "s1_scale_finite": s1["all_scale_finite"],
            "s1_not_worse_than_s0": s1_not_worse,
        },
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    if args.strict:
        ok = (
            s0["all_domain_ok"] and s1["all_domain_ok"]
            and s0["all_scale_finite"] and s1["all_scale_finite"]
            and s1_not_worse
        )
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
