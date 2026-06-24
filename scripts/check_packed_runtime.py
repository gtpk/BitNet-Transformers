#!/usr/bin/env python3
"""Phase 3 test cases: PackedTernaryLinear runtime PoC.

A PackedTernaryLinear holds packed bytes (no dense [out,in] weight parameter)
and unpacks alpha*T on the fly inside forward. This proves the packed artifact
is usable as a runtime module, not just a storage blob.

  PACK-201 layer forward  : PackedTernaryLinear(x) == F.linear(x, S1 alpha*T)
  PACK-202 model logits    : model with packed linears == S1-converted model
  PACK-203 no dense weight : packed linears store uint8 codes, not a float weight
  PACK-204 state round-trip: save/load packed-linear state_dict keeps logits

    .venv/bin/python scripts/check_packed_runtime.py --json-out reports/packed_runtime_tc.json --strict
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama import packing as P  # noqa: E402


def _check(results: list, name: str, passed: bool, detail: str = "") -> None:
    results.append({"id": name, "pass": bool(passed), "detail": detail})
    print(f"{'PASS' if passed else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


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
def run(seed: int = 7) -> tuple[list, dict]:
    results: list = []

    # PACK-201: single packed linear forward equals dense S1 reconstruction
    torch.manual_seed(seed)
    linear = torch.nn.Linear(130, 48, bias=False)
    with torch.no_grad():
        linear.weight.mul_(0.05)
    packed_linear = P.PackedTernaryLinear.from_linear(linear, group_size=64, lambda_value=0.7)
    x = torch.randn(16, 130)
    _, s1_approx, _ = C.quantize_weight(linear.weight, C.S1)
    err = float((packed_linear(x) - F.linear(x, s1_approx)).abs().max())
    _check(results, "PACK-201/layer-forward", err < 1e-5, f"max_err={err:.2e}")

    # Build a model and its S1-converted reference.
    model, config = build_tiny_model(seed)
    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    ref_state, _ = C.convert_state_dict(model.state_dict(), C.S1)
    ref_model = copy.deepcopy(model)
    ref_model.load_state_dict(ref_state)
    ref_model.eval()
    ref_logits = ref_model(input_ids=input_ids).logits.detach()

    # PACK-202: swap target linears for PackedTernaryLinear, logits must match
    runtime = copy.deepcopy(model)
    count = P.replace_target_linears_with_packed(runtime, group_size=64, lambda_value=0.7)
    runtime.eval()
    runtime_logits = runtime(input_ids=input_ids).logits.detach()
    err2 = float((runtime_logits - ref_logits).abs().max())
    _check(results, "PACK-202/model-logits", err2 < 1e-5 and count > 0, f"layers={count} max_err={err2:.2e}")

    # PACK-203: packed linears expose no dense [out,in] weight parameter
    packed_modules = [m for m in runtime.modules() if isinstance(m, P.PackedTernaryLinear)]
    no_dense = all("weight" not in dict(m.named_parameters()) for m in packed_modules)
    has_codes = all(m.packed_codes.dtype == torch.uint8 for m in packed_modules)
    _check(results, "PACK-203/no-dense-weight", no_dense and has_codes and len(packed_modules) > 0,
           f"{len(packed_modules)} packed modules, uint8 codes, no float weight param")

    # PACK-204: state_dict save/load round trip keeps logits
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "runtime.pt"
        torch.save(runtime.state_dict(), path)
        fresh = copy.deepcopy(model)
        P.replace_target_linears_with_packed(fresh, group_size=64, lambda_value=0.7)
        fresh.load_state_dict(torch.load(path, weights_only=False))
        fresh.eval()
        err3 = float((fresh(input_ids=input_ids).logits.detach() - ref_logits).abs().max())
    _check(results, "PACK-204/state-roundtrip", err3 < 1e-5, f"max_err={err3:.2e}")

    # informative: runtime stored bytes vs fp16 dense for the target linears
    stored = sum(m.stored_bytes() for m in packed_modules)
    dense_fp16 = sum(m.out_features * m.in_features * 2 for m in packed_modules)
    info = {"packed_layers": len(packed_modules),
            "target_stored_bytes": stored, "target_fp16_bytes": dense_fp16,
            "target_compression_vs_fp16": dense_fp16 / max(stored, 1)}
    print(f"\n[runtime] target linears stored {stored/1024:.1f}KB vs fp16 {dense_fp16/1024:.1f}KB "
          f"-> {info['target_compression_vs_fp16']:.2f}x")
    return results, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=Path, default=Path("reports/packed_runtime_tc.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results, info = run(args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"results": results, "runtime": info}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    if args.strict and not all(item["pass"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
