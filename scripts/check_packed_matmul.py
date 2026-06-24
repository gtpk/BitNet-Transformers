#!/usr/bin/env python3
"""Phase 4 reference: blocked dequant matmul without full dense materialization.

The question Phase 4 answers honestly: can we compute the linear WITHOUT ever
holding the full dense weight, so peak working memory drops? Latency is NOT the
goal here (a Python loop is slower than a dense matmul); a real kernel comes
after this reference.

  PACK-301 correctness   : packed_linear_matmul == F.linear(to_dense())
  PACK-302 peak memory    : peak transient weight = chunk*in << out*in
  PACK-303 fused module    : PackedTernaryLinear(fused=True) logits == dense path
  PACK-304 latency honesty : measure both paths, report ratio (no speed assert)

    .venv/bin/python scripts/check_packed_matmul.py --json-out reports/packed_matmul_tc.json --strict
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
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


@torch.no_grad()
def run(seed: int = 7) -> tuple[list, dict]:
    results: list = []
    torch.manual_seed(seed)
    out_features, in_features = 256, 256
    weight = torch.randn(out_features, in_features) * 0.05
    x = torch.randn(64, in_features)
    packed = P.PackedTernaryWeight.from_weight(weight, group_size=64, lambda_value=0.7, scheme="trit")
    dense = packed.to_dense()

    # PACK-301: blocked matmul equals dense reference
    chunk = 32
    y_blocked, peak = P.packed_linear_matmul(x, packed, out_chunk=chunk, return_peak=True)
    y_dense = F.linear(x, dense)
    err = float((y_blocked - y_dense).abs().max())
    _check(results, "PACK-301/correctness", err < 1e-5, f"max_err={err:.2e}")

    # PACK-302: peak transient weight is one chunk, not the full matrix
    full = out_features * in_features
    reduction = full / max(peak, 1)
    _check(results, "PACK-302/peak-memory", peak == chunk * in_features and peak < full,
           f"peak_weight_numel={peak} vs full={full} -> {reduction:.1f}x smaller working set")

    # PACK-303: fused runtime module matches the dense path on a real model
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    ref_state, _ = C.convert_state_dict(model.state_dict(), C.S1)
    ref = copy.deepcopy(model); ref.load_state_dict(ref_state); ref.eval()
    ref_logits = ref(input_ids=ids).logits.detach()

    fused = copy.deepcopy(model)
    P.replace_target_linears_with_packed(fused, group_size=64, lambda_value=0.7, fused=True)
    fused.eval()
    fused_err = float((fused(input_ids=ids).logits.detach() - ref_logits).abs().max())
    _check(results, "PACK-303/fused-module", fused_err < 1e-5, f"max_logit_err={fused_err:.2e}")

    # PACK-304: latency honesty -- measure, do not assert speed
    def _time(fn, n=20):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000.0

    dense_ms = _time(lambda: F.linear(x, packed.to_dense()))
    blocked_ms = _time(lambda: P.packed_linear_matmul(x, packed, out_chunk=chunk))
    slowdown = blocked_ms / max(dense_ms, 1e-9)
    _check(results, "PACK-304/latency-measured", blocked_ms > 0 and dense_ms > 0,
           f"dense {dense_ms:.3f}ms, blocked {blocked_ms:.3f}ms -> {slowdown:.1f}x slower (expected; memory win, not speed)")

    info = {
        "peak_weight_numel": peak,
        "full_weight_numel": full,
        "working_set_reduction": reduction,
        "dense_ms": dense_ms,
        "blocked_ms": blocked_ms,
        "blocked_slowdown": slowdown,
    }
    print(f"\n[phase4] working-set {reduction:.1f}x smaller; blocked path {slowdown:.1f}x slower "
          f"than dense (Python-loop reference -- real kernel needed for latency)")
    return results, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=Path, default=Path("reports/packed_matmul_tc.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results, info = run(args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"results": results, "phase4": info}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    if args.strict and not all(item["pass"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
