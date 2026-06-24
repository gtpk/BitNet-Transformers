#!/usr/bin/env python3
"""Test cases for packed ternary weight storage (bitnet_llama/packing.py).

Proves the storage half of b1.58:
  PACK-001 domain        : unpack stays in {-1,0,+1}
  PACK-002 round trip    : unpack(pack(T)) == T for both schemes
  PACK-003 dense match   : to_dense() == conversion.S1 alpha*T (exact)
  PACK-004 sste export   : pack(ScaledBitLinear).to_dense() == layer forward value
  PACK-005 save/load     : load(save(x)).to_dense() == x.to_dense()
  PACK-006 storage       : trit < two_bit < int8 < fp16, trit ~ b1.58 bound

    .venv/bin/python scripts/check_packed_ternary.py --json-out reports/packed_ternary_tc.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama import packing as P  # noqa: E402
from bitnet_llama.module import ScaledBitLinear  # noqa: E402


def _check(results: list, name: str, passed: bool, detail: str = "") -> None:
    results.append({"id": name, "pass": bool(passed), "detail": detail})
    print(f"{'PASS' if passed else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


def run() -> list:
    torch.manual_seed(0)
    results: list = []
    weight = torch.randn(48, 130) * 0.05  # non-multiple in_features exercises padding

    # PACK-001 / PACK-002: domain + round trip for both schemes
    ternary, _, _, _ = P.groupwise_ternary_and_scales(weight, group_size=64, lambda_value=0.7)
    for scheme in P.SCHEMES:
        packed = P._pack(scheme, ternary)
        restored = P._unpack(scheme, packed, ternary.numel()).reshape(ternary.shape)
        domain_ok = set(torch.unique(restored).tolist()).issubset({-1, 0, 1})
        roundtrip_ok = bool(torch.equal(restored, ternary))
        _check(results, f"PACK-001/{scheme}-domain", domain_ok)
        _check(results, f"PACK-002/{scheme}-roundtrip", roundtrip_ok)

    # PACK-003: to_dense equals conversion.S1 reconstruction exactly
    _, s1_approx, _ = C.quantize_weight(weight, C.S1)  # uses group_size 64, lambda 0.7
    for scheme in P.SCHEMES:
        pk = P.PackedTernaryWeight.from_weight(weight, group_size=64, lambda_value=0.7, scheme=scheme)
        max_err = float((pk.to_dense() - s1_approx).abs().max())
        _check(results, f"PACK-003/{scheme}-dense-match-conversion", max_err < 1e-6, f"max_err={max_err:.2e}")

    # PACK-004: exported ScaledBitLinear reconstructs its own forward weight
    layer = ScaledBitLinear(130, 48, bias=False, group_size=64, lambda_value=0.7)
    with torch.no_grad():
        layer.weight.copy_(weight)
    forward_weight = layer.quantize_weight_groupwise().detach()
    pk = P.pack_scaled_bitlinear(layer, scheme="trit")
    max_err = float((pk.to_dense() - forward_weight).abs().max())
    _check(results, "PACK-004/scaled-bitlinear-export", max_err < 1e-6, f"max_err={max_err:.2e}")

    # PACK-005: save/load round trip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "packed.pt"
        pk.save(path)
        loaded = P.PackedTernaryWeight.load(path)
        ok = bool(torch.equal(loaded.to_dense(), pk.to_dense()))
        _check(results, "PACK-005/save-load-roundtrip", ok)

    # PACK-006: storage ordering and b1.58 proximity
    rep = P.storage_report(out_features=48, in_features=130, n_groups=pk.n_groups)
    ordering = rep["trit_bytes"] < rep["two_bit_bytes"] < rep["int8_bytes"] < rep["fp16_bytes"]
    near_ideal = abs(rep["trit_bits_per_elem"] - P.TERNARY_BITS_PER_ELEM) < 0.05
    _check(results, "PACK-006/storage-ordering", ordering,
           f"trit={rep['trit_bytes']} two_bit={rep['two_bit_bytes']} int8={rep['int8_bytes']} fp16={rep['fp16_bytes']}")
    _check(results, "PACK-006/trit-near-b158", near_ideal,
           f"trit_bits_per_elem={rep['trit_bits_per_elem']:.3f} vs {P.TERNARY_BITS_PER_ELEM:.3f}")

    # Informative: a realistic layer's compression
    big = P.storage_report(out_features=512, in_features=2048, n_groups=2048 // 64)
    print(f"\n[512x2048 layer] trit {big['trit_bytes']/1024:.1f}KB vs fp16 {big['fp16_bytes']/1024:.1f}KB "
          f"-> {big['trit_compression_vs_fp16']:.2f}x  (two_bit {big['two_bit_compression_vs_fp16']:.2f}x)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=Path("reports/packed_ternary_tc.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results = run()
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    if args.strict and not all(item["pass"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
