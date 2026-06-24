#!/usr/bin/env python3
"""Run focused TC checks for ScaledBitLinear.

These checks are intentionally tiny and CPU-only. They verify the algorithmic
contract before a larger Colab run:

* SSTE-001: the fake-quantized forward weight matches S1 alpha*T.
* SSTE-002: gradients flow through the latent full-precision weight.
* SSTE-003: optional activation fake quantization keeps finite outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama.module import ScaledBitLinear  # noqa: E402


def check_s1_weight_equivalence() -> dict[str, object]:
    torch.manual_seed(11)
    layer = ScaledBitLinear(
        in_features=10,
        out_features=6,
        bias=False,
        group_size=4,
        lambda_value=0.7,
    )
    with torch.no_grad():
        layer.weight.normal_(mean=0.0, std=0.2)

    config = C.ConversionConfig(
        policy="groupwise_input",
        lambda_value=0.7,
        group_size=4,
        name="tc_s1",
    )
    _, expected, _ = C.quantize_weight(layer.weight, config)
    observed = layer.quantize_weight_groupwise().detach()
    max_abs_diff = float((expected - observed).abs().max())
    return {
        "id": "SSTE-001",
        "name": "S1 alpha*T equivalence",
        "max_abs_diff": max_abs_diff,
        "passed": max_abs_diff <= 1e-6,
    }


def check_gradient_flow() -> dict[str, object]:
    torch.manual_seed(17)
    layer = ScaledBitLinear(
        in_features=8,
        out_features=5,
        bias=True,
        group_size=4,
        lambda_value=0.7,
    )
    x = torch.randn(3, 8)
    target = torch.randn(3, 5)
    loss = F.mse_loss(layer(x), target)
    loss.backward()
    grad = layer.weight.grad
    finite = grad is not None and bool(torch.isfinite(grad).all())
    grad_norm = float(grad.norm()) if grad is not None else 0.0
    return {
        "id": "SSTE-002",
        "name": "STE gradient flow",
        "loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "passed": finite and grad_norm > 0.0,
    }


def check_activation_fake_quant() -> dict[str, object]:
    torch.manual_seed(23)
    layer = ScaledBitLinear(
        in_features=8,
        out_features=5,
        bias=True,
        group_size=4,
        lambda_value=0.7,
        activation_bits=8,
    )
    output = layer(torch.randn(2, 4, 8))
    finite = bool(torch.isfinite(output).all())
    return {
        "id": "SSTE-003",
        "name": "activation fake quant finite output",
        "shape": list(output.shape),
        "passed": finite,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("reports/scaled_bitlinear_tc.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = [
        check_s1_weight_equivalence(),
        check_gradient_flow(),
        check_activation_fake_quant(),
    ]
    payload = {
        "all_passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for item in checks:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{status} {item['id']}: {item['name']}")
    print(f"Wrote {args.json_out}")
    if not payload["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
