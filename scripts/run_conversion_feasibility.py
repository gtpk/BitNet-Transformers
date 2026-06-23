#!/usr/bin/env python3
"""Run dependency-light feasibility checks for ternary BitNet conversion.

This script does not require torch or transformers. It checks whether simple
ternary PTQ policies can approximate linear weights and layer outputs better
than a zero-weight baseline on synthetic but structured matrices.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-12


@dataclass
class QuantizationResult:
    scenario: str
    policy: str
    lambda_value: float
    weight_mse_ratio: float
    output_mse_ratio: float
    sparsity: float
    domain_ok: bool


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, EPS))


def _scale_from_mask(values: np.ndarray, mask: np.ndarray, axis: int | tuple[int, ...] | None) -> np.ndarray:
    masked_abs = np.abs(values) * mask
    denom = np.maximum(np.sum(mask, axis=axis, keepdims=True), 1.0)
    return np.sum(masked_abs, axis=axis, keepdims=True) / denom


def quantize_per_tensor(weight: np.ndarray, lambda_value: float) -> tuple[np.ndarray, np.ndarray]:
    threshold = lambda_value * np.mean(np.abs(weight))
    mask = np.abs(weight) > threshold
    ternary = np.where(mask, np.sign(weight), 0.0).astype(np.float32)
    scale = _scale_from_mask(weight, mask.astype(np.float32), axis=None)
    return ternary, scale * ternary


def quantize_per_channel(weight: np.ndarray, lambda_value: float) -> tuple[np.ndarray, np.ndarray]:
    threshold = lambda_value * np.mean(np.abs(weight), axis=1, keepdims=True)
    mask = np.abs(weight) > threshold
    ternary = np.where(mask, np.sign(weight), 0.0).astype(np.float32)
    scale = _scale_from_mask(weight, mask.astype(np.float32), axis=1)
    return ternary, scale * ternary


def quantize_groupwise_input(
    weight: np.ndarray, lambda_value: float, group_size: int
) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros_like(weight, dtype=np.float32)
    ternary = np.zeros_like(weight, dtype=np.float32)

    for start in range(0, weight.shape[1], group_size):
        end = min(start + group_size, weight.shape[1])
        block = weight[:, start:end]
        threshold = lambda_value * np.mean(np.abs(block), axis=1, keepdims=True)
        mask = np.abs(block) > threshold
        block_ternary = np.where(mask, np.sign(block), 0.0).astype(np.float32)
        scale = _scale_from_mask(block, mask.astype(np.float32), axis=1)
        ternary[:, start:end] = block_ternary
        output[:, start:end] = scale * block_ternary

    return ternary, output


def quantize(weight: np.ndarray, policy: str, lambda_value: float, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    if policy == "per_tensor":
        return quantize_per_tensor(weight, lambda_value)
    if policy == "per_channel":
        return quantize_per_channel(weight, lambda_value)
    if policy == "groupwise_input":
        return quantize_groupwise_input(weight, lambda_value, group_size)
    raise ValueError(f"Unknown policy: {policy}")


def output(weight: np.ndarray, activations: np.ndarray) -> np.ndarray:
    return activations @ weight.T


def make_scenarios(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out_features = 128
    in_features = 256
    samples = 512

    x = rng.normal(0.0, 1.0, size=(samples, in_features)).astype(np.float32)
    w = rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)
    scenarios["gaussian"] = (w, x)

    channel_scale = rng.lognormal(mean=0.0, sigma=1.0, size=(out_features, 1)).astype(np.float32)
    w = rng.normal(0.0, 0.015, size=(out_features, in_features)).astype(np.float32) * channel_scale
    scenarios["channel_scaled"] = (w, x)

    w = (rng.standard_t(df=3.0, size=(out_features, in_features)) * 0.015).astype(np.float32)
    outlier_count = max(1, weight_numel_fraction(w, 0.002))
    flat_indices = rng.choice(w.size, size=outlier_count, replace=False)
    w.reshape(-1)[flat_indices] *= 8.0
    scenarios["heavy_tail"] = (w, x)

    rank = 16
    a = rng.normal(0.0, 0.04, size=(out_features, rank)).astype(np.float32)
    b = rng.normal(0.0, 0.04, size=(rank, in_features)).astype(np.float32)
    w = a @ b + rng.normal(0.0, 0.002, size=(out_features, in_features)).astype(np.float32)
    scenarios["low_rank_plus_noise"] = (w, x)

    correlated_x = x.copy()
    correlated_x[:, :32] *= 3.0
    w = rng.normal(0.0, 0.02, size=(out_features, in_features)).astype(np.float32)
    scenarios["activation_outlier"] = (w, correlated_x)

    return scenarios


def weight_numel_fraction(weight: np.ndarray, fraction: float) -> int:
    return int(np.ceil(weight.size * fraction))


def run(
    lambdas: Iterable[float],
    policies: Iterable[str],
    group_size: int,
    seed: int,
) -> list[QuantizationResult]:
    rng = np.random.default_rng(seed)
    results: list[QuantizationResult] = []

    for scenario, (weight, activations) in make_scenarios(rng).items():
        reference_weight_output = output(weight, activations)
        zero_weight = np.zeros_like(weight)
        zero_weight_mse = mse(weight, zero_weight)
        zero_output_mse = mse(reference_weight_output, output(zero_weight, activations))

        for policy in policies:
            for lambda_value in lambdas:
                ternary, approx_weight = quantize(weight, policy, lambda_value, group_size)
                approx_output = output(approx_weight, activations)
                unique = set(np.unique(ternary).tolist())
                results.append(
                    QuantizationResult(
                        scenario=scenario,
                        policy=policy,
                        lambda_value=float(lambda_value),
                        weight_mse_ratio=safe_ratio(mse(weight, approx_weight), zero_weight_mse),
                        output_mse_ratio=safe_ratio(mse(reference_weight_output, approx_output), zero_output_mse),
                        sparsity=float(np.mean(ternary == 0.0)),
                        domain_ok=unique.issubset({-1.0, 0.0, 1.0}),
                    )
                )

    return results


def summarize(results: list[QuantizationResult]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    scenarios = sorted({result.scenario for result in results})

    for scenario in scenarios:
        scenario_results = [result for result in results if result.scenario == scenario]
        best = min(scenario_results, key=lambda item: item.output_mse_ratio)
        naive_candidates = [
            result
            for result in scenario_results
            if result.policy == "per_tensor" and abs(result.lambda_value - 0.5) < EPS
        ]
        naive = naive_candidates[0] if naive_candidates else scenario_results[0]
        summary[scenario] = {
            "best": asdict(best),
            "naive": asdict(naive),
            "best_improves_zero_weight": best.weight_mse_ratio < 1.0,
            "best_improves_zero_output": best.output_mse_ratio < 1.0,
            "best_not_worse_than_naive_output": best.output_mse_ratio <= naive.output_mse_ratio + EPS,
        }

    return summary


def print_summary(summary: dict[str, dict[str, object]]) -> None:
    print("BitNet conversion feasibility smoke")
    print("=" * 40)
    for scenario, data in summary.items():
        best = data["best"]
        naive = data["naive"]
        print(f"\n[{scenario}]")
        print(
            "  best: "
            f"policy={best['policy']} lambda={best['lambda_value']} "
            f"weight_mse_ratio={best['weight_mse_ratio']:.4f} "
            f"output_mse_ratio={best['output_mse_ratio']:.4f} "
            f"sparsity={best['sparsity']:.3f}"
        )
        print(
            "  naive: "
            f"policy={naive['policy']} lambda={naive['lambda_value']} "
            f"weight_mse_ratio={naive['weight_mse_ratio']:.4f} "
            f"output_mse_ratio={naive['output_mse_ratio']:.4f} "
            f"sparsity={naive['sparsity']:.3f}"
        )
        print(
            "  checks: "
            f"zero_weight={data['best_improves_zero_weight']} "
            f"zero_output={data['best_improves_zero_output']} "
            f"search>=naive={data['best_not_worse_than_naive_output']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.3],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["per_tensor", "per_channel", "groupwise_input"],
    )
    parser.add_argument("--json-out", type=Path, default=Path("reports/conversion_feasibility_smoke.json"))
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if core smoke checks fail.")
    args = parser.parse_args()

    results = run(args.lambdas, args.policies, args.group_size, args.seed)
    summary = summarize(results)
    print_summary(summary)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "group_size": args.group_size,
        "lambdas": args.lambdas,
        "policies": args.policies,
        "summary": summary,
        "results": [asdict(result) for result in results],
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    checks = [
        data["best_improves_zero_weight"]
        and data["best_improves_zero_output"]
        and data["best_not_worse_than_naive_output"]
        and data["best"]["domain_ok"]
        for data in summary.values()
    ]
    if args.strict and not all(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
