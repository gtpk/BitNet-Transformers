#!/usr/bin/env python3
"""Estimate decode-time memory traffic for BitNet-style transformer variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def attention_weight_elements_per_layer(self) -> int:
        q_proj = self.hidden_size * self.hidden_size
        k_proj = self.hidden_size * self.kv_width
        v_proj = self.hidden_size * self.kv_width
        o_proj = self.hidden_size * self.hidden_size
        return q_proj + k_proj + v_proj + o_proj

    @property
    def mlp_weight_elements_per_layer(self) -> int:
        gate_proj = self.hidden_size * self.intermediate_size
        up_proj = self.hidden_size * self.intermediate_size
        down_proj = self.intermediate_size * self.hidden_size
        return gate_proj + up_proj + down_proj

    @property
    def weight_elements_per_layer(self) -> int:
        return self.attention_weight_elements_per_layer + self.mlp_weight_elements_per_layer


@dataclass(frozen=True)
class TrafficPolicy:
    name: str
    weight_bits: float
    kv_bits: float
    scale_bytes_per_output_channel: int = 2
    temp_weight_multiplier: float = 0.0
    activation_scratch_multiplier: float = 1.0
    materializes_dequant_weight: bool = False


@dataclass
class TrafficEstimate:
    policy: str
    context_length: int
    batch_size: int
    weight_read_mb: float
    dequant_temp_mb: float
    kv_read_mb: float
    kv_write_mb: float
    activation_scratch_mb: float
    total_mb_per_token: float
    ideal_tokens_per_second_at_bandwidth: dict[str, float]
    materializes_dequant_weight: bool


def bytes_to_mb(value: float) -> float:
    return value / (1024.0 * 1024.0)


def output_channels_per_layer(shape: ModelShape) -> int:
    return (
        shape.hidden_size
        + shape.kv_width
        + shape.kv_width
        + shape.hidden_size
        + shape.intermediate_size
        + shape.intermediate_size
        + shape.hidden_size
    )


def estimate(
    shape: ModelShape,
    policy: TrafficPolicy,
    context_length: int,
    batch_size: int,
    bandwidth_gbps_values: list[float],
) -> TrafficEstimate:
    weight_bits_bytes = shape.weight_elements_per_layer * policy.weight_bits / 8.0
    scale_bytes = output_channels_per_layer(shape) * policy.scale_bytes_per_output_channel
    weight_read = (weight_bits_bytes + scale_bytes) * shape.num_layers

    dequant_temp = 0.0
    if policy.temp_weight_multiplier:
        fp16_weight_bytes = shape.weight_elements_per_layer * 2.0 * shape.num_layers
        dequant_temp = fp16_weight_bytes * policy.temp_weight_multiplier

    kv_values_per_layer = batch_size * context_length * shape.kv_width * 2
    kv_read = kv_values_per_layer * policy.kv_bits / 8.0 * shape.num_layers

    kv_write_values_per_layer = batch_size * shape.kv_width * 2
    kv_write = kv_write_values_per_layer * policy.kv_bits / 8.0 * shape.num_layers

    hidden_bytes = batch_size * shape.hidden_size * 2.0
    mlp_bytes = batch_size * shape.intermediate_size * 2.0
    attn_score_bytes = batch_size * shape.num_attention_heads * context_length * 2.0
    activation_scratch = (
        hidden_bytes * 8.0 + mlp_bytes * 3.0 + attn_score_bytes * 2.0
    ) * shape.num_layers * policy.activation_scratch_multiplier

    total = weight_read + dequant_temp + kv_read + kv_write + activation_scratch
    ideal_tps = {
        f"{bandwidth:g}_GBps": bandwidth * 1_000_000_000.0 / total
        for bandwidth in bandwidth_gbps_values
    }

    return TrafficEstimate(
        policy=policy.name,
        context_length=context_length,
        batch_size=batch_size,
        weight_read_mb=bytes_to_mb(weight_read),
        dequant_temp_mb=bytes_to_mb(dequant_temp),
        kv_read_mb=bytes_to_mb(kv_read),
        kv_write_mb=bytes_to_mb(kv_write),
        activation_scratch_mb=bytes_to_mb(activation_scratch),
        total_mb_per_token=bytes_to_mb(total),
        ideal_tokens_per_second_at_bandwidth=ideal_tps,
        materializes_dequant_weight=policy.materializes_dequant_weight,
    )


def default_policies(dtype_bits: float) -> list[TrafficPolicy]:
    return [
        TrafficPolicy(
            name="fp16_baseline",
            weight_bits=dtype_bits,
            kv_bits=dtype_bits,
            scale_bytes_per_output_channel=0,
            activation_scratch_multiplier=1.0,
        ),
        TrafficPolicy(
            name="current_py_bitlinear_reference",
            weight_bits=dtype_bits,
            kv_bits=dtype_bits,
            scale_bytes_per_output_channel=0,
            temp_weight_multiplier=2.0,
            activation_scratch_multiplier=1.3,
            materializes_dequant_weight=True,
        ),
        TrafficPolicy(
            name="packed_b1_58_weight_fp16_kv",
            weight_bits=2.0,
            kv_bits=dtype_bits,
            scale_bytes_per_output_channel=2,
            activation_scratch_multiplier=0.8,
        ),
        TrafficPolicy(
            name="packed_b1_58_weight_int8_kv",
            weight_bits=2.0,
            kv_bits=8.0,
            scale_bytes_per_output_channel=2,
            activation_scratch_multiplier=0.8,
        ),
        TrafficPolicy(
            name="packed_b1_58_weight_int4_kv",
            weight_bits=2.0,
            kv_bits=4.0,
            scale_bytes_per_output_channel=2,
            activation_scratch_multiplier=0.8,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-key-value-heads", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    parser.add_argument("--dtype-bits", type=float, default=16.0)
    parser.add_argument("--bandwidth-gbps", type=float, nargs="+", default=[50.0, 100.0, 200.0])
    parser.add_argument("--json-out", type=Path, default=Path("reports/memory_traffic_estimate.json"))
    return parser.parse_args()


def print_table(estimates: list[TrafficEstimate]) -> None:
    headers = [
        "policy",
        "ctx",
        "weight_mb",
        "temp_mb",
        "kv_read_mb",
        "scratch_mb",
        "total_mb",
        "tps@100GBps",
    ]
    print(" ".join(f"{header:>24}" for header in headers))
    print("-" * (25 * len(headers)))
    for item in estimates:
        tps_100 = item.ideal_tokens_per_second_at_bandwidth.get("100_GBps", 0.0)
        print(
            f"{item.policy:>24} "
            f"{item.context_length:>24} "
            f"{item.weight_read_mb:>24.2f} "
            f"{item.dequant_temp_mb:>24.2f} "
            f"{item.kv_read_mb:>24.2f} "
            f"{item.activation_scratch_mb:>24.2f} "
            f"{item.total_mb_per_token:>24.2f} "
            f"{tps_100:>24.1f}"
        )


def main() -> None:
    args = parse_args()
    shape = ModelShape(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
    )
    policies = default_policies(args.dtype_bits)
    estimates = [
        estimate(shape, policy, context_length, args.batch_size, args.bandwidth_gbps)
        for context_length in args.context_lengths
        for policy in policies
    ]

    print_table(estimates)

    payload = {
        "shape": asdict(shape),
        "batch_size": args.batch_size,
        "dtype_bits": args.dtype_bits,
        "context_lengths": args.context_lengths,
        "bandwidth_gbps": args.bandwidth_gbps,
        "estimates": [asdict(item) for item in estimates],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
