# coding=utf-8
"""Torch-based reference for converting an existing LLM checkpoint to a
BitNet-style ternary representation (Phase A of the conversion plan).

This module is intentionally teacher-free and dependency-light: it only needs
``torch`` and operates on a plain ``state_dict``. It implements the S0/S1 rungs
of the conversion ladder described in
``docs/existing_model_to_bitnet_conversion_plan.md``:

    S0: naive per-tensor ternary PTQ
    S1: scaled ternary PTQ (per-output-channel or groupwise-along-input)

Weight representation::

    W_fp      original full precision weight, shape [out_features, in_features]
    T         ternary code in {-1, 0, +1}
    alpha     scale (per-tensor / per-channel / per-group), all >= 0
    W_approx  alpha * T

Layer/output error is measured against calibration activations so the quality
signal is "how well is ``X @ W^T`` preserved", not just weight MSE.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable

import torch

EPS = 1e-12
TERNARY_BITS_PER_ELEM = math.log2(3.0)  # ~1.585, the b1.58 information bound

# Projections we convert first. Matches the conversion plan's layer policy.
TARGET_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# Explicitly excluded from the initial conversion (separate quality/storage story).
EXCLUDE_SUFFIXES: tuple[str, ...] = ("embed_tokens", "lm_head")
EXCLUDE_KEYWORDS: tuple[str, ...] = ("norm",)


@dataclass
class ConversionConfig:
    """A single rung on the conversion ladder."""

    policy: str = "per_channel"  # per_tensor | per_channel | groupwise_input
    lambda_value: float = 0.7
    group_size: int = 64
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.policy}/lambda={self.lambda_value}"


# Canonical presets referenced throughout the plan.
S0 = ConversionConfig(policy="per_tensor", lambda_value=0.5, name="S0")
S1 = ConversionConfig(policy="groupwise_input", lambda_value=0.7, group_size=64, name="S1")


@dataclass
class LayerStat:
    key: str
    shape: tuple[int, ...]
    policy: str
    lambda_value: float
    group_size: int
    sparsity: float
    domain_ok: bool
    scale_finite: bool
    weight_mse: float
    weight_mse_ratio: float  # vs zero-weight baseline; < 1.0 means useful
    improves_zero: bool
    scale_params: int
    ternary_bits: float
    scale_bits: float
    fp16_bits: float
    compression_vs_fp16: float


# --------------------------------------------------------------------------- #
# Key discovery
# --------------------------------------------------------------------------- #
def is_target_weight_key(key: str) -> bool:
    """True for ``...<proj>.weight`` keys we convert; False for excluded keys."""
    if not key.endswith(".weight"):
        return False
    if any(key.endswith(f"{suffix}.weight") for suffix in EXCLUDE_SUFFIXES):
        return False
    if any(word in key for word in EXCLUDE_KEYWORDS):
        return False
    return any(key.endswith(f"{suffix}.weight") for suffix in TARGET_SUFFIXES)


def find_target_keys(state_dict: dict[str, torch.Tensor]) -> list[str]:
    return [key for key in state_dict if is_target_weight_key(key)]


# --------------------------------------------------------------------------- #
# Ternary quantization policies
# --------------------------------------------------------------------------- #
def _ternarize_block(block: torch.Tensor, lambda_value: float, dim: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Ternarize along ``dim`` (None = whole tensor).

    Returns ``(ternary, w_approx)`` where ``ternary`` is in {-1, 0, +1} and
    ``w_approx = alpha * ternary`` with a non-negative scale broadcast over dim.
    """
    absw = block.abs()
    if dim is None:
        threshold = lambda_value * absw.mean()
    else:
        threshold = lambda_value * absw.mean(dim=dim, keepdim=True)
    mask = absw > threshold
    ternary = torch.where(mask, torch.sign(block), torch.zeros_like(block))

    masked_abs = absw * mask
    if dim is None:
        denom = mask.sum().clamp(min=1.0)
        scale = masked_abs.sum() / denom
    else:
        denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
        scale = masked_abs.sum(dim=dim, keepdim=True) / denom
    return ternary, scale * ternary


def quantize_per_tensor(weight: torch.Tensor, lambda_value: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    ternary, approx = _ternarize_block(weight, lambda_value, dim=None)
    return ternary, approx, 1


def quantize_per_channel(weight: torch.Tensor, lambda_value: float) -> tuple[torch.Tensor, torch.Tensor, int]:
    # weight is [out_features, in_features]; one scale per output channel.
    ternary, approx = _ternarize_block(weight, lambda_value, dim=1)
    return ternary, approx, weight.shape[0]


def quantize_groupwise_input(
    weight: torch.Tensor, lambda_value: float, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    # one scale per (output channel, input-block) pair.
    out_features, in_features = weight.shape
    ternary = torch.zeros_like(weight)
    approx = torch.zeros_like(weight)
    n_groups = 0
    for start in range(0, in_features, group_size):
        end = min(start + group_size, in_features)
        block_t, block_a = _ternarize_block(weight[:, start:end], lambda_value, dim=1)
        ternary[:, start:end] = block_t
        approx[:, start:end] = block_a
        n_groups += 1
    return ternary, approx, out_features * n_groups


def quantize_weight(
    weight: torch.Tensor, config: ConversionConfig
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Dispatch to the configured policy. Returns ``(ternary, w_approx, scale_params)``."""
    if weight.dim() != 2:
        raise ValueError(f"expected 2D linear weight, got shape {tuple(weight.shape)}")
    work = weight.detach().float()
    if config.policy == "per_tensor":
        return quantize_per_tensor(work, config.lambda_value)
    if config.policy == "per_channel":
        return quantize_per_channel(work, config.lambda_value)
    if config.policy == "groupwise_input":
        return quantize_groupwise_input(work, config.lambda_value, config.group_size)
    raise ValueError(f"unknown policy: {config.policy}")


# --------------------------------------------------------------------------- #
# Error / storage metrics
# --------------------------------------------------------------------------- #
def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b) ** 2).mean())


def output_error(weight: torch.Tensor, weight_approx: torch.Tensor, activations: torch.Tensor) -> float:
    """Relative L2 error of ``X @ W^T`` after quantization (PTQ-002).

    ``activations`` has shape ``[*, in_features]``.
    """
    w = weight.detach().float()
    wa = weight_approx.detach().float()
    x = activations.detach().float()
    ref = x @ w.t()
    approx = x @ wa.t()
    num = torch.linalg.vector_norm(ref - approx)
    den = torch.linalg.vector_norm(ref).clamp(min=EPS)
    return float(num / den)


def storage_bits(numel: int, scale_params: int, scale_dtype_bits: int = 16) -> tuple[float, float, float, float]:
    """Theoretical storage (MEM-001). Returns (ternary_bits, scale_bits, fp16_bits, compression)."""
    ternary_bits = TERNARY_BITS_PER_ELEM * numel
    scale_bits = scale_dtype_bits * scale_params
    fp16_bits = 16.0 * numel
    total = ternary_bits + scale_bits
    compression = fp16_bits / max(total, EPS)
    return ternary_bits, scale_bits, fp16_bits, compression


def _layer_stat(key: str, weight: torch.Tensor, config: ConversionConfig) -> tuple[torch.Tensor, LayerStat]:
    ternary, approx, scale_params = quantize_weight(weight, config)
    w = weight.detach().float()
    zero_mse = _mse(w, torch.zeros_like(w))
    approx_mse = _mse(w, approx)
    unique = set(torch.unique(ternary).tolist())
    numel = weight.numel()
    t_bits, s_bits, fp16_bits, compression = storage_bits(numel, scale_params)
    stat = LayerStat(
        key=key,
        shape=tuple(weight.shape),
        policy=config.policy,
        lambda_value=config.lambda_value,
        group_size=config.group_size,
        sparsity=float((ternary == 0).float().mean()),
        domain_ok=unique.issubset({-1.0, 0.0, 1.0}),
        scale_finite=bool(torch.isfinite(approx).all()),
        weight_mse=approx_mse,
        weight_mse_ratio=float(approx_mse / max(zero_mse, EPS)),
        improves_zero=approx_mse < zero_mse,
        scale_params=scale_params,
        ternary_bits=t_bits,
        scale_bits=s_bits,
        fp16_bits=fp16_bits,
        compression_vs_fp16=compression,
    )
    return approx, stat


# --------------------------------------------------------------------------- #
# State-dict conversion
# --------------------------------------------------------------------------- #
def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: ConversionConfig,
) -> tuple[dict[str, torch.Tensor], list[LayerStat]]:
    """Return a new state_dict with target linears replaced by ``alpha * T``.

    Non-target tensors (embedding, lm_head, norms, biases) are passed through
    unchanged. Shapes are preserved (CONV-001), so the result loads into the
    original architecture.
    """
    new_state: dict[str, torch.Tensor] = {}
    stats: list[LayerStat] = []
    for key, tensor in state_dict.items():
        if is_target_weight_key(key) and tensor.dim() == 2:
            approx, stat = _layer_stat(key, tensor, config)
            new_state[key] = approx.to(dtype=tensor.dtype, device=tensor.device)
            stats.append(stat)
        else:
            new_state[key] = tensor
    return new_state, stats


def stats_summary(stats: Iterable[LayerStat]) -> dict[str, object]:
    stats = list(stats)
    if not stats:
        return {"num_layers": 0}
    return {
        "num_layers": len(stats),
        "all_domain_ok": all(s.domain_ok for s in stats),
        "all_scale_finite": all(s.scale_finite for s in stats),
        "all_improve_zero": all(s.improves_zero for s in stats),
        "mean_sparsity": sum(s.sparsity for s in stats) / len(stats),
        "mean_weight_mse_ratio": sum(s.weight_mse_ratio for s in stats) / len(stats),
        "mean_compression_vs_fp16": sum(s.compression_vs_fp16 for s in stats) / len(stats),
        "layers": [asdict(s) for s in stats],
    }
