# coding=utf-8
"""Packed ternary weight storage for BitNet-style ternary weights.

This module turns a ternary weight ``T in {-1, 0, +1}`` plus groupwise scales
``alpha`` into a compact byte layout, and reconstructs ``alpha * T`` exactly.
It is the storage half of the b1.58 story: it proves that the theoretical
~1.585 bits/element actually maps to a real byte reduction. It is NOT a fast
matmul kernel — reconstruction returns a dense tensor for a reference matmul.

Two packing schemes (see docs/packed_ternary_format_plan.md):

- ``two_bit``  : 2 bits/trit, 4 trits/byte. Simple, byte-aligned, 2.0 bits/elem.
- ``trit``     : base-3, 5 trits/byte (3**5 = 243 <= 256). 1.6 bits/elem,
                 essentially the b1.58 information bound (log2(3) = 1.585).

The groupwise ternarization matches ``conversion.S1`` and
``module.ScaledBitLinear`` exactly, so a trained ScaledBitLinear layer can be
exported and re-imported without changing its forward value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import conversion as _conv  # single source for the target-layer policy

TERNARY_BITS_PER_ELEM = math.log2(3.0)  # 1.585, the information bound
SCHEMES = ("two_bit", "trit")


# --------------------------------------------------------------------------- #
# Trit <-> byte packing.  Ternary {-1,0,+1} is shifted to {0,1,2} before packing.
# --------------------------------------------------------------------------- #
def _to_shifted(ternary: torch.Tensor) -> torch.Tensor:
    t = ternary.to(torch.int64).reshape(-1)
    if t.numel() and (int(t.min()) < -1 or int(t.max()) > 1):
        raise ValueError("ternary tensor must contain only {-1, 0, +1}")
    return t + 1  # {-1,0,1} -> {0,1,2}


def _pad_to(values: torch.Tensor, multiple: int) -> torch.Tensor:
    pad = (-values.numel()) % multiple
    if pad:
        values = torch.cat([values, torch.zeros(pad, dtype=values.dtype)])
    return values


def pack_two_bit(ternary: torch.Tensor) -> torch.Tensor:
    shifted = _pad_to(_to_shifted(ternary), 4).reshape(-1, 4)
    packed = shifted[:, 0] | (shifted[:, 1] << 2) | (shifted[:, 2] << 4) | (shifted[:, 3] << 6)
    return packed.to(torch.uint8)


def unpack_two_bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    p = packed.to(torch.int64)
    shifted = torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=1).reshape(-1)
    return (shifted[:numel] - 1).to(torch.int8)


def pack_trit(ternary: torch.Tensor) -> torch.Tensor:
    shifted = _pad_to(_to_shifted(ternary), 5).reshape(-1, 5)
    weights = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int64)
    packed = (shifted * weights).sum(dim=1)
    return packed.to(torch.uint8)


def unpack_trit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    p = packed.to(torch.int64)
    digits = []
    for _ in range(5):
        digits.append(p % 3)
        p = p // 3
    shifted = torch.stack(digits, dim=1).reshape(-1)
    return (shifted[:numel] - 1).to(torch.int8)


def _pack(scheme: str, ternary: torch.Tensor) -> torch.Tensor:
    if scheme == "two_bit":
        return pack_two_bit(ternary)
    if scheme == "trit":
        return pack_trit(ternary)
    raise ValueError(f"unknown scheme: {scheme}")


def _unpack(scheme: str, packed: torch.Tensor, numel: int) -> torch.Tensor:
    if scheme == "two_bit":
        return unpack_two_bit(packed, numel)
    if scheme == "trit":
        return unpack_trit(packed, numel)
    raise ValueError(f"unknown scheme: {scheme}")


# --------------------------------------------------------------------------- #
# Groupwise ternarization (matches conversion.S1 / ScaledBitLinear).
# --------------------------------------------------------------------------- #
def groupwise_ternary_and_scales(
    weight: torch.Tensor, group_size: int, lambda_value: float
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return (ternary int8 [out,in], scales float [out,n_groups], group_size, n_groups)."""
    w = weight.detach().float()
    out_features, in_features = w.shape
    gs = in_features if group_size <= 0 else group_size
    n_groups = (in_features + gs - 1) // gs
    ternary = torch.zeros_like(w)
    scales = torch.zeros(out_features, n_groups)
    for index, start in enumerate(range(0, in_features, gs)):
        end = min(start + gs, in_features)
        block = w[:, start:end]
        absb = block.abs()
        threshold = lambda_value * absb.mean(dim=1, keepdim=True)
        mask = absb > threshold
        ternary[:, start:end] = torch.where(mask, torch.sign(block), torch.zeros_like(block))
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        scales[:, index] = ((absb * mask).sum(dim=1, keepdim=True) / denom).squeeze(1)
    return ternary.to(torch.int8), scales, gs, n_groups


@dataclass
class PackedTernaryWeight:
    """A packed ternary weight: byte-packed codes + groupwise fp scales + metadata."""

    scheme: str
    out_features: int
    in_features: int
    group_size: int
    n_groups: int
    packed: torch.Tensor  # uint8
    scales: torch.Tensor  # float32 [out_features, n_groups]

    @classmethod
    def from_weight(
        cls,
        weight: torch.Tensor,
        group_size: int = 64,
        lambda_value: float = 0.7,
        scheme: str = "trit",
    ) -> "PackedTernaryWeight":
        if weight.dim() != 2:
            raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
        ternary, scales, gs, n_groups = groupwise_ternary_and_scales(weight, group_size, lambda_value)
        return cls(
            scheme=scheme,
            out_features=weight.shape[0],
            in_features=weight.shape[1],
            group_size=gs,
            n_groups=n_groups,
            packed=_pack(scheme, ternary),
            scales=scales,
        )

    def unpack_ternary(self) -> torch.Tensor:
        numel = self.out_features * self.in_features
        return _unpack(self.scheme, self.packed, numel).reshape(self.out_features, self.in_features)

    def to_dense(self) -> torch.Tensor:
        """Reconstruct ``alpha * T`` as a dense float tensor (== ScaledBitLinear forward value)."""
        ternary = self.unpack_ternary().float()
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            dtype=self.scales.dtype,
            device=self.scales.device,
        )
        for index, start in enumerate(range(0, self.in_features, self.group_size)):
            end = min(start + self.group_size, self.in_features)
            dense[:, start:end] = ternary[:, start:end] * self.scales[:, index : index + 1]
        return dense

    # -- storage accounting --------------------------------------------------
    def code_bytes(self) -> int:
        return int(self.packed.numel())

    def scale_bytes(self, scale_dtype_bits: int = 16) -> int:
        return int(self.scales.numel() * scale_dtype_bits // 8)

    def nbytes(self, scale_dtype_bits: int = 16) -> int:
        return self.code_bytes() + self.scale_bytes(scale_dtype_bits)

    def state(self) -> dict:
        return {
            "scheme": self.scheme,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "group_size": self.group_size,
            "n_groups": self.n_groups,
            "packed": self.packed,
            "scales": self.scales,
        }

    @classmethod
    def from_state(cls, state: dict) -> "PackedTernaryWeight":
        return cls(**state)

    def save(self, path) -> None:
        torch.save(self.state(), path)

    @classmethod
    def load(cls, path) -> "PackedTernaryWeight":
        return cls.from_state(torch.load(path, weights_only=False))


def pack_scaled_bitlinear(layer, scheme: str = "trit") -> PackedTernaryWeight:
    """Export a trained ``ScaledBitLinear`` layer to packed ternary storage."""
    return PackedTernaryWeight.from_weight(
        layer.weight,
        group_size=layer.group_size,
        lambda_value=layer.lambda_value,
        scheme=scheme,
    )


# --------------------------------------------------------------------------- #
# Model-wide export/import (Phase 2).  Only target linears are packed; embedding,
# lm_head, norms, and biases stay fp16.
# --------------------------------------------------------------------------- #
def _named_target_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _conv.is_target_weight_key(f"{name}.weight"):
            yield name, module


def pack_model(
    model: nn.Module, group_size: int = 64, lambda_value: float = 0.7, scheme: str = "trit"
) -> dict:
    """Pack every target linear of ``model`` into a single artifact dict."""
    layers = {
        name: PackedTernaryWeight.from_weight(module.weight, group_size, lambda_value, scheme)
        for name, module in _named_target_linears(model)
    }
    return {"scheme": scheme, "group_size": group_size, "lambda_value": lambda_value, "layers": layers}


def _set_module_weight(model: nn.Module, name: str, weight: torch.Tensor) -> None:
    module = model.get_submodule(name)
    with torch.no_grad():
        module.weight.copy_(weight.to(dtype=module.weight.dtype, device=module.weight.device))


def unpack_into_model(model: nn.Module, artifact: dict) -> nn.Module:
    """Reconstruct alpha*T into the target linears of ``model`` in place."""
    for name, packed in artifact["layers"].items():
        _set_module_weight(model, name, packed.to_dense())
    return model


def save_packed_model(artifact: dict, path) -> None:
    serial = {key: artifact[key] for key in ("scheme", "group_size", "lambda_value")}
    serial["layers"] = {name: packed.state() for name, packed in artifact["layers"].items()}
    torch.save(serial, path)


def load_packed_model(path) -> dict:
    serial = torch.load(path, weights_only=False)
    artifact = {key: serial[key] for key in ("scheme", "group_size", "lambda_value")}
    artifact["layers"] = {name: PackedTernaryWeight.from_state(s) for name, s in serial["layers"].items()}
    return artifact


def model_storage_report(model: nn.Module, artifact: dict, scale_dtype_bits: int = 16) -> dict:
    """Whole-model storage: packed target linears + fp16 for everything else.

    The per-layer compression (~8.65x for a square linear) is diluted at the
    model level because embedding/lm_head/norms stay fp16, so this reports the
    honest end-to-end number.
    """
    packed_bytes = sum(packed.nbytes(scale_dtype_bits) for packed in artifact["layers"].values())
    target_numel = sum(module.weight.numel() for _, module in _named_target_linears(model))
    all_numel = sum(p.numel() for p in model.parameters())
    other_bytes = (all_numel - target_numel) * 2  # fp16
    total = packed_bytes + other_bytes
    fp16_total = all_numel * 2
    return {
        "packed_target_bytes": packed_bytes,
        "fp16_other_bytes": other_bytes,
        "packed_total_bytes": total,
        "fp16_total_bytes": fp16_total,
        "model_compression_vs_fp16": fp16_total / max(total, 1),
        "target_numel": target_numel,
        "other_numel": all_numel - target_numel,
        "target_fraction": target_numel / max(all_numel, 1),
        "num_packed_layers": len(artifact["layers"]),
    }


# --------------------------------------------------------------------------- #
# Runtime module (Phase 3).  Holds packed bytes (no dense weight param) and
# unpacks alpha*T on the fly for a reference F.linear. NOT a fast kernel.
# --------------------------------------------------------------------------- #
class PackedTernaryLinear(nn.Module):
    """Drop-in linear that stores a packed ternary weight instead of a dense one."""

    def __init__(self, packed: "PackedTernaryWeight", bias: torch.Tensor | None = None):
        super().__init__()
        self.scheme = packed.scheme
        self.out_features = packed.out_features
        self.in_features = packed.in_features
        self.group_size = packed.group_size
        self.n_groups = packed.n_groups
        # packed codes + scales are buffers (no [out,in] float weight parameter)
        self.register_buffer("packed_codes", packed.packed)
        self.register_buffer("scales", packed.scales)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def _as_packed(self) -> "PackedTernaryWeight":
        return PackedTernaryWeight(
            scheme=self.scheme,
            out_features=self.out_features,
            in_features=self.in_features,
            group_size=self.group_size,
            n_groups=self.n_groups,
            packed=self.packed_codes,
            scales=self.scales,
        )

    def dense_weight(self) -> torch.Tensor:
        return self._as_packed().to_dense()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.dense_weight().to(dtype=input.dtype, device=input.device)
        bias = self.bias.to(input.dtype) if self.bias is not None else None
        return F.linear(input, weight, bias)

    def stored_bytes(self, scale_dtype_bits: int = 16) -> int:
        bias_bytes = self.bias.numel() * 2 if self.bias is not None else 0
        return self._as_packed().nbytes(scale_dtype_bits) + bias_bytes

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, group_size: int = 64, lambda_value: float = 0.7, scheme: str = "trit"
    ) -> "PackedTernaryLinear":
        packed = PackedTernaryWeight.from_weight(linear.weight, group_size, lambda_value, scheme)
        return cls(packed, linear.bias)


def replace_target_linears_with_packed(
    model: nn.Module, group_size: int = 64, lambda_value: float = 0.7, scheme: str = "trit"
) -> int:
    """Swap each target ``nn.Linear`` for a ``PackedTernaryLinear`` in place."""
    count = 0
    for name, module in list(_named_target_linears(model)):
        replacement = PackedTernaryLinear.from_linear(module, group_size, lambda_value, scheme)
        parent_path, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child, replacement)
        count += 1
    return count


def storage_report(
    out_features: int, in_features: int, n_groups: int, scale_dtype_bits: int = 16
) -> dict:
    """Theoretical byte cost per scheme vs fp16/int8 and the b1.58 bound."""
    numel = out_features * in_features
    scale_bytes = n_groups * out_features * scale_dtype_bits // 8
    two_bit = math.ceil(numel / 4) + scale_bytes
    trit = math.ceil(numel / 5) + scale_bytes
    ideal = math.ceil(TERNARY_BITS_PER_ELEM * numel / 8) + scale_bytes
    fp16 = numel * 2
    int8 = numel * 1
    return {
        "numel": numel,
        "scale_bytes": scale_bytes,
        "two_bit_bytes": two_bit,
        "trit_bytes": trit,
        "ideal_b158_bytes": ideal,
        "fp16_bytes": fp16,
        "int8_bytes": int8,
        "trit_compression_vs_fp16": fp16 / max(trit, 1),
        "two_bit_compression_vs_fp16": fp16 / max(two_bit, 1),
        "trit_bits_per_elem": 8 * (trit - scale_bytes) / max(numel, 1),
    }
