# coding=utf-8
"""I2_S-style export reference (per-tensor b1.58), Python-only.

Proves the export half before touching bitnet.cpp/GGUF: a model trained with
``PerTensorBitLinear`` (native per-tensor b1.58 STE) can be written as I2_S-style
artifacts (2-bit ternary codes + a single per-tensor ``gamma``) and re-imported
with the dense reconstruction ``gamma * T`` reproducing the original forward
exactly.

I2_S layout this models (bitnet.cpp's direct b1.58 path):

    gamma = mean(|W|)                       # one scalar per weight matrix
    T     = clamp(round(W / gamma), -1, 1)  # ternary {-1,0,+1}
    W_hat = gamma * T

This is the per-tensor counterpart of ``packing.PackedTernaryWeight`` and reuses
the same 2-bit code packing (``packing.pack_two_bit``). It is NOT a kernel — the
runtime model is a dense reconstruction for a reference matmul.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from . import packing as _pk
from .module import PerTensorBitLinear

EPS = 1e-12


@dataclass
class I2SWeight:
    """One I2_S-style weight: 2-bit ternary codes + a per-tensor scale."""

    out_features: int
    in_features: int
    gamma: float           # per-tensor scale = mean(|W|)
    packed: torch.Tensor   # uint8, 2-bit codes (4 trits/byte)

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> "I2SWeight":
        if weight.dim() != 2:
            raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
        w = weight.detach().float()
        gamma = w.abs().mean().clamp(min=EPS)
        ternary = torch.clamp(torch.round(w / gamma), -1.0, 1.0).to(torch.int8)
        return cls(
            out_features=weight.shape[0],
            in_features=weight.shape[1],
            gamma=float(gamma),
            packed=_pk.pack_two_bit(ternary),
        )

    def unpack_ternary(self) -> torch.Tensor:
        numel = self.out_features * self.in_features
        return _pk.unpack_two_bit(self.packed, numel).reshape(self.out_features, self.in_features)

    def to_dense(self) -> torch.Tensor:
        """Reconstruct ``gamma * T`` (== PerTensorBitLinear forward weight value)."""
        return self.gamma * self.unpack_ternary().float()

    def nbytes(self, scale_dtype_bits: int = 16) -> int:
        return int(self.packed.numel()) + scale_dtype_bits // 8  # codes + one scalar

    def state(self) -> dict:
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "gamma": self.gamma,
            "packed": self.packed,
        }

    @classmethod
    def from_state(cls, state: dict) -> "I2SWeight":
        return cls(**state)


# --------------------------------------------------------------------------- #
# Model-level export / import.
# --------------------------------------------------------------------------- #
def export_model_i2s(model: nn.Module) -> dict:
    """Export every ``PerTensorBitLinear`` to an I2_S-style artifact dict."""
    layers = {
        name: I2SWeight.from_weight(module.weight)
        for name, module in model.named_modules()
        if isinstance(module, PerTensorBitLinear)
    }
    if not layers:
        raise ValueError("no PerTensorBitLinear layers found; export expects a per-tensor-native model")
    return {"format": "i2s_per_tensor_b158", "layers": layers}


def import_model_i2s(dense_model: nn.Module, artifact: dict) -> nn.Module:
    """Load ``gamma * T`` into a plain dense model's target linears in place.

    Import target must be a plain ``nn.Linear`` model (the runtime dequantizes to
    ``gamma * T`` then does a normal matmul); importing back into a
    PerTensorBitLinear would re-quantize and change the scale.
    """
    for name, weight in artifact["layers"].items():
        module = dense_model.get_submodule(name)
        with torch.no_grad():
            module.weight.copy_(weight.to_dense().to(dtype=module.weight.dtype, device=module.weight.device))
    return dense_model


def save_i2s(artifact: dict, path) -> None:
    serial = {"format": artifact["format"],
              "layers": {name: w.state() for name, w in artifact["layers"].items()}}
    torch.save(serial, path)


def load_i2s(path) -> dict:
    serial = torch.load(path, weights_only=False)
    return {"format": serial["format"],
            "layers": {name: I2SWeight.from_state(s) for name, s in serial["layers"].items()}}


def i2s_storage_report(model: nn.Module, artifact: dict, scale_dtype_bits: int = 16) -> dict:
    """Whole-model storage: I2_S target linears + fp16 for everything else."""
    target_bytes = sum(w.nbytes(scale_dtype_bits) for w in artifact["layers"].values())
    target_numel = sum(w.out_features * w.in_features for w in artifact["layers"].values())
    all_numel = sum(p.numel() for p in model.parameters())
    other_bytes = (all_numel - target_numel) * 2
    total = target_bytes + other_bytes
    fp16_total = all_numel * 2
    return {
        "i2s_target_bytes": target_bytes,
        "fp16_other_bytes": other_bytes,
        "i2s_total_bytes": total,
        "fp16_total_bytes": fp16_total,
        "target_compression_vs_fp16": (target_numel * 2) / max(target_bytes, 1),
        "model_compression_vs_fp16": fp16_total / max(total, 1),
        "num_layers": len(artifact["layers"]),
        "target_bits_per_elem": 8 * (target_bytes - len(artifact["layers"]) * (scale_dtype_bits // 8)) / max(target_numel, 1),
    }
