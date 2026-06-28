"""I2_S + LoRA / residual sidecar (docs/i2s_lora_sidecar_plan.md).

For one target linear, keep the I2_S ternary base and add a tiny trainable low-rank residual:

    y = gamma*T*x + s * B(A x),   A: r x in,  B: out x r,  s = alpha / r

The base is a PerTensorBitLinear (STE gamma*T, I2_S-exportable). The sidecar is the part of the
FP function one ternary plane cannot express. B is zero-initialized so the wrapped module starts
EXACTLY at the base behaviour (rank-0 reproducible). This is the cheapest test of "1.58 alone may
lack capacity -> add a tiny correction" -- mostly-I2_S + tiny sidecar, NOT pure I2_S.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from bitnet_llama.module import PerTensorBitLinear
from bitnet_llama import conversion as C


class I2SLoRALinear(nn.Module):
    """Wrap a PerTensorBitLinear base with a rank-r LoRA sidecar. base(x) + scale*B(A(x))."""

    def __init__(self, base: PerTensorBitLinear, rank: int, alpha: float, init: str = "zero"):
        super().__init__()
        self.base = base  # PerTensorBitLinear; .weight is the latent FP weight (STE -> gamma*T)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scale = alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Linear(self.in_features, rank, bias=False).to(device=dev, dtype=dt)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False).to(device=dev, dtype=dt)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        if init == "zero":
            nn.init.zeros_(self.lora_B.weight)  # start at base behaviour (sidecar contributes 0)
        elif init == "random":
            nn.init.normal_(self.lora_B.weight, std=1e-3)
        elif init == "svd_residual":
            self._init_svd_residual()
        else:
            raise ValueError(f"unknown sidecar init: {init}")

    @torch.no_grad()
    def _init_svd_residual(self):
        # initialize A,B to the top-r SVD modes of the ternary residual E = W - gamma*T
        W = self.base.weight.detach().float()
        E = W - C.per_tensor_b158_approx(W)
        U, S, Vh = torch.linalg.svd(E, full_matrices=False)
        r = self.rank
        Br = (U[:, :r] * S[:r].sqrt()) / self.scale  # fold the scale so forward reproduces E modes
        Ar = (S[:r].sqrt().unsqueeze(1) * Vh[:r, :])
        self.lora_B.weight.copy_(Br.to(self.lora_B.weight.dtype))
        self.lora_A.weight.copy_(Ar.to(self.lora_A.weight.dtype))

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))

    @torch.no_grad()
    def effective_weight(self):
        """Dense weight that reproduces the forward: gamma*T + scale * (B @ A). For materialize/score
        only -- the deployable form keeps gamma*T as I2_S and B,A separate."""
        return C.per_tensor_b158_approx(self.base.weight) + self.scale * (self.lora_B.weight @ self.lora_A.weight)


def _set_module(root, dotted, mod):
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], mod)


_TARGET_GROUPS = {
    "all": None,
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


def wrap_targets_with_lora(model, rank, alpha, target="all", init="zero", top_layers=0, n_layers=0,
                           layer_names=None):
    """Replace each PerTensorBitLinear (within the target group) with an I2SLoRALinear. Returns the
    count wrapped. layer_names (exact module-name list, EGROW-002) OVERRIDES the group: wrap only
    those. Else target in {all, attn, mlp, top_saliency}; top_saliency wraps the last `top_layers`
    blocks."""
    if rank <= 0:
        return 0
    names_set = set(layer_names) if layer_names else None
    suffixes = _TARGET_GROUPS.get(target if target != "top_saliency" else "all")
    import re
    lo = (n_layers - top_layers) if (target == "top_saliency" and top_layers and n_layers) else None
    wrapped = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, PerTensorBitLinear):
            continue
        if names_set is not None:
            if name not in names_set:
                continue
        else:
            if suffixes is not None and not any(name.endswith(s) for s in suffixes):
                continue
            if lo is not None:
                m = re.search(r"\.layers\.(\d+)\.", name + ".")
                if not (m and int(m.group(1)) >= lo):
                    continue
        _set_module(model, name, I2SLoRALinear(mod, rank, alpha, init))
        wrapped += 1
    return wrapped


def sidecar_accounting(model, i2s_bits_per_weight=2.0):
    """Bytes/ops overhead of the sidecar vs the I2_S target-linear bytes (SIDE-000).
    I2_S target bytes ~= in*out * i2s_bits_per_weight / 8 (project: ~2 bits/weight = 16x vs f32)."""
    sc_params = sc_lin = 0
    i2s_weights = 0
    for mod in model.modules():
        if isinstance(mod, I2SLoRALinear):
            sc_params += mod.rank * (mod.in_features + mod.out_features)
            sc_lin += 1
            i2s_weights += mod.in_features * mod.out_features
        elif isinstance(mod, PerTensorBitLinear):
            i2s_weights += mod.in_features * mod.out_features
    sc_bytes = 2 * sc_params  # fp16 sidecar
    i2s_bytes = i2s_weights * i2s_bits_per_weight / 8.0
    return {
        "sidecar_linears": sc_lin,
        "sidecar_params": sc_params,
        "sidecar_bytes_fp16": sc_bytes,
        "i2s_target_bytes_est": int(i2s_bytes),
        "sidecar_bytes_ratio_vs_target_i2s": round(sc_bytes / max(i2s_bytes, 1), 4),
    }
