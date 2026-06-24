#!/usr/bin/env python3
"""RT-106: isolate the bitnet-vs-Python PPL gap by adding PRE-LINEAR activation
fake-quant to the Python reference and sweeping modes.

The Python model uses the SAME weights bitnet runs (ternary-dense Wq = gamma*T,
from models/tiny_pt_ternary). bitnet.cpp feeds int8 matmul inputs, so we must
quantize the INPUT of each target linear (not just post-linear outputs). We sweep
{none, int8 per-tensor, int8 per-token} and compare PPL on eval.txt to bitnet's
21384.

Verdict:
  Python int8-act PPL ~= 21384  -> weights faithful, gap was activation -> Path A'
  Python int8-act PPL ~= 2000   -> activation not the cause -> codes/order/writer
  in between                    -> both; revisit parser/dequant

    .venv/bin/python scripts/rt106_activation_sweep.py --dir /Users/puka/repository/bitnet.cpp/models/tiny_pt_ternary
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from bitnet_llama import conversion as C  # noqa: E402

EPS = 1e-12


def quant_act(x, mode, bits=8):
    if mode == "none":
        return x
    qmax = (1 << (bits - 1)) - 1  # 127 for int8 (symmetric)
    if mode == "int8_per_tensor":
        g = x.detach().abs().amax().clamp(min=EPS)
    elif mode == "int8_per_token":
        g = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=EPS)
    else:
        raise ValueError(mode)
    scale = g / qmax
    return torch.clamp(torch.round(x / scale), -qmax, qmax) * scale


class ActQuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, mode: str):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.mode = mode

    def forward(self, x):
        return F.linear(quant_act(x, self.mode), self.weight, self.bias)


def wrap_targets(model, mode):
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            parent_path, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_path) if parent_path else model, child, ActQuantLinear(m, mode))
    return model


@torch.no_grad()
def ppl(model, ids, seq_len):
    nseq = ids.numel() // seq_len
    x = ids[: nseq * seq_len].reshape(nseq, seq_len)
    return float(model(input_ids=x, labels=x).loss)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--bitnet-ppl", type=float, default=21384.0)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(args.dir / "tokenizer.json"))
    eval_ids = torch.tensor(tok((args.dir / "eval.txt").read_text())["input_ids"], dtype=torch.long)
    print(f"eval tokens: {eval_ids.numel()}, seq_len {args.seq_len}, bitnet PPL ref = {args.bitnet_ppl}")

    results = {}
    for mode in ["none", "int8_per_tensor", "int8_per_token"]:
        model = AutoModelForCausalLM.from_pretrained(args.dir, dtype=torch.float32).eval()
        wrap_targets(model, mode)
        loss = ppl(model, eval_ids, args.seq_len)
        p = math.exp(min(loss, 20.0))
        results[mode] = p
        print(f"  act={mode:16} loss={loss:.4f}  PPL={p:.2f}")

    base = results["none"]
    closest = min(results, key=lambda k: abs(results[k] - args.bitnet_ppl))
    print(f"\nclosest-to-bitnet mode: {closest} (PPL {results[closest]:.1f} vs bitnet {args.bitnet_ppl})")
    if results[closest] >= 0.6 * args.bitnet_ppl:
        print("VERDICT: activation quant explains most of the gap -> weights faithful -> Path A'")
    elif base < 4000 and all(v < 0.5 * args.bitnet_ppl for v in results.values()):
        print("VERDICT: activation does NOT explain the gap -> codes/order/writer issue -> Path B / dequant parity")
    else:
        print("VERDICT: partial -> activation + weight/order both contribute; revisit parser/dequant")


if __name__ == "__main__":
    main()
