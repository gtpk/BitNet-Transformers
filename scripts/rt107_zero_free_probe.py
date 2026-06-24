#!/usr/bin/env python3
"""RT-107: zero-free probe to confirm/deny the I2_S zero-code hypothesis.

Materialize a ZERO-FREE ternary variant of the trained model:
    gamma = mean(|W|);  T = sign(W) (>=0 -> +1, <0 -> -1, NO zeros);  W_sign = gamma*T
so every weight is +-gamma (codes {00,10} only, like the official pure-sign models).
Export via the same path; compare F16 vs I2_S PPL.

  I2_S(W_sign) ~= F16(W_sign)  -> zero-code path is the culprit (our b1.58 zeros break I2_S)
  I2_S(W_sign) still collapses -> not zeros; I2_S runtime/shape/metadata issue on this build

    .venv/bin/python scripts/rt107_zero_free_probe.py \
      --in-dir /Users/puka/repository/bitnet.cpp/models/tiny_pt_trained \
      --out-dir /Users/puka/repository/bitnet.cpp/models/tiny_pt_sign
"""
from __future__ import annotations
import argparse, math, shutil, sys
from pathlib import Path
import torch, torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from bitnet_llama import conversion as C  # noqa: E402

TOK = ["tokenizer.model","tokenizer.json","tokenizer_config.json","special_tokens_map.json","added_tokens.json","eval.txt"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    model = AutoModelForCausalLM.from_pretrained(args.in_dir, dtype=torch.float32).eval()
    n = 0
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
                W = m.weight.float()
                gamma = W.abs().mean().clamp(min=1e-12)
                T = torch.where(W >= 0, 1.0, -1.0)           # pure sign, NO zeros
                m.weight.copy_(gamma * T)
                n += 1
    print(f"materialized {n} target linears to gamma*sign(W) (zero-free; codes will be {{00,10}})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    for fn in TOK:
        s = args.in_dir / fn
        if s.exists(): shutil.copyfile(s, args.out_dir / fn)

    tok = PreTrainedTokenizerFast(tokenizer_file=str(args.out_dir / "tokenizer.json"))
    ids = torch.tensor(tok((args.out_dir / "eval.txt").read_text())["input_ids"], dtype=torch.long)
    nseq = ids.numel() // args.seq_len
    x = ids[: nseq * args.seq_len].reshape(nseq, args.seq_len)
    with torch.no_grad():
        loss = float(model(input_ids=x, labels=x).loss)
    print(f"Python PPL (zero-free W_sign) = {math.exp(min(loss,20)):.2f}")
    print(f"saved {args.out_dir}")


if __name__ == "__main__":
    main()
