#!/usr/bin/env python3
"""RT-104D: Path A' — feed ALREADY-ternarized dense weights to upstream I2_S.

Insight (proved): upstream I2_S quantize is sign(W) x max|W|. If we feed the
per-tensor b1.58 *materialized* weights Wq = gamma*T (gamma=mean|W|, T in
{-1,0,+1}) instead of the latent FP weights, then max|Wq|=gamma, sign(Wq)=T, and
zeros stay zeros -> Q_max(Wq) = gamma*T = Wq. Lossless repack, no Path B writer.

This script materializes Wq into a dense HF model (from the RT-104A latent dir),
so RT-104D-export can convert+quantize it and we check parity.

    .venv/bin/python scripts/rt104d_quantized_dense.py \
      --in-dir  /Users/puka/repository/bitnet.cpp/models/tiny_pt_trained \
      --out-dir /Users/puka/repository/bitnet.cpp/models/tiny_pt_ternary
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402

TOKENIZER_FILES = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "eval.txt"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    model = AutoModelForCausalLM.from_pretrained(args.in_dir, dtype=torch.float32).eval()

    # materialize per-tensor b1.58: W -> gamma*T (gamma=mean|W|, T=clamp(round(W/gamma),-1,1))
    n_q = 0
    gammas = {}
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
                Wq = C.per_tensor_b158_approx(mod.weight)   # = gamma * T
                mod.weight.copy_(Wq)
                gammas[name] = float(mod.weight.abs().max())  # == gamma (max|Wq|)
                n_q += 1
    print(f"materialized {n_q} target linears to gamma*T (per-tensor b1.58)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    for fn in TOKENIZER_FILES:
        s = args.in_dir / fn
        if s.exists():
            shutil.copyfile(s, args.out_dir / fn)

    # Python PPL of the ternary-dense model on eval.txt (our reference for parity)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(args.out_dir / "tokenizer.json"))
    eval_ids = torch.tensor(tok((args.out_dir / "eval.txt").read_text())["input_ids"], dtype=torch.long)
    nseq = eval_ids.numel() // args.seq_len
    ids = eval_ids[: nseq * args.seq_len].reshape(nseq, args.seq_len)
    with torch.no_grad():
        loss = float(model(input_ids=ids, labels=ids).loss)
    print(f"Python PPL (ternary-dense Wq) on eval.txt = {math.exp(min(loss,20)):.3f}  (loss {loss:.4f})")
    # report a couple gammas (these should equal the stored I2_S scale after upstream quantize)
    for k in list(gammas)[:3]:
        print(f"  gamma(max|Wq|) {k} = {gammas[k]:.6g}")
    print(f"\nSaved ternary-dense HF dir to {args.out_dir}")
    print("Next: convert --outtype f32, llama-quantize I2_S, then llama-perplexity on eval.txt;")
    print("expect PPL near the Python value (lossless repack), not the ~62k latent-path collapse.")


if __name__ == "__main__":
    main()
