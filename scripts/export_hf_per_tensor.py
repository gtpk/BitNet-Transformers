#!/usr/bin/env python3
"""RT-103A: export a tiny per-tensor-native model to an HF directory that
bitnet.cpp's converter can consume, and prove an HF save/load round-trip.

Scope is plumbing, NOT quality: build a small standard `LlamaForCausalLM`
(borrowing a known LLaMA SPM tokenizer so we don't fight tokenizer metadata),
save config + safetensors + tokenizer files, reload via
`AutoModelForCausalLM.from_pretrained`, and assert identical logits.

Path A note: the F32 GGUF must hold the LATENT fp weights (upstream re-quantizes
to I2_S), so we do NOT pre-ternarize here. The per-tensor-native STE training
that makes RT-104 parity meaningful is applied separately; this step only proves
the export pipeline.

    .venv/bin/python scripts/export_hf_per_tensor.py \
      --tokenizer-src /Users/puka/repository/bitnet.cpp/models/bitnet_b1_58-large \
      --out-dir /Users/puka/repository/bitnet.cpp/models/tiny_pt_native
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TOKENIZER_FILES = [
    "tokenizer.model", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-src", type=Path, required=True,
                    help="HF dir to borrow tokenizer files + vocab_size from")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--intermediate-size", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-kv-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

    src_cfg = json.loads((args.tokenizer_src / "config.json").read_text())
    vocab_size = src_cfg["vocab_size"]
    print(f"borrowing tokenizer + vocab_size={vocab_size} from {args.tokenizer_src}")

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=src_cfg.get("max_position_embeddings", 2048),
        rms_norm_eps=src_cfg.get("rms_norm_eps", 1e-5),
        rope_theta=src_cfg.get("rope_theta", 10000.0),
        bos_token_id=src_cfg.get("bos_token_id", 1),
        eos_token_id=src_cfg.get("eos_token_id", 2),
        pad_token_id=src_cfg.get("pad_token_id", None),
        tie_word_embeddings=src_cfg.get("tie_word_embeddings", True),
        torch_dtype="float32",
    )
    torch.manual_seed(args.seed)
    model = LlamaForCausalLM(config).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"built tiny LlamaForCausalLM: {n:,} params, {args.num_layers} layers, hidden {args.hidden_size}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    copied = []
    for fn in TOKENIZER_FILES:
        src = args.tokenizer_src / fn
        if src.exists():
            shutil.copyfile(src, args.out_dir / fn)
            copied.append(fn)
    print(f"saved model + copied tokenizer files: {copied}")
    print("out dir:", sorted(p.name for p in args.out_dir.iterdir()))

    # --- round-trip: reload and compare logits ---
    ids = torch.randint(0, vocab_size, (2, 16))
    with torch.no_grad():
        ref = model(input_ids=ids).logits
    reloaded = AutoModelForCausalLM.from_pretrained(args.out_dir, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        got = reloaded(input_ids=ids).logits
    err = float((ref - got).abs().max())
    print(f"\nRT-103A model logit round-trip: max_err={err:.2e} -> {'PASS' if err < 1e-5 else 'FAIL'}")

    # best-effort tokenizer reload (real validity is RT-103B / converter's job)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.out_dir)
        enc = tok("The history of timekeeping began", return_tensors="pt")
        print(f"tokenizer reload OK: {tok.__class__.__name__}, "
              f"vocab={tok.vocab_size}, sample_len={enc.input_ids.shape[1]}")
    except Exception as e:
        print(f"tokenizer reload via AutoTokenizer needs attention (RT-103B): {type(e).__name__}: {str(e)[:160]}")

    raise SystemExit(0 if err < 1e-5 else 1)


if __name__ == "__main__":
    main()
