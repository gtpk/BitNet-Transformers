#!/usr/bin/env python3
"""RT-104A: build a TRAINED tiny per-tensor-native model + Python reference.

Uses the LLaMA SPM tokenizer (not byte) so bitnet.cpp and Python see the same
token stream later. Trains a tiny LlamaForCausalLM with PerTensorBitLinear
(native per-tensor b1.58 STE) on a small text corpus, then records the Python
reference the bitnet.cpp I2_S run must match in RT-104C:

  - latent-fp model PPL       (what the F32 GGUF holds; upstream re-quantizes it)
  - i2s_export model PPL       (gamma*T reconstruction -> the parity target)
  - per-tensor STE model PPL   (the trained model's own forward)

Also saves the trained model as an HF dir (latent fp weights, Path A) + the eval
text, for RT-104B/C.

    .venv/bin/python scripts/rt104_build_reference.py \
      --tokenizer-src /Users/puka/repository/bitnet.cpp/models/bitnet_b1_58-large \
      --corpus data/tiny_corpus.txt \
      --out-dir /Users/puka/repository/bitnet.cpp/models/tiny_pt_trained \
      --json-out reports/rt104_reference.json --train-steps 300
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama import i2s_export as X  # noqa: E402
from bitnet_llama.module import PerTensorBitLinear  # noqa: E402

TOKENIZER_FILES = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json"]


def is_target(name):
    return C.is_target_weight_key(f"{name}.weight")


def set_module(root, name, mod):
    parent_path, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent_path) if parent_path else root, child, mod)


def replace_with_per_tensor(model):
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and is_target(name):
            repl = PerTensorBitLinear(m.in_features, m.out_features, bias=m.bias is not None)
            with torch.no_grad():
                repl.weight.copy_(m.weight)
                if m.bias is not None:
                    repl.bias.copy_(m.bias)
            set_module(model, name, repl)
            n += 1
    return n


def windows(tokens, seq_len, batch, seed):
    g = torch.Generator().manual_seed(seed)
    usable = tokens.numel() - 1
    starts = torch.randint(0, max(1, usable - seq_len), (batch,), generator=g)
    x = torch.stack([tokens[s:s + seq_len] for s in starts.tolist()]).long()
    return x


@torch.no_grad()
def ppl(model, eval_tokens, seq_len):
    model.eval()
    n = eval_tokens.numel() // seq_len
    if n == 0:
        seq_len = eval_tokens.numel(); n = 1
    ids = eval_tokens[: n * seq_len].reshape(n, seq_len)
    loss = float(model(input_ids=ids, labels=ids).loss)
    return loss, math.exp(min(loss, 20.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-src", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, default=Path("reports/rt104_reference.json"))
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--intermediate-size", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(tokenizer_file=str(args.tokenizer_src / "tokenizer.json"))
    src_cfg = json.loads((args.tokenizer_src / "config.json").read_text())
    vocab_size = src_cfg["vocab_size"]
    text = args.corpus.read_text(encoding="utf-8")
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long)
    split = int(ids.numel() * 0.85)
    train_tokens, eval_tokens = ids[:split], ids[split:]
    print(f"tokenizer vocab={vocab_size} | corpus tokens: {ids.numel()} (train {train_tokens.numel()}, eval {eval_tokens.numel()})")

    config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=args.hidden_size, intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_layers, num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_heads, max_position_embeddings=src_cfg.get("max_position_embeddings", 2048),
        rms_norm_eps=src_cfg.get("rms_norm_eps", 1e-5), bos_token_id=src_cfg.get("bos_token_id", 1),
        eos_token_id=src_cfg.get("eos_token_id", 2), pad_token_id=src_cfg.get("pad_token_id", None),
        tie_word_embeddings=src_cfg.get("tie_word_embeddings", True), torch_dtype="float32",
    )
    torch.manual_seed(args.seed)
    model = LlamaForCausalLM(config)
    replace_with_per_tensor(model)

    # train (native per-tensor STE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    losses = []
    for step in range(args.train_steps):
        x = windows(train_tokens, args.seq_len, args.batch_size, args.seed + step)
        out = model(input_ids=x, labels=x)
        opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
        losses.append(float(out.loss.detach()))
    model.eval()
    print(f"train loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Python references on the eval tokens
    pt_loss, pt_ppl = ppl(model, eval_tokens, args.seq_len)
    # latent fp dense model (what F32 GGUF holds)
    latent = copy.deepcopy(model)
    for name, m in list(latent.named_modules()):
        if isinstance(m, PerTensorBitLinear):
            dense = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None)
            with torch.no_grad():
                dense.weight.copy_(m.weight)
                if m.bias is not None: dense.bias.copy_(m.bias)
            set_module(latent, name, dense)
    latent.eval()
    latent_loss, latent_ppl = ppl(latent, eval_tokens, args.seq_len)
    # i2s_export reconstruction (gamma*T) -> parity target
    artifact = X.export_model_i2s(model)
    i2s_ref = X.import_model_i2s(copy.deepcopy(latent), artifact)
    i2s_loss, i2s_ppl = ppl(i2s_ref, eval_tokens, args.seq_len)
    print(f"PPL  per_tensor_ste={pt_ppl:.3f}  latent_fp={latent_ppl:.3f}  i2s_export(gamma*T)={i2s_ppl:.3f}")

    # generation smoke on the i2s reference
    prompt = eval_tokens[: min(8, eval_tokens.numel())].unsqueeze(0)
    with torch.no_grad():
        g = prompt.clone()
        for _ in range(16):
            lg = i2s_ref(input_ids=g).logits[:, -1, :]
            g = torch.cat([g, lg.argmax(-1, keepdim=True)], dim=1)
    gen = g[0, prompt.shape[1]:].tolist()
    gen_ok = bool(torch.isfinite(i2s_ref(input_ids=prompt).logits).all()) and len(set(gen)) > 1
    print(f"generation smoke (i2s ref): finite+nondegenerate={gen_ok}, unique={len(set(gen))}")

    # save trained model as HF dir (latent fp weights, Path A) + tokenizer + eval text
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latent.save_pretrained(args.out_dir, safe_serialization=True)
    for fn in TOKENIZER_FILES:
        s = args.tokenizer_src / fn
        if s.exists(): shutil.copyfile(s, args.out_dir / fn)
    (args.out_dir / "eval.txt").write_text(tok.decode(eval_tokens.tolist()), encoding="utf-8")

    # HF round-trip check
    from transformers import AutoModelForCausalLM
    rl = AutoModelForCausalLM.from_pretrained(args.out_dir, dtype=torch.float32).eval()
    chk = eval_tokens[: args.seq_len].unsqueeze(0)
    with torch.no_grad():
        rt_err = float((latent(input_ids=chk).logits - rl(input_ids=chk).logits).abs().max())
    print(f"HF export round-trip max_err={rt_err:.2e}")

    payload = {
        "tokenizer": "llama_spm(borrowed)", "vocab_size": vocab_size,
        "config": {"hidden": args.hidden_size, "intermediate": args.intermediate_size,
                   "layers": args.num_layers, "heads": args.num_heads, "seq_len": args.seq_len},
        "train_loss_start": losses[0], "train_loss_end": losses[-1],
        "eval_tokens": int(eval_tokens.numel()),
        "ppl": {"per_tensor_ste": pt_ppl, "latent_fp": latent_ppl, "i2s_export_gamma_T": i2s_ppl},
        "loss": {"per_tensor_ste": pt_loss, "latent_fp": latent_loss, "i2s_export_gamma_T": i2s_loss},
        "generation_nondegenerate": gen_ok,
        "hf_roundtrip_max_err": rt_err,
        "out_dir": str(args.out_dir),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}  and HF dir {args.out_dir}")
    ok = losses[-1] < losses[0] and gen_ok and rt_err < 1e-4
    print("RT-104A:", "PASS" if ok else "CHECK", "(train loss down, generation ok, round-trip exact)")


if __name__ == "__main__":
    main()
