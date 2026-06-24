#!/usr/bin/env python3
"""I2_S export PoC test cases (Python reference, no bitnet.cpp yet).

  PTX-101 layer round-trip  : I2SWeight.to_dense() == PerTensorBitLinear forward weight
  PTX-102 model round-trip  : imported dense model logits == per-tensor-native model
  PTX-103 save/load          : load(save(artifact)) reproduces the same logits
  PTX-104 storage            : I2_S target bytes vs fp16 ratio recorded
  PTX-105 tiny-text PPL       : CE/PPL of base / native / imported recorded; native==imported

    .venv/bin/python scripts/check_i2s_export.py --json-out reports/i2s_export_tc.json --strict
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama import i2s_export as X  # noqa: E402
from bitnet_llama.module import PerTensorBitLinear  # noqa: E402


def _check(results, name, passed, detail=""):
    results.append({"id": name, "pass": bool(passed), "detail": detail})
    print(f"{'PASS' if passed else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


def build_tiny_model(seed):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).eval(), cfg


def to_per_tensor_native(base):
    """Copy base into a model whose target linears are PerTensorBitLinear."""
    model = copy.deepcopy(base)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            repl = PerTensorBitLinear(module.in_features, module.out_features,
                                      bias=module.bias is not None, activation_bits=None)
            with torch.no_grad():
                repl.weight.copy_(module.weight)
                if module.bias is not None:
                    repl.bias.copy_(module.bias)
            parent_path, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_path) if parent_path else model, child, repl)
    return model.eval()


def eval_ce(model, ids):
    with torch.no_grad():
        return float(model(input_ids=ids, labels=ids).loss)


@torch.no_grad()
def run(seed=7):
    results = []
    base, cfg = build_tiny_model(seed)
    pt_model = to_per_tensor_native(base)

    # PTX-101: per-layer reconstruction equals the layer's forward weight value
    layer = [m for m in pt_model.modules() if isinstance(m, PerTensorBitLinear)][0]
    fwd_w = layer.quantize_weight().detach()
    rt_w = X.I2SWeight.from_weight(layer.weight).to_dense()
    err = float((rt_w - fwd_w).abs().max())
    _check(results, "PTX-101/layer-roundtrip", err < 1e-6, f"max_err={err:.2e}")

    # text-ish eval batch from the committed byte fixture (vocab 256)
    text = (REPO_ROOT / "data/tiny_corpus.txt").read_text(encoding="utf-8")
    ids = torch.tensor(list(text.encode("utf-8"))[: 8 * 64], dtype=torch.long).reshape(8, 64)

    ref_logits = pt_model(input_ids=ids).logits.detach()

    # PTX-102: export -> import into a dense model -> identical logits
    artifact = X.export_model_i2s(pt_model)
    imported = X.import_model_i2s(copy.deepcopy(base), artifact)
    imp_logits = imported(input_ids=ids).logits.detach()
    err2 = float((imp_logits - ref_logits).abs().max())
    _check(results, "PTX-102/model-roundtrip", err2 < 1e-5, f"max_logit_err={err2:.2e}")

    # PTX-103: save/load artifact -> import -> identical logits
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.i2s"
        X.save_i2s(artifact, path)
        loaded = X.load_i2s(path)
        imported2 = X.import_model_i2s(copy.deepcopy(base), loaded)
        err3 = float((imported2(input_ids=ids).logits.detach() - ref_logits).abs().max())
    _check(results, "PTX-103/save-load", err3 < 1e-5, f"max_logit_err={err3:.2e}")

    # PTX-104: storage ratio
    rep = X.i2s_storage_report(pt_model, artifact)
    _check(results, "PTX-104/storage", rep["model_compression_vs_fp16"] > 1.0,
           f"target {rep['target_compression_vs_fp16']:.2f}x, model {rep['model_compression_vs_fp16']:.2f}x, "
           f"target_bits/elem={rep['target_bits_per_elem']:.2f}")

    # PTX-105: tiny-text PPL recorded; native and imported must match
    base_ce, pt_ce, imp_ce = eval_ce(base, ids), eval_ce(pt_model, ids), eval_ce(imported, ids)
    ppl = lambda x: math.exp(min(x, 20.0))
    _check(results, "PTX-105/ppl-preserved", abs(pt_ce - imp_ce) < 1e-4,
           f"PPL base={ppl(base_ce):.2f} native={ppl(pt_ce):.2f} imported={ppl(imp_ce):.2f} "
           f"(native-vs-imported CE delta={abs(pt_ce-imp_ce):.2e})")

    info = {"storage": rep, "ce": {"base": base_ce, "native": pt_ce, "imported": imp_ce}}
    return results, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=Path, default=Path("reports/i2s_export_tc.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results, info = run(args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"results": results, "info": info}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")
    if args.strict and not all(r["pass"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
