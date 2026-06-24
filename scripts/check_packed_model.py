#!/usr/bin/env python3
"""Phase 2 test cases: model-wide packed ternary export/import.

Proves that packing a whole model preserves its output exactly and reports the
honest end-to-end storage reduction.

  PACK-101 logit equality : pack -> unpack model logits == S1-converted model
  PACK-102 save/load model : load(save(artifact)) reproduces the same logits
  PACK-103 model storage   : whole-model packed bytes < fp16, ratio reported

    .venv/bin/python scripts/check_packed_model.py --json-out reports/packed_model_tc.json --strict
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama import packing as P  # noqa: E402


def _check(results: list, name: str, passed: bool, detail: str = "") -> None:
    results.append({"id": name, "pass": bool(passed), "detail": detail})
    print(f"{'PASS' if passed else 'FAIL'} {name}{(': ' + detail) if detail else ''}")


def build_tiny_model(seed: int):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config).eval(), config


@torch.no_grad()
def logits_of(model, input_ids):
    return model(input_ids=input_ids).logits.detach()


def run(seed: int = 7) -> tuple[list, dict]:
    results: list = []
    model, config = build_tiny_model(seed)
    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, config.vocab_size, (2, 32))

    # Reference: the S1-converted model (alpha*T applied to target linears).
    ref_state, _ = C.convert_state_dict(model.state_dict(), C.S1)
    ref_model = copy.deepcopy(model)
    ref_model.load_state_dict(ref_state)
    ref_model.eval()
    ref_logits = logits_of(ref_model, input_ids)

    # PACK-101: pack original -> unpack into a copy -> logits must match reference
    artifact = P.pack_model(model, group_size=64, lambda_value=0.7, scheme="trit")
    packed_model = copy.deepcopy(model)
    P.unpack_into_model(packed_model, artifact)
    packed_logits = logits_of(packed_model, input_ids)
    err = float((packed_logits - ref_logits).abs().max())
    _check(results, "PACK-101/logit-equality", err < 1e-5, f"max_logit_err={err:.2e}")

    # PACK-102: save/load artifact -> unpack into a fresh copy -> same logits
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "packed_model.pt"
        P.save_packed_model(artifact, path)
        loaded = P.load_packed_model(path)
        reloaded_model = copy.deepcopy(model)
        P.unpack_into_model(reloaded_model, loaded)
        err2 = float((logits_of(reloaded_model, input_ids) - ref_logits).abs().max())
    _check(results, "PACK-102/save-load-model", err2 < 1e-5, f"max_logit_err={err2:.2e}")

    # PACK-103: whole-model storage report
    rep = P.model_storage_report(model, artifact)
    _check(
        results,
        "PACK-103/model-storage",
        rep["model_compression_vs_fp16"] > 1.0 and rep["num_packed_layers"] > 0,
        f"{rep['num_packed_layers']} layers packed, "
        f"target_fraction={rep['target_fraction']:.2f}, "
        f"compression={rep['model_compression_vs_fp16']:.2f}x "
        f"({rep['packed_total_bytes']/1024:.1f}KB vs fp16 {rep['fp16_total_bytes']/1024:.1f}KB)",
    )
    return results, rep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", type=Path, default=Path("reports/packed_model_tc.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results, rep = run(args.seed)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"results": results, "storage": rep}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")

    if args.strict and not all(item["pass"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
