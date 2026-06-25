#!/usr/bin/env python3
"""RT-114 / SCALE-001: pretrained small-model scale-up of the I2_S export path.

RT-112/113 closed the tiny artifact (per-tensor-native -> I2_S is correct AND
efficient on x86). This checks the gain is not a tiny-toy artifact: take a REAL
pretrained LLaMA (default JackFram/llama-160m, linear-dominated), materialize the
per-tensor b1.58 weights Wq=gamma*T on the target linears (PTQ, NO retrain), and
measure storage / latency / f16-vs-i2_s parity at scale.

Quality is judged by f16-vs-i2_s PARITY, not absolute PPL: PTQ-to-ternary without
adaptation has poor absolute quality by design (RT-104), but i2_s must still equal
f16 of the SAME materialized weights (the runtime-faithfulness claim, SCALE-001d).

To keep the parity test clean, embedding AND lm_head are forced to f16 in the i2_s
artifact (--token-embedding-type f16 --output-tensor-type f16), so ONLY the 84
materialized target linears become I2_S; every other tensor is byte-identical
between the f16 and i2_s GGUFs. Since Wq=gamma*T satisfies max|Wq|=gamma /
sign(Wq)=T, upstream sign*absmax I2_S is lossless on it -> i2_s == f16.

PREREQUISITES: bitnet.cpp built on x86 (see rt112_x86_arena.py). Run from repo root.

USAGE:
  python scripts/rt114_scaleup.py --bitnet /content/bitnet.cpp \
    --model-id JackFram/llama-160m --ctx 64 --threads 2
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402

PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")

# A few real English sentences; repeated to give the perplexity tool enough tokens.
EVAL_PARA = (
    "The history of science is the study of the development of the natural world. "
    "Researchers observe phenomena, form hypotheses, and test them with experiments. "
    "Over centuries this method has reshaped medicine, transport, and communication. "
    "A small change in one field often unlocks unexpected progress in another. "
)


def run(cmd, cwd=None):
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print((r.stdout + r.stderr)[-500:])
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return r.stdout + r.stderr


def materialize_ternary(fp_dir: Path, ternary_dir: Path):
    """Copy the FP model with target linears replaced by Wq = gamma*T (per-tensor b1.58)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(fp_dir, dtype=torch.float32).eval()
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
                mod.weight.copy_(C.per_tensor_b158_approx(mod.weight))
                n += 1
    ternary_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ternary_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(fp_dir).save_pretrained(ternary_dir)
    print(f"materialized {n} target linears to gamma*T -> {ternary_dir}")
    return n


def build_ggufs(bitnet: Path, hf_dir: Path, convert: Path):
    """f32, f16, and i2_s GGUFs in hf_dir. Embedding+lm_head kept f16 in i2_s so only
    the materialized target linears are quantized (clean parity)."""
    f32, f16 = hf_dir / "ggml-model-f32.gguf", hf_dir / "ggml-model-f16.gguf"
    i2s = hf_dir / "ggml-model-i2_s.gguf"
    quant = bitnet / "build/bin/llama-quantize"
    if not f32.exists():
        run(f'python "{convert}" "{hf_dir}" --outtype f32')
    if not f16.exists():
        run(f'python "{convert}" "{hf_dir}" --outtype f16')
    if not i2s.exists():
        run(f'"{quant}" --token-embedding-type f16 --output-tensor-type f16 '
            f'"{f32}" "{i2s}" I2_S 1 1')
    return {"f32": f32, "f16": f16, "i2_s": i2s}


def perplexity(bitnet: Path, gguf: Path, eval_txt: Path, ctx: int, threads: int):
    cmd = f'{bitnet}/build/bin/llama-perplexity -m "{gguf}" -f "{eval_txt}" -c {ctx} -t {threads}'
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = PPL_RE.search(out)
    if not m:
        print("  !! no Final estimate. tail:\n" + out[-400:])
        return None
    print(f"  PPL = {m.group(1)}")
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitnet", type=Path, required=True)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--work", type=Path, default=None, help="default: <bitnet>/models")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-repeat", type=int, default=40)
    ap.add_argument("--with-latent", action="store_true",
                    help="also build+ppl the un-materialized latent model (Path A control)")
    args = ap.parse_args()

    bitnet = args.bitnet.resolve()
    work = (args.work or bitnet / "models").resolve()
    convert = bitnet / "utils/convert-hf-to-gguf-bitnet.py"
    py = sys.executable
    slug = args.model_id.split("/")[-1]
    fp_dir, tern_dir = work / f"{slug}_fp", work / f"{slug}_ternary"

    for p, what in [(bitnet / "build/bin/llama-quantize", "llama-quantize"),
                    (bitnet / "build/bin/llama-perplexity", "llama-perplexity"),
                    (convert, "convert-hf-to-gguf-bitnet.py")]:
        if not p.exists():
            raise SystemExit(f"missing prerequisite {what}: {p}")

    # 1) download pretrained FP model
    if not (fp_dir / "config.json").exists():
        from huggingface_hub import snapshot_download
        fp_dir.mkdir(parents=True, exist_ok=True)
        print(f">>> downloading {args.model_id} -> {fp_dir}")
        snapshot_download(args.model_id, local_dir=str(fp_dir))

    # 2) materialize Wq=gamma*T on target linears
    n_lin = materialize_ternary(fp_dir, tern_dir)

    # 3) eval text (parity is relative; absolute value irrelevant)
    eval_txt = tern_dir / "eval.txt"
    eval_txt.write_text(EVAL_PARA * args.eval_repeat, encoding="utf-8")

    # 4) build GGUFs + perplexity parity for the ternary (Path A') model
    ggufs = build_ggufs(bitnet, tern_dir, convert)
    ppl = {fmt: perplexity(bitnet, g, eval_txt, args.ctx, args.threads) for fmt, g in ggufs.items()}

    # optional latent control (Path A)
    ppl_lat = {}
    if args.with_latent:
        (fp_dir / "eval.txt").write_text(EVAL_PARA * args.eval_repeat, encoding="utf-8")
        lat_ggufs = build_ggufs(bitnet, fp_dir, convert)
        ppl_lat = {fmt: perplexity(bitnet, g, fp_dir / "eval.txt", args.ctx, args.threads)
                   for fmt, g in lat_ggufs.items()}

    # 5) storage + latency via the RT-113 driver on the ternary model
    run(f'{py} scripts/rt113_storage_latency.py --bitnet "{bitnet}" '
        f'--model-dir "{tern_dir}" --threads {args.threads} '
        f'--json-out "{REPO_ROOT}/reports/rt114_{slug}_storage_latency.json"', cwd=REPO_ROOT)

    # 6) summary + verdict
    print("\n" + "=" * 60)
    print(f"RT-114 / SCALE-001  {args.model_id}  ({n_lin} target linears)")
    print("=" * 60)
    print(f"ternary Path A' PPL (eval.txt, ctx={args.ctx}):")
    for fmt in ["f32", "f16", "i2_s"]:
        print(f"  {fmt:<5} {ppl.get(fmt)}")
    if ppl_lat:
        print("latent Path A control PPL:")
        for fmt in ["f32", "f16", "i2_s"]:
            print(f"  {fmt:<5} {ppl_lat.get(fmt)}")
    print("\nSCALE-001d (parity) VERDICT:")
    f16, i2s = ppl.get("f16"), ppl.get("i2_s")
    if f16 and i2s and abs(i2s - f16) <= 0.03 * f16:
        print(f"  PASS: i2_s {i2s} ~= f16 {f16} -> runtime faithful at scale; absolute PPL is")
        print(f"        PTQ-poor by design (needs ternary training, separate track).")
    else:
        print(f"  CHECK: i2_s {i2s} vs f16 {f16} -> parity gap; inspect which tensor diverges.")
    print("storage/latency (SCALE-001a/b): see the RT-113 table above.")


if __name__ == "__main__":
    main()
