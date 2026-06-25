#!/usr/bin/env python3
"""RT-113 / EXPORT-006/007: storage + latency for f32/f16/i2_s on x86 bitnet.cpp.

RT-112 proved correctness (our b1.58 model has F16/F32 parity through x86 I2_S).
RT-113 answers the original efficiency question: does the artifact get smaller and
does it run faster, measured under identical conditions?

Storage: analytic target-linear-only bytes (the part I2_S compresses) AND the whole
on-disk artifact, with ratios. The whole-file ratio is diluted by the f16 embedding
floor on tiny models; target-linear-only is the true I2_S compression and is what a
real model (where linears dominate) converges to.

Latency: llama-bench (prompt-processing pp + token-generation tg tokens/sec) with the
same thread count for every format. tg is memory-bandwidth-bound, so it is where the
2-bit weight traffic reduction shows up.

PREREQUISITES: bitnet.cpp built on x86 (see rt112_x86_arena.py docstring); a model
dir with ggml-model-{f32,f16,i2_s}.gguf + config.json (rt112 produces tiny_pt_ternary).

USAGE:
  python scripts/rt113_storage_latency.py --bitnet /content/bitnet.cpp \
    --model-dir /content/bitnet.cpp/models/tiny_pt_ternary --threads 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

FMTS = ["f32", "f16", "i2_s"]
TS_RE = re.compile(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([0-9.]+)\s*±\s*([0-9.]+)\s*\|")


def target_linear_elems(cfg):
    """Sum of elements in the I2_S-quantized linears (attn q/k/v/o + ffn gate/up/down)."""
    H = cfg["hidden_size"]; I = cfg["intermediate_size"]; L = cfg["num_hidden_layers"]
    n_heads = cfg["num_attention_heads"]; n_kv = cfg.get("num_key_value_heads", n_heads)
    head = H // n_heads
    q = H * H; k = H * (n_kv * head); v = H * (n_kv * head); o = H * H
    gate = H * I; up = H * I; down = I * H
    per_layer = q + k + v + o + gate + up + down
    n_tensors = L * 7
    return L * per_layer, n_tensors


def fmt_bytes(elems, ntensors, fmt):
    if fmt == "f32":
        return elems * 4
    if fmt == "f16":
        return elems * 2
    return elems // 4 + 32 * ntensors        # I2_S: 2-bit codes + 32B trailing scale/tensor


def llama_bench(bitnet, gguf, threads, pp, tg, reps):
    cmd = (f'{bitnet}/build/bin/llama-bench -m "{gguf}" -t {threads} '
           f'-p {pp} -n {tg} -r {reps} 2>/dev/null')
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    res = {}
    for test, ts, sd in TS_RE.findall(out):
        res[test] = (float(ts), float(sd))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitnet", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--pp", type=int, default=64)
    ap.add_argument("--tg", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json-out", type=Path, default=Path("reports/rt113_storage_latency.json"))
    args = ap.parse_args()

    md = args.model_dir.resolve()
    cfg = json.loads((md / "config.json").read_text())
    tl_elems, tl_n = target_linear_elems(cfg)

    # ---- STORAGE ----
    print("=" * 60)
    print("RT-113 STORAGE  (%s)" % md.name)
    print("=" * 60)
    whole = {f: os.path.getsize(md / f"ggml-model-{f}.gguf") for f in FMTS}
    tl = {f: fmt_bytes(tl_elems, tl_n, f) for f in FMTS}
    print(f"target-linear elems={tl_elems:,} ({tl_n} tensors)  embedding masks the whole-file ratio")
    print(f"{'fmt':<6}{'target_lin B':>14}{'whole_file B':>14}")
    for f in FMTS:
        print(f"{f:<6}{tl[f]:>14,}{whole[f]:>14,}")
    storage = {"target_linear_bytes": tl, "whole_bytes": whole,
               "ratio_target_vs_f32": {f: tl[f] / tl["f32"] for f in FMTS},
               "ratio_whole_vs_f32": {f: whole[f] / whole["f32"] for f in FMTS}}
    print(f"\nratio vs f32   target-linear | whole")
    for f in FMTS:
        print(f"  {f:<5}: {storage['ratio_target_vs_f32'][f]:.4f}      | {storage['ratio_whole_vs_f32'][f]:.4f}")

    # ---- LATENCY ----
    print("\n" + "=" * 60)
    print("RT-113 LATENCY  (llama-bench, t=%d, pp%d/tg%d, r%d)" % (args.threads, args.pp, args.tg, args.reps))
    print("=" * 60)
    lat = {}
    for f in FMTS:
        lat[f] = llama_bench(args.bitnet, md / f"ggml-model-{f}.gguf", args.threads, args.pp, args.tg, args.reps)
        print(f"{f:<6} {lat[f]}")
    ppk, tgk = f"pp{args.pp}", f"tg{args.tg}"
    if all(ppk in lat[f] and tgk in lat[f] for f in FMTS):
        print("\nspeedup of i2_s   pp | tg")
        print(f"  vs f32: {lat['i2_s'][ppk][0]/lat['f32'][ppk][0]:.2f}x | {lat['i2_s'][tgk][0]/lat['f32'][tgk][0]:.2f}x")
        print(f"  vs f16: {lat['i2_s'][ppk][0]/lat['f16'][ppk][0]:.2f}x | {lat['i2_s'][tgk][0]/lat['f16'][tgk][0]:.2f}x")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"storage": storage, "latency": lat,
                                          "threads": args.threads, "pp": args.pp, "tg": args.tg,
                                          "reps": args.reps}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")
    print("NOTE: peak RSS is NOT a useful discriminator here — llama.cpp mmaps the model,"
          " so touched-page RSS is ~equal across formats; storage bytes + tg t/s carry the"
          " memory story.")


if __name__ == "__main__":
    main()
