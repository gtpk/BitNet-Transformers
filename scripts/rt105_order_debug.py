#!/usr/bin/env python3
"""RT-105 order debug: find the I2_S element order by isolating where decode breaks.

Uses the ternary-dense model (models/tiny_pt_ternary) whose weights are exactly
Wq = gamma*T, so the true ternary T is known. Cross-checks three sources:
  safetensors Wq  ->  F32 GGUF tensor  ->  I2_S codes
to locate the bug (f32 reader vs i2_s decode vs element order), trying several
order/mapping hypotheses and reporting zero/sign/overall match + first mismatch.

    .venv/bin/python scripts/rt105_order_debug.py --dir /Users/puka/repository/bitnet.cpp/models/tiny_pt_ternary --tensor blk.0.ffn_gate.weight
"""
from __future__ import annotations
import argparse, struct, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.rt104c_scale_check import parse_gguf, gguf_to_hf  # reuse parser


def i2s_raw(b, ds, T, name):
    info = T[name]; numel = int(np.prod(info["dims"]))
    offs = sorted(t["offset"] for t in T.values())
    nxt = min([x for x in offs if x > info["offset"]], default=None)
    end = (ds + nxt) if nxt is not None else len(b)
    raw = np.frombuffer(b[ds + info["offset"]: end], dtype=np.uint8)
    code_len = (numel + 3) // 4
    scale = struct.unpack_from("<f", b, ds + info["offset"] + code_len)[0]
    return raw[:code_len].copy(), scale, numel, info["dims"]


def f32_tensor(b, ds, T, name):
    info = T[name]; numel = int(np.prod(info["dims"]))
    off = ds + info["offset"]
    return np.frombuffer(b[off:off + numel * 4], dtype=np.float32).copy()


# ---- decode hypotheses: codes(uint8[n/4]) -> ternary int8[n] ----
def dec_block_interleave(codes, numel, field_shifts):
    nb = len(codes) // 32
    cb = codes[:nb * 32].reshape(nb, 32).astype(np.uint8)
    m = np.array([-1, 0, 1, 0], dtype=np.int8)
    out = np.zeros(nb * 128, dtype=np.int8).reshape(nb, 128)
    for g, sh in enumerate(field_shifts):
        out[:, g * 32:(g + 1) * 32] = m[(cb >> sh) & 3]
    return out.reshape(-1)[:numel]

def dec_sequential(codes, numel, shifts):
    c = codes.astype(np.uint8)
    m = np.array([-1, 0, 1, 0], dtype=np.int8)
    out = np.zeros(len(c) * 4, dtype=np.int8).reshape(-1, 4)
    for k, sh in enumerate(shifts):
        out[:, k] = m[(c >> sh) & 3]
    return out.reshape(-1)[:numel]


def metrics(dec, Ttrue):
    zero_mask = (Ttrue == 0)
    zp = float(np.mean(dec[zero_mask] == 0)) if zero_mask.any() else 1.0
    nz = ~zero_mask
    sm = float(np.mean(dec[nz] == Ttrue[nz])) if nz.any() else 1.0
    overall = float(np.mean(dec == Ttrue))
    fm = int(np.argmax(dec != Ttrue)) if overall < 1 else -1
    return zp, sm, overall, fm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--tensor", default="blk.0.ffn_gate.weight")
    args = ap.parse_args()
    from safetensors.torch import load_file

    sd = load_file(str(args.dir / "model.safetensors"))
    bi, dsi, Ti = parse_gguf(args.dir / "ggml-model-i2_s.gguf")
    bf, dsf, Tf = parse_gguf(args.dir / "ggml-model-f32.gguf")

    W_st = sd[gguf_to_hf(args.tensor)].float().numpy()   # safetensors [out,in], = gamma*T
    gamma = float(np.abs(W_st).max())
    T_st = np.sign(W_st).astype(np.int8).reshape(-1)      # row-major [out,in]
    Wf = f32_tensor(bf, dsf, Tf, args.tensor)             # f32 gguf, flat
    codes, scale, numel, dims = i2s_raw(bi, dsi, Ti, args.tensor)

    print(f"tensor {args.tensor} dims(gguf)={dims} numel={numel}  gamma(st)={gamma:.6g} scale(i2s)={scale:.6g}")
    # (1) isolate f32 reader: f32-gguf vs safetensors
    print("\n[f32 GGUF vs safetensors]")
    print(f"  sign match row-major(C): {np.mean(np.sign(Wf)==T_st)*100:.2f}%")
    print(f"  sign match col-major(F): {np.mean(np.sign(Wf)==np.sign(W_st).reshape(-1,order='F'))*100:.2f}%")
    print(f"  max|Δ| (C order): {np.abs(Wf - W_st.reshape(-1)).max():.3e}")

    Tf_sign = np.sign(Wf).astype(np.int8)  # ground-truth ternary in F32-GGUF order
    # (2) decode hypotheses vs F32-GGUF-order ternary (same lineage, removes safetensors transforms)
    cands = {
        "block_interleave [6,4,2,0]": dec_block_interleave(codes, numel, [6,4,2,0]),
        "block_interleave [0,2,4,6]": dec_block_interleave(codes, numel, [0,2,4,6]),
        "block_field_rev   [6,4,2,0]->cols rev": dec_block_interleave(codes, numel, [6,4,2,0])[::1],  # placeholder
        "sequential [6,4,2,0]": dec_sequential(codes, numel, [6,4,2,0]),
        "sequential [0,2,4,6]": dec_sequential(codes, numel, [0,2,4,6]),
    }
    print("\n[i2s decode hypotheses vs sign(F32-GGUF)]   zero_preserve / nonzero_sign / overall / first_mismatch")
    for name, dec in cands.items():
        zp, sm, ov, fm = metrics(dec, Tf_sign)
        print(f"  {name:36} {zp*100:6.2f}% {sm*100:6.2f}% {ov*100:6.2f}%  fm={fm}")

    # (3) first 32 side-by-side for the source hypothesis
    dec = dec_block_interleave(codes, numel, [6,4,2,0])
    print("\nfirst 24 elements:  F32sign:", list(Tf_sign[:24]))
    print("                    decoded :", list(dec[:24]))
    print("raw code bytes[:6]:", list(codes[:6]), "-> 2-bit fields:",
          [[(int(c)>>s)&3 for s in (6,4,2,0)] for c in codes[:6]])


if __name__ == "__main__":
    main()
