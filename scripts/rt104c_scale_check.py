#!/usr/bin/env python3
"""RT-104C (decisive): weight-level parity between bitnet.cpp's I2_S tensor and
our Python per-tensor b1.58 reference. Bypasses tokenization / PPL protocol /
activation-int8 confounds by comparing the dequantized weight directly.

For a target linear it:
  1. parses the I2_S GGUF, extracts the tensor's per-tensor fp32 scale + 2-bit
     codes (per the RT-101 layout: 128-elem blocks, MSB fields, 0b00=0/01=+1/10=-1,
     trailing 32-byte scale x8),
  2. reconstructs gamma_bitnet * T_bitnet,
  3. compares to our reference gamma_ours = mean(|W_latent|), T_ours =
     clamp(round(W/gamma),-1,1) from the HF latent safetensors.

If scale and reconstruction match -> upstream I2_S uses our scale semantics ->
Path A preserves quality. If not, we learn the exact layout/scale difference and
go Path B.

    .venv/bin/python scripts/rt104c_scale_check.py \
      --i2s /Users/puka/repository/bitnet.cpp/models/tiny_pt_trained/ggml-model-i2_s.gguf \
      --hf  /Users/puka/repository/bitnet.cpp/models/tiny_pt_trained \
      --tensors blk.0.attn_q.weight blk.0.ffn_gate.weight
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch

GGUF_NAME_MAP = {  # gguf -> hf suffix
    "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
}
_SCALAR = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
           5: ("i", 4), 6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


def _rd(b, o, fmt, sz):
    return struct.unpack_from("<" + fmt, b, o)[0], o + sz


def _rd_str(b, o):
    n, o = _rd(b, o, "Q", 8)
    return b[o:o + n].decode("utf-8"), o + n


def _skip_val(b, o, vtype):
    if vtype in _SCALAR:
        return o + _SCALAR[vtype][1]
    if vtype == 8:  # string
        n, o = _rd(b, o, "Q", 8); return o + n
    if vtype == 9:  # array
        et, o = _rd(b, o, "I", 4); cnt, o = _rd(b, o, "Q", 8)
        if et == 8:
            for _ in range(cnt):
                n, o = _rd(b, o, "Q", 8); o += n
            return o
        return o + cnt * _SCALAR[et][1]
    raise ValueError(f"bad vtype {vtype}")


def parse_gguf(path):
    b = Path(path).read_bytes()
    assert b[:4] == b"GGUF", "not a GGUF"
    o = 4
    ver, o = _rd(b, o, "I", 4)
    n_tensors, o = _rd(b, o, "Q", 8)
    n_kv, o = _rd(b, o, "Q", 8)
    alignment = 32
    for _ in range(n_kv):
        key, o = _rd_str(b, o)
        vtype, o = _rd(b, o, "I", 4)
        if key == "general.alignment":
            val, _o2 = _rd(b, o, _SCALAR[vtype][0], _SCALAR[vtype][1]); alignment = val
        o = _skip_val(b, o, vtype)
    tensors = {}
    for _ in range(n_tensors):
        name, o = _rd_str(b, o)
        nd, o = _rd(b, o, "I", 4)
        dims = []
        for _ in range(nd):
            d, o = _rd(b, o, "Q", 8); dims.append(d)
        ttype, o = _rd(b, o, "I", 4)
        toff, o = _rd(b, o, "Q", 8)
        tensors[name] = {"dims": dims, "type": ttype, "offset": toff}
    data_start = (o + alignment - 1) // alignment * alignment
    return b, data_start, tensors


def unpack_i2s(code_bytes: np.ndarray, numel: int) -> np.ndarray:
    """RT-101 layout: 32-byte blocks of 128 elems; byte gp holds elems
    [gp,32+gp,64+gp,96+gp] at bits [7:6],[5:4],[3:2],[1:0]; 00=0,01=+1,10=-1."""
    n_blocks = len(code_bytes) // 32
    out = np.zeros(n_blocks * 128, dtype=np.int8)
    cb = code_bytes[: n_blocks * 32].reshape(n_blocks, 32).astype(np.uint8)
    code2val = np.array([-1, 0, 1, 0], dtype=np.int8)  # ggml-quants.c: q8 0->-1,1->0,2->+1
    for group in range(4):  # offset 0,32,64,96
        shift = 6 - 2 * group
        codes = (cb >> shift) & 0x3          # [n_blocks, 32]
        out.reshape(n_blocks, 128)[:, group * 32:(group + 1) * 32] = code2val[codes]
    return out[:numel]


def gguf_to_hf(name):
    # blk.N.attn_q.weight -> model.layers.N.self_attn.q_proj.weight
    parts = name.split(".")
    n = parts[1]; key = parts[2]
    return f"model.layers.{n}.{GGUF_NAME_MAP[key]}.weight"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--i2s", type=Path, required=True)
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--tensors", nargs="+", default=["blk.0.attn_q.weight"])
    args = ap.parse_args()

    from safetensors.torch import load_file
    sd = load_file(str(args.hf / "model.safetensors"))
    b, data_start, tensors = parse_gguf(args.i2s)
    names = sorted(tensors)
    print(f"gguf: {len(tensors)} tensors, data_start={data_start}, alignment ok")
    print(f"I2_S type id seen: {sorted(set(t['type'] for t in tensors.values()))}")

    allok = True
    for tname in args.tensors:
        info = tensors[tname]
        dims = info["dims"]  # gguf dims: [in(fast), out(slow)] print order
        numel = int(np.prod(dims))
        # tensor byte span = next tensor offset - this, or file_end - data_start - off
        offs = sorted(t["offset"] for t in tensors.values())
        nxt = min([x for x in offs if x > info["offset"]], default=None)
        end = (data_start + nxt) if nxt is not None else len(b)
        raw = np.frombuffer(b[data_start + info["offset"]: end], dtype=np.uint8)
        code_len = (numel + 3) // 4
        scale_bytes = raw[code_len:code_len + 32]
        gamma_bitnet = struct.unpack_from("<f", scale_bytes.tobytes(), 0)[0]
        T_bitnet = unpack_i2s(raw[:code_len], numel)

        W = sd[gguf_to_hf(tname)].float().numpy()           # [out, in]
        absmax = float(np.abs(W).max())
        absmean = float(np.abs(W).mean())
        # bitnet reference (per ggml-quants.c): scale=absmax, T=sign (deadzone 1e-6)
        T_sign = np.where(np.abs(W) < 1e-6, 0, np.sign(W)).astype(np.int8).reshape(-1)
        # our per-tensor b1.58 reference: scale=mean|W|, T=round(W/scale)
        T_round = np.clip(np.round(W / absmean), -1, 1).astype(np.int8).reshape(-1)

        Tb = T_bitnet
        m_sign = float(np.mean(Tb == T_sign))
        m_round = float(np.mean(Tb == T_round))
        bytes_law = (len(raw) == code_len + 32)
        print(f"\n[{tname}] dims={dims} numel={numel} bytes_law(ceil/4+32)={bytes_law}")
        print(f"  scale: bitnet={gamma_bitnet:.6g}  absmax|W|={absmax:.6g} (ratio {gamma_bitnet/absmax:.4f})  "
              f"mean|W|={absmean:.6g} (ratio {gamma_bitnet/absmean:.3f})")
        print(f"  ternary code match vs sign(absmax)={m_sign*100:.2f}%   vs round(absmean,ours)={m_round*100:.2f}%")
        ok = bytes_law and m_sign > 0.99 and abs(gamma_bitnet / max(absmax, 1e-12) - 1) < 0.02
        allok = allok and ok
        print(f"  -> bitnet I2_S == sign(W)*absmax ? {'YES' if ok else 'NO'}")
    print("\nRT-104C verdict:", "bitnet I2_S = absmax + sign (NOT our absmean+round)" if allok
          else "inconclusive / different layout - inspect")


if __name__ == "__main__":
    main()
