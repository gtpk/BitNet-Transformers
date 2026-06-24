#!/usr/bin/env python3
"""RT-112: our tiny per-tensor-native model on x86 I2_S — latent vs ternary-dense.

RT-111 proved bitnet.cpp I2_S is fine on x86 (official model f32 1.8547 ~ i2_s
1.8548) and the M5 collapse is toolchain-only. RT-112 closes the loop: does OUR
trained tiny per-tensor b1.58 model reach F16/F32/Python parity through the x86
I2_S runtime?

Two I2_S encodings are compared (same eval text, same llama-perplexity tool):

  latent   (Path A)  : convert the trained latent-FP weights -> upstream I2_S
                       re-quantizes with sign*absmax (RT-104C) -> expected to
                       collapse. CONTROL.
  ternary  (Path A') : materialize Wq = gamma*T (gamma=mean|W|) into a dense HF
                       model, THEN upstream I2_S. Since max|Wq|=gamma, sign(Wq)=T,
                       Q_absmax(Wq)=Wq -> lossless repack -> expected F16 parity.
                       If this matches, no Path B byte-writer is needed.

For each path we also build F32 and F16 GGUFs as faithful-runtime anchors and
compare to the Python references from rt104_reference.json.

PREREQUISITES (do these on the x86 Colab box BEFORE this script):
  1. bitnet.cpp cloned @ pinned commit + submodules, BOTH `int8_t * y_col`
     occurrences in src/ggml-bitnet-mad.cpp const-patched, then built via
     `python setup_env.py -hr 1bitLLM/bitnet_b1_58-large -q i2_s` (gives the
     binaries + the tokenizer dir + the official i2_s control).
  2. This repo cloned; run from its root with torch+transformers available.

USAGE (on Colab x86):
  python scripts/rt112_x86_arena.py --bitnet /content/bitnet.cpp
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")


def run(cmd, cwd=None, env=None):
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out[-600:])
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode}): {cmd}")
    return out


def perplexity(bitnet: Path, gguf: Path, eval_txt: Path, ctx: int, threads: int):
    """Return (ppl, stderr) for a GGUF on eval_txt, or (None, ...) if no estimate."""
    binp = bitnet / "build/bin/llama-perplexity"
    cmd = f'{binp} -m "{gguf}" -f "{eval_txt}" -c {ctx} -t {threads}'
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    m = PPL_RE.search(out)
    if not m:
        print("  !! no Final estimate. tail:\n" + out[-500:])
        return None, out
    ppl = float(m.group(1))
    print(f"  PPL = {ppl}  (+/- {m.group(2)})")
    return ppl, out


def build_ggufs(bitnet: Path, hf_dir: Path, convert: Path):
    """Make f32, f16, and i2_s GGUFs inside hf_dir. Returns dict path-by-tag."""
    f32 = hf_dir / "ggml-model-f32.gguf"
    f16 = hf_dir / "ggml-model-f16.gguf"
    i2s = hf_dir / "ggml-model-i2_s.gguf"
    quant = bitnet / "build/bin/llama-quantize"
    if not f32.exists():
        run(f'python "{convert}" "{hf_dir}" --outtype f32')
    if not f16.exists():
        run(f'python "{convert}" "{hf_dir}" --outtype f16')
    if not i2s.exists():
        run(f'"{quant}" --token-embedding-type f16 "{f32}" "{i2s}" I2_S 1 1')
    return {"f32": f32, "f16": f16, "i2_s": i2s}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitnet", type=Path, required=True, help="built bitnet.cpp root")
    ap.add_argument("--tokenizer-src", type=Path, default=None,
                    help="dir with tokenizer.json+config.json (default: <bitnet>/models/bitnet_b1_58-large)")
    ap.add_argument("--work", type=Path, default=None, help="output dir (default: <bitnet>/models)")
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "data/tiny_corpus.txt")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--corpus-repeat", type=int, default=20,
                    help="tile the corpus N times so the eval split is long enough "
                         "for ctx perplexity + rt104d seq_len (parity-safe)")
    ap.add_argument("--train-steps", type=int, default=300)
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt112_x86_arena.json")
    args = ap.parse_args()

    bitnet = args.bitnet.resolve()
    tok_src = (args.tokenizer_src or bitnet / "models/bitnet_b1_58-large").resolve()
    work = (args.work or bitnet / "models").resolve()
    convert = bitnet / "utils/convert-hf-to-gguf-bitnet.py"
    py = sys.executable

    for p, what in [(bitnet / "build/bin/llama-perplexity", "llama-perplexity"),
                    (bitnet / "build/bin/llama-quantize", "llama-quantize"),
                    (convert, "convert-hf-to-gguf-bitnet.py"),
                    (tok_src / "tokenizer.json", "tokenizer.json")]:
        if not p.exists():
            raise SystemExit(f"missing prerequisite {what}: {p}\nSee the PREREQUISITES docstring.")

    latent_dir = work / "tiny_pt_trained"     # Path A (latent FP)
    ternary_dir = work / "tiny_pt_ternary"    # Path A' (gamma*T dense)
    ref_json = REPO_ROOT / "reports/rt104_reference.json"

    # The bundled tiny_corpus is a few hundred tokens; its 15% eval split is then
    # too short for ctx=64 perplexity AND for rt104d's seq_len=128 (-> a reshape
    # [0,128,..] crash). Parity is a same-text comparison, so repeating the corpus
    # is harmless (like RT-111's repeated eval). Repeat until >= a token floor.
    corpus = args.corpus
    if args.corpus_repeat > 1:
        big = work / "rt112_corpus.txt"
        big.write_text(args.corpus.read_text(encoding="utf-8") * args.corpus_repeat,
                       encoding="utf-8")
        corpus = big
        print(f"corpus repeated x{args.corpus_repeat} -> {corpus} ({big.stat().st_size} bytes)")

    # 1) train tiny per-tensor model -> latent HF dir + eval.txt + Python refs
    run(f'{py} scripts/rt104_build_reference.py '
        f'--tokenizer-src "{tok_src}" --corpus "{corpus}" '
        f'--out-dir "{latent_dir}" --json-out "{ref_json}" --train-steps {args.train_steps}',
        cwd=REPO_ROOT)
    # 2) materialize gamma*T -> ternary-dense HF dir (Path A')
    run(f'{py} scripts/rt104d_quantized_dense.py '
        f'--in-dir "{latent_dir}" --out-dir "{ternary_dir}"', cwd=REPO_ROOT)

    eval_txt = latent_dir / "eval.txt"   # identical token stream for every GGUF
    refs = json.loads(ref_json.read_text())["ppl"] if ref_json.exists() else {}

    # 3) build GGUFs + perplexity for both paths
    table = {}
    for tag, hf_dir in [("latent(PathA)", latent_dir), ("ternary(PathA')", ternary_dir)]:
        ggufs = build_ggufs(bitnet, hf_dir, convert)
        row = {}
        for fmt, g in ggufs.items():
            ppl, _ = perplexity(bitnet, g, eval_txt, args.ctx, args.threads)
            row[fmt] = ppl
        table[tag] = row

    # 4) report
    print("\n" + "=" * 64)
    print("RT-112  x86 I2_S arena  (eval.txt, ctx=%d)" % args.ctx)
    print("=" * 64)
    print("Python refs:  per_tensor_ste=%s  latent_fp=%s  i2s_export(gamma*T)=%s"
          % (refs.get("per_tensor_ste"), refs.get("latent_fp"), refs.get("i2s_export_gamma_T")))
    print(f"{'path':<18}{'f32':>12}{'f16':>12}{'i2_s':>12}")
    for tag, row in table.items():
        print(f"{tag:<18}{str(row.get('f32')):>12}{str(row.get('f16')):>12}{str(row.get('i2_s')):>12}")

    # 5) verdict
    print("\nVERDICT:")
    tern = table.get("ternary(PathA')", {})
    lat = table.get("latent(PathA)", {})

    def near(a, b, tol=0.05):
        return a is not None and b is not None and abs(a - b) <= tol * max(abs(b), 1e-9)

    if near(tern.get("i2_s"), tern.get("f16")):
        print("  PASS: Path A' i2_s ~= f16 -> our b1.58 model runs faithfully on x86 I2_S.")
        print("        No Path B byte-writer needed. Next: storage/latency (EXPORT-006/007).")
    elif lat.get("i2_s") and tern.get("i2_s") and tern["i2_s"] < lat["i2_s"] * 0.5:
        print("  PARTIAL: Path A' beats latent but != f16 -> encoding mostly right, residual to chase.")
    else:
        print("  FAIL: Path A' i2_s != f16 while official passed (RT-111) -> our artifact/encoding")
        print("        issue -> go to Path B direct I2_S byte-writer.")
    if lat.get("i2_s") and lat.get("f16") and lat["i2_s"] > lat["f16"] * 2:
        print("  (expected) latent Path A i2_s collapses vs its f16 -> absmax-vs-absmean confirmed.")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"ctx": args.ctx, "python_refs": refs, "table": table}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
