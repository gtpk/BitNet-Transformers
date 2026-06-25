#!/usr/bin/env python3
"""RT-121 / BASE-001 cheap baseline panel for G5.

Question: why not just use existing one-shot quantization?

This script deliberately runs only the cheap/no-new-training panel:

  B0    FP reference, f16 GGUF
  B1    one-shot b1.58 PTQ, Wq=gamma*T -> I2_S
  B2    llama.cpp Q2_K one-shot from FP f32
  B3    llama.cpp Q3_K_M one-shot from FP f32
  B4    llama.cpp Q4_0 one-shot from FP f32
  OURS  previously adapted b1.58 model -> I2_S

All rows are evaluated by the SAME llama-perplexity binary on the SAME eval.txt.
The script does not train. Produce the adapted HF dir first with
scripts/rt116_quality_recovery.py, then pass it via --adapted-dir.

USAGE (Colab/Linux x86 with bitnet.cpp already built):

  python scripts/rt121_baseline_panel.py \
    --bitnet /content/bitnet.cpp \
    --model-id JackFram/llama-160m \
    --adapted-dir /content/bnt/reports/llama-160m_adapted \
    --json-out reports/rt121_baseline_panel.json \
    --markdown-out reports/rt121_baseline_panel.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402

PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")


def run(cmd: str, cwd: Path | None = None) -> str:
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(out[-800:])
    if r.returncode != 0:
        raise SystemExit(f"command failed rc={r.returncode}: {cmd}")
    return out


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {label}: {path}")


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace("/", "_")


def download_hf(model_id: str, out_dir: Path) -> None:
    if (out_dir / "config.json").exists():
        return
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> downloading {model_id} -> {out_dir}")
    snapshot_download(model_id, local_dir=str(out_dir))


def materialize_ptq(fp_dir: Path, ptq_dir: Path) -> int:
    """Save a dense HF model where target linears are Wq=gamma*T (no adaptation)."""
    if (ptq_dir / "config.json").exists():
        return -1
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(fp_dir, dtype=torch.float32).eval()
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
                mod.weight.copy_(C.per_tensor_b158_approx(mod.weight))
                n += 1
    ptq_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ptq_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(fp_dir).save_pretrained(ptq_dir)
    print(f"materialized B1 PTQ Wq=gamma*T on {n} target linears -> {ptq_dir}")
    return n


def build_gguf(convert: Path, hf_dir: Path, outtype: str) -> Path:
    gguf = hf_dir / f"ggml-model-{outtype}.gguf"
    if not gguf.exists():
        run(f'"{sys.executable}" "{convert}" "{hf_dir}" --outtype {outtype}')
    return gguf


def quantize(quant: Path, src_f32: Path, dst: Path, qtype: str) -> Path:
    if dst.exists():
        return dst
    if qtype == "I2_S":
        cmd = (f'"{quant}" --token-embedding-type f16 --output-tensor-type f16 '
               f'"{src_f32}" "{dst}" I2_S 1 1')
    else:
        cmd = (f'"{quant}" --token-embedding-type f16 --output-tensor-type f16 '
               f'"{src_f32}" "{dst}" {qtype}')
    run(cmd)
    return dst


def write_eval_text(args, fp_dir: Path, adapted_dir: Path, eval_txt: Path) -> None:
    if args.eval_text:
        shutil.copyfile(args.eval_text, eval_txt)
        return
    adapted_eval = adapted_dir / "eval.txt"
    if adapted_eval.exists():
        shutil.copyfile(adapted_eval, eval_txt)
        return

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(fp_dir)
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors=None)["input_ids"][: args.ppl_eval_tokens]
    eval_txt.write_text(tok.decode(ids), encoding="utf-8")


def perplexity(ppx: Path, gguf: Path, eval_txt: Path, ctx: int, threads: int) -> float:
    cmd = f'"{ppx}" -m "{gguf}" -f "{eval_txt}" -c {ctx} -t {threads}'
    out = run(cmd)
    m = PPL_RE.search(out)
    if not m:
        raise SystemExit(f"could not parse Final estimate PPL for {gguf}")
    return float(m.group(1))


def mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bitnet", type=Path, required=True)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--adapted-dir", type=Path, required=True,
                    help="HF dir already materialized by rt116 (adapted Wq=gamma*T). No training is done here.")
    ap.add_argument("--work", type=Path, default=None, help="default: <bitnet>/models/rt121_baselines")
    ap.add_argument("--eval-text", type=Path, default=None,
                    help="optional eval.txt to use for every row; else adapted-dir/eval.txt or WikiText validation")
    ap.add_argument("--ppl-eval-tokens", type=int, default=3000)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/rt121_baseline_panel.json")
    ap.add_argument("--markdown-out", type=Path, default=REPO_ROOT / "reports/rt121_baseline_panel.md")
    args = ap.parse_args()

    bitnet = args.bitnet.resolve()
    adapted_dir = args.adapted_dir.resolve()
    slug = model_slug(args.model_id)
    work = (args.work or bitnet / "models/rt121_baselines").resolve()
    fp_dir = work / f"{slug}_fp"
    ptq_dir = work / f"{slug}_ptq_b158"
    eval_txt = work / "eval.txt"

    convert = bitnet / "utils/convert-hf-to-gguf-bitnet.py"
    quant = bitnet / "build/bin/llama-quantize"
    ppx = bitnet / "build/bin/llama-perplexity"
    for path, label in [(convert, "convert-hf-to-gguf-bitnet.py"),
                        (quant, "llama-quantize"),
                        (ppx, "llama-perplexity"),
                        (adapted_dir / "config.json", "adapted HF config")]:
        ensure_file(path, label)

    work.mkdir(parents=True, exist_ok=True)
    download_hf(args.model_id, fp_dir)
    n_ptq = materialize_ptq(fp_dir, ptq_dir)
    write_eval_text(args, fp_dir, adapted_dir, eval_txt)
    print(f"eval text: {eval_txt} ({eval_txt.stat().st_size:,} bytes)")

    fp_f32 = build_gguf(convert, fp_dir, "f32")
    fp_f16 = build_gguf(convert, fp_dir, "f16")
    ptq_f32 = build_gguf(convert, ptq_dir, "f32")
    ptq_i2s = quantize(quant, ptq_f32, ptq_dir / "ggml-model-i2_s.gguf", "I2_S")
    adapted_f32 = build_gguf(convert, adapted_dir, "f32")
    adapted_i2s = quantize(quant, adapted_f32, adapted_dir / "ggml-model-i2_s.gguf", "I2_S")

    k_quants = {
        "B2": ("llama.cpp Q2_K one-shot", "~2.6", "Q2_K"),
        "B3": ("llama.cpp Q3_K_M one-shot", "~3.4", "Q3_K_M"),
        "B4": ("llama.cpp Q4_0 one-shot", "~4.5", "Q4_0"),
    }
    k_paths = {
        key: quantize(quant, fp_f32, fp_dir / f"ggml-model-{qtype.lower()}.gguf", qtype)
        for key, (_, _, qtype) in k_quants.items()
    }

    rows = [
        {"id": "B0", "method": "FP reference (f16)", "target_bits": "16",
         "trains": "no", "gguf": fp_f16},
        {"id": "B1", "method": "RTN ternary one-shot Wq=gamma*T (I2_S)", "target_bits": "1.58 logical / 2.0 stored",
         "trains": "no", "gguf": ptq_i2s},
        {"id": "B2", "method": k_quants["B2"][0], "target_bits": k_quants["B2"][1],
         "trains": "no", "gguf": k_paths["B2"]},
        {"id": "B3", "method": k_quants["B3"][0], "target_bits": k_quants["B3"][1],
         "trains": "no", "gguf": k_paths["B3"]},
        {"id": "B4", "method": k_quants["B4"][0], "target_bits": k_quants["B4"][1],
         "trains": "no", "gguf": k_paths["B4"]},
        {"id": "OURS", "method": "b1.58 + teacher-free CE (linears-only, I2_S)",
         "target_bits": "1.58 logical / 2.0 stored", "trains": "yes (cheap)", "gguf": adapted_i2s},
    ]

    print("\n" + "=" * 72)
    print("RT-121 / BASE-001 baseline panel")
    print("=" * 72)
    for row in rows:
        row["ppl"] = perplexity(ppx, row["gguf"], eval_txt, args.ctx, args.threads)
        row["ce_nats"] = math.log(row["ppl"])
        row["whole_mb"] = mb(row["gguf"])
        row["gguf"] = str(row["gguf"])

    by_id = {row["id"]: row for row in rows}
    q2 = by_id["B2"]["ppl"]
    ours = by_id["OURS"]["ppl"]
    headline = {
        "ours_ppl": ours,
        "q2k_ppl": q2,
        "ours_vs_q2k_ppl_ratio": ours / q2,
        "ours_beats_q2k": ours <= q2,
        "verdict": "PASS_HEADLINE" if ours <= q2 else "NEGATIVE_OR_REFRAME",
    }

    result = {
        "model": args.model_id,
        "ctx": args.ctx,
        "threads": args.threads,
        "eval_txt": str(eval_txt),
        "n_ptq_target_linears": n_ptq,
        "rows": rows,
        "headline": headline,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# RT-121 / BASE-001 baseline panel — {args.model_id}",
        "",
        f"Same `llama-perplexity`, same eval text, ctx `{args.ctx}`, threads `{args.threads}`.",
        "",
        "| id | method | target bits | trains? | PPL | CE nats | whole MB |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append("| {id} | {method} | {target_bits} | {trains} | {ppl:.4f} | {ce_nats:.4f} | {whole_mb:.1f} |".format(**row))
    lines += [
        "",
        "## Headline",
        "",
        f"- OURS / Q2_K PPL ratio: `{headline['ours_vs_q2k_ppl_ratio']:.4f}`",
        f"- Verdict: `{headline['verdict']}`",
        "",
    ]
    if headline["ours_beats_q2k"]:
        lines.append("OURS beats or matches one-shot Q2_K despite lower logical target-bit budget.")
    else:
        lines.append("OURS does not beat one-shot Q2_K at this budget; reframe or improve before headline use.")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
