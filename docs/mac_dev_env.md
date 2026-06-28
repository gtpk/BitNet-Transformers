# Mac Dev Env (Apple Silicon / MPS)

Document position: [Index](./index.md). The local Mac (Apple M5, 24 GB unified memory) for **160M
DINO smokes + scoring/analysis**, running on the MPS backend. Pairs with the GPU boxes:
[Colab](./bitnet_colab_mcp_setup.md) (1.1B gate runs) and the
[Windows 3080 box](./windows_dev_environment.md) (10 GB, branch-killer).

## Env

```text
venv   : .venv  (Python 3.12.13)
torch  : 2.12.1  (MPS available)
deps   : requirements-mac.txt  (transformers/datasets/accelerate/numpy/...; NO bitsandbytes)
```

Create / repair:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements-mac.txt
```

The scripts auto-pick the device `cuda -> mps -> cpu`, so on the Mac they use MPS with no flag.
Always export the MPS CPU-fallback so an unimplemented op degrades instead of crashing:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -X utf8 scripts/dino_i2s_selfdistill_smoke.py --steps 400
```

## What runs here vs not

| workload | here (Mac/MPS)? | why |
| --- | --- | --- |
| 160M DINO smoke / sweep / DIAG | YES | small model, plain adamw, MPS fast enough; the cheap research loop |
| scoring / token analysis (score_dir, dino_diag, score_dino002) | YES | inference only |
| **1.1B DINO training** | NO (impractical) | bitsandbytes is CUDA-only -> plain adamw ~16 GB + slow/flaky MPS -> use Colab/3080 |
| GGUF / bitnet.cpp i2_s runtime | NO | that is the [3080/Windows](./windows_dev_environment.md) + bitnet.cpp path |

## Gotchas

- **`--optim adamw8bit` will fail on Mac** (bitsandbytes CUDA-only). Use `--optim adamw` (the default).
  rt116 only imports bitsandbytes inside the `adamw8bit` branch, so the default path is clean.
- **`--dtype float32`** (the default) is the safe MPS choice; bf16/fp16 training on MPS can be flaky.
- **pyarrow**: 24.x works on Mac. The `pyarrow==21.0.0` pin in the project history is a *Windows-only*
  fix for a `pyarrow.dataset` segfault; do not apply it here.
- MPS is slower than CUDA; keep Mac jobs at 160M. For a quick check use `--code-smoke`.
