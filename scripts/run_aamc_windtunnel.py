#!/usr/bin/env python3
"""AAMC overfit wind-tunnel driver (docs/adaptive_anchor_manifold_controller_plan.md).

Runs 4 arms SEQUENTIALLY on one GPU (built for the RTX 3080 / Qwen-0.5B), in a deliberately
overfit-prone regime (lr 3e-4, 1000 steps), to test whether the AAMC controller REACTS -- i.e.
sees overfit_score rise and raises lambda (and turns on weak DINO only on collapse) -- NOT to
claim final quality. Resumable: skips any arm whose train.json already exists.

Arms:
  fixed020  lambda 0.2 fixed            (under-anchored baseline)
  fixed040  lambda 0.4 fixed            (stronger anchor baseline)
  dynlam    AAMC, lambda 0.2->raise, alpha capped 0  (dynamic-lambda only, DINO never on)
  dyndino   AAMC, lambda 0.2->raise, alpha up to 0.10 (conditional DINO only on collapse)
"""
import subprocess, sys, os, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
PY = sys.executable
BASE = "reports/aamc_wt"
CKBASE = "bnt_ckpt_wt"

COMMON = [
    "--model-id", "Qwen/Qwen2.5-0.5B-Instruct", "--dtype", "bfloat16", "--teacher-dtype", "float16",
    "--seq-len", "256", "--batch", "2", "--grad-accum-steps", "12", "--lr", "3e-4",
    "--train-source", "mixed", "--answer-loss-only", "--base-kl-replay", "--kl-temp", "1.0",
    "--kl-content-only", "--replay-tokens", "200000", "--replay-batch", "1",
    "--telemetry-full", "--telemetry-probe-n", "10", "--seed", "1", "--steps", "1000",
    "--fact-eval-steps", "400,600,800,1000",
]
ARMS = [
    ("fixed020", ["--kl-weight", "0.2"]),
    ("fixed040", ["--kl-weight", "0.4"]),
    ("dynlam",   ["--kl-weight", "0.2", "--aamc", "--aamc-score-every", "200",
                  "--aamc-max-alpha", "0.0", "--aamc-min-step", "300"]),
    ("dyndino",  ["--kl-weight", "0.2", "--aamc", "--aamc-score-every", "200",
                  "--aamc-max-alpha", "0.10", "--aamc-min-step", "300"]),
]

for name, extra in ARMS:
    out = f"{BASE}/{name}"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/train.json"):
        print(f"[windtunnel] {name} already done -> skip", flush=True)
        continue
    cmd = [PY, "-X", "utf8", "scripts/rt116_quality_recovery.py"] + COMMON + extra + [
        "--out-dir", f"{out}/adapted", "--metrics-out", f"{out}/metrics.jsonl",
        "--json-out", f"{out}/train.json", "--ckpt-dir", f"{CKBASE}/{name}", "--ckpt-every-min", "30",
    ]
    print(f"[windtunnel] START {name} :: {' '.join(extra)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"[windtunnel] END {name} rc={rc} ({(time.time()-t0)/60:.1f}m)", flush=True)
print("[windtunnel] ALL DONE", flush=True)
