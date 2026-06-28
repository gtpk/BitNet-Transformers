#!/usr/bin/env python3
"""EGROW-001: per-layer I2_S instability instrumentation (docs/entropy_guided_i2s_growth_plan.md).

Logger ONLY. During a short 160M content-KL + PopQA-blend adaptation, log per target linear:
  flip_rate              F_l = mean 1[T^t != T^{t-1}]                 (ternary bucket churn)
  temporal_entropy       H_time = mean per-weight entropy of recent ternary states (subsampled)
  gradient_cosine_conflict  C_l = mean (1 - cos(g^t, g^{t-1}))
  update_reversal_rate   R_l = mean 1[<dtheta^t, dtheta^{t-1}> < 0]
  output_residual        E_l = E_x ||W x - Q(W) x||^2 / E_x ||W x||^2 (small batch, activation-weighted)
  task_saliency          S_l = mean activation_norm * weight_grad_norm on a FACT/PopQA batch
Then (instability) I_l = mean(norm(H,F,C,R)); (bottleneck) B_l = I_l * norm(E_l) * norm(S_l).
The MULTIPLICATIVE form means a layer is only a growth candidate when it is unstable AND has high
residual AND high task saliency -- separating optimizer noise from a real capacity bottleneck.

This does NOT grow anything; it ranks layers so EGROW-002/SIDE can put a sidecar on the top-k by B_l
(vs random-k). First success: top layers by B_l are stable across 2 seeds and are not just "last
layers only".

USAGE (3080 box): python -X utf8 scripts/egrow001_instrument.py --seed 41 --steps 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bitnet_llama.module import PerTensorBitLinear  # noqa: E402
from rt116_quality_recovery import (  # noqa: E402
    replace_targets, load_corpus, _panel_exclude_set, _instruction_ids_mask,
    base_kl_replay_term, _factual_ids_mask)


def ternary(W):
    g = W.detach().abs().mean().clamp(min=1e-12)
    return torch.clamp(torch.round(W.detach() / g), -1, 1)


def norm01(d):
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return {k: (v - lo) / rng for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40, help="log signals only after this step")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-tokens", type=int, default=800_000)
    ap.add_argument("--blend-file", type=Path, default=REPO_ROOT / "data/popqa_blend_train.jsonl")
    ap.add_argument("--blend-frac", type=float, default=0.05)
    ap.add_argument("--ent-window", type=int, default=16)
    ap.add_argument("--ent-sub", type=int, default=4096, help="weights subsampled per layer for entropy")
    ap.add_argument("--json-out", type=Path, default=REPO_ROOT / "reports/egrow_160m_layer_instability.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device)
    panel_excl = _panel_exclude_set(REPO_ROOT / "data/factual_panel_v1.jsonl")
    train_ids, eval_ids, train_mask = load_corpus(
        "mixed", tok, args.max_train_tokens, 60000, answer_mask=True, exclude_texts=panel_excl,
        factual_blend_file=args.blend_file, factual_blend_frac=args.blend_frac)
    # content-KL teacher + replay (same recipe as the best adaptation)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    r_ids, r_mask = _instruction_ids_mask(tok, exclude_texts=panel_excl)
    r_ids, r_mask = r_ids[:200000], r_mask[:200000]
    special = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=device)
    for i in sorted(set(int(i) for i in (tok.all_special_ids or []) if i is not None)):
        special[i] = True

    n_lin = replace_targets(model)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    targets = {n: m for n, m in model.named_modules() if isinstance(m, PerTensorBitLinear)}
    for m in targets.values():
        m.weight.requires_grad_(True)
    opt = torch.optim.AdamW([m.weight for m in targets.values()], lr=args.lr)
    print(f"device={device} model={args.model_id} seed={args.seed} steps={args.steps} "
          f"target_linears={len(targets)} blend={args.blend_frac}", flush=True)

    # per-layer state
    sub_idx = {n: torch.randperm(m.weight.numel(), device=device)[: args.ent_sub] for n, m in targets.items()}
    prev_T = {n: ternary(m.weight) for n, m in targets.items()}
    ent_win = {n: deque(maxlen=args.ent_window) for n in targets}
    prev_g = {n: None for n in targets}
    prev_dW = {n: None for n in targets}
    acc = {n: {"flip": 0.0, "conf": 0.0, "rev": 0.0, "k": 0} for n in targets}

    g = torch.Generator().manual_seed(args.seed)
    usable = train_ids.numel() - 1
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        starts = torch.randint(0, max(1, usable - args.seq_len), (args.batch,), generator=g).tolist()
        x = torch.stack([train_ids[s:s + args.seq_len] for s in starts]).to(device)
        m_ = torch.stack([train_mask[s:s + args.seq_len] for s in starts]).to(device)
        labels = x.clone(); labels[~m_] = -100
        if (labels != -100).any():
            loss = model(input_ids=x, labels=labels).loss
            kl = base_kl_replay_term(model, teacher, r_ids, r_mask, args.seq_len, 2, 1.0, g, device, special_vocab=special)
            if kl is not None:
                loss = loss + 0.2 * kl
            W_before = {n: m.weight.detach().clone() for n, m in targets.items()}
            loss.backward()
            grads = {n: m.weight.grad.detach().flatten() for n, m in targets.items()}
            opt.step()
            if step >= args.warmup:
                for n, m in targets.items():
                    T = ternary(m.weight)
                    acc[n]["flip"] += (T != prev_T[n]).float().mean().item()
                    dW = (m.weight.detach() - W_before[n]).flatten()
                    if prev_g[n] is not None:
                        acc[n]["conf"] += (1 - F.cosine_similarity(grads[n], prev_g[n], dim=0)).item()
                    if prev_dW[n] is not None:
                        acc[n]["rev"] += float((dW @ prev_dW[n]).item() < 0)
                    acc[n]["k"] += 1
                    ent_win[n].append(T.flatten()[sub_idx[n]].to(torch.int8).cpu())
                    prev_T[n] = T; prev_g[n] = grads[n]; prev_dW[n] = dW
            else:
                for n, m in targets.items():
                    prev_T[n] = ternary(m.weight); prev_g[n] = grads[n]
                    prev_dW[n] = (m.weight.detach() - W_before[n]).flatten()
        if step % 25 == 0:
            print(f"  step {step}/{args.steps}  loss {float(loss):.4f}", flush=True)

    # temporal entropy from the subsampled window
    temp_ent = {}
    for n, win in ent_win.items():
        if len(win) < 2:
            temp_ent[n] = 0.0; continue
        stk = torch.stack(list(win)).float()  # [W, sub]
        H = 0.0
        for k in (-1, 0, 1):
            p = (stk == k).float().mean(0)
            H = H - (p * (p + 1e-12).log()).sum().item()
        temp_ent[n] = H / stk.shape[1]

    # output residual + task saliency on a small batch (activation-weighted)
    model.eval()
    inp_cache = {}
    hooks = [m.register_forward_pre_hook(lambda mod, a, nm=n: inp_cache.__setitem__(nm, a[0].detach()))
             for n, m in targets.items()]
    sb_starts = torch.randint(0, max(1, usable - args.seq_len), (4,), generator=g).tolist()
    xb = torch.stack([train_ids[s:s + args.seq_len] for s in sb_starts]).to(device)
    mb = torch.stack([train_mask[s:s + args.seq_len] for s in sb_starts]).to(device)
    lb = xb.clone(); lb[~mb] = -100
    out = model(input_ids=xb, labels=lb)
    for h in hooks:
        h.remove()
    out_res = {}
    for n, m in targets.items():
        xin = inp_cache[n].reshape(-1, m.in_features).float()
        W = m.weight.detach().float()
        Wq = ternary(W) * W.abs().mean().clamp(min=1e-12)
        num = ((xin @ (W - Wq).T) ** 2).sum().item()
        den = ((xin @ W.T) ** 2).sum().item() + 1e-12
        out_res[n] = num / den
    # task saliency: grad on the (answer-masked) loss, S = act_norm * grad_norm
    model.train()
    for m in targets.values():
        m.weight.grad = None
    out.loss.backward()
    sal = {}
    for n, m in targets.items():
        an = inp_cache[n].float().norm().item()
        gn = m.weight.grad.detach().float().norm().item() if m.weight.grad is not None else 0.0
        sal[n] = an * gn

    flip = {n: acc[n]["flip"] / max(acc[n]["k"], 1) for n in targets}
    conf = {n: acc[n]["conf"] / max(acc[n]["k"], 1) for n in targets}
    rev = {n: acc[n]["rev"] / max(acc[n]["k"], 1) for n in targets}
    nH, nF, nC, nR = norm01(temp_ent), norm01(flip), norm01(conf), norm01(rev)
    nE, nS = norm01(out_res), norm01(sal)
    rows = []
    for n in targets:
        inst = 0.25 * (nH[n] + nF[n] + nC[n] + nR[n])
        B = inst * nE[n] * nS[n]
        mt = n.split(".")[-1]
        rows.append({"layer_name": n, "module_type": mt, "flip_rate": round(flip[n], 5),
                     "temporal_entropy": round(temp_ent[n], 5), "gradient_cosine_conflict": round(conf[n], 5),
                     "update_reversal_rate": round(rev[n], 5), "output_residual": round(out_res[n], 5),
                     "task_saliency": round(sal[n], 3), "instability": round(inst, 4),
                     "bottleneck_score": round(B, 5)})
    rows.sort(key=lambda r: r["bottleneck_score"], reverse=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps({"seed": args.seed, "model": args.model_id, "steps": args.steps,
                                         "layers": rows}, indent=2), encoding="utf-8")
    print("\nTOP-8 bottleneck layers (B_l):", flush=True)
    for r in rows[:8]:
        print(f"  {r['bottleneck_score']:.4f}  {r['layer_name']}  "
              f"(flip {r['flip_rate']:.3f} Hent {r['temporal_entropy']:.3f} res {r['output_residual']:.3f} "
              f"sal {r['task_saliency']:.1f})", flush=True)
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
