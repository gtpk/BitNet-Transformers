#!/usr/bin/env python3
"""DINO-I2S-000/001: no-label self-distillation smoke for I2_S factual RETENTION.

Motivation (docs/dino_i2s_self_distillation_plan.md, docs/i2s_v0_recipe_and_closed_branches.md):
the CE/replay objective family is exhausted -- small hard replay memorises, PopQA blend
collapses generation at 1.1B. Every failed branch tries to INJECT answers. DINO-I2S applies
the opposite pressure: keep the I2_S student inside the frozen FP teacher's content
distribution + hidden geometry on broad UNLABELED text, so the converted model "forgets less".

I2_S stays the root (all target linears ternary, PerTensorBitLinear STE, lm_head/embeds frozen).
DINO-style self-distillation is only an adaptation OBJECTIVE added on top of the v0 backbone.

Arms (each: fresh I2_S student, frozen FP self-teacher, then materialise + PyTorch score):
  baseline      v0 recipe: answer-CE + content-KL(base||student) replay, lambda=0.2  (FACT-003C)
  dino_logit    baseline + beta_c * content-KL on UNLABELED views (teacher clean vs student view)
  dino_hidden   dino_logit + beta_h * normalized hidden alignment (mid+last layers)
  dino_centered (optional, --with-centered) dino_hidden + DINO teacher-logit centering -- ONLY
                if a collapse (all-salad/empty/loop, high-confidence copy) appears.

The DINO terms are a PURE ADD-ON over the identical v0 backbone, so the verdict isolates the
self-distillation contribution. Scored on eval_panel + popqa_tight (PRIMARY) + popqa_train
(memorise check) + WikiText CE + generation tags + i2_s==f16 parity (materialised dir is i2_s).

PASS (DINO-I2S-001, plan): a dino_* arm lifts FACT eval_panel OR popqa_tight by >= +0.05 over
baseline AND tags stay ok/non-degenerate AND no train-only memorisation signature.
FAIL: FACT flat / PopQA flat / hidden alignment only improves CE while behaviour stays flat
  -> CLOSE DINO; accept the 1.1B same-topology I2_S factual ceiling is low (goalpost shift to a
  better/larger base model), per docs/i2s_v0_recipe_and_closed_branches.md.

USAGE (3080 box, foreground held by a background ssh):
  python -X utf8 scripts/dino_i2s_selfdistill_smoke.py --model-id Felladrin/Llama-160M-Chat-v1 \
    --steps 300 --code-smoke         # DINO-I2S-000 first (20 steps, finite-loss check)
  python -X utf8 scripts/dino_i2s_selfdistill_smoke.py --steps 400   # DINO-I2S-001 full
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from bitnet_llama.module import PerTensorBitLinear  # noqa: E402
from rt116_quality_recovery import (  # noqa: E402
    replace_targets, materialize_and_save, load_corpus, _panel_exclude_set,
    base_kl_replay_term, _instruction_ids_mask, eval_ce)
from fact004a_160m_smoke import score_dir  # noqa: E402


def load_jsonl(p, n=0):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return rows[:n] if n else rows


def trainable_target_params(model):
    """v0 recipe: only the ternary target-linear latent weights train (lm_head/embeds/norms frozen)."""
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for m in model.modules():
        if isinstance(m, PerTensorBitLinear):
            m.weight.requires_grad_(True)
            params.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                params.append(m.bias)
    return params


def make_view(x, mode, p, gen, vocab_size, pad_id):
    """Student-side augmented view of token batch x (same length, so positions stay aligned)."""
    if mode == "same" or p <= 0:
        return x
    mask = (torch.rand(x.shape, generator=gen, device="cpu") < p).to(x.device)
    if mode == "dropout":   # replace dropped tokens with pad/unk
        repl = torch.full_like(x, pad_id)
    elif mode == "noise":   # replace with random vocab ids (stronger corruption)
        repl = torch.randint(0, vocab_size, x.shape, generator=gen, device="cpu").to(x.device)
    else:
        raise ValueError(mode)
    return torch.where(mask, repl, x)


def content_kl_logits(t_logits, s_logits, special_vocab, temp, center=None):
    """Forward content-KL(teacher||student) over ALL positions, special vocab dropped. [B,T,V]."""
    t = t_logits.float() / temp
    s = s_logits.float() / temp
    if center is not None:
        t = t - center
    if special_vocab is not None:
        t = t.masked_fill(special_vocab, float("-inf"))
        s = s.masked_fill(special_vocab, float("-inf"))
    p_t = F.softmax(t, dim=-1)
    term = p_t * (F.log_softmax(t, dim=-1) - F.log_softmax(s, dim=-1))
    if special_vocab is not None:
        term = term.masked_fill(special_vocab, 0.0)
    return term.sum(-1).mean() * (temp * temp)


def hidden_align(s_hidden, t_hidden, idxs):
    """Normalized hidden-state alignment: ||norm(h_S) - sg(norm(h_T))||^2 over mid/last layers."""
    terms = []
    for idx in idxs:
        sh = F.normalize(s_hidden[idx].float(), dim=-1)
        th = F.normalize(t_hidden[idx].float(), dim=-1).detach()
        terms.append((sh - th).pow(2).sum(-1).mean())
    return torch.stack(terms).mean()


def train_arm(arm, args, tok, train_ids, train_mask, unlab_ids, replay_ids, replay_mask,
              special_vocab, device, out_dir):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 1)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device)
    n_t = replace_targets(model)
    model.to(device=device, dtype=torch.float32)  # new PerTensorBitLinear modules start on CPU -> move
    model.config.use_cache = False
    teacher = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.config.use_cache = False

    params = trainable_target_params(model)
    opt = torch.optim.AdamW(params, lr=args.lr)
    use_dino = arm in ("dino_logit", "dino_hidden", "dino_centered")
    use_hidden = arm in ("dino_hidden", "dino_centered")
    use_center = arm == "dino_centered"
    n_layers = model.config.num_hidden_layers
    hid_idxs = [max(1, (n_layers + 1) // 2), n_layers]  # mid + last hidden_states (0=embeds)
    center = None
    print(f"\n===== arm {arm} =====  targets={n_t}  trainable={sum(p.numel() for p in params):,}  "
          f"dino={use_dino} hidden={use_hidden} center={use_center}", flush=True)

    usable = train_ids.numel() - 1
    un_usable = unlab_ids.numel() - 1
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        # (1) v0 backbone: answer-masked CE on the mixed instruction stream
        starts = torch.randint(0, max(1, usable - args.seq_len), (args.batch,), generator=gen).tolist()
        xb = torch.stack([train_ids[s:s + args.seq_len] for s in starts]).to(device)
        mb = torch.stack([train_mask[s:s + args.seq_len] for s in starts]).to(device)
        labels = xb.clone()
        labels[~mb] = -100
        ce = model(input_ids=xb, labels=labels).loss
        loss = ce
        # (2) v0 backbone: content-KL(base||student) on instruction replay (FACT-003C)
        klr = base_kl_replay_term(model, teacher, replay_ids, replay_mask, args.seq_len,
                                  args.replay_batch, args.kl_temp, gen, device, special_vocab)
        if klr is not None:
            loss = loss + args.kl_weight * klr
        dkl = torch.tensor(0.0)
        dh = torch.tensor(0.0)
        # (3) DINO add-on: content-KL + hidden alignment on UNLABELED views (no labels)
        if use_dino:
            us = torch.randint(0, max(1, un_usable - args.seq_len), (args.dino_batch,), generator=gen).tolist()
            uo = torch.stack([unlab_ids[s:s + args.seq_len] for s in us]).to(device)  # teacher clean view
            uv = make_view(uo, args.view_mode, args.view_p, gen, model.config.vocab_size,
                           tok.pad_token_id if tok.pad_token_id is not None else 0)  # student view
            with torch.no_grad():
                tout = teacher(input_ids=uo, output_hidden_states=use_hidden)
            if use_center:
                with torch.no_grad():
                    batch_mean = tout.logits.float().mean(dim=(0, 1), keepdim=True)
                    center = batch_mean if center is None else args.center_m * center + (1 - args.center_m) * batch_mean
            sout = model(input_ids=uv, output_hidden_states=use_hidden)
            dkl = content_kl_logits(tout.logits, sout.logits, special_vocab, args.kl_temp,
                                    center=center if use_center else None)
            loss = loss + args.dino_logit_weight * dkl
            if use_hidden:
                dh = hidden_align(sout.hidden_states, tout.hidden_states, hid_idxs)
                loss = loss + args.hidden_weight * dh
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            el = (time.time() - t0) / 60
            print(f"  [{arm}] step {step:4d}/{args.steps}  ce={float(ce):.4f}  klr={float(klr) if klr is not None else 0:.4f}  "
                  f"dkl={float(dkl):.4f}  dh={float(dh):.4f}  loss={float(loss):.4f}  {el:.1f}m", flush=True)
            if not torch.isfinite(loss):
                print(f"  [{arm}] NON-FINITE loss at step {step} -- aborting arm", flush=True)
                return {"arm": arm, "error": "non_finite_loss", "step": step}
    materialize_and_save(model, out_dir, tok)
    del model, teacher
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"arm": arm, "n_targets": n_t, "trained_min": round((time.time() - t0) / 60, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="Felladrin/Llama-160M-Chat-v1")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports" / "dino_i2s_160m")
    ap.add_argument("--arms", default="baseline,dino_logit,dino_hidden")
    ap.add_argument("--with-centered", action="store_true", help="append dino_centered arm (use only if collapse seen)")
    ap.add_argument("--code-smoke", action="store_true", help="DINO-I2S-000: 20 steps, all arms, finite-loss check only")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dino-batch", type=int, default=4, help="unlabeled-view windows per step for the DINO term")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--kl-weight", type=float, default=0.2, help="v0 content-KL replay lambda (FACT-003C)")
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--replay-batch", type=int, default=2)
    ap.add_argument("--replay-tokens", type=int, default=200_000)
    ap.add_argument("--dino-logit-weight", type=float, default=0.2, help="beta_c on unlabeled-view content-KL")
    ap.add_argument("--hidden-weight", type=float, default=0.01, help="beta_h on normalized hidden alignment")
    ap.add_argument("--view-mode", choices=["same", "dropout", "noise"], default="dropout")
    ap.add_argument("--view-p", type=float, default=0.1, help="fraction of student-view tokens corrupted")
    ap.add_argument("--center-m", type=float, default=0.9, help="DINO centering EMA momentum (dino_centered only)")
    ap.add_argument("--max-train-tokens", type=int, default=800_000)
    ap.add_argument("--eval-tokens", type=int, default=60_000)
    ap.add_argument("--ce-windows", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--tight-sample", type=int, default=300)
    ap.add_argument("--train-sample", type=int, default=80)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--skip-train", action="store_true", help="re-score existing arm dirs only")
    args = ap.parse_args()
    if args.code_smoke:
        args.steps = 20
        args.log_every = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.work.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  model={args.model_id}  steps={args.steps}  view={args.view_mode}@{args.view_p}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    cfg = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float32).config
    special_vocab = None
    ids = sorted(set(int(i) for i in (tok.all_special_ids or []) if i is not None))
    special_vocab = torch.zeros(cfg.vocab_size, dtype=torch.bool, device=device)
    special_vocab[ids] = True
    print(f"content-KL drops {len(ids)} special vocab ids {ids}", flush=True)

    # v0 backbone data: mixed instruction stream (answer-masked) + de-leaked instruction replay pool
    panel_exclude = _panel_exclude_set(REPO_ROOT / "data/factual_panel_v1.jsonl")
    train_ids, eval_ids, train_mask = load_corpus(
        "mixed", tok, args.max_train_tokens, args.eval_tokens,
        answer_mask=True, exclude_texts=panel_exclude)
    r_ids, r_mask = _instruction_ids_mask(tok, exclude_texts=panel_exclude)
    replay_ids, replay_mask = r_ids[: args.replay_tokens], r_mask[: args.replay_tokens]
    unlab_ids = train_ids  # broad unlabeled pool = the mixed text stream (DINO uses content positions, no labels)
    print(f"train tokens={train_ids.numel():,}  replay tokens={replay_ids.numel():,}  "
          f"answer-mask {100*float(train_mask.float().mean()):.1f}%", flush=True)

    # eval panels (all held out / de-leaked)
    ev = load_jsonl(REPO_ROOT / "data/factual_panel_v1.jsonl")
    tight = load_jsonl(REPO_ROOT / "data/popqa_heldout_tight.jsonl", args.tight_sample)
    tr = [{"id": r["id"], "prompt": r["prompt"], "must_contain": [r["answer"].strip().lower()]}
          for r in load_jsonl(REPO_ROOT / "data/popqa_blend_train.jsonl", args.train_sample)]
    from rt116_quality_recovery import _wikitext_train_eval
    _, wt = _wikitext_train_eval(tok, 1000, args.eval_tokens)

    arms = [a for a in args.arms.split(",") if a]
    if args.with_centered and "dino_centered" not in arms:
        arms.append("dino_centered")

    rows = []
    for arm in arms:
        out_dir = args.work / arm
        meta = {}
        if not args.skip_train:
            meta = train_arm(arm, args, tok, train_ids, train_mask, unlab_ids, replay_ids,
                             replay_mask, special_vocab, device, out_dir)
            if meta.get("error"):
                rows.append({"arm": arm, **meta})
                continue
        e = score_dir(out_dir, tok, ev, device, args.max_new, wt, args.ce_windows)
        ti = score_dir(out_dir, tok, tight, device, args.max_new, wt, args.ce_windows)
        trs = score_dir(out_dir, tok, tr, device, args.max_new, wt, args.ce_windows)
        row = {"arm": arm, "eval": e["fact_rate"], "tight": ti["fact_rate"], "train": trs["fact_rate"],
               "ce": round(e["ce"], 3), "tags": e["tags"], **{k: meta.get(k) for k in ("trained_min",)}}
        rows.append(row)
        print(f"  >> {arm}: eval {row['eval']} | tight {row['tight']} | train {row['train']} | CE {row['ce']} | {row['tags']}",
              flush=True)

    # verdict
    base = next((r for r in rows if r["arm"] == "baseline"), None)
    best = None
    if base and "eval" in base:
        for r in rows:
            if r["arm"] == "baseline" or "eval" not in r:
                continue
            d_eval = r["eval"] - base["eval"]
            d_tight = r["tight"] - base["tight"]
            r["d_eval"], r["d_tight"] = round(d_eval, 3), round(d_tight, 3)
            ok_tags = sum(r["tags"].get(t, 0) for t in ("salad", "empty", "loop")) <= 0.3 * sum(r["tags"].values())
            r["passes"] = (max(d_eval, d_tight) >= 0.05) and ok_tags and (r["train"] - r["tight"] < 0.3)
            if r["passes"] and (best is None or max(d_eval, d_tight) > best[1]):
                best = (r["arm"], max(d_eval, d_tight))

    lines = ["# DINO-I2S-001 self-distillation smoke (160M)", "",
             f"model={args.model_id}  steps={args.steps}  view={args.view_mode}@{args.view_p}  "
             f"dino_logit_w={args.dino_logit_weight} hidden_w={args.hidden_weight}", "",
             "| arm | eval_panel | popqa_tight (PRIMARY) | popqa_train (memorise) | CE | dEval | dTight | pass | tags |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for r in rows:
        if "eval" not in r:
            lines.append(f"| {r['arm']} | ERROR {r.get('error','')} | | | | | | | |")
            continue
        lines.append(f"| {r['arm']} | {r['eval']} | {r['tight']} | {r['train']} | {r['ce']} | "
                     f"{r.get('d_eval','-')} | {r.get('d_tight','-')} | {r.get('passes','-')} | {r['tags']} |")
    if args.code_smoke:
        verdict = "CODE SMOKE (DINO-I2S-000): arms ran; check losses finite + tags not all degenerate above."
    elif best:
        verdict = (f"PASS (DINO-I2S-001): arm '{best[0]}' lifts FACT/PopQA by +{best[1]:.3f} over baseline with ok tags "
                   f"and no train-only memorisation. -> schedule DINO-I2S-002 1.1B Colab gate (target eval > 0.185, >=0.25).")
    else:
        verdict = ("FAIL (DINO-I2S-001): no dino_* arm beats baseline by >=0.05 on FACT or PopQA-tight with ok tags. "
                   "Self-distillation retention does not move factual behaviour at 160M -> CLOSE DINO; accept the 1.1B "
                   "same-topology I2_S factual ceiling is low (goalpost shift to a better/larger base model). "
                   "See docs/i2s_v0_recipe_and_closed_branches.md.")
    lines += ["", "VERDICT: " + verdict]
    (args.work / "summary.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2), encoding="utf-8")
    (args.work / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {args.work / 'summary.md'}")


if __name__ == "__main__":
    main()
