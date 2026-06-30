#!/usr/bin/env python3
"""RT-116 / TRAIN-001 (QR-001..003): teacher-free b1.58 quality recovery on Llama-160M.

RT-112..115 closed the systems story (small + fast + faithful at scale). The open
question is quality: a one-shot PTQ ternary model has terrible absolute PPL by
design — can a SHORT, teacher-free CE-only adaptation recover a meaningful fraction
of the FP->PTQ loss? See docs/quality_recovery_plan.md for the full ladder.

This implements, all in PyTorch on one held-out text set (apples-to-apples, no
cross-tool noise):

  QR-001  collapse baseline : CE_fp (FP original) vs CE_ptq (Wq=gamma*T, no train)
  QR-002a CE adaptation     : replace target linears with per-tensor b1.58 STE,
                              freeze everything else, train next-token CE only,
                              report recovered_fraction = (CE_ptq-CE_adapted)
                                                          /(CE_ptq-CE_fp)
  QR-003  runtime preserve  : (with --bitnet) export adapted Wq=gamma*T -> f16 & i2_s
                              GGUF, llama-perplexity, check i2_s ~= f16 in nats.

Forward uses Wq=gamma*T (PerTensorBitLinear STE); backward updates the latent FP
weight through the STE. Teacher-free = plain next-token CE, no distillation.

USAGE (GPU strongly recommended for the training step):
  python scripts/rt116_quality_recovery.py --model-id JackFram/llama-160m \
    --steps 300 --seq-len 256 --batch 8 --lr 2e-4 \
    --bitnet /content/bitnet.cpp     # add --bitnet to also run QR-003 export

For the 1.1B budget-scaling run, keep `--batch` as the per-microbatch size that
fits in GPU memory and use `--grad-accum-steps` to raise the effective batch.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bitnet_llama import conversion as C  # noqa: E402
from bitnet_llama.module import PerTensorBitLinear  # noqa: E402
from bitnet_llama.sidecar import (  # noqa: E402
    I2SLoRALinear, wrap_targets_with_lora, sidecar_accounting)


def set_module(root, name, mod):
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, mod)


def replace_targets(model):
    """Swap target nn.Linear -> PerTensorBitLinear (latent = FP weight). Returns count."""
    n = 0
    for name, m in list(model.named_modules()):
        if isinstance(m, nn.Linear) and C.is_target_weight_key(f"{name}.weight"):
            repl = PerTensorBitLinear(m.in_features, m.out_features, bias=m.bias is not None)
            with torch.no_grad():
                repl.weight.copy_(m.weight)
                if m.bias is not None:
                    repl.bias.copy_(m.bias)
            set_module(model, name, repl)
            n += 1
    return n


def materialize_and_save(model, out_dir, tokenizer):
    """Replace PerTensorBitLinear -> dense Linear holding Wq=gamma*T, save HF dir."""
    import copy
    m2 = copy.deepcopy(model).to("cpu").float()
    # pass 1: fold I2_S+sidecar -> dense effective weight (gamma*T + scale*B@A) for scoring; the
    # deployable form keeps gamma*T as I2_S and the sidecar separate, but quality is identical.
    for name, mod in list(m2.named_modules()):
        if isinstance(mod, I2SLoRALinear):
            dense = nn.Linear(mod.in_features, mod.out_features, bias=mod.base.bias is not None)
            with torch.no_grad():
                dense.weight.copy_(mod.effective_weight())
                if mod.base.bias is not None:
                    dense.bias.copy_(mod.base.bias)
            set_module(m2, name, dense)
    # pass 2: remaining un-wrapped ternary bases -> dense gamma*T
    for name, mod in list(m2.named_modules()):
        if isinstance(mod, PerTensorBitLinear):
            dense = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None)
            with torch.no_grad():
                dense.weight.copy_(C.per_tensor_b158_approx(mod.weight))
                if mod.bias is not None:
                    dense.bias.copy_(mod.bias)
            set_module(m2, name, dense)
    out_dir.mkdir(parents=True, exist_ok=True)
    m2.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)


def load_wikitext(tokenizer, max_train_tokens, max_eval_tokens):
    from datasets import load_dataset
    try:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    def join_tok(split, cap):
        # Stream per-document and stop at `cap` tokens, rather than tokenizing the whole
        # concatenated corpus in one call. Equivalent token stream (docs are joined by "\n\n",
        # a natural Llama-tokenizer boundary; BOS kept on the first doc) but bounded memory and
        # no multi-MB single input -- the latter SEGFAULTS the `tokenizers` rust lib on Windows
        # (access violation / exit 5), while being a no-op correctness-wise on Linux.
        sep = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        ids, first = [], True
        for t in ds[split]["text"]:
            if not t.strip():
                continue
            if first:
                ids.extend(tokenizer(t, return_tensors=None)["input_ids"])
                first = False
            else:
                ids.extend(sep)
                ids.extend(tokenizer(t, add_special_tokens=False, return_tensors=None)["input_ids"])
            if len(ids) >= cap:
                break
        return torch.tensor(ids[:cap], dtype=torch.long)

    return join_tok("train", max_train_tokens), join_tok("validation", max_eval_tokens)


def _wikitext_train_eval(tokenizer, max_train_tokens, max_eval_tokens):
    return load_wikitext(tokenizer, max_train_tokens, max_eval_tokens)


def _instruction_text(max_examples=20000):
    """Dolly-15k formatted as 'Q: instruction [context]\\nA: response' (license: CC BY-SA)."""
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    parts = []
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ctx = (ex.get("context") or "").strip()
        q = ex["instruction"].strip() + (("\n" + ctx) if ctx else "")
        parts.append(f"Q: {q}\nA: {ex['response'].strip()}")
    return "\n\n".join(parts)


def _panel_exclude_set(panel_path):
    """Normalized FACT-001 panel prompts + distinctive (>=5 char) answers, for de-leaking the
    instruction/replay stream so adaptation/anchoring never trains on the eval (FACT-003B/C)."""
    out = set()
    for line in open(panel_path):
        if not line.strip():
            continue
        p = json.loads(line)
        out.add(re.sub(r"\s+", " ", p["prompt"]).strip().lower())
        for a in p.get("must_contain", []):
            a = re.sub(r"\s+", " ", a).strip().lower()
            if len(a) >= 5:
                out.add(a)
    return out


def _instruction_ids_mask(tokenizer, max_examples=20000, exclude_texts=None):
    """FACT-003A: Dolly Q/A as a token stream + answer mask (True on response tokens).

    Each example is tokenized as prompt 'Q: ..\\nA:' + answer ' <response>' + '\\n\\n' sep,
    with the answer-mask True only on the response tokens. Under --answer-loss-only the CE
    is computed on these tokens only, so the model is not trained to reproduce the Q/A prompt
    formatting (the thing instruction-only adaptation overfit into empty-answer collapse, FACT-002).

    exclude_texts (FACT-003B/C de-leak): if given, any example whose normalized text contains
    one of these strings (panel prompts/answers) is dropped, so the stream is disjoint from the
    factual eval panel.
    """
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    sep = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    ids, mask = [], []
    dropped = 0
    for ex in ds.select(range(min(max_examples, len(ds)))):
        ctx = (ex.get("context") or "").strip()
        q = ex["instruction"].strip() + (("\n" + ctx) if ctx else "")
        full = f"Q: {q}\nA: {ex['response'].strip()}"
        if exclude_texts:
            norm = re.sub(r"\s+", " ", full).strip().lower()
            if any(x in norm for x in exclude_texts):
                dropped += 1
                continue
        p = tokenizer(f"Q: {q}\nA:", add_special_tokens=False)["input_ids"]
        a = tokenizer(" " + ex["response"].strip(), add_special_tokens=False)["input_ids"]
        ids += p + a + sep
        mask += [0] * len(p) + [1] * len(a) + [0] * len(sep)
    if exclude_texts:
        print(f"  de-leak: dropped {dropped} Dolly examples overlapping the factual panel")
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _factual_ids_mask(tokenizer, path, exclude_texts=None):
    """FACT-003D: protected factual replay set as an answer-masked token stream.

    Reads a jsonl of {"prompt": "Q: ..\\nA:", "answer": " <short answer>"} (the curated atomic
    facts from make_atomic_facts.py) and builds the same prompt-masked / answer-kept stream as
    _instruction_ids_mask, so the factual CE counts ONLY the answer tokens. exclude_texts is a
    de-leak safety net: any item whose normalized text contains a panel prompt/answer is dropped
    (the generator already excludes these; this asserts it again at load)."""
    sep = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    ids, mask = [], []
    dropped = 0
    n = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        ex = json.loads(line)
        prompt, answer = ex["prompt"], ex["answer"]
        if exclude_texts:
            norm = re.sub(r"\s+", " ", prompt + " " + answer).strip().lower()
            if any(x in norm for x in exclude_texts):
                dropped += 1
                continue
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        a = tokenizer(answer if answer.startswith(" ") else " " + answer, add_special_tokens=False)["input_ids"]
        ids += p + a + sep
        mask += [0] * len(p) + [1] * len(a) + [0] * len(sep)
        n += 1
    if exclude_texts and dropped:
        raise ValueError(f"factual replay set {path} LEAKS vs panel: {dropped} items overlap "
                         f"(regenerate with make_atomic_facts.py)")
    print(f"  FACT-003D protected factual replay: {n} atomic facts, {len(ids):,} tokens "
          f"({100*sum(mask)/max(len(mask),1):.1f}% answer)")
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def factual_ce_term(model, fids, fmask, seq_len, fbatch, gen, device):
    """FACT-003D: answer-masked CE on a sampled window of the protected factual stream.

    Direct CE (NOT KL): we want the model to PRESERVE these exact answer tokens, not anchor to a
    base distribution. Returns a scalar loss tensor (grad through student) or None if the sampled
    window has no answer tokens."""
    usable = fids.numel()
    win = min(seq_len, usable)
    starts = torch.randint(0, max(1, usable - win), (fbatch,), generator=gen).tolist()
    fx = torch.stack([fids[s : s + win] for s in starts]).to(device)
    fm = torch.stack([fmask[s : s + win] for s in starts]).to(device)
    if not fm.any():
        return None
    labels = fx.clone()
    labels[~fm] = -100
    if not (labels != -100).any():
        return None
    return model(input_ids=fx, labels=labels).loss


def answer_token_weighted_ce(model, x, m, beta, first_k):
    """ANS-001: answer-masked next-token CE with EXTRA weight beta on the FIRST-k tokens of each answer
    span. gold_rank shows the answer token is reachable but not emitted (the model rambles); concentrating
    gradient on the first answer token(s) pulls the answer to the front / short form. Returns scalar loss."""
    import torch.nn.functional as F
    logits = model(input_ids=x).logits[:, :-1].float()      # predict token t+1
    tgt = x[:, 1:]
    mb = m[:, 1:].bool()                                     # answer mask aligned to targets
    B, Tm1, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(B, Tm1)
    prev = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), mb[:, :-1]], dim=1)
    first = (mb & ~prev).float()                            # answer-span starts (F->T transitions)
    boost = first.clone()
    for k in range(1, first_k):                             # also the k-1 tokens after each start
        boost = boost + torch.cat([torch.zeros(B, k, device=x.device), first[:, :-k]], dim=1)
    w = (1.0 + beta * boost) * mb.float()                   # answer tokens only; first-k up-weighted
    return (ce * w).sum() / w.sum().clamp_min(1.0)


def load_corpus(source, tokenizer, max_train_tokens, max_eval_tokens, answer_mask=False, exclude_texts=None,
                factual_blend_file=None, factual_blend_frac=0.0):
    """Returns (train_ids, eval_ids, train_answer_mask). eval is ALWAYS WikiText validation.

    train_answer_mask[i] is True for tokens whose CE counts under --answer-loss-only: response
    tokens for instruction data, and all content tokens for WikiText (no prompt/answer split).
    When answer_mask=False the mask is all-True and the token stream is byte-identical to the
    FACT-002 runs (the masked-stream tokenization is only built when actually needed).
    exclude_texts de-leaks the instruction stream against the factual panel (FACT-003B).

    FACT-003G mixed-stream factual blend: if factual_blend_frac>0, that fraction of the mixed
    train tokens are protected factual QA BLENDED into the one stream under the SAME answer-only
    CE -- NOT a separate strong loss (which FACT-003D mu*CE was, and it overfit: the model just
    memorised the small set). Here facts are a low-ratio part of the normal Q/A distribution, so
    the lever is "answer questions, keeping facts" not "memorise these N cards"."""
    wt_train, wt_eval = _wikitext_train_eval(tokenizer, max_train_tokens, max_eval_tokens)
    if source == "wikitext":
        return wt_train, wt_eval, torch.ones_like(wt_train, dtype=torch.bool)
    if answer_mask:
        instr, instr_msk = _instruction_ids_mask(tokenizer, exclude_texts=exclude_texts)
    else:
        instr = torch.tensor(tokenizer(_instruction_text())["input_ids"], dtype=torch.long)
        instr_msk = torch.ones_like(instr, dtype=torch.bool)
    if source == "instruction":
        return instr[:max_train_tokens], wt_eval, instr_msk[:max_train_tokens]
    if source == "mixed":
        fac = fac_msk = None
        if factual_blend_file and factual_blend_frac > 0:
            fac, fac_msk = _factual_ids_mask(tokenizer, factual_blend_file, exclude_texts=exclude_texts)
            n_fac = int(max_train_tokens * factual_blend_frac)
            if 0 < fac.numel() < n_fac:
                reps = (n_fac // fac.numel()) + 1
                print(f"  [FACT-003G blend] factual set is {fac.numel():,} tokens but the {100*factual_blend_frac:.0f}% "
                      f"blend budget is {n_fac:,} -> repeated ~{reps}x. SMALL SET => still memorisation risk; "
                      f"scale the factual data to make the blend diverse.", flush=True)
                fac = fac.repeat(reps)[:n_fac]
                fac_msk = fac_msk.repeat(reps)[:n_fac]
            else:
                fac, fac_msk = fac[:n_fac], fac_msk[:n_fac]
        n_fac = fac.numel() if fac is not None else 0
        rest = max_train_tokens - n_fac
        half = rest // 2
        wt_part = wt_train[:half]
        instr_part = instr[: rest - half]
        parts = ([fac] if fac is not None else []) + [instr_part, wt_part]
        mparts = ([fac_msk] if fac_msk is not None else []) + [instr_msk[: rest - half],
                                                               torch.ones_like(wt_part, dtype=torch.bool)]
        if n_fac:
            print(f"  [FACT-003G blend] mixed stream = {n_fac:,} factual ({100*factual_blend_frac:.0f}%) "
                  f"+ {instr_part.numel():,} instruction + {wt_part.numel():,} wikitext tokens", flush=True)
        return torch.cat(parts), wt_eval, torch.cat(mparts)
    raise ValueError(source)


@torch.no_grad()
def eval_ce(model, eval_ids, seq_len, device, max_windows=64):
    model.eval()
    n = min(eval_ids.numel() // seq_len, max_windows)
    ids = eval_ids[: n * seq_len].reshape(n, seq_len).to(device)
    tot, cnt = 0.0, 0
    for i in range(0, n, 8):
        b = ids[i : i + 8]
        loss = model(input_ids=b, labels=b).loss
        tot += float(loss) * b.shape[0]
        cnt += b.shape[0]
    return tot / max(cnt, 1)


def base_kl_replay_term(model, teacher, replay_ids, replay_mask, seq_len, replay_batch, temp, gen, device,
                        special_vocab=None):
    """FACT-003B/C: answer-masked forward-KL(base||student) on a sampled replay window.

    Returns a scalar loss tensor with grad through the student (teacher is frozen / no-grad),
    averaged over answer tokens only, scaled by temp^2 (standard KD). Returns None if the
    sampled window has no answer tokens.

    FACT-003C content-KL: if special_vocab (a [vocab] bool mask, True at EOS/BOS/pad/... ids)
    is given, those vocab columns are removed from BOTH distributions (logits -> -inf) so the
    KL anchors only the base model's CONTENT distribution, not its "when to stop" (EOS) mass.
    Naive base-KL (FACT-003B) copied the chat teacher's early-EOS -> empty collapse; this fixes
    that by never matching EOS/special probability."""
    import torch.nn.functional as F
    usable = replay_ids.numel() - 1
    starts = torch.randint(0, max(1, usable - seq_len), (replay_batch,), generator=gen).tolist()
    rx = torch.stack([replay_ids[s : s + seq_len] for s in starts]).to(device)
    rm = torch.stack([replay_mask[s : s + seq_len] for s in starts]).to(device)
    if not rm.any():
        return None
    with torch.no_grad():
        t = teacher(input_ids=rx).logits.float() / temp
        if special_vocab is not None:
            t = t.masked_fill(special_vocab, float("-inf"))   # drop EOS/special from teacher dist
        p_t = F.softmax(t, dim=-1)
        logp_t = F.log_softmax(t, dim=-1)
    s = model(input_ids=rx).logits.float() / temp
    if special_vocab is not None:
        s = s.masked_fill(special_vocab, float("-inf"))
    logp_s = F.log_softmax(s, dim=-1)
    term = p_t * (logp_t - logp_s)                    # [B,T,V] forward-KL summand
    if special_vocab is not None:
        term = term.masked_fill(special_vocab, 0.0)   # special cols are 0*(-inf)=nan -> set 0
    kl_tok = term.sum(-1)                              # [B,T] KL per token
    return kl_tok[rm].mean() * (temp * temp)          # answer tokens only, KD temp^2 scale


def dino_logit_term(model, teacher, train_ids, seq_len, dino_batch, view_mode, view_p,
                    temp, gen, device, special_vocab, vocab_size, pad_id,
                    center=None, center_m=0.9, do_center=False):
    """DINO-I2S-002/003: no-label self-distillation -- content-KL(teacher_clean || student_view) over
    ALL positions of UNLABELED text windows (no answer labels). DINO-DIAG-001 showed this raises the
    teacher's factual content mass in the student (gold-token logprob/rank up); hidden alignment is
    DISCARDED (it cancelled the gain). The student sees an augmented view (token dropout/noise), the
    frozen teacher sees the clean window; matching the teacher's content distribution on broad text
    is the retention pressure. special_vocab (EOS/special) is dropped from BOTH sides, so EOS
    decisions are never distilled.

    DINO-I2S-003 stabilisation: optional DINO centering (do_center) subtracts an EMA of the teacher's
    per-vocab logit mean from the teacher logits before softmax, to stop the student collapsing onto a
    few dominant tokens (the 1.1B salad failure mode). Returns (term, center) so the EMA persists."""
    import torch.nn.functional as F
    usable = train_ids.numel() - 1
    starts = torch.randint(0, max(1, usable - seq_len), (dino_batch,), generator=gen).tolist()
    uo = torch.stack([train_ids[s : s + seq_len] for s in starts]).to(device)   # teacher clean view
    if view_mode == "same" or view_p <= 0:
        uv = uo
    else:
        vmask = (torch.rand(uo.shape, generator=gen) < view_p).to(device)
        repl = (torch.full_like(uo, pad_id) if view_mode == "dropout"
                else torch.randint(0, vocab_size, uo.shape, generator=gen).to(device))
        uv = torch.where(vmask, repl, uo)                                        # student augmented view
    with torch.no_grad():
        t = teacher(input_ids=uo).logits.float() / temp
        if do_center:                                    # DINO centering (EMA of teacher logit mean)
            batch_c = t.mean(dim=(0, 1))                 # [V]
            center = batch_c if center is None else center_m * center + (1 - center_m) * batch_c
            t = t - center
        if special_vocab is not None:
            t = t.masked_fill(special_vocab, float("-inf"))
        p_t = F.softmax(t, dim=-1)
        logp_t = F.log_softmax(t, dim=-1)
    s = model(input_ids=uv).logits.float() / temp
    if special_vocab is not None:
        s = s.masked_fill(special_vocab, float("-inf"))
    logp_s = F.log_softmax(s, dim=-1)
    term = p_t * (logp_t - logp_s)                       # [B,T,V] forward-KL summand, all positions
    if special_vocab is not None:
        term = term.masked_fill(special_vocab, 0.0)
    return term.sum(-1).mean() * (temp * temp), center


@torch.no_grad()
def _dino_collapse_frac(model, tok, prompts, device):
    """DINO-I2S-003 early-collapse detector: generate on a few held-out prompts (read-only, no grad)
    and return the salad/empty/loop fraction. Used only to STOP a degenerating run early."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from fact004a_160m_smoke import generate, tag
    was_train, was_cache = model.training, model.config.use_cache
    model.eval(); model.config.use_cache = True
    bad = 0
    for p in prompts:
        if tag(generate(model, tok, p, 40, device)) in ("salad", "empty", "loop"):
            bad += 1
    model.config.use_cache = was_cache
    if was_train:
        model.train()
    return bad / max(len(prompts), 1)


@torch.no_grad()
def telemetry_probe(model, tok, probe_rows, device, max_new=40):
    """PYTHIA-LADDER P2: per-log-step collapse-signature probe on held-out panel rows (read-only).

    Returns the telemetry schema fields (docs/pythia_ladder_runbook.md S5): student gold-token
    rank/logp, last-answer-position logit entropy + top1 prob, generation degeneracy
    (salad/loop/empty), and mid/last hidden-state variance -- the quantities that move at a
    scale-dependent collapse onset, before/around visible salad."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from fact004a_160m_smoke import generate, tag
    import torch.nn.functional as F
    was_train, was_cache = model.training, model.config.use_cache
    model.eval()
    gr, gl, ents, top1s, hvm, hvl = [], [], [], [], [], []
    tags = {}
    for r in probe_rows:
        gold = r["must_contain"][0].strip()
        gs = " " + (gold[0].upper() + gold[1:] if gold else gold)
        pids = tok(r["prompt"], return_tensors="pt").input_ids.to(device)
        gids = tok(gs, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        model.config.use_cache = False
        out = model(input_ids=torch.cat([pids, gids], dim=1), output_hidden_states=True)
        logp = F.log_softmax(out.logits[0].float(), dim=-1)
        Lp = pids.shape[1]
        gl.append(sum(logp[Lp - 1 + i, g].item() for i, g in enumerate(gids[0].tolist())) / gids.shape[1])
        dist = logp[Lp - 1]
        first = int(gids[0, 0])
        gr.append(int((dist > dist[first]).sum().item()) + 1)
        pdist = dist.exp()
        ents.append(float(-(pdist * dist).sum()))
        top1s.append(float(pdist.max()))
        hs = out.hidden_states
        nl = len(hs)
        hvm.append(float(hs[max(1, nl // 2)][0].float().var()))
        hvl.append(float(hs[nl - 1][0].float().var()))
        model.config.use_cache = True
        t = tag(generate(model, tok, r["prompt"], max_new, device))
        tags[t] = tags.get(t, 0) + 1
    model.config.use_cache = was_cache
    if was_train:
        model.train()
    n = max(len(probe_rows), 1)
    deg = sum(tags.get(k, 0) for k in ("salad", "empty", "loop")) / n
    return {"gold_rank_mean": round(sum(gr) / n, 1), "gold_logp_mean": round(sum(gl) / n, 3),
            "logit_entropy": round(sum(ents) / n, 3), "top1_prob": round(sum(top1s) / n, 3),
            "degenerate_rate": round(deg, 3), "salad_rate": round(tags.get("salad", 0) / n, 3),
            "loop_rate": round(tags.get("loop", 0) / n, 3), "empty_rate": round(tags.get("empty", 0) / n, 3),
            "hidden_var_mid": round(sum(hvm) / n, 3), "hidden_var_last": round(sum(hvl) / n, 3)}


@torch.no_grad()
def fact_panel_eval(model, tok, panel_rows, device, max_new=40):
    """RFIT peak-hunting: FULL-panel factual eval (generate + exact hit + first-token rank) on the
    LIVE model (i2_s==f16 parity holds, so this matches the materialized score_fact_panel). Returns
    fact_rate + first_token_hit + gold_rank_mean over ALL panel rows -- the real FACT metric that the
    10-row telemetry gold_rank only loosely/noisily proxies. Called at --fact-eval-steps to find the
    step where FACT peaks (Qwen-1.5B over-trains: FACT can peak mid-run then fall)."""
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from fact004a_160m_smoke import generate, hit
    import torch.nn.functional as F
    was_train, was_cache = model.training, model.config.use_cache
    model.eval()
    hits, ft, ranks = 0, 0, []
    for r in panel_rows:
        model.config.use_cache = True
        txt = generate(model, tok, r["prompt"], max_new, device)
        hits += int(hit(txt, r["must_contain"]))
        gold = r["must_contain"][0].strip()
        gs = " " + (gold[0].upper() + gold[1:] if gold else gold)
        pids = tok(r["prompt"], return_tensors="pt").input_ids.to(device)
        gids = tok(gs, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        model.config.use_cache = False
        lp = F.log_softmax(model(torch.cat([pids, gids], dim=1)).logits[0].float(), dim=-1)[pids.shape[1] - 1]
        first = int(gids[0, 0]); rk = int((lp > lp[first]).sum().item()) + 1
        ranks.append(rk); ft += int(rk == 1)
        model.config.use_cache = True
    model.config.use_cache = was_cache
    if was_train:
        model.train()
    n = max(len(panel_rows), 1)
    return {"fact_rate": round(hits / n, 3), "first_token_hit": round(ft / n, 3),
            "fact_gold_rank_mean": round(sum(ranks) / n, 1), "fact_hits": hits, "fact_n": n}


def aamc_controller(hist, cur_lambda, cur_alpha, max_alpha):
    """AAMC Controller Policy V0 (docs/adaptive_anchor_manifold_controller_plan.md).
    hist: list of scored-interval dicts {train_ce, eval_ce, fact, gold_rank, entropy, top1,
    degen_gap, collapse_rate}. Compares the last two intervals -> overfit_score / collapse_score
    (counts of satisfied conditions, a PATTERN not one scalar), then maps to a lambda/alpha move.
    Returns (new_lambda, new_alpha, overfit_score, collapse_score, reason). Clamps: 0.2<=lambda<=0.5,
    0<=alpha<=max_alpha (max_alpha=0 disables DINO -> the dynamic-lambda-only arm)."""
    cur = hist[-1]
    prev = hist[-2] if len(hist) >= 2 else None
    collapse = (int(cur["degen_gap"] >= 0.3) + int(cur["collapse_rate"] >= 0.2))
    overfit = 0
    if prev is not None:
        overfit += int(cur["train_ce"] < prev["train_ce"] - 0.05)   # train stream being memorized
        overfit += int(cur["eval_ce"] >= prev["eval_ce"] - 0.02)    # held-out CE not improving
        overfit += int(cur["fact"] <= prev["fact"] + 1e-9)          # FACT flat/down
        overfit += int(cur["entropy"] < prev["entropy"] - 0.05)     # distribution sharpening
        overfit += int(cur["top1"] > prev["top1"] + 0.02)           # overconfident
    collapse_high = collapse >= 1
    overfit_high = overfit >= 3
    stall = (prev is not None and cur["train_ce"] > 6.0
             and abs(cur["train_ce"] - prev["train_ce"]) < 0.1 and cur["gold_rank"] >= prev["gold_rank"])
    new_lambda, new_alpha, reason = cur_lambda, cur_alpha, "keep"
    if collapse_high and max_alpha > 0:
        new_alpha = min(cur_alpha + 0.05, max_alpha); reason = "collapse->raise alpha"
    elif overfit_high and not collapse_high:
        new_lambda = min(cur_lambda + 0.1, 0.5); reason = "overfit->raise lambda"
    elif stall:
        new_lambda = max(cur_lambda - 0.1, 0.2); new_alpha = 0.0; reason = "stall->lower lambda"
    return round(new_lambda, 3), round(new_alpha, 3), overfit, collapse, reason


def homeostasis_term(model, teacher, input_ids, layer_mode="last", rho=1.0):
    """HOME-001: match base/student hidden-state mean+rms set-points on the same batch.

    This is a cheap biological-homeostasis smoke: the student may adapt, but its
    selected activation statistics should not drift too far from the base model.
    Intended for 160M first; it doubles forward compute for the sampled batch.
    """
    with torch.no_grad():
        t_h = teacher(input_ids=input_ids, output_hidden_states=True).hidden_states
    s_h = model(input_ids=input_ids, output_hidden_states=True).hidden_states
    n = len(s_h)
    if layer_mode == "last":
        idxs = [n - 1]
    elif layer_mode == "mid_last":
        idxs = [max(1, n // 2), n - 1]
    else:
        raise ValueError(layer_mode)

    terms = []
    for idx in idxs:
        sh = s_h[idx].float()
        th = t_h[idx].float()
        s_mean = sh.mean(dim=(0, 1))
        t_mean = th.mean(dim=(0, 1))
        s_rms = sh.pow(2).mean(dim=(0, 1)).sqrt()
        t_rms = th.pow(2).mean(dim=(0, 1)).sqrt()
        terms.append((s_mean - t_mean).pow(2).mean() + rho * (s_rms - t_rms).pow(2).mean())
    return torch.stack(terms).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="JackFram/llama-160m")
    ap.add_argument("--work", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--out-dir", type=Path, default=None, help="adapted HF dir (default <work>/<slug>_adapted)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="number of microbatches per optimizer step; effective batch = batch * grad_accum_steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-tokens", type=int, default=600_000)
    ap.add_argument("--max-eval-tokens", type=int, default=60_000)
    ap.add_argument("--ppl-eval-tokens", type=int, default=3_000,
                    help="tokens written to eval.txt for the GGUF llama-perplexity (QR-003); "
                         "keep small, CPU perplexity is ~per-token")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32",
                    help="model compute dtype (bfloat16 saves memory for 1B+ on a 16GB GPU)")
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                    help="adamw8bit (bitsandbytes) cuts optimizer-state memory ~4x for 1B+")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade compute for activation memory (needed for 1B+ on a 16GB GPU)")
    ap.add_argument("--train-norms", action="store_true",
                    help="QR-002b: also adapt RMSNorm weights (may absorb quantization drift)")
    ap.add_argument("--train-lm-head", action="store_true",
                    help="QR-002c: also adapt lm_head (output distribution retune)")
    ap.add_argument("--bitnet", type=Path, default=None, help="if set, run QR-003 export+perplexity")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--train-source", choices=["wikitext", "instruction", "mixed"], default="wikitext",
                    help="FACT-002: adaptation data (eval is always WikiText validation; factual eval is rt130)")
    ap.add_argument("--answer-loss-only", action="store_true",
                    help="FACT-003A: compute CE only on response tokens of instruction data "
                         "(prompt 'Q:..\\nA:' + separators masked to -100); WikiText content "
                         "tokens always count. Avoids overfitting the Q/A prompt formatting.")
    ap.add_argument("--answer-token-weight", type=float, default=0.0,
                    help="ANS-001: extra weight beta on the first-k tokens of each answer span "
                         "(answer-token-weighted CE) -- pulls the reachable gold token to the front / short "
                         "answer. Needs --answer-loss-only. 0 = uniform answer CE (existing behaviour).")
    ap.add_argument("--answer-token-first-k", type=int, default=1,
                    help="ANS-001: how many leading answer-span tokens get the extra weight (default 1)")
    ap.add_argument("--base-kl-replay", action="store_true",
                    help="FACT-003B: base-anchored adaptation. Add KL(base||student) on the answer "
                         "tokens of a fixed instruction replay set, with the SAME original FP model "
                         "as a frozen self-teacher, so b1.58 keeps the base model's answer behaviour.")
    ap.add_argument("--kl-weight", type=float, default=1.0, help="FACT-003B: weight lambda on the KL replay term")
    ap.add_argument("--kl-temp", type=float, default=1.0, help="FACT-003B: distillation temperature for the KL term")
    ap.add_argument("--replay-tokens", type=int, default=200_000,
                    help="FACT-003B: size of the fixed instruction replay pool the KL anchor samples from")
    ap.add_argument("--replay-batch", type=int, default=2,
                    help="FACT-003B: microbatch of replay windows per step for the KL term (keep small)")
    ap.add_argument("--kl-content-only", action="store_true",
                    help="FACT-003C: drop EOS/special tokens from the base-KL distribution so the "
                         "anchor copies the base model's CONTENT, not its 'when to stop' (EOS) mass. "
                         "Fixes the FACT-003B empty-collapse from a chat teacher's early-EOS.")
    ap.add_argument("--teacher-dtype", choices=["float16", "bfloat16", "float32"], default="float16",
                    help="FACT-003B: dtype of the frozen base teacher (float16 halves its memory)")
    ap.add_argument("--exclude-panel", action="store_true",
                    help="FACT-003B de-leak: drop instruction examples that overlap the factual "
                         "panel from the TRAINING stream too (the replay pool is always de-leaked). "
                         "Use for a clean held-out fact_rate. Needs --panel-file.")
    ap.add_argument("--panel-file", type=Path, default=REPO_ROOT / "data/factual_panel_v1.jsonl",
                    help="factual eval panel to de-leak against (FACT-003B)")
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="resumable-checkpoint dir; use a Drive path (e.g. /content/drive/MyDrive/..) "
                         "to survive a VM recycle. Saves model+optimizer+step every --ckpt-every-min.")
    ap.add_argument("--ckpt-every-min", type=float, default=60.0,
                    help="wall-clock minutes between checkpoints (0 disables)")
    ap.add_argument("--factual-replay", type=Path, default=None,
                    help="FACT-003D: jsonl of protected atomic facts ({prompt,answer}); adds mu*answer-CE")
    ap.add_argument("--factual-weight", type=float, default=1.0, help="FACT-003D: weight mu on the factual CE term")
    ap.add_argument("--factual-batch", type=int, default=4, help="FACT-003D: windows per step for the factual term")
    ap.add_argument("--factual-blend-file", type=Path, default=None,
                    help="FACT-003G: jsonl of factual QA BLENDED into the mixed stream (same answer-CE, not a separate loss)")
    ap.add_argument("--factual-blend-frac", type=float, default=0.0,
                    help="FACT-003G: fraction of mixed train tokens that are blended factual QA (e.g. 0.05)")
    ap.add_argument("--homeostasis-weight", type=float, default=0.0,
                    help="HOME-001: weight eta on activation mean/RMS homeostasis against the frozen base model")
    ap.add_argument("--homeostasis-layers", choices=["last", "mid_last"], default="last",
                    help="HOME-001: hidden-state layers whose activation stats are matched")
    ap.add_argument("--homeostasis-rho", type=float, default=1.0,
                    help="HOME-001: relative weight on RMS drift versus mean drift")
    ap.add_argument("--dino-logit-weight", type=float, default=0.0,
                    help="DINO-I2S-002: weight on no-label self-distillation content-KL over UNLABELED "
                         "views (teacher_clean||student_view). Needs --base-kl-replay (shares the frozen "
                         "teacher). hidden alignment is intentionally NOT included (DINO-DIAG-001).")
    ap.add_argument("--dino-view-mode", choices=["same", "dropout", "noise"], default="dropout",
                    help="DINO-I2S-002: student-view augmentation of the unlabeled window")
    ap.add_argument("--dino-view-p", type=float, default=0.1, help="DINO-I2S-002: fraction of view tokens corrupted")
    ap.add_argument("--dino-batch", type=int, default=2, help="DINO-I2S-002: unlabeled windows per step for the DINO term")
    ap.add_argument("--dino-center", action="store_true",
                    help="DINO-I2S-003: DINO centering (EMA of teacher logit mean subtracted) to prevent salad collapse")
    ap.add_argument("--dino-center-m", type=float, default=0.9, help="DINO-I2S-003: centering EMA momentum")
    ap.add_argument("--dino-warmup-steps", type=int, default=0,
                    help="DINO-I2S-003: linearly ramp the dino weight 0->target over this many steps (0=off)")
    ap.add_argument("--dino-collapse-check-every", type=int, default=0,
                    help="DINO-I2S-003: every N steps generate on a few panel prompts and abort if degenerate (0=off)")
    ap.add_argument("--dino-collapse-salad-thresh", type=float, default=0.5,
                    help="DINO-I2S-003: stop if salad/empty/loop fraction exceeds this on the collapse check")
    ap.add_argument("--dino-collapse-min-step", type=int, default=150,
                    help="DINO-I2S-003: do not run the collapse check before this step (let warmup settle)")
    ap.add_argument("--telemetry-full", action="store_true",
                    help="PYTHIA-LADDER P2: every log step, run a held-out probe and log the collapse-signature "
                         "schema (gold_rank/logp, logit_entropy, top1_prob, degenerate/salad/loop/empty rate, "
                         "hidden_var mid/last) + grad_norm into metrics.jsonl")
    ap.add_argument("--telemetry-probe-n", type=int, default=10, help="PYTHIA-LADDER P2: # panel prompts in the probe")
    ap.add_argument("--aamc", action="store_true", help="AAMC: telemetry-driven controller adjusts lambda "
                    "(kl-weight) and alpha (dino-logit-weight) live at --aamc-score-every intervals per Policy V0 "
                    "(overfit_score -> raise lambda; collapse_score -> raise alpha). docs/adaptive_anchor_manifold_controller_plan.md")
    ap.add_argument("--aamc-score-every", type=int, default=200, help="AAMC: controller score interval (steps)")
    ap.add_argument("--aamc-max-alpha", type=float, default=0.10, help="AAMC: alpha (DINO) clamp upper bound; "
                    "set 0.0 for the dynamic-lambda-only arm (DINO never turns on), 0.10 for conditional-DINO arm")
    ap.add_argument("--dino-start-step", type=int, default=0, help="RFIT-D: keep DINO fully OFF until this step, "
                    "then ramp over --dino-warmup-steps. Applies DINO as a LATE anti-overfit regularizer only.")
    ap.add_argument("--fact-eval-steps", default="", help="RFIT: comma-separated step numbers at which to run a "
                    "FULL-panel FACT eval (generate+hit, all panel rows) on the live model + materialize a per-step "
                    "ckpt <out-dir>_s<step> -- for peak-hunting when FACT may peak mid-run then over-train")
    ap.add_argument("--sidecar-rank", type=int, default=0,
                    help="SIDE: I2_S+LoRA sidecar rank per target linear (0 = disabled, existing behavior)")
    ap.add_argument("--sidecar-alpha", type=float, default=8.0, help="SIDE: LoRA scale = alpha/rank")
    ap.add_argument("--sidecar-target", choices=["all", "attn", "mlp", "top_saliency"], default="all")
    ap.add_argument("--sidecar-train-base", action="store_true",
                    help="SIDE: also adapt the ternary base (co-adapted, SIDE-002); default = frozen base, LoRA only")
    ap.add_argument("--sidecar-init", choices=["zero", "random", "svd_residual"], default="zero")
    ap.add_argument("--sidecar-top-layers", type=int, default=4, help="SIDE: blocks wrapped when --sidecar-target top_saliency")
    ap.add_argument("--sidecar-layers", default="", help="EGROW-002: comma list of exact module names to wrap (overrides --sidecar-target)")
    ap.add_argument("--metrics-out", type=Path, default=None,
                    help="append per-log-step metrics as jsonl (put on Drive to survive VM recycle)")
    ap.add_argument("--tb-logdir", type=Path, default=None,
                    help="TensorBoard logdir for scalar curves (put on Drive); skipped if tensorboard missing")
    ap.add_argument("--resume", action="store_true",
                    help="resume training from <ckpt-dir>/ckpt.pt if present (continues the step count)")
    args = ap.parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    slug = args.model_id.split("/")[-1]
    out_dir = args.out_dir or args.work / f"{slug}_adapted"
    print(f"device={device}  model={args.model_id}")

    tdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=tdtype).to(device)
    if args.grad_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # else ckpt warns: no input needs grad (embeds frozen)

    panel_exclude = (_panel_exclude_set(args.panel_file)
                     if (args.exclude_panel or args.base_kl_replay or args.factual_replay
                         or args.factual_blend_file) else None)
    train_ids, eval_ids, train_mask = load_corpus(
        args.train_source, tok, args.max_train_tokens, args.max_eval_tokens,
        answer_mask=args.answer_loss_only,
        exclude_texts=(panel_exclude if args.exclude_panel else None),
        factual_blend_file=args.factual_blend_file, factual_blend_frac=args.factual_blend_frac)
    print(f"train-source={args.train_source}"
          + ("  [train de-leaked vs panel]" if args.exclude_panel else ""))
    print(f"train tokens={train_ids.numel():,}  eval tokens={eval_ids.numel():,}")
    if args.answer_loss_only:
        print(f"FACT-003A answer-loss-only: {100*float(train_mask.float().mean()):.1f}% of train "
              f"tokens are answer/content (CE masked to these; prompt+sep -> -100)")

    # FACT-003B: frozen base teacher (self-anchor) + fixed instruction replay pool
    teacher = None
    home_teacher = None
    replay_ids = replay_mask = None
    special_vocab = None
    if args.base_kl_replay:
        ttd = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.teacher_dtype]
        teacher = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=ttd).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.config.use_cache = False
        r_ids, r_mask = _instruction_ids_mask(tok, exclude_texts=panel_exclude)  # anchor never reinforces eval
        replay_ids, replay_mask = r_ids[: args.replay_tokens], r_mask[: args.replay_tokens]
        if args.kl_content_only:
            ids = sorted(set(int(i) for i in (tok.all_special_ids or []) if i is not None))
            special_vocab = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=device)
            special_vocab[ids] = True
            print(f"FACT-003C content-KL: dropping {len(ids)} special/EOS vocab ids from the anchor {ids}")
        print(f"FACT-003B base-KL replay: teacher={args.model_id} ({args.teacher_dtype}, frozen); "
              f"replay pool {replay_ids.numel():,} instruction tokens "
              f"({100*float(replay_mask.float().mean()):.1f}% answer); "
              f"lambda={args.kl_weight} temp={args.kl_temp} replay_batch={args.replay_batch}"
              f"{' content-only' if args.kl_content_only else ''}")
    if args.homeostasis_weight > 0:
        if teacher is not None:
            home_teacher = teacher
        else:
            ttd = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.teacher_dtype]
            home_teacher = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=ttd).to(device).eval()
            for p in home_teacher.parameters():
                p.requires_grad_(False)
            home_teacher.config.use_cache = False
        print(f"HOME-001 activation homeostasis: eta={args.homeostasis_weight} "
              f"layers={args.homeostasis_layers} rho={args.homeostasis_rho} teacher={args.teacher_dtype}")
    dino_pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    if args.dino_logit_weight > 0:
        if teacher is None:
            raise SystemExit("--dino-logit-weight needs --base-kl-replay (the DINO term reuses the frozen teacher)")
        print(f"DINO-I2S-002 logit self-distill: weight={args.dino_logit_weight} "
              f"view={args.dino_view_mode}@{args.dino_view_p} dino_batch={args.dino_batch} "
              f"(unlabeled content-KL, EOS/special {'masked' if special_vocab is not None else 'NOT masked'}, no hidden)")

    # FACT-003D: protected factual replay -- direct answer-CE on a panel-disjoint atomic-facts set
    factual_ids = factual_mask = None
    if args.factual_replay:
        factual_ids, factual_mask = _factual_ids_mask(tok, args.factual_replay, exclude_texts=panel_exclude)
        factual_ids, factual_mask = factual_ids.to(device), factual_mask.to(device)
        print(f"FACT-003D protected factual replay: mu={args.factual_weight} "
              f"factual_batch={args.factual_batch} from {args.factual_replay}")

    # QR-001: FP baseline, then PTQ collapse
    ce_fp = eval_ce(model, eval_ids, args.seq_len, device)
    n_lin = replace_targets(model)
    if n_lin == 0:
        raise SystemExit(
            f"replace_targets matched 0 linears for {args.model_id} -- the target-linear matcher "
            f"(bitnet_llama.conversion.TARGET_SUFFIXES) does not cover this architecture, so NOTHING "
            f"would be quantised to b1.58. Add this model's linear names before running (PYTHIA-LADDER P1).")
    model.to(device=device, dtype=tdtype)  # new PerTensorBitLinear modules -> model dtype
    n_side = 0
    if args.sidecar_rank > 0:  # SIDE: wrap target ternary linears with a low-rank LoRA sidecar
        n_layers_cfg = getattr(model.config, "num_hidden_layers", 0)
        side_layers = [s.strip() for s in args.sidecar_layers.split(",") if s.strip()] or None
        n_side = wrap_targets_with_lora(model, args.sidecar_rank, args.sidecar_alpha,
                                        target=args.sidecar_target, init=args.sidecar_init,
                                        top_layers=args.sidecar_top_layers, n_layers=n_layers_cfg,
                                        layer_names=side_layers)
        model.to(device=device, dtype=tdtype)
        acct = sidecar_accounting(model)
        print(f"SIDE: rank={args.sidecar_rank} alpha={args.sidecar_alpha} target={args.sidecar_target} "
              f"train_base={args.sidecar_train_base} init={args.sidecar_init} -> wrapped {n_side} linears; "
              f"sidecar {acct['sidecar_params']:,} params / {acct['sidecar_bytes_fp16']:,}B fp16 "
              f"= {acct['sidecar_bytes_ratio_vs_target_i2s']*100:.2f}% of I2_S target bytes")
    ce_ptq = eval_ce(model, eval_ids, args.seq_len, device)  # B=0 init -> base behaviour, unchanged
    print(f"QR-001  CE_fp={ce_fp:.4f} (ppl {math.exp(ce_fp):.2f})  "
          f"CE_ptq={ce_ptq:.4f} (ppl {math.exp(ce_ptq):.2f})  [{n_lin} target linears]")

    # QR-002a/b/c: freeze all, then unfreeze target linears (+ optional norms / lm_head)
    for p in model.parameters():
        p.requires_grad_(False)
    if args.sidecar_rank > 0:
        # sidecar: LoRA A/B always trainable; ternary base trainable only if --sidecar-train-base;
        # any un-wrapped target base (target=attn/mlp) still adapts normally.
        side_mods = [m for m in model.modules() if isinstance(m, I2SLoRALinear)]
        wrapped_base_ids = {id(m.base) for m in side_mods}
        tparams = []
        for m in side_mods:
            tparams += [m.lora_A.weight, m.lora_B.weight]
            if args.sidecar_train_base:
                tparams.append(m.base.weight)
        tparams += [m.weight for m in model.modules()
                    if isinstance(m, PerTensorBitLinear) and id(m) not in wrapped_base_ids]
    else:
        tparams = [m.weight for m in model.modules() if isinstance(m, PerTensorBitLinear)]
    if args.train_norms:
        tparams += [p for n, p in model.named_parameters() if "norm" in n.lower() and p.dim() == 1]
    if args.train_lm_head:
        tparams += [p for n, p in model.named_parameters() if n.endswith("lm_head.weight")]
    seen, uniq = set(), []
    for p in tparams:
        if id(p) not in seen:
            seen.add(id(p)); p.requires_grad_(True); uniq.append(p)
    tparams = uniq
    arm = ("QR-002a(linears)" + ("+norms" if args.train_norms else "")
           + ("+lmhead" if args.train_lm_head else "") + ("+ansmask" if args.answer_loss_only else "")
           + (f"+anstok{args.answer_token_weight}k{args.answer_token_first_k}" if args.answer_token_weight > 0 else "")
           + (f"+basekl{args.kl_weight}" if args.base_kl_replay else "")
           + ("c" if (args.base_kl_replay and args.kl_content_only) else "")
           + (f"+factrep{args.factual_weight}" if args.factual_replay else "")
           + (f"+blend{args.factual_blend_frac}" if args.factual_blend_file else "")
           + (f"+home{args.homeostasis_weight}" if args.homeostasis_weight > 0 else "")
           + (f"+dino{args.dino_logit_weight}" if args.dino_logit_weight > 0 else "")
           + (f"+side{args.sidecar_rank}{args.sidecar_target[:3]}{'cob' if args.sidecar_train_base else 'frz'}"
              if args.sidecar_rank > 0 else ""))
    print(f"adapting {sum(p.numel() for p in tparams)/1e6:.1f}M params across {len(tparams)} tensors [{arm}]")
    print(f"microbatch={args.batch}  grad_accum={args.grad_accum_steps}  "
          f"effective_batch={args.batch * args.grad_accum_steps}")
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(tparams, lr=args.lr)
    else:
        opt = torch.optim.AdamW(tparams, lr=args.lr)
    g = torch.Generator().manual_seed(args.seed)
    usable = train_ids.numel() - 1

    # resumable checkpointing (survives VM recycle if --ckpt-dir is on Drive)
    import os, time
    ckpt_path = (args.ckpt_dir / "ckpt.pt") if args.ckpt_dir else None
    start_step = 0
    if args.resume and ckpt_path and ckpt_path.exists():
        # map to CPU: loading a multi-GB ckpt onto CUDA would pile the whole thing on the GPU on
        # top of model+teacher+optimizer and OOM the resume. load_state_dict then copies into the
        # already-on-GPU model; del ck frees the CPU copy.
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:
            print(f"  [resume] optimizer state not restored ({e}); continuing with fresh optimizer", flush=True)
        try:
            # RNG continuity is non-critical (only the window-sample order) -> fall back if it fails.
            g.set_state(ck["gen"].cpu().to(torch.uint8))
        except Exception as e:
            print(f"  [resume] RNG state not restored ({e}); continuing with fresh RNG", flush=True)
        start_step = int(ck["step"]) + 1
        del ck
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  [resume] loaded {ckpt_path} -> continue at step {start_step}/{args.steps}", flush=True)

    def save_ckpt(step):
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "gen": g.get_state(), "steps": args.steps}, tmp)
        os.replace(tmp, ckpt_path)
        print(f"  [checkpoint] step {step} -> {ckpt_path}", flush=True)

    # comprehensive logging: per-step scalars to a Drive jsonl + optional TensorBoard (both survive
    # recycle when pointed at Drive). Append mode so resumes keep the prior history.
    metrics_fh = None
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = open(args.metrics_out, "a", encoding="utf-8")
    tb = None
    if args.tb_logdir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            args.tb_logdir.mkdir(parents=True, exist_ok=True)
            tb = SummaryWriter(log_dir=str(args.tb_logdir))
        except Exception as e:
            print(f"  [tb] TensorBoard unavailable ({e}); skipping event logging", flush=True)

    def log_metrics(step, ce, kl, fce, home, elapsed, rate, eta, pct, extra=None):
        rec = {"step": step, "pct": round(pct, 2), "train_ce": round(ce, 4),
               "kl": round(kl, 4) if teacher is not None else None,
               "fce": round(fce, 4) if factual_ids is not None else None,
               "home": round(home, 4) if home_teacher is not None else None,
               "dino": round(extra.get("dino_loss"), 4) if (extra and "dino_loss" in extra) else None,
               "elapsed_min": round(elapsed / 60, 2), "eta_min": round(eta / 60, 2),
               "sec_per_step": round(rate, 2), "arm": arm}
        if extra:
            rec.update(extra)  # PYTHIA-LADDER P2 telemetry schema (gold_rank, entropy, degenerate_rate, grad_norm, ...)
        if metrics_fh:
            metrics_fh.write(json.dumps(rec) + "\n"); metrics_fh.flush()
        if tb:
            tb.add_scalar("train/ce", ce, step)
            if teacher is not None:
                tb.add_scalar("train/kl", kl, step)
            if factual_ids is not None:
                tb.add_scalar("train/fce", fce, step)
            if home_teacher is not None:
                tb.add_scalar("train/home", home, step)
            for k, v in (extra or {}).items():
                if isinstance(v, (int, float)):
                    tb.add_scalar(f"telemetry/{k}", v, step)
            tb.flush()

    # DINO-I2S-003 stabilisation state: centering EMA, collapse-check prompts, early-stop flag
    dino_center = None
    collapsed = False
    collapse_prompts = []
    if args.dino_logit_weight > 0 and args.dino_collapse_check_every > 0:
        try:
            collapse_prompts = [json.loads(l)["prompt"] for l in open(args.panel_file, encoding="utf-8") if l.strip()][:10]
            print(f"DINO-I2S-003 collapse detector: every {args.dino_collapse_check_every} steps from step "
                  f"{args.dino_collapse_min_step}, stop if salad/empty/loop > {args.dino_collapse_salad_thresh} "
                  f"on {len(collapse_prompts)} panel prompts (read-only)", flush=True)
        except Exception as e:
            print(f"  [collapse-check] could not load panel prompts ({e}); detector off", flush=True)
    if args.dino_logit_weight > 0 and (args.dino_center or args.dino_warmup_steps > 0):
        print(f"DINO-I2S-003 stabilisation: center={args.dino_center}(m={args.dino_center_m}) "
              f"warmup_steps={args.dino_warmup_steps}", flush=True)

    # PYTHIA-LADDER P2: full collapse-signature telemetry probe rows (held-out panel, read-only)
    probe_rows = []
    teacher_base = {}
    if args.telemetry_full:
        try:
            probe_rows = [json.loads(l) for l in open(args.panel_file, encoding="utf-8") if l.strip()][: args.telemetry_probe_n]
            print(f"PYTHIA-LADDER telemetry: probing {len(probe_rows)} panel rows every {args.log_every} steps "
                  f"(gold_rank/entropy/top1/degenerate/hidden_var + grad_norm)", flush=True)
            # teacher-relative baseline: probe the FROZEN FP teacher ONCE so collapse is judged
            # relative to the base model's own behaviour (Pythia is base, not chat -- absolute tags
            # are not comparable across model families; the teacher gap is).
            if teacher is not None:
                tprobe = telemetry_probe(teacher, tok, probe_rows, device)  # NOTE: not 'tb' -- that is the TB writer
                teacher_base = {"teacher_degen": tprobe["degenerate_rate"], "teacher_gold_rank": tprobe["gold_rank_mean"],
                                "teacher_gold_logp": tprobe["gold_logp_mean"], "teacher_top1": tprobe["top1_prob"]}
                print(f"  [teacher-baseline] degen={tprobe['degenerate_rate']:.2f} gold_rank={tprobe['gold_rank_mean']:.0f} "
                      f"top1={tprobe['top1_prob']:.3f} ent={tprobe['logit_entropy']:.2f} -- collapse judged RELATIVE to this", flush=True)
        except Exception as e:
            print(f"  [telemetry] could not load probe rows ({e}); telemetry-full off", flush=True)
            probe_rows = []

    # RFIT peak-hunting: full panel + the step set at which to FACT-eval (generate+hit on all rows)
    fact_eval_steps = set(int(s) for s in args.fact_eval_steps.split(",") if s.strip())
    fact_panel = []
    fact_eval_log = None
    if fact_eval_steps:
        fact_panel = [json.loads(l) for l in open(args.panel_file, encoding="utf-8") if l.strip()]
        fact_eval_log = (Path(str(args.metrics_out) + ".facteval.jsonl") if args.metrics_out
                         else (args.out_dir.parent / "fact_eval.jsonl" if args.out_dir else None))
        print(f"RFIT: full-panel FACT eval ({len(fact_panel)} rows) at steps {sorted(fact_eval_steps)}; "
              f"materializing per-step ckpts; log -> {fact_eval_log}", flush=True)

    # AAMC: live-adjustable lambda (teacher anchor) + alpha (DINO). When --aamc, the controller mutates
    # these at --aamc-score-every; otherwise they stay at the fixed CLI values.
    cur_lambda = args.kl_weight
    cur_alpha = args.dino_logit_weight if args.dino_logit_weight > 0 else 0.0
    aamc_hist = []
    aamc_log = (Path(str(args.metrics_out) + ".aamc.jsonl") if args.metrics_out
                else (args.out_dir.parent / "aamc.jsonl" if args.out_dir else None))
    if args.aamc:
        aamc_panel = [json.loads(l) for l in open(args.panel_file, encoding="utf-8") if l.strip()]
        aamc_probe = aamc_panel[: args.telemetry_probe_n]
        print(f"AAMC controller ON: score every {args.aamc_score_every} steps; lambda0={cur_lambda} "
              f"alpha0={cur_alpha} max_alpha={args.aamc_max_alpha}; log -> {aamc_log}", flush=True)

    model.train()
    t_start = time.time()
    last_ckpt = t_start
    for step in range(start_step, args.steps):
        opt.zero_grad(set_to_none=True)
        train_ce_sum = 0.0
        train_kl_sum = 0.0
        train_fce_sum = 0.0
        train_home_sum = 0.0
        train_dino_sum = 0.0
        for _ in range(args.grad_accum_steps):
            starts = torch.randint(0, max(1, usable - args.seq_len), (args.batch,), generator=g)
            sl = starts.tolist()
            x = torch.stack([train_ids[s : s + args.seq_len] for s in sl]).to(device)
            if args.answer_loss_only:
                m = torch.stack([train_mask[s : s + args.seq_len] for s in sl]).to(device)
                labels = x.clone()
                labels[~m] = -100
                if not (labels != -100).any():
                    continue  # window(s) landed entirely on prompt/sep tokens; skip (rare)
                if args.answer_token_weight > 0:  # ANS-001: up-weight the first-k answer-span tokens
                    loss = answer_token_weighted_ce(model, x, m, args.answer_token_weight, args.answer_token_first_k)
                else:
                    loss = model(input_ids=x, labels=labels).loss
            else:
                loss = model(input_ids=x, labels=x).loss
            train_ce_sum += float(loss)
            if teacher is not None:  # FACT-003B: add base-KL replay anchor
                kl = base_kl_replay_term(model, teacher, replay_ids, replay_mask, args.seq_len,
                                         args.replay_batch, args.kl_temp, g, device,
                                         special_vocab=special_vocab)
                if kl is not None:
                    train_kl_sum += float(kl)
                    loss = loss + cur_lambda * kl
            if factual_ids is not None:  # FACT-003D: protected factual replay (direct answer-CE)
                fce = factual_ce_term(model, factual_ids, factual_mask, args.seq_len,
                                      args.factual_batch, g, device)
                if fce is not None:
                    train_fce_sum += float(fce)
                    loss = loss + args.factual_weight * fce
            if home_teacher is not None:
                home = homeostasis_term(model, home_teacher, x, args.homeostasis_layers, args.homeostasis_rho)
                train_home_sum += float(home)
                loss = loss + args.homeostasis_weight * home
            # DINO consistency: AAMC drives it via cur_alpha (controller-set); else RFIT-D fixed-weight
            # path (fully OFF until --dino-start-step, then ramps over --dino-warmup-steps).
            if args.aamc:
                use_dino = teacher is not None and cur_alpha > 0
                eff_dino_w = cur_alpha
            else:
                use_dino = (args.dino_logit_weight > 0 and teacher is not None and (step + 1) >= args.dino_start_step)
                s_since = step + 1 - args.dino_start_step
                eff_dino_w = (args.dino_logit_weight
                              * (min(1.0, (s_since + 1) / args.dino_warmup_steps) if args.dino_warmup_steps > 0 else 1.0))
            if use_dino:
                dino, dino_center = dino_logit_term(
                    model, teacher, train_ids, args.seq_len, args.dino_batch,
                    args.dino_view_mode, args.dino_view_p, args.kl_temp, g, device,
                    special_vocab, model.config.vocab_size, dino_pad_id,
                    center=dino_center, center_m=args.dino_center_m, do_center=args.dino_center)
                train_dino_sum += float(dino)
                loss = loss + eff_dino_w * dino
            (loss / args.grad_accum_steps).backward()
        is_log = step % args.log_every == 0 or step == args.steps - 1
        grad_norm = None
        if args.telemetry_full and is_log:  # PYTHIA-LADDER P2: measure grad norm (max_norm=inf -> no clipping)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(tparams, float("inf")))
        opt.step()
        if is_log:
            done = step - start_step + 1                 # steps done THIS session (for rate/ETA)
            elapsed = time.time() - t_start
            rate = elapsed / done                        # sec per optimizer step
            eta = rate * (args.steps - 1 - step)
            pct = 100.0 * (step + 1) / args.steps         # absolute progress
            extra = {}
            if args.dino_logit_weight > 0:
                extra["dino_loss"] = train_dino_sum / args.grad_accum_steps
            if grad_norm is not None:
                extra["grad_norm"] = round(grad_norm, 4)
            if probe_rows:                               # collapse-signature probe (read-only)
                extra.update(telemetry_probe(model, tok, probe_rows, device))
                extra.update(teacher_base)               # constant FP-teacher baseline for reference
                if teacher_base:                         # teacher-RELATIVE collapse signals (the real criterion)
                    extra["degen_gap"] = round(extra["degenerate_rate"] - teacher_base["teacher_degen"], 3)
                    extra["gold_rank_ratio"] = round(extra["gold_rank_mean"] / max(teacher_base["teacher_gold_rank"], 1.0), 2)
            kl_str = f"  kl={train_kl_sum / args.grad_accum_steps:.4f}" if teacher is not None else ""
            fce_str = f"  fce={train_fce_sum / args.grad_accum_steps:.4f}" if factual_ids is not None else ""
            home_str = f"  home={train_home_sum / args.grad_accum_steps:.4f}" if home_teacher is not None else ""
            dino_str = f"  dino={train_dino_sum / args.grad_accum_steps:.4f}" if args.dino_logit_weight > 0 else ""
            tele_str = (f"  gnorm={extra['grad_norm']:.2f}  degen={extra.get('degenerate_rate', 0):.2f}"
                        f"(gap{extra.get('degen_gap', 0):+.2f})  goldrank={extra.get('gold_rank_mean', 0):.0f}"
                        f"(x{extra.get('gold_rank_ratio', 0):.0f}T)  ent={extra.get('logit_entropy', 0):.2f}"
                        if grad_norm is not None else "")
            print(f"  step {step:4d}/{args.steps} ({pct:5.1f}%)  train_ce={train_ce_sum / args.grad_accum_steps:.4f}{kl_str}{fce_str}{home_str}{dino_str}{tele_str}  "
                  f"elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m  ({rate:.1f}s/step)", flush=True)
            log_metrics(step, train_ce_sum / args.grad_accum_steps, train_kl_sum / args.grad_accum_steps,
                        train_fce_sum / args.grad_accum_steps, train_home_sum / args.grad_accum_steps,
                        elapsed, rate, eta, pct, extra=extra)
        if args.aamc and (step + 1) % args.aamc_score_every == 0:
            ce_now = eval_ce(model, eval_ids, args.seq_len, device, max_windows=32)
            tp = telemetry_probe(model, tok, aamc_probe, device)
            fp = fact_panel_eval(model, tok, aamc_panel, device)
            dg = round(tp["degenerate_rate"] - teacher_base.get("teacher_degen", 0.0), 3) if teacher_base else tp["degenerate_rate"]
            ent = {"step": step + 1, "train_ce": round(train_ce_sum / args.grad_accum_steps, 4), "eval_ce": round(ce_now, 4),
                   "fact": fp["fact_rate"], "first_token_hit": fp["first_token_hit"], "gold_rank": fp["fact_gold_rank_mean"],
                   "entropy": tp["logit_entropy"], "top1": tp["top1_prob"], "degen_gap": dg,
                   "collapse_rate": round(tp["salad_rate"] + tp["loop_rate"] + tp["empty_rate"], 3)}
            aamc_hist.append(ent)
            new_l, new_a, ofs, cs, reason = aamc_controller(aamc_hist, cur_lambda, cur_alpha, args.aamc_max_alpha)
            ent.update({"lambda": cur_lambda, "alpha": cur_alpha, "new_lambda": new_l, "new_alpha": new_a,
                        "overfit_score": ofs, "collapse_score": cs, "action": reason})
            print(f"  [AAMC] step {step+1}: train_ce={ent['train_ce']} eval_ce={ent['eval_ce']} FACT={ent['fact']} "
                  f"fth={ent['first_token_hit']} ent={ent['entropy']} degen_gap={dg} | overfit={ofs} collapse={cs} "
                  f"-> {reason}: lambda {cur_lambda}->{new_l} alpha {cur_alpha}->{new_a}", flush=True)
            if aamc_log:
                with open(aamc_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ent) + "\n")
            cur_lambda, cur_alpha = new_l, new_a
        if fact_eval_steps and (step + 1) in fact_eval_steps:
            fe = fact_panel_eval(model, tok, fact_panel, device)
            fe["step"] = step + 1
            print(f"  [FACT-eval] step {step+1}: FACT {fe['fact_hits']}/{fe['fact_n']}={fe['fact_rate']}  "
                  f"first_token_hit={fe['first_token_hit']}  gold_rank={fe['fact_gold_rank_mean']}", flush=True)
            if fact_eval_log:
                with open(fact_eval_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(fe) + "\n")
            if args.out_dir is not None:
                sd = Path(str(args.out_dir) + f"_s{step+1}")
                materialize_and_save(model, sd, tok)
                print(f"  [FACT-eval] materialized peak-candidate -> {sd}", flush=True)
        if (collapse_prompts and step >= args.dino_collapse_min_step
                and step % args.dino_collapse_check_every == 0):
            sf = _dino_collapse_frac(model, tok, collapse_prompts, device)
            print(f"  [collapse-check] step {step} salad/empty/loop frac={sf:.2f}", flush=True)
            if sf > args.dino_collapse_salad_thresh:
                print(f"  [COLLAPSE DETECTED] {sf:.2f} > {args.dino_collapse_salad_thresh} -> stopping early "
                      f"at step {step} (DINO-I2S-003 stabilisation FAILED)", flush=True)
                collapsed = True
                if ckpt_path:
                    save_ckpt(step)
                break
        if ckpt_path and args.ckpt_every_min > 0 and (time.time() - last_ckpt) >= args.ckpt_every_min * 60:
            save_ckpt(step)
            last_ckpt = time.time()
    if metrics_fh:
        metrics_fh.close()
    if tb:
        tb.close()
    shared_home_teacher = home_teacher is teacher
    if teacher is not None:  # free the frozen base teacher before export
        del teacher
        if shared_home_teacher:
            home_teacher = None
        if device == "cuda":
            torch.cuda.empty_cache()
    if home_teacher is not None:
        del home_teacher
        if device == "cuda":
            torch.cuda.empty_cache()
    ce_adapted = eval_ce(model, eval_ids, args.seq_len, device)
    rec = (ce_ptq - ce_adapted) / max(ce_ptq - ce_fp, 1e-9)
    print(f"QR-002a CE_adapted={ce_adapted:.4f} (ppl {math.exp(ce_adapted):.2f})  "
          f"recovered_fraction={rec:.3f}")

    # save adapted materialized model for QR-003. NOTE: keep eval.txt SHORT — the
    # GGUF llama-perplexity processes every token and 60k tokens is ~13 min/format
    # on a 2-core CPU; a few thousand tokens is plenty for the f16-vs-i2_s parity.
    materialize_and_save(model, out_dir, tok)
    (out_dir / "eval.txt").write_text(tok.decode(eval_ids[: args.ppl_eval_tokens].tolist()),
                                      encoding="utf-8")
    print(f"saved adapted ternary HF dir -> {out_dir}")

    result = {"model": args.model_id, "device": device, "n_target_linears": n_lin,
              "steps": args.steps, "seq_len": args.seq_len, "batch": args.batch, "lr": args.lr,
              "grad_accum_steps": args.grad_accum_steps,
              "effective_batch": args.batch * args.grad_accum_steps,
              "effective_tokens_per_step": args.batch * args.grad_accum_steps * args.seq_len,
              "arm": arm, "train_source": args.train_source, "train_norms": args.train_norms, "train_lm_head": args.train_lm_head,
              "answer_loss_only": args.answer_loss_only,
              "answer_token_weight": args.answer_token_weight, "answer_token_first_k": args.answer_token_first_k,
              "base_kl_replay": args.base_kl_replay, "kl_weight": args.kl_weight, "kl_temp": args.kl_temp,
              "kl_content_only": args.kl_content_only,
              "replay_tokens": args.replay_tokens, "replay_batch": args.replay_batch,
              "factual_replay": str(args.factual_replay) if args.factual_replay else None,
              "factual_weight": args.factual_weight, "factual_batch": args.factual_batch,
              "factual_blend_file": str(args.factual_blend_file) if args.factual_blend_file else None,
              "factual_blend_frac": args.factual_blend_frac,
              "homeostasis_weight": args.homeostasis_weight,
              "homeostasis_layers": args.homeostasis_layers,
              "homeostasis_rho": args.homeostasis_rho,
              "dino_logit_weight": args.dino_logit_weight, "dino_view_mode": args.dino_view_mode,
              "dino_view_p": args.dino_view_p, "dino_batch": args.dino_batch,
              "dino_center": args.dino_center, "dino_center_m": args.dino_center_m,
              "dino_warmup_steps": args.dino_warmup_steps, "dino_start_step": args.dino_start_step,
              "aamc": args.aamc, "aamc_final_lambda": cur_lambda, "aamc_final_alpha": cur_alpha,
              "dino_collapsed_early": collapsed,
              **teacher_base,
              "sidecar_enabled": args.sidecar_rank > 0, "sidecar_rank": args.sidecar_rank,
              "sidecar_alpha": args.sidecar_alpha, "sidecar_target": args.sidecar_target,
              "sidecar_train_base": args.sidecar_train_base, "sidecar_linears": n_side,
              **({f"sidecar_{k}": v for k, v in sidecar_accounting(model).items()} if args.sidecar_rank > 0 else {}),
              "trainable_params": int(sum(p.numel() for p in tparams)),
              "ce_fp": ce_fp, "ce_ptq": ce_ptq, "ce_adapted": ce_adapted,
              "ppl_fp": math.exp(ce_fp), "ppl_ptq": math.exp(ce_ptq), "ppl_adapted": math.exp(ce_adapted),
              "recovered_fraction": rec}

    # QR-003: export adapted -> f16 & i2_s, perplexity parity (needs bitnet.cpp)
    if args.bitnet:
        bn = args.bitnet.resolve()
        conv = bn / "utils/convert-hf-to-gguf-bitnet.py"
        q = bn / "build/bin/llama-quantize"
        ppx = bn / "build/bin/llama-perplexity"
        f32, f16 = out_dir / "ggml-model-f32.gguf", out_dir / "ggml-model-f16.gguf"
        i2s = out_dir / "ggml-model-i2_s.gguf"

        import time

        def sh(c, label=""):
            print(f">>> [QR-003] {label} START ...", flush=True)
            t0 = time.time()
            r = subprocess.run(c, shell=True, capture_output=True, text=True)
            print(f">>> [QR-003] {label} done rc{r.returncode} in {time.time()-t0:.0f}s", flush=True)
            print((r.stdout + r.stderr)[-300:], flush=True)
            return r

        sh(f'python "{conv}" "{out_dir}" --outtype f32', "convert f32 (writes a multi-GB GGUF)")
        sh(f'python "{conv}" "{out_dir}" --outtype f16', "convert f16")
        sh(f'"{q}" --token-embedding-type f16 --output-tensor-type f16 "{f32}" "{i2s}" I2_S 1 1', "quantize i2_s")
        import re
        pr = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")
        qr3 = {}
        for tag, g_ in [("f16", f16), ("i2_s", i2s)]:
            print(f">>> [QR-003] perplexity {tag} START (CPU, ~per-token) ...", flush=True)
            t0 = time.time()
            out = subprocess.run(f'"{ppx}" -m "{g_}" -f "{out_dir}/eval.txt" -c 64 -t 2',
                                 shell=True, capture_output=True, text=True)
            m = pr.search(out.stdout + out.stderr)
            qr3[tag] = float(m.group(1)) if m else None
            print(f"QR-003 adapted {tag} PPL = {qr3[tag]}  ({time.time()-t0:.0f}s)", flush=True)
        if qr3.get("f16") and qr3.get("i2_s"):
            d = math.log(qr3["i2_s"]) - math.log(qr3["f16"])
            print(f"QR-003 adapted i2_s vs f16 = {d:+.4f} nats")
            qr3["i2s_vs_f16_nats"] = d
        result["qr003"] = qr3

    print("\n" + "=" * 56)
    print(f"RT-116 / TRAIN-001  {args.model_id}")
    print("=" * 56)
    print(f"CE  fp={ce_fp:.4f}  ptq={ce_ptq:.4f}  adapted={ce_adapted:.4f}")
    print(f"PPL fp={math.exp(ce_fp):.1f}  ptq={math.exp(ce_ptq):.1f}  adapted={math.exp(ce_adapted):.1f}")
    print(f"recovered_fraction = (ptq-adapted)/(ptq-fp) = {rec:.3f}")
    if rec > 0.3:
        print("QR-002 VERDICT: PASS -- short teacher-free CE recovers a meaningful fraction.")
    elif rec > 0.05:
        print("QR-002 VERDICT: WEAK -- some recovery; try +norms/+lm_head, more steps, or LR.")
    else:
        print("QR-002 VERDICT: FAIL -- no recovery; revisit recipe (norms/lm_head/LR/corpus).")

    jo = args.json_out or args.work / f"rt116_{slug}_recovery.json"
    jo.parent.mkdir(parents=True, exist_ok=True)
    jo.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {jo}")


if __name__ == "__main__":
    main()
