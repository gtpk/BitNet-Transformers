#!/usr/bin/env python3
"""Build FACT-003E length-balanced factual replay data from the protected atomic facts.

FACT-003D deliberately used short atomic QA because it was a clean diagnostic. That is
not representative of PTQ/QAT calibration/adaptation data, where prompt and answer
lengths should cover the range the model will see at inference time.

This script keeps the same protected train/held-out split and the same canonical fact
answers, but expands each fact into several prompt/answer formats:

  short       Q: ... A: <short answer>
  sentence    complete sentence answer
  chat        User/Assistant format
  explain     short answer plus one generic explanation sentence
  long        longer instruction/context prompt

The held-out output remains panel-style with must_contain so scoring checks the same
canonical answer regardless of answer wording.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_answer(rec: dict) -> str:
    if "answer" in rec:
        return str(rec["answer"]).strip()
    must = rec.get("must_contain") or []
    if isinstance(must, str):
        return must.strip()
    return str(must[0]).strip()


def canonical_contains(rec: dict) -> list[str]:
    if "must_contain" in rec:
        must = rec["must_contain"]
        if isinstance(must, str):
            return [must.strip().lower()]
        return [str(m).strip().lower() for m in must if str(m).strip()]
    return [clean_answer(rec).lower()]


def question_from_prompt(prompt: str) -> str:
    text = prompt.strip()
    text = re.sub(r"^Q:\s*", "", text)
    text = re.sub(r"\s*A:\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def entity_value(entity: str) -> str:
    return entity.split(":", 1)[1] if ":" in entity else entity


def sentence_answer(rec: dict, answer: str) -> str:
    category = rec.get("category", "")
    entity = entity_value(rec.get("entity", rec.get("id", "the item")))

    if category == "capital":
        return f"The capital of {entity} is {answer}."
    if category == "currency":
        return f"The currency of {entity} is {answer}."
    if category == "element":
        return f"The chemical symbol for {entity} is {answer}."
    if category == "author":
        return f"{entity} was written by {answer}."
    if category == "continent":
        return f"{entity} is located in {answer}."
    if category == "language":
        return f"A main language spoken in {entity} is {answer}."
    return f"The answer is {answer}."


def make_variants(rec: dict, styles: list[str], split: str) -> list[dict]:
    source_id = rec.get("id") or rec.get("entity")
    question = question_from_prompt(rec["prompt"])
    answer = clean_answer(rec)
    contains = canonical_contains(rec)
    sent = sentence_answer(rec, answer)

    rows = []
    for style in styles:
        if style == "short":
            prompt = rec["prompt"]
            out_answer = " " + answer
            bucket = "short_prompt_short_answer"
        elif style == "sentence":
            prompt = f"Q: {question} Answer in one complete sentence.\nA:"
            out_answer = " " + sent
            bucket = "short_prompt_sentence_answer"
        elif style == "chat":
            prompt = f"User: {question}\nAssistant:"
            out_answer = " " + sent
            bucket = "chat_prompt_sentence_answer"
        elif style == "explain":
            prompt = f"Q: {question} Give the answer and one brief explanation sentence.\nA:"
            out_answer = " " + sent + " This is the requested fact."
            bucket = "short_prompt_explanation_answer"
        elif style == "long":
            prompt = (
                "Context: This is a factual question from a mixed-length calibration set. "
                "Some examples are short and some include extra instructions. Focus only on "
                "the requested fact and answer naturally.\n"
                f"Question: {question}\nAnswer:"
            )
            out_answer = " " + sent
            bucket = "long_prompt_sentence_answer"
        else:
            raise ValueError(f"unknown style: {style}")

        row = {
            "id": f"{source_id}:{style}",
            "source_id": source_id,
            "category": rec.get("category"),
            "entity": rec.get("entity", rec.get("id")),
            "split": split,
            "style": style,
            "length_bucket": bucket,
            "prompt": prompt,
            "must_contain": contains,
        }
        if split == "train":
            row["answer"] = out_answer
        rows.append(row)
    return rows


def length_stats(rows: list[dict]) -> dict:
    if not rows:
        return {}
    prompt_chars = [len(r["prompt"]) for r in rows]
    prompt_words = [len(re.findall(r"\S+", r["prompt"])) for r in rows]
    answer_chars = [len(r.get("answer", " ".join(r.get("must_contain", [])))) for r in rows]
    answer_words = [len(re.findall(r"\S+", r.get("answer", " ".join(r.get("must_contain", []))))) for r in rows]
    return {
        "rows": len(rows),
        "prompt_chars_min": min(prompt_chars),
        "prompt_chars_mean": round(mean(prompt_chars), 1),
        "prompt_chars_max": max(prompt_chars),
        "prompt_words_min": min(prompt_words),
        "prompt_words_mean": round(mean(prompt_words), 1),
        "prompt_words_max": max(prompt_words),
        "answer_words_min": min(answer_words),
        "answer_words_mean": round(mean(answer_words), 1),
        "answer_words_max": max(answer_words),
        "answer_chars_min": min(answer_chars),
        "answer_chars_mean": round(mean(answer_chars), 1),
        "answer_chars_max": max(answer_chars),
    }


def counts_by(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-source", type=Path, default=REPO_ROOT / "data/atomic_facts_train.jsonl")
    ap.add_argument("--heldout-source", type=Path, default=REPO_ROOT / "data/atomic_facts_heldout.jsonl")
    ap.add_argument("--out-train", type=Path, default=REPO_ROOT / "data/atomic_facts_lengthmix_train.jsonl")
    ap.add_argument("--out-heldout", type=Path, default=REPO_ROOT / "data/atomic_facts_lengthmix_heldout.jsonl")
    ap.add_argument("--report-json", type=Path, default=REPO_ROOT / "reports/fact003e_lengthmix_dataset_summary.json")
    ap.add_argument("--report-md", type=Path, default=REPO_ROOT / "reports/fact003e_lengthmix_dataset_summary.md")
    ap.add_argument("--styles", default="short,sentence,chat,explain,long")
    args = ap.parse_args()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    train_src = load_jsonl(args.train_source)
    heldout_src = load_jsonl(args.heldout_source)
    train_rows = [row for rec in train_src for row in make_variants(rec, styles, "train")]
    heldout_rows = [row for rec in heldout_src for row in make_variants(rec, styles, "heldout")]

    write_jsonl(args.out_train, train_rows)
    write_jsonl(args.out_heldout, heldout_rows)

    summary = {
        "styles": styles,
        "source_train_rows": len(train_src),
        "source_heldout_rows": len(heldout_src),
        "train_rows": len(train_rows),
        "heldout_rows": len(heldout_rows),
        "train_by_style": counts_by(train_rows, "style"),
        "heldout_by_style": counts_by(heldout_rows, "style"),
        "train_by_category": counts_by(train_rows, "category"),
        "heldout_by_category": counts_by(heldout_rows, "category"),
        "train_length_stats": length_stats(train_rows),
        "heldout_length_stats": length_stats(heldout_rows),
        "out_train": str(args.out_train),
        "out_heldout": str(args.out_heldout),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# FACT-003E Length-Mixed Factual Replay Dataset",
        "",
        f"styles: `{','.join(styles)}`",
        "",
        "| split | source facts | expanded rows | prompt words min/mean/max | answer words min/mean/max |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| train | {len(train_src)} | {len(train_rows)} | "
        f"{summary['train_length_stats']['prompt_words_min']}/"
        f"{summary['train_length_stats']['prompt_words_mean']}/"
        f"{summary['train_length_stats']['prompt_words_max']} | "
        f"{summary['train_length_stats']['answer_words_min']}/"
        f"{summary['train_length_stats']['answer_words_mean']}/"
        f"{summary['train_length_stats']['answer_words_max']} |",
        f"| heldout | {len(heldout_src)} | {len(heldout_rows)} | "
        f"{summary['heldout_length_stats']['prompt_words_min']}/"
        f"{summary['heldout_length_stats']['prompt_words_mean']}/"
        f"{summary['heldout_length_stats']['prompt_words_max']} | "
        f"{summary['heldout_length_stats']['answer_words_min']}/"
        f"{summary['heldout_length_stats']['answer_words_mean']}/"
        f"{summary['heldout_length_stats']['answer_words_max']} |",
        "",
        "This dataset preserves the protected train/held-out entity split from FACT-003D,",
        "but replaces the single short QA surface form with a mixed-length adaptation set.",
        "",
        f"- train: `{args.out_train}`",
        f"- heldout: `{args.out_heldout}`",
    ]
    args.report_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.out_train}")
    print(f"wrote {args.out_heldout}")
    print(f"wrote {args.report_md}")


if __name__ == "__main__":
    main()
