#!/usr/bin/env python3
"""RDT-001: tiny rate-distortion / cost ledger for I2_S smoke branches.

This is deliberately conservative. It reads the existing small-screen reports
that already contain structured markdown tables and emits a single JSON/MD
ledger that asks:

    how much behavior did each extra byte buy?

It is not a model runner and it should never be used to claim quality by itself.
It is a branch-prioritization tool before spending Colab time.

Usage:
  python scripts/rdt001_cost_ledger.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_md_tables(path: Path) -> list[dict[str, str]]:
    """Parse all simple GitHub-style markdown tables in a file."""
    rows: list[dict[str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith("|"):
            i += 1
            continue
        if i + 1 >= len(lines) or not lines[i + 1].startswith("| ---"):
            i += 1
            continue
        headers = [h.strip() for h in line.strip("|").split("|")]
        j = i + 2
        while j < len(lines) and lines[j].startswith("|"):
            if set(lines[j].replace("|", "").strip()) <= {"-", ":", " "}:
                j += 1
                continue
            cells = [c.strip() for c in lines[j].strip("|").split("|")]
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
            j += 1
        i = j
    return rows


def num(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.replace(",", "").strip()
    value = re.sub(r"\*\*", "", value)
    if value in {"", "—", "-", "None"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def side001_rows() -> list[dict[str, object]]:
    path = REPO_ROOT / "reports/side001_160m.md"
    out = []
    for row in parse_md_tables(path):
        if "rank" not in row or "eval_panel" not in row:
            continue
        rank = int(num(row["rank"]) or 0)
        bytes_fp16 = {0: 0, 2: 847_872, 4: 1_695_744, 8: 3_391_488}.get(rank, 0)
        out.append({
            "branch": "SIDE-001",
            "arm": f"rank{rank}",
            "eval_panel": num(row.get("eval_panel")),
            "popqa_tight": num(row.get("popqa_tight (PRIMARY)")),
            "ce": num(row.get("CE")),
            "extra_bytes": bytes_fp16,
            "is_baseline": rank == 0,
            "notes": row.get("tags", ""),
        })
    return out


def egrow002_rows() -> list[dict[str, object]]:
    path = REPO_ROOT / "reports/egrow002_160m.md"
    out = []
    for row in parse_md_tables(path):
        if "arm" not in row or "eval_panel" not in row:
            continue
        out.append({
            "branch": "EGROW-002",
            "arm": row["arm"],
            "eval_panel": num(row.get("eval_panel")),
            "popqa_tight": num(row.get("popqa_tight")),
            "ce": num(row.get("CE")),
            "extra_bytes": int(num(row.get("sidecar bytes")) or 0),
            "is_baseline": row["arm"].lower() == "none",
            "notes": row.get("tags", ""),
        })
    return out


def wsync_rows() -> list[dict[str, object]]:
    out = []
    for fname, branch in [
        ("rt_wsync_160m.md", "WSYNC-scaling"),
        ("rt_wsync_160m_hi2s.md", "WSYNC-HI2S"),
    ]:
        path = REPO_ROOT / "reports" / fname
        for row in parse_md_tables(path):
            arm = row.get("arm")
            if not arm or "fact_rate" not in row:
                continue
            out.append({
                "branch": branch,
                "arm": re.sub(r"\*\*", "", arm),
                "eval_panel": num(row.get("fact_rate")),
                "popqa_tight": None,
                "ce": num(row.get("CE")),
                "extra_bytes": 0,
                "is_baseline": "per-tensor" in arm.lower() or arm.strip().lower() == "pt",
                "is_reference": arm.strip().lower() == "fp",
                "notes": "data-free",
            })
    return out


def enrich(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Add deltas relative to the no-extra-capacity baseline inside each branch."""
    by_branch: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_branch.setdefault(str(row["branch"]), []).append(row)

    enriched = []
    for branch, group in by_branch.items():
        baseline = None
        for row in group:
            arm = str(row["arm"]).lower()
            if row.get("is_baseline") or arm in {"rank0", "none", "pt (per-tensor b1.58)", "pt"}:
                baseline = row
                break
        if baseline is None:
            baseline = group[0]
        base_eval = baseline.get("eval_panel")
        base_ce = baseline.get("ce")
        for row in group:
            r = dict(row)
            ev = row.get("eval_panel")
            ce = row.get("ce")
            r["delta_eval_vs_branch_base"] = (
                round(float(ev) - float(base_eval), 6)
                if ev is not None and base_eval is not None else None
            )
            r["delta_ce_vs_branch_base"] = (
                round(float(ce) - float(base_ce), 6)
                if ce is not None and base_ce is not None else None
            )
            extra = int(row.get("extra_bytes") or 0)
            delta_eval = r["delta_eval_vs_branch_base"]
            r["eval_gain_per_mb"] = (
                round(float(delta_eval) / (extra / 1_000_000), 6)
                if extra > 0 and delta_eval is not None else None
            )
            enriched.append(r)
    return enriched


def verdict(rows: list[dict[str, object]]) -> str:
    positives = [
        r for r in rows
        if (not r.get("is_reference")
            and not r.get("is_baseline")
            and r.get("delta_eval_vs_branch_base") is not None
            and float(r["delta_eval_vs_branch_base"]) >= 0.05)
    ]
    if positives:
        return "INVESTIGATE: at least one arm clears +0.05 eval over branch baseline."
    return "NO LOW-COST POSITIVE: current PC branches do not buy >=0.05 eval; wait for FACT-003H or a new mechanism."


def main() -> None:
    rows = enrich(wsync_rows() + side001_rows() + egrow002_rows())
    out_json = REPO_ROOT / "reports/rdt001_cost_ledger.json"
    out_md = REPO_ROOT / "reports/rdt001_cost_ledger.md"
    out_json.write_text(json.dumps({"rows": rows, "verdict": verdict(rows)}, indent=2), encoding="utf-8")

    lines = [
        "# RDT-001 Cost Ledger",
        "",
        "This ledger is a branch-prioritization smoke, not a quality claim.",
        "",
        "| branch | arm | eval | Δeval | CE | ΔCE | extra bytes | eval gain / MB | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['branch']} | {r['arm']} | {r.get('eval_panel')} | "
            f"{r.get('delta_eval_vs_branch_base')} | {r.get('ce')} | "
            f"{r.get('delta_ce_vs_branch_base')} | {r.get('extra_bytes')} | "
            f"{r.get('eval_gain_per_mb')} | {r.get('notes')} |"
        )
    lines += ["", "## Verdict", "", verdict(rows)]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(verdict(rows))


if __name__ == "__main__":
    main()
