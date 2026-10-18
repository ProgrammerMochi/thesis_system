"""Section 3.6.4, step 1: AI-assisted pre-annotation helper.

    python ml/ai_prelabel.py batches    # makes data/audit/ai_batches/batch_XX.txt
    python ml/ai_prelabel.py collect    # checks replies, writes ai_labels.csv

Workflow: open a NEW chat for each batch file, paste the whole file, and save
the reply as data/audit/ai_outputs/batch_XX.csv (same number). Then run
`collect`. It refuses to build ai_labels.csv until every batch is complete
and valid, and tells you exactly which batch and which rows to redo.

Record in Section 3.6.4: model name and version, date, the exact prompt
(ml/ai_labeling_prompt.md, committed to git), and that each batch was a
fresh chat. Whoever runs this must NOT read the labels (see README).
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data" / "audit"
PROMPT = ROOT / "ml" / "ai_labeling_prompt.md"
BATCH = 100
LABELS = {"smishing", "spam", "legitimate", "unclassifiable"}
CONF = {"high", "medium", "low"}


def batch_ids(df: pd.DataFrame) -> list[list[str]]:
    ids = df["msg_id"].tolist()
    return [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]


def make_batches() -> None:
    df = pd.read_csv(AUD / "audit_input_blind.csv", dtype=str)
    prompt = PROMPT.read_text(encoding="utf-8").rstrip() + "\n\n"
    out = AUD / "ai_batches"
    out.mkdir(parents=True, exist_ok=True)
    by_id = df.set_index("msg_id")["text_norm"]
    for n, ids in enumerate(batch_ids(df), start=1):
        lines = [f"{i} | {by_id[i].replace(chr(10), ' ⏎ ')}" for i in ids]
        (out / f"batch_{n:02d}.txt").write_text(
            prompt + "\n".join(lines) + "\n", encoding="utf-8")
    (AUD / "ai_outputs").mkdir(exist_ok=True)
    print(f"Wrote {n} batch files to data/audit/ai_batches/ ({len(df)} messages).")
    print("Save each reply as data/audit/ai_outputs/batch_XX.csv")


def clean_reply(raw: str) -> str:
    """Tolerate a markdown fence or a stray sentence before the header."""
    lines = [ln for ln in raw.strip().splitlines() if not ln.strip().startswith("```")]
    start = next((i for i, ln in enumerate(lines)
                  if ln.lower().replace(" ", "").startswith("msg_id,")), None)
    return "\n".join(lines[start:]) if start is not None else ""


def collect() -> None:
    df = pd.read_csv(AUD / "audit_input_blind.csv", dtype=str)
    batches = batch_ids(df)
    good, bad = [], False
    for n, ids in enumerate(batches, start=1):
        f = AUD / "ai_outputs" / f"batch_{n:02d}.csv"
        if not f.exists():
            print(f"batch_{n:02d}: MISSING")
            bad = True
            continue
        text = clean_reply(f.read_text(encoding="utf-8"))
        if not text:
            print(f"batch_{n:02d}: no CSV header found (header must start 'msg_id,')")
            bad = True
            continue
        r = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                        quoting=csv.QUOTE_MINIMAL, on_bad_lines="skip")
        r.columns = [c.strip().lower() for c in r.columns]
        probs = []
        if not {"msg_id", "ai_label", "ai_confidence"} <= set(r.columns):
            print(f"batch_{n:02d}: wrong columns {list(r.columns)}")
            bad = True
            continue
        for c in ("msg_id", "ai_label", "ai_confidence"):
            r[c] = r[c].str.strip().str.lower() if c != "msg_id" else r[c].str.strip()
        missing = [i for i in ids if i not in set(r["msg_id"])]
        extra = [i for i in r["msg_id"] if i not in set(ids)]
        dups = r["msg_id"][r["msg_id"].duplicated()].tolist()
        badlab = r[~r["ai_label"].isin(LABELS) | ~r["ai_confidence"].isin(CONF)]
        if missing: probs.append(f"{len(missing)} ids missing (e.g. {missing[:2]})")
        if extra: probs.append(f"{len(extra)} unknown ids (e.g. {extra[:2]})")
        if dups: probs.append(f"{len(dups)} duplicate ids")
        if len(badlab): probs.append(f"{len(badlab)} invalid label/confidence values")
        if probs:
            print(f"batch_{n:02d}: " + "; ".join(probs) + "  -> redo this batch")
            bad = True
            continue
        if "criteria" not in r.columns:
            r["criteria"] = ""
        good.append(r[["msg_id", "ai_label", "ai_confidence", "criteria"]])
    if bad:
        sys.exit("\nFix the batches above, then run collect again.")
    out = pd.concat(good)
    assert len(out) == len(df) and out["msg_id"].is_unique
    out.to_csv(AUD / "ai_labels.csv", index=False)
    print(f"ai_labels.csv complete: {len(out)} messages.")
    print("(Do not read it. Only the aggregate counts below are safe to look at.)")
    print(out["ai_label"].value_counts().to_string())
    print(out["ai_confidence"].value_counts().to_string())


if __name__ == "__main__":
    cmds = {"batches": make_batches, "collect": collect}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]]()
