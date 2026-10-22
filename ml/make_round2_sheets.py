"""Round 2 (after the codebook rules are clarified and written down).

    python ml/make_round2_sheets.py

Draws FRESH messages: not in the round-1 human sample and not in any
adjudication file, so no annotator sees a message again.
  - 30 shared messages (10 per AI class) labeled by all six annotators;
  - 30 fresh individual messages (10 per AI class) for each annotator in
    audit_config.SUPERSEDED_ROUND1;
  - pair EF's adjudication items, split between two other pairs.
Sheets show text only (emails and mobile numbers masked), no AI/source label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_config import REDO_ADJ_PAIR, REDO_ADJ_TO, SUPERSEDED_ROUND1
from make_audit_sheets import redact

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data" / "audit"
OUT = AUD / "round2"
SEED = 11
PEOPLE = list("ABCDEF")
CLASSES = ["smishing", "spam", "legitimate"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared-per-class", type=int, default=10)
    ap.add_argument("--indiv-per-class", type=int, default=10)
    args = ap.parse_args()

    text = pd.read_csv(AUD / "audit_input_blind.csv", dtype=str)
    ai = pd.read_csv(AUD / "ai_labels.csv", dtype=str)[["msg_id", "ai_label"]]
    df = text.merge(ai, on="msg_id")
    key1 = pd.read_csv(AUD / "human_sample_key_DO_NOT_SHARE.csv", dtype=str)
    used = set(key1["msg_id"])
    adj_files = sorted((AUD / "sheets").glob("adjudication_pair_*.csv"))
    for f in adj_files:
        used |= set(pd.read_csv(f, dtype=str)["msg_id"])
    fresh = df[~df["msg_id"].isin(used)]
    print(f"Fresh pool: {len(fresh)} messages (excluded {len(used)} already used)")

    n_ind = len(SUPERSEDED_ROUND1)
    need = args.shared_per_class + args.indiv_per_class * n_ind
    rng = np.random.default_rng(SEED)
    rows = []
    for c in CLASSES:
        pool = fresh[fresh["ai_label"] == c]
        if len(pool) < need:
            raise SystemExit(
                f"Only {len(pool)} fresh '{c}' messages; need {need}. Lower "
                "--shared-per-class / --indiv-per-class and report the change.")
        pick = pool.iloc[rng.permutation(len(pool))[:need]].copy()
        pick["set"] = ["shared"] * args.shared_per_class + ["individual"] * (need - args.shared_per_class)
        who = ["ALL"] * args.shared_per_class
        for p in SUPERSEDED_ROUND1:
            who += [p] * args.indiv_per_class
        pick["annotator"] = who
        rows.append(pick)
    key = pd.concat(rows)

    (OUT / "sheets").mkdir(parents=True, exist_ok=True)
    for p in PEOPLE:
        mine = key[(key["set"] == "shared") | (key["annotator"] == p)]
        mine = mine.sample(frac=1, random_state=SEED + ord(p))
        sheet = mine[["msg_id", "text_norm"]].rename(columns={"text_norm": "text"})
        sheet["text"] = sheet["text"].map(redact)
        sheet["label"] = ""
        sheet["notes"] = ""
        sheet.to_csv(OUT / "sheets" / f"annotator_{p}.csv", index=False)
    key[["msg_id", "set", "annotator", "ai_label"]].to_csv(
        OUT / "key_DO_NOT_SHARE.csv", index=False)

    # Redo of pair EF's adjudication by two other pairs.
    redo_src = AUD / "sheets" / f"adjudication_pair_{REDO_ADJ_PAIR}.csv"
    redo = pd.read_csv(redo_src, dtype=str)[["msg_id", "text"]].copy()
    redo["agreed_label"] = ""
    redo["notes"] = ""
    for i, pr in enumerate(REDO_ADJ_TO):
        redo.iloc[i::len(REDO_ADJ_TO)].to_csv(
            OUT / "sheets" / f"adjudication_redo_{pr}.csv", index=False)

    n_sh = int((key["set"] == "shared").sum())
    print(f"Shared: {n_sh} | individual per superseded annotator: "
          f"{args.indiv_per_class * len(CLASSES)} | redo adjudication items: {len(redo)} "
          f"(about {len(redo) // len(REDO_ADJ_TO)} per pair)")
    print(f"Sheets in {OUT / 'sheets'}")


if __name__ == "__main__":
    main()
