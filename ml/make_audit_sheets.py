"""Section 3.6.4, step 2: build the six blind human-labeling sheets.

Input:  data/audit/ai_labels.csv with columns
            msg_id, ai_label, ai_confidence
        ai_label in {smishing, spam, legitimate, unclassifiable}
        ai_confidence in {high, medium, low}
Output: data/audit/sheets/annotator_A.csv ... annotator_F.csv
        data/audit/sheets/adjudication_pair_*.csv
        data/audit/human_sample_key_DO_NOT_SHARE.csv

Design (state this in Section 3.6.4):
- SHARED set: 30 messages (10 per AI class) labeled by ALL six people.
  Used for Fleiss' kappa (human vs human).
- INDIVIDUAL set: 180 messages (60 per AI class), different from the shared
  set, split so each person gets 30 (10 per class). Used for Cohen's kappa
  (AI vs human).
- Equal allocation per class (not proportional) so the agreement estimate
  covers the rare smishing and legitimate classes, not only spam.
- Each person therefore labels 60 messages. Shared and individual items are
  shuffled together, so nobody can tell which items everyone else also has.
- Sheets contain msg_id and text only: no AI label, no source label.
- Low-confidence and unclassifiable AI items go to adjudication, split
  across three pairs; each pair labels its items together and agrees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data" / "audit"
SEED = 7
PEOPLE = list("ABCDEF")
CLASSES = ["smishing", "spam", "legitimate"]
SHARED_PER_CLASS = 10
INDIV_PER_PERSON_PER_CLASS = 10


def main() -> None:
    text = pd.read_csv(AUD / "audit_input_blind.csv", dtype=str)
    ai = pd.read_csv(AUD / "ai_labels.csv", dtype=str)
    ai["ai_label"] = ai["ai_label"].str.strip().str.lower()
    ai["ai_confidence"] = ai["ai_confidence"].str.strip().str.lower()
    bad = ai[~ai["ai_label"].isin(CLASSES + ["unclassifiable"])]
    assert bad.empty, f"Unknown AI labels:\n{bad.head()}"
    missing = set(text["msg_id"]) - set(ai["msg_id"])
    assert not missing, f"{len(missing)} messages have no AI label"
    df = text.merge(ai, on="msg_id")
    rng = np.random.default_rng(SEED)

    shared, indiv = [], []
    need = SHARED_PER_CLASS + INDIV_PER_PERSON_PER_CLASS * len(PEOPLE)
    for c in CLASSES:
        pool = df[df["ai_label"] == c]
        if len(pool) < need:
            raise SystemExit(f"Only {len(pool)} '{c}' items; need {need}. "
                             "Lower the per-class counts and report the change.")
        pick = pool.iloc[rng.permutation(len(pool))[:need]]
        shared.append(pick.iloc[:SHARED_PER_CLASS])
        rest = pick.iloc[SHARED_PER_CLASS:].copy()
        rest["annotator"] = np.repeat(PEOPLE, INDIV_PER_PERSON_PER_CLASS)
        indiv.append(rest)
    shared = pd.concat(shared).assign(annotator="ALL", set="shared")
    indiv = pd.concat(indiv).assign(set="individual")

    out = AUD / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for p in PEOPLE:
        mine = pd.concat([shared, indiv[indiv["annotator"] == p]])
        mine = mine.sample(frac=1, random_state=SEED + ord(p))
        sheet = mine[["msg_id", "text_norm"]].rename(columns={"text_norm": "text"})
        sheet["label"] = ""   # smishing / spam / legitimate / unclassifiable
        sheet["notes"] = ""
        sheet.to_csv(out / f"annotator_{p}.csv", index=False)

    key = pd.concat([shared, indiv])[["msg_id", "set", "annotator", "ai_label"]]
    key.to_csv(AUD / "human_sample_key_DO_NOT_SHARE.csv", index=False)

    # Adjudication: items the AI was unsure about, excluding ones already in
    # the human sample (those get human labels anyway).
    adj = df[(df["ai_confidence"] == "low") | (df["ai_label"] == "unclassifiable")]
    adj = adj[~adj["msg_id"].isin(key["msg_id"])].sample(frac=1, random_state=SEED)
    pairs = ["AB", "CD", "EF"]
    for i, pr in enumerate(pairs):
        part = adj.iloc[i::3][["msg_id", "text_norm"]].rename(columns={"text_norm": "text"})
        part["agreed_label"] = ""
        part["notes"] = ""
        part.to_csv(out / f"adjudication_pair_{pr}.csv", index=False)

    print(f"Shared: {len(shared)}  Individual: {len(indiv)}  "
          f"Per person: {len(shared) + len(indiv) // len(PEOPLE)}")
    print(f"Adjudication items: {len(adj)} (about {len(adj) // 3} per pair)")


if __name__ == "__main__":
    main()
