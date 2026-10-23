"""Hand-labeling of message LANGUAGE for the test split (replaces the crude
function-word heuristic in corpus_stats.py, which undercounts Taglish).

    python ml/make_language_sheets.py

Creates one sheet per annotator in data/audit/language/:
  - 30 shared messages labeled by all six (reliability);
  - the remaining test messages divided equally (about 59 each).
Sheets show msg_id and text only: no class label, no AI label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "audit" / "language"
PEOPLE = list("ABCDEF")
N_SHARED = 30
SEED = 21


def main() -> None:
    c = pd.read_csv(ROOT / "data" / "processed" / "corpus.csv", keep_default_na=False)
    t = c[c["split"] == "test"][["msg_id", "text_model"]].rename(columns={"text_model": "text"})
    t = t.sample(frac=1, random_state=SEED).reset_index(drop=True)
    shared, rest = t.iloc[:N_SHARED], t.iloc[N_SHARED:]
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = np.array_split(np.arange(len(rest)), len(PEOPLE))
    for p, idx in zip(PEOPLE, chunks):
        mine = pd.concat([shared, rest.iloc[idx]]).sample(frac=1, random_state=SEED + ord(p))
        mine = mine.assign(language="", notes="")
        mine.to_csv(OUT / f"lang_annotator_{p}.csv", index=False)
    print(f"{len(t)} test messages: {N_SHARED} shared + {[len(i) for i in chunks]} individual")
    print(f"Sheets in {OUT}")


if __name__ == "__main__":
    main()
