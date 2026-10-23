"""Make the minimal private copy of the frozen corpus for Kaggle.

    python ml/export_for_kaggle.py

Writes data/processed/corpus_kaggle.csv with ONLY the columns training needs
(data minimization, RA 10173): no sender, no un-masked text. Upload it as a
PRIVATE Kaggle dataset. Never commit it. Record the printed SHA-256 hashes in
the README / Chapter 3 so the exact frozen corpus can be verified later.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "corpus.csv"
DST = ROOT / "data" / "processed" / "corpus_kaggle.csv"
COLS = ["msg_id", "source", "text_model", "urls", "has_url", "n_chars",
        "template_id", "label3", "label_bin", "label_method", "split"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    df = pd.read_csv(SRC, keep_default_na=False)
    assert df["text_model"].ne("").all(), "empty text_model rows"
    assert set(df["split"]) == {"train", "val", "test"}
    assert set(df["label_bin"].astype(int)) <= {0, 1}
    df[COLS].to_csv(DST, index=False)
    print(f"Wrote {DST.name}: {len(df)} rows, {len(COLS)} columns")
    print(f"SHA-256 corpus.csv         {sha(SRC)}")
    print(f"SHA-256 corpus_kaggle.csv  {sha(DST)}")
    print("Upload corpus_kaggle.csv as a PRIVATE Kaggle dataset. Do not commit it.")


if __name__ == "__main__":
    main()
