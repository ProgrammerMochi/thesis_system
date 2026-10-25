"""Verify that data/processed/corpus.csv is the frozen corpus.

Prints the file's SHA-256 and compares it with the corpus.csv value recorded
in README.md. Exit codes: 0 = match, 1 = mismatch, 2 = file or README value
missing. Read-only: the CSV is only read, never modified.

    python scripts/check_corpus_hash.py
    python scripts/check_corpus_hash.py --split-counts   # also rows per split
"""
import argparse
import csv
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "processed" / "corpus.csv"
README = ROOT / "README.md"
README_RE = re.compile(r"\b([0-9a-f]{64}) \(corpus\.csv\)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_counts(path: Path) -> Counter:
    """Rows per split, reading only the `split` column (no message text)."""
    with open(path, newline="", encoding="utf-8") as f:
        return Counter(row["split"] for row in csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-counts", action="store_true")
    args = ap.parse_args()
    if not CORPUS.exists():
        print(f"missing: {CORPUS.relative_to(ROOT)} (data/ is not in git)")
        return 2
    m = README_RE.search(README.read_text(encoding="utf-8"))
    if not m:
        print("no corpus.csv hash found in README.md")
        return 2
    actual, recorded = sha256(CORPUS), m.group(1)
    print(f"actual   {actual}")
    print(f"README   {recorded}")
    if args.split_counts:
        counts = split_counts(CORPUS)
        for name in ("train", "val", "test"):
            print(f"{name:5s} {counts.get(name, 0)}")
        print(f"total {sum(counts.values())}")
    if actual != recorded:
        print("MISMATCH: corpus.csv is not the recorded frozen corpus")
        return 1
    print("OK: corpus.csv matches README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
