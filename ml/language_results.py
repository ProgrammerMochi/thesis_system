"""Combine the language sheets: Taglish share, reliability, labels for subgroups.

    python ml/language_results.py

Writes data/audit/language/language_labels_test.csv (msg_id, language) and prints
shares with Wilson 95% intervals, Fleiss' kappa on the shared messages (with a
bootstrap interval), and the language mix by class. Shared messages need at
least 4 of 6 annotators to agree, otherwise they are excluded and counted.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from audit_agreement import fleiss_kappa

ROOT = Path(__file__).resolve().parents[1]
LANG = ROOT / "data" / "audit" / "language"
PEOPLE = list("ABCDEF")
CATS = ["english", "tagalog", "taglish", "other"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    rows = []
    for p in PEOPLE:
        s = pd.read_csv(LANG / f"lang_annotator_{p}.csv", dtype=str, keep_default_na=False)
        s["language"] = s["language"].str.strip().str.lower()
        bad = s[~s["language"].isin(CATS)]
        if len(bad):
            raise SystemExit(f"lang_annotator_{p}.csv: {len(bad)} blank/invalid labels "
                             f"(allowed: {', '.join(CATS)})")
        s["who"] = p
        rows.append(s[["msg_id", "language", "who"]])
    a = pd.concat(rows)
    counts = a.groupby("msg_id")["who"].nunique()
    shared_ids = counts[counts == len(PEOPLE)].index
    sh = a[a["msg_id"].isin(shared_ids)]
    table = (sh.groupby(["msg_id", "language"]).size().unstack(fill_value=0)
             .reindex(columns=CATS, fill_value=0).to_numpy())
    kf = fleiss_kappa(table)
    rng = np.random.default_rng(5)
    boots = [fleiss_kappa(table[rng.integers(0, len(table), len(table))]) for _ in range(2000)]
    boots = [b for b in boots if np.isfinite(b)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"Fleiss' kappa on {len(table)} shared messages: {kf:.3f} (95% CI {lo:.3f} to {hi:.3f})")

    final, dropped = {}, 0
    for m, g in a.groupby("msg_id"):
        if m in set(shared_ids):
            v = g["language"].value_counts()
            if v.iloc[0] >= 4:
                final[m] = v.index[0]
            else:
                dropped += 1
        else:
            final[m] = g["language"].iloc[0]
    lab = pd.DataFrame({"msg_id": list(final), "language": list(final.values())})
    lab.to_csv(LANG / "language_labels_test.csv", index=False)
    n = len(lab)
    print(f"\nLanguage labels for {n} test messages ({dropped} shared messages excluded: no 4-of-6 majority)")
    print("\nShare of the test split (Wilson 95% CI):")
    for c in CATS:
        k = int((lab["language"] == c).sum())
        l, u = wilson(k, n)
        print(f"  {c:8s} {k:4d}  {k / n:6.1%}  ({l:.1%} to {u:.1%})")
    k = int(lab["language"].isin(["tagalog", "taglish"]).sum())
    l, u = wilson(k, n)
    print(f"  Tagalog-containing (tagalog + taglish): {k} = {k / n:.1%} ({l:.1%} to {u:.1%})")

    c = pd.read_csv(ROOT / "data" / "processed" / "corpus.csv", keep_default_na=False)
    m = lab.merge(c[["msg_id", "label_bin"]], on="msg_id")
    print("\nLanguage mix by class (0 = legitimate, 1 = malicious):")
    print(pd.crosstab(m["label_bin"].astype(int), m["language"]).reindex(columns=CATS, fill_value=0).to_string())


if __name__ == "__main__":
    main()
