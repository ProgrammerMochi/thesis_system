"""Section 3.6.4, step 3: check the filled sheets and compute agreement.

Put the six filled sheets back in data/audit/sheets/ (same file names).
Reports:
- Cohen's kappa, AI vs human, on the 180 individual items
- Fleiss' kappa, human vs human, on the 30 shared items
- Per-annotator AI agreement (spots one person misreading the codebook)
Decision rule from Section 3.6.4 (declared before human labeling):
- binary kappa (legitimate vs malicious) < 0.70: re-review ALL AI labels;
- only the three-class kappa < 0.70: re-review the smishing/spam sub-labels
  before per-subclass results are reported.
Fleiss' kappa is reported with a bootstrap 95% CI over the shared items.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data" / "audit"
PEOPLE = list("ABCDEF")
VALID = ["smishing", "spam", "legitimate", "unclassifiable"]


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, j] = number of raters who put item i in category j."""
    n = table.sum(axis=1)[0]
    assert (table.sum(axis=1) == n).all(), "every item needs the same number of raters"
    p_j = table.sum(axis=0) / table.sum()
    p_i = (np.square(table).sum(axis=1) - n) / (n * (n - 1))
    p_bar, p_e = p_i.mean(), np.square(p_j).sum()
    return float((p_bar - p_e) / (1 - p_e))


def main() -> None:
    key = pd.read_csv(AUD / "human_sample_key_DO_NOT_SHARE.csv", dtype=str)
    rows, problems = [], []
    for p in PEOPLE:
        s = pd.read_csv(AUD / "sheets" / f"annotator_{p}.csv", dtype=str,
                        keep_default_na=False)
        s["label"] = s["label"].str.strip().str.lower()
        for _, r in s[~s["label"].isin(VALID)].iterrows():
            problems.append(f"annotator {p}, {r['msg_id']}: label={r['label']!r}")
        s["annotator_file"] = p
        rows.append(s[["msg_id", "label", "annotator_file"]])
    if problems:
        print("Fix these labels first (blank or misspelled):")
        print("\n".join(problems[:50]))
        raise SystemExit(1)
    h = pd.concat(rows).merge(key, on="msg_id")

    ind = h[h["set"] == "individual"]
    ind = ind[ind["annotator"] == ind["annotator_file"]]
    k_ai = cohen_kappa_score(ind["ai_label"], ind["label"])
    print(f"Cohen's kappa, AI vs human, three classes (n={len(ind)}): {k_ai:.3f}")
    print("Rows = AI label, columns = human label")
    print(pd.DataFrame(confusion_matrix(ind["ai_label"], ind["label"], labels=VALID),
                       index=VALID, columns=VALID).to_string())
    # Binary: legitimate vs malicious (smishing or spam); unclassifiable excluded.
    b = ind[(ind["ai_label"] != "unclassifiable") & (ind["label"] != "unclassifiable")]
    k_bin = cohen_kappa_score(b["ai_label"] == "legitimate", b["label"] == "legitimate")
    print(f"\nCohen's kappa, AI vs human, binary legit-vs-malicious (n={len(b)}): {k_bin:.3f}")
    print("\nAI-human agreement per annotator (look for an outlier):")
    print(ind.assign(agree=ind["ai_label"] == ind["label"])
          .groupby("annotator")["agree"].mean().round(2).to_string())

    sh = h[h["set"] == "shared"]
    table = (sh.groupby(["msg_id", "label"]).size().unstack(fill_value=0)
             .reindex(columns=VALID, fill_value=0).to_numpy())
    k_f = fleiss_kappa(table)
    rng = np.random.default_rng(7)
    boots = []
    for _ in range(2000):
        t = table[rng.integers(0, len(table), len(table))]
        try:
            v = fleiss_kappa(t)
            if np.isfinite(v):
                boots.append(v)
        except ZeroDivisionError:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nFleiss' kappa, six humans (n={table.shape[0]} items): {k_f:.3f}  "
          f"(bootstrap 95% CI {lo:.3f} to {hi:.3f})")

    print("\nDECISION (Section 3.6.4):")
    if k_bin < 0.70:
        print("  Binary kappa < 0.70: re-review ALL AI labels before training.")
    elif k_ai < 0.70:
        print("  Binary kappa OK, three-class kappa < 0.70: re-review the smishing/spam "
              "sub-labels before reporting per-subclass results.")
    else:
        print("  Both kappas >= 0.70: AI labels accepted.")


if __name__ == "__main__":
    main()
