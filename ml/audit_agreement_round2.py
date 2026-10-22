"""Round 2 agreement. Round-1 results (audit_agreement.py) stay on record;
this reports round 2 SEPARATELY, plus a pooled estimate labeled as post hoc.

    python ml/audit_agreement_round2.py

Targets (audit_config.py) were fixed before round 2:
  Fleiss' kappa on the fresh shared set >= FLEISS_TARGET
  pooled binary AI-human kappa >= BINARY_KAPPA_MIN
Pooled = round-1 individual items from annotators who are NOT superseded
+ round-2 individual items from the superseded annotators.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from audit_agreement import fleiss_kappa
from audit_config import BINARY_KAPPA_MIN, FLEISS_TARGET, SUPERSEDED_ROUND1

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data" / "audit"
PEOPLE = list("ABCDEF")
VALID = ["smishing", "spam", "legitimate", "unclassifiable"]


def load(sheet_dir: Path, key: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in PEOPLE:
        f = sheet_dir / f"annotator_{p}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, dtype=str, keep_default_na=False)
        s["label"] = s["label"].str.strip().str.lower()
        bad = s[~s["label"].isin(VALID)]
        if len(bad):
            raise SystemExit(f"{f.name}: {len(bad)} blank/invalid labels (fix first).")
        s["who"] = p
        rows.append(s[["msg_id", "label", "who"]])
    return pd.concat(rows).merge(key, on="msg_id")


def binary_kappa(d: pd.DataFrame, rng) -> tuple[float, float, float, int]:
    b = d[(d["ai_label"] != "unclassifiable") & (d["label"] != "unclassifiable")]
    a, h = (b["ai_label"] == "legitimate").to_numpy(), (b["label"] == "legitimate").to_numpy()
    k = cohen_kappa_score(a, h)
    ks = []
    for _ in range(3000):
        i = rng.integers(0, len(b), len(b))
        if len(set(a[i])) > 1 and len(set(h[i])) > 1:
            ks.append(cohen_kappa_score(a[i], h[i]))
    lo, hi = np.percentile(ks, [2.5, 97.5])
    return k, lo, hi, len(b)


def main() -> None:
    rng = np.random.default_rng(3)
    key2 = pd.read_csv(AUD / "round2" / "key_DO_NOT_SHARE.csv", dtype=str)
    h2 = load(AUD / "round2" / "sheets", key2)

    sh = h2[h2["set"] == "shared"]
    table = (sh.groupby(["msg_id", "label"]).size().unstack(fill_value=0)
             .reindex(columns=VALID, fill_value=0).to_numpy())
    kf = fleiss_kappa(table)
    boots = []
    for _ in range(2000):
        try:
            v = fleiss_kappa(table[rng.integers(0, len(table), len(table))])
            if np.isfinite(v):
                boots.append(v)
        except ZeroDivisionError:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"ROUND 2, shared set: Fleiss' kappa, six annotators (n={table.shape[0]}): "
          f"{kf:.3f} (95% CI {lo:.3f} to {hi:.3f}); target {FLEISS_TARGET:.2f}")

    ind2 = h2[(h2["set"] == "individual") & (h2["annotator"] == h2["who"])]
    k, l, u, n = binary_kappa(ind2, rng)
    print(f"ROUND 2, individual items of {', '.join(SUPERSEDED_ROUND1)} only: "
          f"binary AI-human kappa {k:.3f} (95% CI {l:.3f} to {u:.3f}, n={n}); "
          "small sample, interpret with care")

    key1 = pd.read_csv(AUD / "human_sample_key_DO_NOT_SHARE.csv", dtype=str)
    h1 = load(AUD / "sheets", key1)
    ind1 = h1[(h1["set"] == "individual") & (h1["annotator"] == h1["who"])
              & (~h1["who"].isin(SUPERSEDED_ROUND1))]
    pooled = pd.concat([ind1, ind2])
    k, l, u, n = binary_kappa(pooled, rng)
    print(f"POOLED (post hoc: round-1 items of {len(set(PEOPLE) - set(SUPERSEDED_ROUND1))} "
          f"annotators + round-2 items of {len(SUPERSEDED_ROUND1)}): binary AI-human kappa "
          f"{k:.3f} (95% CI {l:.3f} to {u:.3f}, n={n}); minimum {BINARY_KAPPA_MIN:.2f}")
    k3 = cohen_kappa_score(pooled["ai_label"], pooled["label"])
    print(f"POOLED three-class kappa (exploratory only): {k3:.3f}")

    print("\nDECISION (targets fixed before round 2):")
    ok_f, ok_b = kf >= FLEISS_TARGET, k >= BINARY_KAPPA_MIN
    print(f"  Human agreement target met: {ok_f}")
    print(f"  Pooled binary kappa target met: {ok_b}")
    if ok_f and ok_b:
        print("  Proceed to merge_labels.py. Report round 1 AND round 2 in Chapter 4.")
    else:
        print("  Not met: do not freeze. Bring this printout to the adviser; options are")
        print("  narrowing training to human-validated labels or a further targeted round.")


if __name__ == "__main__":
    main()
