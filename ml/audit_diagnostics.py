"""Diagnose WHERE the audit disagreement comes from (run after audit_agreement.py).

    python ml/audit_diagnostics.py

This does not change any label or any threshold. It answers four questions:
  1. Do the six humans disagree with each other, or with the AI, or both?
  2. Is one annotator (or two) the outlier?
  3. Is the AI-vs-human disagreement concentrated in one source category
     (for example institutional promotions, the legitimate/spam boundary)?
  4. Which shared messages split the group, and how?

Annotator-removal figures below are DIAGNOSTIC ONLY. Do not drop annotators
from the reported results because they disagree; report all six.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_agreement import fleiss_kappa  # noqa: E402
from prepare_corpus import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data" / "audit"
PEOPLE = list("ABCDEF")
LABELS = ["smishing", "spam", "legitimate", "unclassifiable"]


def fleiss_from(votes: pd.DataFrame, people: list[str]) -> float:
    v = votes[people]
    t = np.array([[int((v.loc[i] == c).sum()) for c in LABELS] for i in v.index])
    return fleiss_kappa(t)


def main() -> None:
    key = pd.read_csv(AUD / "human_sample_key_DO_NOT_SHARE.csv", dtype=str)
    rows = []
    texts = {}
    for p in PEOPLE:
        s = pd.read_csv(AUD / "sheets" / f"annotator_{p}.csv", dtype=str, keep_default_na=False)
        s["label"] = s["label"].str.strip().str.lower()
        for _, r in s.iterrows():
            texts[r["msg_id"]] = r["text"]
        s["who"] = p
        rows.append(s[["msg_id", "label", "who"]])
    h = pd.concat(rows).merge(key, on="msg_id")

    # ---- 1. Do humans agree with each other? ---------------------------
    sh = h[h["set"] == "shared"]
    votes = sh.pivot(index="msg_id", columns="who", values="label")[PEOPLE]
    print("=" * 70)
    print("1. HUMAN vs HUMAN on the 30 shared messages")
    print("=" * 70)
    print(f"Fleiss' kappa, all six: {fleiss_from(votes, PEOPLE):.3f}")
    print("\nPairwise Cohen's kappa (three classes):")
    pk = pd.DataFrame(index=PEOPLE, columns=PEOPLE, dtype=float)
    for a, b in itertools.product(PEOPLE, PEOPLE):
        pk.loc[a, b] = 1.0 if a == b else cohen_kappa_score(votes[a], votes[b])
    print(pk.round(2).to_string())
    print("\nMean agreement of each person with the other five (kappa):")
    print(pd.Series({p: pk.loc[p, [q for q in PEOPLE if q != p]].mean() for p in PEOPLE})
          .round(2).to_string())
    print("\nFleiss' kappa with one person removed (diagnostic only):")
    for p in PEOPLE:
        rest = [q for q in PEOPLE if q != p]
        print(f"  without {p}: {fleiss_from(votes, rest):.3f}")
    best = sorted(((fleiss_from(votes, [x for x in PEOPLE if x not in c]), c)
                   for c in itertools.combinations(PEOPLE, 2)), reverse=True)[:3]
    print("Best three removals of two people:",
          "; ".join(f"{''.join(c)} -> {k:.3f}" for k, c in best))

    # ---- 2. Each person's label habits ---------------------------------
    print("\n" + "=" * 70)
    print("2. LABEL COUNTS PER PERSON (all 60 messages each)")
    print("=" * 70)
    print(pd.crosstab(h["who"], h["label"]).reindex(columns=LABELS, fill_value=0).to_string())

    # ---- 3. AI vs human on the individual set, by source category ------
    corpus = build(verbose=False)[["msg_id", "source", "source_category", "label_source"]]
    ind = h[(h["set"] == "individual") & (h["annotator"] == h["who"])]
    ind = ind.merge(corpus, on="msg_id", how="left")
    print("\n" + "=" * 70)
    print("3. AI vs HUMAN on the 180 individual messages, by SOURCE CATEGORY")
    print("   (Chua categories: ads/gov/notifs/spam; Wandowando = spam_scam)")
    print("=" * 70)
    ct = pd.crosstab([ind["source_category"], ind["ai_label"]], ind["label"])
    print(ct.reindex(columns=LABELS, fill_value=0).to_string())

    ai_leg = ind[ind["ai_label"] == "legitimate"]
    print(f"\nAI said 'legitimate' on {len(ai_leg)} messages; humans said:")
    print(ai_leg["label"].value_counts().to_string())
    print("...of which humans said spam, by source category:")
    print(ai_leg[ai_leg["label"] == "spam"]["source_category"].value_counts().to_string())

    # ---- binary agreement with uncertainty, and with each person removed
    b = ind[(ind["ai_label"] != "unclassifiable") & (ind["label"] != "unclassifiable")]
    a_leg = (b["ai_label"] == "legitimate").to_numpy()
    h_leg = (b["label"] == "legitimate").to_numpy()
    rng = np.random.default_rng(0)
    ks = []
    for _ in range(3000):
        i = rng.integers(0, len(b), len(b))
        if len(set(a_leg[i])) > 1 and len(set(h_leg[i])) > 1:
            ks.append(cohen_kappa_score(a_leg[i], h_leg[i]))
    print(f"\nBinary kappa AI vs human: {cohen_kappa_score(a_leg, h_leg):.3f} "
          f"(bootstrap 95% CI {np.percentile(ks, 2.5):.3f} to {np.percentile(ks, 97.5):.3f}, n={len(b)})")
    print("Binary kappa with one annotator's 30 messages removed (diagnostic only):")
    for p in PEOPLE:
        bb = b[b["who"] != p]
        k = cohen_kappa_score(bb["ai_label"] == "legitimate", bb["label"] == "legitimate")
        print(f"  without {p}: {k:.3f}")

    # ---- 4. The shared messages that split the group -------------------
    print("\n" + "=" * 70)
    print("4. SHARED MESSAGES THAT SPLIT THE GROUP (most split first)")
    print("=" * 70)
    ai_of = key.drop_duplicates("msg_id").set_index("msg_id")["ai_label"]
    src = corpus.set_index("msg_id")["source_category"]
    out = []
    for m, r in votes.iterrows():
        top = r.value_counts().iloc[0]
        out.append((top, m))
    for top, m in sorted(out)[:12]:
        r = votes.loc[m]
        tally = " ".join(f"{c[:4]}={int((r == c).sum())}" for c in LABELS if (r == c).any())
        print(f"[{tally}] AI={ai_of[m][:5]} src={src.get(m, '?')}")
        print("   " + texts[m].replace("\n", " ⏎ ")[:170])


if __name__ == "__main__":
    main()
