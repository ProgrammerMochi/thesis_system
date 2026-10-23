"""Numbers for Chapters 1, 3 and 4, from the frozen corpus (run once, paste output).

    python ml/corpus_stats.py

Prints sizes by split and label, URL counts (including the small subsets that
decide confidence-interval width), where the self-collected messages landed,
label provenance, length and language-mix estimates, and the largest template
families. The language mix is a transparent HEURISTIC (function-word lists);
validate it on a hand-checked sample before putting it in the paper.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "processed" / "corpus.csv", keep_default_na=False)
df["has_url"] = df["has_url"].astype(str).str.lower().eq("true")
df["label_bin"] = df["label_bin"].astype(int)
df["n_chars"] = df["n_chars"].astype(int)

TAG = set("""ang ng sa mga po lang para naman kayo tayo ikaw ako ko mo mong iyong ito hindi
yung ngayon dito sana kung nang pag namin natin nila nyo niyo ninyo kami siya mayroon ba
din rin pa na ka mag may""".split()) - {"may", "pa", "na", "ka", "mag", "din", "rin"}
ENG = set("""the and to of is for you your with on at in this that are will now get our we it
be have free win now""".split())


def lang(t: str) -> str:
    w = set(re.findall(r"[a-zA-Z']+", t.lower()))
    tg, en = len(w & TAG) >= 2, len(w & ENG) >= 2
    return "taglish" if tg and en else "tagalog" if tg else "english" if en else "other"


def title(s: str) -> None:
    print("\n" + "=" * 68 + f"\n{s}\n" + "=" * 68)


title("1. CORPUS")
n_mal = int(df.label_bin.sum())
print(f"Messages: {len(df)} | malicious {n_mal} ({n_mal / len(df):.1%}) | legitimate {len(df) - n_mal}")
print(df["label3"].value_counts().to_string())

title("2. SPLITS (sizes, label mix, URL subsets)")
print(pd.crosstab(df["split"], df["label3"], margins=True).to_string())
print()
g = df.groupby("split")
print(pd.DataFrame({
    "n": g.size(), "pos_rate": g["label_bin"].mean().round(3),
    "with_url": g["has_url"].sum(),
    "mal_no_url": g.apply(lambda d: int(((d.label_bin == 1) & ~d.has_url).sum())),
    "legit_with_url": g.apply(lambda d: int(((d.label_bin == 0) & d.has_url).sum())),
}).to_string())
print("\n(mal_no_url in TEST is the subset behind 'recall without a URL'; "
      "legit_with_url shows how little the corpus can say about legitimate links.)")

title("3. SELF-COLLECTED MESSAGES (shortcut check)")
s = df[df["source"] == "self"]
print(f"{len(s)} messages")
print(pd.crosstab(s["split"], s["label3"]).to_string() if len(s) else "none")

title("4. LABEL PROVENANCE (Chapter 4 must state this)")
pv = df["label_method"].value_counts()
print(pd.DataFrame({"n": pv, "share": (pv / len(df)).round(3)}).to_string())
hv = pv.reindex(["adjudicated", "human", "human_r2", "human_majority",
                 "human_majority_r2", "group"]).fillna(0).sum()
print(f"\nHuman-decided (incl. adjudicated/group): {int(hv)} ({hv / len(df):.1%})")

title("5. URL RATE AND LENGTH")
print(df.groupby("label_bin").agg(n=("msg_id", "size"), url_rate=("has_url", "mean"),
                                  median_chars=("n_chars", "median")).round(3).to_string())

title("6. LANGUAGE MIX (heuristic, validate by hand)")
df["lang"] = df["text_model"].map(lang)
print(df["lang"].value_counts().to_string())
print((df["lang"].value_counts(normalize=True) * 100).round(1).astype(str).add("%").to_string())
print("\nBy class:")
print(pd.crosstab(df["label_bin"], df["lang"], normalize="index").round(3).to_string())
print("\nHand-check sample (read these 30 and count how many labels are wrong):")
for _, r in df.sample(30, random_state=1).iterrows():
    print(f"  [{r['lang']:7s}] {r['text_model'].replace(chr(10), ' / ')[:110]}")

title("7. NEAR-DUPLICATE FAMILIES")
tc = df["template_id"].value_counts()
print(f"Distinct templates: {tc.size} for {len(df)} messages; largest families: {tc.head(5).tolist()}")
