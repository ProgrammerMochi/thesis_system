"""Increment 1: build the corpus, the frozen split, and the audit input.

Usage (from repo root):
    python ml/prepare_corpus.py               # dry run: stats only
    python ml/prepare_corpus.py --write       # writes data/processed/*

Run with --write exactly ONCE at corpus freeze (Oct 21), after the
self-collected file is in place. The split is then never regenerated:
regenerating it after seeing any results is a form of test-set leakage.

Label scheme (decided Oct 18): positive class = smishing + spam (1),
negative = legitimate (0). Source labels below are PROVISIONAL; the
Section 3.6.4 audit supersedes them and also assigns the three-class
sub-label (smishing / spam / legitimate) used for per-subclass recall.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from textnorm import normalize, template_key, to_model_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42
N_FOLDS = 20  # 14 train / 3 val / 3 test folds -> 70/15/15
LEGIT_AUDIT_SAMPLE = 150

CHUA_MAP = {"spam": 1, "ads": 0, "gov": 0, "notifs": 0}  # OTP excluded


def load_chua(path: Path) -> pd.DataFrame:
    rows, malformed = [], []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["date-received", "date-read", "sender", "category", "text"], header
        for i, r in enumerate(reader, start=1):
            if len(r) != 5:
                # Known case: two rows carry two category fields
                # (e.g. "ads","notifs"). Text is the last field.
                malformed.append((i, r[3:-1], r[-1][:40]))
                r = r[:3] + [r[3]] + [r[-1]]
            rows.append(r)
    for m in malformed:
        print(f"  [chua] malformed row {m[0]}: categories={m[1]} text={m[2]!r}")
    df = pd.DataFrame(rows, columns=header)
    df["source"] = "chua"
    df["source_category"] = df["category"]
    return df[["source", "source_category", "sender", "text"]]


def load_wando(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["source"] = "wandowando"
    df["source_category"] = "spam_scam"
    df = df.rename(columns={"masked_celphone_number": "sender"})
    return df[["source", "source_category", "sender", "text"]]


def load_self_collected(path: Path) -> pd.DataFrame | None:
    """Expected columns: text. Already screened for incidental personal
    details at collection time (Section 3.6.4)."""
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["source"] = "self"
    df["source_category"] = "smishing_collected"
    df["sender"] = ""
    return df[["source", "source_category", "sender", "text"]]


def build(verbose: bool = True) -> pd.DataFrame:
    parts = [load_chua(RAW / "scottleechua.csv"), load_wando(RAW / "bwandowando.csv")]
    sc = load_self_collected(RAW / "self_collected.csv")
    if sc is not None:
        parts.append(sc)
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)

    df = df[df["source_category"] != "OTP"]
    df["text_norm"] = df["text"].map(normalize)
    # Chua uses two placeholders for withheld text.
    df = df[~df["text_norm"].isin(["", "<REDACTED>", "<BLANK>"])]
    n_after_redact = len(df)

    df["label_source"] = np.where(
        df["source"] == "chua", df["source_category"].map(CHUA_MAP), 1
    ).astype(int)

    # Within-source, then cross-source exact dedup on normalized text.
    # Conflicts (same text, different source labels) are kept once and
    # flagged for the audit instead of being silently resolved.
    df["n_copies"] = df.groupby("text_norm")["text_norm"].transform("size")
    conflict = df.groupby("text_norm")["label_source"].nunique() > 1
    df["label_conflict"] = df["text_norm"].map(conflict)
    # Prefer the self-collected / Chua row as the kept copy (has sender info).
    order = {"self": 0, "chua": 1, "wandowando": 2}
    df = df.sort_values(by="source", key=lambda s: s.map(order), kind="stable")
    df = df.drop_duplicates("text_norm", keep="first").reset_index(drop=True)

    mt = df["text_norm"].map(to_model_text)
    df["text_model"] = mt.map(lambda x: x[0])
    df["urls"] = mt.map(lambda x: " ".join(x[1]))
    df["has_url"] = df["urls"] != ""
    df["template"] = df["text_model"].map(template_key)
    df["template_id"] = df["template"].map(
        lambda s: hashlib.sha1(s.encode()).hexdigest()[:12]
    )
    # Stable ID from the text itself, so IDs never change when the
    # self-collected file is added later (audit labels join on msg_id).
    df["msg_id"] = df["text_norm"].map(
        lambda s: "m" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:10])
    assert df["msg_id"].is_unique
    df["n_chars"] = df["text_norm"].str.len()

    if verbose:
        print(f"\nRaw rows: {n0}; after OTP/redacted/empty removal: {n_after_redact}")
        print(f"Unique after dedup: {len(df)}")
        print("By source/category:")
        print(df.groupby(["source", "source_category"]).size().to_string())
        print("Provisional labels:", df["label_source"].value_counts().to_dict())
        print("Cross-source label conflicts:", int(df["label_conflict"].sum()))
        print("Distinct templates:", df["template_id"].nunique())
        print("\nURL rate and median length by provisional label:")
        print(df.groupby("label_source").agg(
            n=("msg_id", "size"), url_rate=("has_url", "mean"),
            median_chars=("n_chars", "median")).round(3).to_string())
    return df


def split(df: pd.DataFrame) -> pd.Series:
    """Template-grouped, label-stratified 70/15/15 split."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label_source"], df["template_id"])):
        fold[idx] = k
    return pd.Series(
        np.select([fold < 14, fold < 17], ["train", "val"], "test"), index=df.index
    )


def write_audit(df: pd.DataFrame) -> None:
    """Section 3.6.4 audit input: all candidate-malicious messages from the
    public datasets + 150 random legitimate ones, shuffled and stripped of
    source labels so annotators (AI and human) are blind to them.
    Self-collected messages are labeled manually and are not included."""
    out = ROOT / "data" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    pub = df[df["source"] != "self"]
    rng = np.random.default_rng(SEED)
    pos = pub[pub["label_source"] == 1]
    neg = pub[pub["label_source"] == 0]
    neg_s = neg.iloc[rng.choice(len(neg), LEGIT_AUDIT_SAMPLE, replace=False)]
    audit = pd.concat([pos, neg_s]).sample(frac=1, random_state=SEED)
    audit[["msg_id", "text_norm"]].to_csv(out / "audit_input_blind.csv", index=False)
    audit[["msg_id", "label_source", "source", "label_conflict"]].to_csv(
        out / "audit_key_DO_NOT_SHARE.csv", index=False)
    print(f"Wrote {len(audit)} messages to data/audit/audit_input_blind.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="freeze corpus + split (run once, Oct 21)")
    ap.add_argument("--audit", action="store_true",
                    help="write audit sheets only; does NOT freeze the split")
    args = ap.parse_args()

    df = build()
    if args.audit:
        write_audit(df)
        return
    df["split"] = split(df)

    # Leakage check: no template may appear in two splits.
    leak = df.groupby("template_id")["split"].nunique().gt(1).sum()
    assert leak == 0, f"{leak} templates span multiple splits"
    print("\nSplit sizes and positive rate:")
    print(df.groupby("split").agg(n=("msg_id", "size"),
                                  pos_rate=("label_source", "mean"),
                                  urls=("has_url", "sum")).round(3).to_string())

    if not args.write:
        print("\nDry run. Re-run with --write at corpus freeze.")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "corpus.csv").exists():
        sys.exit("data/processed/corpus.csv exists. The split is frozen; refusing to overwrite.")
    cols = ["msg_id", "source", "source_category", "sender", "text_norm", "text_model",
            "urls", "has_url", "n_chars", "template_id", "label_source",
            "label_conflict", "n_copies", "split"]
    df[cols].to_csv(OUT / "corpus.csv", index=False)

    print(f"\nWrote corpus.csv ({len(df)}).")


if __name__ == "__main__":
    main()
