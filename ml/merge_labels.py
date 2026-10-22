"""Section 3.6.4: merge all label sources into data/audit/final_labels.csv.

Run after the human sheets and pair adjudication files are filled in and
audit_agreement.py has been run. Then run prepare_corpus.py --write once.

Precedence (state this in Section 3.6.4), highest first:
  0. group          full-group decision from data/audit/group_decisions.csv
                   (columns msg_id,label), used only for the items listed in
                   needs_group_decision.csv
  1. adjudicated   pair-agreed label (adjudication_pair_*.csv)
  2. human         the researcher's label on an individual-set message
  3. human_majority  majority of the six labels on a shared-set message
                   (needs at least 4 of 6 to agree, else sent to the group)
  3b. If data/audit/round2/ exists (round 2 was run): the round-1 labels of
      annotators in audit_config.SUPERSEDED_ROUND1 are NOT used; round-1
      shared items are decided by the remaining annotators (ceil(2n/3)
      must agree); round-2 individual labels (method human_r2) and round-2
      shared majorities (human_majority_r2, at least 4 of 6) are added; and
      pair EF's adjudication is replaced by the redo files.
  4. self_pair     pair-assigned label on a self-collected message
  5. ai            AI label on an audited message with no human label
  6. source        source-dataset label (legitimate) on messages outside the
                   audit sample

Messages that still need a group decision (a pair wrote GROUP or left it
blank, or fewer than 4 of 6 agreed on a shared item) are listed in
data/audit/needs_group_decision.csv. final_labels.csv is NOT written until
that list is empty, unless --allow-unresolved is passed (then the
unresolved messages are marked unclassifiable and excluded, and the count
must be reported).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_config import REDO_ADJ_PAIR, REDO_ADJ_TO, SUPERSEDED_ROUND1  # noqa: E402
from prepare_corpus import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "data" / "audit"
PEOPLE = list("ABCDEF")
VALID = {"smishing", "spam", "legitimate", "unclassifiable"}


def read_sheet(path: Path, label_col: str) -> pd.DataFrame:
    s = pd.read_csv(path, dtype=str, keep_default_na=False)
    s[label_col] = s[label_col].str.strip().str.lower()
    return s[["msg_id", label_col]].rename(columns={label_col: "label"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unresolved", action="store_true")
    args = ap.parse_args()

    df = build(verbose=False)
    ai = pd.read_csv(AUD / "ai_labels.csv", dtype=str).set_index("msg_id")["ai_label"]
    key = pd.read_csv(AUD / "human_sample_key_DO_NOT_SHARE.csv", dtype=str)
    in_audit = set(pd.read_csv(AUD / "audit_input_blind.csv", dtype=str)["msg_id"])

    # --- human labels, round 1 ----------------------------------------
    r2 = AUD / "round2"
    r2_on = (r2 / "key_DO_NOT_SHARE.csv").exists()
    superseded = set(SUPERSEDED_ROUND1) if r2_on else set()
    indiv_ids = set(key.loc[key["set"] == "individual", "msg_id"])
    shared_ids = set(key.loc[key["set"] == "shared", "msg_id"])
    human_ind, shared_votes, method_of = {}, {}, {}
    for p in PEOPLE:
        sh = read_sheet(AUD / "sheets" / f"annotator_{p}.csv", "label")
        bad = sh[~sh["label"].isin(VALID)]
        if len(bad):
            sys.exit(f"annotator_{p}.csv has {len(bad)} blank/invalid labels; fix first.")
        if p in superseded:
            continue                      # round-1 labels replaced by round 2
        for _, r in sh.iterrows():
            if r["msg_id"] in indiv_ids:
                human_ind[r["msg_id"]] = r["label"]
                method_of[r["msg_id"]] = "human"
            elif r["msg_id"] in shared_ids:
                shared_votes.setdefault(r["msg_id"], []).append(r["label"])
    n_raters1 = len(PEOPLE) - len(superseded)
    min_agree1 = math.ceil(2 * n_raters1 / 3)
    shared_rule = {m: (n_raters1, min_agree1, "human_majority") for m in shared_votes}

    # --- human labels, round 2 ----------------------------------------
    if r2_on:
        key2 = pd.read_csv(r2 / "key_DO_NOT_SHARE.csv", dtype=str)
        ind2 = set(key2.loc[key2["set"] == "individual", "msg_id"])
        sh2 = set(key2.loc[key2["set"] == "shared", "msg_id"])
        for p in PEOPLE:
            f = r2 / "sheets" / f"annotator_{p}.csv"
            if not f.exists():
                sys.exit(f"round 2 sheet missing: {f.name}")
            sh = read_sheet(f, "label")
            bad = sh[~sh["label"].isin(VALID)]
            if len(bad):
                sys.exit(f"round2 annotator_{p}.csv has {len(bad)} blank/invalid labels.")
            for _, r in sh.iterrows():
                if r["msg_id"] in ind2:
                    human_ind[r["msg_id"]] = r["label"]
                    method_of[r["msg_id"]] = "human_r2"
                elif r["msg_id"] in sh2:
                    shared_votes.setdefault(r["msg_id"], []).append(r["label"])
                    shared_rule[r["msg_id"]] = (len(PEOPLE), math.ceil(2 * len(PEOPLE) / 3),
                                                "human_majority_r2")

    # --- adjudication (pairs) ----------------------------------------
    adj = {}
    pairs = ["AB", "CD", "EF"]
    if r2_on:
        pairs = [x for x in pairs if x != REDO_ADJ_PAIR]
    for pr in pairs:
        f = AUD / "sheets" / f"adjudication_pair_{pr}.csv"
        if not f.exists():
            continue
        a = read_sheet(f, "agreed_label")
        for _, r in a.iterrows():
            adj[r["msg_id"]] = r["label"]   # may be '', 'group', or a label
    if r2_on:
        redo_ids = set(pd.read_csv(AUD / "sheets" / f"adjudication_pair_{REDO_ADJ_PAIR}.csv",
                                   dtype=str)["msg_id"])
        for pr in REDO_ADJ_TO:
            f = r2 / "sheets" / f"adjudication_redo_{pr}.csv"
            if not f.exists():
                sys.exit(f"redo adjudication file missing: {f.name}")
            for _, r in read_sheet(f, "agreed_label").iterrows():
                adj[r["msg_id"]] = r["label"]
        for m in redo_ids:
            adj.setdefault(m, "")          # not redone -> unresolved

    # --- full-group decisions for items a pair/the six could not settle --
    group = {}
    gf = AUD / "group_decisions.csv"
    if gf.exists():
        g = read_sheet(gf, "label")
        badg = g[~g["label"].isin(VALID)]
        if len(badg):
            sys.exit(f"group_decisions.csv has {len(badg)} blank/invalid labels.")
        group = dict(zip(g["msg_id"], g["label"]))

    # --- decide per message -------------------------------------------
    final, method, needs = [], [], []
    for _, r in df.iterrows():
        m = r["msg_id"]
        lab, how = None, None
        if m in group:
            lab, how = group[m], "group"
        elif m in adj:
            if adj[m] in VALID:
                lab, how = adj[m], "adjudicated"
            else:
                needs.append((m, "adjudication unresolved (GROUP or blank)", r["text_norm"]))
        elif m in human_ind:
            lab, how = human_ind[m], method_of[m]
        elif m in shared_votes:
            n_exp, need_n, nm = shared_rule[m]
            v = pd.Series(shared_votes[m]).value_counts()
            if len(shared_votes[m]) == n_exp and v.iloc[0] >= need_n:
                lab, how = v.index[0], nm
            else:
                needs.append((m, f"shared item split: {v.to_dict()}", r["text_norm"]))
        elif r["source"] == "self":
            sl = str(r.get("self_label", "") or "")
            if sl in VALID:
                lab, how = sl, "self_pair"
            else:
                needs.append((m, "self-collected message has no pair label", r["text_norm"]))
        elif m in in_audit:
            lab, how = ai[m], "ai"
        elif r["label_source"] == 0:
            lab, how = "legitimate", "source"
        else:
            needs.append((m, "candidate malicious message missing from audit set", r["text_norm"]))
        final.append(lab)
        method.append(how)
    df["label3"], df["label_method"] = final, method

    if needs:
        nd = pd.DataFrame(needs, columns=["msg_id", "reason", "text"])
        nd["label"] = ""
        nd.to_csv(AUD / "needs_group_decision.csv", index=False)
        print(f"{len(nd)} messages need a group decision -> data/audit/needs_group_decision.csv")
        print(nd["reason"].str.split(":").str[0].value_counts().to_string())
        if not args.allow_unresolved:
            sys.exit("Decide them as a group: fill the 'label' column of needs_group_decision.csv,\n"
                     "save it as data/audit/group_decisions.csv, and rerun.")
        df.loc[df["msg_id"].isin(nd["msg_id"]), ["label3", "label_method"]] = ["unclassifiable", "unresolved"]

    out = df[["msg_id", "label3", "label_method"]]
    out.to_csv(AUD / "final_labels.csv", index=False)

    # --- report (these numbers belong in Chapter 4) --------------------
    print(f"\nfinal_labels.csv written: {len(out)} messages")
    print("\nBy method:\n" + df["label_method"].value_counts().to_string())
    print("\nBy final label:\n" + df["label3"].value_counts().to_string())
    pub = df[df["source"] != "self"]
    print("\nSource label vs final (public messages), binary:")
    mal = pub["label3"].isin(["smishing", "spam"])
    print(pd.crosstab(pub["label_source"].map({0: "source=legit", 1: "source=spam"}),
                      mal.map({True: "final=malicious", False: "final=legit/unclass"})).to_string())
    cand = pub[pub["label_source"] == 1]
    print(f"\nOf {len(cand)} source-spam messages: "
          f"{(cand['label3'] == 'smishing').mean():.1%} smishing, "
          f"{(cand['label3'] == 'spam').mean():.1%} spam, "
          f"{(cand['label3'] == 'legitimate').mean():.1%} legitimate, "
          f"{(cand['label3'] == 'unclassifiable').mean():.1%} unclassifiable")
    print("\nNote: most labels rest on the AI pre-annotation; only the human-sample,")
    print("shared and adjudicated messages were human-validated. Report this split.")


if __name__ == "__main__":
    main()
