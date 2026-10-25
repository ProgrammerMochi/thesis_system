"""Train the lexical URL model on TRAIN and tune the text + URL fusion on
VALIDATION (Chapter 3, Sections 3.2.3 and 3.2.4). Prototype for the title
defense; it never reads the test split.

    python -m ml.train_url_fusion            # writes models/url_fusion_v0/, results/url_fusion/

RULES (fixed on 2026-10-25, committed before any validation result was computed)

Labels. Each link takes the binary label (label_bin) of the message it came
from. This is PROVISIONAL: Chapter 3 specifies URL-level ground truth (Google
Safe Browsing, a threat feed, WHOIS domain age, a manual check), which does
not exist yet. A harmless link inside a scam message is therefore labelled
malicious here.

URL model. Logistic regression on ml/url_model.feature_vector(): 20 lexical
features plus one indicator per public suffix seen in at least 5 distinct
training links ("tld=other" for the rest). Features are standardized with
training means and standard deviations. Hyperparameters are FIXED, not tuned:
L2 penalty, C = 1.0, class_weight = "balanced", lbfgs, max_iter = 5000.
Training rows are the distinct (link, label) pairs of the TRAIN split, so a
link repeated across many copies of a template counts once per label.
Nothing is visited or looked up; shortened links are not expanded.

Message score. p_url = the highest p_url among the message's links; a message
with no link has no p_url and is scored by p_text alone.
Fusion: S = w * p_text + (1 - w) * p_url.

Tuning on VALIDATION (n = 381).
  w grid:          0.00, 0.05, ..., 1.00 (21 values)
  threshold grid:  0.05, 0.06, ..., 0.95 (91 values); malicious if S >= t
  criterion:       maximum F1 with malicious as the positive class
  tie-break:       larger w first (closer to text-only), then the threshold
                   closest to 0.50, then the smaller threshold
The same threshold grid, criterion and tie-break pick the threshold of the
text-only comparison ("text-only, tuned threshold"), so the fused model is
compared with a text model that had the same chance to tune its cut-off.

Reported (all on validation, all OPTIMISTIC because w and the thresholds were
chosen on the same messages): F1, precision, recall and errors for
text-only at 0.50, text-only with its tuned threshold, and fused; an exact
McNemar test of fused vs. tuned text-only; recall on smishing and on
malicious messages without a link; false positives among legitimate messages
with a link; and the URL model alone on validation messages that have a link
(ROC AUC, F1 at 0.50).

Outputs: models/url_fusion_v0/url_model.json and fusion.json (read by the
server), results/url_fusion/val_predictions.csv (msg_id and numbers only)
and results/url_fusion/report.json. No message text or link text is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.url_model import UrlModel, feature_vector, fuse, message_p_url, tld_of  # noqa: E402

EXPECTED = {"train": 1786, "val": 381}   # docs/PROJECT_CONTEXT.md, split table
W_GRID = [round(0.05 * i, 2) for i in range(21)]
T_GRID = [round(0.05 + 0.01 * i, 2) for i in range(91)]
MIN_TLD_COUNT = 5
LR_PARAMS = {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 5000}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(corpus: Path) -> dict[str, list[dict]]:
    """Train and val rows only. Test rows are skipped before any field other
    than `split` is used."""
    csv.field_size_limit(10**9)
    out = {"train": [], "val": []}
    with open(corpus, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] not in out:
                continue
            out[row["split"]].append({
                "msg_id": row["msg_id"], "text_model": row["text_model"],
                "urls": row["urls"].split(" ") if row["urls"] else [],
                "label_bin": int(row["label_bin"]), "label3": row["label3"],
            })
    for split, n in EXPECTED.items():
        if len(out[split]) != n:
            raise SystemExit(f"{split}: expected {n} rows, found {len(out[split])}")
    return out


def train_url_model(rows: list[dict]) -> tuple[dict, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    pairs = sorted({(u, r["label_bin"]) for r in rows for u in r["urls"]})
    tld_counts = Counter(tld_of(u) for u in {u for u, _ in pairs})
    vocab = sorted(t for t, c in tld_counts.items() if t and c >= MIN_TLD_COUNT)
    names, _ = feature_vector("example.com", vocab)
    X = np.array([feature_vector(u, vocab)[1] for u, _ in pairs])
    y = np.array([lab for _, lab in pairs])
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(**LR_PARAMS).fit(scaler.transform(X), y)
    spec = {
        "kind": "logistic_regression", "feature_names": names, "tld_vocab": vocab,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "params": LR_PARAMS, "min_tld_count": MIN_TLD_COUNT,
        "label_source": "message label_bin (provisional; not URL-level ground truth)",
    }
    info = {"train_link_label_pairs": len(pairs), "malicious_pairs": int(y.sum()),
            "legitimate_pairs": int(len(y) - y.sum()), "tld_vocab_size": len(vocab)}
    return spec, info


def text_probs(rows: list[dict], batch: int = 16) -> np.ndarray:
    """p_text from the exact model and preprocessing the server uses."""
    from server.app.engine import RealEngine
    from server.config import Settings
    engine = RealEngine(Settings(warmup=False))
    out = []
    for i in range(0, len(rows), batch):
        out.extend(engine.predict_proba([r["text_model"] for r in rows[i:i + batch]]).tolist())
    return np.array(out)


def metrics(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": t, "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "errors": fp + fn}


def pick(candidates: list[tuple[float, float, dict]]) -> tuple[float, float, dict]:
    """(w, t, metrics) with the max F1; ties: larger w, t closest to 0.5, smaller t."""
    return min(candidates, key=lambda c: (-c[2]["f1"], -c[0], abs(c[1] - 0.5), c[1]))


def subset_recall(y, pred, mask) -> dict:
    m = mask & (y == 1)
    return {"n": int(m.sum()), "caught": int((pred[m] == 1).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=REPO / "data" / "processed" / "corpus.csv")
    ap.add_argument("--model-out", type=Path, default=REPO / "models" / "url_fusion_v0")
    ap.add_argument("--results", type=Path, default=REPO / "results" / "url_fusion")
    args = ap.parse_args()

    rows = load_rows(args.corpus)
    spec, train_info = train_url_model(rows["train"])
    url_model = UrlModel(spec)

    val = rows["val"]
    y = np.array([r["label_bin"] for r in val])
    has_url = np.array([bool(r["urls"]) for r in val])
    smishing = np.array([r["label3"] == "smishing" for r in val])
    print(f"scoring {len(val)} validation messages with the text model ...", flush=True)
    p_text = text_probs(val)
    p_url = [message_p_url(url_model, r["urls"]) for r in val]

    text_05 = metrics(y, p_text, 0.5)
    _, t_text, text_tuned = pick([(1.0, t, metrics(y, p_text, t)) for t in T_GRID])
    cands = []
    for w in W_GRID:
        s = np.array([fuse(pt, pu, w) for pt, pu in zip(p_text, p_url)])
        cands.extend((w, t, metrics(y, s, t)) for t in T_GRID)
    w_best, t_best, fused = pick(cands)
    S = np.array([fuse(pt, pu, w_best) for pt, pu in zip(p_text, p_url)])

    from scipy.stats import binomtest
    from sklearn.metrics import roc_auc_score
    pred_text = (p_text >= t_text).astype(int)
    pred_fused = (S >= t_best).astype(int)
    b = int(((pred_text == y) & (pred_fused != y)).sum())   # fusion broke it
    c = int(((pred_text != y) & (pred_fused == y)).sum())   # fusion fixed it
    mcnemar_p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0

    pu_arr = np.array([p if p is not None else np.nan for p in p_url])
    yu, su = y[has_url], pu_arr[has_url]
    url_only = {"n": int(has_url.sum()), "n_legitimate": int((yu == 0).sum()),
                "roc_auc": round(float(roc_auc_score(yu, su)), 4) if len(set(yu)) == 2 else None,
                **{k: v for k, v in metrics(yu, su, 0.5).items() if k != "threshold"}}

    def breakdown(pred):
        return {"smishing_recall": subset_recall(y, pred, smishing),
                "no_url_malicious_recall": subset_recall(y, pred, ~has_url),
                "legitimate_with_url_fp": {"n": int((has_url & (y == 0)).sum()),
                                           "flagged": int((pred[has_url & (y == 0)] == 1).sum())}}

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "split": "val", "n": len(val),
        "optimistic": "w and thresholds were tuned on these same validation messages",
        "label_source": spec["label_source"], "url_model_train": train_info,
        "text_only_0.5": {**text_05, **breakdown((p_text >= 0.5).astype(int))},
        "text_only_tuned": {**text_tuned, **breakdown(pred_text)},
        "fused": {"w": w_best, **fused, **breakdown(pred_fused)},
        "mcnemar_fused_vs_text_tuned": {"text_right_fused_wrong": b, "text_wrong_fused_right": c,
                                        "exact_p": round(float(mcnemar_p), 4)},
        "url_model_alone_on_val_links": url_only,
        "corpus_sha256": sha256(args.corpus),
        "textnorm_sha256": sha256(REPO / "ml" / "textnorm.py"),
        "url_model_py_sha256": sha256(REPO / "ml" / "url_model.py"),
    }

    args.model_out.mkdir(parents=True, exist_ok=True)
    (args.model_out / "url_model.json").write_text(json.dumps(spec, indent=1))
    fusion = {"w": w_best, "threshold": t_best, "text_model": "roberta_weighted_seed42",
              "basis": "fusion_prototype_val_tuned_message_labels", "rule": "ml/train_url_fusion.py docstring",
              "validation": {k: report[k] for k in ("text_only_0.5", "text_only_tuned", "fused")}}
    (args.model_out / "fusion.json").write_text(json.dumps(fusion, indent=1))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "report.json").write_text(json.dumps(report, indent=1))
    with open(args.results / "val_predictions.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["msg_id", "label_bin", "label3", "has_url", "p_text", "p_url", "S"])
        for r, pt, pu, s in zip(val, p_text, p_url, S):
            wr.writerow([r["msg_id"], r["label_bin"], r["label3"], int(bool(r["urls"])), round(float(pt), 6),
                         "" if pu is None else round(pu, 6), round(float(s), 6)])

    def line(name, m):
        return (f"{name:22s} t={m['threshold']:.2f}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                f"R={m['recall']:.4f}  FP={m['fp']}  FN={m['fn']}  errors={m['errors']}")
    print(f"\nURL model: {train_info}")
    print(f"URL model alone, val messages with a link: {url_only}")
    print("\nVALIDATION (optimistic: tuned on val)")
    print(line("text-only @0.50", text_05))
    print(line("text-only tuned t", text_tuned))
    print(line(f"fused w={w_best:.2f}", fused))
    print(f"McNemar fused vs tuned text: fusion fixed {c}, broke {b}, exact p = {mcnemar_p:.4f}")
    print(f"Wrote {args.model_out} and {args.results}")


if __name__ == "__main__":
    main()
