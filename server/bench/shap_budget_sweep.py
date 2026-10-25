"""SHAP max_evals sweep on VALIDATION messages (design doc, Section 4).

Picks the SHAP evaluation budget the server ships with. It never reads the
test split: rows whose split is not "val" are skipped before their text is
touched (CLAUDE.md rule 1). Outputs hold msg_id and numbers only, never text.

    python -m server.bench.shap_budget_sweep                 # full run, ~2-3 h on an M1
    python -m server.bench.shap_budget_sweep --limit 10 --out results/shap_sweep_smoke

Budgets: 32, 64, 128, 256, 512, against a reference of 2048. Every message
is explained once per budget, with the budgets in a shuffled order per
message (seeded), so CPU heat or background load does not favour one
budget. The explainer is the server's own (ml/explain.py WordExplainer,
same model, batch size 16, 4 torch threads), run on the stored `text_model`,
which is the exact model text the server explains (verified by the
frozen-corpus test in tests/test_textnorm_aligned.py).

SELECTION RULE (fixed on 2026-10-25, before any result was seen)
---------------------------------------------------------------
Choose the SMALLEST budget among 32, 64, 128, 256, 512 for which BOTH hold:
  (a) the median, over validation messages, of the Spearman rank
      correlation between its word weights and the 2048 reference is >= 0.90;
  (b) the 95th percentile of its SHAP latency (explainer call only) is <= 3 s.
If no budget meets both, the rule selects none: the default stays at 256
and the group decides. The script reports the selection; it never changes
the server default (server/config.py), which needs the group's approval.

Pre-declared metric details:
- Units are the whitespace words of text_model (as in the app).
- Spearman uses signed weights, average ranks for ties. Messages with fewer
  than 3 words are excluded from Spearman and top-3 overlap (counted in the
  summary). If the reference weights are all equal the message is excluded;
  if only the candidate's are all equal, its Spearman counts as 0 (a flat
  explanation ranks nothing, and excluding it would flatter small budgets).
- Top-3 overlap: share of the reference's 3 largest |weight| words that are
  also in the candidate's top 3 (ties broken by word position). Reported,
  not part of the rule.
- Deletion faithfulness: rank words by weight toward the PREDICTED class
  (w if p_text >= 0.5, else -w), replace the top k with <mask> (the SHAP
  baseline token, separators kept), and record the drop in the predicted
  class's probability. k = 3 is primary; k = 1 and 5 are also reported.
  Computed only when the message has more than k words. A random-mask
  baseline (mean of 10 random k = 3 masks, seeded by msg_id) is reported
  for scale. Reported, not part of the rule.
- Latency is wall time of WordExplainer.explain() alone; n_evals counts the
  model rows SHAP actually requested, which shows whether the budget binds.
- Reference check: on 25 seeded messages, 2048 is compared with 4096. A
  median Spearman below 0.95 means the reference itself has not converged;
  the summary then marks the selection as inconclusive.

These are MacBook latencies (SOP 5 latency is measured on Cloud Run).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus.csv"
OUT = ROOT / "results" / "shap_sweep"
BUDGETS = (32, 64, 128, 256, 512)
REFERENCE = 2048
REF_CHECK_BUDGET = 4096
REF_CHECK_N = 25
EXPECTED_VAL_N = 381          # docs/PROJECT_CONTEXT.md, split table
SPEARMAN_MIN = 0.90
P95_MAX_S = 3.0
REF_CONVERGED_MIN = 0.95
KS = (1, 3, 5)
SEED = 42

PER_MESSAGE_FIELDS = ["msg_id", "budget", "n_words", "label_bin", "p_text", "base_value",
                      "shap_seconds", "n_evals", "spearman_vs_ref", "top3_overlap",
                      "drop_k1", "drop_k3", "drop_k5", "random_drop_k3"]


# ----------------------------------------------------------------- metrics
def spearman(candidate, reference) -> float | None:
    """None = excluded (fewer than 3 words or flat reference)."""
    from scipy.stats import spearmanr
    c, r = np.asarray(candidate, float), np.asarray(reference, float)
    if len(r) < 3 or np.all(r == r[0]):
        return None
    if np.all(c == c[0]):
        return 0.0
    return float(spearmanr(c, r).statistic)


def top_k(weights, k: int) -> list[int]:
    """Indices of the k largest |weight|, ties broken by word position."""
    order = sorted(range(len(weights)), key=lambda i: (-abs(weights[i]), i))
    return order[:k]


def top3_overlap(candidate, reference) -> float | None:
    if len(reference) < 3:
        return None
    return len(set(top_k(candidate, 3)) & set(top_k(reference, 3))) / 3


def toward_predicted(weights, p_text: float) -> list[float]:
    sign = 1.0 if p_text >= 0.5 else -1.0
    return [sign * w for w in weights]


def predicted_prob(p_text: float, p_ref: float) -> float:
    """Probability of the class predicted for the unmasked text."""
    return p_text if p_ref >= 0.5 else 1.0 - p_text


def select_budget(summary: list[dict]) -> int | None:
    """The pre-declared rule (module docstring). `summary` rows need
    budget, median_spearman and p95_seconds; the reference row is ignored."""
    ok = [r["budget"] for r in summary
          if r["budget"] in BUDGETS
          and r["median_spearman"] is not None and r["median_spearman"] >= SPEARMAN_MIN
          and r["p95_seconds"] <= P95_MAX_S]
    return min(ok) if ok else None


def percentile(xs, q: float) -> float:
    return float(np.percentile(np.asarray(xs, float), q))


# ----------------------------------------------------------------- data
def load_val(path: Path) -> list[dict]:
    """msg_id, text_model and label_bin of validation rows only. Other rows
    are dropped before their text is read into anything."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            rows.append({"msg_id": row["msg_id"], "text_model": row["text_model"],
                         "label_bin": int(row["label_bin"])})
    return rows


def sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), min(n, len(rows))))
    return [rows[i] for i in idx]


# ----------------------------------------------------------------- run
class CountingPredict:
    """Wraps predict_proba to count the model rows SHAP asks for."""

    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, texts):
        self.n += len(texts)
        return self.fn(texts)


def explain_timed(explainer, counter, text):
    counter.n = 0
    t = time.perf_counter()
    ex = explainer.explain(text)
    return ex, time.perf_counter() - t, counter.n


def deletion_drops(engine, masker, text, p_text, weights):
    """Drop in predicted-class probability after masking the top-k words."""
    n = len(weights)
    scores = toward_predicted(weights, p_text)
    texts, keys = [], []
    for k in KS:
        if n > k:
            keep = np.ones(n, dtype=bool)
            keep[sorted(range(n), key=lambda i: (-scores[i], i))[:k]] = False
            texts.append(str(masker(keep, text)[0][0]))
            keys.append(k)
    out = {f"drop_k{k}": None for k in KS}
    if texts:
        p = engine.predict_proba(texts)
        for k, pk in zip(keys, p):
            out[f"drop_k{k}"] = predicted_prob(p_text, p_text) - predicted_prob(float(pk), p_text)
    return out


def random_drop(engine, masker, text, p_text, n_words, msg_id, draws=10, k=3):
    if n_words <= k:
        return None
    rng = random.Random(int(hashlib.sha256(msg_id.encode()).hexdigest()[:8], 16))
    texts = []
    for _ in range(draws):
        keep = np.ones(n_words, dtype=bool)
        keep[rng.sample(range(n_words), k)] = False
        texts.append(str(masker(keep, text)[0][0]))
    p = engine.predict_proba(texts)
    base = predicted_prob(p_text, p_text)
    return float(np.mean([base - predicted_prob(float(x), p_text) for x in p]))


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    counts: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts[row["msg_id"]] = counts.get(row["msg_id"], 0) + 1
    return {m for m, c in counts.items() if c == len(BUDGETS) + 1}


def run_sweep(engine, rows, per_message_path: Path, log=print):
    from ml.explain import WordExplainer, word_spans

    counter = CountingPredict(engine.predict_proba)
    bs = engine.settings.shap_batch_size
    mask_token = engine.tokenizer.mask_token
    explainers = {b: WordExplainer(counter, mask_token, b, bs) for b in BUDGETS + (REFERENCE,)}
    masker = explainers[REFERENCE].masker

    finished = done_ids(per_message_path)
    # A partly written message (interrupted run) is redone from scratch.
    if per_message_path.exists() and finished:
        with open(per_message_path, newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["msg_id"] in finished]
        with open(per_message_path, "w", newline="") as f:
            w = csv.DictWriter(f, PER_MESSAGE_FIELDS)
            w.writeheader()
            w.writerows(kept)
    elif per_message_path.exists():
        per_message_path.unlink()
    todo = [r for r in rows if r["msg_id"] not in finished]
    log(f"{len(rows)} messages, {len(finished)} already done, {len(todo)} to run")

    new = not per_message_path.exists()
    rng = random.Random(SEED)
    t0 = time.perf_counter()
    with open(per_message_path, "a", newline="") as f:
        w = csv.DictWriter(f, PER_MESSAGE_FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(todo, 1):
            text = row["text_model"]
            n_words = len(word_spans(text))
            p_text = float(engine.predict_proba([text])[0])
            order = list(BUDGETS + (REFERENCE,))
            rng.shuffle(order)
            results = {b: explain_timed(explainers[b], counter, text) for b in order}
            ref_w = results[REFERENCE][0].weights
            rnd = random_drop(engine, masker, text, p_text, n_words, row["msg_id"])
            for b in BUDGETS + (REFERENCE,):
                ex, secs, n_evals = results[b]
                w.writerow({"msg_id": row["msg_id"], "budget": b, "n_words": n_words,
                            "label_bin": row["label_bin"], "p_text": p_text,
                            "base_value": ex.base_value, "shap_seconds": secs, "n_evals": n_evals,
                            "spearman_vs_ref": spearman(ex.weights, ref_w),
                            "top3_overlap": top3_overlap(ex.weights, ref_w),
                            **deletion_drops(engine, masker, text, p_text, ex.weights),
                            "random_drop_k3": rnd})
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                log(f"  {i}/{len(todo)} messages, {el/60:.1f} min, ETA {el/i*(len(todo)-i)/60:.0f} min")


def reference_check(engine, rows, path: Path, log=print):
    from ml.explain import WordExplainer
    bs, mt = engine.settings.shap_batch_size, engine.tokenizer.mask_token
    a = WordExplainer(engine.predict_proba, mt, REFERENCE, bs)
    b = WordExplainer(engine.predict_proba, mt, REF_CHECK_BUDGET, bs)
    out = []
    for row in sample(rows, REF_CHECK_N, SEED + 1):
        rho = spearman(a.explain(row["text_model"]).weights, b.explain(row["text_model"]).weights)
        out.append({"msg_id": row["msg_id"], "spearman_2048_vs_4096": rho})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, ["msg_id", "spearman_2048_vs_4096"])
        w.writeheader()
        w.writerows(out)
    vals = [r["spearman_2048_vs_4096"] for r in out if r["spearman_2048_vs_4096"] is not None]
    med = statistics.median(vals) if vals else None
    log(f"reference check: median Spearman 2048 vs 4096 = {med} (n = {len(vals)})")
    return med, len(vals)


def _num(x):
    return None if x in ("", None) else float(x)


def summarize(per_message_path: Path) -> list[dict]:
    with open(per_message_path, newline="") as f:
        rows = list(csv.DictReader(f))
    summary = []
    for b in BUDGETS + (REFERENCE,):
        rs = [r for r in rows if int(r["budget"]) == b]
        rho = [v for v in (_num(r["spearman_vs_ref"]) for r in rs) if v is not None]
        ov = [v for v in (_num(r["top3_overlap"]) for r in rs) if v is not None]
        secs = [float(r["shap_seconds"]) for r in rs]
        rec = {"budget": b, "n_messages": len(rs), "n_spearman": len(rho),
               "median_spearman": statistics.median(rho) if rho else None,
               "q1_spearman": percentile(rho, 25) if rho else None,
               "q3_spearman": percentile(rho, 75) if rho else None,
               "share_spearman_ge_0_9": sum(v >= SPEARMAN_MIN for v in rho) / len(rho) if rho else None,
               "mean_top3_overlap": statistics.mean(ov) if ov else None}
        for k in KS:
            d = [v for v in (_num(r[f"drop_k{k}"]) for r in rs) if v is not None]
            rec[f"n_drop_k{k}"] = len(d)
            rec[f"median_drop_k{k}"] = statistics.median(d) if d else None
            rec[f"mean_drop_k{k}"] = statistics.mean(d) if d else None
        rnd = [v for v in (_num(r["random_drop_k3"]) for r in rs) if v is not None]
        rec["mean_random_drop_k3"] = statistics.mean(rnd) if rnd else None
        rec.update({"median_seconds": statistics.median(secs), "p95_seconds": percentile(secs, 95),
                    "max_seconds": max(secs),
                    "median_n_evals": statistics.median(int(r["n_evals"]) for r in rs)})
        summary.append(rec)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--limit", type=int, default=None, help="seeded subsample (smoke runs)")
    ap.add_argument("--expected-n", type=int, default=EXPECTED_VAL_N)
    ap.add_argument("--skip-ref-check", action="store_true")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)   # progress shows up in a redirected log

    from server.config import Settings
    from server.app.engine import RealEngine
    from server.app.timing import StageTimer

    rows = load_val(args.corpus)
    if len(rows) != args.expected_n:
        sys.exit(f"expected {args.expected_n} validation rows, found {len(rows)}; stopping")
    if args.limit:
        rows = sample(rows, args.limit, SEED)
    args.out.mkdir(parents=True, exist_ok=True)

    settings = Settings.from_env()
    engine = RealEngine(settings)
    engine.warm_up()
    engine.analyze("Warm-up three, check this out: example.com", True, StageTimer())

    run_sweep(engine, rows, args.out / "per_message.csv")
    ref_med, ref_n = (None, 0) if args.skip_ref_check else \
        reference_check(engine, rows, args.out / "reference_check.csv")

    summary = summarize(args.out / "per_message.csv")
    chosen = select_budget(summary)
    converged = ref_med is not None and ref_med >= REF_CONVERGED_MIN
    for r in summary:
        r["meets_spearman"] = r["median_spearman"] is not None and r["median_spearman"] >= SPEARMAN_MIN
        r["meets_latency"] = r["p95_seconds"] <= P95_MAX_S
        r["selected"] = r["budget"] == chosen
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    meta = engine.meta()
    run = {"date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "split": "val",
           "n_messages": len(rows), "limit": args.limit, "budgets": list(BUDGETS),
           "reference": REFERENCE, "rule": {"spearman_min": SPEARMAN_MIN, "p95_max_s": P95_MAX_S},
           "selected_budget": chosen, "reference_check_median": ref_med, "reference_check_n": ref_n,
           "reference_converged": converged, "server_default_max_evals": settings.shap_max_evals,
           "machine": platform.platform(), "cpu_count": os.cpu_count(),
           "model_version": meta["model_version"], "weights_sha256": meta["weights_sha256"],
           "textnorm_sha256": meta["textnorm_sha256"], "versions": meta["versions"],
           "torch_threads": settings.torch_threads, "batch_size": settings.shap_batch_size}
    (args.out / "run.json").write_text(json.dumps(run, indent=2))

    print("\nbudget  median_rho  top3  drop_k3  rand_k3  lat_med  lat_p95  evals")
    for r in summary:
        f3 = lambda x: "  n/a " if x is None else f"{x:6.3f}"
        print(f"{r['budget']:6d}  {f3(r['median_spearman'])}    {f3(r['mean_top3_overlap'])} "
              f"{f3(r['median_drop_k3'])}   {f3(r['mean_random_drop_k3'])}  "
              f"{r['median_seconds']:6.2f}s  {r['p95_seconds']:6.2f}s  {r['median_n_evals']:5.0f}")
    print(f"\nRule (fixed before results): smallest budget with median Spearman >= {SPEARMAN_MIN} "
          f"and p95 <= {P95_MAX_S} s")
    print(f"SELECTED: {chosen if chosen else 'none (default stays at 256; group decides)'}")
    if not args.skip_ref_check and not converged:
        print(f"WARNING: reference check median {ref_med} < {REF_CONVERGED_MIN}: "
              "the 2048 reference may not have converged; treat the selection as inconclusive.")
    print(f"Server default unchanged (max_evals = {settings.shap_max_evals}). Outputs in {args.out}")


if __name__ == "__main__":
    main()
