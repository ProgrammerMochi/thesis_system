"""The SHAP budget sweep: metric helpers, the pre-declared selection rule,
the validation-only loader, and one tiny end-to-end run."""
import csv

import pytest

from server.bench import shap_budget_sweep as sweep

CANARY = "CANARY test split text must never be read"


def test_spearman_rules():
    assert sweep.spearman([0.3, 0.2, 0.1], [0.3, 0.2, 0.1]) == pytest.approx(1.0)
    assert sweep.spearman([0.1, 0.2], [0.2, 0.1]) is None            # fewer than 3 words
    assert sweep.spearman([0.1, 0.2, 0.3], [0.5, 0.5, 0.5]) is None  # flat reference
    assert sweep.spearman([0.2, 0.2, 0.2], [0.1, 0.2, 0.3]) == 0.0   # flat candidate


def test_top3_overlap_uses_abs_weight_and_position_ties():
    ref = [0.5, -0.4, 0.3, 0.01]
    assert sweep.top3_overlap([-0.5, 0.4, 0.3, 0.0], ref) == 1.0
    assert sweep.top3_overlap([0.0, 0.0, 0.0, 0.9], ref) == pytest.approx(2 / 3)  # ties -> idx 0, 1
    assert sweep.top3_overlap([0.1, 0.2], [0.2, 0.1]) is None


def test_predicted_class_direction():
    assert sweep.toward_predicted([0.2, -0.1], 0.9) == [0.2, -0.1]
    assert sweep.toward_predicted([0.2, -0.1], 0.1) == [-0.2, 0.1]
    assert sweep.predicted_prob(0.3, 0.1) == pytest.approx(0.7)


def _row(b, rho, p95):
    return {"budget": b, "median_spearman": rho, "p95_seconds": p95}


def test_selection_rule():
    rows = [_row(32, 0.70, 0.4), _row(64, 0.91, 0.8), _row(128, 0.95, 1.5),
            _row(256, 0.99, 4.6), _row(512, 0.99, 9.0), _row(2048, 1.0, 30.0)]
    assert sweep.select_budget(rows) == 64
    rows[1]["p95_seconds"] = 3.01                        # latency fails at 64
    assert sweep.select_budget(rows) == 128
    assert sweep.select_budget([_row(32, 0.9, 3.0)]) == 32  # both bounds inclusive
    assert sweep.select_budget([_row(32, 0.89, 1.0), _row(64, None, 1.0), _row(256, 0.99, 3.5)]) is None
    assert sweep.select_budget([_row(2048, 1.0, 0.1)]) is None   # the reference is never a candidate


def _corpus(path, n_val=4):
    cols = ["msg_id", "text_model", "label_bin", "split"]
    texts = ["Your account is locked. Verify now at [LINK] to avoid suspension.",
             "Hi Ma, nakarating na ako sa bahay. Kain na tayo mamaya.",
             "Congratulations! Ikaw ay nanalo ng [AMOUNT]. I-claim na sa [LINK]",
             "Tol, tuloy pa ba tayo bukas? Dala ka ng bola.", "Ok"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n_val):
            w.writerow([f"v{i}", texts[i % len(texts)], i % 2, "val"])
        w.writerow(["t0", CANARY, 1, "test"])
        w.writerow(["r0", "a train message", 0, "train"])


def test_loader_reads_validation_only(tmp_path):
    p = tmp_path / "corpus.csv"
    _corpus(p)
    rows = sweep.load_val(p)
    assert [r["msg_id"] for r in rows] == ["v0", "v1", "v2", "v3"]
    assert all(CANARY not in r["text_model"] for r in rows)


def test_end_to_end_tiny_model(tmp_path, tiny_model_dir, monkeypatch, capsys):
    monkeypatch.setenv("SMISHING_MODEL_DIR", str(tiny_model_dir))
    corpus, out = tmp_path / "corpus.csv", tmp_path / "out"
    _corpus(corpus, n_val=5)
    sweep.main(["--corpus", str(corpus), "--out", str(out), "--expected-n", "5", "--skip-ref-check"])

    per = list(csv.DictReader(open(out / "per_message.csv")))
    assert len(per) == 5 * (len(sweep.BUDGETS) + 1)
    assert {r["msg_id"] for r in per} == {f"v{i}" for i in range(5)}
    assert set(per[0]) == set(sweep.PER_MESSAGE_FIELDS)          # numbers and msg_id only
    ref = [r for r in per if r["budget"] == str(sweep.REFERENCE) and r["spearman_vs_ref"]]
    assert all(float(r["spearman_vs_ref"]) == pytest.approx(1.0) for r in ref)
    assert all(int(r["n_evals"]) <= int(r["budget"]) + 2 for r in per)

    summary = list(csv.DictReader(open(out / "summary.csv")))
    assert [int(r["budget"]) for r in summary] == list(sweep.BUDGETS) + [sweep.REFERENCE]
    for f in out.iterdir():
        assert CANARY not in f.read_text() and "Verify now" not in f.read_text()
    assert "Server default unchanged" in capsys.readouterr().out

    # Resuming skips finished messages and keeps the file consistent.
    sweep.main(["--corpus", str(corpus), "--out", str(out), "--expected-n", "5", "--skip-ref-check"])
    assert len(list(csv.DictReader(open(out / "per_message.csv")))) == len(per)


def test_wrong_val_count_stops(tmp_path):
    p = tmp_path / "corpus.csv"
    _corpus(p)
    with pytest.raises(SystemExit):
        sweep.main(["--corpus", str(p), "--out", str(tmp_path / "o")])
