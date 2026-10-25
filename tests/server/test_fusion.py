"""Text + URL fusion in REAL mode, with the tiny random text model (conftest.py)
and a hand-made URL model, so nothing depends on local model files. Checks
plumbing: S = w * p_text + (1 - w) * p_url, link offsets, fallbacks."""
import json

import pytest
from fastapi.testclient import TestClient

from ml.url_model import feature_vector
from server.app.main import create_app
from server.app.schemas import AnalyzeResponse
from server.config import Settings


@pytest.fixture(scope="module")
def fusion_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fusion")
    vocab = ["top"]
    names, _ = feature_vector("example.com", vocab)
    coef = [3.0 if n == "tld=top" else (-2.0 if n == "is_https" else 0.0) for n in names]
    (d / "url_model.json").write_text(json.dumps({
        "tld_vocab": vocab, "feature_names": names, "mean": [0.0] * len(names),
        "scale": [1.0] * len(names), "coef": coef, "intercept": -1.0, "label_source": "test"}))
    (d / "fusion.json").write_text(json.dumps({"w": 0.5, "threshold": 0.3}))
    return d


def _client(tiny_model_dir, **kw):
    return TestClient(create_app(Settings(mode="real", model_dir=tiny_model_dir, warmup=False,
                                          shap_max_evals=64, torch_threads=1, **kw)))


def _u16_slice(text, a, b):
    return text.encode("utf-16-le")[2 * a:2 * b].decode("utf-16-le")


def test_fused_score_and_link_offsets(tiny_model_dir, fusion_dir):
    text = "😀 Verify at evil-site.top/login. Also https://safe.example.com now"
    with _client(tiny_model_dir, fusion_dir=fusion_dir) as c:
        body = c.post("/v1/analyze?explain=false", json={"text": text}).json()
    r = AnalyzeResponse.model_validate(body)
    assert r.scores.w == 0.5 and r.scores.p_url is not None
    assert r.threat_probability == pytest.approx(0.5 * r.scores.p_text + 0.5 * r.scores.p_url, abs=1e-3)
    assert r.scores.fused == r.threat_probability
    assert r.decision.threshold == 0.5 and "threshold_0.5_provisional" in r.decision.basis
    assert [(_u16_slice(text, a.start, a.end)) for a in r.url_analysis] == [
        "evil-site.top/login", "https://safe.example.com"]
    risky, safe = r.url_analysis
    assert risky.p_url > safe.p_url and r.scores.p_url == risky.p_url
    assert risky.top_features[0].feature == "tld=top"
    # the response carries offsets, never the link text
    assert "evil-site" not in json.dumps(body)


def test_no_link_uses_p_text_alone(tiny_model_dir, fusion_dir):
    with _client(tiny_model_dir, fusion_dir=fusion_dir) as c:
        r = c.post("/v1/analyze?explain=false", json={"text": "Hi Ma, kain na tayo mamaya."}).json()
    assert r["scores"]["p_url"] is None and r["url_analysis"] == []
    assert r["threat_probability"] == r["scores"]["p_text"]


def test_tuned_threshold_only_when_enabled(tiny_model_dir, fusion_dir):
    with _client(tiny_model_dir, fusion_dir=fusion_dir, use_tuned_threshold=True) as c:
        r = c.post("/v1/analyze?explain=false", json={"text": "visit evil.top"}).json()
        meta = c.get("/v1/meta").json()
    assert r["decision"]["threshold"] == 0.3 and "threshold_val_tuned" in r["decision"]["basis"]
    assert r["verdict"] == ("malicious" if r["threat_probability"] >= 0.3 else "legitimate")
    assert meta["fusion"]["w"] == 0.5 and meta["threshold"] == 0.3


def test_missing_files_fall_back_to_text_only(tiny_model_dir, tmp_path):
    for d in (None, tmp_path):
        with _client(tiny_model_dir, fusion_dir=d) as c:
            r = c.post("/v1/analyze?explain=false", json={"text": "visit evil.top now"}).json()
        assert r["scores"]["p_url"] is None and r["scores"]["w"] is None and r["url_analysis"] == []
        assert r["decision"]["basis"] == "p_text_only_provisional"
        assert r["threat_probability"] == r["scores"]["p_text"]
