"""REAL mode plumbing on a tiny random model (see conftest.py): textnorm,
classifier, SHAP and offset mapping. Not an accuracy test."""
import pytest
from fastapi.testclient import TestClient

from server.app.main import create_app
from server.app.schemas import AnalyzeResponse
from server.config import Settings


@pytest.fixture(scope="module")
def client(tiny_model_dir):
    app = create_app(Settings(mode="real", model_dir=tiny_model_dir, warmup=False,
                              shap_max_evals=64, torch_threads=1))
    with TestClient(app) as c:
        yield c


def _u16(s):
    return len(s.encode("utf-16-le")) // 2


def test_valid_response_and_additivity(client):
    text = "Hi  Ma, visit\nbit.ly/abc123 ngayon! Call 0917-123-4567"
    body = AnalyzeResponse.model_validate(client.post("/v1/analyze", json={"text": text}).json())
    assert body.verdict == ("malicious" if body.threat_probability >= 0.5 else "legitimate")
    assert body.explanation.computed and not body.explanation.truncated
    total = body.explanation.base_value + sum(a.weight for a in body.text_attributions)
    assert abs(total - body.scores.p_text) < 0.01
    assert "shortener_domain" in body.url_flags


def test_offsets_point_into_sent_text_after_emoji(client):
    text = "😀😀 Click  https://shorten.is/BDO_Update now."
    r = client.post("/v1/analyze", json={"text": text}).json()
    units = text.encode("utf-16-le")
    spans = {units[2 * a["start"]:2 * a["end"]].decode("utf-16-le") for a in r["text_attributions"]}
    assert all(0 <= a["start"] < a["end"] <= _u16(text) for a in r["text_attributions"])
    # every reported span is an exact substring of what the phone sent
    assert spans <= {"😀😀", "Click", "https://shorten.is/BDO_Update", "now."}


def test_deterministic(client):
    t = "Your account is locked. Verify now at x.icu/login"
    a = client.post("/v1/analyze", json={"text": t}).json()
    b = client.post("/v1/analyze", json={"text": t}).json()
    assert a == b


def test_truncated_flag(client):
    long_text = " ".join(["salita"] * 400)
    r = client.post("/v1/analyze?explain=false", json={"text": long_text[:1600]}).json()
    assert r["explanation"]["truncated"] is True and r["text_attributions"] == []


def test_alignment_guard(client, monkeypatch):
    import server.app.engine as eng
    monkeypatch.setattr(eng, "to_model_text_aligned", lambda t: ("different", [], [], True))
    r = client.post("/v1/analyze", json={"text": "Hello there friend"}).json()
    assert r["text_attributions"] == [] and r["explanation"]["computed"] is False
    assert client.get("/v1/meta").json()["alignment_mismatches"] >= 1


def test_server_timing_has_stages(client):
    r = client.post("/v1/analyze", json={"text": "Kain na tayo"})
    for stage in ("queue", "preprocess", "classify", "shap", "map", "total"):
        assert f"{stage};dur=" in r.headers["server-timing"]


def test_nfc_fallback_omits_attributions(client):
    r = client.post("/v1/analyze", json={"text": "Annyeong \u1100\u1161 friend"}).json()
    assert r["text_attributions"] == [] and r["explanation"]["computed"] is False
    assert client.get("/v1/meta").json()["nfc_fallbacks"] >= 1


def test_malformed_url_does_not_500(client):
    r = client.post("/v1/analyze", json={"text": "Login at http://[secure].bdo.com now"})
    assert r.status_code == 200 and "malformed_url" in r.json()["url_flags"]


@pytest.mark.parametrize("text", ["Ok", "  Ok!  ", "https://bit.ly/x", "Ok po", "Salamat po Ma"])
def test_very_short_messages_get_highlights(client, text):
    """shap's text clustering fails on one word; the explainer computes that
    case exactly (f(text) - f(<mask>)) instead of returning a 500."""
    r = client.post("/v1/analyze", json={"text": text})
    assert r.status_code == 200
    body = AnalyzeResponse.model_validate(r.json())
    assert body.explanation.computed
    total = body.explanation.base_value + sum(a.weight for a in body.text_attributions)
    assert abs(total - body.scores.p_text) < 0.01
