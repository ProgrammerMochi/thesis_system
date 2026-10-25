"""STUB mode: the contract the Android client is built against."""
from fastapi.testclient import TestClient

from server.app.main import create_app
from server.app.schemas import AnalyzeResponse
from server.config import Settings


def client():
    return TestClient(create_app(Settings(mode="stub")))


def test_health_ready_meta():
    with client() as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/readyz").status_code == 200
        assert c.get("/v1/meta").json()["mode"] == "stub"


def test_stub_response_is_valid_figure3():
    text = "😀 Verify now at bit.ly/x"
    with client() as c:
        r = c.post("/v1/analyze", json={"text": text})
    assert r.status_code == 200
    body = AnalyzeResponse.model_validate(r.json())
    assert body.verdict == "malicious" and body.threat_probability == 0.98
    assert body.decision.basis == "stub_fixed_response"
    utf16_len = len(text.encode("utf-16-le")) // 2
    for a in body.text_attributions:
        assert 0 <= a.start < a.end <= utf16_len
    # first word is the emoji: 2 UTF-16 units
    assert (body.text_attributions[0].start, body.text_attributions[0].end) == (0, 2)
    assert "total;dur=" in r.headers["server-timing"]


def test_stub_single_word_and_explain_false():
    with client() as c:
        one = c.post("/v1/analyze", json={"text": "Hello"}).json()
        none = c.post("/v1/analyze?explain=false", json={"text": "Hello there"}).json()
    assert len(one["text_attributions"]) == 1
    assert none["text_attributions"] == [] and none["explanation"]["computed"] is False


def test_rejects_extra_fields_and_empty_text():
    with client() as c:
        r = c.post("/v1/analyze", json={"text": "hi", "sender": "BDO"})
        assert r.status_code == 422 and r.json()["errors"][0]["type"] == "extra_forbidden"
        assert c.post("/v1/analyze", json={"text": ""}).status_code == 422
