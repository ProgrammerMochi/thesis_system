"""The two-call contract (server/README.md): for the same text,
explain=false (call 1, verdict) and explain=true (call 2, highlights) must
agree on everything except the explanation."""
import pytest
from fastapi.testclient import TestClient

from server.app.main import create_app
from server.config import Settings

SAME = ("verdict", "threat_probability", "scores", "decision", "url_flags", "model_version")
TEXTS = [
    "Your account is locked. Verify now at bit.ly/abc123 to avoid suspension.",
    "Hi Ma, nakarating na ako sa bahay. Kain na tayo mamaya.",
    "😀 Login at http://[secure].bdo.com now, OTP is 482910",
    "Promo ngayong linggo lang! Visit www.example-shop.ph\n\nThanks po.",
]


@pytest.fixture(scope="module", params=["stub", "real"])
def client(request, tiny_model_dir):
    settings = (Settings(mode="stub") if request.param == "stub" else
                Settings(mode="real", model_dir=tiny_model_dir, warmup=False,
                         shap_max_evals=64, torch_threads=1))
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.mark.parametrize("text", TEXTS)
def test_verdict_identical_with_and_without_explain(client, text):
    first = client.post("/v1/analyze?explain=false", json={"text": text})
    second = client.post("/v1/analyze", json={"text": text})
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    for key in SAME:
        assert a[key] == b[key], key
    assert a["text_attributions"] == [] and a["explanation"]["computed"] is False
    assert b["explanation"]["computed"] is True
    assert "shap;dur=" not in first.headers["server-timing"]
