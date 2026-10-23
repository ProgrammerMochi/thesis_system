"""Zero retention (Chapter 3, Section 3.5.3): message text must never appear
in a response body, a header, or any log line, for normal AND failing
requests."""
import logging

import pytest
from fastapi.testclient import TestClient

from server.app.main import create_app
from server.config import Settings

CANARY = "ZqCANARY7x41"


def _assert_absent(responses, caplog, capfd):
    for r in responses:
        assert CANARY not in r.text
        assert all(CANARY not in v for v in r.headers.values())
    out, err = capfd.readouterr()
    assert CANARY not in out and CANARY not in err
    assert CANARY not in caplog.text


@pytest.fixture
def app():
    return create_app(Settings(mode="stub", max_chars=50))


def test_canary_never_echoed(app, caplog, capfd):
    caplog.set_level(logging.DEBUG)
    with TestClient(app) as c:
        rs = [
            c.post("/v1/analyze", json={"text": f"ok {CANARY}"}),                 # valid
            c.post("/v1/analyze", json={"text": CANARY * 10}),                    # too long
            c.post("/v1/analyze", json={"text": [CANARY]}),                       # wrong type
            c.post("/v1/analyze", json={"text": "x", "sender": CANARY}),          # extra field
            c.post("/v1/analyze", content=f'{{"text": "{CANARY}',                 # malformed JSON
                   headers={"content-type": "application/json"}),
            c.post("/v1/analyze", content=CANARY * 7000,                          # oversized body
                   headers={"content-type": "application/json"}),
        ]
    assert [r.status_code for r in rs] == [200, 422, 422, 422, 422, 413]
    _assert_absent(rs, caplog, capfd)


def test_canary_not_in_500(app, caplog, capfd, monkeypatch):
    caplog.set_level(logging.DEBUG)

    def boom(text, explain, timer):
        raise ValueError(f"failed on {text}")   # an exception message that quotes the text

    with TestClient(app) as c:
        monkeypatch.setattr(c.app.state.engine, "analyze", boom)
        r = c.post("/v1/analyze", json={"text": CANARY})
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    _assert_absent([r], caplog, capfd)
