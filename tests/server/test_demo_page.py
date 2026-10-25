"""The /demo browser page: served, self-contained, and switchable off."""
import re

from fastapi.testclient import TestClient

from server.app.main import create_app
from server.config import Settings


def test_demo_page_served_and_root_redirects():
    with TestClient(create_app(Settings(mode="stub"))) as c:
        r = c.get("/demo")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "/v1/analyze?explain=" in r.text
        assert c.get("/", follow_redirects=False).headers["location"] == "/demo"


def test_demo_page_loads_nothing_external():
    """Must work over a phone hotspot with no internet."""
    with TestClient(create_app(Settings(mode="stub"))) as c:
        html = c.get("/demo").text
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=", html)
    assert "localStorage" not in html and "document.cookie" not in html


def test_demo_page_can_be_disabled():
    with TestClient(create_app(Settings(mode="stub", demo_page=False))) as c:
        assert c.get("/demo").status_code == 404
        assert c.get("/", follow_redirects=False).status_code == 404
