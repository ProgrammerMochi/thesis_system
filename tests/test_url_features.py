"""Lexical URL flags. Run: python -m pytest tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from url_features import message_url_flags, url_flags  # noqa: E402


def test_flags():
    assert url_flags("https://bit.ly/abc") == ["shortener_domain"]
    assert url_flags("bit.ly/abc") == ["shortener_domain", "no_https"]
    assert url_flags("http://192.168.1.5:8080/login") == ["ip_host", "no_https", "nonstandard_port"]
    assert url_flags("https://xn--bdo-xyz.com/") == ["punycode_host"]
    assert url_flags("https://bdo.com.ph@evil.icu/x") == ["at_in_url"]
    assert url_flags("https://www.bdo.com.ph/") == []


def test_message_union_and_empty():
    assert message_url_flags([]) == []
    assert message_url_flags(["https://bit.ly/a", "http://x.icu"]) == ["shortener_domain", "no_https"]


def test_malformed_urls_do_not_raise():
    # urlsplit reads square brackets as a broken IPv6 host and raises ValueError.
    assert url_flags("http://[secure].bdo.com") == ["malformed_url"]
    assert url_flags("http://a]b.com") == ["malformed_url"]
    assert message_url_flags(["http://[secure].bdo.com", "bit.ly/x"]) == [
        "shortener_domain", "no_https", "malformed_url"]
