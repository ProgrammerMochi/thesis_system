"""Lexical URL flags (Chapter 3, Section 3.2.3), shared by the server and,
later, the Increment 2 URL classifier.

Increment 3 reports only yes/no flags that need no tuned cut-off. Count
features (length, hyphens, subdomain depth) need thresholds; those will be
derived from the TRAIN split in Increment 2, not chosen by hand, and p_url
is not computed yet.
"""
from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

# Researcher-compiled list of public link shorteners seen in PH SMS. It is a
# convenience list, not an authoritative registry; Increment 2 replaces it
# with a documented source and reports its coverage.
SHORTENERS = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly", "shorturl.at",
    "rb.gy", "ow.ly", "tiny.cc", "s.id", "shorten.is", "rebrand.ly", "t.ly",
    "v.gd", "bitly.com",
})

FLAG_NAMES = ("ip_host", "shortener_domain", "punycode_host", "at_in_url",
              "no_https", "nonstandard_port", "malformed_url")


def _split(url: str):
    has_scheme = "://" in url[:10]
    parts = urlsplit(url if has_scheme else "http://" + url)
    return has_scheme, parts


def url_flags(url: str) -> list[str]:
    """Flags for one URL, in FLAG_NAMES order. A URL that urlsplit cannot
    parse (for example stray square brackets, read as a broken IPv6 host)
    gets only `malformed_url`: no other flag can be computed reliably, and
    the request must not fail because of it."""
    try:
        has_scheme, parts = _split(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return ["malformed_url"]
    flags = set()
    try:
        ipaddress.ip_address(host)
        flags.add("ip_host")
    except ValueError:
        pass
    if host.removeprefix("www.") in SHORTENERS:
        flags.add("shortener_domain")
    if any(label.startswith("xn--") for label in host.split(".")):
        flags.add("punycode_host")
    if "@" in parts.netloc:
        flags.add("at_in_url")
    if not (has_scheme and parts.scheme.lower() == "https"):
        flags.add("no_https")
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is not None and port not in (80, 443):
        flags.add("nonstandard_port")
    return [f for f in FLAG_NAMES if f in flags]


def message_url_flags(urls: list[str]) -> list[str]:
    """Union of flags over all URLs in a message, in FLAG_NAMES order."""
    found = {f for u in urls for f in url_flags(u)}
    return [f for f in FLAG_NAMES if f in found]
