"""Lexical URL model (Chapter 3, Section 3.2.3): features, scoring, and the
per-feature contributions that explain p_url.

Shared by the training script (ml/train_url_fusion.py) and the server, so a
URL is turned into numbers in exactly one place. The model is a logistic
regression stored as plain JSON (feature names, standardization constants,
coefficients), so the server needs only numpy to apply it and every
contribution can be read off directly: contribution_j = coef_j * z_j, in
log-odds, and p_url = sigmoid(intercept + sum_j contribution_j).

Links are NOT visited: shortened links are not expanded, no DNS or WHOIS
lookup is made. Every feature comes from the link's characters. The public
suffix list is the snapshot bundled with tldextract (no network fetch).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import tldextract

from ml.url_features import _split, url_flags

# Bundled public suffix snapshot only: never fetches, never writes a cache.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

# Words that phishing URLs often carry in the host or path. A researcher-
# compiled list (the group's choice), not taken from a published source.
SUSPICIOUS_WORDS = ("verify", "login", "log-in", "signin", "secure", "account", "update",
                    "confirm", "unlock", "reactivate", "suspend", "kyc", "otp", "wallet",
                    "claim", "prize", "reward", "bonus", "gift", "free", "win", "promo")

NUMERIC_FEATURES = (
    "log_url_len", "log_host_len", "log_path_len", "n_host_labels", "n_subdomain_labels",
    "has_www", "n_hyphens_host", "n_digits_host", "digit_ratio_host", "n_path_segments",
    "has_query", "has_scheme", "is_https", "ip_host", "shortener_domain", "punycode_host",
    "at_in_url", "nonstandard_port", "malformed_url", "n_suspicious_words",
)

# Plain-language names, for the demo page and the paper's tables.
FEATURE_TEXT = {
    "log_url_len": "link length", "log_host_len": "domain length", "log_path_len": "path length",
    "n_host_labels": "number of dot-separated parts in the domain",
    "n_subdomain_labels": "number of subdomains", "has_www": "starts with www",
    "n_hyphens_host": "hyphens in the domain", "n_digits_host": "digits in the domain",
    "digit_ratio_host": "share of digits in the domain", "n_path_segments": "path depth",
    "has_query": "has a ?query part", "has_scheme": "starts with http(s)://",
    "is_https": "uses https", "ip_host": "bare IP address instead of a name",
    "shortener_domain": "link shortener", "punycode_host": "look-alike (punycode) domain",
    "at_in_url": "'@' inside the link", "nonstandard_port": "unusual port",
    "malformed_url": "malformed link", "n_suspicious_words": "words like verify/login/claim",
}


def tld_of(url: str) -> str:
    """Public suffix of the link's host ('com.ph', 'top', ...); 'ip' for an
    IP address, '' if none can be found."""
    if "ip_host" in url_flags(url):
        return "ip"
    try:
        _, parts = _split(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return ""
    return _EXTRACT(host).suffix.lower()


def numeric_features(url: str) -> dict[str, float]:
    flags = set(url_flags(url))
    try:
        has_scheme, parts = _split(url)
        host = (parts.hostname or "").lower()
        path, query, scheme = parts.path or "", parts.query, parts.scheme.lower()
    except ValueError:
        has_scheme, host, path, query, scheme = "://" in url[:10], "", "", "", ""
    labels = [x for x in host.split(".") if x]
    sub = _EXTRACT(host).subdomain if host and "ip_host" not in flags else ""
    sub_labels = [x for x in sub.split(".") if x and x != "www"]
    digits = sum(c.isdigit() for c in host)
    low = url.lower()
    return {
        "log_url_len": math.log1p(len(url)),
        "log_host_len": math.log1p(len(host)),
        "log_path_len": math.log1p(len(path.strip("/"))),
        "n_host_labels": float(len(labels)),
        "n_subdomain_labels": float(len(sub_labels)),
        "has_www": float(host.startswith("www.")),
        "n_hyphens_host": float(host.count("-")),
        "n_digits_host": float(digits),
        "digit_ratio_host": digits / len(host) if host else 0.0,
        "n_path_segments": float(len([s for s in path.split("/") if s])),
        "has_query": float(bool(query)),
        "has_scheme": float(has_scheme),
        "is_https": float(has_scheme and scheme == "https"),
        "ip_host": float("ip_host" in flags),
        "shortener_domain": float("shortener_domain" in flags),
        "punycode_host": float("punycode_host" in flags),
        "at_in_url": float("at_in_url" in flags),
        "nonstandard_port": float("nonstandard_port" in flags),
        "malformed_url": float("malformed_url" in flags),
        "n_suspicious_words": float(sum(w in low for w in SUSPICIOUS_WORDS)),
    }


def feature_vector(url: str, tld_vocab: list[str]) -> tuple[list[str], list[float]]:
    """Numeric features, then one indicator per vocabulary TLD plus 'tld=other'."""
    num = numeric_features(url)
    tld = tld_of(url)
    names = list(NUMERIC_FEATURES) + [f"tld={t}" for t in tld_vocab] + ["tld=other"]
    values = [num[n] for n in NUMERIC_FEATURES] + [float(tld == t) for t in tld_vocab]
    values.append(float(tld not in tld_vocab))
    return names, values


def feature_text(name: str) -> str:
    if name.startswith("tld="):
        t = name[4:]
        return "uncommon domain ending" if t == "other" else f"domain ends in .{t}"
    return FEATURE_TEXT.get(name, name)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


class UrlModel:
    """Logistic regression over feature_vector(), loaded from JSON."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.tld_vocab = list(spec["tld_vocab"])
        self.names = list(spec["feature_names"])
        self.mean = [float(x) for x in spec["mean"]]
        self.scale = [float(x) if float(x) > 0 else 1.0 for x in spec["scale"]]
        self.coef = [float(x) for x in spec["coef"]]
        self.intercept = float(spec["intercept"])
        names, _ = feature_vector("example.com", self.tld_vocab)
        if names != self.names:
            raise ValueError("url model was trained with a different feature set")

    @classmethod
    def load(cls, path: Path) -> "UrlModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def contributions(self, url: str) -> list[tuple[str, float]]:
        """(feature, coef * standardized value) for every feature, in log-odds."""
        _, x = feature_vector(url, self.tld_vocab)
        return [(n, c * (v - m) / s)
                for n, c, v, m, s in zip(self.names, self.coef, x, self.mean, self.scale)]

    def score(self, url: str) -> float:
        return _sigmoid(self.intercept + sum(c for _, c in self.contributions(url)))

    def explain(self, url: str, k: int = 3) -> tuple[float, list[dict]]:
        """p_url and the k features that moved this link's score most."""
        contrib = self.contributions(url)
        p = _sigmoid(self.intercept + sum(c for _, c in contrib))
        top = sorted((c for c in contrib if abs(c[1]) >= 1e-6), key=lambda c: -abs(c[1]))[:k]
        return p, [{"feature": n, "label": feature_text(n), "contribution": round(c, 4)} for n, c in top]


def message_p_url(model: UrlModel, urls: list[str]) -> float | None:
    """A message's p_url is its riskiest link; None when it has no link."""
    return max((model.score(u) for u in urls), default=None)


def fuse(p_text: float, p_url: float | None, w: float) -> float:
    """S = w * p_text + (1 - w) * p_url; p_text alone when there is no link."""
    return p_text if p_url is None else w * p_text + (1.0 - w) * p_url
