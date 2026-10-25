"""Lexical URL model: features, scoring, contributions, fusion.
Run: python -m pytest tests/"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.url_model import (NUMERIC_FEATURES, UrlModel, feature_vector, fuse,  # noqa: E402
                          message_p_url, numeric_features, tld_of)


def test_tld_uses_public_suffix_offline():
    assert tld_of("https://www.bdo.com.ph/x") == "com.ph"
    assert tld_of("bdo-secure-login.top/verify") == "top"
    assert tld_of("GoJackpot1.vegas") == "vegas"
    assert tld_of("http://192.168.1.5:8080/a") == "ip"
    assert tld_of("http://[secure].bdo.com") == ""


def test_numeric_features():
    f = numeric_features("http://bdo-secure-login.top/verify/account?id=1")
    assert f["n_hyphens_host"] == 2 and f["has_query"] == 1 and f["is_https"] == 0
    assert f["has_scheme"] == 1 and f["n_path_segments"] == 2 and f["n_suspicious_words"] >= 3
    g = numeric_features("https://a.b.c.example.co.uk/")
    assert g["n_subdomain_labels"] == 3 and g["is_https"] == 1 and g["log_path_len"] == 0
    assert numeric_features("www.bdo.com.ph")["n_subdomain_labels"] == 0   # www is not counted
    assert numeric_features("bit.ly/x")["shortener_domain"] == 1
    assert numeric_features("http://[secure].bdo.com")["malformed_url"] == 1
    assert set(numeric_features("x.com")) == set(NUMERIC_FEATURES)


def test_feature_vector_tld_indicators():
    names, x = feature_vector("evil.top", ["com", "top"])
    assert names[-3:] == ["tld=com", "tld=top", "tld=other"] and x[-3:] == [0.0, 1.0, 0.0]
    _, x = feature_vector("evil.icu", ["com", "top"])
    assert x[-3:] == [0.0, 0.0, 1.0]


def _model(coef_by_name: dict, intercept=0.0, vocab=("top",)):
    names, _ = feature_vector("example.com", list(vocab))
    return UrlModel({"tld_vocab": list(vocab), "feature_names": names, "mean": [0.0] * len(names),
                     "scale": [1.0] * len(names), "intercept": intercept,
                     "coef": [coef_by_name.get(n, 0.0) for n in names]})


def test_score_equals_sigmoid_of_contributions():
    m = _model({"tld=top": 2.0, "is_https": -1.5}, intercept=-0.5)
    p, top = m.explain("evil.top/login")
    assert p == pytest.approx(1 / (1 + math.exp(-1.5)))
    assert top[0]["feature"] == "tld=top" and top[0]["contribution"] == 2.0
    assert m.score("https://evil.top") == pytest.approx(1 / (1 + math.exp(0.0)))


def test_rejects_mismatched_feature_set():
    m = _model({})
    spec = dict(m.spec, feature_names=m.spec["feature_names"][:-1])
    with pytest.raises(ValueError):
        UrlModel(spec)


def test_message_score_and_fusion():
    m = _model({"tld=top": 3.0})
    assert message_p_url(m, []) is None
    assert message_p_url(m, ["safe.com", "evil.top"]) == pytest.approx(m.score("evil.top"))
    assert fuse(0.2, None, 0.3) == 0.2                    # no link: p_text alone
    assert fuse(0.2, 1.0, 0.25) == pytest.approx(0.8)
