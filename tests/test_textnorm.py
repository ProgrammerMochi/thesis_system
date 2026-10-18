"""Contract tests for preprocessing. The Kotlin on-device masker must pass
the PHONE and OTP cases below too (port them to an Android unit test).
Run: python -m pytest tests/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from textnorm import normalize, template_key, to_model_text  # noqa: E402

CASES = [
    ("Get P3K now! visit lvv65a118l.uk/?x=1. Call 0917-123-4567",
     "Get [AMOUNT] now! visit [LINK]. Call [PHONE_NUM]", ["lvv65a118l.uk/?x=1"]),
    ("Email juan.dela.cruz@gmail.com today.Please reply",
     "Email [EMAIL] today.Please reply", []),
    ("Your OTP is 482910. Ref 1234567890", "Your OTP is [OTP]. Ref [REF]", []),
    ("Bisitahin ang https://shorten.is/BDO_Update ngayon!",
     "Bisitahin ang [LINK] ngayon!", ["https://shorten.is/BDO_Update"]),
    ("Login sa GoJackpot1.vegas, +639171234567", "Login sa [LINK], [PHONE_NUM]",
     ["GoJackpot1.vegas"]),
    ("Spend P50K at the Mar. 14-16 sale", "Spend [AMOUNT] at the Mar. 14-16 sale", []),
]


def test_model_text():
    for raw, want_text, want_urls in CASES:
        got_text, got_urls = to_model_text(raw)
        assert got_text == want_text, (raw, got_text)
        assert got_urls == want_urls, (raw, got_urls)


def test_literal_newlines_and_casing():
    assert normalize("Hi!\\nBDO   Advisory") == "Hi!\nBDO Advisory"


def test_template_groups_variants():
    a, _ = to_model_text("Win P500 at a1.xyz code 12")
    b, _ = to_model_text("Win P900 at b2.xyz code 34")
    assert template_key(a) == template_key(b)
