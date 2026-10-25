"""to_model_text_aligned() must produce EXACTLY the text to_model_text()
produces (the model was trained on the latter), and its character map must
point back into the input. Run: python -m pytest tests/"""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from textnorm import source_span, to_model_text, to_model_text_aligned  # noqa: E402
from test_textnorm import CASES  # noqa: E402

# Researcher-written edge cases (none from data/), one per position-shifting step.
EDGE = [
    "",
    "   ",
    "Hello  world\t\tagain  now",
    "  leading and trailing  ",
    "Line one\\nLine two\\r\\nLine three\\tTabbed",
    "Para one\n\n\n  Para two",
    "Café and piña",                     # combining accents -> NFC
    "Å angstrom sign, 가 jamo",          # singleton and Hangul NFC
    "Ctrl\u0007char\u0085here",
    "Emoji 😀🔥 then visit bit.ly/abc123). Now!",
    "Click (www.Example-Bank.com/verify), please.",
    "Mail me@x.com or go to x.com/y?z=1",
    "Your verification code: 1234. PIN ay 98765",
    "Send P1,500.00 or ₱20 or PHP 3k to 0917 123 4567 ref 123456789012",
    "Already masked: [PHONE_NUM] [OTP] [LINK] [AMOUNT]",
    "Mixed \n spaces\n \n\nend",
    "https://a.b/c... and http://d.e/f!!",
]


def _check(raw):
    want_text, want_urls = to_model_text(raw)
    text, urls, spans, _ = to_model_text_aligned(raw)
    assert text == want_text, (raw, text, want_text)
    assert urls == want_urls, raw
    assert len(spans) == len(text)
    n = len(str(raw))
    for s, e in spans:
        assert 0 <= s <= e <= n
    starts = [s for s, _ in spans]
    assert starts == sorted(starts), raw


@pytest.mark.parametrize("raw", [c[0] for c in CASES] + EDGE)
def test_identical_model_text(raw):
    _check(raw)


def test_kept_characters_point_at_themselves():
    raw = "  Hi  Juan, visit bit.ly/x now. Call 09171234567!"
    text, _, spans, exact = to_model_text_aligned(raw)
    assert exact
    for ch, (s, e) in zip(text, spans):
        if e - s == 1 and ch not in " \n":
            assert raw[s] == ch


def test_placeholder_maps_to_whole_original():
    raw = "Visit https://shorten.is/BDO_Update ngayon! Call 0917-123-4567."
    text, _, spans, exact = to_model_text_aligned(raw)
    assert exact
    i = text.index("[LINK]")
    s, e = source_span(spans, i, i + len("[LINK]"))
    assert raw[s:e] == "https://shorten.is/BDO_Update"
    i = text.index("[PHONE_NUM]")
    s, e = source_span(spans, i, i + len("[PHONE_NUM]"))
    assert raw[s:e] == "0917-123-4567"


def test_trailing_punctuation_kept_separately():
    raw = "Go to lvv65a118l.uk/?x=1. Now"
    text, _, spans, exact = to_model_text_aligned(raw)
    assert exact
    assert text == "Go to [LINK]. Now"
    dot = text.index("[LINK]") + len("[LINK]")
    assert raw[spans[dot][0]:spans[dot][1]] == "."


def test_nfc_fallback_is_reported():
    # Hangul jamo compose across clusters, so the per-cluster NFC differs from
    # whole-string NFC; the text must still be identical, but exact=False.
    raw = "Hi \u1100\u1161 there"
    text, _, spans, exact = to_model_text_aligned(raw)
    assert text == to_model_text(raw)[0] and exact is False
    assert to_model_text_aligned("Cafe\u0301 ok")[3] is True


hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402

_ALPHABET = st.sampled_from(list(
    "abcXYZ019 .,:/-_@()!?\n\t ́\u0007₱😀") + [
    "\\n", "\\r\\n", "\\t", "http://", "www.", ".com", ".icu", "bit.ly/", "OTP is ",
    "code: ", "P500", "PHP 1,000", "09171234567", "+63 917 123 4567", "12345678901",
    "juan@x.com", "[LINK]", "  ", "\n\n",
])


@settings(max_examples=3000, deadline=None)
@given(st.lists(_ALPHABET, max_size=40).map("".join))
def test_identical_model_text_fuzz(raw):
    _check(raw)


CORPUS = ROOT / "data" / "processed" / "corpus.csv"


@pytest.mark.skipif(not CORPUS.exists(), reason="frozen corpus not present (data/ is not in git)")
def test_identical_on_frozen_corpus_train_val():
    """Every train/val message: aligned text == to_model_text == the stored
    text_model the model was trained on. Test-split rows are skipped before
    their text is used, because the test split is sealed (CLAUDE.md rule 1)."""
    n = 0
    with open(CORPUS, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                continue
            text, _, _, _ = to_model_text_aligned(row["text_norm"])
            assert text == row["text_model"], row["msg_id"]
            n += 1
    assert n > 2000, n
