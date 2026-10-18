"""Text normalization, URL extraction, and PII masking.

This module is the single source of truth for preprocessing. The training
pipeline (ml/prepare_corpus.py) and the FastAPI server must import THESE
functions so that training-time and inference-time text are identical
(Chapter 3, Section 3.1). Any rule changed here changes the model's input
distribution: retrain after editing.

The Android client masks phone numbers and OTPs on-device (Section 3.5.2).
Its Kotlin regexes must reproduce PHONE_RE and OTP_RE exactly; the test
vectors in tests/test_textnorm.py are the contract both sides are checked
against.
"""
from __future__ import annotations

import re
import unicodedata

LINK_TOKEN = "[LINK]"

# --- URL extraction --------------------------------------------------------
# Scheme URLs, www-prefixed hosts, and bare domains. Scam domains use cheap,
# unusual TLDs (.icu, .bid, .uk, .at, .vegas ...), so a closed TLD list
# misses most of them (a closed list missed ~800 distinct URLs in this
# corpus). Bare domains instead require an all-LOWERCASE alphabetic TLD of
# 2-24 letters. This rejects the dominant false positive in SMS text, a
# missing space after a period ("today.Please", "Out.More", "2.Mag"), while
# keeping mixed-case hosts such as "GoJackpot1.vegas". Residual errors are
# measured in Increment 2 by validating hosts against the Public Suffix List
# (tldextract) and reported.
URL_RE = re.compile(
    r"(?:[Hh][Tt][Tt][Pp][Ss]?://[^\s<>\"']+)"
    r"|(?:[Ww][Ww][Ww]\.[^\s<>\"']+)"
    r"|(?:(?<![@\w.])[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*\.[a-z]{2,24}\b"
    r"(?:/[^\s<>\"']*)?)"
)
_TRAILING_PUNCT = ".,;:!?)]}'\""

# --- PII masking -----------------------------------------------------------
# PH mobile numbers: 09XXXXXXXXX, +639XXXXXXXXX, 639XXXXXXXXX, with optional
# spaces or hyphens between groups.
PHONE_RE = re.compile(r"(?<!\d)(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# One-time passwords: a 4-8 digit code that follows an OTP keyword.
OTP_RE = re.compile(
    r"(?i)\b(otp|one[- ]time (?:pin|password)|verification code|code|pin)"
    r"(\s*(?:is|:|ay)?\s*)(\d{4,8})\b"
)
# Peso amounts: P500, PHP 1,500.00, Php7,890, ₱20
AMOUNT_RE = re.compile(r"(?i)(?:₱|\bphp\s?|\bp(?=\d))\s?\d[\d,]*(?:\.\d+)?(?:[km]\b)?")
# Long reference numbers (8+ digits) that are not phone numbers.
REF_RE = re.compile(r"(?<!\d)\d{8,}(?!\d)")

_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f-\u009f]")


def normalize(text: str) -> str:
    """Light cleaning only. Casing is preserved (the model is cased)."""
    t = unicodedata.normalize("NFC", str(text))
    # The Chua export stores line breaks as the two characters backslash-n.
    t = t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    t = _CONTROL_RE.sub(" ", t)
    t = re.sub(r"[ \t\u00a0]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return t.strip()


def extract_urls(text: str) -> list[str]:
    urls = []
    for m in URL_RE.finditer(text):
        u = m.group(0).rstrip(_TRAILING_PUNCT)
        if u:
            urls.append(u)
    return urls


def _link_sub(m: re.Match) -> str:
    """Replace a URL with LINK_TOKEN but keep trailing sentence punctuation."""
    u = m.group(0)
    core = u.rstrip(_TRAILING_PUNCT)
    return LINK_TOKEN + u[len(core):] if core else u


def mask_pii(text: str) -> str:
    """Mask PII. Order matters: emails before URLs-in-text, OTP before REF."""
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = PHONE_RE.sub("[PHONE_NUM]", t)
    t = OTP_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}[OTP]", t)
    t = AMOUNT_RE.sub("[AMOUNT]", t)
    t = REF_RE.sub("[REF]", t)
    return t


def to_model_text(text: str) -> tuple[str, list[str]]:
    """normalize -> extract and replace URLs -> mask PII.

    URLs are replaced before PII masking so that digits inside URLs are not
    rewritten as [REF] or [AMOUNT]. Returns (model_text, urls).
    """
    # Emails first: "first.last@gmail.com" would otherwise match URL_RE.
    t = EMAIL_RE.sub("[EMAIL]", normalize(text))
    urls = extract_urls(t)
    t = URL_RE.sub(_link_sub, t)
    t = mask_pii(t)
    return t, urls


def template_key(model_text: str) -> str:
    """Key for near-duplicate grouping in the split (Chapter 2, Section 2.3):
    lowercase, every digit -> 0, whitespace collapsed."""
    t = re.sub(r"\d", "0", model_text.lower())
    return re.sub(r"\s+", " ", t).strip()
