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


# --- Offset tracking (server only) ----------------------------------------
# The FastAPI server must report SHAP attributions as character offsets into
# the text the phone sent (Chapter 3, Section 3.2.4), but the steps above
# shift positions (whitespace collapsing, URL -> [LINK], masking). The
# functions below replay EXACTLY the same steps with the same regexes while
# recording, for every output character, the span of input characters that
# produced it. to_model_text() above is deliberately left untouched: the model
# was trained on its output. tests/test_textnorm_aligned.py asserts that both
# paths produce identical text, and the server checks it again per request.

Span = tuple[int, int]


def _tsub(pattern: re.Pattern, repl, t: str, spans: list[Span]) -> tuple[str, list[Span]]:
    """Tracked re.sub. `repl(m)` returns pieces (text, start, end) in the
    coordinates of `t`. A piece that equals t[start:end] keeps its per-
    character spans; any other piece maps every character to the union of
    spans[start:end] (a replacement token covers the whole matched text)."""
    out, out_spans, pos = [], [], 0
    for m in pattern.finditer(t):
        out.append(t[pos:m.start()])
        out_spans.extend(spans[pos:m.start()])
        for text, a, b in repl(m):
            out.append(text)
            if t[a:b] == text:
                out_spans.extend(spans[a:b])
            else:
                covered = spans[a:b] or spans[max(a - 1, 0):a] or [(0, 0)]
                union = (min(s for s, _ in covered), max(e for _, e in covered))
                out_spans.extend([union] * len(text))
        pos = m.end()
    out.append(t[pos:])
    out_spans.extend(spans[pos:])
    return "".join(out), out_spans


def _const(token: str):
    return lambda m: [(token, m.start(), m.end())]


def _nfc_tracked(text: str) -> tuple[str, list[Span], bool]:
    """NFC per cluster (a base character plus following combining marks).
    If that ever differs from whole-string NFC (e.g. Hangul jamo), fall back
    to mapping every character to the whole input and report exact=False, so
    the server omits attributions instead of highlighting the whole message."""
    out, spans, i = [], [], 0
    while i < len(text):
        j = i + 1
        while j < len(text) and unicodedata.combining(text[j]):
            j += 1
        piece = unicodedata.normalize("NFC", text[i:j])
        out.append(piece)
        spans.extend([(i, j)] * len(piece) if len(piece) != j - i else [(k, k + 1) for k in range(i, j)])
        i = j
    t = "".join(out)
    whole = unicodedata.normalize("NFC", text)
    if t != whole:
        return whole, [(0, len(text))] * len(whole), False
    return t, spans, True


def _normalize_tracked(text: str) -> tuple[str, list[Span], bool]:
    """Mirror of normalize(). The bool is False if the NFC fallback fired."""
    t, spans, exact = _nfc_tracked(str(text))
    for lit, rep in (("\\r\\n", "\n"), ("\\n", "\n"), ("\\t", " ")):
        t, spans = _tsub(re.compile(re.escape(lit)), _const(rep), t, spans)
    t, spans = _tsub(_CONTROL_RE, _const(" "), t, spans)
    t, spans = _tsub(re.compile(r"[ \t ]+"), _const(" "), t, spans)
    t, spans = _tsub(re.compile(r"\n\s*\n+"), _const("\n"), t, spans)
    left = len(t) - len(t.lstrip())
    right = len(t.rstrip())
    return t[left:right], spans[left:right], exact


def _link_pieces(m: re.Match):
    """Mirror of _link_sub()."""
    u = m.group(0)
    core = u.rstrip(_TRAILING_PUNCT)
    if not core:
        return [(u, m.start(), m.end())]
    cut = m.start() + len(core)
    return [(LINK_TOKEN, m.start(), cut), (u[len(core):], cut, m.end())]


def _otp_pieces(m: re.Match):
    """Mirror of the OTP lambda in mask_pii()."""
    return [(m.group(1), m.start(1), m.end(1)), (m.group(2), m.start(2), m.end(2)),
            ("[OTP]", m.start(3), m.end(3))]


def to_model_text_aligned(text: str) -> tuple[str, list[str], list[Span], bool]:
    """Same output as to_model_text(), plus spans[i] = (start, end) of the
    input characters (Python code points) that produced model_text[i], and
    `exact` (False if the NFC fallback fired and the spans are coarse)."""
    t, spans, exact = _normalize_tracked(text)
    t, spans = _tsub(EMAIL_RE, _const("[EMAIL]"), t, spans)
    urls = extract_urls(t)
    t, spans = _tsub(URL_RE, _link_pieces, t, spans)
    # mask_pii(), step by step
    t, spans = _tsub(EMAIL_RE, _const("[EMAIL]"), t, spans)
    t, spans = _tsub(PHONE_RE, _const("[PHONE_NUM]"), t, spans)
    t, spans = _tsub(OTP_RE, _otp_pieces, t, spans)
    t, spans = _tsub(AMOUNT_RE, _const("[AMOUNT]"), t, spans)
    t, spans = _tsub(REF_RE, _const("[REF]"), t, spans)
    return t, urls, spans, exact


def source_span(spans: list[Span], start: int, end: int) -> Span:
    """Map a [start, end) span of model text back to the input text."""
    covered = spans[start:end]
    return min(s for s, _ in covered), max(e for _, e in covered)
