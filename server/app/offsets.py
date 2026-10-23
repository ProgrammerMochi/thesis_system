"""Model-text spans -> spans of the text the phone sent, in UTF-16 units.

Python indexes strings by code point; Kotlin (the Android client) by UTF-16
code unit. An emoji is one code point but two UTF-16 units, so without this
conversion every highlight after an emoji would be shifted."""
from __future__ import annotations

from itertools import accumulate

from ml.textnorm import source_span


def utf16_index(text: str) -> list[int]:
    """utf16[i] = UTF-16 offset of code point i (length len(text) + 1)."""
    return [0] + list(accumulate(2 if ord(c) > 0xFFFF else 1 for c in text))


def to_sent_text(spans_model: list[tuple[int, int]], char_map: list[tuple[int, int]],
                 sent_text: str) -> list[tuple[int, int]]:
    u16 = utf16_index(sent_text)
    out = []
    for a, b in spans_model:
        s, e = source_span(char_map, a, b)
        out.append((u16[s], u16[e]))
    return out
