"""SHAP explanation of p(malicious) at word level (Chapter 3, Section 3.2.4).

The server and the SOP 4 faithfulness script must both use this module, so
the faithfulness result describes the explainer the app actually shows.

Method choices (state them in the paper):
- Partition explainer: Owen values over a hierarchy of neighbouring words.
  Its cost is bounded by `max_evals`, not by message length.
- Unit: whitespace-separated words of the model text. A placeholder such as
  [LINK] is one unit, so its weight maps back to the whole original URL.
- Baseline: a removed word is replaced by the tokenizer's mask token
  (<mask>). The separators around it are kept, so the unmasked input is
  byte-identical to the model text and base + sum(weights) = p_text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import shap

WORD_SPLIT = r"\s+"


class WordMasker(shap.maskers.Text):
    """shap's Text masker over whitespace words, with one change: masking
    keeps the original separators. The stock masker collapses all whitespace
    (newlines included), so its "nothing masked" input would differ from the
    model text and SHAP would explain a slightly different string."""

    def __init__(self, mask_token: str):
        super().__init__(tokenizer=WORD_SPLIT, mask_token=mask_token, collapse_mask_token=False)

    def __call__(self, mask, s):
        mask = self._standardize_mask(mask, s)
        self._update_s_cache(s)
        parts = []
        for keep, seg in zip(mask, self._segments_s):
            if keep:
                parts.append(seg)
            else:
                word = seg.rstrip()
                parts.append(self.mask_token + seg[len(word):])
        return (np.array(["".join(parts)]),)


@dataclass(frozen=True)
class Explanation:
    spans: list[tuple[int, int]]   # word spans in the model text
    weights: list[float]
    base_value: float


def word_spans(model_text: str) -> list[tuple[int, int]]:
    """The units SHAP assigns weights to, as [start, end) in model text."""
    import re
    return [m.span() for m in re.finditer(r"\S+", model_text)]


class WordExplainer:
    def __init__(self, predict_proba: Callable[[list[str]], np.ndarray], mask_token: str,
                 max_evals: int, batch_size: int):
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.masker = WordMasker(mask_token)
        self.explainer = shap.explainers.Partition(
            lambda texts: predict_proba([str(t) for t in texts]), self.masker)

    def explain(self, model_text: str) -> Explanation:
        spans = word_spans(model_text)
        if not spans:
            return Explanation([], [], 0.0)
        out = self.explainer([model_text], max_evals=self.max_evals,
                             batch_size=self.batch_size, silent=True)
        values = np.asarray(out.values[0], dtype=float).reshape(-1)
        base = float(np.asarray(out.base_values).reshape(-1)[0])
        if len(values) != len(spans):
            raise RuntimeError("SHAP unit count does not match word spans")
        return Explanation(spans, values.tolist(), base)
