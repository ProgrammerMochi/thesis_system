"""Request and response bodies. The four Figure 3 fields keep their names;
the extra fields make the provisional basis of each verdict visible."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def make_request_model(max_chars: int):
    class AnalyzeRequest(BaseModel):
        # Only the (phone-masked) text. Sender ID and timestamp are not used by
        # any model, so they are not accepted (data minimization, RA 10173).
        model_config = ConfigDict(extra="forbid")
        text: str = Field(min_length=1, max_length=max_chars)

    return AnalyzeRequest


class Attribution(BaseModel):
    start: int   # UTF-16 code units into the text the phone sent
    end: int
    weight: float


class Scores(BaseModel):
    p_text: float
    p_url: Optional[float] = None   # riskiest link; None when there is no link or no URL model
    fused: Optional[float] = None   # S = w * p_text + (1 - w) * p_url (p_text when no link)
    w: Optional[float] = None       # fusion weight tuned on validation; None = text-only


class UrlFeature(BaseModel):
    feature: str
    label: str            # plain-language name
    contribution: float   # log-odds added to this link's score


class UrlAnalysis(BaseModel):
    # UTF-16 offsets of the link in the text the phone sent (never the link
    # text itself); None if the link could not be placed exactly.
    start: Optional[int]
    end: Optional[int]
    p_url: float
    flags: list[str]
    top_features: list[UrlFeature]


class Decision(BaseModel):
    threshold: float
    basis: str


class ExplanationInfo(BaseModel):
    base_value: Optional[float]
    unit: Literal["word"] = "word"
    offset_unit: Literal["utf16"] = "utf16"
    truncated: bool
    computed: bool   # false when ?explain=false or the alignment guard fired


class AnalyzeResponse(BaseModel):
    verdict: Literal["malicious", "legitimate"]
    threat_probability: float
    text_attributions: list[Attribution]
    url_flags: list[str]
    scores: Scores
    decision: Decision
    explanation: ExplanationInfo
    url_analysis: list[UrlAnalysis] = []
    model_version: str
