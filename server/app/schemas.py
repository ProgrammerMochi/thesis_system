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
    p_url: Optional[float] = None   # Increment 2


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
    model_version: str
