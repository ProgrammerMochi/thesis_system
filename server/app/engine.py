"""The two engines behind POST /v1/analyze.

StubEngine returns a fixed, valid Figure 3 response without loading the
model, so the Android client can be built against the real contract.
RealEngine runs textnorm -> RoBERTa -> SHAP -> offset mapping.

Zero retention: message text lives only in local variables of one call. It
is never logged, cached, or stored on self.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import Path

from ml.textnorm import to_model_text, to_model_text_aligned
from ml.url_features import message_url_flags
from server.config import MALICIOUS_INDEX, REPO_ROOT, Settings
from server.app.offsets import to_sent_text, utf16_index
from server.app.timing import StageTimer

log = logging.getLogger("smishing.engine")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _response(settings: Settings, p: float, attributions, url_flags, base_value,
              truncated: bool, computed: bool, model_version: str, basis: str | None = None) -> dict:
    return {
        "verdict": "malicious" if p >= settings.threshold else "legitimate",
        "threat_probability": round(p, 4),   # p_text until fusion exists
        "text_attributions": attributions,
        "url_flags": url_flags,
        "scores": {"p_text": round(p, 4), "p_url": None},
        "decision": {"threshold": settings.threshold, "basis": basis or settings.decision_basis},
        "explanation": {"base_value": None if base_value is None else round(base_value, 4),
                        "unit": "word", "offset_unit": "utf16",
                        "truncated": truncated, "computed": computed},
        "model_version": model_version,
    }


class StubEngine:
    """Fixed verdict and probability (the Figure 3 values). Attributions sit
    on the first two words of the sent text so offsets are always valid."""
    name = "stub"
    ready = True

    def __init__(self, settings: Settings):
        self.settings = settings

    def meta(self) -> dict:
        return {"mode": "stub", "model_version": "stub"}

    def analyze(self, text: str, explain: bool, timer: StageTimer) -> dict:
        u16 = utf16_index(text)
        words = [m.span() for m in re.finditer(r"\S+", text)][:2]
        attributions = [{"start": u16[s], "end": u16[e], "weight": w}
                        for (s, e), w in zip(words, (0.31, 0.22))] if explain else []
        return _response(self.settings, 0.98, attributions, ["shortener_domain", "no_https"],
                         0.62 if explain else None, False, explain, "stub", basis="stub_fixed_response")


class RealEngine:
    name = "real"

    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from ml.explain import WordExplainer

        self.settings = settings
        self.ready = False
        self._torch = torch
        torch.set_num_threads(settings.torch_threads)
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_dir)
        if not self.tokenizer.is_fast:
            raise RuntimeError("a fast tokenizer is required")
        self.model = AutoModelForSequenceClassification.from_pretrained(settings.model_dir)
        self.model.eval()
        if self.model.config.num_labels != 2:
            raise RuntimeError("expected a 2-label classifier")
        self.explainer = WordExplainer(self.predict_proba, self.tokenizer.mask_token,
                                       settings.shap_max_evals, settings.shap_batch_size)
        self.lock = threading.Lock()
        weights = settings.model_dir / "model.safetensors"
        self._meta = {
            "mode": "real",
            "model_version": settings.model_dir.name,
            "weights_sha256": _sha256(weights) if weights.exists() else None,
            "textnorm_sha256": _sha256(REPO_ROOT / "ml" / "textnorm.py"),
            "threshold": settings.threshold,
            "decision_basis": settings.decision_basis,
            "max_len": settings.max_len,
            "malicious_index": MALICIOUS_INDEX,
            "shap": {"explainer": "Partition", "unit": "word", "mask_token": self.tokenizer.mask_token,
                     "max_evals": settings.shap_max_evals, "batch_size": settings.shap_batch_size},
            "torch_threads": settings.torch_threads,
            "alignment_mismatches": 0,
            "nfc_fallbacks": 0,
        }

    def warm_up(self):
        for t in ("Warm-up message one, walang laman.", "Second warm-up: visit example.com now"):
            self.analyze(t, True, StageTimer())

    def meta(self) -> dict:
        import shap
        import transformers
        return {**self._meta, "versions": {"torch": self._torch.__version__,
                                           "transformers": transformers.__version__,
                                           "shap": shap.__version__}}

    def predict_proba(self, texts: list[str]):
        """p(malicious) for already-preprocessed model texts."""
        torch = self._torch
        enc = self.tokenizer(texts, truncation=True, max_length=self.settings.max_len,
                             padding=True, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**enc).logits
        return torch.softmax(logits.float(), -1)[:, MALICIOUS_INDEX].numpy()

    def analyze(self, text: str, explain: bool, timer: StageTimer) -> dict:
        with timer.stage("queue"):
            self.lock.acquire()
        try:
            with timer.stage("preprocess"):
                model_text, urls = to_model_text(text)
                aligned_text, _, char_map, exact = to_model_text_aligned(text)
                flags = message_url_flags(urls)
                n_tokens = len(self.tokenizer(model_text)["input_ids"])
            with timer.stage("classify"):
                p = float(self.predict_proba([model_text])[0])
            attributions, base, computed = [], None, False
            if explain and model_text:
                if aligned_text != model_text:
                    # Never show a highlight we cannot place exactly.
                    self._meta["alignment_mismatches"] += 1
                    log.warning("alignment guard fired; attributions omitted")
                elif not exact:
                    # NFC fallback: every span would cover the whole message.
                    self._meta["nfc_fallbacks"] += 1
                    log.warning("NFC fallback fired; attributions omitted")
                else:
                    with timer.stage("shap"):
                        ex = self.explainer.explain(model_text)
                    with timer.stage("map"):
                        sent = to_sent_text(ex.spans, char_map, text)
                        attributions = sorted(
                            ({"start": s, "end": e, "weight": round(w, 4)}
                             for (s, e), w in zip(sent, ex.weights) if round(w, 4) != 0),
                            key=lambda a: -abs(a["weight"]))
                    base, computed = ex.base_value, True
            return _response(self.settings, p, attributions, flags, base,
                             n_tokens > self.settings.max_len, computed, self.settings.model_dir.name)
        finally:
            self.lock.release()


def build_engine(settings: Settings):
    if settings.mode == "stub":
        return StubEngine(settings)
    if settings.mode == "real":
        return RealEngine(settings)
    raise ValueError("SMISHING_MODE must be 'real' or 'stub'")
