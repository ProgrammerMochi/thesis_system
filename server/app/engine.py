"""The two engines behind POST /v1/analyze.

StubEngine returns a fixed, valid Figure 3 response without loading the
model, so the Android client can be built against the real contract.
RealEngine runs textnorm -> RoBERTa -> SHAP -> offset mapping.

Zero retention: message text lives only in local variables of one call. It
is never logged, cached, or stored on self.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path

from ml.textnorm import to_model_text, to_model_text_aligned
from ml.url_features import message_url_flags, url_flags
from ml.url_model import UrlModel, fuse
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


class Fusion:
    """The URL model and the validation-tuned fusion weight (ml/train_url_fusion.py).
    Loaded from settings.fusion_dir; None (text-only) if the files are missing."""

    def __init__(self, directory: Path):
        self.url_model = UrlModel.load(directory / "url_model.json")
        spec = json.loads((directory / "fusion.json").read_text(encoding="utf-8"))
        self.w = float(spec["w"])
        self.tuned_threshold = float(spec["threshold"])
        self.meta = {"dir": directory.name, "w": self.w, "tuned_threshold": self.tuned_threshold,
                     "url_model_sha256": _sha256(directory / "url_model.json"),
                     "label_source": self.url_model.spec.get("label_source")}

    @classmethod
    def load(cls, settings: Settings) -> "Fusion | None":
        d = settings.fusion_dir
        if d is None or not (d / "url_model.json").exists() or not (d / "fusion.json").exists():
            log.warning("URL fusion files not found; deciding on p_text alone")
            return None
        return cls(d)


def decision_rule(settings: Settings, fusion: Fusion | None) -> tuple[float, str]:
    """(threshold, basis) in effect. The validation-tuned threshold is used only
    when SMISHING_USE_TUNED_THRESHOLD=1; otherwise 0.5 stays (provisional)."""
    if fusion is None:
        return settings.threshold, settings.decision_basis
    if settings.use_tuned_threshold:
        return fusion.tuned_threshold, "fusion_prototype_w_and_threshold_val_tuned_message_labels"
    return settings.threshold, "fusion_prototype_w_val_tuned_threshold_0.5_provisional"


def url_analysis(fusion: Fusion | None, urls: list[str], link_spans: list | None) -> list[dict]:
    """One entry per link: UTF-16 offsets into the sent text (never the link
    text), p_url, flags and the features that moved p_url most."""
    if fusion is None:
        return []
    out = []
    for i, u in enumerate(urls):
        p, top = fusion.url_model.explain(u)
        start, end = link_spans[i] if link_spans else (None, None)
        out.append({"start": start, "end": end, "p_url": round(p, 4),
                    "flags": url_flags(u), "top_features": top})
    return out


def _response(settings: Settings, p: float, attributions, url_flags, base_value,
              truncated: bool, computed: bool, model_version: str, basis: str | None = None,
              p_url: float | None = None, w: float | None = None, threshold: float | None = None,
              links: list | None = None) -> dict:
    s = p if w is None else fuse(p, p_url, w)
    threshold = settings.threshold if threshold is None else threshold
    return {
        "verdict": "malicious" if s >= threshold else "legitimate",
        "threat_probability": round(s, 4),   # S; equals p_text when there is no link or no fusion
        "text_attributions": attributions,
        "url_flags": url_flags,
        "scores": {"p_text": round(p, 4), "p_url": None if p_url is None else round(p_url, 4),
                   "fused": round(s, 4), "w": w},
        "decision": {"threshold": threshold, "basis": basis or settings.decision_basis},
        "explanation": {"base_value": None if base_value is None else round(base_value, 4),
                        "unit": "word", "offset_unit": "utf16",
                        "truncated": truncated, "computed": computed},
        "url_analysis": links or [],
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
        # One fixed link entry on the last word, so clients can build the link card.
        last = re.search(r"\S+\s*$", text)
        link = {"start": u16[last.start()], "end": u16[last.start() + len(last.group(0).rstrip())],
                "p_url": 0.97, "flags": ["shortener_domain", "no_https"],
                "top_features": [{"feature": "shortener_domain", "label": "link shortener", "contribution": 1.1},
                                 {"feature": "is_https", "label": "uses https", "contribution": 0.4}]} if last else None
        return _response(self.settings, 0.98, attributions, ["shortener_domain", "no_https"],
                         0.62 if explain else None, False, explain, "stub", basis="stub_fixed_response",
                         links=[link] if link else [])


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
        self.fusion = Fusion.load(settings)
        self.threshold, self.basis = decision_rule(settings, self.fusion)
        self.lock = threading.Lock()
        weights = settings.model_dir / "model.safetensors"
        self._meta = {
            "mode": "real",
            "model_version": settings.model_dir.name,
            "weights_sha256": _sha256(weights) if weights.exists() else None,
            "textnorm_sha256": _sha256(REPO_ROOT / "ml" / "textnorm.py"),
            "threshold": self.threshold,
            "decision_basis": self.basis,
            "fusion": self.fusion.meta if self.fusion else None,
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

    @staticmethod
    def _link_spans(text, aligned_text, model_text, char_map, exact, n_urls):
        """UTF-16 spans of each link in the sent text, from the [LINK] tokens of
        the aligned model text; None when they cannot be placed exactly."""
        if not n_urls or aligned_text != model_text or not exact:
            return None
        starts = [m.start() for m in re.finditer(re.escape("[LINK]"), model_text)]
        if len(starts) != n_urls:
            return None
        return to_sent_text([(a, a + len("[LINK]")) for a in starts], char_map, text)

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
            with timer.stage("url"):
                links = url_analysis(self.fusion, urls, self._link_spans(text, aligned_text, model_text,
                                                                         char_map, exact, len(urls)))
                p_url = max((x["p_url"] for x in links), default=None)
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
                             n_tokens > self.settings.max_len, computed, self.settings.model_dir.name,
                             basis=self.basis, p_url=p_url, threshold=self.threshold, links=links,
                             w=self.fusion.w if self.fusion else None)
        finally:
            self.lock.release()


def build_engine(settings: Settings):
    if settings.mode == "stub":
        return StubEngine(settings)
    if settings.mode == "real":
        return RealEngine(settings)
    raise ValueError("SMISHING_MODE must be 'real' or 'stub'")
