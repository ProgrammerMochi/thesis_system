"""Sanity check that server.config.ID2LABEL points the right way.

The messages below were written for this test (none come from data/). This
is a direction check on obvious cases, not an evaluation: it fails only if
index 1 does not mean malicious. Needs the server environment and the saved
model in models/; skipped otherwise.
Run: python -m pytest tests/
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))
from server.config import ID2LABEL, MALICIOUS_INDEX, MAX_LEN, MODEL_DIR  # noqa: E402
from textnorm import to_model_text  # noqa: E402

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not (MODEL_DIR / "config.json").exists():
    pytest.skip(f"saved model not found at {MODEL_DIR}", allow_module_level=True)

MALICIOUS = [
    "BDO Alert: Your account has been locked. Verify your details now at "
    "bdo-secure-verify.top/login to avoid suspension.",
    "Congratulations! Ikaw ay nanalo ng P50,000 sa GCash raffle. I-claim na "
    "sa gcash-promo.icu/claim bago mag-expire.",
    "Hi po, need ng part-time worker, P3,000 per day, like lang ng videos. "
    "Message sa telegram: t.me/easyjob_ph",
]
LEGITIMATE = [
    "Hi Ma, nakarating na ako sa bahay. Kain na tayo mamaya pag uwi ni Papa.",
    "Your OTP is 482910. Do not share this code with anyone. "
    "BPI will never ask for your OTP.",
    "Wag mo kalimutan yung payong mo, malakas daw ulan mamaya sabi ni Tita.",
]

# A casual personal message scored 0.54 (malicious side) on 2026-10-23. Kept
# visible as a known weakness rather than dropped from the test: the
# legitimate class is mostly institutional SMS, so chatty personal messages
# may be under-represented. One message proves nothing either way; the
# hard-negative set and validation error analysis should check this.
PERSONAL_BORDERLINE = [
    "Tol, tuloy pa ba tayo sa basketball bukas ng 6pm? Dala ka ng bola.",
]


@pytest.fixture(scope="module")
def p_malicious():
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    def score(texts):
        model_texts = [to_model_text(t)[0] for t in texts]
        enc = tok(model_texts, truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        return torch.softmax(logits, -1)[:, MALICIOUS_INDEX].tolist()

    return score


def test_mapping_is_binary():
    assert ID2LABEL == {0: "legitimate", 1: "malicious"}


def test_malicious_side(p_malicious):
    for text, p in zip(MALICIOUS, p_malicious(MALICIOUS)):
        assert p > 0.5, (text, p)


def test_legitimate_side(p_malicious):
    for text, p in zip(LEGITIMATE, p_malicious(LEGITIMATE)):
        assert p < 0.5, (text, p)


@pytest.mark.xfail(reason="known weakness on casual personal SMS; see comment", strict=False)
def test_personal_borderline(p_malicious):
    for text, p in zip(PERSONAL_BORDERLINE, p_malicious(PERSONAL_BORDERLINE)):
        assert p < 0.5, (text, p)
