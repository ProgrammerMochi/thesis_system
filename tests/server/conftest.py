"""Shared fixtures. A tiny, randomly initialised RoBERTa with the same six
placeholder tokens stands in for the saved model, so REAL mode (tokenizer,
classifier, SHAP, offsets) is tested without the 436 MB weights. Its
predictions are meaningless; tests check plumbing, not accuracy."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PLACEHOLDERS = ["[LINK]", "[PHONE_NUM]", "[OTP]", "[AMOUNT]", "[REF]", "[EMAIL]"]
CORPUS = [
    "Hi Ma, nakarating na ako sa bahay. Kain na tayo mamaya.",
    "Your account is locked. Verify now at the link to avoid suspension.",
    "Congratulations! Ikaw ay nanalo. I-claim na bago mag-expire.",
    "Tol, tuloy pa ba tayo bukas? Dala ka ng bola.",
    "Promo ngayong linggo lang, bili na! Thanks po.",
] * 20


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tokenizers import ByteLevelBPETokenizer

    d = tmp_path_factory.mktemp("tiny_roberta")
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(CORPUS, vocab_size=400, min_frequency=1,
                            special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    bpe.save_model(str(d))
    import json
    vocab = json.loads((d / "vocab.json").read_text())
    merges = [tuple(line.split()) for line in (d / "merges.txt").read_text().splitlines()[1:] if line]
    tok = transformers.RobertaTokenizer(vocab=vocab, merges=merges)
    tok.add_special_tokens({"additional_special_tokens": PLACEHOLDERS})
    config = transformers.RobertaConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2,
                                        num_attention_heads=2, intermediate_size=64,
                                        max_position_embeddings=300, num_labels=2,
                                        pad_token_id=tok.pad_token_id)
    import torch
    torch.manual_seed(0)
    model = transformers.RobertaForSequenceClassification(config)
    tok.save_pretrained(d)
    model.save_pretrained(d)
    return d
