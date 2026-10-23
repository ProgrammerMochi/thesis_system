"""Server configuration for the Increment 3 FastAPI inference service."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "roberta_weighted_seed42"

# Same truncation length the model was fine-tuned with (covers >99.9% of
# train+val messages; see docs/PROJECT_CONTEXT.md, Section 1).
MAX_LEN = 256

# Output index -> class. The saved config.json has no id2label, so the
# mapping lives here and models/ is left untouched. Source: the Increment 1
# notebook (notebooks/increment1_baselines_roberta.ipynb, Kaggle run) trains
# with num_labels=2 on label_bin (1 = malicious), takes p(malicious) as
# softmax(logits)[:, 1] in Cell 5, and reports every validation metric with
# pos_label=1. tests/test_label_mapping.py sanity-checks the direction.
ID2LABEL = {0: "legitimate", 1: "malicious"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
MALICIOUS_INDEX = LABEL2ID["malicious"]
