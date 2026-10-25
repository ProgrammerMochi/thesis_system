"""Server configuration for the Increment 3 FastAPI inference service."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "roberta_weighted_seed42"
FUSION_DIR = REPO_ROOT / "models" / "url_fusion_v0"

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


@dataclass(frozen=True)
class Settings:
    """Runtime settings, read from SMISHING_* environment variables.

    mode: "real" loads the saved model; "stub" returns a fixed, valid
    Figure 3 response without loading anything (for Android development).
    The default is "real" so a forgotten variable never shows fake verdicts.
    """
    mode: str = "real"
    model_dir: Path = MODEL_DIR
    max_len: int = MAX_LEN
    # Provisional: 0.5 on p_text only, because URL fusion (Increment 2) and the
    # validation-selected threshold (Chapter 3, Section 3.2.4) do not exist yet.
    threshold: float = 0.5
    decision_basis: str = "p_text_only_provisional"
    # About ten concatenated SMS. Longer requests are rejected with 422.
    max_chars: int = 1600
    # SHAP evaluation budget. Provisional until the validation sweep in the
    # design doc (Section 4) picks it; recorded in /v1/meta.
    shap_max_evals: int = 256
    shap_batch_size: int = 16
    torch_threads: int = 4
    warmup: bool = True
    # Browser demo page at /demo (defense fallback for the Android client).
    demo_page: bool = True
    # URL model + fusion weight from ml/train_url_fusion.py. Missing files mean
    # text-only. The fusion's validation-tuned threshold is applied only when
    # use_tuned_threshold is set (SMISHING_USE_TUNED_THRESHOLD=1); until the
    # group approves it, the verdict threshold stays at `threshold` (0.5).
    fusion_dir: Path | None = FUSION_DIR
    use_tuned_threshold: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            mode=env("SMISHING_MODE", cls.mode),
            model_dir=Path(env("SMISHING_MODEL_DIR", str(cls.model_dir))),
            threshold=float(env("SMISHING_THRESHOLD", cls.threshold)),
            max_chars=int(env("SMISHING_MAX_CHARS", cls.max_chars)),
            shap_max_evals=int(env("SMISHING_SHAP_MAX_EVALS", cls.shap_max_evals)),
            shap_batch_size=int(env("SMISHING_SHAP_BATCH_SIZE", cls.shap_batch_size)),
            torch_threads=int(env("SMISHING_TORCH_THREADS", cls.torch_threads)),
            warmup=env("SMISHING_WARMUP", "1") != "0",
            demo_page=env("SMISHING_DEMO_PAGE", "1") != "0",
            fusion_dir=(None if env("SMISHING_FUSION_DIR", "") == "none"
                        else Path(env("SMISHING_FUSION_DIR", str(FUSION_DIR)))),
            use_tuned_threshold=env("SMISHING_USE_TUNED_THRESHOLD", "0") == "1",
        )
