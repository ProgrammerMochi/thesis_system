# Increment 3: FastAPI server

Run every command from the repo root with the venv active (`source .venv/bin/activate`).

- Server only: `pip install -r server/requirements.txt`
- Server and tests: `pip install -r server/requirements-dev.txt`, then `python -m pytest tests/`

## Two modes

| Mode | Command | What it does |
| --- | --- | --- |
| Stub | `SMISHING_MODE=stub python -m server` | Fixed, valid Figure 3 response. Loads no model. For building the Android client. |
| Real | `python -m server` | Loads `models/roberta_weighted_seed42`, classifies, explains with SHAP. |

The server listens on port 8000 on all interfaces, so a phone on the same Wi-Fi
can reach `http://<laptop-ip>:8000`. One worker, no access log.

## Endpoints

- `POST /v1/analyze` with `{"text": "..."}` (the text the phone already masked). Add `?explain=false` to skip SHAP.
- `GET /healthz`, `GET /readyz` (200 once the model is loaded), `GET /v1/meta` (model, hashes, settings, versions).

Response fields: `verdict` (`malicious` or `legitimate`), `threat_probability`
(p_text until URL fusion exists), `text_attributions` (`start`/`end` in UTF-16
units of the text the phone sent, `weight` = SHAP value toward malicious),
`url_flags`, plus `scores`, `decision`, `explanation`, `model_version`.

**Provisional:** the verdict uses p_text only with threshold 0.5. The
validation-selected threshold and URL fusion come in Increment 2.

## Settings (environment variables)

`SMISHING_MODE` (real/stub), `SMISHING_MODEL_DIR`, `SMISHING_THRESHOLD` (0.5),
`SMISHING_MAX_CHARS` (1600), `SMISHING_SHAP_MAX_EVALS` (256, provisional until the
validation sweep), `SMISHING_SHAP_BATCH_SIZE` (16), `SMISHING_TORCH_THREADS` (4),
`SMISHING_WARMUP` (1; set 0 to measure cold start), `SMISHING_HOST`, `SMISHING_PORT`.

## Zero retention

No request body or message text is logged, cached or stored. Error responses
never echo input (FastAPI's default 422 body does, so it is replaced).
`tests/server/test_privacy.py` checks this with a canary string.

## Latency benchmark

    python -m server.bench.latency --restarts 10 --warm 20 --out bench_latency.csv

Reports process cold start, first-request cold start, and warm classification
and SHAP times separately (median, p95, max). These are laptop demo numbers,
not the SOP 5 result, which must be measured on the deployed Cloud Run setup.

## SHAP budget sweep (validation only)

    python -m server.bench.shap_budget_sweep            # all 381 validation messages, ~2-3 h
    python -m server.bench.shap_budget_sweep --limit 10 --out results/shap_sweep_smoke

Compares max_evals 32, 64, 128, 256 and 512 against a 2048 reference on
validation messages (never test): Spearman correlation of word weights, top-3
word overlap, deletion faithfulness, and SHAP latency. The selection rule was
fixed in the script's docstring before any result was seen. Results go to
`results/shap_sweep/` (msg_id and numbers only; not committed). The script
never changes the server default.
