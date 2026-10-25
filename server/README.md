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

## Two-call flow (verdict first, highlights second)

Classification takes about 30 ms on the M1 laptop; SHAP takes up to about
4.8 s (Oct 23 benchmark, `max_evals=256`). So the app asks twice, on the same
endpoint:

1. `POST /v1/analyze?explain=false` with `{"text": ...}` → show the verdict
   and `threat_probability` right away. `text_attributions` is `[]` and
   `explanation.computed` is `false`.
2. `POST /v1/analyze` (explain defaults to true) with **the same text** →
   draw the highlights from `text_attributions`.

The server keeps no text between the calls (zero retention, Chapter 3
Section 3.5.3), so the phone must resend it; there is no request ID to look
up. Both calls return the same `verdict`, `threat_probability`, `scores`,
`decision` and `url_flags` for the same text
(`tests/server/test_two_call_flow.py`); only `text_attributions` and
`explanation` differ. If call 2 fails or times out, the app keeps the verdict
from call 1 and shows it without highlights.

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

Reports process cold start, first-request cold start, and the two headline
numbers of the two-call flow: **time to verdict** (`explain=false`) and
**time to highlights** (`explain=true`), each as median, p95 and max. These are laptop demo numbers,
not the SOP 5 result, which must be measured on the deployed Cloud Run setup.
