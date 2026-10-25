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

## Demo page (for the defense)

Start the server, then open `http://localhost:8000/` (or `http://<laptop-ip>:8000/`
from a phone on the same Wi-Fi). Paste a message, or click an example, and the
page shows the verdict first, then colours the words that pushed it toward
malicious (red) or legitimate (green). It uses the same two calls as the
Android app and loads nothing from the internet, so it works on a hotspot. It
does not mask phone numbers or OTPs the way the app will, so demo with made-up
messages only. Turn it off with `SMISHING_DEMO_PAGE=0` (for example on Cloud Run).

## Endpoints

- `POST /v1/analyze` with `{"text": "..."}` (the text the phone already masked). Add `?explain=false` to skip SHAP.
- `GET /healthz`, `GET /readyz` (200 once the model is loaded), `GET /v1/meta` (model, hashes, settings, versions).

Response fields: `verdict` (`malicious` or `legitimate`), `threat_probability`
(the fused score S; equals p_text when there is no link or no URL model),
`text_attributions` (`start`/`end` in UTF-16 units of the text the phone sent,
`weight` = SHAP value toward malicious), `url_flags`, `url_analysis` (one entry
per link: UTF-16 `start`/`end`, `p_url`, `flags`, `top_features` with
log-odds contributions; never the link text), plus `scores` (`p_text`,
`p_url`, `fused`, `w`), `decision`, `explanation`, `model_version`.

## URL model and fusion (prototype)

    python -m ml.train_url_fusion     # train on TRAIN, tune w on VAL; writes models/url_fusion_v0/

A logistic regression over lexical link features (`ml/url_model.py`) gives each
link a `p_url`; a message's `p_url` is its riskiest link. The server reports
`S = w * p_text + (1 - w) * p_url` (p_text alone when there is no link), with
`w` from `models/url_fusion_v0/fusion.json`. The tuning rule is in the
training script's docstring and was committed before any validation result.
Links are never visited and shortened links are not expanded.

**Provisional:** links take the label of their message, not URL-level ground
truth. The validation-tuned threshold is applied only with
`SMISHING_USE_TUNED_THRESHOLD=1`; by default the verdict threshold stays at
0.5 until the group approves the tuned one. Without the model files the server
falls back to text-only.

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
`decision`, `url_flags` and `url_analysis` for the same text
(`tests/server/test_two_call_flow.py`); only `text_attributions` and
`explanation` differ. If call 2 fails or times out, the app keeps the verdict
from call 1 and shows it without highlights.

**Provisional:** the verdict threshold is 0.5 (see "URL model and fusion").

## Settings (environment variables)

`SMISHING_MODE` (real/stub), `SMISHING_MODEL_DIR`, `SMISHING_THRESHOLD` (0.5),
`SMISHING_MAX_CHARS` (1600), `SMISHING_SHAP_MAX_EVALS` (256, provisional until the
validation sweep), `SMISHING_SHAP_BATCH_SIZE` (16), `SMISHING_TORCH_THREADS` (4),
`SMISHING_WARMUP` (1; set 0 to measure cold start), `SMISHING_DEMO_PAGE` (1),
`SMISHING_FUSION_DIR` (`models/url_fusion_v0`; `none` = text-only), `SMISHING_USE_TUNED_THRESHOLD` (0),
`SMISHING_HOST`, `SMISHING_PORT`.

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
