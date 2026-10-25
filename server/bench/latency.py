"""Latency benchmark for Increment 3 (design doc, Section 5).

Measures, per Chapter 3 (Section 3.3.3), classification and SHAP separately
and cold start apart from warm requests:
  - process cold start: server launch -> /readyz 200
  - first-request cold start: first /v1/analyze with SMISHING_WARMUP=0
  - warm requests: N two-call pairs per message, as the app does them:
    call 1 ?explain=false (time to verdict), then call 2 with SHAP;
    time to highlights = call 1 + call 2
Server stage times come from the Server-Timing header. The CSV holds numbers
only (message index, never text).

    python -m server.bench.latency --restarts 10 --warm 20 --out bench.csv

These are MacBook demo numbers; SOP 5 must be measured on the deployed
Cloud Run configuration.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MESSAGES = [l for l in (Path(__file__).with_name("synthetic.txt")).read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]


def post(base, text, explain):
    req = urllib.request.Request(f"{base}/v1/analyze?explain={'true' if explain else 'false'}",
                                 data=json.dumps({"text": text}).encode(),
                                 headers={"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
        header = r.headers.get("Server-Timing", "")
    client_ms = (time.perf_counter() - t) * 1000
    stages = {p.split(";dur=")[0].strip(): float(p.split(";dur=")[1]) for p in header.split(",") if ";dur=" in p}
    return client_ms, stages


def wait_ready(base, timeout=300):
    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        try:
            with urllib.request.urlopen(f"{base}/readyz", timeout=2) as r:
                if r.status == 200:
                    return (time.perf_counter() - t) * 1000
        except (urllib.error.URLError, ConnectionError):
            pass
        time.sleep(0.1)
    raise SystemExit("server did not become ready")


def summarize(rows, kind, key, label=None):
    vals = sorted(r[key] for r in rows if r["kind"] == kind and r.get(key) is not None)
    if not vals:
        return
    p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
    print(f"{label or kind + ' ' + key:30s} n={len(vals):4d} median={statistics.median(vals):8.1f} "
          f"p95={p95:8.1f} max={vals[-1]:8.1f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--warm", type=int, default=20, help="warm requests per message and mode")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", default="bench_latency.csv")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    env = {**os.environ, "SMISHING_WARMUP": "0", "SMISHING_PORT": str(a.port), "SMISHING_HOST": "127.0.0.1"}
    rows = []
    for run in range(a.restarts):
        proc = subprocess.Popen([sys.executable, "-m", "server"], cwd=ROOT, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            rows.append({"run": run, "kind": "process_cold", "msg": "", "client_ms": wait_ready(base)})
            ms, st = post(base, MESSAGES[run % len(MESSAGES)], True)
            rows.append({"run": run, "kind": "first_request", "msg": run % len(MESSAGES), "client_ms": ms, **st})
            if run == 0:
                with urllib.request.urlopen(f"{base}/v1/meta") as r:
                    meta = json.loads(r.read())
                # The app's two-call flow: call 1 (explain=false) then call 2
                # (explain=true) with the same text, back to back.
                for i, m in enumerate(MESSAGES):
                    for _ in range(a.warm):
                        ms1, st1 = post(base, m, False)
                        ms2, st2 = post(base, m, True)
                        rows.append({"run": run, "kind": "warm_classify", "msg": i, "client_ms": ms1, **st1})
                        rows.append({"run": run, "kind": "warm_shap", "msg": i, "client_ms": ms2, **st2})
                        rows.append({"run": run, "kind": "two_call", "msg": i, "client_ms": ms1 + ms2})
        finally:
            proc.terminate()
            proc.wait()
    keys = ["run", "kind", "msg", "client_ms", "queue", "preprocess", "classify", "shap", "map", "total"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("meta:", json.dumps({k: meta.get(k) for k in ("model_version", "torch_threads", "shap", "versions")}))
    print("== headline (warm, client-side, laptop only; network and phone not included) ==")
    summarize(rows, "warm_classify", "client_ms", "TIME TO VERDICT (call 1)")
    summarize(rows, "two_call", "client_ms", "TIME TO HIGHLIGHTS (call 1+2)")
    summarize(rows, "warm_shap", "client_ms", "  of which call 2 alone")
    print("== detail ==")
    summarize(rows, "process_cold", "client_ms")
    for kind in ("first_request", "warm_classify", "warm_shap"):
        for key in ("client_ms", "classify", "shap", "total"):
            summarize(rows, kind, key)
    print(f"wrote {a.out} (numbers only)")


if __name__ == "__main__":
    main()
