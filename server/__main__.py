"""python -m server  ->  one worker, no access log (zero retention)."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.app.main:create_app", factory=True,
        host=os.environ.get("SMISHING_HOST", "0.0.0.0"),
        port=int(os.environ.get("SMISHING_PORT", "8000")),
        workers=1,          # the model is 436 MB; a second worker doubles RAM and skews latency
        access_log=False,   # access lines carry client IPs; nothing about a request is logged
    )
