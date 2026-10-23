"""Per-stage wall-clock timing, returned in the standard Server-Timing header
(never logged together with message text)."""
from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.ms: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = (time.perf_counter() - t) * 1000

    def header(self) -> str:
        total = (time.perf_counter() - self._t0) * 1000
        parts = [f"{k};dur={v:.1f}" for k, v in self.ms.items()]
        return ", ".join(parts + [f"total;dur={total:.1f}"])
