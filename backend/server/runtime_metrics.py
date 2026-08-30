"""Small dependency-free runtime metrics for the local digital-twin server."""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class RuntimeMetrics:
    """Record bounded latency windows and monotonic counters.

    The schema deliberately follows the counter/histogram split used by modern
    observability systems, while remaining usable without an external metrics
    service.  An OpenTelemetry exporter can consume the same names later.
    """

    def __init__(self, window_size: int = 2048) -> None:
        if int(window_size) <= 0:
            raise ValueError("window_size must be positive")
        self.started_monotonic = time.monotonic()
        self._window_size = int(window_size)
        self._counters: dict[str, int] = {}
        self._windows: dict[str, deque[float]] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        self._counters[str(name)] = self._counters.get(str(name), 0) + int(amount)

    def observe(self, name: str, value: float) -> None:
        number = float(value)
        if not math.isfinite(number):
            return
        window = self._windows.setdefault(
            str(name), deque(maxlen=self._window_size)
        )
        window.append(number)

    def snapshot(self) -> dict[str, Any]:
        windows: dict[str, dict[str, float | int | None]] = {}
        for name, values in self._windows.items():
            samples = list(values)
            windows[name] = {
                "samples": len(samples),
                "mean": sum(samples) / len(samples) if samples else None,
                "p50": _percentile(samples, 50.0),
                "p95": _percentile(samples, 95.0),
                "p99": _percentile(samples, 99.0),
                "maximum": max(samples) if samples else None,
            }
        return {
            "uptime_s": max(0.0, time.monotonic() - self.started_monotonic),
            "counters": dict(self._counters),
            "windows": windows,
        }
