"""Live fleet metrics — parse TensorFold's Prometheus endpoint (read-only).

The spark cluster exposes /metrics unauthenticated (read-only by design).
This module turns that text into the same metric dicts the rest of
Chalkboard/Forerunner consume, so the monitors work against LIVE fleet
data, not just recorded traces.
"""
from __future__ import annotations

import re
import urllib.request

TF_METRIC_RE = re.compile(r"^(tensorfold_[a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$")


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text -> {metric: value} (last wins)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = TF_METRIC_RE.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def fetch_metrics(url: str, timeout: float = 5.0) -> dict[str, float]:
    """Fetch and parse a Prometheus endpoint."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return parse_prometheus(resp.read().decode("utf-8"))


def chalkboard_view(metrics: dict[str, float]) -> dict[str, float]:
    """Chalkboard's lens on live metrics: cache efficiency + throughput.

    - cache_hit_ratio: cached_tokens / prompt_tokens (shared-prefix reuse)
    - decode_tok_s: completion tokens per decode second
    - error_rate: errors / requests
    """
    prompt = metrics.get("tensorfold_prompt_tokens_total", 0.0)
    cached = metrics.get("tensorfold_cached_tokens_total", 0.0)
    completion = metrics.get("tensorfold_completion_tokens_total", 0.0)
    decode_s = metrics.get("tensorfold_decode_seconds_total", 0.0)
    reqs = metrics.get("tensorfold_requests_total", 0.0)
    errs = metrics.get("tensorfold_request_errors_total", 0.0)
    return {
        "cache_hit_ratio": cached / prompt if prompt else 0.0,
        "decode_tok_s": completion / decode_s if decode_s else 0.0,
        "error_rate": errs / reqs if reqs else 0.0,
        "requests_inflight": metrics.get("tensorfold_requests_inflight", 0.0),
        "requests_stalled": metrics.get("tensorfold_requests_stalled", 0.0),
        "engine_fatal": metrics.get("tensorfold_engine_fatal", 0.0),
    }


def forerunner_window(metrics: dict[str, float]) -> dict[str, float]:
    """Forerunner's lens: a flat metric window from live scrape."""
    return chalkboard_view(metrics)
