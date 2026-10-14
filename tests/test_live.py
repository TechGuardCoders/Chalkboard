"""Tests for live TensorFold metrics parsing (recorded snapshot, no network)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from live import parse_prometheus, chalkboard_view, forerunner_window

SNAPSHOT = """# HELP tensorfold_cached_tokens_total prompt tokens resumed instead of prefilled
# TYPE tensorfold_cached_tokens_total counter
tensorfold_cached_tokens_total{model="glm-5.3-flash"} 16722048
tensorfold_requests_total{model="glm-5.3-flash"} 238
tensorfold_request_errors_total{model="glm-5.3-flash"} 0
tensorfold_prompt_tokens_total{model="glm-5.3-flash"} 17199400
tensorfold_completion_tokens_total{model="glm-5.3-flash"} 92337
tensorfold_decode_seconds_total{model="glm-5.3-flash"} 1809.8
tensorfold_requests_inflight{model="glm-5.3-flash"} 1
tensorfold_requests_stalled{model="glm-5.3-flash"} 0
tensorfold_engine_fatal{model="glm-5.3-flash"} 0
"""


def test_parse_prometheus_extracts_values():
    m = parse_prometheus(SNAPSHOT)
    assert m["tensorfold_requests_total"] == 238
    assert m["tensorfold_decode_seconds_total"] == 1809.8


def test_parse_prometheus_ignores_comments_and_garbage():
    m = parse_prometheus("# comment\nnot a metric line\n" + SNAPSHOT)
    assert "tensorfold_requests_total" in m
    assert len(m) == 9


def test_cache_hit_ratio_from_real_fleet_snapshot():
    v = chalkboard_view(parse_prometheus(SNAPSHOT))
    # 16722048 / 17199400 — the fleet's actual shared-prefix reuse
    assert 0.95 < v["cache_hit_ratio"] < 0.99


def test_decode_throughput():
    v = chalkboard_view(parse_prometheus(SNAPSHOT))
    assert 40 < v["decode_tok_s"] < 60  # ~51 tok/s aggregate


def test_error_rate_zero_on_healthy_snapshot():
    v = chalkboard_view(parse_prometheus(SNAPSHOT))
    assert v["error_rate"] == 0.0
    assert v["engine_fatal"] == 0.0


def test_empty_metrics_no_div_zero():
    v = chalkboard_view({})
    assert v["cache_hit_ratio"] == 0.0
    assert v["decode_tok_s"] == 0.0
    assert v["error_rate"] == 0.0


def test_forerunner_window_is_flat_dict():
    w = forerunner_window(parse_prometheus(SNAPSHOT))
    assert isinstance(w, dict)
    assert all(isinstance(k, str) for k in w)
    assert w["requests_inflight"] == 1.0
