"""Chalkboard tests — the math must match hand-computed values exactly."""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from chalkboard import (
    kv_per_token, model_kv_bytes, total_kv_bytes, max_context_for_vram,
    oom_risk, monitor, parse_kv_samples, utilization_stats,
)


# ---------------------------------------------------------------- math (hand-checked)

def test_kv_per_token_known_value():
    # 32 layers x 8 kv-heads x 128 head_dim x 2 (K+V) x 2 bytes = 131072 B/token
    assert kv_per_token(32, 8, 128, bits=16) == 131072.0


def test_kv_per_token_gqa_smaller_than_mha():
    # GQA (8 kv heads) must be cheaper than MHA (32 heads) at same layers/dim
    assert kv_per_token(32, 8, 128) < kv_per_token(32, 32, 128)


def test_kv_per_token_fp8_half_of_fp16():
    assert kv_per_token(32, 8, 128, bits=8) == kv_per_token(32, 8, 128, bits=16) / 2


def test_model_kv_bytes_hand_checked():
    # 32L x 8H x 128D x 2 x 2B = 131072 B/token x 32768 tokens = 4,294,967,296
    assert model_kv_bytes(32, 8, 128, 32768, bits=16) == 4_294_967_296


def test_total_kv_bytes_scales_with_batch():
    one = model_kv_bytes(32, 8, 128, 8192, bits=16)
    assert total_kv_bytes(32, 8, 128, 8192, batch_size=4, bits=16) == 4 * one


def test_max_context_for_vram():
    # budget 8GB, per-token 131072 B, batch 1 -> 8e9/131072 = 61035 contexts
    n = max_context_for_vram(32, 8, 128, vram_bytes=8_000_000_000, batch_size=1, bits=16)
    assert n == int(8_000_000_000 / 131072)


def test_max_context_scales_inverse_with_batch():
    b1 = max_context_for_vram(32, 8, 128, vram_bytes=8_000_000_000, batch_size=1)
    b4 = max_context_for_vram(32, 8, 128, vram_bytes=8_000_000_000, batch_size=4)
    assert b4 == b1 // 4


def test_max_context_zero_when_weights_exceed_vram():
    assert max_context_for_vram(32, 8, 128, vram_bytes=1_000_000_000,
                                model_weights_bytes=2_000_000_000) == 0


# ---------------------------------------------------------------- oom verdicts

def test_oom_when_over_budget():
    v = oom_risk(32, 8, 128, 32768, batch_size=4, vram_bytes=8_000_000_000, bits=16)
    assert not v.fits
    assert v.verdict == "oom"
    assert v.headroom_pct < 0


def test_ok_when_plenty_of_headroom():
    v = oom_risk(32, 8, 128, 8192, batch_size=1, vram_bytes=64_000_000_000, bits=16)
    assert v.fits
    assert v.verdict == "ok"
    assert v.headroom_pct > 20


def test_tight_when_under_20pct_headroom():
    # need ~4.3GB + weights 0; give 5GB -> ~14% headroom
    v = oom_risk(32, 8, 128, 32768, batch_size=1, vram_bytes=5_000_000_000, bits=16)
    assert v.fits
    assert v.verdict == "tight"


def test_oom_risk_accounts_for_weights():
    without = oom_risk(32, 8, 128, 8192, 1, vram_bytes=10_000_000_000, bits=16)
    with_w = oom_risk(32, 8, 128, 8192, 1, vram_bytes=10_000_000_000, bits=16,
                      model_weights_bytes=4_000_000_000)
    assert with_w.headroom_pct < without.headroom_pct


# ---------------------------------------------------------------- parsing

def test_parse_kv_samples_from_trace_dict():
    trace = {"traces": [{"samples": [{"kvCache": 0.1}, {"kvCache": 0.5}]},
                        {"samples": [{"kvCache": 0.9}]}]}
    assert parse_kv_samples(trace) == [0.1, 0.5, 0.9]


def test_parse_kv_samples_from_file(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"traces": [{"samples": [{"kvCache": 0.3}]}]}))
    assert parse_kv_samples(str(f)) == [0.3]


def test_parse_kv_samples_from_float_list():
    assert parse_kv_samples([0.2, 0.4]) == [0.2, 0.4]


def test_parse_kv_samples_rejects_garbage():
    with pytest.raises(ValueError):
        parse_kv_samples({"foo": 1})


# ---------------------------------------------------------------- stats

def test_utilization_stats_percentiles():
    samples = [float(i) / 100 for i in range(1, 101)]  # 0.01..1.00
    s = utilization_stats(samples)
    assert s["n"] == 100
    assert s["p50"] == 0.51  # index 50 of sorted 0.01..1.00
    assert s["peak"] == 1.0
    assert s["verdict"] == "critical"  # peak >= 0.90


def test_utilization_ok_below_warn():
    s = utilization_stats([0.1, 0.2, 0.3])
    assert s["verdict"] == "ok"


def test_utilization_warn_band():
    s = utilization_stats([0.85])
    assert s["verdict"] == "warn"


def test_utilization_critical_band():
    s = utilization_stats([0.95])
    assert s["verdict"] == "critical"


def test_utilization_empty_is_no_data():
    assert utilization_stats([])["verdict"] == "no-data"


def test_utilization_counts_above_warn():
    s = utilization_stats([0.1, 0.85, 0.95])  # 2 of 3 above 0.80
    assert s["pct_samples_above_warn"] == pytest.approx(66.7, abs=0.1)


# ---------------------------------------------------------------- monitor end-to-end

def test_monitor_end_to_end(tmp_path):
    f = tmp_path / "trace.json"
    f.write_text(json.dumps({"traces": [{"samples": [{"kvCache": v} for v in
                                                        [0.1, 0.3, 0.85, 0.95]]}]}))
    m = monitor(str(f))
    assert m["n"] == 4
    assert m["verdict"] == "critical"
    assert m["peak"] == 0.95


def test_monitor_real_spinal_trace():
    """The actual Spinal trace from the fleet must parse and stay in a sane band."""
    p = Path.home() / "Spinal" / "traces" / "spinal_2026-09-16_225647.json"
    if not p.exists():
        pytest.skip("spinal trace not on this machine")
    m = monitor(str(p))
    assert m["n"] > 0
    assert 0.0 <= m["peak"] <= 1.0
    assert m["verdict"] in ("ok", "warn", "critical")
