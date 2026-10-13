"""
Chalkboard — KV cache memory calculator and monitor.

Most production OOMs are KV cache math errors, not model size. Chalkboard
does two things:

1. CALCULATE — predict KV cache VRAM per model and config BEFORE deployment:
   bytes = layers x kv_heads x head_dim x 2 (K+V) x bytes_per_element
         x batch_size x context_len x (tp_shards absorbed into kv_heads)

2. MONITOR — track live cache utilization from a metrics endpoint
   (vLLM/TensorFold expose kv_cache_usage_perc; Spinal traces record
   kvCache samples). Alert thresholds: warn at 80%, critical at 90%.

Public API:
    model_kv_bytes(...)          -> bytes for one sequence
    total_kv_bytes(...)          -> bytes for batch x context
    kv_per_token(...)            -> bytes per token (the constant)
    max_context_for_vram(...)    -> solve context from a VRAM budget
    oom_risk(...)                -> (fits, headroom_pct, verdict)
    parse_kv_samples(...)        -> utilization series from trace/metrics
    utilization_stats(...)       -> p50/p95/max + threshold verdicts

CLI:
    python chalkboard.py calc --layers 32 --kv-heads 8 --head-dim 128 \
        --bits 16 --batch 4 --context 32768
    python chalkboard.py monitor spinal_trace.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

__version__ = "0.1.0"

WARN_THRESHOLD = 0.80
CRIT_THRESHOLD = 0.90


# ---------------------------------------------------------------- calculate

def kv_per_token(layers: int, kv_heads: int, head_dim: int, bits: int = 16) -> float:
    """Bytes of KV cache per token, per sequence.

    K and V each store kv_heads x head_dim elements per layer per token.
    GQA/MQA: kv_heads < attention_heads is exactly what this captures.
    """
    bytes_per_element = bits / 8
    return layers * kv_heads * head_dim * 2 * bytes_per_element


def model_kv_bytes(layers: int, kv_heads: int, head_dim: int, context_len: int,
                   bits: int = 16) -> int:
    """KV cache bytes for ONE sequence of context_len tokens."""
    return int(kv_per_token(layers, kv_heads, head_dim, bits) * context_len)


def total_kv_bytes(layers: int, kv_heads: int, head_dim: int, context_len: int,
                   batch_size: int, bits: int = 16) -> int:
    """KV cache bytes for batch_size sequences at context_len each."""
    return model_kv_bytes(layers, kv_heads, head_dim, context_len, bits) * batch_size


def max_context_for_vram(layers: int, kv_heads: int, head_dim: int,
                         vram_bytes: int, batch_size: int = 1, bits: int = 16,
                         model_weights_bytes: int = 0, overhead_bytes: int = 0) -> int:
    """Largest context_len that fits: (vram - weights - overhead) / (per-token x batch)."""
    budget = vram_bytes - model_weights_bytes - overhead_bytes
    if budget <= 0:
        return 0
    per_tok = kv_per_token(layers, kv_heads, head_dim, bits) * batch_size
    return int(budget / per_tok)


@dataclass
class OomVerdict:
    fits: bool
    headroom_pct: float      # positive = spare VRAM, negative = over budget
    verdict: str             # ok | tight | oom
    detail: str

    def to_dict(self) -> dict:
        return {"fits": self.fits, "headroom_pct": round(self.headroom_pct, 1),
                "verdict": self.verdict, "detail": self.detail}


def oom_risk(layers: int, kv_heads: int, head_dim: int, context_len: int,
             batch_size: int, vram_bytes: int, bits: int = 16,
             model_weights_bytes: int = 0, overhead_bytes: int = 0) -> OomVerdict:
    """Will this config OOM? Verdict: ok (>20% headroom), tight (<20%), oom."""
    need = total_kv_bytes(layers, kv_heads, head_dim, context_len, batch_size, bits) \
        + model_weights_bytes + overhead_bytes
    if vram_bytes <= 0:
        return OomVerdict(False, -100.0, "oom", "no VRAM budget given")
    headroom = (vram_bytes - need) / vram_bytes * 100
    if headroom < 0:
        return OomVerdict(False, headroom, "oom",
                          f"need {need/1e9:.1f}GB, have {vram_bytes/1e9:.1f}GB — over by {abs(headroom):.0f}%")
    if headroom < 20:
        return OomVerdict(True, headroom, "tight",
                          f"fits with {headroom:.0f}% headroom — risky under load spikes")
    return OomVerdict(True, headroom, "ok",
                      f"fits with {headroom:.0f}% headroom")


# ---------------------------------------------------------------- monitor

def parse_kv_samples(source) -> list[float]:
    """Extract a kv-cache utilization series (0..1) from a source.

    Accepts:
    - a Spinal trace dict (traces[].samples[].kvCache)
    - a path to one (JSON file)
    - a list of floats
    """
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        data = source

    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [float(v) for v in data]

    if isinstance(data, dict) and "traces" in data:
        vals = []
        for t in data["traces"]:
            for s in t.get("samples", []):
                if "kvCache" in s:
                    vals.append(float(s["kvCache"]))
        return vals

    raise ValueError("unrecognized kv sample source")


def utilization_stats(samples: list[float], warn: float = WARN_THRESHOLD,
                       crit: float = CRIT_THRESHOLD) -> dict:
    """Stats + threshold verdicts for a utilization series (0..1)."""
    if not samples:
        return {"n": 0, "verdict": "no-data"}
    s = sorted(samples)
    n = len(s)
    p50 = s[n // 2]
    p95 = s[min(n - 1, int(0.95 * n))]
    peak = s[-1]
    if peak >= crit:
        verdict = "critical"
    elif peak >= warn:
        verdict = "warn"
    else:
        verdict = "ok"
    return {
        "n": n, "p50": round(p50, 4), "p95": round(p95, 4), "peak": round(peak, 4),
        "warn_threshold": warn, "crit_threshold": crit,
        "pct_samples_above_warn": round(sum(1 for v in samples if v >= warn) / n * 100, 1),
        "verdict": verdict,
    }


def monitor(source, warn: float = WARN_THRESHOLD, crit: float = CRIT_THRESHOLD) -> dict:
    """One-call monitor: parse + stats."""
    return utilization_stats(parse_kv_samples(source), warn, crit)


# ---------------------------------------------------------------- CLI

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Chalkboard — KV cache calculator + monitor")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calc", help="predict KV cache bytes")
    c.add_argument("--layers", type=int, required=True)
    c.add_argument("--kv-heads", type=int, required=True)
    c.add_argument("--head-dim", type=int, required=True)
    c.add_argument("--context", type=int, required=True)
    c.add_argument("--batch", type=int, default=1)
    c.add_argument("--bits", type=int, default=16)
    c.add_argument("--vram-gb", type=float, default=0, help="VRAM budget for OOM check")
    c.add_argument("--weights-gb", type=float, default=0)

    m = sub.add_parser("monitor", help="monitor kv utilization from a trace file")
    m.add_argument("trace")

    args = p.parse_args()
    if args.cmd == "calc":
        need = total_kv_bytes(args.layers, args.kv_heads, args.head_dim,
                              args.context, args.batch, args.bits)
        out = {"kv_bytes": need, "kv_gb": round(need / 1e9, 3),
               "per_token_bytes": kv_per_token(args.layers, args.kv_heads, args.head_dim, args.bits)}
        if args.vram_gb:
            v = oom_risk(args.layers, args.kv_heads, args.head_dim, args.context,
                         args.batch, int(args.vram_gb * 1e9), args.bits,
                         int(args.weights_gb * 1e9))
            out["oom"] = v.to_dict()
        print(json.dumps(out, indent=2))
    else:
        print(json.dumps(monitor(args.trace), indent=2))


if __name__ == "__main__":
    main()
