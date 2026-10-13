# Chalkboard

**KV cache memory calculator and monitor.**

Most production OOMs are KV cache math errors, not model size. Chalkboard predicts KV cache VRAM per model and config *before* deployment, then tracks live cache utilization against alert thresholds.

## The math

```
KV bytes per token = layers × kv_heads × head_dim × 2 (K+V) × bytes_per_element
```

GQA/MQA is exactly what `kv_heads < attention_heads` captures — fewer KV heads, proportionally less cache. A 32-layer / 8-kv-head / 128-head-dim model at FP16 costs **131,072 bytes per token**; at 32K context that's **4.29 GB per sequence**.

## Calculate (before deployment)

```bash
python chalkboard.py calc --layers 32 --kv-heads 8 --head-dim 128 \
    --context 32768 --batch 4 --bits 16 --vram-gb 24 --weights-gb 14
```

```json
{
  "kv_bytes": 17179869184,
  "kv_gb": 17.18,
  "per_token_bytes": 131072.0,
  "oom": { "fits": false, "headroom_pct": -13.9, "verdict": "oom",
           "detail": "need 31.2GB, have 24.0GB — over by 14%" }
}
```

```python
from chalkboard import oom_risk, max_context_for_vram

v = oom_risk(32, 8, 128, context_len=32768, batch_size=4, vram_bytes=24e9)
v.verdict   # 'oom' | 'tight' | 'ok'

# solve context from a VRAM budget
max_context_for_vram(32, 8, 128, vram_bytes=24e9, batch_size=4)   # tokens
```

## Monitor (in production)

Reads `kvCache` samples from Spinal traces (or any float series), reports p50/p95/peak, and verdicts against thresholds (warn 80%, critical 90%):

```bash
python chalkboard.py monitor spinal_trace.json
```

```json
{ "n": 90, "p50": 0.0421, "p95": 0.1068, "peak": 0.1069,
  "pct_samples_above_warn": 0.0, "verdict": "ok" }
```

## Part of the TechGuardCoders portfolio

Consumes [Spinal](https://github.com/TechGuardCoders/Spinal) traces directly — the fleet's own KV utilization data is the first thing it monitors. Pairs with the TensorFold engine's 1M-token context pool: the calculator says what a config *should* cost, the monitor says what it *is* costing.

## Tests

```bash
python -m pytest tests/ -q   # 24 tests
```

The math is hand-checked against known values (131,072 B/token for the reference config), GQA-vs-MHA and FP8-vs-FP16 relationships are asserted, and the monitor runs against the real Spinal trace from the fleet.
