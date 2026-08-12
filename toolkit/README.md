# UCM metrics view

This branch carries the metrics-view subset of `ucm-toolkit`. It collects a
Prometheus/OpenMetrics endpoint into SQLite and evaluates the built-in
`metrics_lite` preset from a terminal.

```bash
pip install -e toolkit
ucm-toolkit run metrics-view check --url http://127.0.0.1:8000/metrics --config metrics_lite
ucm-toolkit run metrics-view start --url http://127.0.0.1:8000/metrics --config metrics_lite
ucm-toolkit run metrics-view query --config metrics_lite --window 5m
```

The tier hit-rate preset uses vLLM HBM/external token counters and UCM final
shard-source counters. It does not require a TP-size parameter. YuanRong L2 is
shown only as a diagnostic rate and is excluded from DRAM/SSD attribution.
