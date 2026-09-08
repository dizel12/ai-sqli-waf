# Latency & Size Benchmark

_Batch=1 `predict_proba` timing on held-out test texts._

| model | p50 ms | p99 ms | model_bytes (MB) |
|---|---|---|---|
| baseline | 0.81 | 1.32 | 24.00 |
| cnn | 0.41 | 0.69 | 0.54 |
| distilbert | 38.40 | 39.82 | 256.33 |
