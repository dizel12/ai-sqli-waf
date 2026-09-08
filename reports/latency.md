# Latency & Size Benchmark

_`predict_proba` timing on held-out test texts, batch=1 and batch=32._

| model | p50 ms | p99 ms | p50 ms (batch=32) | p99 ms (batch=32) | model_bytes (MB) |
|---|---|---|---|---|---|
| baseline | 0.75 | 1.02 | 1.48 | 1.55 | 24.00 |
| cnn | 0.30 | 0.67 | 4.69 | 5.19 | 0.54 |
| distilbert | 32.19 | 39.62 | 849.31 | 940.47 | 256.33 |
