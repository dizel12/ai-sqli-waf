# Model Report

| model | F1 | PR-AUC | ROC-AUC | Recall@FPR0.1% | Adv. detect | p50 ms | size |
|---|---|---|---|---|---|---|---|
| baseline | 0.971 | 0.999 | 0.999 | 0.971 | 0.808 | 0.81 | 25168448 |
| cnn | 1.000 | 1.000 | 1.000 | 1.000 | 0.885 | 0.41 | 563332 |
| distilbert | 1.000 | 1.000 | 1.000 | 1.000 | 0.904 | 38.40 | 268778634 |

_Active model rationale: see spec §6. Regenerate with `python -m ml.evaluate`._

## Notes

The held-out clean test split is small and easily separated: `baseline`, `cnn`, and `distilbert` all reach roughly 1.0 F1 on it, so the clean-split metrics do not discriminate between the models. The **adversarial detection rate** on obfuscated payloads (and recall-at-FPR) is therefore the metric that actually separates them.

`cnn` is the active model because it has the best adversarial-detection vs. latency vs. size trade-off. Per `reports/latency.md`, `cnn` scores a request in about half a millisecond (p50) from a sub-megabyte artifact, while `distilbert` needs tens of milliseconds and hundreds of megabytes for only a few extra points of adversarial recall, and `baseline` is both slower and far larger than `cnn` while detecting the fewest obfuscated attacks.

PR / ROC overlays: `reports/pr_curve.png`, `reports/roc_curve.png` (regenerated locally, not committed).
