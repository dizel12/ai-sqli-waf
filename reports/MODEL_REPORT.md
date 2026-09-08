# Model Report

| model | F1 | PR-AUC | ROC-AUC | Recall@FPR0.1% | Adv. detect | p50 ms | size |
|---|---|---|---|---|---|---|---|
| baseline | 0.971 | 0.999 | 0.999 | 0.971 | 0.808 | 0.75 | 25168448 |
| cnn | 1.000 | 1.000 | 1.000 | 1.000 | 0.885 | 0.30 | 563332 |
| distilbert | 1.000 | 1.000 | 1.000 | 1.000 | 0.904 | 32.19 | 268778634 |

_Active model rationale: see spec §6. Regenerate with `python -m ml.evaluate`._

## Notes

The held-out clean test split is small and easily separated: `baseline`, `cnn`, and `distilbert` all reach roughly 1.0 F1 on it, so the clean-split metrics do not discriminate between the models. The **adversarial detection rate** on obfuscated payloads (and recall-at-FPR) is therefore the metric that actually separates them.

`cnn` is the active model because it has the best adversarial-detection vs. latency vs. size trade-off. Per `reports/latency.md`, `cnn` scores a request in about half a millisecond (p50) from a sub-megabyte artifact, while `distilbert` needs tens of milliseconds and hundreds of megabytes for only a few extra points of adversarial recall, and `baseline` is both slower and far larger than `cnn` while detecting the fewest obfuscated attacks.

PR / ROC overlays: `reports/pr_curve.png`, `reports/roc_curve.png` (regenerated locally, not committed).

Corpus & split: the committed pipeline trains on the hand-authored seed corpus only (~175 malicious + ~173 benign values). `datasets/download.py` is a stub, so `datasets/raw/` is empty and the 3-way split is filled by slicing those two files into `seed_shard_*` buckets — sharding makes the source-disjoint split effectively random, and near-duplicate removal is the only remaining leakage guard. This is why the clean held-out split is easily separated (all models near 1.0 F1) and the adversarial-technique breakdown in `reports/adversarial.md` is the discriminating metric.

Explainability: char n-gram / CNN character-importance visualizations (spec §6) are a known omission in this build — the model comparison rests on the adversarial-technique breakdown in `reports/adversarial.md`.
