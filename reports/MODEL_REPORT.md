# Model Report

| model | F1 | PR-AUC | ROC-AUC | Recall@FPR0.1% | Adv. detect | p50 ms | size |
|---|---|---|---|---|---|---|---|
| baseline | 0.971 | 0.999 | 0.999 | 0.971 | 0.808 | 0.75 | 25168448 |
| cnn | 1.000 | 1.000 | 1.000 | 1.000 | 0.885 | 0.42 | 563332 |
| distilbert | 1.000 | 1.000 | 1.000 | 1.000 | 0.904 | 37.34 | 268778634 |

_Active model rationale: see spec §6. Regenerate with `python -m ml.evaluate`._
