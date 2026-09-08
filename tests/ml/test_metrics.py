import numpy as np
from ml.eval.metrics import binary_metrics, recall_at_fpr, confusion_at


def test_binary_metrics_perfect_separation():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(y, s, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_recall_at_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = np.array([0] * 1000 + [1] * 1000)
    s = np.concatenate([rng.uniform(0, 0.6, 1000), rng.uniform(0.4, 1.0, 1000)])
    out = recall_at_fpr(y, s, max_fpr=0.01)
    assert 0.0 <= out["recall"] <= 1.0
    assert out["fpr"] <= 0.01 + 1e-9


def test_confusion_at_counts():
    y = [1, 1, 0, 0]
    s = [0.9, 0.4, 0.6, 0.1]
    c = confusion_at(y, s, 0.5)
    assert c == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
