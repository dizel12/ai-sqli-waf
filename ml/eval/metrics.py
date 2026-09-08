from __future__ import annotations
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support, roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: Sequence[int], scores: Sequence[float],
                   threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "threshold": float(threshold),
    }


def recall_at_fpr(y_true, scores, max_fpr: float = 0.001) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= max_fpr
    if not ok.any():
        return {"recall": 0.0, "threshold": 1.0, "fpr": 0.0}
    idx = np.argmax(tpr * ok)  # best tpr among allowed points
    return {"recall": float(tpr[idx]), "threshold": float(thr[idx]),
            "fpr": float(fpr[idx])}


def confusion_at(y_true, scores, threshold) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
