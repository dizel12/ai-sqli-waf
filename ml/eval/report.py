from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.eval.metrics import binary_metrics, recall_at_fpr


def _latency_ms(detector, samples: list[str], reps: int = 50) -> dict:
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        detector.predict_proba(samples[:1])
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99))}


def _model_bytes(detector) -> int:
    tmp = Path(".eval_size_probe")
    tmp.mkdir(exist_ok=True)
    detector.save(tmp)
    total = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp.rmdir()
    return total


def evaluate_model(detector, test_df: pd.DataFrame,
                   adv_df: pd.DataFrame) -> dict:
    scores = detector.predict_proba(test_df["text"].tolist())
    y = test_df["label"].tolist()
    test_block = binary_metrics(y, scores, 0.5)
    test_block["recall_at_fpr_0.1pct"] = recall_at_fpr(y, scores, 0.001)

    adv_scores = detector.predict_proba(adv_df["text"].tolist())
    hit = adv_scores >= 0.5
    by_tech: dict[str, float] = {}
    for tech in sorted(adv_df["technique"].unique()):
        mask = (adv_df["technique"] == tech).to_numpy()
        by_tech[tech] = float(hit[mask].mean())

    return {
        "test": test_block,
        "adversarial": {"detection_rate": float(hit.mean()),
                        "by_technique": by_tech},
        "latency_ms": _latency_ms(detector, test_df["text"].tolist()),
        "model_bytes": _model_bytes(detector),
    }


def write_model_report(eval_json: dict, path: Path) -> None:
    lines = ["# Model Report", "",
             "| model | F1 | PR-AUC | ROC-AUC | Recall@FPR0.1% | Adv. detect | p50 ms | size |",
             "|---|---|---|---|---|---|---|---|"]
    for name, e in sorted(eval_json.items()):
        t = e["test"]
        lines.append(
            f"| {name} | {t['f1']:.3f} | {t['pr_auc']:.3f} | {t['roc_auc']:.3f} "
            f"| {t['recall_at_fpr_0.1pct']['recall']:.3f} "
            f"| {e['adversarial']['detection_rate']:.3f} "
            f"| {e['latency_ms']['p50']:.2f} | {e['model_bytes']} |")
    lines += ["", "_Active model rationale: see spec §6. Regenerate with "
              "`python -m ml.evaluate`._", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
