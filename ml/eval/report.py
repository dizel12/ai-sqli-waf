from __future__ import annotations
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from ml.eval.metrics import (  # noqa: E402
    binary_metrics, confusion_at, recall_at_fpr,
)


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
                   adv_df: pd.DataFrame, scores=None) -> dict:
    if scores is None:
        scores = detector.predict_proba(test_df["text"].tolist())
    scores = np.asarray(scores, dtype=float)
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
        "confusion": confusion_at(y, scores, 0.5),
    }


def fp_fn_rows(texts, y_true, scores, threshold: float = 0.5,
               top: int = 30) -> tuple[list[tuple[str, float]],
                                       list[tuple[str, float]]]:
    """Top false positives / false negatives, worst case first.

    FP: benign (y==0) scored >= threshold, highest score first.
    FN: malicious (y==1) scored < threshold, lowest score first.
    """
    texts = list(texts)
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    fp = [(texts[i], float(s[i]))
          for i in range(len(texts)) if s[i] >= threshold and y[i] == 0]
    fn = [(texts[i], float(s[i]))
          for i in range(len(texts)) if s[i] < threshold and y[i] == 1]
    fp.sort(key=lambda r: r[1], reverse=True)
    fn.sort(key=lambda r: r[1])
    return fp[:top], fn[:top]


def _write_rows_csv(path: Path, rows: list[tuple[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["text", "score"])
        for text, score in rows:
            w.writerow([text, f"{score:.6f}"])


def write_fp_fn_csvs(name: str, texts, y_true, scores, out_dir: Path,
                     threshold: float = 0.5, top: int = 30) -> None:
    fp, fn = fp_fn_rows(texts, y_true, scores, threshold, top)
    _write_rows_csv(out_dir / f"{name}_fp.csv", fp)
    _write_rows_csv(out_dir / f"{name}_fn.csv", fn)


def write_curves(scores_by_model: dict[str, np.ndarray], y_true,
                 out_dir: Path) -> None:
    """PR and ROC overlay curves, one line per model, on the test scores."""
    y = np.asarray(y_true, dtype=int)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in sorted(scores_by_model.items()):
        precision, recall, _ = precision_recall_curve(y, np.asarray(s, float))
        ax.plot(recall, precision, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (held-out test split)")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower left")
    fig.savefig(out_dir / "pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, s in sorted(scores_by_model.items()):
        fpr, tpr, _ = roc_curve(y, np.asarray(s, float))
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (held-out test split)")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.legend(loc="lower right")
    fig.savefig(out_dir / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def write_confusion_md(eval_json: dict, path: Path) -> None:
    lines = ["# Confusion Matrices", "",
             "_Held-out test split, decision threshold 0.5._", ""]
    for name, e in sorted(eval_json.items()):
        c = e["confusion"]
        lines += [
            f"## {name}",
            "",
            "| | predicted malicious | predicted benign |",
            "|---|---|---|",
            f"| **actual malicious** | {c['tp']} (TP) | {c['fn']} (FN) |",
            f"| **actual benign** | {c['fp']} (FP) | {c['tn']} (TN) |",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_latency_md(eval_json: dict, path: Path) -> None:
    lines = ["# Latency & Size Benchmark", "",
             "_Batch=1 `predict_proba` timing on held-out test texts._", "",
             "| model | p50 ms | p99 ms | model_bytes (MB) |",
             "|---|---|---|---|"]
    for name, e in sorted(eval_json.items()):
        lat = e["latency_ms"]
        mb = e["model_bytes"] / (1024 * 1024)
        lines.append(
            f"| {name} | {lat['p50']:.2f} | {lat['p99']:.2f} | {mb:.2f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


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
    lines += [
        "## Notes",
        "",
        "The held-out clean test split is small and easily separated: "
        "`baseline`, `cnn`, and `distilbert` all reach roughly 1.0 F1 on it, "
        "so the clean-split metrics do not discriminate between the models. "
        "The **adversarial detection rate** on obfuscated payloads (and "
        "recall-at-FPR) is therefore the metric that actually separates them.",
        "",
        "`cnn` is the active model because it has the best "
        "adversarial-detection vs. latency vs. size trade-off. Per "
        "`reports/latency.md`, `cnn` scores a request in about half a "
        "millisecond (p50) from a sub-megabyte artifact, while `distilbert` "
        "needs tens of milliseconds and hundreds of megabytes for only a few "
        "extra points of adversarial recall, and `baseline` is both slower "
        "and far larger than `cnn` while detecting the fewest obfuscated "
        "attacks.",
        "",
        "PR / ROC overlays: `reports/pr_curve.png`, `reports/roc_curve.png` "
        "(regenerated locally, not committed).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
