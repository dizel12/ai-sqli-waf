from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from ml.eval.report import (
    evaluate_model, write_adversarial_table, write_confusion_md, write_curves,
    write_fp_fn_csvs, write_latency_md, write_model_report,
)
from ml.models.base import get_detector
import ml.models.baseline  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent


def _load_detector(name: str, art_dir: Path):
    if name == "cnn":
        import ml.models.cnn  # noqa: F401
    elif name == "distilbert":
        import ml.models.distilbert  # noqa: F401
    return get_detector(name).load(art_dir)


def evaluate_all(artifacts_dir: Path, processed_dir: Path,
                 datasets_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    adv_df = pd.read_csv(datasets_dir / "adversarial_testset.csv")

    results: dict = {}
    scores_by_model: dict = {}
    for sub in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        if not (sub / "run.json").exists():
            continue
        det = _load_detector(sub.name, sub)
        scores = det.predict_proba(test_df["text"].tolist())
        scores_by_model[sub.name] = scores
        results[sub.name] = evaluate_model(det, test_df, adv_df, scores=scores)
        write_fp_fn_csvs(sub.name, test_df["text"].tolist(),
                         test_df["label"].tolist(), scores, out_dir)

    (out_dir / "eval.json").write_text(json.dumps(results, indent=2),
                                       encoding="utf-8", newline="\n")
    write_model_report(results, out_dir / "MODEL_REPORT.md")
    write_confusion_md(results, out_dir / "confusion.md")
    write_latency_md(results, out_dir / "latency.md")
    write_adversarial_table(results, out_dir / "adversarial.md")
    if scores_by_model:
        write_curves(scores_by_model, test_df["label"].tolist(), out_dir)
    return results


def main() -> None:
    evaluate_all(_ROOT / "ml" / "artifacts",
                 _ROOT / "datasets" / "processed",
                 _ROOT / "datasets",
                 _ROOT / "reports")


if __name__ == "__main__":
    main()
