from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from ml.eval.report import evaluate_model, write_model_report
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
    for sub in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        if not (sub / "run.json").exists():
            continue
        det = _load_detector(sub.name, sub)
        results[sub.name] = evaluate_model(det, test_df, adv_df)

    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    write_model_report(results, _ROOT / "reports" / "MODEL_REPORT.md")
    return results


def main() -> None:
    evaluate_all(_ROOT / "ml" / "artifacts",
                 _ROOT / "datasets" / "processed",
                 _ROOT / "datasets",
                 _ROOT / "reports")


if __name__ == "__main__":
    main()
