from __future__ import annotations
import time
from pathlib import Path

from ml.models.base import get_detector
import ml.models.baseline  # noqa: F401

_ALL = ["baseline", "cnn", "distilbert"]


class ModelRegistry:
    def __init__(self, artifacts_dir: Path, active_model: str = "cnn",
                 threshold: float = 0.5, best_effort_budget_ms: float = 120.0):
        self.threshold = threshold
        self.best_effort_budget_ms = best_effort_budget_ms
        self.models: dict = {}
        for name in _ALL:
            sub = artifacts_dir / name
            if not (sub / "run.json").exists():
                continue
            if name == "cnn":
                import ml.models.cnn  # noqa: F401
            elif name == "distilbert":
                import ml.models.distilbert  # noqa: F401
            self.models[name] = get_detector(name).load(sub)
        if not self.models:
            raise RuntimeError("no model artifacts found")
        self.active_model = active_model if active_model in self.models \
            else next(iter(self.models))

    def score_values(self, values: list[str]) -> list[dict]:
        active = self.models[self.active_model]
        active_scores = active.predict_proba(values)

        others: dict[str, list] = {}
        start = time.perf_counter()
        for name, det in self.models.items():
            if name == self.active_model:
                continue
            if (time.perf_counter() - start) * 1000.0 > self.best_effort_budget_ms:
                break
            try:
                others[name] = det.predict_proba(values).tolist()
            except Exception:
                others[name] = None

        results: list[dict] = []
        for i, v in enumerate(values):
            scores: dict[str, float | None] = {n: None for n in _ALL}
            scores[self.active_model] = float(active_scores[i])
            for name, arr in others.items():
                scores[name] = None if arr is None else float(arr[i])
            score = float(active_scores[i])
            results.append({
                "value": v,
                "scores": scores,
                "active_model": self.active_model,
                "score": score,
                "decision": "malicious" if score >= self.threshold else "benign",
                "threshold": self.threshold,
            })
        return results
