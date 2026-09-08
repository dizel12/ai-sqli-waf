from __future__ import annotations
import concurrent.futures
import time
import warnings
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
        if active_model in self.models:
            self.active_model = active_model
        else:
            fallback = next(iter(self.models))
            warnings.warn(
                f"ACTIVE_MODEL={active_model!r} not available; "
                f"falling back to {fallback!r}")
            self.active_model = fallback
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def score_values(self, values: list[str]) -> list[dict]:
        # The ACTIVE model is scored synchronously and unconditionally; it
        # drives the block decision and must always be present.
        active = self.models[self.active_model]
        active_scores = active.predict_proba(values)

        # Non-active ("best-effort") models are advisory only. Per spec §12 they
        # must never block the forward path past ``best_effort_budget_ms`` — so
        # each runs on a worker thread under a hard per-model deadline and the
        # first one to blow the remaining budget stops the whole best-effort
        # pass (its scores, and every later model's, stay ``None``).
        others: dict[str, list] = {}
        start = time.perf_counter()
        for name, det in self.models.items():
            if name == self.active_model:
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            remaining_ms = self.best_effort_budget_ms - elapsed_ms
            if remaining_ms <= 0:
                break
            fut = self._pool.submit(det.predict_proba, values)
            try:
                others[name] = fut.result(timeout=remaining_ms / 1000.0).tolist()
            except Exception:
                # concurrent.futures.TimeoutError (budget blown) or any model
                # error: give up on this and every remaining best-effort model.
                fut.cancel()
                others[name] = None
                break

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
