from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.models.base import Detector, register


@register
class BaselineDetector(Detector):
    name = "baseline"

    def __init__(self) -> None:
        self._pipe = Pipeline([
            ("vec", HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      n_features=2 ** 20, alternate_sign=False,
                                      norm="l2")),
            ("clf", CalibratedClassifierCV(
                LinearSVC(class_weight="balanced"), method="sigmoid", cv=3)),
        ])

    def fit(self, tt, tl, vt, vl) -> None:
        self._pipe.fit(tt, tl)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._pipe.predict_proba(texts)[:, 1].astype(float)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipe, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> "BaselineDetector":
        obj = cls()
        obj._pipe = joblib.load(path / "model.joblib")
        return obj
