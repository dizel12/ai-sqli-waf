from __future__ import annotations
import abc
from pathlib import Path

import numpy as np

REGISTRY: dict[str, type["Detector"]] = {}


def register(cls: type["Detector"]) -> type["Detector"]:
    REGISTRY[cls.name] = cls
    return cls


def get_detector(name: str) -> type["Detector"]:
    return REGISTRY[name]


class Detector(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def fit(self, train_texts: list[str], train_labels: list[int],
            val_texts: list[str], val_labels: list[int]) -> None: ...

    @abc.abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    @abc.abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "Detector": ...
