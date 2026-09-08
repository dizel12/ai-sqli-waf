import numpy as np
import pytest

from ml.models.base import Detector, register, get_detector, REGISTRY


@register
class _Dummy(Detector):
    name = "dummy"

    def fit(self, tt, tl, vt, vl):
        self._fitted = True

    def predict_proba(self, texts):
        return np.full(len(texts), 0.5, dtype=float)

    def save(self, path):
        path.write_text("ok")

    @classmethod
    def load(cls, path):
        return cls()


def test_registry_roundtrip():
    assert get_detector("dummy") is _Dummy
    assert "dummy" in REGISTRY


def test_predict_proba_contract():
    d = _Dummy()
    out = d.predict_proba(["a", "b", "c"])
    assert out.shape == (3,)
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_unknown_detector_raises():
    with pytest.raises(KeyError):
        get_detector("nope")
