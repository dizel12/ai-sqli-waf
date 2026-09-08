import numpy as np
from fastapi.testclient import TestClient

from ml.models.base import Detector, REGISTRY


class _Fake(Detector):
    name = "baseline"

    def fit(self, *a):
        ...

    def predict_proba(self, texts):
        return np.array([0.95 if "'" in t or "--" in t else 0.02
                         for t in texts], dtype=float)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.joblib").write_text("x")

    @classmethod
    def load(cls, path):
        return cls()


def _client(tmp_path, monkeypatch):
    art = tmp_path / "artifacts" / "baseline"
    art.mkdir(parents=True)
    (art / "run.json").write_text("{}")
    (art / "model.joblib").write_text("x")
    from inference_svc.app import create_app
    monkeypatch.setitem(REGISTRY, "baseline", _Fake)
    return TestClient(create_app(artifacts_dir=tmp_path / "artifacts",
                                 active_model="baseline", threshold=0.5))


def test_predict_flags_malicious_and_benign(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/predict", json={"values": ["' OR 1=1 -- ", "wireless mouse"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["decision"] == "malicious"
    assert res[0]["score"] >= 0.5
    assert res[1]["decision"] == "benign"
    assert res[0]["active_model"] == "baseline"


def test_health_lists_models(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    h = c.get("/health").json()
    assert h["status"] == "ok"
    assert "baseline" in h["models_loaded"]
