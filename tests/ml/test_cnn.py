import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ml.models.cnn import CNNDetector

MAL = ["' OR 1=1 -- ", "admin' -- ", "' UNION SELECT a,b,c -- ",
       "1; DROP TABLE users -- ", "' OR SLEEP(5) -- ", "\") OR (\"1\"=\"1"]
BEN = ["wireless mouse", "mechanical keyboard", "laptop stand under $50",
       "O'Brien", "standing desk", "usb-c hub 45 dollars"]


@pytest.mark.slow
def test_cnn_overfits_tiny_set_and_roundtrips(tmp_path):
    d = CNNDetector()
    x = MAL + BEN
    y = [1] * len(MAL) + [0] * len(BEN)
    d.fit(x, y, x, y)
    p = d.predict_proba(["' OR 1=1 -- ", "wireless mouse"])
    assert p.shape == (2,)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[0] > 0.5 > p[1]

    d.save(tmp_path)
    d2 = CNNDetector.load(tmp_path)
    assert np.allclose(d.predict_proba(x), d2.predict_proba(x), atol=1e-5)
