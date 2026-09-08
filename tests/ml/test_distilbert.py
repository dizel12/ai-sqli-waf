import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")
from ml.models.distilbert import DistilBertDetector

MAL = ["' OR 1=1 -- ", "admin' -- ", "' UNION SELECT a,b -- ",
       "1; DROP TABLE users -- "]
BEN = ["wireless mouse", "mechanical keyboard", "standing desk", "O'Brien"]


@pytest.mark.slow
def test_distilbert_contract_and_roundtrip(tmp_path):
    d = DistilBertDetector()
    x, y = MAL + BEN, [1, 1, 1, 1, 0, 0, 0, 0]
    d.fit(x, y, x, y)
    p = d.predict_proba(["' OR 1=1 -- ", "wireless mouse"])
    assert p.shape == (2,)
    assert ((p >= 0) & (p <= 1)).all()

    d.save(tmp_path)
    d2 = DistilBertDetector.load(tmp_path)
    assert np.allclose(d.predict_proba(x), d2.predict_proba(x), atol=1e-4)
