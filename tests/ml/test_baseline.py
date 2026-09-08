import numpy as np
from ml.models.baseline import BaselineDetector


TRAIN_X = ["' OR 1=1 -- ", "admin' -- ", "' UNION SELECT a,b -- ",
           "mouse", "wireless keyboard", "laptop stand"]
TRAIN_Y = [1, 1, 1, 0, 0, 0]


def test_baseline_learns_seed_separation(tmp_path):
    d = BaselineDetector()
    d.fit(TRAIN_X, TRAIN_Y, TRAIN_X, TRAIN_Y)
    p = d.predict_proba(["' OR 1=1 -- ", "keyboard"])
    assert p.shape == (2,)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[0] > p[1]


def test_baseline_save_load_roundtrip(tmp_path):
    d = BaselineDetector()
    d.fit(TRAIN_X, TRAIN_Y, TRAIN_X, TRAIN_Y)
    before = d.predict_proba(["' OR 1=1 -- "])
    d.save(tmp_path)
    d2 = BaselineDetector.load(tmp_path)
    after = d2.predict_proba(["' OR 1=1 -- "])
    assert np.allclose(before, after)
