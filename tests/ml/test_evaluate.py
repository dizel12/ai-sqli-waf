import pandas as pd
from ml.models.baseline import BaselineDetector
from ml.eval.report import evaluate_model


def test_evaluate_model_shape(tmp_path):
    tr = pd.DataFrame(
        {"text": ["' OR 1=1 -- ", "admin' -- ", "' UNION SELECT 1 -- ",
                  "mouse", "keyboard", "lamp"],
         "label": [1, 1, 1, 0, 0, 0]})
    d = BaselineDetector()
    d.fit(tr["text"].tolist(), tr["label"].tolist(),
          tr["text"].tolist(), tr["label"].tolist())

    test_df = tr.copy()
    adv_df = pd.DataFrame({"text": ["' /*!UNION*/ SELECT 1 -- ",
                                    "' Or 1=1 -- "],
                           "technique": ["inline-comment", "case-mixing"]})
    res = evaluate_model(d, test_df, adv_df)
    assert set(res) == {"test", "adversarial", "latency_ms", "model_bytes"}
    assert 0.0 <= res["adversarial"]["detection_rate"] <= 1.0
    assert "p50" in res["latency_ms"]
