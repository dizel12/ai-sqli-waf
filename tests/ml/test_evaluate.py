import pandas as pd
from ml.models.baseline import BaselineDetector
from ml.eval.report import (
    _latency_ms, evaluate_model, fp_fn_rows, write_adversarial_table,
)


def _tiny_detector():
    tr = pd.DataFrame(
        {"text": ["' OR 1=1 -- ", "admin' -- ", "' UNION SELECT 1 -- ",
                  "mouse", "keyboard", "lamp"],
         "label": [1, 1, 1, 0, 0, 0]})
    d = BaselineDetector()
    d.fit(tr["text"].tolist(), tr["label"].tolist(),
          tr["text"].tolist(), tr["label"].tolist())
    return d, tr


def test_evaluate_model_shape(tmp_path):
    d, tr = _tiny_detector()
    test_df = tr.copy()
    adv_df = pd.DataFrame({"text": ["' /*!UNION*/ SELECT 1 -- ",
                                    "' Or 1=1 -- "],
                           "technique": ["inline-comment", "case-mixing"]})
    res = evaluate_model(d, test_df, adv_df)
    assert set(res) == {"test", "adversarial", "latency_ms", "model_bytes",
                        "confusion"}
    assert 0.0 <= res["adversarial"]["detection_rate"] <= 1.0
    assert "p50" in res["latency_ms"]


def test_evaluate_model_confusion_key(tmp_path):
    d, tr = _tiny_detector()
    adv_df = pd.DataFrame({"text": ["' Or 1=1 -- "], "technique": ["case-mixing"]})
    res = evaluate_model(d, tr.copy(), adv_df)
    conf = res["confusion"]
    assert set(conf) == {"tp", "fp", "tn", "fn"}
    assert all(isinstance(v, int) for v in conf.values())
    # 6 rows, 3 positives / 3 negatives -> counts sum to the row count
    assert conf["tp"] + conf["fp"] + conf["tn"] + conf["fn"] == len(tr)


def test_fp_fn_rows_columns_and_worst_first_ordering():
    texts = ["fp_high", "fp_low", "tn_ok", "fn_low", "fn_high", "tp_ok"]
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.95, 0.60, 0.10, 0.05, 0.40, 0.99]
    fp, fn = fp_fn_rows(texts, y_true, scores, threshold=0.5, top=30)

    # FP = benign scored >= 0.5, highest (worst) first
    assert [t for t, _ in fp] == ["fp_high", "fp_low"]
    assert fp[0][1] >= fp[1][1]
    # FN = malicious scored < 0.5, lowest (worst) first
    assert [t for t, _ in fn] == ["fn_low", "fn_high"]
    assert fn[0][1] <= fn[1][1]
    # rows are (text, float) pairs
    assert all(isinstance(t, str) and isinstance(s, float) for t, s in fp + fn)


def test_latency_ms_has_batch1_and_batch32_keys():
    d, tr = _tiny_detector()
    lat = _latency_ms(d, tr["text"].tolist(), reps=2)
    assert set(lat) == {"p50", "p99", "p50_batch32", "p99_batch32"}
    assert all(isinstance(v, float) and v >= 0.0 for v in lat.values())


def test_write_adversarial_table_has_row_per_technique(tmp_path):
    eval_json = {
        "baseline": {"adversarial": {"by_technique": {
            "inline-comment": 0.5, "case-mixing": 0.9}}},
        "cnn": {"adversarial": {"by_technique": {
            "inline-comment": 0.8, "whitespace": 1.0}}},
    }
    path = tmp_path / "adversarial.md"
    write_adversarial_table(eval_json, path)
    text = path.read_text(encoding="utf-8")
    for tech in ("inline-comment", "case-mixing", "whitespace"):
        assert any(line.startswith(f"| {tech} |")
                   for line in text.splitlines()), tech
    assert "| technique | baseline | cnn |" in text


def test_fp_fn_rows_respects_top_limit():
    texts = [f"b{i}" for i in range(10)]
    y_true = [0] * 10
    scores = [0.5 + i / 100 for i in range(10)]
    fp, fn = fp_fn_rows(texts, y_true, scores, threshold=0.5, top=3)
    assert len(fp) == 3
    assert fn == []
    # highest scores kept, sorted worst-first
    assert [t for t, _ in fp] == ["b9", "b8", "b7"]
