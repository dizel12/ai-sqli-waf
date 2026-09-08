import json
import pandas as pd
from ml.train import train


def _make_parquet(dir_, name, rows):
    dir_.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["text", "label", "source"]).to_parquet(
        dir_ / f"{name}.parquet", index=False)


def test_train_writes_artifact_and_runjson(tmp_path):
    proc = tmp_path / "processed"
    rows = [{"text": t, "label": y, "source": "s"} for t, y in [
        ("' OR 1=1 -- ", 1), ("admin' -- ", 1), ("' UNION SELECT 1 -- ", 1),
        ("mouse", 0), ("keyboard", 0), ("desk lamp", 0)]]
    _make_parquet(proc, "train", rows)
    _make_parquet(proc, "val", rows)

    art = train("baseline", proc, tmp_path / "artifacts")
    assert (art / "model.joblib").exists()
    run = json.loads((art / "run.json").read_text())
    assert run["model"] == "baseline"
    assert run["n_train"] == 6
    assert "train_seconds" in run
