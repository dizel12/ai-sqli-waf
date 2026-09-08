from pathlib import Path
import csv

DATA = Path(__file__).resolve().parents[2] / "datasets"


def test_seed_files_have_enough_lines():
    mal = (DATA / "seed" / "malicious.txt").read_text(encoding="utf-8").splitlines()
    ben = (DATA / "seed" / "benign.txt").read_text(encoding="utf-8").splitlines()
    assert len([x for x in mal if x.strip()]) >= 120
    assert len([x for x in ben if x.strip()]) >= 120
    assert all(x.strip() for x in mal), "no blank lines allowed"
    assert all(x.strip() for x in ben), "no blank lines allowed"


def test_adversarial_testset_shape():
    rows = list(csv.DictReader((DATA / "adversarial_testset.csv").open(encoding="utf-8")))
    assert len(rows) >= 40
    assert set(rows[0].keys()) == {"text", "technique"}
    assert all(r["text"] and r["technique"] for r in rows)


def test_build_produces_disjoint_parquets(tmp_path):
    import pandas as pd
    from ml.data.pipeline import build

    counts = build(DATA, tmp_path, seed=13)
    assert set(counts) == {"train", "val", "test"}
    assert all(v > 0 for v in counts.values())

    frames = {name: pd.read_parquet(tmp_path / f"{name}.parquet")
              for name in counts}
    for name, df in frames.items():
        assert list(df.columns) == ["text", "label", "source"]
        assert df["label"].isin([0, 1]).all()
        assert set(df["label"].tolist()) == {0, 1}, f"{name} split is single-label"

    src_to_split = {}
    for name, df in frames.items():
        for s in df["source"].unique():
            assert s not in src_to_split
            src_to_split[s] = name
