from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from ml.data.clean import normalize_value, dedupe_exact
from ml.data.split import remove_near_duplicates, source_disjoint_split

MIN_LEN, MAX_LEN = 3, 2048


def _rows_from_lines(lines, label: int, source: str) -> list[dict]:
    out: list[dict] = []
    for raw in lines:
        t = normalize_value(raw)
        if MIN_LEN <= len(t) <= MAX_LEN:
            out.append({"text": t, "label": label, "source": source})
    # exact-dedupe within this source
    kept = set()
    deduped = []
    for r in out:
        if r["text"] in kept:
            continue
        kept.add(r["text"])
        deduped.append(r)
    return deduped


def load_seed(datasets_dir: Path, n_shards: int = 5) -> list[dict]:
    """Read the seed corpus and spread it across ``n_shards`` mixed-label sources.

    Each ``seed_shard_k`` contains BOTH malicious and benign lines so that a
    3-way source-disjoint split cannot leave any split empty or single-label.

    Note: sharding is purely a mechanism to let the source-disjoint 3-way split
    fill all three splits from a two-file corpus. Because every ``seed_shard_k``
    is just an ``i % n_shards`` slice of the same ``malicious.txt`` /
    ``benign.txt``, "source" here carries no provenance meaning and the split is
    effectively random. ``remove_near_duplicates`` (in ``build``) is the only
    remaining leakage control.
    """
    seed = datasets_dir / "seed"
    mal = (seed / "malicious.txt").read_text(encoding="utf-8").splitlines()
    ben = (seed / "benign.txt").read_text(encoding="utf-8").splitlines()

    mal_rows = _rows_from_lines(mal, 1, "seed_malicious")
    ben_rows = _rows_from_lines(ben, 0, "seed_benign")

    rows: list[dict] = []
    for i, r in enumerate(mal_rows):
        rows.append({"text": r["text"], "label": 1,
                     "source": f"seed_shard_{i % n_shards}"})
    for i, r in enumerate(ben_rows):
        rows.append({"text": r["text"], "label": 0,
                     "source": f"seed_shard_{i % n_shards}"})
    return rows


def load_raw(datasets_dir: Path) -> list[dict]:
    from datasets.download import SOURCES

    raw = datasets_dir / "raw"
    if not raw.exists():
        return []
    by_name = {s["name"]: s for s in SOURCES}
    rows: list[dict] = []
    for dat in sorted(raw.glob("*.dat")):
        meta = by_name.get(dat.stem)
        if not meta:
            continue
        label = 1 if meta["kind"] == "malicious" else 0
        lines = dat.read_text(encoding="utf-8", errors="ignore").splitlines()
        rows.extend(_rows_from_lines(lines, label, dat.stem))
    return rows


def build(datasets_dir: Path, out_dir: Path, seed: int = 13) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_seed(datasets_dir) + load_raw(datasets_dir)
    rows = remove_near_duplicates(rows, threshold=0.9)
    parts = source_disjoint_split(rows, seed=seed)
    counts: dict[str, int] = {}
    for name, part in parts.items():
        df = pd.DataFrame(part, columns=["text", "label", "source"])
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        counts[name] = len(df)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[2]
    ap.add_argument("--datasets", type=Path, default=here / "datasets")
    ap.add_argument("--out", type=Path, default=here / "datasets" / "processed")
    args = ap.parse_args()
    print(build(args.datasets, args.out))


if __name__ == "__main__":
    main()
