from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

from ml.models.base import get_detector
import ml.models.baseline  # noqa: F401  (registers "baseline")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _sha256_frames(*frames: pd.DataFrame) -> str:
    h = hashlib.sha256()
    for df in frames:
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def train(model_name: str, processed_dir: Path, artifacts_dir: Path,
          seed: int = 13) -> Path:
    tr = pd.read_parquet(processed_dir / "train.parquet")
    va = pd.read_parquet(processed_dir / "val.parquet")
    detector = get_detector(model_name)()

    t0 = time.time()
    detector.fit(tr["text"].tolist(), tr["label"].tolist(),
                 va["text"].tolist(), va["label"].tolist())
    elapsed = time.time() - t0

    out = artifacts_dir / model_name
    out.mkdir(parents=True, exist_ok=True)
    detector.save(out)
    (out / "run.json").write_text(json.dumps({
        "model": model_name,
        "seed": seed,
        "n_train": len(tr),
        "n_val": len(va),
        "train_seconds": round(elapsed, 3),
        "git_commit": _git_commit(),
        "data_sha256": _sha256_frames(tr, va),
    }, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["baseline", "cnn", "distilbert"])
    ap.add_argument("--processed", type=Path,
                    default=_ROOT / "datasets" / "processed")
    ap.add_argument("--artifacts", type=Path, default=_HERE / "artifacts")
    args = ap.parse_args()
    if args.model == "cnn":
        import ml.models.cnn  # noqa: F401
    elif args.model == "distilbert":
        import ml.models.distilbert  # noqa: F401
    path = train(args.model, args.processed, args.artifacts)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
