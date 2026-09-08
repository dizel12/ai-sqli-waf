"""Fetch public datasets into datasets/raw/. Network-optional."""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

RAW = Path(__file__).parent / "raw"

SOURCES: list[dict] = [
    # Fill url/sha256 when finalized. license MUST be set before use.
    {"name": "payloads_all_the_things_sqli", "url": "", "license": "MIT",
     "sha256": "", "kind": "malicious"},
    {"name": "sqlmap_payloads", "url": "", "license": "GPL-2.0",
     "sha256": "", "kind": "malicious"},
    {"name": "kaggle_sqli", "url": "", "license": "CC0-1.0",
     "sha256": "", "kind": "malicious"},
    {"name": "csic_2010_http", "url": "", "license": "research-use",
     "sha256": "", "kind": "benign"},
]


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def fetch(dest: Path = RAW, offline: bool = False) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src in SOURCES:
        if not src["license"]:
            raise ValueError(f"source {src['name']} has no license set")
        if offline or not src["url"]:
            print(f"skip {src['name']} (offline or no url)")
            continue
        out = dest / f"{src['name']}.dat"
        with urlopen(src["url"]) as resp:  # noqa: S310 - trusted list
            out.write_bytes(resp.read())
        if src["sha256"] and _sha256(out) != src["sha256"]:
            raise ValueError(f"sha256 mismatch for {src['name']}")
        written.append(out)
    return written


if __name__ == "__main__":
    fetch(offline="--offline" in sys.argv)
