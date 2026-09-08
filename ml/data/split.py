from __future__ import annotations
import random
from collections import defaultdict

from datasketch import MinHash

from ml.data.clean import normalize_value


def _shingles(text: str, k: int = 3) -> set[str]:
    s = normalize_value(text)
    if len(s) <= k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def _minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text):
        m.update(sh.encode("utf-8"))
    return m


def remove_near_duplicates(rows, threshold: float = 0.9, num_perm: int = 64):
    kept: list = []
    kept_hashes: list[MinHash] = []
    for r in rows:
        mh = _minhash(r["text"], num_perm)
        if any(mh.jaccard(k) >= threshold for k in kept_hashes):
            continue
        kept.append(r)
        kept_hashes.append(mh)
    return kept


def source_disjoint_split(rows, ratios=(0.6, 0.2, 0.2), seed: int = 13):
    by_source: dict[str, list] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    total = len(rows)
    targets = {"train": ratios[0] * total,
               "val": ratios[1] * total,
               "test": ratios[2] * total}
    counts = {"train": 0, "val": 0, "test": 0}
    parts: dict[str, list] = {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    sources = sorted(by_source, key=lambda s: (-len(by_source[s]), s))
    # deterministic tie-break shuffle within equal sizes handled by name sort
    for src in sources:
        deficit = {k: targets[k] - counts[k] for k in parts}
        pick = max(sorted(parts), key=lambda k: deficit[k])
        parts[pick].extend(by_source[src])
        counts[pick] += len(by_source[src])
    for k in parts:
        rng.shuffle(parts[k])
    return parts
