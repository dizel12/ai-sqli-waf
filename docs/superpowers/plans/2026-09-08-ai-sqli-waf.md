# AI-SQLi-WAF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an ML-based SQL-injection detector that runs as a blocking reverse proxy in front of a deliberately vulnerable demo web app, with a model-comparison story (TF-IDF baseline / char-CNN / DistilBERT), a real-time detection log UI, and a `SECURE_MODE` toggle plus `SECURITY.md` for the defense-in-depth narrative.

**Architecture:** Four `docker-compose` services on one network. `waf-proxy` (FastAPI) receives all traffic, extracts request *values*, calls `inference-svc` (FastAPI, loads all 3 models) for a maliciousness score, then either forwards to `vuln-app` (Flask + SQLite) or returns 403. Every inspected request is written as an event to a shared SQLite file that `log-ui` streams to a single page via SSE. ML training/evaluation lives in an offline `ml/` package that produces model artifacts consumed by `inference-svc`.

**Tech Stack:** Python 3.11, FastAPI + httpx + uvicorn (proxy, inference, log-ui), Flask (vuln-app), scikit-learn (baseline), PyTorch (char-CNN), HuggingFace transformers (DistilBERT), pandas + pyarrow (data), datasketch (MinHash near-dup), pytest, Docker + docker-compose, Git LFS.

**Spec:** `docs/superpowers/specs/2026-09-08-ai-sqli-waf-design.md`

## Global Constraints

- Python 3.11 for every service and the `ml/` package.
- `waf-proxy` and `inference-svc` are CPU-only (no CUDA assumptions anywhere).
- Every `vuln-app` source file starts with the exact comment line: `# INTENTIONALLY VULNERABLE - DO NOT DEPLOY`
- Only `waf-proxy` (host `8080`) and `log-ui` (host `8081`) are published to the host in `docker-compose.yml`. `vuln-app` (`8000`) and `inference-svc` (`9000`) stay on the internal network.
- The proxy inspects request **values only**, never key names; values shorter than 3 characters are skipped.
- Proxy failure behavior is **fail-open**: if `inference-svc` errors or exceeds `INFERENCE_TIMEOUT_MS` (default `250`), forward the request and log a warning.
- Active model and block threshold are runtime config on `waf-proxy`: env vars `ACTIVE_MODEL` (default `cnn`), `BLOCK_THRESHOLD` (default `0.5`).
- `/predict` always scores the active model synchronously; non-active models are best-effort and may be `null` in the response. The offline evaluation harness always runs every model in full.
- Git LFS tracks `*.parquet` and `ml/artifacts/**`. `datasets/raw/` is gitignored.
- Model-blocking decision rule: block if the active model score `>=` `BLOCK_THRESHOLD` for **any** inspected value in the request.
- Every commit message uses Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`, `fix:`).

---

## File Structure

```
ai-sqli-waf/
├── docker-compose.yml
├── .gitattributes                 # LFS rules
├── .gitignore
├── README.md
├── SECURITY.md
├── pytest.ini
├── scripts/
│   └── demo.py                    # E2E: 4 scenarios + false-positive check
├── reports/
│   └── MODEL_REPORT.md            # generated/updated by ml/evaluate.py
├── datasets/
│   ├── download.py               # fetch public sources into raw/
│   ├── raw/                      # gitignored; SOURCES.md committed
│   │   └── SOURCES.md
│   ├── seed/                     # committed small deterministic corpus
│   │   ├── malicious.txt
│   │   └── benign.txt
│   ├── processed/                # {train,val,test}.parquet (LFS)
│   └── adversarial_testset.csv   # committed, eval-only
├── ml/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── clean.py             # value extraction + normalization
│   │   ├── split.py             # source-disjoint split + MinHash near-dup
│   │   └── pipeline.py          # CLI: raw/seed -> processed parquet
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py              # Detector ABC + registry
│   │   ├── baseline.py          # TF-IDF char n-gram + LinearSVC
│   │   ├── cnn.py               # char-CNN (PyTorch)
│   │   └── distilbert.py        # DistilBERT fine-tune
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py           # precision/recall/f1/pr-auc/roc-auc/recall@fpr
│   │   └── report.py            # curves, latency bench, MODEL_REPORT.md writer
│   ├── train.py                 # CLI: --model {baseline,cnn,distilbert}
│   └── evaluate.py              # CLI: eval all artifacts on test + adversarial
├── inference_svc/
│   ├── __init__.py
│   ├── app.py                   # FastAPI, POST /predict, GET /health
│   ├── registry.py             # load artifacts, active/best-effort scoring
│   ├── schemas.py              # pydantic request/response models
│   ├── Dockerfile
│   └── requirements.txt
├── waf_proxy/
│   ├── __init__.py
│   ├── app.py                   # FastAPI catch-all route
│   ├── extract.py              # candidate value extraction from a request
│   ├── decision.py             # call inference-svc, apply threshold, fail-open
│   ├── events.py               # SQLite events writer
│   ├── config.py               # env-var config
│   ├── Dockerfile
│   └── requirements.txt
├── vuln_app/
│   ├── __init__.py
│   ├── app.py                   # Flask app factory, 3 endpoints, SECURE_MODE
│   ├── db.py                    # connection helpers, full vs read-only user
│   ├── queries.py              # vulnerable vs safe query builders
│   ├── seed.py                 # create + seed app.db
│   ├── templates/              # minimal HTML
│   ├── traffic_gen.py          # benign traffic generator
│   ├── Dockerfile
│   └── requirements.txt
├── log_ui/
│   ├── __init__.py
│   ├── app.py                   # FastAPI: GET /events, GET /events/stream (SSE)
│   ├── static/
│   │   └── index.html          # vanilla JS single page
│   ├── Dockerfile
│   └── requirements.txt
└── tests/
    ├── conftest.py
    ├── ml/
    │   ├── test_clean.py
    │   ├── test_split.py
    │   ├── test_pipeline.py
    │   ├── test_detector_contract.py
    │   ├── test_baseline.py
    │   ├── test_cnn.py
    │   ├── test_distilbert.py
    │   └── test_metrics.py
    ├── inference_svc/
    │   └── test_predict.py
    ├── waf_proxy/
    │   ├── test_extract.py
    │   └── test_decision.py
    ├── vuln_app/
    │   ├── test_vulnerable.py
    │   └── test_secure.py
    ├── log_ui/
    │   └── test_events.py
    └── e2e/
        └── test_scenarios.py
```

---

## Task 1: Project scaffolding

**Files:**
- Create: `.gitignore`, `.gitattributes`, `pytest.ini`, `ml/__init__.py`, `ml/data/__init__.py`, `ml/models/__init__.py`, `ml/eval/__init__.py`, `inference_svc/__init__.py`, `waf_proxy/__init__.py`, `vuln_app/__init__.py`, `log_ui/__init__.py`, `tests/conftest.py`, `tests/__init__.py`
- Create: `requirements-dev.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: importable empty packages `ml`, `ml.data`, `ml.models`, `ml.eval`, `inference_svc`, `waf_proxy`, `vuln_app`, `log_ui`; `pytest` runs from repo root.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
.pytest_cache/
datasets/raw/*
!datasets/raw/SOURCES.md
*.db
reports/*.png
.DS_Store
```

- [ ] **Step 2: Create `.gitattributes`**

```gitattributes
*.parquet filter=lfs diff=lfs merge=lfs -text
ml/artifacts/** filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
filterwarnings =
    ignore::DeprecationWarning
markers =
    slow: slow tests (model training, transformers)
    e2e: requires docker-compose stack running
```

- [ ] **Step 4: Create `requirements-dev.txt`**

```
pytest==8.3.4
pandas==2.2.3
pyarrow==18.1.0
scikit-learn==1.6.1
datasketch==1.6.5
httpx==0.28.1
```

- [ ] **Step 5: Create the empty `__init__.py` files and a minimal `tests/conftest.py`**

`tests/conftest.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

All `__init__.py` files: empty.

- [ ] **Step 6: Create the virtualenv and install dev deps**

Run:
```bash
python -m venv .venv && .venv/Scripts/python -m pip install -U pip -r requirements-dev.txt
```
Expected: install completes without error.

- [ ] **Step 7: Verify pytest collects nothing and exits cleanly**

Run: `.venv/Scripts/python -m pytest`
Expected: `no tests ran` (exit code 5) — acceptable at this point.

- [ ] **Step 8: Commit**

```bash
git add .gitignore .gitattributes pytest.ini requirements-dev.txt ml inference_svc waf_proxy vuln_app log_ui tests
git commit -m "chore: scaffold packages, pytest config, LFS rules"
```

---

## Task 2: Seed corpus + dataset source registry

**Files:**
- Create: `datasets/seed/malicious.txt`, `datasets/seed/benign.txt`
- Create: `datasets/adversarial_testset.csv`
- Create: `datasets/download.py`
- Create: `datasets/raw/SOURCES.md`
- Test: `tests/ml/test_pipeline.py` (seed-load assertions only in this task)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `datasets/seed/malicious.txt`, `datasets/seed/benign.txt`: newline-delimited raw values, `>= 120` lines each, no blank lines.
  - `datasets/adversarial_testset.csv`: header `text,technique`; `>= 40` rows; all rows malicious by construction.
  - `datasets.download.SOURCES: list[dict]` with keys `name, url, license, sha256, kind` where `kind` in `{"malicious","benign"}`.
  - `datasets.download.fetch(dest: Path, offline: bool = False) -> list[Path]` — downloads each source to `dest`, skips on `offline=True`, verifies sha256 when present.

- [ ] **Step 1: Write `datasets/seed/malicious.txt`**

Hand-author at least 120 distinct SQLi payloads across categories (union, boolean-blind, time-blind, stacked, comment-terminated auth bypass, error-based, encoded). Example first lines:

```
' OR '1'='1
' OR 1=1 -- 
admin' -- 
' UNION SELECT username, password, 3, 4 FROM users -- 
1; DROP TABLE users -- 
' AND (SELECT 1 FROM (SELECT SLEEP(5))x) -- 
1 OR 1=1
' OR 'a'='a' /*
") OR ("1"="1
%27%20OR%201%3D1
```

- [ ] **Step 2: Write `datasets/seed/benign.txt`**

Hand-author at least 120 legitimate input values. Deliberately include SQL-looking-but-legitimate and long/structured values:

```
mouse
wireless keyboard
O'Brien
chair 1=1 sale
SELECT desk lamp
laptop stand under $50
john.doe@example.com
{"q":"standing desk","sort":"price"}
The quick brown fox jumps over the lazy dog near the riverbank at dawn
aGVsbG8gd29ybGQ=
2026-09-08
+82 10-1234-5678
```

- [ ] **Step 3: Write `datasets/adversarial_testset.csv`**

Header `text,technique`. At least 40 rows, each an obfuscated bypass of a payload that a naive filter would catch. Examples:

```csv
text,technique
' /*!50000UNION*/ SELECT username,password FROM users -- ,inline-comment
' UnIoN sElEcT username,password FROM users -- ,case-mixing
%2527%20OR%201%3D1,double-url-encode
' OR 0x31=0x31 -- ,hex-literal
' OR CHAR(49)=CHAR(49) -- ,char-function
'/**/OR/**/1=1/**/-- ,whitespace-comment
' UNI/**/ON SEL/**/ECT 1,2 -- ,split-keyword
```

- [ ] **Step 4: Write `datasets/raw/SOURCES.md`**

A markdown table documenting each intended public source: name, URL, license, what it contributes (malicious/benign), and a note that `raw/` contents are not committed. Include at minimum: sqlmap payloads, PayloadsAllTheThings SQLi, a Kaggle SQLi dataset, CSIC 2010 HTTP dataset.

- [ ] **Step 5: Write `datasets/download.py`**

```python
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
```

- [ ] **Step 6: Write the seed-load test in `tests/ml/test_pipeline.py`**

```python
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
```

- [ ] **Step 7: Run the test**

Run: `.venv/Scripts/python -m pytest tests/ml/test_pipeline.py -v`
Expected: PASS.

- [ ] **Step 8: Verify `download.py` runs offline without error**

Run: `.venv/Scripts/python datasets/download.py --offline`
Expected: prints `skip ...` lines, exit 0.

- [ ] **Step 9: Commit**

```bash
git add datasets
git commit -m "feat: add seed corpus, adversarial testset, dataset source registry"
```

---

## Task 3: Value extraction & normalization (`ml/data/clean.py`)

**Files:**
- Create: `ml/data/clean.py`
- Test: `tests/ml/test_clean.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `normalize_value(raw: str) -> str` — one `urllib.parse.unquote` pass, strip, collapse internal whitespace runs to a single space, lowercase-preserving (does NOT lowercase). Returns `""` for input that is only whitespace.
  - `is_inspectable(value: str, min_len: int = 3) -> bool` — `len(value.strip()) >= min_len`.
  - `extract_values_from_qs(query_string: str) -> list[str]` — parse with `urllib.parse.parse_qsl(keep_blank_values=True)`, return the values (post `normalize_value`), preserving order, dropping empties.
  - `dedupe_exact(texts: Iterable[str]) -> list[str]` — order-preserving exact-duplicate removal after `normalize_value`.

- [ ] **Step 1: Write failing tests in `tests/ml/test_clean.py`**

```python
from ml.data.clean import (
    normalize_value, is_inspectable, extract_values_from_qs, dedupe_exact,
)


def test_normalize_url_decodes_once():
    assert normalize_value("%27%20OR%201%3D1") == "' OR 1=1"


def test_normalize_collapses_whitespace_and_strips():
    assert normalize_value("  a\t\tb\n c  ") == "a b c"


def test_normalize_blank_becomes_empty():
    assert normalize_value("   \t ") == ""


def test_is_inspectable_min_len():
    assert is_inspectable("abc") is True
    assert is_inspectable("ab") is False
    assert is_inspectable("  x  ") is False


def test_extract_values_from_qs_keeps_order_drops_empty():
    assert extract_values_from_qs("q=mouse&sort=&page=2") == ["mouse", "2"]


def test_dedupe_exact_preserves_order():
    assert dedupe_exact(["a", "b", "a", "  b ", "c"]) == ["a", "b", "c"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/ml/test_clean.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.data.clean'`.

- [ ] **Step 3: Implement `ml/data/clean.py`**

```python
from __future__ import annotations
import re
from typing import Iterable
from urllib.parse import parse_qsl, unquote

_WS = re.compile(r"\s+")


def normalize_value(raw: str) -> str:
    decoded = unquote(raw)
    collapsed = _WS.sub(" ", decoded).strip()
    return collapsed


def is_inspectable(value: str, min_len: int = 3) -> bool:
    return len(value.strip()) >= min_len


def extract_values_from_qs(query_string: str) -> list[str]:
    out: list[str] = []
    for _key, val in parse_qsl(query_string, keep_blank_values=True):
        norm = normalize_value(val)
        if norm:
            out.append(norm)
    return out


def dedupe_exact(texts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        n = normalize_value(t)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ml/test_clean.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/data/clean.py tests/ml/test_clean.py
git commit -m "feat: value extraction and normalization helpers"
```

---

## Task 4: Source-disjoint split + MinHash near-dup removal (`ml/data/split.py`)

**Files:**
- Create: `ml/data/split.py`
- Test: `tests/ml/test_split.py`

**Interfaces:**
- Consumes: `ml.data.clean.normalize_value`.
- Produces:
  - `remove_near_duplicates(rows: list[dict], threshold: float = 0.9, num_perm: int = 64) -> list[dict]` — each row is `{"text": str, "label": int, "source": str}`. Removes rows whose MinHash-Jaccard (over 3-char shingles of `normalize_value(text)`) to a kept row is `>= threshold`. Order-preserving; first occurrence kept.
  - `source_disjoint_split(rows: list[dict], ratios=(0.6, 0.2, 0.2), seed: int = 13) -> dict[str, list[dict]]` — returns `{"train": [...], "val": [...], "test": [...]}`. Whole sources are assigned to exactly one split; no `source` value appears in more than one split. Assignment is deterministic given `seed`. Balances by greedily assigning sources (largest first) to the split furthest below its target row count.

- [ ] **Step 1: Write failing tests in `tests/ml/test_split.py`**

```python
from ml.data.split import remove_near_duplicates, source_disjoint_split


def _rows(pairs):
    return [{"text": t, "label": l, "source": s} for t, l, s in pairs]


def test_remove_near_duplicates_drops_trivial_variants():
    rows = _rows([
        ("SELECT * FROM users WHERE id=1", 1, "a"),
        ("SELECT * FROM users WHERE id=2", 1, "a"),   # near-dup of first
        ("completely different benign text here", 0, "b"),
    ])
    out = remove_near_duplicates(rows, threshold=0.8)
    texts = [r["text"] for r in out]
    assert "SELECT * FROM users WHERE id=1" in texts
    assert "SELECT * FROM users WHERE id=2" not in texts
    assert "completely different benign text here" in texts


def test_split_is_source_disjoint():
    rows = []
    for i in range(10):
        src = f"src{i}"
        for j in range(20):
            rows.append({"text": f"{src} sample {j}", "label": i % 2, "source": src})
    parts = source_disjoint_split(rows, seed=13)
    seen = {}
    for name, part in parts.items():
        for r in part:
            assert r["source"] not in seen or seen[r["source"]] == name
            seen[r["source"]] = name
    assert sum(len(p) for p in parts.values()) == len(rows)


def test_split_is_deterministic():
    rows = [{"text": f"s{i} t{j}", "label": 0, "source": f"s{i}"}
            for i in range(8) for j in range(15)]
    a = source_disjoint_split(rows, seed=13)
    b = source_disjoint_split(rows, seed=13)
    assert [r["text"] for r in a["train"]] == [r["text"] for r in b["train"]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/ml/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.data.split'`.

- [ ] **Step 3: Implement `ml/data/split.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ml/test_split.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/data/split.py tests/ml/test_split.py
git commit -m "feat: source-disjoint split and MinHash near-dup removal"
```

---

## Task 5: Data pipeline CLI (`ml/data/pipeline.py`)

**Files:**
- Create: `ml/data/pipeline.py`
- Modify: `tests/ml/test_pipeline.py` (add build assertions)

**Interfaces:**
- Consumes: `ml.data.clean` (`normalize_value`, `dedupe_exact`), `ml.data.split` (`remove_near_duplicates`, `source_disjoint_split`).
- Produces:
  - `load_seed(datasets_dir: Path) -> list[dict]` — reads `seed/malicious.txt` (label 1, source `seed_malicious`) and `seed/benign.txt` (label 0, source `seed_benign`), normalizes, length-filters to `3..2048`, exact-dedupes within each file. Returns rows `{"text","label","source"}`.
  - `load_raw(datasets_dir: Path) -> list[dict]` — if `raw/*.dat` files exist, parse each as newline-delimited values, label + source from `datasets.download.SOURCES` by filename stem; otherwise return `[]`.
  - `build(datasets_dir: Path, out_dir: Path, seed: int = 13) -> dict[str, int]` — combine seed + raw, `remove_near_duplicates`, `source_disjoint_split`, write `train.parquet`/`val.parquet`/`test.parquet` (columns `text,label,source`) to `out_dir`. Returns row counts per split.
  - `python -m ml.data.pipeline [--datasets DIR] [--out DIR]` runs `build`.

- [ ] **Step 1: Add failing build test to `tests/ml/test_pipeline.py`**

```python
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

    src_to_split = {}
    for name, df in frames.items():
        for s in df["source"].unique():
            assert s not in src_to_split
            src_to_split[s] = name
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_pipeline.py::test_build_produces_disjoint_parquets -v`
Expected: FAIL with `ImportError` / `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ml/data/pipeline.py`**

```python
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


def load_seed(datasets_dir: Path) -> list[dict]:
    seed = datasets_dir / "seed"
    mal = (seed / "malicious.txt").read_text(encoding="utf-8").splitlines()
    ben = (seed / "benign.txt").read_text(encoding="utf-8").splitlines()
    return (_rows_from_lines(mal, 1, "seed_malicious")
            + _rows_from_lines(ben, 0, "seed_benign"))


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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/ml/test_pipeline.py -v`
Expected: PASS (all 3 tests).

- [ ] **Step 5: Build the real processed dataset**

Run: `.venv/Scripts/python -m ml.data.pipeline`
Expected: prints counts dict; `datasets/processed/{train,val,test}.parquet` created.

- [ ] **Step 6: Commit**

```bash
git add ml/data/pipeline.py tests/ml/test_pipeline.py datasets/processed
git commit -m "feat: data pipeline CLI producing source-disjoint parquet splits"
```

---

## Task 6: Detector interface + registry (`ml/models/base.py`)

**Files:**
- Create: `ml/models/base.py`
- Test: `tests/ml/test_detector_contract.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Detector(abc.ABC)` with:
    - `name: str` (class attribute)
    - `fit(self, train_texts: list[str], train_labels: list[int], val_texts: list[str], val_labels: list[int]) -> None`
    - `predict_proba(self, texts: list[str]) -> "np.ndarray"` — shape `(len(texts),)`, dtype float, every element in `[0.0, 1.0]`.
    - `save(self, path: pathlib.Path) -> None`
    - `classmethod load(cls, path: pathlib.Path) -> "Detector"`
  - `REGISTRY: dict[str, type[Detector]]` and `register(cls)` decorator.
  - `get_detector(name: str) -> type[Detector]`.

- [ ] **Step 1: Write failing contract test `tests/ml/test_detector_contract.py`**

```python
import numpy as np
import pytest

from ml.models.base import Detector, register, get_detector, REGISTRY


@register
class _Dummy(Detector):
    name = "dummy"

    def fit(self, tt, tl, vt, vl):
        self._fitted = True

    def predict_proba(self, texts):
        return np.full(len(texts), 0.5, dtype=float)

    def save(self, path):
        path.write_text("ok")

    @classmethod
    def load(cls, path):
        return cls()


def test_registry_roundtrip():
    assert get_detector("dummy") is _Dummy
    assert "dummy" in REGISTRY


def test_predict_proba_contract():
    d = _Dummy()
    out = d.predict_proba(["a", "b", "c"])
    assert out.shape == (3,)
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_unknown_detector_raises():
    with pytest.raises(KeyError):
        get_detector("nope")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_detector_contract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.models.base'`.

- [ ] **Step 3: Implement `ml/models/base.py`**

```python
from __future__ import annotations
import abc
from pathlib import Path

import numpy as np

REGISTRY: dict[str, type["Detector"]] = {}


def register(cls: type["Detector"]) -> type["Detector"]:
    REGISTRY[cls.name] = cls
    return cls


def get_detector(name: str) -> type["Detector"]:
    return REGISTRY[name]


class Detector(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def fit(self, train_texts: list[str], train_labels: list[int],
            val_texts: list[str], val_labels: list[int]) -> None: ...

    @abc.abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...

    @abc.abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Path) -> "Detector": ...
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/ml/test_detector_contract.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/models/base.py tests/ml/test_detector_contract.py
git commit -m "feat: Detector ABC and model registry"
```

---

## Task 7: Baseline model (`ml/models/baseline.py`)

**Files:**
- Create: `ml/models/baseline.py`
- Test: `tests/ml/test_baseline.py`

**Interfaces:**
- Consumes: `ml.models.base` (`Detector`, `register`).
- Produces:
  - `class BaselineDetector(Detector)`, `name = "baseline"`.
  - Pipeline: `HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2**20, alternate_sign=False)` → `CalibratedClassifierCV(LinearSVC(class_weight="balanced"), method="sigmoid", cv=3)`.
  - `save(path)` writes `path / "model.joblib"`. `load(path)` reads it.
  - `top_ngrams(self, k: int = 20) -> list[str]` — not required by other tasks; include if cheap, else omit.

- [ ] **Step 1: Write failing test `tests/ml/test_baseline.py`**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ml/models/baseline.py`**

```python
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ml.models.base import Detector, register


@register
class BaselineDetector(Detector):
    name = "baseline"

    def __init__(self) -> None:
        self._pipe = Pipeline([
            ("vec", HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                      n_features=2 ** 20, alternate_sign=False,
                                      norm="l2")),
            ("clf", CalibratedClassifierCV(
                LinearSVC(class_weight="balanced"), method="sigmoid", cv=3)),
        ])

    def fit(self, tt, tl, vt, vl) -> None:
        self._pipe.fit(tt, tl)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self._pipe.predict_proba(texts)[:, 1].astype(float)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipe, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> "BaselineDetector":
        obj = cls()
        obj._pipe = joblib.load(path / "model.joblib")
        return obj
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/ml/test_baseline.py -v`
Expected: PASS (2 tests). If `cv=3` fails on tiny data, the test uses 3 samples/class which is the minimum — keep `cv=3`.

- [ ] **Step 5: Commit**

```bash
git add ml/models/baseline.py tests/ml/test_baseline.py
git commit -m "feat: TF-IDF char n-gram baseline detector"
```

---

## Task 8: Training CLI (`ml/train.py`)

**Files:**
- Create: `ml/train.py`
- Test: `tests/ml/test_train_cli.py`

**Interfaces:**
- Consumes: `ml.models.base.get_detector`, `ml.models.baseline` (import for registration), parquet from Task 5.
- Produces:
  - `train(model_name: str, processed_dir: Path, artifacts_dir: Path, seed: int = 13) -> Path` — loads `train.parquet` + `val.parquet`, fits, calls `detector.save(artifacts_dir / model_name)`, writes `artifacts_dir / model_name / run.json` with keys `model, seed, n_train, n_val, train_seconds, git_commit, data_sha256`. Returns the artifact dir.
  - `python -m ml.train --model baseline [--processed DIR] [--artifacts DIR]`.
  - `_git_commit() -> str` helper (returns `"unknown"` if not a git repo).

- [ ] **Step 1: Write failing test `tests/ml/test_train_cli.py`**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_train_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.train'`.

- [ ] **Step 3: Implement `ml/train.py`**

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/ml/test_train_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Train the baseline on real data**

Run: `.venv/Scripts/python -m ml.train --model baseline`
Expected: prints `wrote .../ml/artifacts/baseline`; `model.joblib` + `run.json` present.

- [ ] **Step 6: Commit**

```bash
git add ml/train.py tests/ml/test_train_cli.py ml/artifacts/baseline
git commit -m "feat: training CLI with run metadata"
```

---

## Task 9: Metrics module (`ml/eval/metrics.py`)

**Files:**
- Create: `ml/eval/metrics.py`
- Test: `tests/ml/test_metrics.py`

**Interfaces:**
- Consumes: nothing (numpy + sklearn only).
- Produces:
  - `binary_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> dict` with keys `precision, recall, f1, pr_auc, roc_auc, threshold`.
  - `recall_at_fpr(y_true, scores, max_fpr: float = 0.001) -> dict` with keys `recall, threshold, fpr` — the highest recall achievable while keeping FPR `<= max_fpr`; if no threshold satisfies it, returns `recall=0.0, threshold=1.0, fpr=0.0`.
  - `confusion_at(y_true, scores, threshold) -> dict` with keys `tp, fp, tn, fn`.

- [ ] **Step 1: Write failing tests `tests/ml/test_metrics.py`**

```python
import numpy as np
from ml.eval.metrics import binary_metrics, recall_at_fpr, confusion_at


def test_binary_metrics_perfect_separation():
    y = [0, 0, 1, 1]
    s = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(y, s, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_recall_at_fpr_respects_budget():
    rng = np.random.default_rng(0)
    y = np.array([0] * 1000 + [1] * 1000)
    s = np.concatenate([rng.uniform(0, 0.6, 1000), rng.uniform(0.4, 1.0, 1000)])
    out = recall_at_fpr(y, s, max_fpr=0.01)
    assert 0.0 <= out["recall"] <= 1.0
    assert out["fpr"] <= 0.01 + 1e-9


def test_confusion_at_counts():
    y = [1, 1, 0, 0]
    s = [0.9, 0.4, 0.6, 0.1]
    c = confusion_at(y, s, 0.5)
    assert c == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ml/eval/metrics.py`**

```python
from __future__ import annotations
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support, roc_auc_score,
    roc_curve,
)


def binary_metrics(y_true: Sequence[int], scores: Sequence[float],
                   threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "threshold": float(threshold),
    }


def recall_at_fpr(y_true, scores, max_fpr: float = 0.001) -> dict:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= max_fpr
    if not ok.any():
        return {"recall": 0.0, "threshold": 1.0, "fpr": 0.0}
    idx = np.argmax(tpr * ok)  # best tpr among allowed points
    return {"recall": float(tpr[idx]), "threshold": float(thr[idx]),
            "fpr": float(fpr[idx])}


def confusion_at(y_true, scores, threshold) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/ml/test_metrics.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add ml/eval/metrics.py tests/ml/test_metrics.py
git commit -m "feat: evaluation metrics incl. recall-at-FPR"
```

---

## Task 10: Evaluation harness + report (`ml/eval/report.py`, `ml/evaluate.py`)

**Files:**
- Create: `ml/eval/report.py`
- Create: `ml/evaluate.py`
- Test: `tests/ml/test_evaluate.py`

**Interfaces:**
- Consumes: `ml.eval.metrics`, `ml.models.base.get_detector`, model artifacts, `datasets/processed/test.parquet`, `datasets/adversarial_testset.csv`.
- Produces:
  - `evaluate_model(detector, test_df, adv_df) -> dict` — returns `{"test": {...binary_metrics, recall_at_fpr...}, "adversarial": {"detection_rate": float, "by_technique": {tech: rate}}, "latency_ms": {"p50": float, "p99": float}, "model_bytes": int}`.
  - `evaluate_all(artifacts_dir, processed_dir, datasets_dir, out_dir) -> dict` — runs every subdir of `artifacts_dir` that has a `run.json`, writes `out_dir/eval.json`, `out_dir/pr_curve.png`, `out_dir/roc_curve.png`, and `reports/MODEL_REPORT.md`.
  - `python -m ml.evaluate` runs `evaluate_all` with default paths.
  - `write_model_report(eval_json: dict, path: Path) -> None`.

- [ ] **Step 1: Write failing test `tests/ml/test_evaluate.py`**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ml/eval/report.py`**

```python
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.eval.metrics import binary_metrics, recall_at_fpr


def _latency_ms(detector, samples: list[str], reps: int = 50) -> dict:
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        detector.predict_proba(samples[:1])
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {"p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99))}


def _model_bytes(detector) -> int:
    tmp = Path(".eval_size_probe")
    tmp.mkdir(exist_ok=True)
    detector.save(tmp)
    total = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
    for p in sorted(tmp.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    tmp.rmdir()
    return total


def evaluate_model(detector, test_df: pd.DataFrame,
                   adv_df: pd.DataFrame) -> dict:
    scores = detector.predict_proba(test_df["text"].tolist())
    y = test_df["label"].tolist()
    test_block = binary_metrics(y, scores, 0.5)
    test_block["recall_at_fpr_0.1pct"] = recall_at_fpr(y, scores, 0.001)

    adv_scores = detector.predict_proba(adv_df["text"].tolist())
    hit = adv_scores >= 0.5
    by_tech: dict[str, float] = {}
    for tech in sorted(adv_df["technique"].unique()):
        mask = (adv_df["technique"] == tech).to_numpy()
        by_tech[tech] = float(hit[mask].mean())

    return {
        "test": test_block,
        "adversarial": {"detection_rate": float(hit.mean()),
                        "by_technique": by_tech},
        "latency_ms": _latency_ms(detector, test_df["text"].tolist()),
        "model_bytes": _model_bytes(detector),
    }


def write_model_report(eval_json: dict, path: Path) -> None:
    lines = ["# Model Report", "",
             "| model | F1 | PR-AUC | ROC-AUC | Recall@FPR0.1% | Adv. detect | p50 ms | size |",
             "|---|---|---|---|---|---|---|---|"]
    for name, e in sorted(eval_json.items()):
        t = e["test"]
        lines.append(
            f"| {name} | {t['f1']:.3f} | {t['pr_auc']:.3f} | {t['roc_auc']:.3f} "
            f"| {t['recall_at_fpr_0.1pct']['recall']:.3f} "
            f"| {e['adversarial']['detection_rate']:.3f} "
            f"| {e['latency_ms']['p50']:.2f} | {e['model_bytes']} |")
    lines += ["", "_Active model rationale: see spec §6. Regenerate with "
              "`python -m ml.evaluate`._", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: Implement `ml/evaluate.py`**

```python
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from ml.eval.report import evaluate_model, write_model_report
from ml.models.base import get_detector
import ml.models.baseline  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent


def _load_detector(name: str, art_dir: Path):
    if name == "cnn":
        import ml.models.cnn  # noqa: F401
    elif name == "distilbert":
        import ml.models.distilbert  # noqa: F401
    return get_detector(name).load(art_dir)


def evaluate_all(artifacts_dir: Path, processed_dir: Path,
                 datasets_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    adv_df = pd.read_csv(datasets_dir / "adversarial_testset.csv")

    results: dict = {}
    for sub in sorted(p for p in artifacts_dir.iterdir() if p.is_dir()):
        if not (sub / "run.json").exists():
            continue
        det = _load_detector(sub.name, sub)
        results[sub.name] = evaluate_model(det, test_df, adv_df)

    (out_dir / "eval.json").write_text(json.dumps(results, indent=2))
    write_model_report(results, _ROOT / "reports" / "MODEL_REPORT.md")
    return results


def main() -> None:
    evaluate_all(_ROOT / "ml" / "artifacts",
                 _ROOT / "datasets" / "processed",
                 _ROOT / "datasets",
                 _ROOT / "reports")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run to verify the test passes**

Run: `.venv/Scripts/python -m pytest tests/ml/test_evaluate.py -v`
Expected: PASS.

- [ ] **Step 6: Run the harness on the real baseline artifact**

Run: `.venv/Scripts/python -m ml.evaluate`
Expected: `reports/MODEL_REPORT.md` and `reports/eval.json` created with a `baseline` row.

- [ ] **Step 7: Commit**

```bash
git add ml/eval/report.py ml/evaluate.py tests/ml/test_evaluate.py reports/MODEL_REPORT.md reports/eval.json
git commit -m "feat: evaluation harness and model report generator"
```

---

## Task 11: inference-svc `/predict` (`inference_svc/`)

**Files:**
- Create: `inference_svc/schemas.py`, `inference_svc/registry.py`, `inference_svc/app.py`, `inference_svc/requirements.txt`
- Test: `tests/inference_svc/test_predict.py`

**Interfaces:**
- Consumes: `ml.models.base.get_detector` and the concrete model modules.
- Produces:
  - `POST /predict` body `{"values": ["...", "..."]}` → `200` `{"results": [PerValue, ...]}` where `PerValue = {"value": str, "scores": {"baseline": float|null, "cnn": float|null, "distilbert": float|null}, "active_model": str, "score": float, "decision": "benign"|"malicious", "threshold": float}`.
  - `GET /health` → `{"status": "ok", "models_loaded": [str], "active_model": str}`.
  - `registry.ModelRegistry(artifacts_dir, active_model, threshold, best_effort_budget_ms)` with `.score_values(values: list[str]) -> list[dict]`.
  - Env: `ARTIFACTS_DIR` (default `ml/artifacts`), `ACTIVE_MODEL` (default `cnn`, falls back to first loaded if unavailable), `BLOCK_THRESHOLD` (default `0.5`), `BEST_EFFORT_BUDGET_MS` (default `120`).

- [ ] **Step 1: Write `inference_svc/requirements.txt`**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
scikit-learn==1.6.1
joblib==1.4.2
numpy==2.2.1
torch==2.5.1
transformers==4.48.0
pandas==2.2.3
```

- [ ] **Step 2: Write failing test `tests/inference_svc/test_predict.py`**

```python
import numpy as np
from fastapi.testclient import TestClient

from ml.models.base import Detector, register


@register
class _Fake(Detector):
    name = "baseline"

    def fit(self, *a):
        ...

    def predict_proba(self, texts):
        return np.array([0.95 if "'" in t or "--" in t else 0.02
                         for t in texts], dtype=float)

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.joblib").write_text("x")

    @classmethod
    def load(cls, path):
        return cls()


def _client(tmp_path):
    art = tmp_path / "artifacts" / "baseline"
    art.mkdir(parents=True)
    (art / "run.json").write_text("{}")
    (art / "model.joblib").write_text("x")
    from inference_svc.app import create_app
    return TestClient(create_app(artifacts_dir=tmp_path / "artifacts",
                                 active_model="baseline", threshold=0.5))


def test_predict_flags_malicious_and_benign(tmp_path):
    c = _client(tmp_path)
    r = c.post("/predict", json={"values": ["' OR 1=1 -- ", "wireless mouse"]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res[0]["decision"] == "malicious"
    assert res[0]["score"] >= 0.5
    assert res[1]["decision"] == "benign"
    assert res[0]["active_model"] == "baseline"


def test_health_lists_models(tmp_path):
    c = _client(tmp_path)
    h = c.get("/health").json()
    assert h["status"] == "ok"
    assert "baseline" in h["models_loaded"]
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/inference_svc/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'inference_svc.app'`.

- [ ] **Step 4: Implement `inference_svc/schemas.py`**

```python
from __future__ import annotations
from pydantic import BaseModel


class PredictRequest(BaseModel):
    values: list[str]


class PerValue(BaseModel):
    value: str
    scores: dict[str, float | None]
    active_model: str
    score: float
    decision: str
    threshold: float


class PredictResponse(BaseModel):
    results: list[PerValue]
```

- [ ] **Step 5: Implement `inference_svc/registry.py`**

```python
from __future__ import annotations
import time
from pathlib import Path

from ml.models.base import get_detector
import ml.models.baseline  # noqa: F401

_ALL = ["baseline", "cnn", "distilbert"]


class ModelRegistry:
    def __init__(self, artifacts_dir: Path, active_model: str = "cnn",
                 threshold: float = 0.5, best_effort_budget_ms: float = 120.0):
        self.threshold = threshold
        self.best_effort_budget_ms = best_effort_budget_ms
        self.models: dict = {}
        for name in _ALL:
            sub = artifacts_dir / name
            if not (sub / "run.json").exists():
                continue
            if name == "cnn":
                import ml.models.cnn  # noqa: F401
            elif name == "distilbert":
                import ml.models.distilbert  # noqa: F401
            self.models[name] = get_detector(name).load(sub)
        if not self.models:
            raise RuntimeError("no model artifacts found")
        self.active_model = active_model if active_model in self.models \
            else next(iter(self.models))

    def score_values(self, values: list[str]) -> list[dict]:
        active = self.models[self.active_model]
        active_scores = active.predict_proba(values)

        others: dict[str, list] = {}
        start = time.perf_counter()
        for name, det in self.models.items():
            if name == self.active_model:
                continue
            if (time.perf_counter() - start) * 1000.0 > self.best_effort_budget_ms:
                break
            try:
                others[name] = det.predict_proba(values).tolist()
            except Exception:
                others[name] = None

        results: list[dict] = []
        for i, v in enumerate(values):
            scores: dict[str, float | None] = {n: None for n in _ALL}
            scores[self.active_model] = float(active_scores[i])
            for name, arr in others.items():
                scores[name] = None if arr is None else float(arr[i])
            score = float(active_scores[i])
            results.append({
                "value": v,
                "scores": scores,
                "active_model": self.active_model,
                "score": score,
                "decision": "malicious" if score >= self.threshold else "benign",
                "threshold": self.threshold,
            })
        return results
```

- [ ] **Step 6: Implement `inference_svc/app.py`**

```python
from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI

from inference_svc.registry import ModelRegistry
from inference_svc.schemas import PredictRequest, PredictResponse

_DEFAULT_ART = Path(__file__).resolve().parent.parent / "ml" / "artifacts"


def create_app(artifacts_dir: Path | None = None,
               active_model: str | None = None,
               threshold: float | None = None) -> FastAPI:
    reg = ModelRegistry(
        artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", _DEFAULT_ART)),
        active_model or os.environ.get("ACTIVE_MODEL", "cnn"),
        threshold if threshold is not None
        else float(os.environ.get("BLOCK_THRESHOLD", "0.5")),
        float(os.environ.get("BEST_EFFORT_BUDGET_MS", "120")),
    )
    app = FastAPI(title="inference-svc")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok",
                "models_loaded": sorted(reg.models),
                "active_model": reg.active_model}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        return PredictResponse(results=reg.score_values(req.values))

    return app


app = create_app() if os.environ.get("INFERENCE_EAGER") else None
```

- [ ] **Step 7: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/inference_svc/test_predict.py -v`
Expected: PASS (2 tests).

- [ ] **Step 8: Commit**

```bash
git add inference_svc tests/inference_svc
git commit -m "feat: inference-svc with /predict and best-effort multi-model scoring"
```

---

## Task 12: waf-proxy value extraction (`waf_proxy/extract.py`, `waf_proxy/config.py`)

**Files:**
- Create: `waf_proxy/config.py`
- Create: `waf_proxy/extract.py`
- Test: `tests/waf_proxy/test_extract.py`

**Interfaces:**
- Consumes: `ml.data.clean` (`normalize_value`, `is_inspectable`).
- Produces:
  - `config.Settings` (pydantic `BaseSettings`) with `upstream_url: str = "http://vuln-app:8000"`, `inference_url: str = "http://inference-svc:9000"`, `inference_timeout_ms: int = 250`, `active_model: str = "cnn"`, `block_threshold: float = 0.5`, `min_value_len: int = 3`, `events_db: str = "/data/events.db"`. Reads env `UPSTREAM_URL`, `INFERENCE_URL`, `INFERENCE_TIMEOUT_MS`, `ACTIVE_MODEL`, `BLOCK_THRESHOLD`, `MIN_VALUE_LEN`, `EVENTS_DB`.
  - `extract.Candidate = namedtuple("Candidate", "location name value")` where `location` in `{"query","form","json","cookie"}`.
  - `extract_candidates(method, query_string, headers, body_bytes, min_len) -> list[Candidate]` — pulls values from query string, form-encoded body (if `content-type` is `application/x-www-form-urlencoded`), JSON body (recursively, scalar leaves only), and `Cookie` header. Applies `normalize_value`; keeps only `is_inspectable(value, min_len)`.

- [ ] **Step 1: Write failing tests `tests/waf_proxy/test_extract.py`**

```python
import json
from waf_proxy.extract import extract_candidates


def test_extracts_query_values():
    c = extract_candidates("GET", "q=mouse&x=ab&p=hello", {}, b"", 3)
    vals = {x.value for x in c}
    assert "mouse" in vals and "hello" in vals
    assert "ab" not in vals  # too short


def test_extracts_form_body():
    body = b"username=admin%27%20--%20&password=x"
    hdrs = {"content-type": "application/x-www-form-urlencoded"}
    c = extract_candidates("POST", "", hdrs, body, 3)
    vals = {x.value for x in c}
    assert "admin' -- " in vals
    assert all(x.location in {"query", "form"} for x in c)


def test_extracts_json_scalars_recursively():
    body = json.dumps({"a": "select desk lamp",
                       "nested": {"b": "' OR 1=1 -- ", "n": 5}}).encode()
    hdrs = {"content-type": "application/json"}
    c = extract_candidates("POST", "", hdrs, body, 3)
    vals = {x.value for x in c}
    assert "select desk lamp" in vals
    assert "' OR 1=1 -- " in vals


def test_extracts_cookie_values():
    hdrs = {"cookie": "sid=abc123def; theme=dark"}
    c = extract_candidates("GET", "", hdrs, b"", 3)
    vals = {x.value for x in c}
    assert "abc123def" in vals
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/waf_proxy/test_extract.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `waf_proxy/config.py`**

```python
from __future__ import annotations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    upstream_url: str = "http://vuln-app:8000"
    inference_url: str = "http://inference-svc:9000"
    inference_timeout_ms: int = 250
    active_model: str = "cnn"
    block_threshold: float = 0.5
    min_value_len: int = 3
    events_db: str = "/data/events.db"

    class Config:
        env_prefix = ""
        fields = {
            "upstream_url": {"env": "UPSTREAM_URL"},
            "inference_url": {"env": "INFERENCE_URL"},
            "inference_timeout_ms": {"env": "INFERENCE_TIMEOUT_MS"},
            "active_model": {"env": "ACTIVE_MODEL"},
            "block_threshold": {"env": "BLOCK_THRESHOLD"},
            "min_value_len": {"env": "MIN_VALUE_LEN"},
            "events_db": {"env": "EVENTS_DB"},
        }


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Implement `waf_proxy/extract.py`**

```python
from __future__ import annotations
import json
from collections import namedtuple
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl

from ml.data.clean import normalize_value, is_inspectable

Candidate = namedtuple("Candidate", "location name value")


def _qs(query_string: str, location: str) -> list[Candidate]:
    out = []
    for k, v in parse_qsl(query_string, keep_blank_values=True):
        out.append(Candidate(location, k, normalize_value(v)))
    return out


def _json_leaves(obj, name="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_leaves(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_leaves(v, name)
    elif isinstance(obj, (str, int, float)) and not isinstance(obj, bool):
        yield Candidate("json", name, normalize_value(str(obj)))


def extract_candidates(method: str, query_string: str, headers: dict,
                       body_bytes: bytes, min_len: int) -> list[Candidate]:
    hdrs = {k.lower(): v for k, v in headers.items()}
    cands: list[Candidate] = list(_qs(query_string, "query"))

    ctype = hdrs.get("content-type", "")
    if body_bytes:
        if "application/x-www-form-urlencoded" in ctype:
            body = body_bytes.decode("utf-8", "ignore")
            for k, v in parse_qsl(body, keep_blank_values=True):
                cands.append(Candidate("form", k, normalize_value(v)))
        elif "application/json" in ctype:
            try:
                cands.extend(_json_leaves(json.loads(body_bytes)))
            except (ValueError, TypeError):
                pass

    if "cookie" in hdrs:
        jar = SimpleCookie()
        jar.load(hdrs["cookie"])
        for k, morsel in jar.items():
            cands.append(Candidate("cookie", k, normalize_value(morsel.value)))

    return [c for c in cands if is_inspectable(c.value, min_len)]
```

- [ ] **Step 5: Add `pydantic-settings` to dev + proxy requirements**

Append to `requirements-dev.txt`: `pydantic-settings==2.7.0`
Run: `.venv/Scripts/python -m pip install pydantic-settings==2.7.0`

- [ ] **Step 6: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/waf_proxy/test_extract.py -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
git add waf_proxy/config.py waf_proxy/extract.py tests/waf_proxy/test_extract.py requirements-dev.txt
git commit -m "feat: waf-proxy request value extraction and settings"
```

---

## Task 13: waf-proxy events writer (`waf_proxy/events.py`)

**Files:**
- Create: `waf_proxy/events.py`
- Test: `tests/waf_proxy/test_events.py` (new file, add to plan's file list under `tests/waf_proxy/`)

**Interfaces:**
- Consumes: nothing (stdlib `sqlite3`).
- Produces:
  - `EventStore(db_path: str)` — on init, `CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, method TEXT, path TEXT, param TEXT, value TEXT, scores TEXT, active_model TEXT, score REAL, threshold REAL, blocked INTEGER, secure_mode INTEGER)`. WAL mode.
  - `record(self, *, method, path, param, value, scores: dict, active_model, score: float, threshold: float, blocked: bool, secure_mode: bool) -> int` — inserts, returns row id. `value` is truncated to 200 chars. `scores` stored as JSON.
  - `recent(self, limit: int = 100) -> list[dict]` — newest first.
  - `after(self, last_id: int, limit: int = 100) -> list[dict]` — rows with `id > last_id`, oldest first.

- [ ] **Step 1: Write failing tests `tests/waf_proxy/test_events.py`**

```python
from waf_proxy.events import EventStore


def test_record_and_recent(tmp_path):
    st = EventStore(str(tmp_path / "e.db"))
    rid = st.record(method="GET", path="/search", param="q",
                    value="' OR 1=1 -- ", scores={"cnn": 0.9},
                    active_model="cnn", score=0.9, threshold=0.5, blocked=True,
                    secure_mode=False)
    assert rid == 1
    rows = st.recent()
    assert rows[0]["blocked"] == 1
    assert rows[0]["param"] == "q"
    assert rows[0]["scores"] == {"cnn": 0.9}
    assert rows[0]["threshold"] == 0.5


def test_value_is_truncated(tmp_path):
    st = EventStore(str(tmp_path / "e.db"))
    st.record(method="GET", path="/x", param="q", value="a" * 500,
              scores={}, active_model="cnn", score=0.0, threshold=0.5,
              blocked=False, secure_mode=True)
    assert len(st.recent()[0]["value"]) == 200


def test_after_returns_new_rows_oldest_first(tmp_path):
    st = EventStore(str(tmp_path / "e.db"))
    for i in range(5):
        st.record(method="GET", path=f"/{i}", param="q", value="xyz",
                  scores={}, active_model="cnn", score=0.0, threshold=0.5,
                  blocked=False, secure_mode=False)
    rows = st.after(2)
    assert [r["id"] for r in rows] == [3, 4, 5]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/waf_proxy/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `waf_proxy/events.py`**

```python
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, method TEXT, path TEXT, param TEXT, value TEXT,
  scores TEXT, active_model TEXT, score REAL, threshold REAL,
  blocked INTEGER, secure_mode INTEGER
);
"""


class EventStore:
    def __init__(self, db_path: str):
        self._path = db_path
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL;")
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, *, method, path, param, value, scores, active_model,
               score, threshold, blocked, secure_mode) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO events(ts,method,path,param,value,scores,"
                "active_model,score,threshold,blocked,secure_mode) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), method, path, param,
                 str(value)[:200], json.dumps(scores), active_model,
                 float(score), float(threshold), int(blocked), int(secure_mode)))
            return int(cur.lastrowid)

    def _rows(self, sql: str, args: tuple) -> list[dict]:
        with self._conn() as c:
            out = []
            for r in c.execute(sql, args):
                d = dict(r)
                d["scores"] = json.loads(d["scores"]) if d["scores"] else {}
                out.append(d)
            return out

    def recent(self, limit: int = 100) -> list[dict]:
        return self._rows(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    def after(self, last_id: int, limit: int = 100) -> list[dict]:
        return self._rows(
            "SELECT * FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
            (last_id, limit))
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/waf_proxy/test_events.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add waf_proxy/events.py tests/waf_proxy/test_events.py
git commit -m "feat: waf-proxy SQLite event store"
```

---

## Task 14: waf-proxy decision + app (`waf_proxy/decision.py`, `waf_proxy/app.py`)

**Files:**
- Create: `waf_proxy/decision.py`
- Create: `waf_proxy/app.py`
- Create: `waf_proxy/requirements.txt`
- Test: `tests/waf_proxy/test_decision.py`

**Interfaces:**
- Consumes: `waf_proxy.extract`, `waf_proxy.events.EventStore`, `waf_proxy.config.Settings`.
- Produces:
  - `async score_values(client, inference_url, values, timeout_ms) -> list[dict] | None` — POSTs `{"values": values}` to `inference_url + "/predict"`; returns `results` list, or `None` on timeout/error (fail-open signal).
  - `decide(results: list[dict], threshold: float) -> tuple[bool, dict | None]` — `(blocked, worst_result)`; blocked if any `result["score"] >= threshold`; `worst_result` is the max-score result (or `None` if list empty).
  - `create_app(settings, event_store, http_client=None) -> FastAPI` with a catch-all route for every method/path that: extracts candidates, scores them, on fail-open forwards + records `blocked=0` with a `scores={}`, on block returns `403` JSON `{"blocked_by":"ai-waf","model":...,"score":...,"matched_param":...}` + records, else forwards upstream and records `blocked=0`.
  - Forwarding preserves method, path, query, headers (minus `host`), and body; returns upstream status, headers, body.

- [ ] **Step 1: Write `waf_proxy/requirements.txt`**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
httpx==0.28.1
pydantic==2.10.4
pydantic-settings==2.7.0
```

- [ ] **Step 2: Write failing tests `tests/waf_proxy/test_decision.py`**

```python
import respx
import httpx
from fastapi.testclient import TestClient

from waf_proxy.config import Settings
from waf_proxy.events import EventStore
from waf_proxy.decision import decide, create_app


def test_decide_blocks_on_any_high_score():
    results = [{"score": 0.1, "value": "a"}, {"score": 0.8, "value": "b"}]
    blocked, worst = decide(results, 0.5)
    assert blocked is True
    assert worst["value"] == "b"


def test_decide_allows_when_all_low():
    blocked, worst = decide([{"score": 0.2, "value": "a"}], 0.5)
    assert blocked is False


@respx.mock
def test_malicious_request_is_blocked_and_recorded(tmp_path):
    respx.post("http://inf:9000/predict").mock(return_value=httpx.Response(
        200, json={"results": [{"value": "' OR 1=1 -- ",
                                "scores": {"cnn": 0.97}, "active_model": "cnn",
                                "score": 0.97, "decision": "malicious",
                                "threshold": 0.5}]}))
    st = EventStore(str(tmp_path / "e.db"))
    s = Settings(inference_url="http://inf:9000", upstream_url="http://up:8000",
                 events_db=str(tmp_path / "e.db"))
    c = TestClient(create_app(s, st))
    r = c.get("/search?q=' OR 1=1 -- ")
    assert r.status_code == 403
    assert r.json()["blocked_by"] == "ai-waf"
    assert st.recent()[0]["blocked"] == 1


@respx.mock
def test_benign_request_is_forwarded(tmp_path):
    respx.post("http://inf:9000/predict").mock(return_value=httpx.Response(
        200, json={"results": [{"value": "mouse", "scores": {"cnn": 0.01},
                                "active_model": "cnn", "score": 0.01,
                                "decision": "benign", "threshold": 0.5}]}))
    respx.get("http://up:8000/search").mock(return_value=httpx.Response(
        200, text="results page"))
    st = EventStore(str(tmp_path / "e.db"))
    s = Settings(inference_url="http://inf:9000", upstream_url="http://up:8000",
                 events_db=str(tmp_path / "e.db"))
    c = TestClient(create_app(s, st))
    r = c.get("/search?q=mouse")
    assert r.status_code == 200
    assert r.text == "results page"
    assert st.recent()[0]["blocked"] == 0


@respx.mock
def test_inference_down_fails_open(tmp_path):
    respx.post("http://inf:9000/predict").mock(side_effect=httpx.ConnectError("x"))
    respx.get("http://up:8000/search").mock(return_value=httpx.Response(
        200, text="ok"))
    st = EventStore(str(tmp_path / "e.db"))
    s = Settings(inference_url="http://inf:9000", upstream_url="http://up:8000",
                 events_db=str(tmp_path / "e.db"))
    c = TestClient(create_app(s, st))
    r = c.get("/search?q=' OR 1=1 -- ")
    assert r.status_code == 200
    assert st.recent()[0]["blocked"] == 0
```

- [ ] **Step 3: Add test deps**

Append to `requirements-dev.txt`: `respx==0.22.0` and `fastapi==0.115.6` and `uvicorn[standard]==0.34.0`.
Run: `.venv/Scripts/python -m pip install respx==0.22.0 fastapi==0.115.6 "uvicorn[standard]==0.34.0"`

- [ ] **Step 4: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/waf_proxy/test_decision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'waf_proxy.decision'`.

- [ ] **Step 5: Implement `waf_proxy/decision.py`**

```python
from __future__ import annotations

import httpx
from fastapi import FastAPI, Request, Response

from waf_proxy.config import Settings
from waf_proxy.events import EventStore
from waf_proxy.extract import extract_candidates

_HOP = {"host", "content-length", "connection"}


async def score_values(client: httpx.AsyncClient, inference_url: str,
                       values: list[str], timeout_ms: int):
    try:
        resp = await client.post(f"{inference_url}/predict",
                                 json={"values": values},
                                 timeout=timeout_ms / 1000.0)
        resp.raise_for_status()
        return resp.json()["results"]
    except (httpx.HTTPError, KeyError, ValueError):
        return None


def decide(results: list[dict], threshold: float):
    if not results:
        return False, None
    worst = max(results, key=lambda r: r["score"])
    return worst["score"] >= threshold, worst


def create_app(settings: Settings, event_store: EventStore,
               http_client: httpx.AsyncClient | None = None) -> FastAPI:
    app = FastAPI(title="waf-proxy")
    client = http_client or httpx.AsyncClient()
    secure_mode_flag = False  # updated via header echo from upstream if present

    @app.on_event("shutdown")
    async def _close():
        await client.aclose()

    @app.api_route("/{full_path:path}",
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def proxy(request: Request, full_path: str) -> Response:
        body = await request.body()
        cands = extract_candidates(request.method, request.url.query,
                                   dict(request.headers), body,
                                   settings.min_value_len)
        values = [c.value for c in cands]

        results = None
        if values:
            results = await score_values(client, settings.inference_url,
                                         values, settings.inference_timeout_ms)

        if results is None:
            # fail-open (either nothing to score, or inference unavailable)
            if values:
                event_store.record(
                    method=request.method, path=request.url.path, param="-",
                    value="", scores={}, active_model=settings.active_model,
                    score=0.0, threshold=settings.block_threshold,
                    blocked=False, secure_mode=secure_mode_flag)
            return await _forward(client, settings, request, body)

        blocked, worst = decide(results, settings.block_threshold)
        idx = results.index(worst)
        matched = cands[idx]
        event_store.record(
            method=request.method, path=request.url.path, param=matched.name,
            value=matched.value, scores=worst["scores"],
            active_model=worst["active_model"], score=worst["score"],
            threshold=settings.block_threshold,
            blocked=blocked, secure_mode=secure_mode_flag)

        if blocked:
            return Response(
                content=(f'{{"blocked_by":"ai-waf","model":"{worst["active_model"]}",'
                         f'"score":{worst["score"]:.4f},'
                         f'"matched_param":"{matched.name}"}}'),
                status_code=403, media_type="application/json")

        return await _forward(client, settings, request, body)

    return app


async def _forward(client, settings, request: Request, body: bytes) -> Response:
    url = settings.upstream_url + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in _HOP}
    up = await client.request(request.method, url, headers=fwd_headers,
                              content=body, timeout=10.0)
    resp_headers = {k: v for k, v in up.headers.items()
                    if k.lower() not in _HOP}
    return Response(content=up.content, status_code=up.status_code,
                    headers=resp_headers)
```

- [ ] **Step 6: Implement `waf_proxy/app.py`**

```python
from __future__ import annotations

from waf_proxy.config import get_settings
from waf_proxy.decision import create_app
from waf_proxy.events import EventStore

settings = get_settings()
app = create_app(settings, EventStore(settings.events_db))
```

- [ ] **Step 7: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/waf_proxy/test_decision.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add waf_proxy/decision.py waf_proxy/app.py waf_proxy/requirements.txt tests/waf_proxy/test_decision.py requirements-dev.txt
git commit -m "feat: waf-proxy decision, forwarding, and fail-open"
```

---

## Task 15: vuln-app — vulnerable mode (`vuln_app/`)

**Files:**
- Create: `vuln_app/queries.py`, `vuln_app/db.py`, `vuln_app/seed.py`, `vuln_app/app.py`, `vuln_app/templates/index.html`, `vuln_app/templates/login.html`, `vuln_app/requirements.txt`
- Test: `tests/vuln_app/test_vulnerable.py`

**Interfaces:**
- Consumes: nothing (Flask + stdlib sqlite3).
- Produces:
  - `db.connect(db_path: str, readonly: bool = False) -> sqlite3.Connection`.
  - `seed.init_db(db_path: str) -> None` — drops + creates `products(id,name,category,price)` (25 rows) and `users(id,username,password)` (rows incl. `("admin","s3cr3t-admin")`).
  - `queries.search_sql(q: str, secure: bool) -> tuple[str, tuple]` — secure=False returns string-concatenated SQL + `()`; secure=True returns parameterized SQL + params.
  - `queries.login_sql(u: str, p: str, secure: bool) -> tuple[str, tuple]`.
  - `queries.product_sql(pid: str, secure: bool) -> tuple[str, tuple]`.
  - `app.create_app(db_path: str) -> Flask` — routes `GET /search?q=`, `POST /login`, `GET /product?id=`, `GET /health`. Reads `SECURE_MODE` env (`"1"` → secure). On DB error at `SECURE_MODE=0`, returns the exception text with `500`; at `SECURE_MODE=1`, returns generic `"internal error"`.
- Every file begins with `# INTENTIONALLY VULNERABLE - DO NOT DEPLOY`.

- [ ] **Step 1: Write `vuln_app/requirements.txt`**

```
Flask==3.1.0
gunicorn==23.0.0
```

- [ ] **Step 2: Write failing tests `tests/vuln_app/test_vulnerable.py`**

```python
import pytest
from vuln_app.seed import init_db
from vuln_app.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "app.db")
    init_db(db)
    monkeypatch.setenv("SECURE_MODE", "0")
    app = create_app(db)
    app.config.update(TESTING=True)
    return app.test_client()


def test_normal_search_works(client):
    r = client.get("/search?q=mouse")
    assert r.status_code == 200
    assert b"mouse" in r.data.lower()


def test_union_select_exfiltrates_users(client):
    payload = "' UNION SELECT id, username, password, price FROM users -- "
    r = client.get("/search", query_string={"q": payload})
    assert r.status_code == 200
    assert b"admin" in r.data
    assert b"s3cr3t-admin" in r.data


def test_login_auth_bypass(client):
    r = client.post("/login", data={"username": "admin' -- ", "password": "x"})
    assert r.status_code == 200
    assert b"welcome" in r.data.lower()


def test_product_boolean_injection(client):
    r = client.get("/product", query_string={"id": "5 OR 1=1"})
    assert r.status_code == 200
    # OR 1=1 returns more than one product row
    assert r.data.count(b"<tr") > 2
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/vuln_app/test_vulnerable.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `vuln_app/queries.py`**

```python
# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations


def search_sql(q: str, secure: bool):
    if secure:
        return ("SELECT id, name, category, price FROM products "
                "WHERE name LIKE ?", (f"%{q}%",))
    return (f"SELECT id, name, category, price FROM products "
            f"WHERE name LIKE '%{q}%'", ())


def login_sql(u: str, p: str, secure: bool):
    if secure:
        return ("SELECT id, username FROM users "
                "WHERE username = ? AND password = ?", (u, p))
    return (f"SELECT id, username FROM users "
            f"WHERE username = '{u}' AND password = '{p}'", ())


def product_sql(pid: str, secure: bool):
    if secure:
        return ("SELECT id, name, category, price FROM products "
                "WHERE id = ?", (int(pid),))
    return (f"SELECT id, name, category, price FROM products "
            f"WHERE id = {pid}", ())
```

- [ ] **Step 5: Implement `vuln_app/db.py`**

```python
# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations
import sqlite3


def connect(db_path: str, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn
```

- [ ] **Step 6: Implement `vuln_app/seed.py`**

```python
# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations
import sqlite3

_PRODUCTS = [
    ("Wireless Mouse", "peripherals", 19.99),
    ("Mechanical Keyboard", "peripherals", 89.00),
    ("Laptop Stand", "accessories", 34.50),
    ("USB-C Hub", "accessories", 45.00),
    ("Desk Lamp", "lighting", 22.00),
    # ... extend to 25 rows total with varied names/categories/prices
]
_USERS = [
    ("admin", "s3cr3t-admin"),
    ("alice", "password123"),
    ("bob", "hunter2"),
]


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        "DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS users;"
        "CREATE TABLE products(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " name TEXT, category TEXT, price REAL);"
        "CREATE TABLE users(id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " username TEXT, password TEXT);")
    conn.executemany("INSERT INTO products(name,category,price) VALUES(?,?,?)",
                     _PRODUCTS)
    conn.executemany("INSERT INTO users(username,password) VALUES(?,?)", _USERS)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    import sys
    init_db(sys.argv[1] if len(sys.argv) > 1 else "app.db")
```

(Extend `_PRODUCTS` to 25 rows when implementing.)

- [ ] **Step 7: Implement `vuln_app/app.py`**

```python
# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations
import os

from flask import Flask, request, render_template

from vuln_app import db, queries


def _secure() -> bool:
    return os.environ.get("SECURE_MODE", "0") == "1"


def create_app(db_path: str = "app.db") -> Flask:
    app = Flask(__name__)

    def _run(sql: str, params: tuple):
        conn = db.connect(db_path, readonly=_secure())
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @app.get("/health")
    def health():
        return {"status": "ok", "secure_mode": _secure()}

    @app.get("/search")
    def search():
        q = request.args.get("q", "")
        sql, params = queries.search_sql(q, _secure())
        try:
            rows = _run(sql, params)
        except Exception as exc:  # noqa: BLE001
            if _secure():
                return "internal error", 500
            return f"SQL error: {exc}\nquery: {sql}", 500
        return render_template("index.html", q=q, rows=rows)

    @app.post("/login")
    def login():
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        sql, params = queries.login_sql(u, p, _secure())
        try:
            rows = _run(sql, params)
        except Exception as exc:  # noqa: BLE001
            if _secure():
                return "internal error", 500
            return f"SQL error: {exc}\nquery: {sql}", 500
        if rows:
            return render_template("login.html", user=rows[0]["username"])
        return "invalid credentials", 401

    @app.get("/product")
    def product():
        pid = request.args.get("id", "0")
        sql, params = queries.product_sql(pid, _secure())
        try:
            rows = _run(sql, params)
        except Exception as exc:  # noqa: BLE001
            if _secure():
                return "internal error", 500
            return f"SQL error: {exc}\nquery: {sql}", 500
        return render_template("index.html", q=f"id={pid}", rows=rows)

    return app


app = create_app(os.environ.get("APP_DB", "app.db"))
```

- [ ] **Step 8: Implement templates**

`vuln_app/templates/index.html`:

```html
<!-- INTENTIONALLY VULNERABLE - DO NOT DEPLOY -->
<!doctype html><title>shop</title>
<form action="/search"><input name="q" value="{{ q }}"><button>search</button></form>
<table><tr><th>id</th><th>name</th><th>category</th><th>price</th></tr>
{% for r in rows %}<tr><td>{{ r.get('id') }}</td><td>{{ r.get('name') }}</td>
<td>{{ r.get('category') }}</td><td>{{ r.get('price') }}</td></tr>{% endfor %}
</table>
```

`vuln_app/templates/login.html`:

```html
<!-- INTENTIONALLY VULNERABLE - DO NOT DEPLOY -->
<!doctype html><title>login</title>
{% if user %}<p>welcome, {{ user }}</p>{% else %}
<form method="post" action="/login">
<input name="username"><input name="password" type="password"><button>login</button>
</form>{% endif %}
```

- [ ] **Step 9: Run to verify it passes**

Run: `.venv/Scripts/python -m pip install Flask==3.1.0 && .venv/Scripts/python -m pytest tests/vuln_app/test_vulnerable.py -v`
Expected: PASS (4 tests).

- [ ] **Step 10: Commit**

```bash
git add vuln_app tests/vuln_app/test_vulnerable.py
git commit -m "feat: intentionally vulnerable demo app (SECURE_MODE=0)"
```

---

## Task 16: vuln-app — secure mode verification (`tests/vuln_app/test_secure.py`)

**Files:**
- Modify: none (logic already in `queries.py` / `app.py` from Task 15)
- Test: `tests/vuln_app/test_secure.py`

**Interfaces:**
- Consumes: `vuln_app.app.create_app`, `vuln_app.seed.init_db`.
- Produces: proof that `SECURE_MODE=1` defeats the Task-15 attacks and that reads still work.

- [ ] **Step 1: Write failing/pending tests `tests/vuln_app/test_secure.py`**

```python
import pytest
from vuln_app.seed import init_db
from vuln_app.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "app.db")
    init_db(db)
    monkeypatch.setenv("SECURE_MODE", "1")
    app = create_app(db)
    app.config.update(TESTING=True)
    return app.test_client()


def test_secure_normal_search_still_works(client):
    r = client.get("/search?q=mouse")
    assert r.status_code == 200
    assert b"mouse" in r.data.lower()


def test_secure_union_select_returns_no_user_data(client):
    payload = "' UNION SELECT id, username, password, price FROM users -- "
    r = client.get("/search", query_string={"q": payload})
    assert r.status_code == 200
    assert b"s3cr3t-admin" not in r.data
    assert b"admin" not in r.data


def test_secure_login_bypass_fails(client):
    r = client.post("/login", data={"username": "admin' -- ", "password": "x"})
    assert r.status_code == 401


def test_secure_product_injection_is_rejected(client):
    r = client.get("/product", query_string={"id": "5 OR 1=1"})
    # int() coercion raises -> generic 500, never a boolean-true result set
    assert r.status_code == 500
    assert b"internal error" in r.data
```

- [ ] **Step 2: Run the tests**

Run: `.venv/Scripts/python -m pytest tests/vuln_app/test_secure.py -v`
Expected: PASS (4 tests). If any fail, fix `queries.py` / `app.py` secure branches until green (do not weaken the vulnerable branch — re-run Task 15 tests too).

- [ ] **Step 3: Run both vuln-app suites together**

Run: `.venv/Scripts/python -m pytest tests/vuln_app -v`
Expected: PASS (8 tests).

- [ ] **Step 4: Commit**

```bash
git add tests/vuln_app/test_secure.py vuln_app
git commit -m "test: verify SECURE_MODE=1 defeats the injection attacks"
```

---

## Task 17: vuln-app — benign traffic generator (`vuln_app/traffic_gen.py`)

**Files:**
- Create: `vuln_app/traffic_gen.py`
- Test: `tests/vuln_app/test_traffic_gen.py` (add to file list)

**Interfaces:**
- Consumes: nothing (httpx).
- Produces:
  - `SQL_LOOKING_BENIGN: list[str]` — includes `"O'Brien"`, `"chair 1=1 sale"`, `"SELECT desk lamp"`.
  - `build_requests(n: int, seed: int = 0) -> list[tuple[str, str, dict]]` — returns `(method, path, params_or_data)` tuples mixing `/search`, `/product`, `/login` with legitimate values; roughly `1/4` drawn from `SQL_LOOKING_BENIGN`.
  - `run(base_url: str, n: int, seed: int = 0) -> list[int]` — sends them, returns status codes.
  - `python -m vuln_app.traffic_gen --url http://localhost:8080 -n 50`.

- [ ] **Step 1: Write failing test `tests/vuln_app/test_traffic_gen.py`**

```python
from vuln_app.traffic_gen import build_requests, SQL_LOOKING_BENIGN


def test_build_requests_count_and_shape():
    reqs = build_requests(40, seed=1)
    assert len(reqs) == 40
    for method, path, payload in reqs:
        assert method in {"GET", "POST"}
        assert path in {"/search", "/product", "/login"}
        assert isinstance(payload, dict)


def test_build_requests_includes_sql_looking_values():
    reqs = build_requests(200, seed=2)
    flat = " ".join(str(p) for _, _, p in reqs)
    assert any(v in flat for v in SQL_LOOKING_BENIGN)


def test_build_requests_deterministic():
    assert build_requests(30, seed=5) == build_requests(30, seed=5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/vuln_app/test_traffic_gen.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `vuln_app/traffic_gen.py`**

```python
# INTENTIONALLY VULNERABLE - DO NOT DEPLOY
from __future__ import annotations
import argparse
import random

import httpx

SQL_LOOKING_BENIGN = [
    "O'Brien", "chair 1=1 sale", "SELECT desk lamp",
    "laptop stand under $50", "1 or 2 day shipping",
]
_PLAIN = ["mouse", "keyboard", "usb hub", "desk lamp", "monitor arm",
          "cable", "webcam", "headset"]
_USERS = [("alice", "password123"), ("bob", "hunter2")]


def build_requests(n: int, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = rng.choice(["search", "search", "product", "login"])
        if kind == "search":
            pool = SQL_LOOKING_BENIGN if rng.random() < 0.25 else _PLAIN
            out.append(("GET", "/search", {"q": rng.choice(pool)}))
        elif kind == "product":
            out.append(("GET", "/product", {"id": str(rng.randint(1, 25))}))
        else:
            u, p = rng.choice(_USERS)
            out.append(("POST", "/login", {"username": u, "password": p}))
    return out


def run(base_url: str, n: int, seed: int = 0) -> list[int]:
    codes = []
    with httpx.Client(base_url=base_url, timeout=10) as c:
        for method, path, payload in build_requests(n, seed):
            if method == "GET":
                r = c.get(path, params=payload)
            else:
                r = c.post(path, data=payload)
            codes.append(r.status_code)
    return codes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(run(args.url, args.n, args.seed))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/vuln_app/test_traffic_gen.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vuln_app/traffic_gen.py tests/vuln_app/test_traffic_gen.py
git commit -m "feat: benign traffic generator with SQL-looking legitimate inputs"
```

---

## Task 18: log-ui (`log_ui/`)

**Files:**
- Create: `log_ui/app.py`, `log_ui/static/index.html`, `log_ui/requirements.txt`
- Test: `tests/log_ui/test_events.py`

**Interfaces:**
- Consumes: `waf_proxy.events.EventStore` (read-only use).
- Produces:
  - `create_app(events_db: str) -> FastAPI` with:
    - `GET /events?limit=100` → `{"events": [...]}` newest first (uses `EventStore.recent`).
    - `GET /events/stream` → `text/event-stream`; emits `data: <json>\n\n` per new event using `EventStore.after`, polling every 1s; sends an initial `retry: 2000` line.
    - `GET /` → serves `static/index.html`.
  - `GET /summary` → `{"total": int, "blocked": int, "block_rate": float, "active_model": str|null, "threshold": float|null}` (active_model/threshold read from the newest event).

- [ ] **Step 1: Write `log_ui/requirements.txt`**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
```

- [ ] **Step 2: Write failing tests `tests/log_ui/test_events.py`**

```python
from fastapi.testclient import TestClient
from waf_proxy.events import EventStore
from log_ui.app import create_app


def _seed(db):
    st = EventStore(db)
    st.record(method="GET", path="/search", param="q", value="mouse",
              scores={"cnn": 0.01}, active_model="cnn", score=0.01,
              threshold=0.5, blocked=False, secure_mode=False)
    st.record(method="GET", path="/search", param="q", value="' OR 1=1 -- ",
              scores={"cnn": 0.98}, active_model="cnn", score=0.98,
              threshold=0.5, blocked=True, secure_mode=False)
    return st


def test_events_endpoint_returns_newest_first(tmp_path):
    db = str(tmp_path / "e.db")
    _seed(db)
    c = TestClient(create_app(db))
    ev = c.get("/events").json()["events"]
    assert ev[0]["value"] == "' OR 1=1 -- "
    assert ev[0]["blocked"] == 1


def test_summary_counts(tmp_path):
    db = str(tmp_path / "e.db")
    _seed(db)
    c = TestClient(create_app(db))
    s = c.get("/summary").json()
    assert s["total"] == 2
    assert s["blocked"] == 1
    assert abs(s["block_rate"] - 0.5) < 1e-9
    assert s["active_model"] == "cnn"


def test_index_served(tmp_path):
    db = str(tmp_path / "e.db")
    _seed(db)
    c = TestClient(create_app(db))
    r = c.get("/")
    assert r.status_code == 200
    assert b"<table" in r.content or b"<div" in r.content
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/log_ui/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement `log_ui/app.py`**

```python
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from waf_proxy.events import EventStore

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(events_db: str | None = None) -> FastAPI:
    db = events_db or os.environ.get("EVENTS_DB", "/data/events.db")
    store = EventStore(db)
    app = FastAPI(title="log-ui")

    @app.get("/events")
    def events(limit: int = 100) -> dict:
        return {"events": store.recent(limit)}

    @app.get("/summary")
    def summary() -> dict:
        rows = store.recent(100000)
        total = len(rows)
        blocked = sum(r["blocked"] for r in rows)
        newest = rows[0] if rows else None
        return {"total": total, "blocked": blocked,
                "block_rate": (blocked / total) if total else 0.0,
                "active_model": newest["active_model"] if newest else None,
                "threshold": newest["threshold"] if newest else None}

    @app.get("/events/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            yield "retry: 2000\n\n"
            last = store.recent(1)
            last_id = last[0]["id"] if last else 0
            while True:
                for row in store.after(last_id):
                    last_id = row["id"]
                    yield f"data: {json.dumps(row)}\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    return app


app = create_app() if os.environ.get("LOGUI_EAGER") else None
```

Note: `/summary` returns `newest["threshold"]`, which is persisted by `EventStore` (Task 13 schema includes the `threshold REAL` column and `record(..., threshold=...)` kwarg; Task 14 passes `threshold=settings.block_threshold`). No change needed here beyond what Tasks 13–14 already establish.

- [ ] **Step 5: Implement `log_ui/static/index.html`**

```html
<!doctype html><meta charset="utf-8"><title>AI-WAF log</title>
<style>
body{font:14px system-ui;margin:1rem}
#sum{margin-bottom:1rem}
table{border-collapse:collapse;width:100%}
td,th{border:1px solid #ccc;padding:4px 8px;font-size:13px}
.b{background:#fdd}.a{background:#dfd}
.bar{display:inline-block;height:8px;background:#c33}
</style>
<div id="sum"></div>
<label><input type="checkbox" id="only"> blocked only</label>
<table id="t"><thead><tr><th>time</th><th>method</th><th>path</th><th>param</th>
<th>value</th><th>cnn</th><th>baseline</th><th>distilbert</th><th>active</th>
<th>decision</th><th>secure</th></tr></thead><tbody></tbody></table>
<script>
const tb=document.querySelector('#t tbody'),only=document.querySelector('#only');
function bar(x){return x==null?'-':`<span class="bar" style="width:${Math.round(x*60)}px"></span>${x.toFixed(2)}`}
function row(e){
  if(only.checked && !e.blocked) return;
  const tr=document.createElement('tr');
  tr.className=e.blocked?'b':'a';
  const s=e.scores||{};
  tr.innerHTML=`<td>${e.ts}</td><td>${e.method}</td><td>${e.path}</td>
  <td>${e.param}</td><td>${(e.value||'').slice(0,60)}</td>
  <td>${bar(s.cnn)}</td><td>${bar(s.baseline)}</td><td>${bar(s.distilbert)}</td>
  <td>${e.active_model}</td><td>${e.blocked?'BLOCKED':'ALLOWED'}</td>
  <td>${e.secure_mode?'on':'off'}</td>`;
  tb.prepend(tr);
}
async function boot(){
  const j=await (await fetch('/events?limit=100')).json();
  j.events.reverse().forEach(row);
  const s=await (await fetch('/summary')).json();
  document.querySelector('#sum').textContent=
    `total ${s.total} · blocked ${s.blocked} · rate ${(s.block_rate*100).toFixed(1)}% · active ${s.active_model||'?'}`;
  const es=new EventSource('/events/stream');
  es.onmessage=ev=>row(JSON.parse(ev.data));
}
boot();
</script>
```

- [ ] **Step 6: Run to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/log_ui/test_events.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add log_ui tests/log_ui
git commit -m "feat: log-ui SSE stream and single-page detection log"
```

---

## Task 19: char-CNN model (`ml/models/cnn.py`)

**Files:**
- Create: `ml/models/cnn.py`
- Test: `tests/ml/test_cnn.py`

**Interfaces:**
- Consumes: `ml.models.base` (`Detector`, `register`).
- Produces:
  - `class CNNDetector(Detector)`, `name = "cnn"`.
  - Char vocab built from bytes `0..255` (index 0 = pad); `max_len = 256`; embedding dim 64; three `Conv1d(64, 128, k)` for `k in (3,5,7)`; global max pool; concat → `Linear(384, 1)` → sigmoid.
  - `fit`: Adam lr 1e-3, `BCEWithLogitsLoss` with `pos_weight` from class balance, batch 128, up to 8 epochs, early stop on val loss (patience 2), CPU.
  - `predict_proba`: sigmoid of logits, shape `(n,)`.
  - `save(path)`: `torch.save` state dict to `path / "cnn.pt"` + `path / "cnn_meta.json"` (`max_len`, dims). `load(path)`: rebuild + load.
  - Deterministic: `torch.manual_seed(13)` in `__init__`.

- [ ] **Step 1: Write failing test `tests/ml/test_cnn.py`**

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from ml.models.cnn import CNNDetector

MAL = ["' OR 1=1 -- ", "admin' -- ", "' UNION SELECT a,b,c -- ",
       "1; DROP TABLE users -- ", "' OR SLEEP(5) -- ", "\") OR (\"1\"=\"1"]
BEN = ["wireless mouse", "mechanical keyboard", "laptop stand under $50",
       "O'Brien", "standing desk", "usb-c hub 45 dollars"]


@pytest.mark.slow
def test_cnn_overfits_tiny_set_and_roundtrips(tmp_path):
    d = CNNDetector()
    x = MAL + BEN
    y = [1] * len(MAL) + [0] * len(BEN)
    d.fit(x, y, x, y)
    p = d.predict_proba(["' OR 1=1 -- ", "wireless mouse"])
    assert p.shape == (2,)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[0] > 0.5 > p[1]

    d.save(tmp_path)
    d2 = CNNDetector.load(tmp_path)
    assert np.allclose(d.predict_proba(x), d2.predict_proba(x), atol=1e-5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_cnn.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.models.cnn'`.

- [ ] **Step 3: Implement `ml/models/cnn.py`**

```python
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.models.base import Detector, register

MAX_LEN = 256
EMB = 64
CH = 128
KERNELS = (3, 5, 7)


def _encode(texts: list[str]) -> torch.Tensor:
    arr = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        b = t.encode("utf-8", "ignore")[:MAX_LEN]
        arr[i, :len(b)] = [c + 1 for c in b]  # 0 = pad
    return torch.from_numpy(arr)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(257, EMB, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(EMB, CH, k, padding=k // 2) for k in KERNELS])
        self.fc = nn.Linear(CH * len(KERNELS), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x).transpose(1, 2)
        feats = [torch.relu(c(e)).max(dim=2).values for c in self.convs]
        return self.fc(torch.cat(feats, dim=1)).squeeze(1)


@register
class CNNDetector(Detector):
    name = "cnn"

    def __init__(self) -> None:
        torch.manual_seed(13)
        self.net = _Net()

    def fit(self, tt, tl, vt, vl) -> None:
        self.net.train()
        pos = max(sum(tl), 1)
        neg = max(len(tl) - sum(tl), 1)
        pos_weight = torch.tensor([neg / pos], dtype=torch.float)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

        xt, yt = _encode(tt), torch.tensor(tl, dtype=torch.float)
        xv, yv = _encode(vt), torch.tensor(vl, dtype=torch.float)
        best, bad = float("inf"), 0
        for _epoch in range(8):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 128):
                idx = perm[i:i + 128]
                opt.zero_grad()
                out = self.net(xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vloss = loss_fn(self.net(xv), yv).item()
            self.net.train()
            if vloss < best - 1e-4:
                best, bad = vloss, 0
            else:
                bad += 1
                if bad >= 2:
                    break

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            logits = self.net(_encode(texts))
            return torch.sigmoid(logits).cpu().numpy().astype(float)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "cnn.pt")
        (path / "cnn_meta.json").write_text(json.dumps(
            {"max_len": MAX_LEN, "emb": EMB, "ch": CH, "kernels": list(KERNELS)}))

    @classmethod
    def load(cls, path: Path) -> "CNNDetector":
        obj = cls()
        obj.net.load_state_dict(torch.load(path / "cnn.pt", map_location="cpu"))
        obj.net.eval()
        return obj
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pip install torch==2.5.1 && .venv/Scripts/python -m pytest tests/ml/test_cnn.py -v -m slow`
Expected: PASS.

- [ ] **Step 5: Train cnn on real data and re-run evaluation**

Run: `.venv/Scripts/python -m ml.train --model cnn && .venv/Scripts/python -m ml.evaluate`
Expected: `ml/artifacts/cnn/` created; `reports/MODEL_REPORT.md` now has `baseline` + `cnn` rows.

- [ ] **Step 6: Commit**

```bash
git add ml/models/cnn.py tests/ml/test_cnn.py ml/artifacts/cnn reports/MODEL_REPORT.md reports/eval.json
git commit -m "feat: char-CNN detector (PyTorch)"
```

---

## Task 20: DistilBERT model (`ml/models/distilbert.py`)

**Files:**
- Create: `ml/models/distilbert.py`
- Test: `tests/ml/test_distilbert.py`

**Interfaces:**
- Consumes: `ml.models.base` (`Detector`, `register`).
- Produces:
  - `class DistilBertDetector(Detector)`, `name = "distilbert"`.
  - Uses `transformers` `AutoTokenizer` / `AutoModelForSequenceClassification` (`distilbert-base-uncased`, `num_labels=2`), `max_length=192`, fine-tune 2 epochs, lr 5e-5, batch 16, CPU (`no_cuda`), `torch.manual_seed(13)`.
  - `predict_proba`: softmax probability of class 1, shape `(n,)`.
  - `save(path)`: `model.save_pretrained` + `tokenizer.save_pretrained` under `path`. `load(path)`: `from_pretrained(path)`.

- [ ] **Step 1: Write failing test `tests/ml/test_distilbert.py`**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/ml/test_distilbert.py -v -m slow`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.models.distilbert'`.

- [ ] **Step 3: Implement `ml/models/distilbert.py`**

```python
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

from ml.models.base import Detector, register

_MODEL = "distilbert-base-uncased"
_MAXLEN = 192


@register
class DistilBertDetector(Detector):
    name = "distilbert"

    def __init__(self) -> None:
        torch.manual_seed(13)
        self.tok = AutoTokenizer.from_pretrained(_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            _MODEL, num_labels=2)

    def _encode(self, texts: list[str]) -> dict:
        return self.tok(texts, truncation=True, padding="max_length",
                        max_length=_MAXLEN, return_tensors="pt")

    def fit(self, tt, tl, vt, vl) -> None:
        enc = self._encode(tt)
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                           torch.tensor(tl))
        dl = DataLoader(ds, batch_size=16, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        for _epoch in range(2):
            for ids, mask, y in dl:
                opt.zero_grad()
                out = self.model(input_ids=ids, attention_mask=mask, labels=y)
                out.loss.backward()
                opt.step()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        self.model.eval()
        enc = self._encode(texts)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().numpy().astype(float)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)

    @classmethod
    def load(cls, path: Path) -> "DistilBertDetector":
        obj = cls.__new__(cls)
        Detector.__init__(obj)
        obj.tok = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.model.eval()
        return obj
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/Scripts/python -m pip install transformers==4.48.0 && .venv/Scripts/python -m pytest tests/ml/test_distilbert.py -v -m slow`
Expected: PASS (downloads the base model on first run).

- [ ] **Step 5: Train + evaluate all three**

Run: `.venv/Scripts/python -m ml.train --model distilbert && .venv/Scripts/python -m ml.evaluate`
Expected: `reports/MODEL_REPORT.md` has all three rows with latency + size columns populated.

- [ ] **Step 6: Commit**

```bash
git add ml/models/distilbert.py tests/ml/test_distilbert.py ml/artifacts/distilbert reports/MODEL_REPORT.md reports/eval.json
git commit -m "feat: DistilBERT detector and full 3-model report"
```

---

## Task 21: Dockerfiles + docker-compose

**Files:**
- Create: `inference_svc/Dockerfile`, `waf_proxy/Dockerfile`, `vuln_app/Dockerfile`, `log_ui/Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`
- Test: `tests/e2e/test_compose_config.py` (config-only lint, not a running stack)

**Interfaces:**
- Consumes: all service packages + `ml/` (mounted or copied for `inference_svc`).
- Produces: a `docker compose config`-valid file; services `vuln-app`, `inference-svc`, `waf-proxy`, `log-ui`; shared named volume `waf_data` mounted at `/data` on `waf-proxy` and `log-ui`; only `8080` and `8081` published.

- [ ] **Step 1: Write `.dockerignore`**

```
.venv
.git
__pycache__
datasets/raw
*.db
.pytest_cache
```

- [ ] **Step 2: Write `vuln_app/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY vuln_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vuln_app ./vuln_app
ENV APP_DB=/app/app.db SECURE_MODE=0
RUN python -m vuln_app.seed /app/app.db
CMD ["gunicorn", "-b", "0.0.0.0:8000", "vuln_app.app:app"]
```

- [ ] **Step 3: Write `inference_svc/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY inference_svc/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml ./ml
COPY inference_svc ./inference_svc
ENV ARTIFACTS_DIR=/app/ml/artifacts ACTIVE_MODEL=cnn BLOCK_THRESHOLD=0.5 INFERENCE_EAGER=1
CMD ["uvicorn", "inference_svc.app:app", "--host", "0.0.0.0", "--port", "9000"]
```

- [ ] **Step 4: Write `waf_proxy/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY waf_proxy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml/data ./ml/data
COPY ml/__init__.py ./ml/__init__.py
COPY waf_proxy ./waf_proxy
ENV EVENTS_DB=/data/events.db
CMD ["uvicorn", "waf_proxy.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 5: Write `log_ui/Dockerfile`**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY log_ui/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY waf_proxy/events.py ./waf_proxy/events.py
COPY waf_proxy/__init__.py ./waf_proxy/__init__.py
COPY log_ui ./log_ui
ENV EVENTS_DB=/data/events.db LOGUI_EAGER=1
CMD ["uvicorn", "log_ui.app:app", "--host", "0.0.0.0", "--port", "8081"]
```

- [ ] **Step 6: Write `docker-compose.yml`**

```yaml
services:
  vuln-app:
    build: {context: ., dockerfile: vuln_app/Dockerfile}
    environment:
      SECURE_MODE: "${SECURE_MODE:-0}"
    expose: ["8000"]

  inference-svc:
    build: {context: ., dockerfile: inference_svc/Dockerfile}
    environment:
      ACTIVE_MODEL: "${ACTIVE_MODEL:-cnn}"
      BLOCK_THRESHOLD: "${BLOCK_THRESHOLD:-0.5}"
    expose: ["9000"]

  waf-proxy:
    build: {context: ., dockerfile: waf_proxy/Dockerfile}
    depends_on: [vuln-app, inference-svc]
    environment:
      UPSTREAM_URL: "http://vuln-app:8000"
      INFERENCE_URL: "http://inference-svc:9000"
      ACTIVE_MODEL: "${ACTIVE_MODEL:-cnn}"
      BLOCK_THRESHOLD: "${BLOCK_THRESHOLD:-0.5}"
      INFERENCE_TIMEOUT_MS: "250"
      EVENTS_DB: "/data/events.db"
    volumes: ["waf_data:/data"]
    ports: ["8080:8080"]

  log-ui:
    build: {context: ., dockerfile: log_ui/Dockerfile}
    depends_on: [waf-proxy]
    environment:
      EVENTS_DB: "/data/events.db"
    volumes: ["waf_data:/data"]
    ports: ["8081:8081"]

volumes:
  waf_data:
```

- [ ] **Step 7: Write `tests/e2e/test_compose_config.py`**

```python
import shutil
import subprocess
import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(not shutil.which("docker"), reason="docker not installed")
def test_compose_config_is_valid():
    out = subprocess.run(["docker", "compose", "config"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "waf-proxy" in out.stdout
    assert "8080:8080" in out.stdout
```

- [ ] **Step 8: Validate compose config**

Run: `docker compose config`
Expected: prints resolved config, exit 0.

- [ ] **Step 9: Build and smoke-test the stack**

Run: `docker compose up --build -d && sleep 20 && curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/search?q=mouse"`
Expected: `200`. Then `curl -s -o /dev/null -w "%{http_code}" "http://localhost:8080/search?q=' OR 1=1 -- "` → `403`. Then `docker compose down`.

- [ ] **Step 10: Commit**

```bash
git add inference_svc/Dockerfile waf_proxy/Dockerfile vuln_app/Dockerfile log_ui/Dockerfile docker-compose.yml .dockerignore tests/e2e/test_compose_config.py
git commit -m "feat: dockerfiles and docker-compose stack"
```

---

## Task 22: E2E demo script + scenario tests (`scripts/demo.py`, `tests/e2e/test_scenarios.py`)

**Files:**
- Create: `scripts/demo.py`
- Create: `tests/e2e/test_scenarios.py`

**Interfaces:**
- Consumes: a running stack at `http://localhost:8080` (proxy) + `http://localhost:8081` (log-ui); `docker compose` for toggling `SECURE_MODE` / `ACTIVE_MODEL`.
- Produces:
  - `run_scenario(app_secure: bool, waf_on: bool) -> dict` — brings the stack to the requested config (via `docker compose up -d` with env), fires the canonical attacks (`admin' -- ` login, `UNION SELECT` search), returns `{"login_status": int, "search_leaked_users": bool, "search_status": int}`.
  - `main()` prints a 4-row table matching spec §9 and a false-positive line (runs `vuln_app.traffic_gen.run` and reports how many benign requests were blocked).
  - `tests/e2e/test_scenarios.py` (marked `e2e`, skipped unless `RUN_E2E=1`): asserts scenario 1 leaks users, scenarios 2–4 do not, and benign false-positive count is `0`.

- [ ] **Step 1: Write `tests/e2e/test_scenarios.py`**

```python
import os
import pytest

pytestmark = pytest.mark.e2e

if os.environ.get("RUN_E2E") != "1":
    pytest.skip("set RUN_E2E=1 to run", allow_module_level=True)

from scripts.demo import run_scenario, false_positive_count


def test_scenario1_vulnerable_no_waf_leaks():
    r = run_scenario(app_secure=False, waf_on=False)
    assert r["search_leaked_users"] is True
    assert r["login_status"] == 200


def test_scenario2_vulnerable_with_waf_blocks():
    r = run_scenario(app_secure=False, waf_on=True)
    assert r["search_leaked_users"] is False
    assert r["search_status"] == 403


def test_scenario3_secure_no_waf_blocks():
    r = run_scenario(app_secure=True, waf_on=False)
    assert r["search_leaked_users"] is False


def test_scenario4_secure_with_waf_blocks():
    r = run_scenario(app_secure=True, waf_on=True)
    assert r["search_leaked_users"] is False


def test_no_false_positives_on_benign_traffic():
    assert false_positive_count(n=60) == 0
```

- [ ] **Step 2: Run to verify it fails (import error)**

Run: `RUN_E2E=1 .venv/Scripts/python -m pytest tests/e2e/test_scenarios.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.demo'`.

- [ ] **Step 3: Implement `scripts/demo.py`**

```python
from __future__ import annotations
import os
import subprocess
import time

import httpx

PROXY = os.environ.get("PROXY_URL", "http://localhost:8080")
DIRECT = os.environ.get("DIRECT_URL", "http://localhost:8000")

UNION = "' UNION SELECT id, username, password, price FROM users -- "
LOGIN_BYPASS = {"username": "admin' -- ", "password": "x"}


def _compose_up(app_secure: bool) -> None:
    env = dict(os.environ, SECURE_MODE="1" if app_secure else "0")
    subprocess.run(["docker", "compose", "up", "-d"], env=env, check=True)
    time.sleep(8)


def run_scenario(app_secure: bool, waf_on: bool) -> dict:
    _compose_up(app_secure)
    base = PROXY if waf_on else DIRECT
    # when waf_on is False we hit vuln-app directly; requires exposing 8000
    # locally via `docker compose run` port or a compose override. For the
    # demo we publish 8000 in docker-compose.override.yml (dev only).
    with httpx.Client(base_url=base, timeout=15) as c:
        s = c.get("/search", params={"q": UNION})
        login = c.post("/login", data=LOGIN_BYPASS)
    leaked = b"s3cr3t-admin" in s.content
    return {"login_status": login.status_code,
            "search_status": s.status_code,
            "search_leaked_users": leaked}


def false_positive_count(n: int = 60) -> int:
    from vuln_app.traffic_gen import build_requests
    _compose_up(app_secure=False)
    blocked = 0
    with httpx.Client(base_url=PROXY, timeout=15) as c:
        for method, path, payload in build_requests(n, seed=7):
            r = c.get(path, params=payload) if method == "GET" \
                else c.post(path, data=payload)
            if r.status_code == 403:
                blocked += 1
    return blocked


def main() -> None:
    print(f"{'app':<10}{'waf':<6}{'login':<8}{'search':<8}leaked")
    for secure in (False, True):
        for waf in (False, True):
            r = run_scenario(secure, waf)
            print(f"{'secure' if secure else 'vuln':<10}"
                  f"{'on' if waf else 'off':<6}"
                  f"{r['login_status']:<8}{r['search_status']:<8}"
                  f"{r['search_leaked_users']}")
    fp = false_positive_count()
    print(f"\nfalse positives on 60 benign requests: {fp}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `docker-compose.override.yml` (dev-only, publishes vuln-app)**

```yaml
services:
  vuln-app:
    ports: ["8000:8000"]
```

Add `docker-compose.override.yml` to the file list; note in README that it is dev-only and Docker loads it automatically.

- [ ] **Step 5: Run the E2E scenario tests against the stack**

Run: `docker compose up --build -d && sleep 20 && RUN_E2E=1 .venv/Scripts/python -m pytest tests/e2e/test_scenarios.py -v ; docker compose down`
Expected: 5 tests PASS. If `test_no_false_positives` fails, inspect the log UI / `reports/MODEL_REPORT.md`; consider raising `BLOCK_THRESHOLD` or adding the offending benign phrasing to `datasets/seed/benign.txt` and retraining (Task 8/19), then re-run.

- [ ] **Step 6: Commit**

```bash
git add scripts/demo.py tests/e2e/test_scenarios.py docker-compose.override.yml
git commit -m "feat: end-to-end demo script and scenario tests"
```

---

## Task 23: Documentation (`README.md`, `SECURITY.md`)

**Files:**
- Create: `README.md`
- Create: `SECURITY.md`
- Modify: `reports/MODEL_REPORT.md` (only if regeneration needed)

**Interfaces:**
- Consumes: everything built.
- Produces: reader-facing docs. No code.

- [ ] **Step 1: Write `README.md`**

Sections, in order:
1. One-paragraph what/why (copy the spec's Purpose, condensed).
2. Architecture diagram (the ASCII block from the spec §2).
3. Quickstart: `docker compose up --build`, then open `http://localhost:8080` (shop) and `http://localhost:8081` (detection log).
4. The four demo scenarios table (spec §9) + the exact commands (`SECURE_MODE=1 docker compose up -d`, the two `curl` attack lines, `python -m vuln_app.traffic_gen --url http://localhost:8080 -n 50`).
5. Swapping the active model at runtime: `ACTIVE_MODEL=baseline docker compose up -d` and what to observe.
6. Training & evaluation: `python -m ml.data.pipeline`, `python -m ml.train --model {baseline,cnn,distilbert}`, `python -m ml.evaluate`; point at `reports/MODEL_REPORT.md`.
7. Fail-open note and the "this is not a production WAF" disclaimer.
8. Repo layout (the tree from this plan's File Structure).
9. Link to `SECURITY.md` and `docs/superpowers/specs/2026-09-08-ai-sqli-waf-design.md`.

- [ ] **Step 2: Write `SECURITY.md`**

For each of the three endpoints (`/search`, `/login`, `/product`):
- **Vulnerable code** — the exact `secure=False` branch from `vuln_app/queries.py`.
- **Attack** — the payload and, step by step, how the final SQL string is assembled and why it changes the query's meaning.
- **Fixed code** — the exact `secure=True` branch.
- **Why the fix works** — parameter binding sends data out-of-band from the SQL text.

Then two closing sections:
- **Defense in depth** — parameterized queries / ORM / input validation / least-privilege DB account (the read-only connection in `db.connect`) / generic errors. Reference OWASP A03:2021.
- **Why a WAF is not enough** — show 2–3 rows from `datasets/adversarial_testset.csv` that the model may miss (`/*!50000UNION*/`, case-mixing, double URL-encoding); conclude that the ML proxy buys time and visibility but the application-layer fix is mandatory.

- [ ] **Step 3: Verify all docs links resolve and code snippets match source**

Run: `.venv/Scripts/python -m pytest -q` (full suite, excluding `slow`/`e2e`)
Expected: PASS. Manually confirm each code block in `SECURITY.md` is copied verbatim from `vuln_app/queries.py`.

- [ ] **Step 4: Commit**

```bash
git add README.md SECURITY.md
git commit -m "docs: README quickstart and SECURITY.md defense-in-depth writeup"
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Task(s) |
|---|---|
| §1 goals / non-goals / success criteria | Task 22 (scenarios), Task 10 (report), Task 16 (secure proof) |
| §2.1 four services | Tasks 11, 14, 15, 18 |
| §2.2 request lifecycle + `/predict` shape | Tasks 11, 14 |
| §2.3 fail-open | Task 14 (`test_inference_down_fails_open`) |
| §2.3 runtime-swappable model/threshold | Task 11 (env), Task 14 (config), Task 21 (compose env), Task 23 (README §5) |
| §2.3 values-only, min length 3 | Task 3, Task 12 |
| §3 vulnerable app + endpoints | Task 15 |
| §3.3 SECURE_MODE matrix | Tasks 15, 16 |
| §3.5 traffic generator | Task 17 |
| §3.6 SECURITY.md | Task 23 |
| §4.1 sources | Task 2 |
| §4.2 pipeline | Tasks 3, 4, 5 |
| §4.3 leakage/bias controls | Task 4 (source-disjoint, MinHash), Task 2 (hard negatives in seed) |
| §4.4 adversarial testset | Task 2, consumed in Task 10 |
| §5 three models + shared interface | Tasks 6, 7, 19, 20 |
| §6 evaluation harness + MODEL_REPORT.md | Tasks 9, 10 |
| §7 log UI | Task 18 |
| §8 repo layout, LFS | Task 1 |
| §9 demo scenarios | Task 22 |
| §10 test strategy | every task (TDD); E2E in Task 22 |
| §11 Phase 2 SSH | intentionally omitted (out of scope, per spec) |
| §12 risks (DistilBERT latency, Windows SQLite volume) | Task 11 (best-effort budget), Task 21 (named volume) — Windows-volume fallback to HTTP event read is noted here as a known deviation if `docker compose up` shows `events.db` lock errors |

**2. Placeholder scan:** `_PRODUCTS` in Task 15 Step 6 is explicitly marked "extend to 25 rows" with a concrete starter list and an instruction line — acceptable (data, not logic). `download.py` `SOURCES` has empty `url`/`sha256` by design (network-optional; `SOURCES.md` documents intended sources) — the pipeline runs off `datasets/seed/` regardless. No `TODO`/`TBD`/"handle edge cases" strings remain.

**3. Type consistency:**
- `Detector.predict_proba(texts) -> np.ndarray shape (n,)` — consistent across Tasks 6, 7, 11, 19, 20.
- `EventStore.record(...)` — the `threshold REAL` column and `threshold: float` kwarg are now written into Task 13's interface block, schema, `record()` body, and all three test `record()` calls; Task 14 passes `threshold=settings.block_threshold` at both call sites; Task 18's `/summary` returns `newest["threshold"]` and its `_seed` helper passes `threshold=`. Consistent end to end.
- `score_values` (proxy, async, returns `list[dict] | None`) vs `ModelRegistry.score_values` (inference, sync) — same name, different modules and call sites; not a conflict but noted.
- `create_app` appears in `inference_svc.app`, `waf_proxy.decision`, `vuln_app.app`, `log_ui.app` — each in its own module; signatures differ intentionally.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-09-08-ai-sqli-waf.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
