# Dataset source registry

Public datasets that the full data pipeline may ingest to augment the hand-authored
seed corpus in `datasets/seed/`. Fetching is performed by `datasets/download.py`, which
is **network-optional**: with `--offline` (or when a source has no `url`) it prints a
`skip` line and exits 0, and the pipeline runs entirely off `datasets/seed/`.

> **Note:** Everything written under `datasets/raw/` is gitignored (`datasets/raw/*`)
> **except this file** (`!datasets/raw/SOURCES.md`). Downloaded corpora are never
> committed; only this registry is tracked. Re-run `python datasets/download.py` to
> repopulate `raw/` locally.

| Name | URL | License | Contributes | Notes |
|------|-----|---------|-------------|-------|
| PayloadsAllTheThings SQLi | https://github.com/swisskyrepo/PayloadsAllTheThings/tree/master/SQL%20Injection | MIT | malicious | Curated SQL injection payloads and intruder wordlists across many DB engines. |
| sqlmap payloads | https://github.com/sqlmapproject/sqlmap/tree/master/data/xml/payloads | GPL-2.0 | malicious | Boolean-blind, error-based, time-blind, UNION and stacked-query payload templates extracted from sqlmap's detection XML. |
| Kaggle SQLi dataset (`sajid576/sql-injection-dataset`) | https://www.kaggle.com/datasets/sajid576/sql-injection-dataset | CC0-1.0 | malicious | Labeled SQL-injection vs plain-query text samples; used to broaden malicious coverage. Requires Kaggle auth to download. |
| CSIC 2010 HTTP dataset | https://www.isi.csic.es/dataset/ | research-use | benign | Normal (and anomalous) HTTP requests to a web app; the normal split supplies realistic benign parameter values. Research-use terms — not redistributed. |

## How each maps into the pipeline

- **malicious** sources are concatenated, de-duplicated, and near-deduped (MinHash/LSH via
  `datasketch`) against `datasets/seed/malicious.txt`.
- **benign** sources supply hard-negative parameter values that are de-duplicated against
  `datasets/seed/benign.txt`.
- `datasets/adversarial_testset.csv` is **eval-only** and is never mixed into training data,
  regardless of which sources are fetched.
