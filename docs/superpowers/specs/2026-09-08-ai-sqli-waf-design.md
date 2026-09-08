# AI-SQLi-WAF — Design Spec

**Date:** 2026-09-08
**Status:** Approved (brainstorming), pending implementation plan
**Type:** Architectural / new project
**Repo:** `C:\Users\sod09\Documents\GitHub\ai-sqli-waf`

---

## 1. Purpose & Goals

A portfolio and learning project that demonstrates **machine-learning-based
detection of SQL injection attacks**, deployed as a blocking reverse proxy in
front of a deliberately vulnerable demo web application.

The project also demonstrates **defense in depth**: the ML proxy is a
mitigation, not a cure. The demo app ships with a `SECURE_MODE` toggle and a
`SECURITY.md` write-up showing the real application-layer fixes (parameterized
queries, input validation, least-privilege DB accounts) and the limits of a
WAF.

### Learning objectives

- Text classification end to end: data collection, leakage-aware splitting,
  representation choices, class imbalance, evaluation metrics.
- Model progression and comparison: classical baseline → char-level CNN
  (PyTorch) → fine-tuned DistilBERT, compared on accuracy, latency, and size.
- Serving an ML model behind a stable interface, and wiring it into a
  request-blocking proxy with sensible failure behavior.
- Containerized multi-service delivery with `docker-compose`.

### Non-goals

- Not a production WAF. Not hardened for real deployment.
- Not an SSH defense project. SSH log anomaly detection is recorded as an
  optional Phase 2 (see §10) and is **out of scope** for this spec.
- No full analytics dashboard — only a lightweight real-time detection log UI.

### Success criteria

- `docker compose up` brings up all four services; a demo script runs four
  scenarios (see §9) with the expected outcomes.
- `reports/MODEL_REPORT.md` contains a 3-model comparison with PR/ROC curves,
  a latency benchmark, and a written rationale for the active model choice.
- The active model achieves a materially better *recall at FPR ≤ 0.1%* than
  the classical baseline on the held-out test set, and a measurably higher
  detection rate on the adversarial (obfuscation) test set.
- `SECURE_MODE=1` provably defeats the canonical attacks that succeed at
  `SECURE_MODE=0`, verified by tests.

---

## 2. System Architecture

Approach chosen: **thin proxy + dedicated inference service + log UI** (4
containers on one `docker-compose` network).

```
                       ┌──────────────────────────────┐
  client  ── :8080 ──▶  │          waf-proxy           │
                       │  (FastAPI + httpx)            │
                       │  1. extract candidate values │
                       │  2. POST /predict ───────────┼──▶ inference-svc  :9000
                       │  3a. benign → forward ───────┼──▶ vuln-app       :8000
                       │  3b. malicious → 403 + event │
                       │  4. append event to events.db│
                       └───────────────┬──────────────┘
                                       │ SQLite (events.db, shared volume)
                                       ▼
                              log-ui  :8081  (SSE + static page)
```

### 2.1 Services

| Service | Stack | Role | Ports |
|---|---|---|---|
| `vuln-app` | Flask + SQLite | Intentionally SQLi-vulnerable demo app (search + login + product), with `SECURE_MODE` toggle | internal 8000 |
| `waf-proxy` | FastAPI + httpx | Receive all traffic → extract inspectable values → call `inference-svc` → forward or block → record event | **8080 (exposed)** |
| `inference-svc` | FastAPI + PyTorch / scikit-learn | Load all 3 models; `POST /predict` returns per-model scores + decision | internal 9000 |
| `log-ui` | FastAPI + vanilla HTML/JS | Single page; streams detection events via SSE | 8081 |

Events are written by `waf-proxy` to a shared SQLite file (`events.db` on a
compose volume) and read by `log-ui`. No message broker.

### 2.2 Request lifecycle

1. Client calls e.g. `http://localhost:8080/search?q=...`.
2. `waf-proxy` collects **values** (not key names) from: URL query params,
   form-encoded body, JSON body, cookies. (Selected headers optional, off by
   default.)
3. Each value ≥ 3 chars is sent to `inference-svc POST /predict`.
4. Response shape:
   ```json
   {
     "scores": { "baseline": 0.02, "cnn": 0.03, "distilbert": 0.01 },
     "active_model": "cnn",
     "score": 0.03,
     "decision": "benign",
     "threshold": 0.5
   }
   ```
   The **active model** is always scored synchronously and drives `score` /
   `decision`. Non-active models are best-effort: their entries in `scores`
   may be `null` if scoring them would exceed the request budget (see §12).
   The offline evaluation harness (§6) always runs every model in full.
5. If the **active model** score ≥ threshold for any inspected value → block.
6. Benign: forward the original request unchanged to `vuln-app`, return its
   response.
7. Malicious: return
   `403 {"blocked_by":"ai-waf","model":"cnn","score":0.87,"matched_param":"q"}`
   and append an event to `events.db`.
8. `log-ui` pushes the new event to the page.

### 2.3 Policy decisions

- **Fail-open.** If `inference-svc` errors or exceeds the timeout (default
  250 ms), the request is **forwarded** and a warning is logged. Rationale:
  demo availability for a learning project. The trade-off (a real WAF might
  fail-closed) is documented in `README.md` and `SECURITY.md`.
- **Runtime-swappable active model and threshold** via `waf-proxy` env vars
  (`ACTIVE_MODEL`, `BLOCK_THRESHOLD`), so a demo can show the baseline missing
  a payload the CNN catches.
- **Values only.** Key names are not inspected. Values shorter than 3 chars
  are skipped.
- **Every inspected request produces an event** (ALLOWED or BLOCKED) so the
  log UI shows total volume, not just blocks.

---

## 3. Vulnerable Demo App (`vuln-app`)

Flask + SQLite (`app.db`), initialized on container start by `seed.py`. Every
source file carries a header comment:
`# INTENTIONALLY VULNERABLE - DO NOT DEPLOY`.

### 3.1 Schema (seeded)

- `products(id, name, category, price)` — 20–30 rows.
- `users(id, username, password)` — 3–4 rows, plaintext passwords (demo only).

### 3.2 Endpoints — vulnerable at `SECURE_MODE=0`

| Route | Vulnerability | Normal use | Attack example |
|---|---|---|---|
| `GET /search?q=` | `... WHERE name LIKE '%{q}%'` string concat | `q=mouse` | `q=' UNION SELECT username,password,3,4 FROM users -- ` |
| `POST /login` | `... WHERE username='{u}' AND password='{p}'` | valid login | `u=admin' -- ` auth bypass |
| `GET /product?id=` | `... WHERE id={id}` unvalidated integer | `id=5` | `id=5 OR 1=1`, error-based |

DB errors are returned in the response body (enables error-based SQLi demo).

### 3.3 `SECURE_MODE` toggle

| Aspect | `SECURE_MODE=0` (default) | `SECURE_MODE=1` |
|---|---|---|
| Queries | string concatenation | parameterized (`execute("... WHERE id=?", (id,))`) |
| Input | none | type / length / whitelist validation |
| DB account | full privileges | read-only account (least privilege) |
| Errors | stack trace exposed | generic error message |

Enables a 4-way demo matrix (see §9).

### 3.4 UI

Minimal HTML: a search box + results table, and a login form. Almost no
styling — function only.

### 3.5 Benign traffic generator (`vuln-app/traffic_gen.py`)

Generates random legitimate requests for the demo and false-positive testing.
Deliberately mixes in **SQL-looking but legitimate** input, e.g.
`q=O'Brien`, `q=SELECT desk lamp`, `q=chair 1=1 sale`. If the proxy blocks
these, that is a false positive and is shown as such.

### 3.6 `SECURITY.md`

For each vulnerability:

- Vulnerable code (before) / safe code (after), side by side.
- Step-by-step of how the query is assembled and why the attack works.
- OWASP reference (A03:2021 – Injection); parameterization / ORM / input
  validation / least privilege / the role and limits of a WAF.
- A "Why a WAF is not enough" section: obfuscation bypasses
  (`/*!50000UNION*/`, encoding, case mixing), therefore the application-layer
  fix is mandatory.

---

## 4. Data Pipeline

### 4.1 Sources

| Label | Sources | Approx. size |
|---|---|---|
| **Malicious (1)** | sqlmap payload lists; PayloadsAllTheThings / SQLi; Kaggle "SQL Injection Dataset" (several, merged); public SQLi research datasets | 30–50k |
| **Benign (0)** | CSIC 2010 HTTP dataset (normal requests); general search-term / form-input corpora; `traffic_gen.py` output; a hand-built "SQL-looking but legitimate" set | 40–60k |

Target roughly balanced (50:50 – 60:40). Imbalance handled with
`class_weight` / oversampling. Evaluation always references the PR curve.

### 4.2 Normalization pipeline (`ml/data/pipeline.py`)

1. **Acquire** — each source downloaded by `datasets/download.py` into
   `datasets/raw/` verbatim; licenses recorded in `datasets/raw/SOURCES.md`.
   `raw/` is **not committed**.
2. **Extract** — pull only *values* (query-string values, form values, JSON
   values) from HTTP dumps — the same unit the proxy inspects.
3. **Clean** — one URL-decode pass, dedupe, whitespace normalization, length
   filter (3–2048 chars).
4. **Label** — source-based label + a manual audit sample (200 per class
   eyeballed).
5. **Split — by source** into train/val/test (60/20/20) so the same payload
   list cannot straddle train and test.
6. **Output** — `datasets/processed/{train,val,test}.parquet`, columns
   `text, label, source`. Committed via **Git LFS**.

### 4.3 Leakage & bias controls

- **Source-disjoint split** (step 5).
- **Near-duplicate removal** — normalize, then hash-dedupe; MinHash to drop
  near-duplicates (e.g. `?id=1` vs `?id=2`).
- **"Easy negatives" trap** — deliberately include long sentences, special
  characters, base64, and JSON among benign samples so the model cannot
  simply learn "long + punctuation ⇒ malicious".
- **Frozen holdout** — `test.parquet` is never inspected during model
  development; used exactly once, at the end.

### 4.4 Separate artifact: `datasets/adversarial_testset.csv`

Evaluation-only, never trained on. 200–300 hand-built obfuscation bypasses:
inline comments `/*!UNION*/`, case mixing, double URL-encoding, `CHAR()` /
`0x` encoding, whitespace → `/**/`, `UNION` → `UNI/**/ON`. Detection rate on
this set is a headline portfolio metric.

---

## 5. Models

Location `ml/models/`. Shared contract so training, evaluation, and
`inference-svc` treat all three identically:

```python
class Detector:
    def fit(self, train, val) -> None: ...
    def predict_proba(self, texts: list[str]) -> np.ndarray: ...   # shape (n,), range [0,1]
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "Detector": ...
```

| Model | Representation | Architecture | Notes |
|---|---|---|---|
| **baseline** | char n-gram (3–5) TF-IDF via `HashingVectorizer` | `LinearSVC` + `CalibratedClassifierCV` (for probabilities) | trains in seconds; can surface top contributing n-grams |
| **cnn** | char-level integer encoding, embedding dim 64 | `Conv1d` with kernels 3/5/7 → global max pool → FC → sigmoid (PyTorch) | **default active model**; target CPU inference < 5 ms for a single value |
| **distilbert** | `distilbert-base-uncased` tokenizer | HF `AutoModelForSequenceClassification` fine-tune, 2–3 epochs | performance ceiling; measure inference latency + model size to argue it is overkill for the proxy |

- Training: `ml/train.py --model {baseline,cnn,distilbert}` → writes
  `ml/artifacts/{model}/` + `metrics.json`.
- Reproducibility: fixed seed; dataset hash recorded; `ml/artifacts/{model}/run.json`
  holds hyperparameters, git commit hash, training time.
- Artifacts committed via **Git LFS**.

---

## 6. Evaluation Harness (`ml/evaluate.py`)

Runs all models against two fixed sets — `datasets/processed/test.parquet`
and `datasets/adversarial_testset.csv` — in one pass. Outputs to `reports/`.

- **Metrics:** Precision, Recall, F1, PR-AUC, ROC-AUC, and — as the
  operational headline — **Recall at FPR ≤ 0.1%** (false positives are
  costly for a WAF).
- **Curves:** PR and ROC with all three models overlaid
  (`reports/pr_curve.png`, `reports/roc_curve.png`).
- **Confusion matrices:** one per model.
- **Latency benchmark:** batch=1 and batch=32 inference time p50/p99, plus
  model file size → `reports/latency.md`.
- **Adversarial detection-rate table:** recall per bypass technique.
- **Error analysis:** top 30 false positives / false negatives per model
  dumped to `reports/{model}_fp.csv` / `reports/{model}_fn.csv`.
- **Explainability:** baseline — highest-contribution n-grams; cnn —
  character importance (gradient / occlusion) on a few example inputs.

### Final artifact: `reports/MODEL_REPORT.md`

3-model comparison table + curves + a written rationale for choosing `cnn`
as the active model (performance vs latency vs size). This is the primary
document to show in a portfolio review.

---

## 7. Detection Log UI (`log-ui`)

- Single page, no framework — vanilla JS + minimal CSS.
- `waf-proxy` writes to `events.db` (SQLite). `log-ui` serves
  `GET /events/stream` (SSE) for new events and `GET /events?limit=100` for
  the initial load.
- Row: timestamp · method + path · inspected param name · value (truncated) ·
  3-model scores (bars) · active model · **BLOCKED / ALLOWED** badge ·
  `SECURE_MODE` state.
- Header summary: total requests / blocked count / block rate / current
  active model + threshold.
- Filter toggle: "blocked only".

---

## 8. Repository Layout & Ops

```
ai-sqli-waf/
├── docker-compose.yml
├── README.md                  # architecture diagram, 4 demo scenarios, how to run
├── SECURITY.md                # per-vuln before/after, WAF limits
├── reports/MODEL_REPORT.md    # 3-model comparison (evaluation output)
├── vuln-app/                  # Flask + SQLite, SECURE_MODE toggle, seed.py, traffic_gen.py
├── waf-proxy/                 # FastAPI proxy, writes events.db, env-var model/threshold
├── inference-svc/             # FastAPI, loads 3 models, POST /predict
├── log-ui/                    # SSE + static page
├── ml/
│   ├── data/pipeline.py       # acquire → clean → split
│   ├── models/                # baseline.py, cnn.py, distilbert.py (shared interface)
│   ├── train.py  evaluate.py
│   └── artifacts/             # trained models (Git LFS)
├── datasets/                  # raw/ (download script, gitignored), processed/ (LFS), adversarial_testset.csv
└── tests/
```

- Python 3.11. Separate `requirements.txt` per service. `waf-proxy` and
  `inference-svc` are CPU-only.
- **Git LFS** for model artifacts and `.parquet` files (LFS is configured
  globally). `datasets/raw/` is gitignored; `datasets/download.py` fetches it.
- `docker-compose.yml` wires the four services on one network; only
  `waf-proxy` (8080) and `log-ui` (8081) are published to the host.

---

## 9. Demo Scenarios

Driven by a script (`scripts/demo.sh` or `scripts/demo.py`) after
`docker compose up`:

| # | App mode | WAF | Attack | Expected |
|---|---|---|---|---|
| 1 | vulnerable (`SECURE_MODE=0`) | off | `admin' -- ` login / `UNION SELECT` search | **succeeds** — data exfiltrated |
| 2 | vulnerable | on | same | **blocked** by ML (403); obfuscated variant may still slip through → shows WAF limits |
| 3 | secure (`SECURE_MODE=1`) | off | same | **blocked** by parameterized queries |
| 4 | secure | on | same | **blocked** — layered defense |

Also: run `traffic_gen.py` and confirm the SQL-looking legitimate inputs are
**not** blocked (false-positive check), visible in the log UI.

---

## 10. Testing Strategy

`tests/`, pytest. Implementation follows the test-driven-development skill.

- **Data pipeline:** no split leakage (zero source overlap across
  train/val/test), label distribution, near-dup removal works.
- **Model interface:** all three satisfy the `predict_proba` contract
  (shape, range [0,1]) on small dummy data.
- **inference-svc:** `/predict` response schema; 6 known payloads → malicious,
  6 legitimate inputs → benign (active-model threshold).
- **waf-proxy:** malicious request → 403 + event recorded; benign → 200 +
  forwarded (mocked upstream); `inference-svc` down → fail-open + warning.
- **vuln-app:** at `SECURE_MODE=1`, `admin' -- ` login bypass **fails**
  (remediation verification).
- **E2E (compose):** a script runs the four §9 scenarios via curl after
  `docker compose up` and asserts the expected responses.

### Suggested implementation order (TDD)

data pipeline → baseline model → evaluation harness → inference-svc →
waf-proxy → vuln-app (vulnerable + secure) → log-ui → cnn → distilbert →
E2E → docs.

---

## 11. Phase 2 (Optional, Out of Scope Here)

SSH log anomaly detection: Cowrie honeypot session logs → per-session command
sequences as features → anomaly / malicious classification. Would add a
`POST /predict/ssh` endpoint to `inference-svc` with a separate model.
Recorded as a future extension only; not part of this spec.

---

## 12. Open Risks

- **Dataset quality / licensing.** Public SQLi datasets vary in quality and
  license. Mitigation: `SOURCES.md` records provenance and license per
  source; manual audit sample; drop sources with unclear licensing.
- **Benign data realism.** If benign samples are too clean, metrics will be
  optimistic. Mitigation: the "easy negatives trap" controls in §4.3 and the
  false-positive demo check in §9.
- **DistilBERT latency.** May be too slow for the 250 ms proxy budget even as
  a non-active model if shadow-scored inline. Mitigation: shadow-scoring of
  non-active models is asynchronous / best-effort and never blocks the
  forward path.
- **Windows + Docker file sharing** for the shared `events.db` volume.
  Mitigation: if SQLite-over-bind-mount is flaky on Windows, switch `log-ui`
  to read events from `waf-proxy` over HTTP instead of a shared file.
