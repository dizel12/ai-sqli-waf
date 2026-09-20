# ai-sqli-waf

A portfolio and learning project that demonstrates **machine-learning-based
detection of SQL injection attacks**, deployed as a blocking reverse proxy in
front of a deliberately vulnerable demo web application. Every request to the
demo shop passes through `waf-proxy`, which extracts the inspectable *values*
from the query string, form body, JSON body, and cookies, scores them with a
character-level CNN served by `inference-svc`, and returns `403` when the
active model's score crosses the block threshold. The project also tells a
**defense-in-depth** story: the ML proxy is a mitigation, not a cure. The demo
app ships with a `SECURE_MODE` toggle and a companion [`SECURITY.md`](SECURITY.md)
that shows the real application-layer fixes (parameterized queries, input
validation, least-privilege DB accounts) and the limits of a WAF.

## Architecture

Four containers on one `docker-compose` network. Only `waf-proxy` (8080) and
`log-ui` (8081) are published to the host.

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

| Service | Stack | Role |
|---|---|---|
| `vuln-app` | Flask + SQLite | Intentionally SQLi-vulnerable demo app (`/search`, `/login`, `/product`), with `SECURE_MODE` toggle |
| `waf-proxy` | FastAPI + httpx | Receive all traffic → extract inspectable values → call `inference-svc` → forward or block → record event |
| `inference-svc` | FastAPI + PyTorch / scikit-learn | Load all 3 models; `POST /predict` returns per-model scores + decision |
| `log-ui` | FastAPI + vanilla HTML/JS | Single page; streams detection events via SSE |

## Quickstart

Requires a running Docker daemon (Docker Desktop on Windows/macOS).

```bash
docker compose up --build
```

Then open:

- <http://localhost:8080> — the demo shop, proxied and inspected by the WAF
- <http://localhost:8081> — the real-time detection log (every inspected
  request shows as `ALLOWED` or `BLOCKED`, with the per-model scores)

## Demo scenarios

The four-way matrix below (app mode × WAF) is the core demo. Run each cell by
setting `SECURE_MODE` and choosing whether to hit `waf-proxy` (`:8080`) or the
app directly (`:8000`, published only by the dev override — see the last
section).

| # | App mode | WAF | Attack | Expected |
|---|---|---|---|---|
| 1 | vulnerable (`SECURE_MODE=0`) | off | `admin' -- ` login / `UNION SELECT` search | **succeeds** — data exfiltrated |
| 2 | vulnerable | on | same | **blocked** by ML (`403`); an obfuscated variant may still slip through → shows WAF limits |
| 3 | secure (`SECURE_MODE=1`) | off | same | **blocked** by parameterized queries |
| 4 | secure | on | same | **blocked** — layered defense |

Bring the stack up in secure or vulnerable mode:

```bash
# vulnerable (default)
docker compose up -d

# secure
SECURE_MODE=1 docker compose up -d
```

The two canonical attacks, as curl (against the proxy on `:8080`; swap to
`:8000` to hit the app directly):

```bash
# UNION-based data exfiltration through /search
curl -G 'http://localhost:8080/search' \
  --data-urlencode "q=' UNION SELECT id, username, password, 0 FROM users -- "

# authentication bypass through /login
curl 'http://localhost:8080/login' \
  --data-urlencode "username=admin' -- " \
  --data-urlencode 'password=x'
```

Generate benign traffic (also mixes in SQL-looking-but-legitimate input such
as `O'Brien` and `SELECT desk lamp` for a false-positive check):

```bash
python -m vuln_app.traffic_gen --url http://localhost:8080 -n 50
```

Run all four scenarios plus the false-positive check in one shot:

```bash
python scripts/demo.py
```

`scripts/demo.py` brings the stack up in each mode, fires the UNION and
login-bypass attacks through the proxy and directly, checks whether
`s3cr3t-admin` leaked, then replays 60 benign requests and prints how many
were wrongly blocked.

## Swapping the active model at runtime

`waf-proxy` and `inference-svc` read `ACTIVE_MODEL` (`baseline`, `cnn`, or
`distilbert`) and `BLOCK_THRESHOLD` from the environment. The active model is
scored synchronously and drives the block decision; the others are best-effort
and shown in the log UI for comparison.

```bash
ACTIVE_MODEL=baseline docker compose up -d
```

Re-run an obfuscated attack and watch the detection log at
<http://localhost:8081>: the header summary shows the current active model
and threshold, and every row shows all three models' scores side by side.

### Live-measured comparison (`baseline` vs `cnn`)

Run against the actual stack, not the offline eval set, using the same
requests against `ACTIVE_MODEL=baseline` and then `ACTIVE_MODEL=cnn`:

| request | value | baseline score | cnn score | outcome |
|---|---|---:|---:|---|
| plain SQLi (`' OR 1=1 --`) | via `/search` | 0.9999 | 0.9996 | both **BLOCK** |
| UNION exfiltration | `' UNION SELECT id, username, password, 0 FROM users -- ` | 0.9973 | 0.9893 | both **BLOCK** |
| login bypass | `admin' --` | 0.9977 | 0.9736 | both **BLOCK** |
| `+`-as-space obfuscation | `'+OR+1=1--+` | **0.135** | **0.989** | baseline **ALLOWS the attack through**, cnn blocks |
| hard-negative benign search | `SELECT desk lamp` | **0.512** | 0.424 | baseline **false-positives** (score crosses 0.5), cnn allows correctly |
| benign traffic (8 hard-negative samples: `chair 1=1 sale`, `O'Brien`, `1 or 2 day shipping`, …) | — | 1/8 blocked | 0/8 blocked | baseline false-positive rate 12.5% vs cnn 0% on this sample |

On obvious, unobfuscated SQLi both models are equally decisive. The gap shows
up exactly where the project's design intended it to: `baseline`'s char
n-gram features latch onto short numeric/keyword patterns (`SELECT`, `1=1`)
regardless of context, producing both a live false positive on a legitimate
search and a miss on a trivially re-spaced attack payload that `cnn`'s
learned representation handles correctly.

> **Note on `reports/adversarial.md` vs. live behavior.** The offline
> adversarial evaluation in `ml/evaluate.py` scores the raw corpus text
> directly, without passing it through `waf_proxy/extract.py`'s
> `normalize_value()` (one `unquote()` pass + whitespace-run collapse) the
> way a live request is scored. For **whitespace-substitution** techniques
> (tab/newline in `datasets/adversarial_testset.csv`), that collapse
> normalizes the obfuscation away for both models before they ever see it —
> live, `baseline` and `cnn` both score `'\tOR\t1=1\t-- ` at ~0.999, not the
> 0.000 vs 1.000 the offline table reports for that row. For **encoding**
> techniques (`double-url-encode`, `+`-as-space), the offline numbers do
> track live behavior, because those bypasses survive the single decode
> pass. Treat `reports/adversarial.md` as a lower bound on `cnn`/`distilbert`
> and an *inaccurate* (too pessimistic) number for whitespace-based rows —
> the live comparison above is the one to trust for deployment behavior.

## Training & evaluation

```bash
python -m ml.data.pipeline                       # acquire → clean → source-disjoint split → datasets/processed/*.parquet
python -m ml.train --model baseline              # writes ml/artifacts/baseline/
python -m ml.train --model cnn                   # writes ml/artifacts/cnn/
python -m ml.train --model distilbert            # writes ml/artifacts/distilbert/
python -m ml.evaluate                            # writes reports/MODEL_REPORT.md, confusion.md, latency.md, curves
```

The 3-model comparison (F1, PR-AUC, ROC-AUC, recall at FPR ≤ 0.1%, adversarial
detection rate, latency, size) and the rationale for `cnn` as the active model
are in [`reports/MODEL_REPORT.md`](reports/MODEL_REPORT.md).

**What the committed pipeline actually trains on.** The pipeline runs off the
hand-authored **seed corpus only** — roughly 175 malicious and 173 benign
values in `datasets/seed/`, which the source-disjoint split divides into
204 / 67 / 68 train / val / test rows. `datasets/download.py` is a stub (public
dataset URLs and licenses are still pending), so `datasets/raw/` stays empty
and `load_raw` returns nothing. The 3-way split is therefore filled by slicing
those two files into `seed_shard_*` buckets, which makes it effectively random;
near-duplicate removal is the only leakage control that still bites.
Consequently the held-out clean split is small and easily separated — all three
models score about 1.0 F1 on it — and the **adversarial detection rate** (see
[`reports/adversarial.md`](reports/adversarial.md)) is the metric that actually
discriminates between the models.

**Note:** the trained DistilBERT artifact (~269 MB) is gitignored
(`ml/artifacts/distilbert/`). Run `python -m ml.train --model distilbert`
locally if you want to serve it live; its comparison numbers are already
recorded in `reports/MODEL_REPORT.md`, so evaluation and the demo work without
it.

## Fail-open, and what this is not

- **Fail-open.** If `inference-svc` errors or exceeds the timeout (default
  250 ms), `waf-proxy` **forwards** the request and logs a warning. This keeps
  the demo available for a learning project; a real WAF would more likely
  fail-closed.
- **This is NOT a production WAF.** It is not hardened, not rate-limited, not
  tuned against evasion, and the model is trained on public payload lists. It
  exists to demonstrate ML text classification end to end and the layered
  relationship between a WAF and an application-layer fix. Do not put it in
  front of anything real. The `vuln-app` is intentionally vulnerable and every
  one of its source files says so.

## Repository layout

```
ai-sqli-waf/
├── docker-compose.yml           # 4 services on one network; publishes 8080 + 8081
├── docker-compose.override.yml  # dev-only, auto-loaded; also publishes vuln-app :8000
├── README.md
├── SECURITY.md                  # per-endpoint before/after, defense in depth, WAF limits
├── vuln_app/                    # Flask + SQLite demo app; queries.py, db.py, seed.py, traffic_gen.py
├── waf_proxy/                   # FastAPI proxy: extract.py, decision.py, events.py, config.py
├── inference_svc/               # FastAPI, loads 3 models, POST /predict
├── log_ui/                      # SSE + static detection-log page
├── ml/
│   ├── data/                    # pipeline.py (acquire → clean → split), clean.py, split.py
│   ├── models/                  # baseline.py, cnn.py, distilbert.py (shared Detector interface)
│   ├── eval/                    # metrics.py, report.py
│   ├── train.py  evaluate.py
│   └── artifacts/               # trained models (distilbert/ gitignored)
├── datasets/
│   ├── seed/                    # committed seed corpus (malicious.txt, benign.txt)
│   ├── raw/                     # gitignored; datasets/download.py fetches it
│   ├── processed/               # train/val/test .parquet (Git LFS)
│   └── adversarial_testset.csv  # evaluation-only obfuscation bypasses
├── reports/                     # MODEL_REPORT.md, latency.md, confusion.md
├── scripts/demo.py              # runs the four scenarios + false-positive check
└── tests/                       # pytest: ml, inference_svc, waf_proxy, vuln_app, log_ui, e2e
```

## See also

- [`SECURITY.md`](SECURITY.md) — the vulnerable vs. fixed query for each
  endpoint, how each attack is assembled, defense in depth, and why a WAF is
  not enough.
- [`docs/superpowers/specs/2026-09-08-ai-sqli-waf-design.md`](docs/superpowers/specs/2026-09-08-ai-sqli-waf-design.md)
  — the full design spec (architecture, data pipeline, models, evaluation).

## A note on `docker-compose.override.yml`

`docker-compose.override.yml` is **dev-only** and is auto-loaded by
`docker compose` with no extra flags. It publishes the vulnerable app's port
`8000` directly to the host so the demo script and curl examples can bypass the
proxy and hit `vuln-app` unfiltered. Remove or ignore this file for any
non-local use — exposing `vuln-app` directly defeats the entire point of the
proxy.
