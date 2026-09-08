# SECURITY.md — SQL injection in `vuln-app`, and how to actually fix it

`vuln-app` is **intentionally vulnerable** (`# INTENTIONALLY VULNERABLE - DO
NOT DEPLOY` heads every source file). Its three endpoints — `/search`,
`/login`, `/product` — build SQL by string interpolation when
`SECURE_MODE=0`, and by parameter binding when `SECURE_MODE=1`. Both branches
live in `vuln_app/queries.py`; the code blocks below are copied verbatim from
that file.

This maps to **OWASP A03:2021 – Injection**. The ML proxy in front of the app
is a mitigation that buys time and visibility; the fixes in this document are
the actual remedy.

---

## `/search` — `WHERE name LIKE '%…%'` string concatenation

### Vulnerable code (`SECURE_MODE=0`)

```python
def search_sql(q: str, secure: bool):
    ...
    return (f"SELECT id, name, category, price FROM products "
            f"WHERE name LIKE '%{q}%'", ())
```

The user-controlled `q` is pasted straight into the SQL text and the
parameter tuple is empty (`()`).

### Attack

Request: `GET /search?q=' UNION SELECT id, username, password, 0 FROM users -- `

Step by step, how the final SQL string is assembled:

1. The template is `SELECT id, name, category, price FROM products WHERE name LIKE '%{q}%'`.
2. Substitute `q = ' UNION SELECT id, username, password, 0 FROM users -- `:
   ```sql
   SELECT id, name, category, price FROM products WHERE name LIKE '%' UNION SELECT id, username, password, 0 FROM users -- %'
   ```
3. The leading `'` in the payload closes the string literal that the template
   opened with `'%`. What was meant to be *data inside a `LIKE` pattern* is now
   parsed as *SQL syntax*.
4. `UNION SELECT id, username, password, 0 FROM users` appends a second result
   set with the same column count (4). The `0` is a filler for the `price`
   column so the arities match.
5. `-- ` comments out the trailing `%'` that the template still contributes,
   so the statement parses cleanly.

Result: the response table now contains every row of `users`, including
plaintext passwords. The query's meaning changed from "find products whose
name matches a pattern" to "also dump the users table" because the payload
crossed the boundary between SQL code and string data.

### Fixed code (`SECURE_MODE=1`)

```python
def search_sql(q: str, secure: bool):
    if secure:
        return ("SELECT id, name, category, price FROM products "
                "WHERE name LIKE ?", (f"%{q}%",))
```

### Why the fix works

The SQL text is now a fixed constant with a single `?` placeholder. The value
`f"%{q}%"` is passed **out of band** in the parameter tuple. The database
driver sends the statement and the parameter separately; the parameter is
bound as a typed value and is never parsed as SQL. A `q` of
`' UNION SELECT … -- ` is treated as a literal `LIKE` pattern containing those
characters — it matches no product name and changes nothing about the
statement's structure.

---

## `/login` — `WHERE username='…' AND password='…'` string concatenation

### Vulnerable code (`SECURE_MODE=0`)

```python
def login_sql(u: str, p: str, secure: bool):
    ...
    return (f"SELECT id, username FROM users "
            f"WHERE username = '{u}' AND password = '{p}'", ())
```

### Attack

Request: `POST /login` with `username=admin' -- ` and `password=x`

Step by step:

1. The template is `SELECT id, username FROM users WHERE username = '{u}' AND password = '{p}'`.
2. Substitute `u = admin' -- ` and `p = x`:
   ```sql
   SELECT id, username FROM users WHERE username = 'admin' -- ' AND password = 'x'
   ```
3. The `'` after `admin` closes the `username` string literal.
4. `-- ` starts a SQL line comment, so `' AND password = 'x'` — the entire
   password check — is discarded.
5. The statement the database actually runs is
   `SELECT id, username FROM users WHERE username = 'admin'`.

Result: it returns the `admin` row with no password verification. `vuln_app/app.py`
treats any returned row as a successful login, so the attacker is authenticated
as `admin`. The query's meaning changed from "match username **and** password"
to "match username only" because the payload injected a comment that deleted
the second condition.

### Fixed code (`SECURE_MODE=1`)

```python
def login_sql(u: str, p: str, secure: bool):
    if secure:
        return ("SELECT id, username FROM users "
                "WHERE username = ? AND password = ?", (u, p))
```

### Why the fix works

Both conditions are fixed SQL with `?` placeholders; `u` and `p` are bound as
values. `admin' -- ` becomes a literal username string that is compared with
`=` against the `username` column. There is no row whose username is the
literal string `admin' -- `, so the login fails. The comment sequence
never reaches the SQL parser because it is inside a bound parameter, not the
statement text. Verified by the remediation test in `tests/vuln_app/`.

---

## `/product` — `WHERE id = …` unvalidated integer

### Vulnerable code (`SECURE_MODE=0`)

```python
def product_sql(pid: str, secure: bool):
    ...
    return (f"SELECT id, name, category, price FROM products "
            f"WHERE id = {pid}", ())
```

`pid` comes from `request.args.get("id", "0")` — it is a **string**, and it is
interpolated without quotes.

### Attack

Request: `GET /product?id=5 OR 1=1`

Step by step:

1. The template is `SELECT id, name, category, price FROM products WHERE id = {pid}`.
2. Substitute `pid = 5 OR 1=1`:
   ```sql
   SELECT id, name, category, price FROM products WHERE id = 5 OR 1=1
   ```
3. Because `pid` is not quoted, the injected text `OR 1=1` is parsed as a
   boolean expression, not as part of an integer literal.
4. `id = 5 OR 1=1` is true for every row, so the `WHERE` clause no longer
   filters anything.

Result: the endpoint returns the entire `products` table instead of one row.
A variant such as `id=5 AND (SELECT …)` or a deliberately malformed
`id=5'` triggers a database error whose text is returned in the response body
(`vuln_app/app.py` returns `f"SQL error: {exc}\nquery: {sql}"` at
`SECURE_MODE=0`), enabling error-based extraction. The query's meaning changed
from "one product by id" to "all products" because an unquoted string was
allowed to contribute operators.

### Fixed code (`SECURE_MODE=1`)

```python
def product_sql(pid: str, secure: bool):
    if secure:
        return ("SELECT id, name, category, price FROM products "
                "WHERE id = ?", (int(pid),))
```

### Why the fix works

Two independent defenses: `int(pid)` rejects anything that is not a valid
integer with a `ValueError` before a query is ever built (input validation),
and the surviving integer is bound to a `?` placeholder rather than
interpolated. `5 OR 1=1` never gets to the database — `int("5 OR 1=1")`
raises, and `vuln_app/app.py` returns a generic `"internal error", 500` at
`SECURE_MODE=1` instead of echoing the SQL.

---

## Defense in depth

No single control is sufficient. `SECURE_MODE=1` turns on all of these at once:

1. **Parameterized queries (primary fix).** Every statement is a fixed string
   with `?` placeholders; user data travels in the parameter tuple and is
   bound as typed values, never parsed as SQL. This is the control that
   actually closes the vulnerability class. An ORM or query builder gives the
   same guarantee as long as raw-SQL escape hatches are not used with
   interpolated input.
2. **Input validation.** `/product` calls `int(pid)` and refuses
   non-integers. Type, length, and whitelist checks at the edge shrink the
   attack surface and catch malformed input early — but they are a
   supplement, not a replacement, for binding.
3. **Least-privilege database account.** At `SECURE_MODE=1`, `vuln_app/db.py`
   opens the connection read-only:

   ```python
   def connect(db_path: str, readonly: bool = False) -> sqlite3.Connection:
       if readonly:
           conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
       else:
           conn = sqlite3.connect(db_path)
   ```

   `app.py` passes `readonly=_secure()`. Even if an injection were to get
   through, `INSERT` / `UPDATE` / `DROP` would fail because the connection has
   no write privilege. In a real deployment this is a dedicated DB user
   granted only `SELECT` on only the tables the app needs.
4. **Generic error messages.** At `SECURE_MODE=0` the app returns
   `f"SQL error: {exc}\nquery: {sql}"`; at `SECURE_MODE=1` it returns
   `"internal error", 500`. Leaking the query text and driver exceptions
   hands an attacker a free oracle for blind and error-based extraction.

Reference: **OWASP A03:2021 – Injection**
(<https://owasp.org/Top10/A03_2021-Injection/>).

## Why a WAF is not enough

The ML proxy scores request *values* and blocks the ones that look malicious.
Obfuscation defeats pattern-like detectors. These rows are taken verbatim from
`datasets/adversarial_testset.csv` (the evaluation-only bypass set):

| `text` | `technique` | Bypass idea |
|---|---|---|
| `' /*!50000UNION*/ SELECT username,password FROM users -- ` | `inline-comment` | MySQL version-gated comment `/*!50000…*/` hides the `UNION` keyword from a naive tokenizer while MySQL still executes it |
| `' UnIoN sElEcT username,password FROM users -- ` | `case-mixing` | SQL keywords are case-insensitive, so `UnIoN sElEcT` runs but does not match a case-sensitive signature |
| `%2527%2520OR%25201%253D1` | `double-url-encode` | Double URL-encoding: `%25` decodes to `%`, so one decode pass yields `%27%20OR%201%3D1` and only a second pass reveals `' OR 1=1` |

Per `reports/MODEL_REPORT.md`, adversarial detection rate is **0.808**
(`baseline`), **0.885** (`cnn`, the active model), and **0.904**
(`distilbert`) — all well below `1.0`. Roughly one obfuscated payload in nine
gets past the active model, and a motivated attacker iterates until one does.

Conclusion: **the application-layer fix is mandatory.** Parameterized queries
close the vulnerability regardless of how the payload is encoded, because the
data never reaches the SQL parser. The ML proxy is worth running as a second
layer — it buys time, blocks the easy attempts, and gives real-time
visibility into what is being tried (see the detection log at
`http://localhost:8081`) — but it is not a substitute for fixing the queries.
