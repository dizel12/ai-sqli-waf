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
        sql = "<not built>"
        try:
            sql, params = queries.search_sql(q, _secure())
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
        sql = "<not built>"
        try:
            sql, params = queries.login_sql(u, p, _secure())
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
        sql = "<not built>"
        try:
            sql, params = queries.product_sql(pid, _secure())
            rows = _run(sql, params)
        except Exception as exc:  # noqa: BLE001
            if _secure():
                return "internal error", 500
            return f"SQL error: {exc}\nquery: {sql}", 500
        return render_template("index.html", q=f"id={pid}", rows=rows)

    return app


app = create_app(os.environ.get("APP_DB", "app.db"))
