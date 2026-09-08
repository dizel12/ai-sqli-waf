from __future__ import annotations
import os
import subprocess
import time

import httpx

PROXY = os.environ.get("PROXY_URL", "http://localhost:8080")
DIRECT = os.environ.get("DIRECT_URL", "http://localhost:8000")

UNION = "' UNION SELECT id, username, password, 0 FROM users -- "
LOGIN_BYPASS = {"username": "admin' -- ", "password": "x"}


def _compose_up(app_secure: bool) -> None:
    env = dict(os.environ, SECURE_MODE="1" if app_secure else "0")
    subprocess.run(["docker", "compose", "up", "-d"], env=env, check=True)
    time.sleep(15)


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
