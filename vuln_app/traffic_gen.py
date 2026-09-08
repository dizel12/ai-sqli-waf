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
