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
