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
