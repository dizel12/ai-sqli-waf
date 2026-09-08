from __future__ import annotations

import json
import os

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
    secure_mode_flag = os.environ.get("SECURE_MODE", "0") == "1"

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

        if not results:
            # fail-open (nothing to score, inference unavailable, or empty results)
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
                content=json.dumps({
                    "blocked_by": "ai-waf",
                    "model": worst["active_model"],
                    "score": round(worst["score"], 4),
                    "matched_param": matched.name,
                }),
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
