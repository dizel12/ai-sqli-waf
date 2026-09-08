from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from waf_proxy.events import EventStore

_STATIC = Path(__file__).resolve().parent / "static"


def create_app(events_db: str | None = None) -> FastAPI:
    db = events_db or os.environ.get("EVENTS_DB", "/data/events.db")
    store = EventStore(db)
    app = FastAPI(title="log-ui")

    @app.get("/events")
    def events(limit: int = 100) -> dict:
        return {"events": store.recent(limit)}

    @app.get("/summary")
    def summary() -> dict:
        rows = store.recent(100000)
        total = len(rows)
        blocked = sum(r["blocked"] for r in rows)
        newest = rows[0] if rows else None
        return {"total": total, "blocked": blocked,
                "block_rate": (blocked / total) if total else 0.0,
                "active_model": newest["active_model"] if newest else None,
                "threshold": newest["threshold"] if newest else None}

    @app.get("/events/stream")
    async def stream() -> StreamingResponse:
        async def gen():
            yield "retry: 2000\n\n"
            last = store.recent(1)
            last_id = last[0]["id"] if last else 0
            while True:
                for row in store.after(last_id):
                    last_id = row["id"]
                    yield f"data: {json.dumps(row)}\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    return app


app = create_app() if os.environ.get("LOGUI_EAGER") else None
