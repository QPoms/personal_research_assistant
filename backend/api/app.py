"""FastAPI composition for runtime state and replayable event streams."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from backend.api.sse import resolve_event_cursor, stream_run_events
from backend.config import Settings, load_settings
from backend.runtime.run_state import RunRecord, RunStatus
from backend.storage.sqlite_runtime import SQLiteRuntimeStore


def create_app(settings: Settings | None = None) -> FastAPI:
    configured = settings or load_settings()
    store = SQLiteRuntimeStore(configured.resolved_runtime_database_path())
    app = FastAPI(title="Personal Research Assistant", version="0.1.0")
    app.state.settings = configured
    app.state.runtime_store = store

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/runs", response_model=list[RunRecord])
    async def list_runs(
        status: RunStatus | None = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> tuple[RunRecord, ...]:
        return await asyncio.to_thread(
            store.list_runs,
            status=status.value if status is not None else None,
            limit=limit,
        )

    @app.get("/api/v1/runs/{run_id}", response_model=RunRecord)
    async def get_run(run_id: str) -> RunRecord:
        record = await asyncio.to_thread(store.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return record

    @app.get("/api/v1/runs/{run_id}/events")
    async def run_events(
        request: Request,
        run_id: str,
        after_seq: Annotated[int | None, Query(ge=0)] = None,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> StreamingResponse:
        record = await asyncio.to_thread(store.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        try:
            cursor = resolve_event_cursor(after_seq, last_event_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return StreamingResponse(
            stream_run_events(request, store, run_id, after_seq=cursor),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    return app


__all__ = ["create_app"]
