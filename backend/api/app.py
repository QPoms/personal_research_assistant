"""FastAPI composition for runtime state and replayable event streams."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from backend.api.sse import resolve_event_cursor, stream_run_events
from backend.config import PROJECT_ROOT, Settings, load_settings
from backend.runtime.supervisor import RunSupervisor
from backend.runtime.run_state import RunRecord, RunStatus
from backend.storage.sqlite_runtime import SQLiteRuntimeStore, WorkerLease
from research_tools.source_policy import SourceMode


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    source_mode: SourceMode = SourceMode.MIXED
    guidelines: list[str] = Field(default_factory=list)
    run_id: str | None = Field(default=None, min_length=1, max_length=128)


class RunOperationResponse(BaseModel):
    run_id: str
    accepted: bool = True
    operation: str


class CancelRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="Run cancelled by API request.", min_length=1)


def create_app(
    settings: Settings | None = None,
    supervisor: RunSupervisor | None = None,
) -> FastAPI:
    configured = settings or load_settings()
    store = SQLiteRuntimeStore(configured.resolved_runtime_database_path())
    run_supervisor = supervisor or RunSupervisor(configured, store=store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await run_supervisor.start()
        try:
            yield
        finally:
            await run_supervisor.shutdown()

    app = FastAPI(
        title="Personal Research Assistant", version="0.1.0", lifespan=lifespan
    )
    app.state.settings = configured
    app.state.runtime_store = store
    app.state.run_supervisor = run_supervisor

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

    @app.post(
        "/api/v1/runs", response_model=RunOperationResponse, status_code=202
    )
    async def create_run(body: CreateRunRequest) -> RunOperationResponse:
        try:
            run_id = await run_supervisor.start_run(
                body.query,
                body.source_mode.value,
                guidelines=body.guidelines,
                run_id=body.run_id,
            )
        except ValueError as exc:
            status_code = 409 if "already" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return RunOperationResponse(run_id=run_id, operation="start")

    @app.get("/api/v1/runs/{run_id}", response_model=RunRecord)
    async def get_run(run_id: str) -> RunRecord:
        record = await asyncio.to_thread(store.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return record

    @app.post(
        "/api/v1/runs/{run_id}/resume",
        response_model=RunOperationResponse,
        status_code=202,
    )
    async def resume_run(run_id: str) -> RunOperationResponse:
        try:
            await run_supervisor.resume_run(run_id)
        except (FileNotFoundError, LookupError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return RunOperationResponse(run_id=run_id, operation="resume")

    @app.post(
        "/api/v1/runs/{run_id}/cancel",
        response_model=RunOperationResponse,
        status_code=202,
    )
    async def cancel_run(
        run_id: str, body: CancelRunRequest = CancelRunRequest()
    ) -> RunOperationResponse:
        record = await asyncio.to_thread(store.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        if not await run_supervisor.cancel_run(run_id, body.reason):
            raise HTTPException(status_code=409, detail="run is not active")
        return RunOperationResponse(run_id=run_id, operation="cancel")

    @app.get("/api/v1/runs/{run_id}/worker", response_model=WorkerLease | None)
    async def get_worker(run_id: str) -> WorkerLease | None:
        record = await asyncio.to_thread(store.get_run, run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="run not found")
        return await asyncio.to_thread(store.get_worker_lease, run_id)

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

    frontend_dist = PROJECT_ROOT / "frontend" / "dist"
    if frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


__all__ = ["create_app"]
