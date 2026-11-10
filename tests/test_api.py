from datetime import UTC, datetime
import asyncio
import json

from fastapi.testclient import TestClient
import pytest

import backend.api.app as api_app
from backend.api import create_app
from backend.api.sse import stream_run_events
from backend.config import Settings
from backend.events import RunEventType
from backend.runtime.run_state import (
    RunRecord,
    RunStage,
    RunStatus,
    transition_run_record,
)
from backend.storage.sqlite_runtime import SQLiteRuntimeStore


def _seed_completed_run(settings: Settings, run_id: str = "run-1") -> RunRecord:
    store = SQLiteRuntimeStore(settings.resolved_runtime_database_path())
    record = RunRecord(
        run_id=run_id,
        query="topic",
        source_mode="mixed",
        status=RunStatus.RUNNING,
        stage=RunStage.BUILDING_AGENT,
        started_at=datetime.now(UTC),
        report_path=f"outputs/run_{run_id}/report.md",
    )
    store.commit(record, RunEventType.RUN_CREATED)
    record = transition_run_record(
        record, status=RunStatus.RUNNING, stage=RunStage.RUNNING_AGENT
    )
    store.commit(record, RunEventType.RUN_STAGE_CHANGED)
    record = transition_run_record(
        record, status=RunStatus.RUNNING, stage=RunStage.VALIDATING_REPORT
    )
    store.commit(record, RunEventType.RUN_STAGE_CHANGED)
    record = transition_run_record(
        record,
        status=RunStatus.COMPLETED,
        stage=RunStage.COMPLETED,
        finished_at=datetime.now(UTC),
    )
    store.commit(record, RunEventType.RUN_COMPLETED)
    return record


def _seed_running_run(settings: Settings, run_id: str = "run-1") -> RunRecord:
    store = SQLiteRuntimeStore(settings.resolved_runtime_database_path())
    record = RunRecord(
        run_id=run_id,
        query="topic",
        source_mode="web",
        status=RunStatus.RUNNING,
        stage=RunStage.BUILDING_AGENT,
        started_at=datetime.now(UTC),
        report_path=f"outputs/run_{run_id}/report.md",
    )
    store.commit(record, RunEventType.RUN_CREATED)
    return record


def _sse_events(response) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_run_snapshot_and_list_api_read_canonical_sqlite(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    expected = _seed_completed_run(settings)
    client = TestClient(create_app(settings))

    response = client.get("/api/v1/runs/run-1")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

    response = client.get("/api/v1/runs", params={"status": "completed"})
    assert response.status_code == 200
    assert [item["run_id"] for item in response.json()] == [expected.run_id]
    assert client.get("/api/v1/runs/missing").status_code == 404


def test_sse_replays_only_events_after_query_cursor(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    _seed_completed_run(settings)
    client = TestClient(create_app(settings))

    response = client.get("/api/v1/runs/run-1/events", params={"after_seq": 2})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache, no-transform"
    events = _sse_events(response)
    assert [event["seq"] for event in events] == [3, 4]
    assert "id: 3" in response.text
    assert "event: run.completed" in response.text


def test_sse_last_event_id_resumes_and_never_moves_cursor_backwards(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    _seed_completed_run(settings)
    client = TestClient(create_app(settings))

    response = client.get(
        "/api/v1/runs/run-1/events",
        params={"after_seq": 1},
        headers={"Last-Event-ID": "3"},
    )

    assert [event["seq"] for event in _sse_events(response)] == [4]


def test_sse_rejects_invalid_cursor_and_unknown_run(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    _seed_completed_run(settings)
    client = TestClient(create_app(settings))

    invalid = client.get(
        "/api/v1/runs/run-1/events", headers={"Last-Event-ID": "event-3"}
    )
    assert invalid.status_code == 400
    assert "non-negative" in invalid.json()["detail"]
    assert client.get("/api/v1/runs/missing/events").status_code == 404


def test_sse_does_not_truncate_terminal_history_at_page_boundary(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    record = _seed_running_run(settings)
    store = SQLiteRuntimeStore(settings.resolved_runtime_database_path())
    for index in range(105):
        store.commit(
            record,
            RunEventType.TOOL_COMPLETED,
            task_id=f"task-{index}",
        )
    record = transition_run_record(
        record,
        status=RunStatus.FAILED,
        stage=RunStage.BUILDING_AGENT,
        finished_at=datetime.now(UTC),
        error="test failure",
    )
    store.commit(record, RunEventType.RUN_FAILED)

    response = TestClient(create_app(settings)).get("/api/v1/runs/run-1/events")

    events = _sse_events(response)
    assert len(events) == 107
    assert [event["seq"] for event in events] == list(range(1, 108))


@pytest.mark.asyncio
async def test_sse_tails_running_ledger_until_terminal_event(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    record = _seed_running_run(settings)
    store = SQLiteRuntimeStore(settings.resolved_runtime_database_path())

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def collect() -> list[str]:
        return [
            item
            async for item in stream_run_events(
                ConnectedRequest(),  # type: ignore[arg-type]
                store,
                "run-1",
                poll_interval_seconds=0.001,
            )
        ]

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0.01)
    failed = transition_run_record(
        record,
        status=RunStatus.FAILED,
        stage=RunStage.BUILDING_AGENT,
        finished_at=datetime.now(UTC),
        error="test failure",
    )
    store.commit(failed, RunEventType.RUN_FAILED)

    chunks = await collector
    assert [
        json.loads(line.removeprefix("data: "))["seq"]
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: ")
    ] == [1, 2]


def test_api_serves_built_frontend_without_shadowing_api(monkeypatch, tmp_path):
    frontend = tmp_path / "frontend" / "dist"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text(
        "<!doctype html><title>Runtime console</title>", encoding="utf-8"
    )
    monkeypatch.setattr(api_app, "PROJECT_ROOT", tmp_path)
    client = TestClient(api_app.create_app(Settings(outputs_dir=tmp_path / "outputs")))

    root = client.get("/")
    health = client.get("/api/v1/health")

    assert root.status_code == 200
    assert "Runtime console" in root.text
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}


class _StubSupervisor:
    def __init__(self) -> None:
        self.calls = []

    async def start(self):
        self.calls.append(("lifespan_start",))
        return ()

    async def shutdown(self):
        self.calls.append(("lifespan_shutdown",))

    async def start_run(self, query, source_mode, *, guidelines=None, run_id=None):
        self.calls.append(("start", query, source_mode, guidelines, run_id))
        return run_id or "generated"

    async def resume_run(self, run_id):
        self.calls.append(("resume", run_id))
        return run_id

    async def cancel_run(self, run_id, reason):
        self.calls.append(("cancel", run_id, reason))
        return True


def test_run_control_api_delegates_to_supervisor(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    _seed_running_run(settings, "active")
    supervisor = _StubSupervisor()

    with TestClient(create_app(settings, supervisor=supervisor)) as client:  # type: ignore[arg-type]
        created = client.post(
            "/api/v1/runs",
            json={
                "query": "research topic",
                "source_mode": "web",
                "guidelines": ["Be concise"],
                "run_id": "requested",
            },
        )
        resumed = client.post("/api/v1/runs/active/resume")
        cancelled = client.post(
            "/api/v1/runs/active/cancel", json={"reason": "user request"}
        )

    assert created.status_code == 202
    assert created.json() == {
        "run_id": "requested",
        "accepted": True,
        "operation": "start",
    }
    assert resumed.status_code == 202
    assert cancelled.status_code == 202
    assert supervisor.calls == [
        ("lifespan_start",),
        ("start", "research topic", "web", ["Be concise"], "requested"),
        ("resume", "active"),
        ("cancel", "active", "user request"),
        ("lifespan_shutdown",),
    ]


def test_run_control_api_validates_source_and_unknown_cancel(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    supervisor = _StubSupervisor()
    client = TestClient(create_app(settings, supervisor=supervisor))  # type: ignore[arg-type]

    invalid = client.post(
        "/api/v1/runs", json={"query": "topic", "source_mode": "filesystem"}
    )
    missing = client.post("/api/v1/runs/missing/cancel", json={})

    assert invalid.status_code == 422
    assert missing.status_code == 404
