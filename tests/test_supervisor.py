from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import backend.research.run_manager as run_manager
from backend.config import Settings
from backend.events import RunEventType
from backend.runtime.run_journal import RunJournal
from backend.runtime.run_state import RunRecord, RunStage, RunStatus
from backend.runtime.supervisor import RunSupervisor
from backend.storage.run_paths import create_run_paths
from backend.storage.sqlite_runtime import SQLiteRuntimeStore


class _BlockingGraph:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def ainvoke(self, payload, **kwargs):
        self.started.set()
        await asyncio.Future()


async def _wait_inactive(supervisor: RunSupervisor, run_id: str) -> None:
    for _ in range(100):
        if not await supervisor.is_active(run_id):
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"run stayed active: {run_id}")


@pytest.mark.asyncio
async def test_supervisor_heartbeats_and_cooperatively_cancels(
    monkeypatch, tmp_path
):
    graph = _BlockingGraph()
    monkeypatch.setattr(
        run_manager,
        "build_research_agent",
        lambda *args, **kwargs: SimpleNamespace(
            graph=graph, cost_tracker=SimpleNamespace(total=0.0)
        ),
    )
    settings = Settings(outputs_dir=tmp_path)
    supervisor = RunSupervisor(
        settings,
        worker_id="worker-test",
        heartbeat_interval_seconds=0.01,
    )

    run_id = await supervisor.start_run("topic", "web", run_id="managed")
    await graph.started.wait()
    first = supervisor.store.get_worker_lease(run_id)
    assert first is not None and first.state == "active"
    second = first
    for _ in range(100):
        await asyncio.sleep(0.01)
        second = supervisor.store.get_worker_lease(run_id)
        if second is not None and second.heartbeat_at > first.heartbeat_at:
            break
    assert second is not None and second.heartbeat_at > first.heartbeat_at

    assert await supervisor.cancel_run(run_id, "cancelled in test") is True
    await _wait_inactive(supervisor, run_id)

    record = supervisor.store.get_run(run_id)
    lease = supervisor.store.get_worker_lease(run_id)
    assert record is not None and record.status is RunStatus.CANCELLED
    assert lease is not None and lease.state == "finished"


@pytest.mark.asyncio
async def test_supervisor_shutdown_persists_resumable_interruption(
    monkeypatch, tmp_path
):
    graph = _BlockingGraph()
    monkeypatch.setattr(
        run_manager,
        "build_research_agent",
        lambda *args, **kwargs: SimpleNamespace(
            graph=graph, cost_tracker=SimpleNamespace(total=0.0)
        ),
    )
    settings = Settings(outputs_dir=tmp_path)
    supervisor = RunSupervisor(settings, worker_id="worker-test")

    await supervisor.start_run("topic", "web", run_id="shutdown")
    await graph.started.wait()
    await supervisor.shutdown()

    record = supervisor.store.get_run("shutdown")
    assert record is not None
    assert record.status is RunStatus.INTERRUPTED
    assert supervisor.store.read_events("shutdown")[-1].event_type is RunEventType.RUN_INTERRUPTED


@pytest.mark.asyncio
async def test_watchdog_marks_expired_worker_as_interrupted(tmp_path):
    settings = Settings(outputs_dir=tmp_path)
    paths = create_run_paths(tmp_path, "orphan")
    journal = RunJournal.create(
        settings.resolved_runtime_database_path(),
        paths,
        RunRecord(
            run_id="orphan",
            query="topic",
            source_mode="web",
            status=RunStatus.RUNNING,
            stage=RunStage.RUNNING_AGENT,
            started_at=datetime.now(UTC),
            report_path=str(paths.report),
        ),
    )
    store = SQLiteRuntimeStore(settings.resolved_runtime_database_path())
    store.claim_worker("orphan", "dead-worker")
    supervisor = RunSupervisor(settings, store=store, stale_after_seconds=0)

    recovered = await supervisor.recover_stale_once()

    assert recovered == ("orphan",)
    record = store.get_run("orphan")
    lease = store.get_worker_lease("orphan")
    assert record is not None and record.status is RunStatus.INTERRUPTED
    assert record.error is not None and "heartbeat expired" in record.error
    assert lease is not None and lease.state == "stale"
    event = store.read_events("orphan")[-1]
    assert event.event_type is RunEventType.RUN_INTERRUPTED
    assert event.payload["reason"] == "worker_heartbeat_expired"
