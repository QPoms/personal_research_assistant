from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from backend.events import RunEventType
from backend.runtime.run_state import (
    RunRecord,
    RunStage,
    RunStatus,
    transition_run_record,
)
from backend.storage.sqlite_runtime import SQLiteRuntimeStore


def _record(run_id: str = "run-1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        query="topic",
        source_mode="hybrid",
        status=RunStatus.RUNNING,
        stage=RunStage.BUILDING_AGENT,
        started_at=datetime.now(UTC),
        report_path=f"outputs/run_{run_id}/report.md",
    )


def test_sqlite_store_commits_snapshot_and_event_together(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    record = _record()
    created = store.commit(record, RunEventType.RUN_CREATED)
    running = transition_run_record(
        record, status=RunStatus.RUNNING, stage=RunStage.RUNNING_AGENT
    )
    changed = store.commit(running, RunEventType.RUN_STAGE_CHANGED)

    assert store.get_run("run-1") == running
    assert [created.seq, changed.seq] == [1, 2]
    assert store.read_events("run-1") == (created, changed)
    assert store.read_events("run-1", after_seq=1, limit=1) == (changed,)
    assert store.list_runs(status="running") == (running,)


def test_sqlite_store_rejects_event_without_created_snapshot(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")

    with pytest.raises(ValueError, match="first run event"):
        store.commit(_record(), RunEventType.TOOL_STARTED)

    assert store.get_run("run-1") is None
    assert store.read_events("run-1") == ()


def test_sqlite_store_rejects_out_of_contract_snapshot(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    record = _record()
    store.commit(record, RunEventType.RUN_CREATED)
    changed = record.model_copy(update={"query": "silently changed"})

    with pytest.raises(ValueError, match="outside the transition contract"):
        store.commit(changed, RunEventType.RUN_STAGE_CHANGED)

    assert store.get_run("run-1") == record
    assert len(store.read_events("run-1")) == 1


def test_sqlite_store_assigns_monotonic_sequences_under_concurrency(tmp_path):
    database = tmp_path / "runtime.sqlite3"
    store = SQLiteRuntimeStore(database)
    record = _record()
    store.commit(record, RunEventType.RUN_CREATED)

    def append_tool_event(index: int):
        return SQLiteRuntimeStore(database).commit(
            record,
            RunEventType.TOOL_COMPLETED,
            task_id=f"task-{index}",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        events = tuple(executor.map(append_tool_event, range(12)))

    assert sorted(event.seq for event in events) == list(range(2, 14))
    assert [event.seq for event in store.read_events("run-1")] == list(range(1, 14))


def test_sqlite_store_tracks_worker_lease_lifecycle(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    store.commit(_record(), RunEventType.RUN_CREATED)

    claimed = store.claim_worker("run-1", "worker-a")
    heartbeat = store.heartbeat_worker("run-1", "worker-a")

    assert claimed.state == "active"
    assert heartbeat.heartbeat_at >= claimed.heartbeat_at
    assert store.get_worker_lease("run-1") == heartbeat
    assert store.stale_worker_leases(before=datetime.now(UTC)) == (heartbeat,)

    store.finish_worker("run-1", "worker-a", state="finished")
    finished = store.get_worker_lease("run-1")
    assert finished is not None
    assert finished.state == "finished"
    assert store.stale_worker_leases(before=datetime.now(UTC)) == ()


def test_sqlite_store_rejects_unknown_or_inactive_worker_lease(tmp_path):
    store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")

    with pytest.raises(LookupError, match="unknown run"):
        store.claim_worker("missing", "worker-a")

    store.commit(_record(), RunEventType.RUN_CREATED)
    store.claim_worker("run-1", "worker-a")
    store.finish_worker("run-1", "worker-a", state="finished")
    with pytest.raises(LookupError, match="unavailable"):
        store.heartbeat_worker("run-1", "worker-a")
