"""SQLite adapter for atomic run snapshots and ordered runtime events."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.events import RunEvent, RunEventType
from backend.runtime.run_state import RunRecord, transition_run_record


@dataclass(frozen=True, slots=True)
class WorkerLease:
    run_id: str
    worker_id: str
    state: str
    started_at: datetime
    heartbeat_at: datetime


class SQLiteRuntimeStore:
    """Canonical local store for run state and its append-only event stream."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_status_updated
                    ON runs(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_run_events_type
                    ON run_events(run_id, event_type, seq);

                CREATE TABLE IF NOT EXISTS worker_leases (
                    run_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    heartbeat_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_worker_leases_state_heartbeat
                    ON worker_leases(state, heartbeat_at);
                """
            )

    def commit(
        self,
        record: RunRecord,
        event_type: RunEventType,
        *,
        source: str = "run_manager",
        agent: str | None = None,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        """Atomically persist the next snapshot and its corresponding event."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT record_json FROM runs WHERE run_id = ?", (record.run_id,)
            ).fetchone()
            if row is None:
                if event_type is not RunEventType.RUN_CREATED:
                    raise ValueError("first run event must be run.created")
            else:
                previous = RunRecord.model_validate_json(row["record_json"])
                expected = transition_run_record(
                    previous,
                    status=record.status,
                    stage=record.stage,
                    finished_at=record.finished_at,
                    web_research_cost=record.web_research_cost,
                    error=record.error,
                )
                if expected != record:
                    raise ValueError(
                        "run record changed outside the transition contract"
                    )

            seq = connection.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM run_events WHERE run_id = ?",
                (record.run_id,),
            ).fetchone()[0]
            event = RunEvent(
                seq=seq,
                run_id=record.run_id,
                event_type=event_type,
                source=source,
                status=record.status.value,
                stage=record.stage.value,
                agent=agent,
                task_id=task_id,
                payload=payload or {},
            )
            connection.execute(
                """
                INSERT INTO runs(run_id, status, stage, updated_at, record_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    stage = excluded.stage,
                    updated_at = excluded.updated_at,
                    record_json = excluded.record_json
                """,
                (
                    record.run_id,
                    record.status.value,
                    record.stage.value,
                    event.occurred_at.isoformat(),
                    record.model_dump_json(),
                ),
            )
            connection.execute(
                """
                INSERT INTO run_events(
                    run_id, seq, event_id, event_type, occurred_at, event_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.seq,
                    event.event_id,
                    event.event_type.value,
                    event.occurred_at.isoformat(),
                    event.model_dump_json(),
                ),
            )
            connection.commit()
            return event
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT record_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return RunRecord.model_validate_json(row["record_json"]) if row else None

    def list_runs(
        self, *, status: str | None = None, limit: int = 100
    ) -> tuple[RunRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        sql = "SELECT record_json FROM runs"
        parameters: list[Any] = []
        if status is not None:
            sql += " WHERE status = ?"
            parameters.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(RunRecord.model_validate_json(row["record_json"]) for row in rows)

    def read_events(
        self, run_id: str, *, after_seq: int = 0, limit: int | None = None
    ) -> tuple[RunEvent, ...]:
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        sql = (
            "SELECT event_json FROM run_events "
            "WHERE run_id = ? AND seq > ? ORDER BY seq"
        )
        parameters: list[Any] = [run_id, after_seq]
        if limit is not None:
            sql += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return tuple(RunEvent.model_validate_json(row["event_json"]) for row in rows)

    def claim_worker(self, run_id: str, worker_id: str) -> WorkerLease:
        """Create or replace the active local-worker lease for a known run."""

        now = datetime.now(UTC)
        with self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                is None
            ):
                raise LookupError(f"unknown run: {run_id}")
            connection.execute(
                """
                INSERT INTO worker_leases(
                    run_id, worker_id, state, started_at, heartbeat_at
                ) VALUES (?, ?, 'active', ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    worker_id = excluded.worker_id,
                    state = 'active',
                    started_at = excluded.started_at,
                    heartbeat_at = excluded.heartbeat_at
                """,
                (run_id, worker_id, now.isoformat(), now.isoformat()),
            )
        return WorkerLease(run_id, worker_id, "active", now, now)

    def heartbeat_worker(self, run_id: str, worker_id: str) -> WorkerLease:
        now = datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE worker_leases
                SET heartbeat_at = ?
                WHERE run_id = ? AND worker_id = ? AND state = 'active'
                """,
                (now.isoformat(), run_id, worker_id),
            )
            if cursor.rowcount != 1:
                raise LookupError(f"active worker lease is unavailable: {run_id}")
            row = connection.execute(
                "SELECT * FROM worker_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._lease_from_row(row)

    def finish_worker(self, run_id: str, worker_id: str, *, state: str) -> None:
        if state not in {"finished", "stale"}:
            raise ValueError("worker lease state must be finished or stale")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE worker_leases
                SET state = ?
                WHERE run_id = ? AND worker_id = ?
                """,
                (state, run_id, worker_id),
            )

    def get_worker_lease(self, run_id: str) -> WorkerLease | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM worker_leases WHERE run_id = ?", (run_id,)
            ).fetchone()
        return self._lease_from_row(row) if row else None

    def stale_worker_leases(self, *, before: datetime) -> tuple[WorkerLease, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM worker_leases
                WHERE state = 'active' AND heartbeat_at < ?
                ORDER BY heartbeat_at
                """,
                (before.isoformat(),),
            ).fetchall()
        return tuple(self._lease_from_row(row) for row in rows)

    @staticmethod
    def _lease_from_row(row: sqlite3.Row) -> WorkerLease:
        return WorkerLease(
            run_id=row["run_id"],
            worker_id=row["worker_id"],
            state=row["state"],
            started_at=datetime.fromisoformat(row["started_at"]),
            heartbeat_at=datetime.fromisoformat(row["heartbeat_at"]),
        )


__all__ = ["SQLiteRuntimeStore", "WorkerLease"]
