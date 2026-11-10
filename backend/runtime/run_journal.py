"""Coordinate canonical SQLite state with human-readable run projections."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.events import RunEventType
from backend.storage.run_paths import RunPaths
from backend.storage.sqlite_runtime import SQLiteRuntimeStore

from .run_state import RunRecord, RunStage, RunStatus, transition_run_record


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class RunJournal:
    """Persist transitions transactionally, then refresh file projections."""

    def __init__(
        self,
        store: SQLiteRuntimeStore,
        paths: RunPaths,
        record: RunRecord,
    ) -> None:
        self.store = store
        self.paths = paths
        self.record = record

    @classmethod
    def create(
        cls, database_path: Path, paths: RunPaths, record: RunRecord
    ) -> "RunJournal":
        store = SQLiteRuntimeStore(database_path)
        store.commit(
            record,
            RunEventType.RUN_CREATED,
            payload={"query": record.query, "source_mode": record.source_mode},
        )
        journal = cls(store, paths, record)
        journal.project()
        return journal

    def transition(
        self,
        event_type: RunEventType,
        *,
        status: RunStatus,
        stage: RunStage,
        finished_at: datetime | None = None,
        web_research_cost: float | None = None,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunRecord:
        next_record = transition_run_record(
            self.record,
            status=status,
            stage=stage,
            finished_at=finished_at,
            web_research_cost=web_research_cost,
            error=error,
        )
        event_payload = {"web_research_cost": next_record.web_research_cost}
        if error:
            event_payload["error"] = error
        if payload:
            event_payload.update(payload)
        self.store.commit(next_record, event_type, payload=event_payload)
        self.record = next_record
        self.project()
        return next_record

    def project(self) -> None:
        _atomic_write(
            self.paths.root / "run.json", self.record.model_dump_json(indent=2)
        )
        events = self.store.read_events(self.record.run_id)
        content = "".join(f"{event.model_dump_json()}\n" for event in events)
        _atomic_write(self.paths.events, content)


__all__ = ["RunJournal"]
