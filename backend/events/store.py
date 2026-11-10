"""Append-only JSONL event storage for one local research run."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .models import RunEvent, RunEventType


class JsonlRunEventStore:
    """Persist a single-process event stream with strict monotonic sequence IDs."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        existing = self.read()
        if any(event.run_id != run_id for event in existing):
            raise ValueError("event log contains a different run_id")
        self._next_seq = existing[-1].seq + 1 if existing else 1

    def append(
        self,
        event_type: RunEventType,
        *,
        source: str = "run_manager",
        status: str | None = None,
        stage: str | None = None,
        agent: str | None = None,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RunEvent:
        with self._lock:
            event = RunEvent(
                seq=self._next_seq,
                run_id=self.run_id,
                event_type=event_type,
                source=source,
                status=status,
                stage=stage,
                agent=agent,
                task_id=task_id,
                payload=payload or {},
            )
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(event.model_dump_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._next_seq += 1
            return event

    def read(
        self, *, after_seq: int = 0, limit: int | None = None
    ) -> tuple[RunEvent, ...]:
        if after_seq < 0:
            raise ValueError("after_seq must be non-negative")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        if not self.path.exists():
            return ()

        events: list[RunEvent] = []
        previous_seq = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                event = RunEvent.model_validate_json(line)
                if event.seq <= previous_seq:
                    raise ValueError(
                        f"event sequence is not strictly increasing at line {line_number}"
                    )
                previous_seq = event.seq
                if event.seq <= after_seq:
                    continue
                events.append(event)
                if limit is not None and len(events) >= limit:
                    break
        return tuple(events)


__all__ = ["JsonlRunEventStore"]
