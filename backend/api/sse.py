"""Server-Sent Event encoding and SQLite-backed cursor replay."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import time

from fastapi import Request

from backend.events import RunEvent
from backend.runtime.run_state import RunStatus
from backend.storage.sqlite_runtime import SQLiteRuntimeStore


_TERMINAL_STATUSES = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.INTERRUPTED,
    RunStatus.CANCELLED,
}


def resolve_event_cursor(after_seq: int | None, last_event_id: str | None) -> int:
    """Resolve query/header cursors without ever replaying older events."""

    cursors = [after_seq or 0]
    if last_event_id is not None:
        value = last_event_id.strip()
        if not value.isdecimal():
            raise ValueError("Last-Event-ID must be a non-negative event sequence")
        cursors.append(int(value))
    cursor = max(cursors)
    if cursor < 0:
        raise ValueError("event cursor must be non-negative")
    return cursor


def encode_sse(event: RunEvent) -> str:
    """Encode one event with its durable sequence as the SSE resume ID."""

    return (
        f"id: {event.seq}\n"
        f"event: {event.event_type.value}\n"
        f"data: {event.model_dump_json()}\n\n"
    )


async def stream_run_events(
    request: Request,
    store: SQLiteRuntimeStore,
    run_id: str,
    *,
    after_seq: int = 0,
    page_size: int = 100,
    poll_interval_seconds: float = 0.25,
    heartbeat_seconds: float = 15.0,
) -> AsyncIterator[str]:
    """Replay history, then tail the ledger until disconnect or terminal state."""

    cursor = after_seq
    last_emit = time.monotonic()
    while True:
        events = await asyncio.to_thread(
            store.read_events,
            run_id,
            after_seq=cursor,
            limit=page_size,
        )
        for event in events:
            cursor = event.seq
            last_emit = time.monotonic()
            yield encode_sse(event)

        if await request.is_disconnected():
            return

        record = await asyncio.to_thread(store.get_run, run_id)
        if record is None:
            return
        if record.status in _TERMINAL_STATUSES and len(events) < page_size:
            # The event read may have raced immediately before the atomic
            # terminal snapshot/event commit. Once the terminal snapshot is
            # visible, a second read is a barrier that must also see its event.
            trailing = await asyncio.to_thread(
                store.read_events,
                run_id,
                after_seq=cursor,
                limit=1,
            )
            if not trailing:
                return
            continue

        now = time.monotonic()
        if now - last_emit >= heartbeat_seconds:
            last_emit = now
            yield ": keep-alive\n\n"
        await asyncio.sleep(poll_interval_seconds)


__all__ = ["encode_sse", "resolve_event_cursor", "stream_run_events"]
