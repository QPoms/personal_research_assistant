"""Durable LangGraph checkpoints for local interrupted-run recovery."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def checkpoint_config(run_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": run_id, "checkpoint_ns": ""}}


@asynccontextmanager
async def open_checkpoint_saver(path: Path) -> AsyncIterator[AsyncSqliteSaver]:
    """Open one async saver with strict, non-pickle deserialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serializer = JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=None,
    )
    async with aiosqlite.connect(path) as connection:
        saver = AsyncSqliteSaver(connection, serde=serializer)
        await saver.setup()
        yield saver


async def has_checkpoint(saver: AsyncSqliteSaver, run_id: str) -> bool:
    return await saver.aget_tuple(checkpoint_config(run_id)) is not None


def graph_invoke_options(run_id: str) -> dict[str, Any]:
    """Return the stable invocation contract used for create and resume."""

    return {
        "config": checkpoint_config(run_id),
        "durability": "sync",
    }


__all__ = [
    "checkpoint_config",
    "graph_invoke_options",
    "has_checkpoint",
    "open_checkpoint_saver",
]
