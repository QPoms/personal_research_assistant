from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from backend.runtime.checkpoints import (
    checkpoint_config,
    has_checkpoint,
    open_checkpoint_saver,
)


class CounterState(TypedDict):
    value: int


@pytest.mark.asyncio
async def test_sqlite_checkpoint_resumes_at_next_graph_node(tmp_path):
    builder = StateGraph(CounterState)
    builder.add_node("first", lambda state: {"value": state["value"] + 1})
    builder.add_node("second", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    path = tmp_path / "checkpoints.sqlite3"
    config = checkpoint_config("run-1")

    async with open_checkpoint_saver(path) as saver:
        graph = builder.compile(checkpointer=saver)
        partial = await graph.ainvoke(
            {"value": 0},
            config=config,
            interrupt_after=["first"],
            durability="sync",
        )
        snapshot = await graph.aget_state(config)

    assert partial == {"value": 1}
    assert snapshot.next == ("second",)

    async with open_checkpoint_saver(path) as saver:
        assert await has_checkpoint(saver, "run-1") is True
        graph = builder.compile(checkpointer=saver)
        completed = await graph.ainvoke(None, config=config, durability="sync")
        snapshot = await graph.aget_state(config)

    assert completed == {"value": 2}
    assert snapshot.next == ()
