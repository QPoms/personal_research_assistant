from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.events import JsonlRunEventStore, RunEventType


def test_jsonl_event_store_assigns_monotonic_sequences_and_replays(tmp_path):
    path = tmp_path / "events.jsonl"
    store = JsonlRunEventStore(path, "run-1")

    created = store.append(
        RunEventType.RUN_CREATED,
        status="running",
        stage="building_agent",
        payload={"query": "topic"},
    )
    store.append(
        RunEventType.RUN_STAGE_CHANGED,
        status="running",
        stage="running_agent",
    )
    completed = store.append(
        RunEventType.RUN_COMPLETED,
        status="completed",
        stage="completed",
    )

    assert created.seq == 1
    assert completed.seq == 3
    assert [event.seq for event in store.read(after_seq=1)] == [2, 3]
    assert (
        JsonlRunEventStore(path, "run-1").append(RunEventType.REPORT_UPDATED).seq == 4
    )


def test_jsonl_event_store_serializes_concurrent_publishers(tmp_path):
    store = JsonlRunEventStore(tmp_path / "events.jsonl", "run-1")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda index: store.append(
                    RunEventType.TOOL_COMPLETED,
                    task_id=f"T{index}",
                    payload={"index": index},
                ),
                range(32),
            )
        )

    assert [event.seq for event in store.read()] == list(range(1, 33))


def test_event_store_rejects_invalid_replay_arguments(tmp_path):
    store = JsonlRunEventStore(tmp_path / "events.jsonl", "run-1")

    with pytest.raises(ValueError, match="after_seq"):
        store.read(after_seq=-1)
    with pytest.raises(ValueError, match="limit"):
        store.read(limit=0)
