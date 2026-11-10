from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import backend.research.run_manager as run_manager
from backend.config import Settings
from backend.events import JsonlRunEventStore, RunEventType


class _Graph:
    def __init__(self, result, *, before_return=None):
        self.result = result
        self.before_return = before_return

    async def ainvoke(self, payload):
        if self.before_return is not None:
            self.before_return()
        return self.result


def _settings(tmp_path):
    return Settings(outputs_dir=tmp_path)


def _events(root, run_id):
    return JsonlRunEventStore(root / "events.jsonl", run_id).read()


@pytest.mark.asyncio
async def test_agent_build_failure_transitions_run_to_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        run_manager,
        "build_research_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
    )

    with pytest.raises(RuntimeError, match="build failed"):
        await run_manager.run_research(
            "topic", "web", _settings(tmp_path), run_id="build-failure"
        )

    metadata = json.loads(
        (tmp_path / "run_build-failure" / "run.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
    assert metadata["stage"] == "building_agent"
    assert metadata["finished_at"] is not None
    assert metadata["error"] == "RuntimeError: build failed"
    events = _events(tmp_path / "run_build-failure", "build-failure")
    assert [event.event_type for event in events] == [
        RunEventType.RUN_CREATED,
        RunEventType.RUN_FAILED,
    ]
    assert [event.seq for event in events] == [1, 2]


@pytest.mark.asyncio
async def test_missing_report_writes_agent_trace_and_final_summary(
    monkeypatch, tmp_path
):
    result = {
        "messages": [
            {
                "role": "assistant",
                "content": "I stopped before writing the report.",
                "tool_calls": [{"name": "write_todos", "args": {"todos": []}}],
            }
        ]
    }
    built = SimpleNamespace(
        graph=_Graph(result),
        cost_tracker=SimpleNamespace(total=0.0),
    )
    monkeypatch.setattr(run_manager, "build_research_agent", lambda *a, **k: built)

    with pytest.raises(RuntimeError, match="I stopped before writing"):
        await run_manager.run_research(
            "topic", "web", _settings(tmp_path), run_id="missing-report"
        )

    root = tmp_path / "run_missing-report"
    trace = json.loads((root / "agent-trace.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert trace["final_summary"] == "I stopped before writing the report."
    assert trace["messages"][0]["tool_calls"][0]["name"] == "write_todos"
    assert metadata["status"] == "failed"
    assert metadata["stage"] == "validating_report"
    assert "last message" in metadata["error"]
    events = _events(root, "missing-report")
    assert events[-1].event_type is RunEventType.RUN_FAILED
    assert events[-1].stage == "validating_report"


@pytest.mark.asyncio
async def test_successful_run_uses_virtual_report_path(monkeypatch, tmp_path):
    run_root = tmp_path / "run_success"

    def write_report():
        (run_root / "report.md").write_text(
            "# Report\n\nReferences\n" + "x" * 200, encoding="utf-8"
        )

    built = SimpleNamespace(
        graph=_Graph(
            {"messages": [{"role": "assistant", "content": "done /report.md"}]},
            before_return=write_report,
        ),
        cost_tracker=SimpleNamespace(total=0.0),
    )
    monkeypatch.setattr(run_manager, "build_research_agent", lambda *a, **k: built)

    result = await run_manager.run_research(
        "topic", "web", _settings(tmp_path), run_id="success"
    )

    assert result.report_path == run_root / "report.md"
    metadata = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["stage"] == "completed"
    events = _events(run_root, "success")
    assert [event.event_type for event in events] == [
        RunEventType.RUN_CREATED,
        RunEventType.RUN_STAGE_CHANGED,
        RunEventType.RUN_STAGE_CHANGED,
        RunEventType.RUN_COMPLETED,
    ]
    assert [event.seq for event in events] == [1, 2, 3, 4]
