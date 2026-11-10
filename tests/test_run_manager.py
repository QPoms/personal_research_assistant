from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import backend.research.run_manager as run_manager
from backend.config import Settings
from backend.events import JsonlRunEventStore, RunEventType
from backend.runtime.cancellation import CancellationToken, RunCancelledError
from backend.runtime.run_state import RunStatus
from backend.storage.sqlite_runtime import SQLiteRuntimeStore


class _Graph:
    def __init__(self, result, *, before_return=None):
        self.result = result
        self.before_return = before_return

        self.invocations = []

    async def ainvoke(self, payload, **kwargs):
        self.invocations.append((payload, kwargs))
        if self.before_return is not None:
            self.before_return()
        return self.result


class _BlockingGraph:
    def __init__(self):
        self.started = asyncio.Event()
        self.cleaned_up = asyncio.Event()

    async def ainvoke(self, payload, **kwargs):
        self.started.set()
        try:
            await asyncio.Future()
        finally:
            self.cleaned_up.set()


class _InterruptingGraph:
    async def ainvoke(self, payload, **kwargs):
        raise asyncio.CancelledError("process interrupted")


def _settings(tmp_path):
    return Settings(outputs_dir=tmp_path)


def _events(root, run_id):
    return JsonlRunEventStore(root / "events.jsonl", run_id).read()


@pytest.mark.asyncio
async def test_requested_cancellation_is_persisted_as_cancelled(monkeypatch, tmp_path):
    graph = _BlockingGraph()
    built = SimpleNamespace(
        graph=graph,
        cost_tracker=SimpleNamespace(total=0.0),
    )
    token = CancellationToken()
    monkeypatch.setattr(run_manager, "build_research_agent", lambda *a, **k: built)

    task = asyncio.create_task(
        run_manager.run_research(
            "topic",
            "web",
            _settings(tmp_path),
            run_id="cancelled",
            cancellation_token=token,
        )
    )
    await graph.started.wait()
    token.cancel("requested from UI")

    with pytest.raises(RunCancelledError, match="requested from UI"):
        await task

    root = tmp_path / "run_cancelled"
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "cancelled"
    assert metadata["stage"] == "running_agent"
    assert metadata["error"] == "RunCancelledError: requested from UI"
    assert graph.cleaned_up.is_set()
    events = _events(root, "cancelled")
    assert events[-1].event_type is RunEventType.RUN_CANCELLED
    canonical = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3").get_run("cancelled")
    assert canonical is not None
    assert canonical.status is RunStatus.CANCELLED


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
    canonical = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3").get_run(
        "build-failure"
    )
    assert canonical is not None
    assert canonical.status is RunStatus.FAILED


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
    canonical_store = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3")
    canonical = canonical_store.get_run("success")
    assert canonical is not None
    assert canonical.status is RunStatus.COMPLETED
    assert canonical_store.read_events("success") == events


@pytest.mark.asyncio
async def test_interrupted_run_resumes_from_same_checkpoint_thread(
    monkeypatch, tmp_path
):
    run_root = tmp_path / "run_resume-me"
    resumed_graph = _Graph(
        {"messages": [{"role": "assistant", "content": "resumed"}]},
        before_return=lambda: (run_root / "report.md").write_text(
            "# Report\n\nReferences\n" + "x" * 200, encoding="utf-8"
        ),
    )
    built_agents = iter(
        [
            SimpleNamespace(
                graph=_InterruptingGraph(),
                cost_tracker=SimpleNamespace(total=0.0),
            ),
            SimpleNamespace(
                graph=resumed_graph,
                cost_tracker=SimpleNamespace(total=0.25),
            ),
        ]
    )
    build_calls = []

    def build_agent(*args, **kwargs):
        build_calls.append(kwargs)
        return next(built_agents)

    async def checkpoint_exists(*args, **kwargs):
        return True

    monkeypatch.setattr(run_manager, "build_research_agent", build_agent)
    monkeypatch.setattr(run_manager, "has_checkpoint", checkpoint_exists)
    settings = _settings(tmp_path)

    with pytest.raises(asyncio.CancelledError, match="process interrupted"):
        await run_manager.run_research(
            "topic",
            "web",
            settings,
            run_id="resume-me",
            guidelines=["Use a concise table."],
        )

    interrupted = SQLiteRuntimeStore(tmp_path / "runtime.sqlite3").get_run("resume-me")
    assert interrupted is not None
    assert interrupted.status is RunStatus.INTERRUPTED
    assert interrupted.guidelines == ("Use a concise table.",)

    result = await run_manager.resume_research("resume-me", settings)

    assert result.summary == "resumed"
    assert result.web_research_cost == 0.25
    payload, invoke_options = resumed_graph.invocations[0]
    assert payload is None
    assert invoke_options["config"]["configurable"]["thread_id"] == "resume-me"
    assert invoke_options["durability"] == "sync"
    assert build_calls[1]["guidelines"] == ["Use a concise table."]
    assert build_calls[1]["checkpointer"] is not None
    events = _events(run_root, "resume-me")
    assert RunEventType.RUN_INTERRUPTED in [event.event_type for event in events]
    assert RunEventType.RUN_RESUMED in [event.event_type for event in events]
    assert events[-1].event_type is RunEventType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_resume_requires_durable_checkpoint(tmp_path):
    paths = run_manager.create_run_paths(tmp_path, "no-checkpoint")
    record = run_manager.RunRecord(
        run_id=paths.run_id,
        query="topic",
        source_mode="web",
        status=run_manager.RunStatus.RUNNING,
        stage=run_manager.RunStage.BUILDING_AGENT,
        started_at=run_manager.datetime.now(run_manager.UTC),
        report_path=str(paths.report),
    )
    run_manager.RunJournal.create(tmp_path / "runtime.sqlite3", paths, record)

    with pytest.raises(ValueError, match="no durable LangGraph checkpoint"):
        await run_manager.resume_research("no-checkpoint", _settings(tmp_path))
