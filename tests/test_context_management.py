from __future__ import annotations

from types import SimpleNamespace
from datetime import UTC, datetime

from deepagents.backends import FilesystemBackend
from deepagents import create_deep_agent
from deepagents.middleware.summarization import SummarizationToolMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from backend.config import Settings
from backend.events import RunEventType
from backend.research.run_manager import _record_context_compaction
from backend.runtime.context_management import (
    ContextCompactionNotice,
    ObservableSummarizationMiddleware,
    build_context_middleware_stack,
)
from backend.runtime.run_journal import RunJournal
from backend.runtime.run_state import RunRecord, RunStage, RunStatus
from backend.storage.run_paths import create_run_paths
from backend.runtime.working_memory import (
    WorkingMemoryMiddleware,
    initial_working_memory,
    update_working_memory,
)


class _ToolCallingFakeModel(FakeListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def test_context_stack_replaces_default_summary_and_reports_compaction(tmp_path):
    notices = []
    settings = Settings(
        context_summarization_trigger_tokens=10_000,
        context_summarization_keep_messages=8,
        context_summary_input_tokens=4_000,
    )
    stack = build_context_middleware_stack(
        settings,
        FakeListChatModel(responses=["summary"]),
        FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True),
        agent="chief",
        on_compaction=notices.append,
    )

    summarization = stack[0]
    assert isinstance(summarization, ObservableSummarizationMiddleware)
    assert summarization.name == "SummarizationMiddleware"
    assert isinstance(stack[1], SummarizationToolMiddleware)
    summarization._notify(  # type: ignore[attr-defined]
        SimpleNamespace(
            command=SimpleNamespace(
                update={
                    "_summarization_event": {
                        "cutoff_index": 12,
                        "file_path": "/conversation_history/session.md",
                    }
                }
            )
        ),
        (30, 10_500),
    )
    assert notices[0].agent == "chief"
    assert notices[0].history_path == "/conversation_history/session.md"
    assert notices[0].trigger_tokens == 10_000


def test_working_memory_is_compact_private_state():
    initial = initial_working_memory("research objective")
    command = update_working_memory.func(
        objective="research objective",
        runtime=SimpleNamespace(tool_call_id="call-1"),
        current_focus="compare findings",
        selected_evidence_ids=["ev-1"],
        open_questions=["Is the exception release-specific?"],
        decisions=["Separate Web and private evidence."],
        artifact_paths=["/sections/01-background.md"],
    )

    assert initial["objective"] == "research objective"
    assert command.update["working_memory"]["selected_evidence_ids"] == ["ev-1"]
    assert WorkingMemoryMiddleware.state_schema.__annotations__["working_memory"]


def test_deep_agent_compacts_and_offloads_history_with_observer(tmp_path):
    notices = []
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    model = _ToolCallingFakeModel(responses=["compact summary", "final answer"])
    summarization = ObservableSummarizationMiddleware(
        model,
        backend=backend,
        agent="chief",
        trigger_tokens=20,
        keep_messages=1,
        summary_input_tokens=200,
        on_compaction=notices.append,
    )
    graph = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[summarization],
    )

    graph.invoke(
        {
            "messages": [
                HumanMessage(content="old question " * 30),
                AIMessage(content="old answer " * 30),
                HumanMessage(content="continue with the latest task"),
            ]
        }
    )

    assert len(notices) == 1
    history_path = notices[0].history_path
    assert history_path is not None
    assert (tmp_path / history_path.lstrip("/")).is_file()


def test_compaction_notice_is_appended_to_canonical_run_events(tmp_path):
    paths = create_run_paths(tmp_path, "context-event")
    journal = RunJournal.create(
        tmp_path / "runtime.sqlite3",
        paths,
        RunRecord(
            run_id=paths.run_id,
            query="topic",
            source_mode="web",
            status=RunStatus.RUNNING,
            stage=RunStage.RUNNING_AGENT,
            started_at=datetime.now(UTC),
            report_path=str(paths.report),
        ),
    )
    _record_context_compaction(
        journal,
        ContextCompactionNotice(
            agent="researcher",
            messages_before=30,
            approximate_tokens_before=100_500,
            cutoff_index=18,
            history_path="/conversation_history/session.md",
            trigger_tokens=100_000,
            keep_messages=12,
        ),
    )

    event = journal.store.read_events(paths.run_id)[-1]
    assert event.event_type is RunEventType.CONTEXT_COMPACTED
    assert event.agent == "researcher"
    assert event.payload["history_path"] == "/conversation_history/session.md"
