"""Create, execute, and record one isolated research run."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from backend.config import Settings
from backend.events import RunEventType
from backend.runtime.cancellation import (
    CancellationToken,
    RunCancelledError,
    await_with_cancellation,
)
from backend.runtime.checkpoints import (
    graph_invoke_options,
    has_checkpoint,
    open_checkpoint_saver,
)
from backend.runtime.context_management import ContextCompactionNotice
from backend.runtime.run_journal import RunJournal
from backend.runtime.run_state import RunRecord, RunStage, RunStatus
from backend.storage.run_paths import RunPaths, create_run_paths, load_run_paths
from backend.runtime.working_memory import initial_working_memory
from deep_agents.builder import build_research_agent
from deep_agents.paths import VIRTUAL_REPORT_PATH
from research_tools.source_policy import SourceMode, coerce_source_mode


@dataclass(frozen=True, slots=True)
class ResearchRunResult:
    run_id: str
    source_mode: str
    run_dir: Path
    report_path: Path
    summary: str
    web_research_cost: float


def _last_message_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
    messages = result.get("messages") or []
    if not messages:
        return "Research completed."
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    if isinstance(content, str):
        return content
    return str(content or "Research completed.")


def _compact_value(value: Any, *, limit: int = 1200) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + " ...[truncated]"
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item, limit=400)
            for key, item in list(value.items())[:24]
        }
    if isinstance(value, (list, tuple)):
        return [_compact_value(item, limit=400) for item in value[:24]]
    return _compact_value(str(value), limit=limit)


def _message_trace(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        content = message.get("content")
        name = message.get("name")
        tool_calls = message.get("tool_calls") or ()
        role = message.get("role") or message.get("type") or "dict"
        status = message.get("status")
    else:
        content = getattr(message, "content", None)
        name = getattr(message, "name", None)
        tool_calls = getattr(message, "tool_calls", ()) or ()
        role = getattr(message, "type", type(message).__name__)
        status = getattr(message, "status", None)
    return {
        "role": str(role),
        "name": name,
        "status": status,
        "content": _compact_value(content),
        "tool_calls": _compact_value(tool_calls),
    }


def _write_agent_trace(paths: RunPaths, result: Any) -> None:
    messages = result.get("messages") if isinstance(result, dict) else None
    payload = {
        "schema_version": "agent-trace/v1",
        "messages": [_message_trace(message) for message in (messages or ())],
        "final_summary": _compact_value(_last_message_text(result)),
    }
    (paths.root / "agent-trace.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    if isinstance(result, dict) and isinstance(result.get("working_memory"), dict):
        (paths.root / "working-memory.json").write_text(
            json.dumps(
                result["working_memory"], ensure_ascii=False, indent=2, default=str
            ),
            encoding="utf-8",
        )


def _record_context_compaction(
    journal: RunJournal, notice: ContextCompactionNotice
) -> None:
    record = journal.store.get_run(journal.record.run_id)
    if record is None or record.status is not RunStatus.RUNNING:
        return
    journal.store.commit(
        record,
        RunEventType.CONTEXT_COMPACTED,
        source="context_manager",
        agent=notice.agent,
        payload=notice.payload(),
    )


def _user_request(record: RunRecord) -> str:
    return (
        f"Research topic: {record.query}\n"
        f"Authorized source mode: {record.source_mode}.\n"
        "Complete the Chief Editor workflow and write the final report to "
        f"the virtual absolute path {VIRTUAL_REPORT_PATH}."
    )


async def _execute_research(
    journal: RunJournal,
    paths: RunPaths,
    settings: Settings,
    mode: SourceMode,
    token: CancellationToken,
    *,
    graph_input: dict[str, Any] | None,
) -> ResearchRunResult:
    built = None
    base_cost = journal.record.web_research_cost
    try:
        async with open_checkpoint_saver(
            settings.resolved_checkpoint_database_path()
        ) as checkpointer:
            built = build_research_agent(
                settings,
                source_mode=mode,
                run_dir=paths.root,
                guidelines=list(journal.record.guidelines),
                cancellation_token=token,
                checkpointer=checkpointer,
                context_event_callback=lambda notice: _record_context_compaction(
                    journal, notice
                ),
                run_id=paths.run_id,
            )
            if journal.record.stage is RunStage.BUILDING_AGENT:
                journal.transition(
                    RunEventType.RUN_STAGE_CHANGED,
                    status=RunStatus.RUNNING,
                    stage=RunStage.RUNNING_AGENT,
                )
            result = await await_with_cancellation(
                built.graph.ainvoke(
                    graph_input,
                    **graph_invoke_options(paths.run_id),
                ),
                token,
            )

        token.raise_if_cancelled()
        _write_agent_trace(paths, result)
        total_cost = base_cost + built.cost_tracker.total
        if journal.record.stage is not RunStage.VALIDATING_REPORT:
            journal.transition(
                RunEventType.RUN_STAGE_CHANGED,
                status=RunStatus.RUNNING,
                stage=RunStage.VALIDATING_REPORT,
                web_research_cost=total_cost,
            )
        if not paths.report.is_file():
            summary = _last_message_text(result).strip().replace("\n", " ")
            if len(summary) > 500:
                summary = summary[:500] + " ...[truncated]"
            raise RuntimeError(
                f"Agent completed without creating {VIRTUAL_REPORT_PATH}; "
                f"last message: {summary or '[empty]'}"
            )
        token.raise_if_cancelled()
        summary = _last_message_text(result)
        journal.transition(
            RunEventType.RUN_COMPLETED,
            finished_at=datetime.now(UTC),
            status=RunStatus.COMPLETED,
            stage=RunStage.COMPLETED,
            web_research_cost=total_cost,
        )
        return ResearchRunResult(
            run_id=paths.run_id,
            source_mode=mode.value,
            run_dir=paths.root,
            report_path=paths.report,
            summary=summary,
            web_research_cost=total_cost,
        )
    except RunCancelledError as exc:
        web_cost = base_cost + (built.cost_tracker.total if built else 0.0)
        journal.transition(
            RunEventType.RUN_CANCELLED,
            finished_at=datetime.now(UTC),
            status=RunStatus.CANCELLED,
            stage=journal.record.stage,
            web_research_cost=web_cost,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    except (KeyboardInterrupt, asyncio.CancelledError) as exc:
        web_cost = base_cost + (built.cost_tracker.total if built else 0.0)
        journal.transition(
            RunEventType.RUN_INTERRUPTED,
            finished_at=datetime.now(UTC),
            status=RunStatus.INTERRUPTED,
            stage=journal.record.stage,
            web_research_cost=web_cost,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    except Exception as exc:
        web_cost = base_cost + (built.cost_tracker.total if built else 0.0)
        journal.transition(
            RunEventType.RUN_FAILED,
            finished_at=datetime.now(UTC),
            status=RunStatus.FAILED,
            stage=journal.record.stage,
            web_research_cost=web_cost,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


async def run_research(
    query: str,
    source_mode: SourceMode | str,
    settings: Settings,
    *,
    guidelines: list[str] | None = None,
    run_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
    started_event: asyncio.Event | None = None,
) -> ResearchRunResult:
    query = query.strip()
    if not query:
        raise ValueError("Research query must not be empty")
    mode = coerce_source_mode(source_mode)
    token = cancellation_token or CancellationToken()
    token.raise_if_cancelled()
    if mode in {SourceMode.PERSONAL, SourceMode.MIXED}:
        settings.validate_for_personal_research()

    paths = create_run_paths(settings.resolved_outputs_dir(), run_id)
    started_at = datetime.now(UTC)
    journal = RunJournal.create(
        settings.resolved_runtime_database_path(),
        paths,
        RunRecord(
            run_id=paths.run_id,
            query=query,
            source_mode=mode.value,
            status=RunStatus.RUNNING,
            stage=RunStage.BUILDING_AGENT,
            started_at=started_at,
            report_path=str(paths.report),
            guidelines=tuple(guidelines or ()),
        ),
    )
    if started_event is not None:
        started_event.set()
    return await _execute_research(
        journal,
        paths,
        settings,
        mode,
        token,
        graph_input={
            "messages": [{"role": "user", "content": _user_request(journal.record)}],
            "working_memory": initial_working_memory(journal.record.query),
        },
    )


async def resume_research(
    run_id: str,
    settings: Settings,
    *,
    cancellation_token: CancellationToken | None = None,
    started_event: asyncio.Event | None = None,
) -> ResearchRunResult:
    """Continue an interrupted or orphaned run from its latest graph checkpoint."""

    paths = load_run_paths(settings.resolved_outputs_dir(), run_id)
    journal = RunJournal.open(settings.resolved_runtime_database_path(), paths)
    if journal.record.status not in {RunStatus.RUNNING, RunStatus.INTERRUPTED}:
        raise ValueError(
            f"run {run_id} cannot resume from status {journal.record.status.value}"
        )
    mode = coerce_source_mode(journal.record.source_mode)
    if mode in {SourceMode.PERSONAL, SourceMode.MIXED}:
        settings.validate_for_personal_research()
    token = cancellation_token or CancellationToken()
    token.raise_if_cancelled()

    async with open_checkpoint_saver(
        settings.resolved_checkpoint_database_path()
    ) as checkpointer:
        if not await has_checkpoint(checkpointer, run_id):
            raise ValueError(f"run {run_id} has no durable LangGraph checkpoint")

    journal.transition(
        RunEventType.RUN_RESUMED,
        status=RunStatus.RUNNING,
        stage=journal.record.stage,
        payload={"checkpoint_thread_id": run_id},
    )
    if started_event is not None:
        started_event.set()
    return await _execute_research(
        journal,
        paths,
        settings,
        mode,
        token,
        graph_input=None,
    )


__all__ = ["ResearchRunResult", "resume_research", "run_research"]
