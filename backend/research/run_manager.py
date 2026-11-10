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
from backend.runtime.run_journal import RunJournal
from backend.runtime.run_state import RunRecord, RunStage, RunStatus
from backend.storage.run_paths import RunPaths, create_run_paths
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


async def run_research(
    query: str,
    source_mode: SourceMode | str,
    settings: Settings,
    *,
    guidelines: list[str] | None = None,
    run_id: str | None = None,
    cancellation_token: CancellationToken | None = None,
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
        ),
    )

    user_request = (
        f"Research topic: {query}\n"
        f"Authorized source mode: {mode.value}.\n"
        "Complete the Chief Editor workflow and write the final report to "
        f"the virtual absolute path {VIRTUAL_REPORT_PATH}."
    )

    built = None
    stage = RunStage.BUILDING_AGENT
    try:
        built = build_research_agent(
            settings,
            source_mode=mode,
            run_dir=paths.root,
            guidelines=guidelines,
            cancellation_token=token,
        )
        stage = RunStage.RUNNING_AGENT
        journal.transition(
            RunEventType.RUN_STAGE_CHANGED,
            status=RunStatus.RUNNING,
            stage=stage,
        )
        result = await await_with_cancellation(
            built.graph.ainvoke(
                {"messages": [{"role": "user", "content": user_request}]}
            ),
            token,
        )
        token.raise_if_cancelled()
        _write_agent_trace(paths, result)
        stage = RunStage.VALIDATING_REPORT
        journal.transition(
            RunEventType.RUN_STAGE_CHANGED,
            status=RunStatus.RUNNING,
            stage=stage,
            web_research_cost=built.cost_tracker.total,
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
        finished_at = datetime.now(UTC)
        summary = _last_message_text(result)
        journal.transition(
            RunEventType.RUN_COMPLETED,
            finished_at=finished_at,
            status=RunStatus.COMPLETED,
            stage=RunStage.COMPLETED,
            web_research_cost=built.cost_tracker.total if built else 0.0,
        )
        return ResearchRunResult(
            run_id=paths.run_id,
            source_mode=mode.value,
            run_dir=paths.root,
            report_path=paths.report,
            summary=summary,
            web_research_cost=built.cost_tracker.total,
        )
    except RunCancelledError as exc:
        web_cost = built.cost_tracker.total if built else 0.0
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
        web_cost = built.cost_tracker.total if built else 0.0
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
        web_cost = built.cost_tracker.total if built else 0.0
        journal.transition(
            RunEventType.RUN_FAILED,
            finished_at=datetime.now(UTC),
            status=RunStatus.FAILED,
            stage=journal.record.stage,
            web_research_cost=web_cost,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
