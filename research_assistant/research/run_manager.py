"""Create, execute, and record one isolated research run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from research_assistant.agents.builder import build_research_agent
from research_assistant.config import Settings
from research_assistant.research.source_policy import SourceMode, coerce_source_mode
from research_assistant.storage.run_paths import RunPaths, create_run_paths


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


def _write_metadata(
    paths: RunPaths,
    *,
    query: str,
    source_mode: SourceMode,
    started_at: datetime,
    finished_at: datetime | None = None,
    status: str,
    web_research_cost: float = 0.0,
    error: str | None = None,
) -> None:
    payload = {
        "run_id": paths.run_id,
        "query": query,
        "source_mode": source_mode.value,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat() if finished_at else None,
        "web_research_cost": web_research_cost,
        "report_path": str(paths.report),
        "error": error,
    }
    (paths.root / "run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def run_research(
    query: str,
    source_mode: SourceMode | str,
    settings: Settings,
    *,
    guidelines: list[str] | None = None,
    run_id: str | None = None,
) -> ResearchRunResult:
    query = query.strip()
    if not query:
        raise ValueError("Research query must not be empty")
    mode = coerce_source_mode(source_mode)
    if mode in {SourceMode.PERSONAL, SourceMode.MIXED}:
        settings.validate_for_personal_research()

    paths = create_run_paths(settings.outputs_dir, run_id)
    started_at = datetime.now(UTC)
    _write_metadata(
        paths,
        query=query,
        source_mode=mode,
        started_at=started_at,
        status="running",
    )

    built = build_research_agent(
        settings,
        source_mode=mode,
        run_dir=paths.root,
        guidelines=guidelines,
    )
    user_request = (
        f"Research topic: {query}\n"
        f"Authorized source mode: {mode.value}.\n"
        "Complete the Chief Editor workflow and write the final report to report.md."
    )

    try:
        result = await built.graph.ainvoke(
            {"messages": [{"role": "user", "content": user_request}]}
        )
        if not paths.report.is_file():
            raise RuntimeError("Agent completed without creating report.md")
        finished_at = datetime.now(UTC)
        summary = _last_message_text(result)
        _write_metadata(
            paths,
            query=query,
            source_mode=mode,
            started_at=started_at,
            finished_at=finished_at,
            status="completed",
            web_research_cost=built.cost_tracker.total,
        )
        return ResearchRunResult(
            run_id=paths.run_id,
            source_mode=mode.value,
            run_dir=paths.root,
            report_path=paths.report,
            summary=summary,
            web_research_cost=built.cost_tracker.total,
        )
    except Exception as exc:
        _write_metadata(
            paths,
            query=query,
            source_mode=mode,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            status="failed",
            web_research_cost=built.cost_tracker.total,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


