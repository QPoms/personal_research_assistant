"""Build the exact research-tool sets authorized for one run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.config import Settings
from research_tools.source_policy import (
    DEEP_RESEARCH,
    PERSONAL_KNOWLEDGE_SEARCH,
    QUICK_SEARCH,
    AgentRole,
    SourceMode,
    coerce_source_mode,
    select_allowed_tools,
)
from research_tools.gpt_researcher import CostTracker, build_web_research_tools
from research_tools.personal_knowledge import build_personal_knowledge_tool


@dataclass(slots=True)
class ResearchToolSets:
    chief: list[Any]
    researcher: list[Any]
    cost_tracker: CostTracker


def build_research_tool_sets(settings: Settings, source_mode: SourceMode | str) -> ResearchToolSets:
    mode = coerce_source_mode(source_mode)
    registry: dict[str, Any] = {}
    cost_tracker = CostTracker()

    if mode in {SourceMode.WEB, SourceMode.MIXED}:
        quick_search, deep_research = build_web_research_tools(
            max_parallel_researchers=settings.max_parallel_researchers,
            max_search_results=settings.max_search_results,
            timeout_seconds=settings.research_tool_timeout_seconds,
            cost_tracker=cost_tracker,
        )
        registry[QUICK_SEARCH] = quick_search
        registry[DEEP_RESEARCH] = deep_research

    if mode in {SourceMode.PERSONAL, SourceMode.MIXED}:
        registry[PERSONAL_KNOWLEDGE_SEARCH] = build_personal_knowledge_tool(settings)

    return ResearchToolSets(
        chief=select_allowed_tools(registry, mode, AgentRole.CHIEF),
        researcher=select_allowed_tools(registry, mode, AgentRole.RESEARCHER),
        cost_tracker=cost_tracker,
    )


