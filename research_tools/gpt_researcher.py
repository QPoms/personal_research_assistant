"""GPT Researcher web tools adapted from its Deep Agents example.

Adapted from ``deep_agents/tools.py`` in GPT Researcher 0.16.0. The source
selection is deliberately closed over and fixed to ``web``. See
``THIRD_PARTY_NOTICES.md`` for provenance and the upstream license caveat.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from gpt_researcher import GPTResearcher

from backend.runtime.cancellation import CancellationToken


@dataclass(slots=True)
class CostTracker:
    """Concurrency-safe accumulated GPT Researcher cost for one app run."""

    total: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def add(self, value: Any) -> None:
        if isinstance(value, dict):
            numeric = sum(
                float(item) for item in value.values() if isinstance(item, (int, float))
            )
        elif isinstance(value, (int, float)):
            numeric = float(value)
        else:
            numeric = 0.0
        async with self._lock:
            self.total += numeric


def build_web_research_tools(
    *,
    max_parallel_researchers: int = 3,
    max_search_results: int = 5,
    timeout_seconds: float = 900.0,
    cost_tracker: CostTracker | None = None,
    cancellation_token: CancellationToken | None = None,
):
    """Return the upstream-style ``quick_search`` and ``deep_research`` tools.

    GPT Researcher's complete engine remains an installed dependency. This
    adapter does not reimplement retrieval, scraping, planning, or reporting.
    """

    semaphore = asyncio.Semaphore(max_parallel_researchers)
    tracker = cost_tracker or CostTracker()

    def _check_cancelled() -> None:
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()

    async def _record_cost(researcher: GPTResearcher) -> None:
        await tracker.add(researcher.get_costs())

    async def quick_search(query: str) -> str:
        """Run GPT Researcher's fast web search and return snippets with URLs."""
        _check_cancelled()
        async with semaphore:
            _check_cancelled()
            async with asyncio.timeout(timeout_seconds):
                researcher = GPTResearcher(
                    query=query,
                    report_type="research_report",
                    report_source="web",
                    verbose=False,
                )
                results = await researcher.quick_search(query)
                _check_cancelled()
                await _record_cost(researcher)

        if isinstance(results, list):
            selected = results[:max_search_results]
            formatted = "\n\n".join(
                (
                    f"[{index}] {item.get('title') or ''}\n"
                    f"{item.get('body') or item.get('content') or ''}\n"
                    f"URL: {item.get('href') or item.get('url') or ''}"
                ).strip()
                for index, item in enumerate(selected, 1)
                if isinstance(item, dict)
            )
            return formatted or "No results found."
        return str(results or "No results found.")

    async def deep_research(query: str, parent_query: str = "") -> str:
        """Run one complete GPT Researcher web pipeline for a report section."""
        _check_cancelled()
        report_type = "subtopic_report" if parent_query else "research_report"
        async with semaphore:
            _check_cancelled()
            async with asyncio.timeout(timeout_seconds):
                researcher = GPTResearcher(
                    query=query,
                    report_type=report_type,
                    report_source="web",
                    parent_query=parent_query,
                    verbose=False,
                )
                await researcher.conduct_research()
                _check_cancelled()
                report = await researcher.write_report()
                _check_cancelled()
                await _record_cost(researcher)
                sources = researcher.get_source_urls()

        if sources:
            report += "\n\n### Sources\n\n" + "\n".join(f"- {url}" for url in sources)
        return report

    # Useful to the run manager without exposing cost as an LLM tool.
    quick_search.cost_tracker = tracker  # type: ignore[attr-defined]
    deep_research.cost_tracker = tracker  # type: ignore[attr-defined]
    return quick_search, deep_research
