"""LangChain Tool for embedded local private-knowledge retrieval."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.tools import tool

from backend.config import Settings
from backend.runtime.model_factory import deep_agent_model
from personal_knowledge.config import RetrievalCfg
from personal_knowledge.research import (
    LangChainStructuredResearchModel,
    LocalResearchPipeline,
)
from personal_knowledge.service import LocalPersonalKnowledgeService


def build_personal_knowledge_tool(settings: Settings):
    """Build a local Tool; no HTTP, authentication, or workspace is involved."""
    settings.validate_for_personal_research()
    root = settings.personal_knowledge_root.expanduser().resolve()
    # The migrated retrieval core resolves both corpus and index paths beneath
    # this application-owned root. It is configuration, never an LLM argument.
    os.environ["PERSONAL_KNOWLEDGE_ROOT"] = str(root)
    cfg = RetrievalCfg(
        specification_root=settings.personal_knowledge_corpus_dir,
        index_dir=settings.personal_knowledge_index_dir,
        target_release=settings.personal_knowledge_release,
    )
    core = LocalPersonalKnowledgeService(
        cfg,
        max_results=settings.personal_knowledge_max_results,
        allow_index_build=settings.personal_knowledge_allow_index_build,
    )
    if settings.personal_knowledge_research_enabled:
        if not settings.personal_knowledge_allow_external_model:
            raise ValueError(
                "Local research would send selected private evidence excerpts "
                "to the configured model provider. Set "
                "PERSONAL_KNOWLEDGE_ALLOW_EXTERNAL_MODEL=true only after "
                "explicitly authorizing that disclosure."
            )
        model_name = (
            settings.personal_knowledge_research_model
            or settings.deep_agent_model
        )
        research_model = LangChainStructuredResearchModel(deep_agent_model(
            model_name,
            max_output_tokens=settings.personal_knowledge_model_max_output_tokens,
            use_responses_api=settings.deep_agent_use_responses_api,
        ))
        service: Any = LocalResearchPipeline(
            core,
            research_model,
            max_tasks=settings.personal_knowledge_max_tasks,
            max_rounds=settings.personal_knowledge_max_retrieval_rounds,
            max_parallel_tasks=settings.personal_knowledge_max_parallel_tasks,
            max_tool_calls_per_task=(
                settings.personal_knowledge_max_tool_calls_per_task
            ),
        )
        search_method = service.research
    else:
        service = core
        search_method = service.search

    @tool("search_personal_knowledge")
    async def search_personal_knowledge(query: str) -> dict[str, Any]:
        """Research the local corpus and return a grounded evidence brief.

        A local Planner decomposes complex questions, bounded Researchers run
        hybrid retrieval, and a local Synthesizer returns findings plus selected
        source-verified evidence. The Chief Editor still writes the final report.
        Corpus paths, Release, indexes, and limits are application configuration.
        """
        try:
            result = await asyncio.to_thread(search_method, query)
            payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
            if settings.personal_knowledge_research_enabled:
                payload["answer"] = payload["local_summary"]
                payload["evidence_status"] = (
                    "retrieved" if payload["evidence"] else "insufficient"
                )
                payload["failure"] = None if payload["evidence"] else {
                    "category": "no_valid_evidence",
                    "message": "; ".join(payload["unresolved_gaps"]),
                }
            return payload
        except Exception as exc:
            return {
                "status": "error",
                "query": query,
                "answer": None,
                "evidence": [],
                "evidence_status": "insufficient",
                "unresolved_gaps": [],
                "plan": {"tasks": []},
                "task_results": [],
                "findings": [],
                "conflicts": [],
                "local_summary": "",
                "pipeline_warnings": [],
                "failure": {
                    "category": type(exc).__name__,
                    "message": str(exc),
                    "retryable": False,
                },
            }

    return search_personal_knowledge

