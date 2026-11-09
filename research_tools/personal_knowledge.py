"""LangChain Tool for embedded local private-knowledge retrieval."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from langchain_core.tools import tool

from backend.config import Settings
from personal_knowledge.config import RetrievalCfg
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
    service = LocalPersonalKnowledgeService(
        cfg,
        max_results=settings.personal_knowledge_max_results,
        allow_index_build=settings.personal_knowledge_allow_index_build,
    )

    @tool("search_personal_knowledge")
    async def search_personal_knowledge(query: str) -> dict[str, Any]:
        """Search the configured local private corpus and return verified evidence.

        The corpus root, Release, indexes, and retrieval limits are fixed by
        application configuration and cannot be selected by the model.
        """
        try:
            return await asyncio.to_thread(service.search, query)
        except Exception as exc:
            return {
                "status": "error",
                "query": query,
                "answer": None,
                "evidence": [],
                "evidence_status": "insufficient",
                "unresolved_gaps": [],
                "failure": {
                    "category": type(exc).__name__,
                    "message": str(exc),
                    "retryable": False,
                },
            }

    return search_personal_knowledge

