"""Chief-only tools for deliberate cross-session memory recall and storage."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from backend.config import Settings

from .models import MemoryKind, MemorySourceRef, MemoryTag
from .store import MarkdownMemoryStore


class RecallMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=10)


class SaveMemoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=6_000)
    kind: MemoryKind
    confidence: float = Field(ge=0, le=1)
    tags: list[MemoryTag] = Field(default_factory=list, max_length=20)
    source_refs: list[MemorySourceRef] = Field(default_factory=list, max_length=20)


class LongTermMemoryMiddleware(AgentMiddleware):
    """Expose bounded memory operations without preloading the whole store."""

    def __init__(
        self,
        store: MarkdownMemoryStore,
        *,
        source_run_id: str | None,
        max_results: int,
        excerpt_chars: int,
    ) -> None:
        self.store = store
        self.source_run_id = source_run_id
        self.max_results = max_results
        self.excerpt_chars = excerpt_chars

        def recall_memory(query: str, limit: int = max_results) -> list[dict[str, Any]]:
            """Recall relevant prior memory; never treat it as citable source evidence."""

            selected = self.store.recall(
                query,
                limit=min(limit, self.max_results),
                excerpt_chars=self.excerpt_chars,
            )
            return [item.model_dump(mode="json") for item in selected]

        def save_memory(
            title: str,
            content: str,
            kind: MemoryKind,
            confidence: float,
            tags: list[str] | None = None,
            source_refs: list[str] | None = None,
        ) -> dict[str, Any]:
            """Save durable verified knowledge or preferences, never credentials."""

            entry = self.store.save(
                title=title,
                content=content,
                kind=kind,
                confidence=confidence,
                tags=tags or [],
                source_refs=source_refs or [],
                source_run_id=self.source_run_id,
            )
            return {
                "memory_id": entry.memory_id,
                "saved": True,
                "path": f"entries/{entry.memory_id}.md",
            }

        self.tools = [
            StructuredTool.from_function(
                recall_memory,
                name="recall_long_term_memory",
                description=(
                    "BM25-recall a few prior Markdown memories for planning. "
                    "Memory may be stale or wrong and is never report evidence; "
                    "verify important claims with authorized research tools."
                ),
                args_schema=RecallMemoryInput,
            ),
            StructuredTool.from_function(
                save_memory,
                name="save_long_term_memory",
                description=(
                    "Persist a durable preference, verified finding, decision, or "
                    "procedure for later runs. Save only compact conclusions with "
                    "source references; never save raw tool output or credentials."
                ),
                args_schema=SaveMemoryInput,
            ),
        ]


def build_long_term_memory_middleware(
    settings: Settings, *, source_run_id: str | None
) -> LongTermMemoryMiddleware:
    return LongTermMemoryMiddleware(
        MarkdownMemoryStore(settings.resolved_long_term_memory_dir()),
        source_run_id=source_run_id,
        max_results=settings.long_term_memory_max_results,
        excerpt_chars=settings.long_term_memory_excerpt_chars,
    )


__all__ = ["LongTermMemoryMiddleware", "build_long_term_memory_middleware"]
